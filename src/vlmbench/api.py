"""CPU-only API evaluation using the existing frozen-suite protocol."""
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import re
import signal
import socket
import time
from urllib.parse import urlsplit

from .common import append, json_hash, now, sha, verify_suite, write


class APIError(RuntimeError):
    """A sanitized transport or response error; no request headers are retained."""


def provider_limit(error):
    """Stop on provider caps without retrying or guessing their reset time."""
    text = str(error).lower()
    return bool(re.search(r'http (402|429)\b', text) or any(marker in text for marker in (
        'insufficient_quota', 'quota exceeded', 'quota_exhausted', 'resource_exhausted',
        'exceeded your current quota', 'insufficient balance', 'credit balance',
        'billing hard limit', '额度不足', '余额不足', '配额已用尽', '超出配额')))


class APIClient:
    def __init__(self, credentials, timeout=300):
        try:
            config = json.loads(Path(credentials).expanduser().read_text())
            self._key = config['api_key']
            self.model = config['model']
            self.base_url = config['base_url'].rstrip('/')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise ValueError('Credentials must contain base_url, api_key and model strings.') from None
        if not all(isinstance(v, str) and v.strip() for v in (self._key, self.model, self.base_url)):
            raise ValueError('Credentials must contain nonempty base_url, api_key and model strings.')
        self._url = urlsplit(self.base_url)
        if (self._url.scheme not in ('http', 'https') or not self._url.hostname
                or self._url.username or self._url.password or self._url.query or self._url.fragment):
            raise ValueError('base_url must be an HTTP(S) URL without userinfo, query or fragment.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', self.model):
            raise ValueError('model must be a filesystem-safe model ID.')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('timeout must be None or a finite positive number of seconds.')
        self.timeout = timeout

    def clean(self, value):
        """Redact even a server response that echoes the credential."""
        if isinstance(value, str):
            return value.replace(self._key, '[REDACTED]')
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, dict):
            return {self.clean(k): self.clean(v) for k, v in value.items()}
        return value

    @property
    def public_config(self):
        return self.clean({
            'backend': 'openai_compatible_api', 'base_url': self.base_url,
            'requested_model': self.model, 'upstream_revision': None,
            'identity_note': 'Proxy-advertised ID; underlying vendor model/version not independently verified.',
            'temperature': 0, 'max_new_tokens': 128, 'stream': False,
            'thinking_policy': 'Provider default for the selected model alias; no thinking override.',
            'timeout_seconds': self.timeout, 'concurrency': 1, 'automatic_retries': 0,
            'latency_kind': 'Client end-to-end including media encoding, proxy, network and remote processing.',
            'token_semantics': 'Preserve provider usage; completion_tokens and reasoning_tokens are not assumed additive.',
        })

    def messages(self, case, root, no_image=False):
        content = []
        if not no_image:
            for media in case['media']:
                path = (Path(root) / media['path']).resolve()
                if not path.is_relative_to(Path(root).resolve()):
                    raise ValueError('Media path escapes suite.')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != media['sha256']:
                    raise ValueError('Media SHA256 mismatch.')
                mime = mimetypes.guess_type(path.name)[0]
                if not mime or not mime.startswith('image/'):
                    raise ValueError('Unsupported media type.')
                if 'timestamp_seconds' in media:
                    content.append({'type': 'text', 'text': f'Frame at {media["timestamp_seconds"]:.6f} seconds:'})
                content.append({'type': 'image_url', 'image_url': {
                    'url': f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')}})
        content.append({'type': 'text', 'text': case['prompt']})
        return [{'role': 'user', 'content': content}]

    def generate(self, case, root, no_image=False):
        started = time.perf_counter()
        body = json.dumps({'model': self.model, 'messages': self.messages(case, root, no_image),
                           'temperature': 0, 'max_tokens': 128, 'stream': False}).encode()
        deadline = None if self.timeout is None else started + self.timeout

        def remaining():
            if deadline is None:
                return None
            seconds = deadline - time.perf_counter()
            if seconds <= 0:
                raise TimeoutError('API request deadline exceeded.')
            return seconds

        connection_type = http.client.HTTPSConnection if self._url.scheme == 'https' else http.client.HTTPConnection
        connection = None
        try:
            connection = connection_type(self._url.hostname, self._url.port, timeout=remaining())
            connection.connect()
            connection.sock.settimeout(remaining())
            connection.request('POST', self._url.path.rstrip('/') + '/chat/completions', body=body,
                               headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'})
            connection.sock.settimeout(remaining())
            response = connection.getresponse()
            # Keep a socket reference when Connection: close detaches it from HTTPConnection.
            response_socket = connection.sock or response.fp.raw._sock
            chunks = []
            size = 0
            while True:
                response_socket.settimeout(remaining())
                chunk = response.read1(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise APIError('API response exceeds 8 MiB.')
                chunks.append(chunk)
                if response.isclosed():
                    break
            raw = b''.join(chunks).decode('utf-8', errors='replace')
            if response.status != 200:
                raise APIError(f'HTTP {response.status}: {self.clean(raw)[:2000]}')
            data = self.clean(json.loads(raw))
            if not isinstance(data, dict) or not isinstance(data.get('choices'), list) or not data['choices']:
                raise APIError('API response has no choices.')
            choice = data['choices'][0]
            message = choice.get('message', {})
            answer = message.get('content')
            if isinstance(answer, list):
                answer = ''.join(p.get('text', '') for p in answer if isinstance(p, dict) and p.get('type') == 'text')
            if answer is None:
                answer = ''
            if not isinstance(answer, str):
                raise APIError('API response content is not text.')
            usage = data.get('usage') or {}
            if not isinstance(usage, dict):
                raise APIError('API usage must be an object.')
            details = usage.get('completion_tokens_details') or {}
            reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
            return self.clean({
                'answer': answer, 'requested_model': self.model, 'response_model': data.get('model'),
                'response_id': data.get('id'), 'finish_reason': choice.get('finish_reason'),
                'length_limit_reached': choice.get('finish_reason') == 'length',
                'input_tokens': usage.get('prompt_tokens'), 'generated_tokens': usage.get('completion_tokens'),
                'reasoning_tokens': reasoning, 'api_usage': usage,
                'reasoning_content': message.get('reasoning_content') or message.get('reasoning'),
                'total_seconds': time.perf_counter() - started,
                'preprocess_seconds': None, 'generation_seconds': None,
                'ttft_seconds': None, 'ttft_kind': 'Not measured: non-streaming API request.',
                'request_body_sha256': hashlib.sha256(body).hexdigest(),
                'processor_settings': {'input_mode': 'original images; chronological multi-image sampled videos',
                                       'image_count': 0 if no_image else len(case['media']),
                                       'media_resized_by_client': False},
            })
        except Exception as exc:
            raise APIError(self.clean(f'{type(exc).__name__}: {exc}')) from None
        finally:
            if connection is not None:
                connection.close()


def request_row(client, case, suite, no_image):
    condition = 'no_image' if no_image else 'image'
    base = {'id': case['id'], 'condition': condition, 'dataset': case['dataset'],
            'category': case['category'], 'model': client.clean(client.model),
            'input_sha256': case['input_sha256'], 'started_utc': now(),
            'request_sha256': json_hash({'input_sha256': case['input_sha256'], 'condition': condition})}
    started = time.perf_counter()
    try:
        generated = client.generate(case, suite, no_image)
        row = {**base, **generated, 'status': 'ok', 'completed_utc': now()}
        if not generated['answer'].strip():
            row.update(status='error', error_type='EmptyAnswer', error='API returned an empty answer.')
        return row
    except Exception as exc:
        return {**base, 'status': 'error', 'error_type': type(exc).__name__,
                'error': client.clean(str(exc)), 'total_seconds': time.perf_counter() - started,
                'completed_utc': now(), 'provider_limit_detected': provider_limit(exc)}


def parallel_requests(client, jobs, suite, concurrency, save):
    """Bound issued requests; stop dispatching after errors, draining paid calls."""
    pending = set()
    cursor = consecutive_errors = peak = 0
    stopped = False
    stop_reason = None
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='vlm-api') as pool:
        try:
            while pending or cursor < len(jobs) and not stopped:
                while len(pending) < concurrency and cursor < len(jobs) and not stopped:
                    case, no_image = jobs[cursor]
                    pending.add(pool.submit(request_row, client, case, suite, no_image))
                    cursor += 1
                    peak = max(peak, len(pending))
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                rows = [(future, future.result()) for future in done]
                for future, row in sorted(rows, key=lambda item: item[1]['completed_utc']):
                    save(row)
                    pending.remove(future)
                    consecutive_errors = 0 if row['status'] == 'ok' else consecutive_errors + 1
                    if row.get('provider_limit_detected'):
                        stop_reason = 'provider_limit'
                        stopped = True
                    stopped = stopped or consecutive_errors >= 3
                    if stopped and stop_reason is None:
                        stop_reason = 'three_consecutive_errors'
        except BaseException:
            # Requests already issued cannot be cancelled at the provider. Save
            # their responses before the outer runner records unsent jobs.
            for future in list(pending):
                try:
                    save(future.result())
                except BaseException:
                    continue
            raise
    return {'dispatched_requests': cursor, 'peak_in_flight': peak,
            'circuit_breaker_tripped': stopped,
            'stop_reason': stop_reason,
            'error_policy': 'Provider cap or three consecutive errors stop dispatch; in-flight requests are drained.'}


def run_api(suite, credentials, output, view='core', timeout=300, concurrency=1, remaining=None, skip_warmup=False):
    """One saved result for every planned request, including interruption/failure."""
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError('concurrency must be an integer from 1 to 8.')
    if remaining is not None and view != 'core':
        raise ValueError('A remaining-request list is supported only for the core view.')
    suite, output = Path(suite).resolve(), Path(output)
    cases = verify_suite(suite)
    views = json.loads((suite / 'views.json').read_text())
    byid = {c['id']: c for c in cases}
    ids = views[view]
    order = [i for i in views['smoke'] if i in ids] + [i for i in ids if i not in views['smoke']]
    jobs = [(byid[i], False) for i in order]
    jobs += [(byid[i], True) for i in views['no_image_diagnostic'] if i in ids]
    client = APIClient(credentials, timeout)
    selection = None
    if remaining is not None:
        selection = json.loads(Path(remaining).read_text())
        if selection['model'] != client.model or selection['suite_sha256'] != sha(suite / 'freeze.json'):
            raise ValueError('Remaining-request model or frozen suite differs from this run.')
        requested = selection['requests']
        wanted = {(r['id'], r['condition']) for r in requested}
        allowed = {(c['id'], 'no_image' if no else 'image') for c, no in jobs}
        if not wanted or len(wanted) != len(requested) or not wanted <= allowed:
            raise ValueError('Remaining requests must be a nonempty unique subset of the frozen core protocol.')
        if selection['remaining_requests'] != len(requested):
            raise ValueError('Remaining-request count mismatch.')
        for row in requested:
            if row['input_sha256'] != byid[row['id']]['input_sha256'] or row['status'] not in ('error', 'not_run'):
                raise ValueError('Remaining input hash or status is invalid; successful calls cannot be repeated.')
        jobs = [(c, no) for c, no in jobs if (c['id'], 'no_image' if no else 'image') in wanted]
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    meta = {'model': client.model, 'started_utc': now(), 'pid': os.getpid(), 'host': socket.gethostname(),
            'python': platform.python_version(), 'config': {**client.public_config, 'concurrency': concurrency}, 'view': view,
            'planned_requests': len(jobs), 'suite_sha256': sha(suite / 'freeze.json'),
            'source_sha256': {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')}}
    if selection is not None:
        write(output / 'selection.json', client.clean({'model':client.model,'suite_sha256':meta['suite_sha256'],
              'remaining_requests':len(jobs),'requests':[{k:r[k] for k in ('id','condition','input_sha256','status')} for r in selection['requests']],
              'source_remaining_sha256':sha(remaining)}))
        meta['selection_sha256'] = sha(output / 'selection.json')
    write(output / 'run.json', meta)
    write(output / 'planned.json', [{'id': c['id'], 'condition': 'no_image' if no else 'image',
                                  'input_sha256': c['input_sha256']} for c, no in jobs])
    seen = set()
    statuses = []
    previous_handlers = {}
    interruption_requested = False

    def interrupted(signum, frame):
        nonlocal interruption_requested
        if not interruption_requested:
            interruption_requested = True
            raise KeyboardInterrupt(f'Signal {signum}')

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        if skip_warmup:
            write(output / 'warmup.json', {'warmup': True, 'status': 'skipped',
                                         'reason': 'Explicit --skip-warmup; no API call issued.'})
            print('WARMUP_SKIPPED', client.clean(client.model), flush=True)
        else:
            warmup_case, warmup_no_image = jobs[0]
            try:
                warmup = client.generate(warmup_case, suite, warmup_no_image)
            except Exception as exc:
                write(output / 'warmup.json', {'id': warmup_case['id'], 'warmup': True, 'status': 'error',
                                             'error': client.clean(str(exc))})
                raise
            write(output / 'warmup.json', {**warmup, 'id': warmup_case['id'], 'warmup': True})
            if not warmup['answer'].strip():
                raise APIError('Warmup returned an empty answer.')
            print('WARMUP_OK', client.clean(client.model), flush=True)
        consecutive_errors = 0
        with (output / 'outputs.jsonl').open('x') as stream:
            def save(row):
                key = (row['id'], row['condition'])
                if key in seen:
                    return
                append(stream, client.clean(row))
                seen.add(key)
                statuses.append(row['status'])
                print('RESULT', len(seen), '/', len(jobs), *key, row['status'],
                      round(row.get('total_seconds', 0), 3), flush=True)
            meta['formal_started_utc'] = now()
            started = time.perf_counter()
            try:
                if concurrency > 1:
                    meta['scheduler'] = parallel_requests(client, jobs, suite, concurrency, save)
                else:
                    stop_reason = None
                    for case, no_image in jobs:
                        if consecutive_errors >= 3:
                            stop_reason = 'three_consecutive_errors'
                            break
                        row = request_row(client, case, suite, no_image)
                        save(row)
                        consecutive_errors = 0 if row['status'] == 'ok' else consecutive_errors + 1
                        if row.get('provider_limit_detected'):
                            stop_reason = 'provider_limit'
                            break
                    meta['scheduler'] = {'dispatched_requests': len(seen), 'peak_in_flight': 1,
                                         'circuit_breaker_tripped': stop_reason is not None,
                                         'stop_reason': stop_reason}
            finally:
                meta['formal_completed_utc'] = now()
                meta['formal_wall_seconds'] = time.perf_counter() - started
    except BaseException as exc:
        meta['fatal_error'] = client.clean(f'{type(exc).__name__}: {exc}')
    finally:
        with (output / 'outputs.jsonl').open('a') as stream:
            for case, no_image in jobs:
                condition = 'no_image' if no_image else 'image'
                if (case['id'], condition) not in seen:
                    append(stream, {'id': case['id'], 'condition': condition, 'model': client.clean(client.model),
                                    'dataset': case['dataset'], 'category': case['category'],
                                    'input_sha256': case['input_sha256'], 'status': 'not_run',
                                    'error': ('Provider cap detected; deferred without automatic retry.' if
                                              meta.get('scheduler', {}).get('stop_reason') == 'provider_limit' else
                                              'Stopped after three consecutive API errors.' if
                                              meta.get('scheduler', {}).get('circuit_breaker_tripped') else
                                              'API worker exited before this request; see run.json.')})
                    statuses.append('not_run')
        meta.update(completed_utc=now(), output_count=len(statuses), ok_count=statuses.count('ok'),
                    all_requests_ok=len(statuses) == len(jobs) and all(s == 'ok' for s in statuses))
        write(output / 'run.json', client.clean(meta))
        rows = [json.loads(line) for line in (output / 'outputs.jsonl').read_text().splitlines() if line.strip()]
        write(output / 'remaining.json', {'model': client.clean(client.model), 'suite_sha256': meta['suite_sha256'],
              'created_utc': now(), 'successful_requests': meta['ok_count'],
              'run_sha256': sha(output / 'run.json'), 'outputs_sha256': sha(output / 'outputs.jsonl'),
              'remaining_requests': len(rows) - meta['ok_count'],
              'requests': [{k:r[k] for k in ('id','condition','input_sha256','status')} for r in rows if r['status']!='ok'],
              'policy':'Preserve successful calls; defer remaining requests to a later run without automatic retry.'})
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if view == 'smoke':
        from .batch import smoke_gate
        write(output / 'smoke-gate.json', smoke_gate(output))
    return meta
