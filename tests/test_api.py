import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest
from PIL import Image

from vlmbench.api import APIClient, APIError, provider_limit, run_api
from vlmbench.common import json_hash, jsonl, sha, write
from vlmbench.score import compare, score_run
from review_fixture import review_fixture


KEY = 'test-credential-never-write-to-artifacts'


@contextmanager
def api_server(tmp_path, replies):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            entry = {'path': self.path, 'auth': self.headers.get('Authorization'),
                     'body': json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                     'started_monotonic': time.monotonic()}
            received.append(entry)
            index = received.index(entry)
            reply = replies[min(index, len(replies) - 1)]
            status, data = reply[:2]
            if len(reply) > 2:
                time.sleep(reply[2])
            entry['ready_monotonic'] = time.monotonic()
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    credentials = tmp_path / 'credentials.json'
    write(credentials, {'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                        'api_key': KEY, 'model': 'test-high'})
    try:
        yield credentials, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def reply(answer='A', **extra):
    return (200, {'model': 'proxy-model-id', 'id': 'response-1',
                  'choices': [{'message': {'content': answer, 'reasoning_content': 'private analysis'}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 111,
                            'completion_tokens_details': {'reasoning_tokens': 100}}, **extra})


def suite_fixture(root, count=5):
    root.mkdir()
    Image.new('RGB', (4, 3), 'blue').save(root / 'frame.png')
    media = [{'path': 'frame.png', 'sha256': sha(root / 'frame.png'), 'width': 4, 'height': 3,
              'timestamp_seconds': second} for second in (0.125, 0.875)]
    cases = []
    for i in range(count):
        case = {'id': f'fixture:{i}', 'dataset': 'MMStar', 'category': 'fixture', 'task': 'mcq',
                'question': 'Which?', 'choices': ['blue', 'red'], 'media': media,
                'prompt': 'Which?\nA. blue\nB. red\nAnswer with only the option letter.'}
        case['input_sha256'] = json_hash({k: case[k] for k in ['question', 'choices', 'media', 'task', 'prompt']})
        cases.append(case)
    (root / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    (root / 'references.jsonl').write_text(''.join(json.dumps({'id': c['id'], 'answers': ['A']}) + '\n' for c in cases))
    write(root / 'views.json', {'smoke': [cases[0]['id']], 'core': [c['id'] for c in cases],
                               'no_image_diagnostic': [cases[0]['id']]})
    names = ['cases.jsonl', 'references.jsonl', 'views.json', 'frame.png']
    write(root / 'freeze.json', {'case_count': count,
          'files': [{'file': n, 'bytes': (root / n).stat().st_size, 'sha256': sha(root / n)} for n in names]})
    return cases


def test_no_timeout_waits_for_delayed_response(tmp_path):
    root = tmp_path / 'suite'
    case = suite_fixture(root)[0]
    with api_server(tmp_path, [(*reply(), 0.08)]) as (credentials, received):
        with pytest.raises(APIError, match='[Tt]imed out|deadline'):
            APIClient(credentials, timeout=0.01).generate(case, root)
        client = APIClient(credentials, timeout=None)
        assert client.public_config['timeout_seconds'] is None
        assert client.generate(case, root)['answer'] == 'A'
        # The first deadline may expire during preprocessing on a loaded host.
        assert 1 <= len(received) <= 2


def test_preprocessing_deadline_is_wrapped_without_opening_connection(tmp_path, monkeypatch):
    from vlmbench import api
    root = tmp_path / 'suite'
    case = suite_fixture(root)[0]
    credentials = tmp_path / 'credentials.local.json'
    write(credentials, {'base_url': 'http://127.0.0.1:1/v1', 'api_key': KEY, 'model': 'fixture'})
    client = APIClient(credentials, timeout=0.01)
    times = iter([0.0, 1.0])
    monkeypatch.setattr(api.time, 'perf_counter', lambda: next(times))
    def unexpected_connection(*args, **kwargs):
        raise AssertionError('Expired request must not create a connection')
    monkeypatch.setattr(api.http.client, 'HTTPConnection', unexpected_connection)
    with pytest.raises(APIError, match='TimeoutError: API request deadline exceeded'):
        client.generate(case, root)


def test_skip_warmup_issues_only_planned_requests(tmp_path):
    root = tmp_path / 'suite'
    cases = suite_fixture(root, count=2)
    output = tmp_path / 'run'
    with api_server(tmp_path, [reply()]) as (credentials, received):
        meta = run_api(root, credentials, output, timeout=None, concurrency=4, skip_warmup=True)
    assert meta['all_requests_ok'] and meta['config']['timeout_seconds'] is None
    assert len(received) == len(cases) + 1 == meta['planned_requests']
    assert json.loads((output / 'warmup.json').read_text())['status'] == 'skipped'


@pytest.mark.parametrize('flags,expected', [([], 300), (['--timeout', '12'], 12), (['--no-timeout'], None)])
def test_cli_timeout_modes(monkeypatch, flags, expected):
    from vlmbench import api, cli
    captured = []
    monkeypatch.setattr(api, 'run_api', lambda *args: captured.append(args) or
                        {'model': 'fixture', 'view': 'core', 'output_count': 1,
                         'ok_count': 1, 'all_requests_ok': True})
    monkeypatch.setattr('sys.argv', ['vlmbench', 'api-run', '--suite', 'suite', '--output', 'output',
                                  '--credentials', 'credentials.local.json', '--skip-warmup', *flags])
    cli.main()
    assert captured[0][4] == expected and captured[0][7] is True


def test_original_images_timestamps_and_no_image_protocol(tmp_path):
    root = tmp_path / 'suite'
    case = suite_fixture(root)[0]
    with api_server(tmp_path, [reply()]) as (credentials, received):
        client = APIClient(credentials)
        messages = client.messages(case, root)
        content = messages[0]['content']
        assert content[0] == {'type': 'text', 'text': 'Frame at 0.125000 seconds:'}
        assert content[2]['text'] == 'Frame at 0.875000 seconds:'
        assert base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1]) == (root / 'frame.png').read_bytes()
        assert content[-1] == {'type': 'text', 'text': case['prompt']}
        assert client.messages(case, root, True)[0]['content'] == [content[-1]]
        assert len(client.messages({**case, 'media': case['media'][:1]}, root)[0]['content']) == 3
        result = client.generate(case, root)
        assert received[0]['auth'] == 'Bearer ' + KEY
        assert received[0]['path'] == '/v1/chat/completions'
        assert received[0]['body']['messages'] == messages
        assert received[0]['body']['temperature'] == 0 and received[0]['body']['max_tokens'] == 128
        assert result['answer'] == 'A' and result['response_model'] == 'proxy-model-id'
        assert result['reasoning_tokens'] == 100 and result['generated_tokens'] == 1
        assert result['ttft_seconds'] is None and result['generation_seconds'] is None
        (root / 'frame.png').write_bytes(b'tampered')
        with pytest.raises(ValueError, match='SHA256'):
            client.messages(case, root)


def test_successful_run_is_scoreable_and_does_not_save_credentials(tmp_path):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root)
    with api_server(tmp_path, [reply()]) as (credentials, received):
        meta = run_api(root, credentials, output)
        assert meta['all_requests_ok'] and meta['planned_requests'] == 6
        assert len(received) == 7  # One warmup, five image requests, one diagnostic.
        assert received[-1]['body']['messages'][0]['content'] == [{'type': 'text', 'text': jsonl(root/'cases.jsonl')[0]['prompt']}]
        assert len(jsonl(output / 'outputs.jsonl')) == 6
        result = score_run(root, output, tmp_path / 'scores', review_fixture(root,[output],tmp_path/'review'))
        assert result['overall']['correct'] == result['overall']['n'] == 5
        assert result['overall']['reasoning_tokens']['median'] == 100
        assert result['overall']['tokens_per_generation_second']['n'] == 0
        assert result['api_token_usage']['total_tokens']['sum'] == 666
        assert result['api_token_usage']['reasoning_tokens']['sum'] == 600
        for p in output.iterdir():
            assert KEY not in p.read_text()
        with pytest.raises(FileExistsError):
            run_api(root, credentials, output)


def test_three_errors_stop_requests_and_preserve_scoring_denominator(tmp_path):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root)
    with api_server(tmp_path, [reply(), (503, {'error': 'echo ' + KEY})]) as (credentials, received):
        meta = run_api(root, credentials, output)
        assert not meta['all_requests_ok'] and len(received) == 4
        rows = jsonl(output / 'outputs.jsonl')
        assert [r['status'] for r in rows] == ['error'] * 3 + ['not_run'] * 3
        assert '[REDACTED]' in rows[0]['error']
        result = score_run(root, output, tmp_path / 'scores', review_fixture(root,[output],tmp_path/'review'))
        assert result['overall']['n'] == result['overall']['failed'] == 5
        for p in output.iterdir():
            assert KEY not in p.read_text()


def test_warmup_failure_and_empty_answers_fail_gate(tmp_path):
    root = tmp_path / 'suite'
    suite_fixture(root)
    with api_server(tmp_path, [reply('')]) as (credentials, received):
        output = tmp_path / 'warmup-error'
        meta = run_api(root, credentials, output, view='smoke')
        assert len(received) == 1 and not meta['all_requests_ok']
        assert len(jsonl(output / 'outputs.jsonl')) == 2
        assert not json.loads((output / 'smoke-gate.json').read_text())['passed']
    with api_server(tmp_path, [reply(), reply('')]) as (credentials, received):
        output = tmp_path / 'empty-output'
        meta = run_api(root, credentials, output, view='smoke')
        assert len(received) == 3 and not meta['all_requests_ok']
        assert all(r['status'] == 'error' for r in jsonl(output / 'outputs.jsonl'))


def test_timeout_and_redaction_of_successful_response(tmp_path):
    root = tmp_path / 'suite'
    case = suite_fixture(root)[0]
    with api_server(tmp_path, [reply(KEY)]) as (credentials, received):
        result = APIClient(credentials).generate(case, root)
        assert result['answer'] == '[REDACTED]'
    with api_server(tmp_path, [(*reply(), 0.1)]) as (credentials, received):
        with pytest.raises(APIError, match='(?i)(timeout|timed out|deadline)'):
            APIClient(credentials, timeout=0.02).generate(case, root)
        assert len(received) == 1


def test_interruption_fills_remaining_jobs(tmp_path, monkeypatch):
    root = tmp_path / 'suite'
    suite_fixture(root)
    with api_server(tmp_path, [reply()]) as (credentials, received):
        original = APIClient.generate
        calls = 0

        def interrupted(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt('interrupted')
            return original(self, *args, **kwargs)

        monkeypatch.setattr(APIClient, 'generate', interrupted)
        output = tmp_path / 'interrupted'
        meta = run_api(root, credentials, output)
        rows = jsonl(output / 'outputs.jsonl')
        assert len(rows) == 6 and rows[0]['status'] == 'ok'
        assert all(r['status'] == 'not_run' for r in rows[1:])
        assert 'KeyboardInterrupt' in meta['fatal_error']


def test_credentials_reloaded_at_start_and_unsafe_urls_rejected(tmp_path):
    credentials = tmp_path / 'credentials.json'
    write(credentials, {'base_url': 'http://127.0.0.1:8045/v1', 'api_key': KEY, 'model': 'first'})
    assert APIClient(credentials).model == 'first'
    assert APIClient(credentials).timeout == 300
    for timeout in [0, -1, float('inf'), float('nan')]:
        with pytest.raises(ValueError, match='finite positive'):
            APIClient(credentials, timeout=timeout)
    write(credentials, {'base_url': 'http://127.0.0.1:8045/v1', 'api_key': KEY, 'model': 'second'})
    assert APIClient(credentials).model == 'second'
    write(credentials, {'base_url': 'http://host/v1?api_key=' + KEY, 'api_key': KEY, 'model': 'second'})
    with pytest.raises(ValueError, match='without userinfo') as exc:
        APIClient(credentials)
    assert KEY not in str(exc.value)


def test_parallel_requests_are_bounded_and_preserve_complete_outputs(tmp_path):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root, count=8)
    with api_server(tmp_path, [(*reply(), 0.06)]) as (credentials, received):
        meta = run_api(root, credentials, output, concurrency=4)
    assert meta['all_requests_ok'] and len(received) == 10
    assert meta['config']['concurrency'] == 4
    assert meta['scheduler']['peak_in_flight'] == 4
    assert meta['formal_wall_seconds'] > 0
    events = [(r['started_monotonic'], 1) for r in received[1:]]
    events += [(r['ready_monotonic'], -1) for r in received[1:]]
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    assert 2 <= peak <= 4
    rows = jsonl(output / 'outputs.jsonl')
    planned = json.loads((output / 'planned.json').read_text())
    assert len(rows) == 9 and len({(r['id'], r['condition']) for r in rows}) == 9
    assert {(r['id'], r['condition']) for r in rows} == {(r['id'], r['condition']) for r in planned}
    for path in output.iterdir():
        assert KEY not in path.read_text()


def test_parallel_errors_stop_dispatch_and_drain_in_flight_requests(tmp_path):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root, count=10)
    with api_server(tmp_path, [reply(), (503, {'error': KEY}, 0.04)]) as (credentials, received):
        meta = run_api(root, credentials, output, concurrency=4)
    rows = jsonl(output / 'outputs.jsonl')
    assert not meta['all_requests_ok'] and meta['scheduler']['circuit_breaker_tripped']
    assert 4 <= len(received) - 1 <= 6
    assert len(rows) == 11
    assert sum(r['status'] == 'error' for r in rows) == len(received) - 1
    assert sum(r['status'] == 'not_run' for r in rows) == 11 - (len(received) - 1)
    assert all(r['status'] in ('error', 'not_run') for r in rows)
    assert KEY not in (output / 'outputs.jsonl').read_text()


@pytest.mark.parametrize('concurrency', [1, 4])
def test_provider_cap_stops_dispatch_and_saves_remaining_without_retry(tmp_path, concurrency):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root, count=10)
    with api_server(tmp_path, [reply(), (429, {'error':'RESOURCE_EXHAUSTED'}, 0.04)]) as (credentials, received):
        meta=run_api(root,credentials,output,concurrency=concurrency)
    assert meta['scheduler']['stop_reason']=='provider_limit'
    assert len(received)==1+concurrency
    remaining=json.loads((output/'remaining.json').read_text())
    assert remaining['successful_requests']==0 and remaining['remaining_requests']==11
    assert len(remaining['requests'])==11
    assert all(r['status'] in ('error','not_run') for r in remaining['requests'])
    assert any(r.get('provider_limit_detected') for r in jsonl(output/'outputs.jsonl'))


def test_provider_cap_recognition_does_not_treat_generic_failures_as_quota():
    for text in ['HTTP 402: payment required','HTTP 429: rate limited','quota exceeded','余额不足']:
        assert provider_limit(text)
    assert not provider_limit('HTTP 503: service unavailable')


def test_resume_calls_only_remaining_requests_and_preserves_prior_outputs(tmp_path):
    root,first,second=tmp_path/'suite',tmp_path/'first',tmp_path/'second'
    suite_fixture(root)
    with api_server(tmp_path,[reply(),reply(),(429,{'error':'quota exceeded'})]) as (credentials,received):
        original=run_api(root,credentials,first)
    assert original['ok_count']==1
    original_hash=sha(first/'outputs.jsonl')
    remaining=json.loads((first/'remaining.json').read_text())
    wanted={(r['id'],r['condition']) for r in remaining['requests']}
    assert remaining['remaining_requests']==5
    with api_server(tmp_path,[reply()]) as (credentials,received):
        resumed=run_api(root,credentials,second,concurrency=4,remaining=first/'remaining.json')
    assert resumed['all_requests_ok'] and resumed['planned_requests']==5
    assert len(received)==6  # One warmup on a remaining case, five selected requests.
    rows=jsonl(second/'outputs.jsonl')
    assert {(r['id'],r['condition']) for r in rows}==wanted
    assert ('fixture:0','image') not in wanted
    assert json.loads((second/'warmup.json').read_text())['id']!='fixture:0'
    assert sha(first/'outputs.jsonl')==original_hash
    assert json.loads((second/'remaining.json').read_text())['remaining_requests']==0
    assert resumed['selection_sha256']==sha(second/'selection.json')


@pytest.mark.parametrize('invalid', ['model','suite_sha256','input_sha256','status','duplicate'])
def test_resume_rejects_changed_protocol_and_successful_requests(tmp_path,invalid):
    root=tmp_path/'suite';case=suite_fixture(root)[0]
    row={'id':case['id'],'condition':'image','input_sha256':case['input_sha256'],'status':'not_run'}
    selection={'model':'test-high','suite_sha256':sha(root/'freeze.json'),'remaining_requests':1,'requests':[row]}
    if invalid in ('model','suite_sha256'):selection[invalid]='changed'
    elif invalid=='duplicate':selection['requests'].append(row);selection['remaining_requests']=2
    elif invalid=='status':row['status']='ok'
    else:row['input_sha256']='changed'
    path=tmp_path/'remaining.json';write(path,selection)
    with api_server(tmp_path,[reply()]) as (credentials,received):
        with pytest.raises(ValueError):run_api(root,credentials,tmp_path/'output',remaining=path)
        assert not received
    assert not (tmp_path/'output').exists()


def test_parallel_interruption_saves_already_issued_responses(tmp_path, monkeypatch):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root, count=8)

    def interrupted_wait(*args, **kwargs):
        raise KeyboardInterrupt('stop dispatch')

    monkeypatch.setattr('vlmbench.api.wait', interrupted_wait)
    with api_server(tmp_path, [(*reply(), 0.04)]) as (credentials, received):
        meta = run_api(root, credentials, output, concurrency=4)
    rows = jsonl(output / 'outputs.jsonl')
    assert len(received) == 5 and len(rows) == 9
    assert sum(r['status'] == 'ok' for r in rows) == 4
    assert sum(r['status'] == 'not_run' for r in rows) == 5
    assert 'KeyboardInterrupt' in meta['fatal_error']


@pytest.mark.parametrize('concurrency', [0, -1, 9, True, 1.5])
def test_invalid_concurrency_rejected_before_requests(tmp_path, concurrency):
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='integer from 1 to 8'):
        run_api(tmp_path / 'absent-suite', tmp_path / 'absent-key', output, concurrency=concurrency)
    assert not output.exists()


def test_api_report_keeps_usage_context_and_portable_images(tmp_path):
    root, output = tmp_path / 'suite', tmp_path / 'output'
    suite_fixture(root)
    with api_server(tmp_path, [reply()]) as (credentials, received):
        run_api(root, credentials, output)
    scores = tmp_path / 'scores'
    compare(root, [output], scores, review_fixture(root,[output],tmp_path/'review'))
    html = (scores / 'review.html').read_text()
    data = json.loads(html.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0])
    assert len(data['cases']) == 5 and len(data['assets']) == 1
    assert data['execution_contexts']['test-high']['backend'] == 'openai_compatible_api'
    assert data['cases'][0]['models']['test-high']['image']['reasoning_tokens'] == 100
    assert 'Reasoning' in (scores / 'README.md').read_text()
    assert KEY not in html
