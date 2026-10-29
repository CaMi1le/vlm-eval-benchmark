"""Read completed summaries and build an offline dashboard; no runner imports."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

AXES = [
    ('MMStar', '综合感知与推理', '覆盖六类综合视觉题，细分能力和样本范围以本轮冻结清单为准。'),
    ('BLINK', '空间与多视角', '包括空间、多视角和其他视觉感知子任务；具体覆盖以本轮冻结清单为准。'),
    ('OCRBench', '文字与文档', '文字读取、文档问答及手写公式；依据原图复核字符和单位。'),
    ('ChartQA', '图表理解', '图表读取与数值推理；数值题沿用原容差规则。'),
    ('POPE', '物体存在性', '图中物体是否存在的 yes/no 判断，不代表全部幻觉类型。'),
    ('MVBench', '短视频理解', '每题输入统一抽取的 8 帧，不含音频；子任务及窗口以冻结清单为准。'),
]
LABELS = {
    'minicpm47-bf16': 'MiniCPM-V 4.7 · BF16',
    'minicpm47-awq': 'MiniCPM-V 4.7 · INT4 AWQ',
    'qwen38-27b-awq': 'Qwen3.8 27B · AWQ',
    'qwen35-4b-bf16': 'Qwen3.5 4B · BF16',
    'minicpm46-awq': 'MiniCPM-V 4.6 · AWQ',
    'internvl35-8b-awq': 'InternVL3.5 8B · AWQ',
    'glm46v-flash-awq': 'GLM-4.6V-Flash · AWQ',
    'gemini-3.8-flash-high': 'Gemini 3.8 Flash High · API',
}
CONTEXT_FIELDS = ('gpu_count', 'gpu_name', 'gpu_memory_mib', 'precision', 'revision', 'downsample_mode', 'single_image_max_slices', 'multi_image_max_slices', 'backend', 'host', 'thinking', 'thinking_policy', 'max_new_tokens',
                  'max_model_len', 'max_time_seconds', 'sampling', 'eval_phase',
                  'requested_model', 'upstream_revision', 'identity_note', 'latency_kind')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def speed_score(seconds, reference=1.0):
    if not number(reference) or reference <= 0:
        raise ValueError('Speed reference must be finite and positive')
    if not number(seconds) or seconds <= 0:
        return None
    return 100 * reference / (reference + seconds)


def context_for(summary, score_directory):
    context = {k: v for k, v in (summary.get('execution_context') or {}).items() if k in CONTEXT_FIELDS}
    source = None
    selection = score_directory.parent / 'provenance/selected-runs.json'
    if selection.is_file():
        selected = json.loads(selection.read_text()).get(summary['model'])
        if selected:
            run = score_directory.parent / selected['selected_run'] / 'result/run.json'
            if run.is_file():
                config = json.loads(run.read_text()).get('config', {})
                context = {**{k: config[k] for k in CONTEXT_FIELDS if k in config}, **context}
                source = {'file': str(run), 'sha256': digest(run)}
    return context, source


def collect(score_paths, reference=1.0):
    speed_score(1, reference)
    models = {}; sources = []; suite = None; core_count = None; dataset_counts = None; review_hash = None; effective_ids = None; protocol = None; scoring_policy = None
    for supplied in score_paths:
        path = Path(supplied).resolve()
        if path.is_dir():
            path = path / 'comparison.json'
        original = path.read_bytes()
        summaries = json.loads(original)
        if not isinstance(summaries, list) or not summaries:
            raise ValueError(f'Expected a completed comparison.json: {path}')
        source = {'file': str(path), 'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
        sources.append(source)
        for summary in summaries:
            current_suite = summary.get('suite_sha256')
            if not current_suite or (suite is not None and suite != current_suite):
                raise ValueError('Cannot combine different frozen suites')
            suite = current_suite
            overall = summary['overall']; groups = summary['datasets']
            counts = {key: value['n'] for key, value in groups.items()}
            full_counts={key:value['run_n'] for key,value in groups.items()}
            if set(groups) != {a[0] for a in AXES} or overall['run_n'] < 144 or len(set(full_counts.values())) != 1 or sum(counts.values()) != overall['n'] or sum(full_counts.values()) != overall['run_n']:
                raise ValueError('Dashboard requires a completed core and consistent reviewed denominators')
            current_protocol=summary.get('scoring_protocol','reviewed-content')
            if protocol is not None and protocol!=current_protocol:
                raise ValueError('Scoring protocols differ; reviewed and public-label results cannot be mixed')
            protocol=current_protocol
            standard=protocol=='vlm-public-label-scoring-1'
            if standard and (not summary.get('scoring_policy_sha256') or summary.get('manual_content_review') is not False):
                raise ValueError('Public-label scoring policy evidence is missing')
            if standard:
                if scoring_policy is not None and scoring_policy!=summary['scoring_policy_sha256']:
                    raise ValueError('Public-label scoring policies differ')
                scoring_policy=summary['scoring_policy_sha256']
            current_review=summary.get('case_set_sha256') if standard else summary.get('reviewed_case_set_sha256')
            if not current_review or (review_hash is not None and current_review != review_hash):
                raise ValueError('Reviewed case sets differ or review evidence is missing')
            review_hash=current_review
            ids=summary.get('effective_case_ids')
            if not isinstance(ids,list) or len(set(ids)) != overall['n'] or (effective_ids is not None and ids != effective_ids):
                raise ValueError('Effective case sets differ')
            effective_ids=ids
            if core_count is not None and (core_count != overall['n'] or counts != dataset_counts):
                raise ValueError('Cannot combine different completed core views')
            core_count, dataset_counts = overall['n'], counts
            if any(not isinstance(groups[a[0]]['correct'], int) or not 0 <= groups[a[0]]['correct'] <= groups[a[0]]['n'] for a in AXES):
                raise ValueError('Invalid dataset correct counts')
            if sum(groups[a[0]]['correct'] for a in AXES) != overall['correct']:
                raise ValueError('Dataset totals do not match core')
            if not isinstance(overall['ok'], int) or not 0 <= overall['ok'] <= overall['run_n']:
                raise ValueError('Invalid successful request count')
            context, context_source = context_for(summary, path.parent)
            backend = context.get('backend', '')
            group = 'api' if backend == 'openai_compatible_api' else 'local' if backend.startswith(('transformers_', 'vllm_')) else 'unknown'
            latency = overall['total_seconds']
            median = latency.get('median'); p90 = latency.get('p90')
            # Fast successful fragments must not make a failed/unfinished run look fast.
            complete_timing = overall['ok'] == overall['run_n'] and latency.get('n') == overall['run_n'] and number(median) and median > 0
            record = {
                'accuracy': {a[0]: {'n': groups[a[0]]['n'], 'correct': groups[a[0]]['correct'],
                                  'percent': 100 * groups[a[0]]['correct'] / groups[a[0]]['n'] if groups[a[0]]['n'] else None} for a in AXES},
                'core': {'n': core_count, 'correct': overall['correct'], 'ok': overall['ok'], 'run_n': overall['run_n'], 'failed': overall['run_n'] - overall['ok']},
                'speed': {'score': speed_score(median, reference) if complete_timing else None,
                          'median_seconds': median if number(median) and median > 0 else None,
                          'p90_seconds': p90 if number(p90) and p90 > 0 else None,
                          'timed_requests': latency.get('n', 0), 'scope': 'core-image-successful-requests',
                          'missing_reason': None if complete_timing else f'成功请求及有效计时未覆盖原始全部 {overall["run_n"]} 题，速度轴留空。'},
                'outputs_sha256': summary.get('outputs_sha256'),
                'summary_source_sha256': source['sha256'],
            }
            if summary['model'] in models:
                raise ValueError('Only one reviewed result per model: ' + summary['model'])
            label=LABELS.get(summary['model'],summary['model'])
            if group=='local' and isinstance(context.get('thinking'),bool):
                label+=' · thinking' if context['thinking'] else ' · 非 thinking'
            models[summary['model']]={'id':summary['model'],'label':label,
                                      'group':group,'context':context,'context_source':context_source,'metrics':record}
    return {'schema': 'vlm-seven-axis-2' if protocol=='vlm-public-label-scoring-1' else 'vlm-reviewed-seven-axis-1', 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'scoring_protocol':protocol,'grading_note':'冻结公开标签自动评分；统一最终答案提取，格式单列，未做全量人工复核。' if protocol=='vlm-public-label-scoring-1' else '历史统一内容复核成绩。',
            'suite_sha256': suite, ('case_set_sha256' if protocol=='vlm-public-label-scoring-1' else 'reviewed_case_set_sha256'):review_hash, 'core_items': core_count, 'run_items':summaries[0]['overall']['run_n'], 'dataset_counts': dataset_counts,
            'axes': [{'id': a, 'label': b, 'description': c} for a, b, c in AXES] +
                    [{'id': 'speed', 'label': '响应速度', 'description': '完整响应耗时中位数的固定尺度映射；越快越外。'}],
            'speed_reference_seconds': reference, 'speed_formula': '100 × reference_seconds / (reference_seconds + median_seconds)',
            'models': list(models.values()), 'sources': sources}


def build_dashboard(score_paths, output, reference=1.0):
    data = collect(score_paths, reference)
    out = Path(output).resolve()
    if any(out.is_relative_to(Path(s['file']).parent) for s in data['sources']):
        raise ValueError('Dashboard output must be separate from input score directories')
    contents = []
    for source in data['sources']:
        content = Path(source['file']).read_bytes()
        if hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError('Source changed while building; retry with a completed snapshot')
        contents.append(content)
    # Published dashboards refer to their own snapshots, not the creator's disk.
    data['sources'] = [{**source, 'file': f'source-summaries/{i:02d}-comparison.json'}
                       for i, source in enumerate(data['sources'], 1)]
    payload = json.dumps(data, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    from .dashboard_view import TEMPLATE
    html = TEMPLATE.replace('DASHBOARD_PAYLOAD', payload)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'dashboard.html').write_text(html)
    (out / 'data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    snapshots = out / 'source-summaries'; snapshots.mkdir()
    for i, content in enumerate(contents, 1):
        (snapshots / f'{i:02d}-comparison.json').write_bytes(content)
    manifest = {'schema': data['schema'], 'generated_utc': data['generated_utc'], 'sources': data['sources'],
                'files': [{'file': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                          for p in sorted(out.rglob('*')) if p.is_file()]}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return data
