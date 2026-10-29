"""One content grade, using complete, hash-bound visual/text review evidence.

Canonical answers are reviewed interpretations, never extracted by searching for
the reference in a model's reasoning. Unreviewed or changed inputs fail closed.
"""
import json
import math
import re
from pathlib import Path
from .common import json_hash, jsonl, sha

SCHEMA = 'vlm-content-review-1'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def text(value, formula=False):
    value = str(value).strip().strip('$').strip()
    return re.sub(r'\s+', '', value) if formula else re.sub(r'\s+', ' ', value).casefold()

UNITS = {
    'ratio': ('ratio', 1), 'percent': ('ratio', .01),
    'count': ('count', 1), 'million_count': ('count', 1e6), 'billion_count': ('count', 1e9),
    'USD': ('USD', 1), 'million_USD': ('USD', 1e6), 'billion_USD': ('USD', 1e9),
    'GBP': ('GBP', 1), 'million_GBP': ('GBP', 1e6), 'billion_GBP': ('GBP', 1e9),
    'tonne_per_hectare': ('yield', 1), 'kg_per_hectare': ('yield', .001),
    'per_thousand': ('per_thousand', 1), 'index': ('index', 1),
    'per_100000': ('ratio', .00001), 'year': ('time_year', 1),
}

def quantity(value, target):
    if not isinstance(value, dict) or set(value) != {'value', 'unit'}:
        return None
    number = value['value']
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
        return None
    source = UNITS.get(value['unit']); destination = UNITS.get(target)
    if source is None or destination is None or source[0] != destination[0]:
        return None
    return number * source[1] / destination[1]

def equivalent(case, value, gold):
    kind = case['kind']
    if value is None:
        return False
    if kind == 'number':
        got = quantity(value, case['unit']); expected = quantity(gold, case['unit'])
        if got is None or expected is None:
            return False
        tolerance = case.get('relative_tolerance', 0)
        return abs(got - expected) <= abs(expected) * tolerance + 1e-12
    if kind == 'list':
        # These questions ask for years; no 5% year tolerance or missing members.
        return isinstance(value, list) and sorted(value) == sorted(gold)
    if kind in ('text', 'formula'):
        return text(value, kind == 'formula') == text(gold, kind == 'formula')
    if kind in ('choice', 'yesno', 'trend'):
        return value == gold
    raise ValueError('Unknown reviewed answer kind: ' + kind)

def grade(case, judgment, status='ok'):
    eligible = case['decision'] != 'exclude'
    correct = status == 'ok' and any(equivalent(case, judgment['canonical'], g) for g in case['answers'])
    return {'eligible': eligible, 'correct': correct if eligible else None,
            'parsed': judgment['canonical'], 'review_reason': judgment['reason'],
            'case_review_reason': case['reason'], 'case_decision': case['decision'],
            'format_issue': judgment['format_issue']}

def load_review(manifest_path, suite, cases, references, model, run, raw):
    path = Path(manifest_path); manifest = json.loads(path.read_text()); root = path.parent
    require(manifest.get('schema') == SCHEMA, 'Missing or unsupported review manifest')
    require(manifest['suite_sha256'] == sha(Path(suite) / 'freeze.json'), 'Review suite hash mismatch')
    for name in ('cases.jsonl', 'answers.jsonl', 'policy.json'):
        require(manifest['files'].get(name) == sha(root / name), 'Review evidence hash mismatch: ' + name)
    policy = json.loads((root / 'policy.json').read_text())
    require(policy.get('schema') == SCHEMA, 'Missing review policy')
    decisions = jsonl(root / 'cases.jsonl'); byid = {c['id']: c for c in decisions}
    require(len(byid) == len(decisions) == len(cases) and set(byid) == set(cases), 'Incomplete case review')
    require(manifest['case_count'] == len(decisions), 'Review case count mismatch')
    for cid, case in cases.items():
        reviewed = byid[cid]
        require(reviewed['input_sha256'] == case['input_sha256'] and reviewed['case_sha256'] == json_hash(case)
                and reviewed['reference_sha256'] == json_hash(references[cid]), 'Case review hash mismatch: ' + cid)
        require(reviewed.get('reason') and reviewed['decision'] in ('retain', 'correct_reference', 'exclude'), 'Missing case rationale')
        require(reviewed.get('kind') in ('choice', 'yesno', 'text', 'formula', 'number', 'list', 'trend'), 'Missing answer type')
        require(reviewed['decision'] == 'exclude' or reviewed.get('answers'), 'Missing reviewed reference')
    source = manifest['runs'].get(model)
    require(source is not None, 'Model has no review evidence')
    for name in ('outputs.jsonl', 'run.json', 'planned.json'):
        require(source.get(name) == sha(Path(run) / name), 'Reviewed run hash mismatch: ' + name)
    answers = jsonl(root / 'answers.jsonl')
    require(manifest['answer_count'] == len(answers), 'Review answer count mismatch')
    require(len({(a['model'], a['id'], a['condition']) for a in answers}) == len(answers), 'Duplicate answer reviews')
    require(set(a['model'] for a in answers) == set(manifest['runs']), 'Incomplete model review')
    for name, expected in manifest['runs'].items():
        require(sum(a['model'] == name for a in answers) == expected['answer_count'], 'Incomplete answer review: ' + name)
    judgments = {(a['id'], a['condition']): a for a in answers if a['model'] == model}
    require(set(judgments) == {(r['id'], r['condition']) for r in raw}, 'Incomplete answer review')
    for row in raw:
        judgment = judgments[(row['id'], row['condition'])]
        require(judgment['raw_row_sha256'] == json_hash(row) and judgment['input_sha256'] == row['input_sha256'],
                'Answer review hash mismatch: ' + row['id'])
        require(judgment.get('reason') and 'canonical' in judgment and isinstance(judgment.get('format_issue'), bool), 'Missing answer rationale')
        # Validate typed values, including excluded cases, so malformed evidence cannot pass silently.
        result = grade(byid[row['id']], judgment, row['status'])
        require('correct' not in judgment or judgment['correct'] == result['correct'], 'Reviewed verdict conflicts with canonical content')
    return byid, judgments, manifest, policy
