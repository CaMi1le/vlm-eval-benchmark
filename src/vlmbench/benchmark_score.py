"""Offline public-label grading; human review is an optional separate workflow."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re

from .answer_parser import PROTOCOL, parse_answer
from .common import json_hash, jsonl, now, sha, verify_suite, write
from .score import aggregate, api_token_usage, distribution
from .quantities import compatible,quantity

UNIT_FILE=Path(__file__).with_name('data')/'answer-units-public-v2.json'

POLICY = {
    'schema':'vlm-public-label-scoring-1','answer_parser':PROTOCOL,
    'labels':'Unmodified frozen published labels. All planned core image questions count; no manual exclusions/corrections.',
    'extraction':'Same gold-independent parser for every model; answer field only, never provider reasoning_content.',
    'mcq':'Explicit final declaration, answer box, standalone terminal letter, leading option label, or exact option text; never search the reference in reasoning.',
    'yesno':'Explicit final yes/no, ignoring case and presentation wrappers; unresolved counts wrong, never defaults to yes.',
    'OCRBench':'Published substring rule: case-insensitive except handwritten formula; formula removes spaces, retaining case.',
    'ChartQA':'5% relative tolerance for scalar numbers; zero must be numerically zero. Otherwise normalized case-insensitive exact match. Lists require all members; year lists have no numeric tolerance.',
    'chart_adaptations':'Uniform answer-box/Markdown/answer-prefix removal, thousands separators, finite numeric formatting, ratio/list separators, and input-bound physical unit conversion. Explicit units must have the correct currency, dimension and magnitude. Bare-number label/base conventions are fixed in unit metadata, not selected per model.',
    'chart_sentence_extraction':'Gold-independent final-sentence quantities, unique year declarations, quoted entities and single copular conclusions. Retain explicit scales/units; reject alternatives and conflicting quantities. Never search thoughts or reference strings for a matching answer.',
    'chart_terminal_extraction':'Before sentence extraction, prefer a separate short final paragraph or standalone terminal quantity/list/yes-no; retain all declared units, alternatives and unknown text on that surface. Answer boxes retain priority. No reference-dependent selection or search in thought blocks.',
    'unit_metadata_sha256':sha(UNIT_FILE) if UNIT_FILE.exists() else None,
    'format':'Strict final-response format is reported separately and does not decide content correctness. Native thought blocks and terminal control tokens are excluded; Markdown and explanations remain format differences.',
    'failures':'Failures, empty and unresolved final answers stay in the planned denominator.',
    'recovery':'Select first-run successes and explicit timeout recovery outcomes only; preserve both attempts. Speed uses per-case cumulative observed request time for recovered cases, excluding between-run waiting.',
    'source_framework_commit':'1e5bd63935e2a506d574b9579bb3cae867c4d4bb',
    'source_files':[
        'vendor/VLMEvalKit/vlmeval/utils/matching_util.py',
        'vendor/VLMEvalKit/vlmeval/dataset/utils/ocrbench.py',
        'vendor/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py',
        'vendor/VLMEvalKit/vlmeval/dataset/utils/yorn.py'],
    'scope':'Internal stratified subsets, shared original inputs. Not full official leaderboards or equal-hardware/compute ranking.'}


def scalar(text):
    value=str(text).strip().rstrip('.').strip()
    if re.fullmatch(r'[+-]?\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?%?',value):value=value.replace(',','').replace(' ','')
    match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(%)?',value)
    if not match:return None
    number=float(match.group(1))
    return (number,bool(match.group(2))) if math.isfinite(number) else None


def list_value(text):
    value=str(text).strip()
    if value.startswith('[') and value.endswith(']'):value=value[1:-1]
    # A valid thousands-formatted scalar must not become a list.
    if scalar(value) is not None:return None
    parts=re.split(r'\s*(?:,|\band\b|;)\s*',value,flags=re.I)
    if len(parts)<2 or any(not p for p in parts):return None
    return sorted(re.sub(r'\s+',' ',p.strip(' \'"')).casefold() for p in parts)


def chart_correct(question, prediction, target, context=None):
    if context is not None and scalar(target) is not None:
        return compatible(prediction,target,context,question)
    if re.search(r'\bratio\b',question,re.I) and scalar(target) is not None:
        return compatible(prediction,target,{'dimension':'ratio','reference_scale':1},question)
    got,expected=scalar(prediction),scalar(target)
    if got is not None and expected is not None:
        a,ap=got;b,bp=expected
        if re.search(r'\b(?:what|which)\s+(?:(?:is|was)\s+the\s+)?year\b',question,re.I):
            return a==b and ap==bp
        percentage_question=bool(re.search(r'\bpercent(?:age)?\b',question,re.I)) and not re.match(
            r'\s*(?:what|which)\s+(?:year|company|gender|platform)',question,re.I)
        # Labels such as 5.1 are percentage points for "what percentage ...?".
        if not (percentage_question and not bp):
            if ap:a/=100
            if bp:b/=100
        return abs(a-b)<=abs(b)*.05+1e-12 if b else a==0
    left,right=list_value(prediction),list_value(target)
    if left is not None or right is not None:return left is not None and right is not None and left==right
    norm=lambda v:re.sub(r'\s+',' ',str(v).strip()).replace('–','-').replace('—','-').casefold()
    if re.search(r'increase or decrease',question,re.I):
        unchanged={'no','neither','constant','neither increase nor decrease','stays the same'}
        if norm(prediction) in unchanged and norm(target) in unchanged:return True
    return norm(prediction)==norm(target)


def score_answer(case, reference, answer, status='ok'):
    parsed=parse_answer(case,answer)
    task=case['task'];value=parsed['parsed'];correct=False
    if status=='ok' and value is not None and str(value).strip():
        if task in ('mcq','yesno'):correct=value in reference['answers']
        elif task=='chart':correct=any(chart_correct(case['question'],value,g,case.get('unit_context')) for g in reference['answers'])
        elif task=='ocr':
            formula=case['category']=='Handwritten Mathematical Expression Recognition'
            def norm(text):
                text=str(text).strip().replace('\n',' ')
                return text.replace(' ','') if formula else text.lower()
            correct=any(norm(g) in norm(value) for g in reference['answers'])
        else:raise ValueError('Unknown task: '+task)
    raw=str(answer).strip()
    format_text=re.sub(r'<think>.*?</think>','',raw,flags=re.S|re.I).strip()
    if '<think>' in format_text.lower():format_text=''
    format_text=re.sub(r'(?:<\|(?:im_end|endoftext|user|assistant|end|eot_id)\|>\s*)+$','',format_text).strip()
    fmt=format_text in [chr(65+i) for i in range(len(case['choices']))] if task=='mcq' else format_text.lower() in ('yes','no') if task=='yesno' else None
    return {**parsed,'correct':bool(correct),'format_ok':fmt,'format_scope':'Final response excluding native thought blocks/control tokens','eligible':True,
            'unit_context':case.get('unit_context'),'quantity_candidates':quantity(value,case.get('unit_context'),case.get('question','')) if task=='chart' and value is not None else None,
            'format_issue':fmt is False,'extraction_method':parsed['method'],'extraction_issue':parsed['issue'],
            'scoring_answer':value,'normalization_applied':str(value).strip()!=raw if value is not None else False}


def summary_for(rows):
    result=aggregate(rows)
    result['total_seconds']=distribution([r.get('timing_seconds') for r in rows if r['status']=='ok'])
    result['extraction_methods']=dict(Counter(r['extraction_method'] for r in rows))
    result['unresolved_final_answers']=sum(r['parsed'] is None for r in rows)
    return result


def with_units(suite,cases,refs):
    if UNIT_FILE.exists():
        units=json.loads(UNIT_FILE.read_text())
        if units['suite_sha256']==sha(Path(suite)/'freeze.json'):
            for cid,context in units['cases'].items():
                assert context['input_sha256']==cases[cid]['input_sha256'],'Unit metadata input mismatch'
                assert context['reference_sha256']==json_hash(refs[cid]),'Unit metadata reference mismatch'
                assert context['media_sha256']==[m['sha256'] for m in cases[cid]['media']],'Unit metadata image mismatch'
                cases[cid]={**cases[cid],'unit_context':context}
    return cases


def score_run(suite, run, destination, *, cases=None, references=None, views=None):
    suite,run,out=Path(suite),Path(run),Path(destination)
    cases=cases or {c['id']:c for c in verify_suite(suite)}
    refs=references or {r['id']:r for r in jsonl(suite/'references.jsonl')}
    cases=with_units(suite,cases,refs)
    views=views or json.loads((suite/'views.json').read_text())
    meta=json.loads((run/'run.json').read_text());raw=jsonl(run/'outputs.jsonl')
    planned=json.loads((run/'planned.json').read_text())
    assert meta['suite_sha256']==sha(suite/'freeze.json'),'Wrong frozen suite'
    keys={(r['id'],r['condition']) for r in raw}
    expected={(i,'image') for i in views['core']} | {(i,'no_image') for i in views['no_image_diagnostic']}
    assert keys==expected=={(r['id'],r['condition']) for r in planned}
    assert len(raw)==len(planned)==len(keys),'Duplicate or incomplete responses'
    assert all(r['model']==meta['model'] and r['input_sha256']==cases[r['id']]['input_sha256'] for r in raw),'Changed input/model'
    rows=[]
    for row in raw:
        case=cases[row['id']]
        recovery=row.get('recovery',{})
        elapsed=row.get('total_seconds')
        if elapsed is not None:elapsed+=recovery.get('first_attempt_seconds') or 0
        rows.append({**row,'raw_row_sha256':json_hash(row),'task':case['task'],'gold':refs[row['id']]['answers'],
            **score_answer(case,refs[row['id']],row.get('answer',''),row['status']),
            'timing_seconds':elapsed,'case_review_reason':'Frozen published reference; no manual change.',
            'review_reason':'Automatic '+PROTOCOL+'; not a manual content review.'})
    core=[r for r in rows if r['condition']=='image']
    groups=defaultdict(list);categories=defaultdict(list)
    for row in core:groups[row['dataset']].append(row);categories[row['dataset']+'/'+row['category']].append(row)
    summary={'model':meta['model'],'scored_utc':now(),'suite_sha256':meta['suite_sha256'],
        'outputs_sha256':sha(run/'outputs.jsonl'),'references_sha256':sha(suite/'references.jsonl'),
        'scoring_protocol':POLICY['schema'],'scoring_policy_sha256':json_hash(POLICY),
        'scoring_code_sha256':sha(__file__),'parser_code_sha256':sha(Path(__file__).with_name('answer_parser.py')),
        'chart_parser_code_sha256':sha(Path(__file__).with_name('chart_answers.py')),
        'quantity_code_sha256':sha(Path(__file__).with_name('quantities.py')),
        'case_set_sha256':json_hash([{'id':cases[i]['id'],'input_sha256':cases[i]['input_sha256'],'reference':refs[i]} for i in sorted(views['core'])]),
        'effective_case_ids':sorted(views['core']),
        'overall':summary_for(core),'datasets':{d:summary_for(v) for d,v in groups.items()},
        'categories':{c:summary_for(v) for c,v in categories.items()},
        'views':{v:summary_for([r for r in core if r['id'] in ids]) for v,ids in views.items() if v in ('regression','delta')},
        'no_image_diagnostic':summary_for([r for r in rows if r['condition']=='no_image']),
        'execution_context':{**meta['config']},'api_token_usage':api_token_usage(rows),
        'scope':POLICY['scope'],'manual_content_review':False}
    if any(r.get('recovery') for r in rows):
        summary['execution_context']['latency_kind']='Per-case cumulative observed client request time, including original timeout for recovered cases; excludes between-run waiting.'
        summary['timeout_recovery']={'requests':sum(bool(r.get('recovery')) for r in rows),
            'source_attempts_preserved':True,'timing_kind':summary['execution_context']['latency_kind']}
    out.mkdir(parents=True,exist_ok=False)
    write(out/'summary.json',summary)
    (out/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return summary,rows


def compare(suite, runs, destination):
    suite,out=Path(suite),Path(destination)
    cases={c['id']:c for c in verify_suite(suite)};refs={r['id']:r for r in jsonl(suite/'references.jsonl')}
    cases=with_units(suite,cases,refs)
    views=json.loads((suite/'views.json').read_text());out.mkdir(parents=True,exist_ok=False)
    summaries=[];allrows={}
    for run in runs:
        run=Path(run);name=json.loads((run/'run.json').read_text())['model']
        assert name not in allrows,'Duplicate model'
        if not isinstance(name,str) or not name or not (out/name).resolve().is_relative_to(out.resolve()) or (out/name).resolve()==out.resolve():
            raise ValueError('Model identifier escapes score output directory')
        result,rows=score_run(suite,run,out/name,cases=cases,references=refs,views=views)
        summaries.append(result);allrows[name]=rows
    assert len({s['case_set_sha256'] for s in summaries})==1
    write(out/'comparison.json',summaries);write(out/'scoring-policy.json',POLICY)
    with (out/'comparison.csv').open('w',newline='') as stream:
        fields=['model','id','dataset','category','condition','status','gold','answer','parsed','correct','format_ok','extraction_method','extraction_issue','normalization_applied','total_seconds','timing_seconds','raw_row_sha256']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for rows in allrows.values():writer.writerows(rows)
    unresolved=[r for rows in allrows.values() for r in rows if r['parsed'] is None]
    (out/'unresolved.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unresolved))
    return summaries,allrows
