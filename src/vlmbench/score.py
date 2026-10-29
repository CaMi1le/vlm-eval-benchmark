"""Content scoring from complete, reviewed evidence. No model or judge calls."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from .common import jsonl, json_hash, now, sha, verify_suite, write
from .reviewed import grade, load_review, require

def distribution(values):
    values=sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:return {'n':0,'median':None,'p90':None,'min':None,'max':None}
    return {'n':len(values),'median':statistics.median(values),'p90':values[max(0,math.ceil(.9*len(values))-1)],'min':values[0],'max':values[-1]}

def api_token_usage(rows):
    reported=[r for r in rows if isinstance(r.get('api_usage'),dict)]
    if not reported:return None
    result={'requests_with_usage':len(reported),'planned_requests':len(rows),
            'scope':'Reported usage only; excludes warmup. Provider fields are preserved without assuming additive token semantics.'}
    for name,extract in [('prompt_tokens',lambda r:r['api_usage'].get('prompt_tokens')),
                         ('completion_tokens',lambda r:r['api_usage'].get('completion_tokens')),
                         ('reasoning_tokens',lambda r:r.get('reasoning_tokens')),
                         ('total_tokens',lambda r:r['api_usage'].get('total_tokens'))]:
        values=[extract(r) for r in reported]
        numbers=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)]
        result[name]={'sum':sum(numbers) if numbers else None,'reported_requests':len(numbers)}
    return result

def aggregate(rows):
    full_rows=rows;rows=[r for r in rows if r.get('eligible',True)]
    n=len(rows);correct=sum(bool(r['correct']) for r in rows);format_rows=[r for r in full_rows if r['format_ok'] is not None];ok=[r for r in full_rows if r['status']=='ok']
    # Wilson is descriptive; correlated items / tiny strata do not support rank claims.
    phat=correct/n if n else 0;z=1.96;den=1+z*z/n if n else 1
    center=(phat+z*z/(2*n))/den if n else 0;delta=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den if n else 0
    result={'n':n,'correct':correct,'accuracy':correct/n if n else None,'wilson95_descriptive':[center-delta,center+delta],
            'run_n':len(full_rows),'excluded':len(full_rows)-n,'ok':len(ok),'failed':len(full_rows)-len(ok),'unparsed':sum(r['task'] in ('mcq','yesno') and r['parsed'] is None for r in rows),
            'format_n':len(format_rows),'format_correct':sum(r['format_ok'] for r in format_rows),
            'total_seconds':distribution([r.get('total_seconds') for r in ok]),'ttft_seconds':distribution([r.get('ttft_seconds') for r in ok]),
            'preprocess_seconds':distribution([r.get('preprocess_seconds') for r in ok]),'generated_tokens':distribution([r.get('generated_tokens') for r in ok]),
            'reasoning_tokens':distribution([r.get('reasoning_tokens') for r in ok]),
            'tokens_per_generation_second':distribution([r['generated_tokens']/r['generation_seconds'] for r in ok if (r.get('generation_seconds') or 0)>0 and r.get('generated_tokens') is not None]),
            'length_limit_reached':sum(bool(r.get('length_limit_reached')) for r in ok)}
    if rows and all(r['task']=='yesno' for r in rows):
        tp=sum(r['gold']==['yes'] and r['parsed']=='yes' for r in rows)
        fp=sum(r['gold']==['no'] and r['parsed']=='yes' for r in rows)
        positives=sum(r['gold']==['yes'] for r in rows)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/positives if positives else 0
        result['pope']={'tp':tp,'fp':fp,'gold_positives':positives,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
                        'yes_ratio':sum(r['parsed']=='yes' for r in rows)/n,'invalid':sum(r['parsed'] is None for r in rows)}
    return result

def score_run(suite, run, destination, review_manifest):
    suite=Path(suite);run=Path(run);out=Path(destination)
    cases={c['id']:c for c in verify_suite(suite)};refs={r['id']:r for r in jsonl(suite/'references.jsonl')}
    meta=json.loads((run/'run.json').read_text())
    require(meta['suite_sha256']==sha(suite/'freeze.json'),'Wrong frozen suite')
    planned=json.loads((run/'planned.json').read_text());raw=jsonl(run/'outputs.jsonl')
    expected={(r['id'],r['condition']) for r in planned};actual=[(r['id'],r['condition']) for r in raw]
    require(len(expected)==len(planned) and len(actual)==len(set(actual)) and set(actual)==expected,'Incomplete or duplicate outputs')
    require(all(r['input_sha256']==cases[r['id']]['input_sha256'] for r in raw),'Input mismatch')
    decisions,judgments,manifest,policy=load_review(review_manifest,suite,cases,refs,meta['model'],run,raw)
    rows=[]
    for r in raw:
        c=cases[r['id']];decision=decisions[r['id']];answer=r.get('answer','').strip()
        fmt=answer in list('ABCD') if c['task']=='mcq' else answer.lower() in ('yes','no') if c['task']=='yesno' else None
        rows.append({**r,'task':c['task'],'original_gold':refs[r['id']]['answers'],'gold':decision['answers'],
                     **grade(decision,judgments[(r['id'],r['condition'])],r['status']),'format_ok':fmt})
    core=[r for r in rows if r['condition']=='image'];groups=defaultdict(list);categories=defaultdict(list)
    for r in core:groups[r['dataset']].append(r);categories[r['dataset']+'/'+r['category']].append(r)
    diag=[];bykey={(r['id'],r['condition']):r for r in rows}
    for r in rows:
        if r['condition']=='no_image':
            image=bykey[(r['id'],'image')];diag.append({'id':r['id'],'eligible':r['eligible'],'image_correct':image['correct'],'no_image_correct':r['correct'],
                                                       'image_answer':image.get('answer'),'no_image_answer':r.get('answer')})
    eligible_diag=[r for r in diag if r['eligible']]
    summary={'model':meta['model'],'reviewed_utc':policy.get('reviewed_utc'),'suite_sha256':meta['suite_sha256'],'outputs_sha256':sha(run/'outputs.jsonl'),'references_sha256':sha(suite/'references.jsonl'),
             'scoring_code_sha256':sha(__file__),'review_code_sha256':sha(Path(__file__).with_name('reviewed.py')),
             'review_manifest_sha256':sha(review_manifest),'reviewed_case_set_sha256':json_hash(decisions),
             'effective_case_ids':sorted(r['id'] for r in core if r['eligible']),
             'excluded_cases':[{'id':r['id'],'reason':r['case_review_reason']} for r in core if not r['eligible']],
             'overall':aggregate(core),'datasets':{k:aggregate(v) for k,v in groups.items()},'categories':{k:aggregate(v) for k,v in categories.items()},
             'no_image_diagnostic':{'n':len(eligible_diag),'run_n':len(diag),'image_correct':sum(bool(r['image_correct']) for r in eligible_diag),'no_image_correct':sum(bool(r['no_image_correct']) for r in eligible_diag),'pairs':diag},
             'scope':'Reviewed internal frozen subset. Quality uses shared eligible cases; runtime metrics retain all original requests and exclude warmup.'}
    context_fields=['backend','host','thinking','thinking_policy','max_new_tokens','requested_model','upstream_revision','identity_note','latency_kind','token_semantics']
    summary['execution_context']={k:meta.get('config',{})[k] for k in context_fields if k in meta.get('config',{})}
    summary['api_token_usage']=api_token_usage(rows)
    out.mkdir(parents=True,exist_ok=True);write(out/'summary.json',summary)
    (out/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    fields=['id','dataset','category','condition','status','eligible','original_gold','gold','answer','parsed','correct','case_decision','case_review_reason','review_reason','format_ok','format_issue','total_seconds','ttft_seconds','input_tokens','generated_tokens','reasoning_tokens','response_model','finish_reason']
    with (out/'scored.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return summary

def compare(suite, runs, destination, review_manifest):
    out=Path(destination);summaries=[];allrows={}
    # Validate every selected model before writing any score artifacts.
    for run in runs:
        run=Path(run);meta=json.loads((run/'run.json').read_text());model=meta['model']
        require(model not in allrows,'Only one selected attempt per model')
        cases={c['id']:c for c in verify_suite(suite)};refs={r['id']:r for r in jsonl(Path(suite)/'references.jsonl')}
        load_review(review_manifest,suite,cases,refs,model,run,jsonl(run/'outputs.jsonl'))
        allrows[model]={}
    for run in runs:
        run=Path(run);model=json.loads((run/'run.json').read_text())['model']
        summary=score_run(suite,run,out/model,review_manifest);summaries.append(summary);allrows[model]={(r['id'],r['condition']):r for r in jsonl(out/model/'scored.jsonl')}
    require(len({s['suite_sha256'] for s in summaries})==1,'Frozen suites differ')
    require(len({s['reviewed_case_set_sha256'] for s in summaries})==1,'Reviewed case sets differ')
    require(all(s['effective_case_ids']==summaries[0]['effective_case_ids'] for s in summaries),'Effective case sets differ')
    require(all({k:v['n'] for k,v in s['datasets'].items()}=={k:v['n'] for k,v in summaries[0]['datasets'].items()} for s in summaries),'Dataset denominators differ')
    keys=set.intersection(*(set(rows) for rows in allrows.values()))
    require(all(set(rows)==keys for rows in allrows.values()),'A/B planned sets differ')
    fields=['id','condition','dataset','category','eligible','gold','case_review_reason']+[m+'_'+k for m in allrows for k in ['answer','parsed','correct','format_issue','review_reason','status','seconds']]
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key in sorted(keys):
            first=next(iter(allrows.values()))[key];row={k:first[k] for k in fields[:7]}
            for m,rows in allrows.items():
                r=rows[key];row.update({m+'_'+k:r.get('total_seconds') if k=='seconds' else r.get(k) for k in ['answer','parsed','correct','format_issue','review_reason','status','seconds']})
            writer.writerow(row)
    write(out/'comparison.json',summaries)
    write(out/'review-policy.json',json.loads((Path(review_manifest).parent/'policy.json').read_text()))
    from .report import gallery,markdown
    gallery(suite,out,list(allrows));markdown(summaries,out)
    return summaries
