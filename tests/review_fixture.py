"""Explicit review evidence for tiny synthetic test runs, not a production judge."""
import json
from pathlib import Path
from vlmbench.common import json_hash,jsonl,sha,write
from vlmbench.reviewed import SCHEMA

def review_fixture(suite,runs,destination):
    destination=Path(destination);destination.mkdir(exist_ok=True)
    cases=jsonl(suite/'cases.jsonl');refs={r['id']:r for r in jsonl(suite/'references.jsonl')}
    decisions=[{'id':c['id'],'input_sha256':c['input_sha256'],'case_sha256':json_hash(c),'reference_sha256':json_hash(refs[c['id']]),
               'decision':'retain','reason':'Synthetic image is blue: option A.','answers':['A'],'kind':'choice'} for c in cases]
    records=[];sources={}
    for run in runs:
        model=json.loads((run/'run.json').read_text())['model'];rows=jsonl(run/'outputs.jsonl')
        sources[model]={n:sha(run/n) for n in ('outputs.jsonl','run.json','planned.json')};sources[model]['answer_count']=len(rows)
        for r in rows:
            records.append({'model':model,'id':r['id'],'condition':r['condition'],'input_sha256':r['input_sha256'],'raw_row_sha256':json_hash(r),
                            'canonical':'A' if r.get('answer','').strip() in ('A','<|begin_of_box|>A<|end_of_box|>') else None,
                            'format_issue':r.get('answer','').strip()!='A','reason':'Explicit synthetic fixture judgment.'})
    for name,rows in [('cases.jsonl',decisions),('answers.jsonl',records)]:
        (destination/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(destination/'policy.json',{'schema':SCHEMA,'rules':['Synthetic fixture review.']})
    write(destination/'manifest.json',{'schema':SCHEMA,'suite_sha256':sha(suite/'freeze.json'),
         'files':{n:sha(destination/n) for n in ('cases.jsonl','answers.jsonl','policy.json')},'runs':sources,'case_count':len(cases),'answer_count':len(records)})
    return destination/'manifest.json'
