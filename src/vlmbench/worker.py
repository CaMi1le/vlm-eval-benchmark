"""Offline batch worker; one output or explicit failure per planned request."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import time
import traceback
from .common import append,json_hash,now,sha,verify_suite,write
from .selection import planned_jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--view',choices=['smoke','core','regression','delta'],default='core');p.add_argument('--diagnostics',action='store_true');p.add_argument('--model-path')
    p.add_argument('--selection',type=Path);p.add_argument('--warmup-case-id')
    a=p.parse_args();cases=verify_suite(a.suite);cfg=json.loads(a.models.read_text())[a.model]
    assert socket.gethostname()==cfg['host'] or os.environ.get('VLM_EVAL_HOST')==cfg['host']
    if a.model_path:cfg['path']=a.model_path
    byid={c['id']:c for c in cases}
    jobs=[(byid[i],condition=='no_image') for i,condition in planned_jobs(a.suite,cases,a.view,a.diagnostics,a.selection)]
    assert jobs, 'Empty request selection'
    a.output.mkdir(parents=True,exist_ok=False)
    meta={'model':a.model,'started_utc':now(),'pid':os.getpid(),'host':socket.gethostname(),'config':cfg,'suite_sha256':sha(a.suite/'freeze.json'),
        'models_config_sha256':sha(a.models),'view':a.view,'planned_requests':len(jobs),'python':platform.python_version(),
        'source_sha256':{str(p.relative_to(Path(__file__).parent)):sha(p) for p in Path(__file__).parent.rglob('*.py')}}
    if a.selection:meta['selection_sha256']=sha(a.selection)
    meta['warmup_case_id']=a.warmup_case_id or jobs[0][0]['id']
    write(a.output/'run.json',meta);write(a.output/'planned.json',[{'id':c['id'],'condition':'no_image' if no else 'image','input_sha256':c['input_sha256']} for c,no in jobs])
    adapter=None;seen=set()
    try:
        from .adapters import load_adapter
        adapter=load_adapter(cfg);meta['runtime']=adapter.metadata;write(a.output/'run.json',meta);print('MODEL_LOADED',flush=True)
        warmcase=byid[a.warmup_case_id] if a.warmup_case_id else jobs[0][0]
        warmup=adapter.generate(warmcase,a.suite,False if a.warmup_case_id else jobs[0][1]);warmup.update(warmup=True,id=warmcase['id']);write(a.output/'warmup.json',warmup)
        with (a.output/'outputs.jsonl').open('x') as stream:
            failures=0
            for case,no_image in jobs:
                condition='no_image' if no_image else 'image'
                base={'id':case['id'],'condition':condition,'dataset':case['dataset'],'category':case['category'],'model':a.model,'input_sha256':case['input_sha256'],
                    'request_sha256':json_hash({'input_sha256':case['input_sha256'],'condition':condition}),'started_utc':now()}
                write(a.output/'active-request.json',{**base,'started_monotonic':time.monotonic()})
                try:
                    if failures>=3:raise RuntimeError('Stopped after three consecutive inference failures')
                    row={**base,**adapter.generate(case,a.suite,no_image),'status':'ok','completed_utc':now()};failures=0
                except Exception as e:
                    row={**base,'status':'error','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()};failures+=1;gc.collect()
                append(stream,row);seen.add((case['id'],condition));(a.output/'active-request.json').unlink(missing_ok=True);print('RESULT',case['id'],condition,row['status'],round(row.get('total_seconds',0),3),flush=True)
        meta['completed_utc']=now()
    except BaseException:
        meta['fatal_error']=traceback.format_exc()
        raise
    finally:
        with (a.output/'outputs.jsonl').open('a') as stream:
            for c,no in jobs:
                condition='no_image' if no else 'image'
                if (c['id'],condition) not in seen:append(stream,{'id':c['id'],'condition':condition,'model':a.model,'dataset':c['dataset'],'category':c['category'],
                    'input_sha256':c['input_sha256'],'status':'not_run','error':'Worker failed before this request; see worker log.'})
        meta['worker_finally_utc']=now();write(a.output/'run.json',meta)
        # Process exit releases the engine and its child processes; supervisor verifies.

if __name__=='__main__':main()
