"""Continue after a request timeout using a fresh, owned engine for remaining cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import jsonl, json_hash, now, sha, write, verify_suite
from .selection import planned_jobs
from .time_policy import enforce_deadline


def completed_prefix(rows, timeout, seconds, phase='nonthinking'):
    """Repair placeholders are queued again, except the timed-out request itself."""
    result=[]
    for row in rows:
        value=enforce_deadline(row, seconds, timeout, phase)
        if value['status']!='not_run':result.append(value)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--model',required=True)
    p.add_argument('--models-config',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=900);p.add_argument('--suite',default='public-vlm-v2')
    p.add_argument('--warmup-case-id')
    p.add_argument('--phase',choices=['nonthinking','thinking'],default='nonthinking')
    a=p.parse_args();root=a.root.resolve();suite=root/'data'/a.suite
    cfg=json.loads(a.models_config.read_text())[a.model];assert cfg['thinking'] is (a.phase=='thinking') and a.seconds>0
    cases=verify_suite(suite);byid={c['id']:c for c in cases}
    jobs=planned_jobs(suite,cases,'core',True,a.selection)
    planned=[{'id':i,'condition':c,'input_sha256':byid[i]['input_sha256']} for i,c in jobs]
    directory=root/'runs'/a.run_id/a.model;directory.mkdir(parents=True,exist_ok=False)
    result=directory/'result';result.mkdir();write(result/'planned.json',planned)
    state={'started_utc':now(),'model':a.model,'seconds':a.seconds,'chunks':[],'finished':False}
    outputs={};last_meta=None;releases=[]
    while len(outputs)<len(jobs):
        pending=[r for r in planned if (r['id'],r['condition']) not in outputs]
        runid=a.run_id+'-chunk-'+str(len(state['chunks'])+1)
        selection=root/'provenance'/f'{runid}-{a.model}.json'
        write(selection,{'suite_sha256':sha(suite/'freeze.json'),'requests':pending,
                         'rule':'Remaining original requests; never retry a timed-out model-case.'})
        command=[sys.executable,'-B','-m','vlmbench.supervise','--root',str(root),'--run-id',runid,
                 '--model',a.model,'--suite',a.suite,'--models-config',str(a.models_config),
                 '--selection',str(selection),'--diagnostics','--deadline','0',
                 '--request-deadline',str(a.seconds),'--warmup-case-id',a.warmup_case_id or jobs[0][0]]
        with (directory/(runid+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['active']={'run_id':runid,'pid':child.pid};write(directory/'bounded-state.json',state)
            code=child.wait()
        chunk=root/'runs'/runid/a.model
        release=json.loads((chunk/'gpu-release.json').read_text())
        assert release['worker_gone'] and release['container_gone'] and release['target_gpus_idle']
        rows=jsonl(chunk/'result/outputs.jsonl');timeout=release.get('request_timeout')
        accepted=completed_prefix(rows,timeout,a.seconds,a.phase)
        last_meta=json.loads((chunk/'result/run.json').read_text())
        if code and not timeout:
            # Preserve explicit failures from an engine failure, without looping forever.
            accepted=[enforce_deadline(r,a.seconds,phase=a.phase) for r in rows]
        assert accepted,'Chunk produced no terminal outcomes'
        for row in accepted:
            key=row['id'],row['condition'];assert key not in outputs;outputs[key]=row
        releases.append({'file':str((chunk/'gpu-release.json').relative_to(root)), 'sha256':sha(chunk/'gpu-release.json'),**release})
        state['chunks'].append({'run_id':runid,'exit_code':code,'outputs_sha256':sha(chunk/'result/outputs.jsonl'),
                                'terminal':len(accepted),'timeout':timeout})
        state['recorded']=len(outputs);write(directory/'bounded-state.json',state)
        (result/'outputs.jsonl').write_text(''.join(json.dumps(outputs[i,c],ensure_ascii=False)+'\n' for i,c in jobs if (i,c) in outputs))
    last_meta.update(composite=True,planned_requests=len(planned),completed_utc=now(),
                     request_deadline_seconds=a.seconds,bounded_chunks=state['chunks'])
    write(result/'run.json',last_meta)
    write(directory/'gpu-release.json',{'aggregate':True,'finished_utc':now(),'worker_gone':True,'container_gone':True,
          'target_gpus_idle':True,'chunks':releases,'before':releases[0]['before'],'after':releases[-1]['after']})
    state.update(finished=True,finished_utc=now());state.pop('active',None);write(directory/'bounded-state.json',state)


if __name__=='__main__':main()
