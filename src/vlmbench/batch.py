"""Run each model's smoke and core in separate resource-owned processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from .common import jsonl,now,write

def supervisor_command(root, model, runid, view, suite, deadline, models_config=None):
    command = ['python3','-B','-m','vlmbench.supervise','--root',str(root),'--model',model,
            '--run-id',runid,'--suite',suite,'--view',view,'--diagnostics','--deadline',str(deadline)]
    if models_config:command+=['--models-config',str(models_config)]
    return command

def smoke_gate(result):
    result=Path(result);planned=json.loads((result/'planned.json').read_text());rows=jsonl(result/'outputs.jsonl')
    expected={(r['id'],r['condition']) for r in planned};actual={(r['id'],r['condition']) for r in rows}
    failures=[{'id':r['id'],'condition':r['condition'],'status':r['status']} for r in rows if r['status']!='ok' or not isinstance(r.get('answer'),str) or not r['answer'].strip()]
    return {'checked_utc':now(),'passed':len(rows)==len(planned)==len(actual) and actual==expected and not failures,
            'planned':len(planned),'outputs':len(rows),'failures':failures,'rule':'Technical gate only: complete nonempty outputs. Accuracy/format/truncation do not select models or cases.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--suite',default='public-vlm-v1');p.add_argument('--deadline',type=int,default=2700)
    p.add_argument('--lane',default='',help='Unique state/log suffix for concurrent model batches on this host')
    p.add_argument('--models-config',type=Path)
    a=p.parse_args()
    assert not a.lane or all(c.isalnum() or c in '-_' for c in a.lane)
    root=a.root.resolve();configs=json.loads((a.models_config or root/'configs/models.json').read_text());host=socket.gethostname()
    assert all(configs[m]['host']==host for m in a.models)
    records=root/'runs'/a.run_id;records.mkdir(parents=True,exist_ok=True)
    state_name=f'batch-{host}'+('-'+a.lane if a.lane else '');state_path=records/(state_name+'.json')
    assert not state_path.exists(),'A batch with this ID already exists'
    state={'pid':os.getpid(),'host':host,'started_utc':now(),'models':a.models,'suite':a.suite,'deadline_seconds':a.deadline,'lane':a.lane,'results':{}};write(state_path,state);child=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (records/(state_name+'.log')).open('x') as log:
            for model in a.models:
                state['active_model']=model;state['results'][model]={};write(state_path,state)
                for view,runid in [('smoke',a.run_id+'-smoke'),('core',a.run_id)]:
                    state['active_view']=view;write(state_path,state)
                    command=supervisor_command(root,model,runid,view,a.suite,a.deadline,a.models_config)
                    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait();child=None;state['results'][model][view+'_exit_code']=code
                    if view=='smoke':
                        try:gate=smoke_gate(root/'runs'/runid/model/'result')
                        except Exception:gate={'passed':False,'error':traceback.format_exc()}
                        write(root/'runs'/runid/model/'smoke-gate.json',gate);state['results'][model]['smoke_gate']=gate
                        if code or not gate['passed']:break
                    write(state_path,state)
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=120)
        state.pop('active_model',None);state.pop('active_view',None);state['finished_utc']=now();write(state_path,state)
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
