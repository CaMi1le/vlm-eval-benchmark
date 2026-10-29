"""Host-side lifecycle owner: exclusive cooperating lock, GPU snapshots, cleanup."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from .common import append, jsonl, now, sha, verify_suite, write
from .selection import planned_jobs
from .gpu_policy import enforce_gpu_policy
from .thinking_pause import enforce_thinking_pause

def snapshot():
    def query(kind, fields):
        p=subprocess.run(['nvidia-smi','--query-'+kind+'='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
        return [dict(zip(fields,[v.strip() for v in r])) for r in csv.reader(io.StringIO(p.stdout)) if r]
    return {'utc':now(),'gpus':query('gpu',['index','uuid','name','memory.total','memory.used','utilization.gpu','driver_version']),
            'processes':query('compute-apps',['gpu_uuid','pid','process_name','used_memory'])}

def idle(snap, indices):
    selected=[g for g in snap['gpus'] if int(g['index']) in indices]
    return len(selected)==len(indices) and all(int(g['memory.used'])<=256 for g in selected) and not any(p['gpu_uuid'] in {g['uuid'] for g in selected} for p in snap['processes'])

def repair_missing(output, cases, jobs, model):
    """Preserve partial bytes separately and make every planned failure explicit."""
    output.mkdir(parents=True,exist_ok=True);path=output/'outputs.jsonl';seen=set();valid=[];broken=[]
    if path.exists():
        for line in path.read_bytes().splitlines():
            try:
                r=json.loads(line);key=(r['id'],r['condition'])
                if key in seen:raise ValueError('Duplicate output')
                seen.add(key);valid.append(line)
            except (ValueError,KeyError):broken.append(line)
    if broken:
        (output/'interrupted-fragment.bin').write_bytes(b'\n'.join(broken));path.write_bytes(b'\n'.join(valid)+(b'\n' if valid else b''))
    byid={c['id']:c for c in cases}
    with path.open('a') as f:
        for i,condition in jobs:
            if (i,condition) not in seen:
                c=byid[i];append(f,{'id':i,'condition':condition,'model':model,'dataset':c['dataset'],'category':c['category'],
                    'input_sha256':c['input_sha256'],'status':'not_run','error':'Worker exited without this result; see lifecycle and worker log.'})

def expired_request(directory, seconds, clock=None):
    """Watch the actual formal request, excluding load and warmup."""
    path=Path(directory)/'active-request.json'
    if not seconds or not path.exists():return None
    try:active=json.loads(path.read_text())
    except (FileNotFoundError,ValueError):return None
    elapsed=(time.monotonic() if clock is None else clock)-active['started_monotonic']
    return {**active,'deadline_seconds':seconds,'observed_elapsed_seconds':elapsed} if elapsed>=seconds else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--suite',default='public-vlm-v1');p.add_argument('--view',choices=['core','smoke','regression','delta'],default='core')
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--deadline',type=int,default=2700)
    p.add_argument('--models-config',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--request-deadline',type=float,default=0);p.add_argument('--warmup-case-id')
    p.add_argument('--worker-module',choices=['vlmbench.worker','vlmbench.batch_probe'],default='vlmbench.worker')
    a=p.parse_args();root=a.root.resolve();models_config=(a.models_config or root/'configs/models.json').resolve();cfg=json.loads(models_config.read_text())[a.model]
    assert models_config.is_relative_to(root), 'Config must be included in the isolated campaign'
    if a.selection:
        a.selection=a.selection.resolve();assert a.selection.is_relative_to(root)
    assert socket.gethostname()==cfg['host'];assert all(x.isalnum() or x in '-_' for x in a.run_id)
    gpu_policy=enforce_gpu_policy(root,cfg)
    enforce_thinking_pause(root,a.model,cfg)
    suite=root/'data'/a.suite;cases=verify_suite(suite);views=json.loads((suite/'views.json').read_text())
    jobs=planned_jobs(suite,cases,a.view,a.diagnostics,a.selection)
    run=root/'runs'/a.run_id/a.model;run.mkdir(parents=True,exist_ok=False)
    (run/'supervisor.pid').write_text(str(os.getpid())+'\n');write(run/'planned.json',[{'id':i,'condition':c} for i,c in jobs])
    locks=[];worker=None;container=None;before=None;error=None;request_timeout=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        (root/'locks').mkdir(exist_ok=True)
        for gpu in cfg['gpus']:
            lock=(root/'locks'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);locks.append(lock)
        while True:
            gpu_policy=enforce_gpu_policy(root,cfg)
            enforce_thinking_pause(root,a.model,cfg)
            before=snapshot()
            if idle(before,cfg['gpus']):break
            write(run/'waiting.json',before);time.sleep(15)
        write(run/'gpu-before.json',before)
        env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),TOKENIZERS_PARALLELISM='false',
                 OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
        args=['-B','-m',a.worker_module,'--suite',str(suite),'--models',str(models_config),'--model',a.model,'--output',str(run/'result'),'--view',a.view]
        if a.diagnostics:args+=['--diagnostics']
        if a.selection:args+=['--selection',str(a.selection)]
        if a.warmup_case_id:args+=['--warmup-case-id',a.warmup_case_id]
        if cfg['backend'].startswith('transformers_'):cmd=[cfg['python']]+args
        else:
            container=f'aibox-vlm-{a.run_id}-{a.model}'
            cache=root/'cache'/a.model;cache.mkdir(parents=True,exist_ok=True)
            cmd=['docker','run','--rm','--name',container,'--label','aibox.vlm.run='+a.run_id,'--network','none',
                 '--gpus','device='+','.join(map(str,cfg['gpus'])),'--shm-size','8g','--user',f'{os.getuid()}:{os.getgid()}',
                 '-v',f'{root}:{root}','-v',f'{cfg["path"]}:/model:ro','-w',str(root)]
            for patch in cfg.get('runtime_bind_mounts',[]):
                source=(root/patch['file']).resolve()
                assert source.is_relative_to(root) and sha(source)==patch['sha256'],'Runtime patch checksum mismatch'
                cmd+=['-v',f'{source}:{patch["container_path"]}:ro']
            docker_env={k:env[k] for k in ['PYTHONPATH','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONUNBUFFERED']}
            docker_env.update(VLM_EVAL_HOST=socket.gethostname(),HF_HOME=str(cache/'hf'),XDG_CACHE_HOME=str(cache),VLLM_CACHE_ROOT=str(cache/'vllm'),
                              XDG_CONFIG_HOME=str(cache/'config'),VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                              TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
                              TORCH_EXTENSIONS_DIR=str(cache/'torch-extensions'),VLLM_WORKER_MULTIPROC_METHOD='spawn')
            for k,v in docker_env.items():cmd+=['-e',f'{k}={v}']
            cmd+=['--entrypoint','python',cfg['docker_image']]+args+['--model-path','/model']
        gpu_policy=enforce_gpu_policy(root,cfg)
        enforce_thinking_pause(root,a.model,cfg)
        write(run/'launch.json',{'command':cmd,'started_utc':now(),'container':container,'deadline_seconds':a.deadline,'request_deadline_seconds':a.request_deadline,'gpu_resource_policy':gpu_policy})
        with (run/'worker.log').open('wb') as log,(run/'gpu-monitor.jsonl').open('w') as mon:
            worker=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (run/'worker.pid').write_text(str(worker.pid)+'\n');start=time.monotonic()
            while worker.poll() is None:
                snap=snapshot();append(mon,snap)
                # Observe, but never kill an unrelated PID based on GPU presence.
                if a.deadline and time.monotonic()-start>a.deadline:raise TimeoutError('Worker deadline exceeded')
                request_timeout=expired_request(run/'result',a.request_deadline)
                if request_timeout:raise TimeoutError('Formal request deadline exceeded')
                time.sleep(0.25 if a.request_deadline else 2)
    except BaseException:
        error=traceback.format_exc()
    finally:
        # Docker CLI termination alone does not terminate a detached engine.
        if container:
            probe=subprocess.run(['docker','inspect','--format','{{index .Config.Labels "aibox.vlm.run"}}',container],capture_output=True,text=True,timeout=15)
            if probe.returncode==0:
                if probe.stdout.strip()!=a.run_id:raise RuntimeError('Refusing cleanup: container ownership mismatch')
                subprocess.run(['docker','stop','-t','0' if request_timeout else '15',container],capture_output=True,timeout=30)
                subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30)
        if worker is not None:
            try:os.killpg(worker.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=15)
        after=snapshot()
        for _ in range(15):
            if idle(after,cfg['gpus']):break
            time.sleep(2);after=snapshot()
        container_gone=not container or subprocess.run(['docker','inspect',container],capture_output=True,timeout=15).returncode!=0
        write(run/'gpu-release.json',{'finished_utc':now(),'error':error,'worker_pid':worker.pid if worker else None,
            'worker_exit_code':worker.returncode if worker else None,'worker_gone':worker is None or not Path(f'/proc/{worker.pid}').exists(),
            'container':container,'container_gone':container_gone,'target_gpus_idle':idle(after,cfg['gpus']),'before':before,'after':after,'request_timeout':request_timeout})
        repair_missing(run/'result',cases,jobs,a.model)
        for lock in locks:lock.close()
    print(json.dumps({'model':a.model,'run_id':a.run_id,'error':error,'exit_code':worker.returncode if worker else None,'gpus_idle':idle(after,cfg['gpus'])}),flush=True)
    if error or worker is None or worker.returncode:raise SystemExit(1)

if __name__=='__main__':main()
