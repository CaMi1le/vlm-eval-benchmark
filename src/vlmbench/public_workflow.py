"""Portable entrypoints around the existing evaluator, scorer and dashboard."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from .common import jsonl, sha, verify_suite, write


def build_report(suite, runs, output):
    from .benchmark_score import compare
    from .benchmark_report import gallery
    from .dashboard import build_dashboard
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summaries,rows=compare(suite,runs,output/'scores')
    gallery(suite,output,summaries,rows,{},introduction=(
        ('SCRIPTED SYNTHETIC DEMO — no model inference; scores and timings are illustrative. ' if all(s['execution_context'].get('backend')=='synthetic_demo' for s in summaries) else '') +
        'Frozen-input evaluation: the same final-answer parser and scoring rules apply to all models. '
        'Failures stay in the denominator. Hardware, runtime, sampling and output budgets must be compared alongside scores. '
        '<a href="dashboard/dashboard.html">Radar chart</a> · <a href="scores/comparison.csv">Per-case CSV</a>'))
    build_dashboard([output/'scores/comparison.json'],output/'dashboard')
    lines=['# Evaluation report','','[Responses and images](review.html) · [Radar chart](dashboard/dashboard.html)',
           '', '| Model | Correct / planned | Accuracy | Median seconds |','|---|---:|---:|---:|']
    for s in summaries:
        v=s['overall'];median=v['total_seconds']['median']
        lines.append(f"| {s['model']} | {v['correct']}/{v['n']} | {v['accuracy']:.2%} | {median if median is not None else 'unavailable'} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return {'output':str(output),'models':len(summaries),'cases_per_model':[s['overall']['n'] for s in summaries]}


def run_local(args):
    """Create an execution snapshot; the supervisor remains the GPU lifecycle owner."""
    from .selection import planned_jobs
    root=Path(args.workspace).resolve();suite=Path(args.suite).resolve()
    if root.is_relative_to(suite):raise ValueError('Workspace must be outside the source suite')
    if root.exists():raise FileExistsError(f'Workspace already exists: {root}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('Invalid run ID')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.model):raise ValueError('Invalid model ID')
    if not math.isfinite(args.seconds) or args.seconds<=0:raise ValueError('Request deadline must be positive and finite')
    models=json.loads(Path(args.models).read_text());cfg=dict(models[args.model])
    cfg['path']=str(Path(cfg['path']).expanduser().resolve())
    if not Path(cfg['path']).is_dir():raise ValueError('Download and verify the model before running')
    if cfg.get('host') in (None,'auto'):cfg['host']=socket.gethostname()
    if cfg['host']!=socket.gethostname():raise ValueError('Model configuration targets another host')
    if not cfg.get('gpus') or any(type(g) is not int or g<0 for g in cfg['gpus']):raise ValueError('Set physical GPU indices in the model configuration')
    if len(set(cfg['gpus']))!=len(cfg['gpus']):raise ValueError('GPU indices must be unique')
    if cfg.get('runtime_bind_mounts'):raise ValueError('Advanced runtime patches require the documented supervisor workflow')
    cases=verify_suite(suite);jobs=planned_jobs(suite,cases,args.view,True)
    if not jobs:raise ValueError('Empty suite selection')
    if args.warmup_case_id and args.warmup_case_id not in {c['id'] for c in cases}:raise ValueError('Warmup case is not in the suite')
    root.mkdir(parents=True)
    package=Path(__file__).parent
    shutil.copytree(package,root/'src/vlmbench',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Copy inputs: edits to the user's source suite cannot mutate an active run.
    shutil.copytree(suite,root/'data/suite')
    config=root/'configs/models.json';write(config,{args.model:cfg})
    write(root/'configs/gpu-resource-policy.json',{'schema':'vlm-gpu-resource-policy-1','hosts':{
        cfg['host']:{'allowed_gpus':cfg['gpus'],'reserved_gpus':[]}}})
    byid={c['id']:c for c in cases};selection=root/'provenance/selection.json'
    write(selection,{'suite_sha256':sha(suite/'freeze.json'),'requests':[
        {'id':i,'condition':condition,'input_sha256':byid[i]['input_sha256']} for i,condition in jobs]})
    command=[sys.executable,'-B','-m','vlmbench.bounded_batch','--root',str(root),
        '--run-id',args.run_id,'--model',args.model,'--models-config',str(config),
        '--suite','suite','--selection',str(selection),'--seconds',str(args.seconds),
        '--phase','thinking' if cfg.get('thinking') else 'nonthinking',
        '--warmup-case-id',args.warmup_case_id or jobs[0][0]]
    code=subprocess.call(command,env=dict(os.environ,PYTHONPATH=str(root/'src')))
    if code:return code
    result=root/'runs'/args.run_id/args.model/'result'
    rows=jsonl(result/'outputs.jsonl')
    print(json.dumps({'result':str(result),'planned':len(jobs),'ok':sum(r['status']=='ok' for r in rows)}))
    return 0 if len(rows)==len(jobs) and all(r['status']=='ok' for r in rows) else 1
