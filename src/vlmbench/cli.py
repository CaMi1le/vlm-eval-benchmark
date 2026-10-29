import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Frozen public VLM regression benchmark')
    sub=p.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('fetch');fetch.add_argument('--config',type=Path,default=Path('configs/datasets.json'));fetch.add_argument('--cache',type=Path,required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--config',type=Path,default=Path('configs/datasets.json'));prep.add_argument('--cache',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    check=sub.add_parser('verify');check.add_argument('--suite',type=Path,required=True)
    api=sub.add_parser('api-run',help='CPU API evaluation; includes fixed no-image diagnostics')
    api.add_argument('--credentials',type=Path,required=True,help='Local JSON containing base_url, api_key and model; never commit this file')
    api.add_argument('--suite',type=Path,required=True);api.add_argument('--output',type=Path,required=True)
    api.add_argument('--view',choices=['smoke','core'],default='core')
    deadline=api.add_mutually_exclusive_group()
    deadline.add_argument('--timeout',type=float,default=300,help='Per-request deadline in seconds (default: 300)')
    deadline.add_argument('--no-timeout',dest='timeout',action='store_const',const=None,help='Wait without a client connection/read deadline; upstream limits may still apply')
    api.add_argument('--concurrency',type=int,default=1,help='Maximum simultaneous requests, 1–8 (default: 1)')
    api.add_argument('--remaining',type=Path,help='Resume only failed/unissued requests from remaining.json into a new output directory')
    api.add_argument('--skip-warmup',action='store_true',help='Issue no separate warmup request, for an already warmed-up recovery run')
    score=sub.add_parser('score');score.add_argument('--suite',type=Path,required=True);score.add_argument('--runs',type=Path,nargs='+',required=True);score.add_argument('--output',type=Path,required=True)
    score.add_argument('--review-manifest',type=Path,help='Optional: use the historical reviewed-content protocol instead of public-label automatic scoring')
    expanded=sub.add_parser('prepare-expanded',help='Extend a frozen v1 suite to the stratified v2 suite')
    expanded.add_argument('--config',type=Path,default=Path('configs/datasets-v2.json'))
    expanded.add_argument('--cache',type=Path,required=True);expanded.add_argument('--base-suite',type=Path,required=True);expanded.add_argument('--output',type=Path,required=True)
    radar=sub.add_parser('radar',help='Build a portable interactive seven-axis HTML dashboard')
    radar.add_argument('--scores',type=Path,nargs='+',required=True);radar.add_argument('--output',type=Path,required=True)
    radar.add_argument('--speed-reference-seconds',type=float,default=1.0)
    report=sub.add_parser('report',help='Score complete runs and generate the gallery and radar chart')
    report.add_argument('--suite',type=Path,required=True);report.add_argument('--runs',type=Path,nargs='+',required=True);report.add_argument('--output',type=Path,required=True)
    demo=sub.add_parser('demo',help='Synthetic CPU-only example; no model calls, downloads, or benchmark claims')
    demo.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('run',help='Run one local model in a new isolated workspace, with per-request deadlines')
    run.add_argument('--workspace',type=Path,required=True);run.add_argument('--suite',type=Path,required=True)
    run.add_argument('--models',type=Path,required=True);run.add_argument('--model',required=True)
    run.add_argument('--run-id',default='evaluation');run.add_argument('--seconds',type=float,default=900)
    run.add_argument('--view',choices=['core','smoke'],default='core');run.add_argument('--warmup-case-id')
    args=p.parse_args()
    if args.command=='fetch':
        from .fetch import fetch
        fetch(args.config,args.cache)
    elif args.command=='prepare':
        from .prepare import prepare
        prepare(args.config,args.cache,args.output)
    elif args.command=='verify':
        from .common import verify_suite
        print(json.dumps({'verified_cases':len(verify_suite(args.suite))}))
    elif args.command=='prepare-expanded':
        from .expand import prepare_expanded
        result=prepare_expanded(args.config,args.cache,args.base_suite,args.output)
        print(json.dumps({k:result[k] for k in ['suite','core_count','inherited_count','delta_count']}))
    elif args.command=='radar':
        from .dashboard import build_dashboard
        result=build_dashboard(args.scores,args.output,args.speed_reference_seconds)
        print(json.dumps({'models':len(result['models']),'output':str(args.output/'dashboard.html')}))
    elif args.command=='report':
        from .public_workflow import build_report
        print(json.dumps(build_report(args.suite,args.runs,args.output)))
    elif args.command=='demo':
        from .demo import demo
        print(json.dumps(demo(args.output)))
    elif args.command=='run':
        from .public_workflow import run_local
        raise SystemExit(run_local(args))
    elif args.command=='api-run':
        from .api import run_api
        meta=run_api(args.suite,args.credentials,args.output,args.view,args.timeout,args.concurrency,args.remaining,args.skip_warmup)
        print(json.dumps({'model':meta['model'],'view':meta['view'],'outputs':meta['output_count'],'ok':meta['ok_count']}))
        if not meta['all_requests_ok']:raise SystemExit(1)
    elif args.command=='score':
        if args.review_manifest:
            from .score import compare
            summaries=compare(args.suite,args.runs,args.output,args.review_manifest)
        else:
            from .benchmark_score import compare
            summaries,_=compare(args.suite,args.runs,args.output)
        print(json.dumps([{s['model']:{k:(v['correct'],v['n']) for k,v in s['datasets'].items()}} for s in summaries]))

if __name__=='__main__':main()
