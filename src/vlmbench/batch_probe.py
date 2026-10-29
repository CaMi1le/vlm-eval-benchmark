"""Isolated batch=2 feasibility probe; never merged into the serial benchmark."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
from .common import append,json_hash,now,sha,verify_suite,write
from .generation import messages_for,template_kwargs,template_evidence,vllm_sampling_kwargs,response_fields
from .selection import planned_jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--model-path');p.add_argument('--view',default='core');p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args();cases=verify_suite(a.suite);byid={c['id']:c for c in cases};cfg=json.loads(a.models.read_text())[a.model]
    if a.model_path:cfg['path']=a.model_path
    assert cfg['family']=='glm46v' and cfg['thinking'] is True and cfg['probe_max_num_seqs']==2
    selected=planned_jobs(a.suite,cases,'core',False,a.selection);assert len(selected)==4 and all(c=='image' for _,c in selected)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'planned.json',[{'id':i,'condition':c,'input_sha256':byid[i]['input_sha256']} for i,c in selected])
    meta={'model':a.model,'config':cfg,'suite_sha256':sha(a.suite/'freeze.json'),'started_utc':now(),
          'planned_requests':4,'diagnostic_only':True,'production_per_request_abort_tested':False,
          'note':'Same max_num_seqs=2 engine, serial submissions versus two-item submissions. Not serial-baseline ranking data.'}
    write(a.output/'run.json',meta)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    processor=AutoProcessor.from_pretrained(cfg['path'],local_files_only=True,trust_remote_code=False)
    kwargs={'size':{'shortest_edge':cfg['min_pixels'],'longest_edge':cfg['max_pixels']}}
    start=time.perf_counter()
    llm=LLM(model=cfg['path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=cfg['max_model_len'],
            max_num_seqs=2,gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=cfg['enforce_eager'],
            enable_prefix_caching=False,limit_mm_per_prompt={'image':8,'video':0},mm_processor_cache_gb=0,
            mm_processor_kwargs=kwargs,trust_remote_code=False,disable_log_stats=False,seed=0)
    actual=llm.llm_engine.vllm_config.scheduler_config.max_num_seqs;assert actual==2
    meta['runtime']={'load_seconds':time.perf_counter()-start,'actual_max_num_seqs':actual,
                     'thinking_control':template_evidence(processor,cfg),
                     'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','compressed-tensors']}}
    write(a.output/'run.json',meta)
    prepared=[]
    for i,_ in selected:
        c=byid[i];text=processor.apply_chat_template(messages_for(cfg,c,a.suite),tokenize=False,add_generation_prompt=True,**template_kwargs(cfg))
        images=[]
        for m in c['media']:
            with Image.open(a.suite/m['path']) as im:images.append(im.convert('RGB'))
        request={'prompt':text,'multi_modal_data':{'image':images},'mm_processor_kwargs':kwargs}
        params=SamplingParams(**vllm_sampling_kwargs(cfg),seed=int(c['input_sha256'][:8],16))
        prepared.append((c,text,request,params))
    warmup=[]
    for c,text,request,params in prepared:
        start=time.perf_counter();r=llm.generate([request],[params],use_tqdm=False)[0]
        warmup.append({'id':c['id'],'seconds':time.perf_counter()-start,'tokens':len(r.outputs[0].token_ids)})
    write(a.output/'warmup.json',{'excluded':True,'cases':warmup})
    def run(size,destination):
        rows=[];walls=[]
        with destination.open('x') as stream:
            for first in range(0,len(prepared),size):
                group=prepared[first:first+size];started=now();start=time.perf_counter()
                write(a.output/'active-request.json',{'id':group[0][0]['id'],'condition':'image','started_monotonic':time.monotonic(),
                      'batch_case_ids':[c['id'] for c,_,_,_ in group],'probe_note':'Synchronous diagnostic group; independent request cancellation is not tested here.'})
                results=llm.generate([x[2] for x in group],[x[3] for x in group],use_tqdm=False)
                elapsed=time.perf_counter()-start;walls.append(elapsed);assert len(results)==len(group)
                for (c,text,_,_),result in zip(group,results):
                    output=result.outputs[0];ids=list(output.token_ids);metrics=result.metrics
                    row={'id':c['id'],'condition':'image','model':a.model,'dataset':c['dataset'],'category':c['category'],
                         'input_sha256':c['input_sha256'],'request_sha256':json_hash({'input_sha256':c['input_sha256'],'condition':'image'}),
                         'status':'ok','started_utc':started,'completed_utc':now(),
                         **response_fields(processor,ids,cfg,text),'generated_ids':ids,'generated_tokens':len(ids),
                         'input_tokens':len(result.prompt_token_ids),'total_seconds':elapsed,'batch_size':size,
                         'latency_kind':'Synchronous whole-group wall time; not isolated single-request latency.',
                         'finish_reason':output.finish_reason,'length_limit_reached':output.finish_reason=='length',
                         'engine_metrics':{k:v for k,v in vars(metrics).items() if isinstance(v,(int,float,str,bool,type(None)))} if metrics else None}
                    append(stream,row);rows.append(row)
                (a.output/'active-request.json').unlink(missing_ok=True)
        seconds=sum(walls);tokens=sum(r['generated_tokens'] for r in rows)
        return rows,{'requests':len(rows),'engine_wall_seconds':seconds,'generated_tokens':tokens,'tokens_per_second':tokens/seconds,'requests_per_second':len(rows)/seconds}
    serial,one=run(1,a.output/'serial-outputs.jsonl');batched,two=run(2,a.output/'outputs.jsonl')
    pairs=[{'id':a['id'],'token_ids_identical':a['generated_ids']==b['generated_ids'],'final_answer_identical':a['final_answer_text']==b['final_answer_text']} for a,b in zip(serial,batched)]
    write(a.output/'summary.json',{'diagnostic_only':True,'serial_submissions':one,'batch2_submissions':two,'paired_outputs':pairs,
          'wall_speedup':one['engine_wall_seconds']/two['engine_wall_seconds'],'production_per_request_abort_tested':False,
          'limitations':'Four frozen cases, one seed, shared host. Engine cache budget is held constant at max_num_seqs=2. No official score or seven-axis speed replacement.'})
    meta['completed_utc']=now();write(a.output/'run.json',meta)


if __name__=='__main__':main()
