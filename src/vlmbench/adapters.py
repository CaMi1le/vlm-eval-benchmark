"""In-process model adapters. Neither adapter starts an HTTP service."""
from collections import Counter
import importlib.metadata
from pathlib import Path
import time
from .generation import messages_for, template_kwargs, template_evidence, torch_generation_kwargs, vllm_sampling_kwargs, response_fields, output_budget

class MiniCPM:
    def __init__(self,config):
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        self.torch=torch;self.cfg=config;torch.set_num_threads(8);torch.manual_seed(0)
        self.processor=AutoProcessor.from_pretrained(config['path'],local_files_only=True,trust_remote_code=False)
        start=time.perf_counter()
        self.model=AutoModelForImageTextToText.from_pretrained(config['path'],dtype=torch.bfloat16,device_map='balanced',max_memory={0:'40GiB',1:'40GiB'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
        devices=Counter(str(p.device) for p in self.model.parameters())
        assert set(devices)=={'cuda:0','cuda:1'} and all(p.dtype==torch.bfloat16 for p in self.model.parameters())
        assert not any(str(d) in ('cpu','disk','meta') for d in self.model.hf_device_map.values())
        self.metadata={'load_seconds':time.perf_counter()-start,'parameter_devices':dict(devices),'parameter_count':sum(p.numel() for p in self.model.parameters()),'device_map':self.model.hf_device_map,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','accelerate']},'cuda':torch.version.cuda,
            'parameter_bytes_by_gpu':{str(i):sum(p.numel()*p.element_size() for p in self.model.parameters() if p.device.index==i) for i in range(2)}}
        self.cfg.setdefault('max_model_len', self.model.config.text_config.max_position_embeddings)
        self.metadata['thinking_control'] = template_evidence(self.processor, self.cfg)
    def generate(self,case,root,no_image=False):
        torch=self.torch;processor=self.processor
        for i in range(2):torch.cuda.synchronize(i);torch.cuda.reset_peak_memory_stats(i)
        start=time.perf_counter()
        slices=self.cfg['single_image_max_slices'] if len(case['media'])==1 else self.cfg['multi_image_max_slices']
        kwargs={**template_kwargs(self.cfg),'downsample_mode':self.cfg['downsample_mode'],'max_slice_nums':slices}
        messages=messages_for(self.cfg,case,root,no_image)
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',**kwargs).to('cuda:0')
        for i in range(2):torch.cuda.synchronize(i)
        gen_start=time.perf_counter()
        class Streamer:
            def __init__(self):self.prompt=True;self.first=None
            def put(self,value):
                if self.prompt:self.prompt=False
                elif self.first is None:self.first=time.perf_counter()
            def end(self):pass
        streamer=Streamer()
        with torch.inference_mode():
            torch.manual_seed(int(case['input_sha256'][:8],16))
            output=self.model.generate(**inputs,**torch_generation_kwargs(self.cfg,inputs['input_ids'].shape[-1]),streamer=streamer,downsample_mode=self.cfg['downsample_mode'])
        for i in range(2):torch.cuda.synchronize(i)
        end=time.perf_counter();n=inputs['input_ids'].shape[-1];ids=output[0,n:].tolist()
        row={**response_fields(processor,ids,self.cfg,rendered),'input_tokens':n,'generated_tokens':len(ids),'generated_ids':ids,
            'preprocess_seconds':gen_start-start,'ttft_seconds':streamer.first-start if streamer.first else None,'ttft_kind':'host first generated token, includes preprocessing',
            'total_seconds':end-start,'generation_seconds':end-gen_start,'length_limit_reached':len(ids)>=output_budget(self.cfg,n),'effective_max_new_tokens':output_budget(self.cfg,n),
            'time_limit_reached':bool(self.cfg.get('max_time_seconds') and end-gen_start>=self.cfg['max_time_seconds']),
            'peak_allocated_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],'peak_reserved_gib':[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)],
            'processor_settings':kwargs,'processed_tensor_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}}
        del inputs,output
        return row

class Qwen:
    def __init__(self,config):
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        self.cfg=config;self.SamplingParams=SamplingParams
        self.processor=AutoProcessor.from_pretrained(config['path'],local_files_only=True,trust_remote_code=False)
        start=time.perf_counter()
        self.model=LLM(model=config['path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=config['max_model_len'],
            max_num_seqs=1,gpu_memory_utilization=config['gpu_memory_utilization'],enforce_eager=config['enforce_eager'],
            enable_prefix_caching=False,limit_mm_per_prompt={'image':8},mm_processor_cache_gb=config['mm_processor_cache_gb'],mm_processor_kwargs={'max_pixels':config['max_pixels']},
            trust_remote_code=False,disable_log_stats=False,seed=0)
        self.metadata={'load_seconds':time.perf_counter()-start,'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','vllm','compressed-tensors']},
            'processor':'Qwen3VLProcessor','precision':config['precision'],'mtp':'retained weights, speculative decoding not enabled','engine_settings':{k:config[k] for k in ['max_model_len','max_pixels','gpu_memory_utilization','enforce_eager','mm_processor_cache_gb']}}
        self.metadata['thinking_control'] = template_evidence(self.processor, self.cfg)
    def generate(self,case,root,no_image=False):
        from PIL import Image
        start=time.perf_counter()
        messages=messages_for(self.cfg,case,root,no_image)
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs(self.cfg))
        images=[]
        if not no_image:
            for m in case['media']:
                with Image.open(Path(root)/m['path']) as image:images.append(image.convert('RGB'))
        request={'prompt':text}
        if images:request['multi_modal_data']={'image':images}
        gen_start=time.perf_counter()
        result=self.model.generate([request],self.SamplingParams(**vllm_sampling_kwargs(self.cfg),seed=int(case['input_sha256'][:8],16)),use_tqdm=False)[0]
        end=time.perf_counter();output=result.outputs[0];metrics=result.metrics
        ttft=None
        if metrics is not None and getattr(metrics,'first_token_time',None) is not None:
            ttft=(gen_start-start)+metrics.first_token_time-metrics.arrival_time
        return {**response_fields(self.processor,list(output.token_ids),self.cfg,text),'engine_output_text':output.text,'input_tokens':len(result.prompt_token_ids),'generated_tokens':len(output.token_ids),'generated_ids':list(output.token_ids),
            'preprocess_seconds':gen_start-start,'ttft_seconds':ttft,'ttft_kind':'engine first token plus client preprocessing; null when unavailable',
            'total_seconds':end-start,'generation_seconds':end-gen_start,'finish_reason':output.finish_reason,'length_limit_reached':output.finish_reason=='length',
            'effective_max_new_tokens':output_budget(self.cfg,len(result.prompt_token_ids)),'processor_settings':{**template_kwargs(self.cfg),'max_pixels':self.cfg['max_pixels'],'input_mode':'chronological multi-image for sampled videos'},
            'engine_metrics':{k:v for k,v in vars(metrics).items() if isinstance(v,(int,float,str,bool,type(None)))} if metrics is not None else None}

def load_adapter(config):
    if config['backend'] in ('transformers_qwen','vllm_multimodal'):
        from .additional_adapters import NativeQwen,MultimodalVLLM
        return {'transformers_qwen':NativeQwen,'vllm_multimodal':MultimodalVLLM}[config['backend']](config)
    return {'transformers_minicpm':MiniCPM,'vllm_qwen':Qwen}[config['backend']](config)
