"""Additional fixed-runtime adapters; canonical cases and scoring stay unchanged."""
import importlib.metadata
from collections import Counter
import json
from pathlib import Path
import time
from .generation import messages_for, template_kwargs, template_evidence, torch_generation_kwargs, vllm_sampling_kwargs, response_fields, output_budget

class NativeQwen:
    def __init__(self,config):
        import torch
        from transformers import AutoModelForImageTextToText,AutoProcessor
        self.torch=torch;self.cfg=config;torch.set_num_threads(8);torch.manual_seed(0)
        self.processor=AutoProcessor.from_pretrained(config['path'],local_files_only=True,trust_remote_code=False)
        assert 'longest_edge' in self.processor.image_processor.size
        self.processor.image_processor.size['longest_edge']=config['max_pixels']
        start=time.perf_counter()
        self.model=AutoModelForImageTextToText.from_pretrained(config['path'],dtype=torch.bfloat16,device_map={'':'cuda:0'},
            attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
        assert all(p.device.type=='cuda' and p.device.index==0 and p.dtype==torch.bfloat16 for p in self.model.parameters())
        self.metadata={'load_seconds':time.perf_counter()-start,'processor':type(self.processor).__name__,'parameter_count':sum(p.numel() for p in self.model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in self.model.parameters()),'parameter_devices':['cuda:0'],'cuda':torch.version.cuda,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','accelerate']},
            'image_processor_size':{k:self.processor.image_processor.size[k] for k in ['shortest_edge','longest_edge']}}
        self.metadata['thinking_control'] = template_evidence(self.processor, self.cfg)

    def generate(self,case,root,no_image=False):
        torch=self.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        messages=messages_for(self.cfg,case,root,no_image)
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs(self.cfg))
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',**template_kwargs(self.cfg)).to('cuda:0')
        n=inputs['input_ids'].shape[-1]
        budget=output_budget(self.cfg,n)
        torch.cuda.synchronize();gen_start=time.perf_counter()
        class Streamer:
            def __init__(self):self.prompt=True;self.first=None
            def put(self,value):
                if self.prompt:self.prompt=False
                elif self.first is None:self.first=time.perf_counter()
            def end(self):pass
        streamer=Streamer()
        with torch.inference_mode():
            torch.manual_seed(int(case['input_sha256'][:8],16))
            output=self.model.generate(**inputs,**torch_generation_kwargs(self.cfg,n),streamer=streamer)
        torch.cuda.synchronize();end=time.perf_counter();ids=output[0,n:].tolist()
        return {**response_fields(self.processor,ids,self.cfg,rendered),'input_tokens':n,'generated_tokens':len(ids),'generated_ids':ids,
            'preprocess_seconds':gen_start-start,'ttft_seconds':streamer.first-start if streamer.first else None,'ttft_kind':'host first generated token, includes preprocessing',
            'generation_seconds':end-gen_start,'total_seconds':end-start,'length_limit_reached':len(ids)>=budget,'effective_max_new_tokens':budget,
            'time_limit_reached':bool(self.cfg.get('max_time_seconds') and end-gen_start>=self.cfg['max_time_seconds']),
            'processor_settings':{**template_kwargs(self.cfg),'max_pixels':self.cfg['max_pixels']},
            'processed_tensor_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
            'peak_allocated_gib':[torch.cuda.max_memory_allocated()/2**30],'peak_reserved_gib':[torch.cuda.max_memory_reserved()/2**30]}

def multimodal_kwargs(config,case):
    multi=len(case['media'])>1
    if config['family'] in ('minicpm46', 'minicpm47'):
        return {'downsample_mode':config['downsample_mode'],'max_slice_nums':config['multi_image_max_slices'] if multi else config['single_image_max_slices']}
    if config['family']=='internvl':
        # This backend forwards processor kwargs to its video processor too.
        # Set the image patch budget in hf_overrides, where it belongs.
        return {}
    return {'size':{'shortest_edge':config['min_pixels'],'longest_edge':config['max_pixels']}}

def runtime_hf_overrides(config):
    overrides=dict(config.get('hf_overrides',{}))
    if config.get('quantization_skip_prefixes'):
        quant=json.loads((Path(config['path'])/'config.json').read_text())['quantization_config']
        quant['modules_to_not_convert']=list(dict.fromkeys(quant['modules_to_not_convert']+config['quantization_skip_prefixes']))
        overrides['quantization_config']=quant
    return overrides

def inspect_quantized_experts(model):
    """Run inside a vLLM worker; return metadata, never model tensors."""
    import torch
    experts = []
    for name, module in model.named_modules():
        if not name.endswith('.experts'):
            continue
        # vLLM 0.32 splits the MoE runner from its RoutedExperts weights.
        routed = getattr(module, 'routed_experts', module)
        method = getattr(routed, 'quant_method', None)
        if method is None:
            continue
        tensors = [
            {'name': key, 'dtype': str(value.dtype), 'shape': list(value.shape),
             'device': str(value.device), 'bytes': value.numel() * value.element_size()}
            for key, value in list(routed.named_parameters(recurse=False))
            + list(routed.named_buffers(recurse=False))
        ]
        experts.append({'name': name, 'module': type(module).__name__,
                        'routed_module': type(routed).__name__,
                        'quant_method': type(method).__name__, 'tensors': tensors})
    return {'model_class': type(model).__name__, 'experts': experts,
            'parameter_devices': dict(Counter(str(p.device) for p in model.parameters())),
            'parameter_bytes_by_dtype': {
                dtype: sum(p.numel() * p.element_size() for p in model.parameters()
                           if str(p.dtype) == dtype)
                for dtype in sorted({str(p.dtype) for p in model.parameters()})},
            'cuda': torch.version.cuda}


def verify_quantized_experts(workers, expected_layers):
    assert workers, 'No worker-side quantization evidence'
    for worker in workers:
        experts = worker['experts']
        assert len(experts) == expected_layers, 'Missing routed expert layers'
        assert all(d.startswith('cuda:') for d in worker['parameter_devices']), 'Model offloaded from GPU'
        for expert in experts:
            assert 'compressed' in expert['quant_method'].lower(), expert
            packed = [t for t in expert['tensors']
                      if 'weight' in t['name'] and t['dtype'] in ('torch.int32', 'torch.uint8', 'torch.int8')]
            assert len(packed) >= 2, 'Expert projections are not packed integer weights'
            assert all(t['device'].startswith('cuda:') for t in packed)


class QuantizationInspector:
    """Named worker RPC returning plain metadata under vLLM's default serializer."""
    def inspect_awq_experts(self):
        return inspect_quantized_experts(self.get_model())


class MultimodalVLLM:
    def __init__(self,config):
        from transformers import AutoProcessor,AutoTokenizer
        from vllm import LLM,SamplingParams
        self.cfg=config;self.SamplingParams=SamplingParams
        if config['family']=='internvl':
            self.processor=AutoTokenizer.from_pretrained(config['path'],local_files_only=True,trust_remote_code=False)
        else:self.processor=AutoProcessor.from_pretrained(config['path'],local_files_only=True,trust_remote_code=False)
        start=time.perf_counter()
        maximum={'media':[{}]}  # Profile the largest allowed single-image budget.
        overrides=runtime_hf_overrides(config)
        worker_options = ({'worker_extension_cls': 'vlmbench.additional_adapters.QuantizationInspector'}
                          if config.get('verify_quantized_expert_layers') else {})
        self.model=LLM(model=config['path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=config['max_model_len'],max_num_seqs=1,
            gpu_memory_utilization=config['gpu_memory_utilization'],enforce_eager=config['enforce_eager'],enable_prefix_caching=False,
            limit_mm_per_prompt={'image':8,'video':0},mm_processor_cache_gb=0,mm_processor_kwargs=multimodal_kwargs(config,maximum),
            trust_remote_code=config['trust_remote_code'],hf_overrides=overrides,disable_log_stats=False,seed=0,**worker_options)
        self.metadata={'load_seconds':time.perf_counter()-start,'processor':type(self.processor).__name__,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','vllm','compressed-tensors']},
            'engine_settings':{k:config[k] for k in ['max_model_len','gpu_memory_utilization','enforce_eager','mm_processor_cache_gb']},
            'assistant_prefix':config.get('assistant_prefix'),'runtime_hf_overrides':overrides,
            'thinking_control':template_evidence(self.processor,self.cfg)}
        if config.get('verify_quantized_expert_layers'):
            quant = json.loads((Path(config['path'])/'config.json').read_text())['quantization_config']
            assert quant['quant_method'] == 'compressed-tensors'
            assert all(g['weights']['num_bits'] == 4 and g['weights']['type'] == 'int'
                       and g['weights']['group_size'] == 128 for g in quant['config_groups'].values())
            evidence = self.model.collective_rpc('inspect_awq_experts')
            self.metadata['quantized_experts'] = evidence
            verify_quantized_experts(evidence, config['verify_quantized_expert_layers'])

    def generate(self,case,root,no_image=False):
        from PIL import Image
        start=time.perf_counter();messages=messages_for(self.cfg,case,root,no_image)
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs(self.cfg))
        if self.cfg.get('assistant_prefix'):text+=self.cfg['assistant_prefix']
        images=[]
        if not no_image:
            for m in case['media']:
                with Image.open(Path(root)/m['path']) as image:images.append(image.convert('RGB'))
        kwargs=multimodal_kwargs(self.cfg,case);request={'prompt':text}
        if images:request.update(multi_modal_data={'image':images},mm_processor_kwargs=kwargs)
        gen_start=time.perf_counter()
        result=self.model.generate([request],self.SamplingParams(**vllm_sampling_kwargs(self.cfg),seed=int(case['input_sha256'][:8],16)),use_tqdm=False)[0]
        end=time.perf_counter();output=result.outputs[0];metrics=result.metrics;ttft=None
        if metrics is not None and getattr(metrics,'first_token_time',None) is not None:ttft=gen_start-start+metrics.first_token_time-metrics.arrival_time
        return {**response_fields(self.processor,list(output.token_ids),self.cfg,text),'engine_output_text':output.text,'input_tokens':len(result.prompt_token_ids),'generated_tokens':len(output.token_ids),'generated_ids':list(output.token_ids),
            'preprocess_seconds':gen_start-start,'ttft_seconds':ttft,'ttft_kind':'engine first token plus client preprocessing; null when unavailable',
            'generation_seconds':end-gen_start,'total_seconds':end-start,'length_limit_reached':output.finish_reason=='length','finish_reason':output.finish_reason,
            'effective_max_new_tokens':output_budget(self.cfg,len(result.prompt_token_ids)),
            'processor_settings':{**template_kwargs(self.cfg),**kwargs,'assistant_prefix':self.cfg.get('assistant_prefix'),'input_mode':'chronological multi-image for sampled videos'},
            'engine_metrics':{k:v for k,v in vars(metrics).items() if isinstance(v,(int,float,str,bool,type(None)))} if metrics is not None else None}
