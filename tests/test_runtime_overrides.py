import json
from vlmbench.additional_adapters import runtime_hf_overrides,multimodal_kwargs
from vlmbench.common import sha

def test_awq_alias_fix_keeps_original_exclusions_and_source_file(tmp_path):
    model={'quantization_config':{'quant_method':'awq','bits':4,'modules_to_not_convert':['in_proj_a','model.vision_tower']}}
    path=tmp_path/'config.json';path.write_text(json.dumps(model));before=sha(path)
    result=runtime_hf_overrides({'path':str(tmp_path),'quantization_skip_prefixes':['vpm','vit_merger']})
    assert sha(path)==before
    q=result['quantization_config'];assert q['quant_method']=='awq' and q['bits']==4
    assert set(q['modules_to_not_convert'])=={'in_proj_a','model.vision_tower','vpm','vit_merger'}

def test_internvl_image_budget_does_not_enter_video_processor_kwargs():
    cfg={'family':'internvl','hf_overrides':{'max_dynamic_patch':4}}
    assert runtime_hf_overrides(cfg)=={'max_dynamic_patch':4}
    assert multimodal_kwargs(cfg,{'media':[{}]*8})=={}
