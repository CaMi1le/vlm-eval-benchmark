import json
import pytest
from vlmbench.common import sha
from vlmbench.generation import output_budget, torch_generation_kwargs, vllm_sampling_kwargs, response_fields, messages_for, template_evidence
from vlmbench.selection import planned_jobs


def test_natural_eos_uses_entire_remaining_context_and_has_no_wall_timeout():
    config={'max_model_len':131072,'max_new_tokens':None}
    assert output_budget(config,8291)==122781
    assert torch_generation_kwargs(config,8291)=={'do_sample':False,'max_new_tokens':122781}
    assert vllm_sampling_kwargs(config)=={'temperature':0,'max_tokens':None}
    assert output_budget(dict(config,max_new_tokens=128),8291)==128
    with pytest.raises(ValueError):output_budget(config,131072)


def test_native_thinking_system_and_sampling_are_explicit():
    case={'media':[],'prompt':'Answer with only the option letter.'}
    config={'thinking':True,'thinking_system_prompt':'Official R1 prompt','sampling':{'do_sample':True,'temperature':.6,'top_p':.95}}
    assert messages_for(config,case,'.',True)[0]['content']=='Official R1 prompt'
    assert messages_for(dict(config,thinking=False),case,'.',True)[0]['role']=='user'
    assert vllm_sampling_kwargs(config)=={'temperature':.6,'top_p':.95,'max_tokens':None}


class Tokenizer:
    all_special_tokens=['<think>','</think>','<|im_end|>']
    def decode(self,ids,skip_special_tokens=False):
        text=''.join(ids)
        if skip_special_tokens:
            for token in self.all_special_tokens:text=text.replace(token,'')
        return text


def test_prompt_opened_reasoning_is_not_mistaken_for_final_answer():
    raw='The answer might be A. Wait, reconsider.\n</think>\nB<|im_end|>'
    row=response_fields(Tokenizer(),list(raw),{'thinking':True},'assistant\n<think>\n')
    assert row['raw_generated_text']==raw
    assert row['answer'].startswith('<think>')
    assert row['final_answer_text']=='B' and not row['reasoning_incomplete']
    from vlmbench.answer_parser import parse_choice
    assert parse_choice(row['answer'],['x','y'])['parsed']=='B'
    incomplete=response_fields(Tokenizer(),list('Answer is A'),{'thinking':True},'assistant\n<think>\n')
    assert incomplete['final_answer_text']=='' and incomplete['reasoning_incomplete']
    assert parse_choice(incomplete['answer'],['x','y'])['parsed'] is None


def test_terminal_native_eos_cannot_break_numeric_final_answer():
    from vlmbench.answer_parser import parse_text
    assert parse_text('<think>try 1</think>\n1055<|user|>')['parsed']=='1055'


def test_thinking_content_never_decides_accuracy_or_final_format():
    from vlmbench.benchmark_score import score_answer
    case={'task':'mcq','choices':['one','two']};reference={'answers':['B']}
    result=score_answer(case,reference,'<think>The answer might be A.</think>\nB<|im_end|>')
    assert result['correct'] and result['format_ok']
    assert not score_answer(case,reference,'<think>Answer: B')['correct']
    assert score_answer(case,reference,'**B**')['format_ok'] is False


def test_flag_ignored_by_template_is_rejected(tmp_path):
    class Ignored:
        def apply_chat_template(self,*args,**kwargs):return 'same prompt'
    with pytest.raises(AssertionError,match='actual prompt'):
        template_evidence(Ignored(),{'path':str(tmp_path),'thinking':True})


def test_request_subset_is_hash_bound_and_preserves_diagnostic_conditions(tmp_path):
    (tmp_path/'freeze.json').write_text('{}')
    (tmp_path/'views.json').write_text(json.dumps({'core':['a','b'],'smoke':['b'],'no_image_diagnostic':['a']}))
    cases=[{'id':'a','input_sha256':'aaa'},{'id':'b','input_sha256':'bbb'}]
    request={'id':'a','condition':'no_image','input_sha256':'aaa'}
    selection={'suite_sha256':sha(tmp_path/'freeze.json'),'requests':[request]}
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    assert planned_jobs(tmp_path,cases,diagnostics=True,selection=path)==[('a','no_image')]
    with pytest.raises(AssertionError):planned_jobs(tmp_path,cases,diagnostics=False,selection=path)
    request['input_sha256']='tampered';path.write_text(json.dumps(selection))
    with pytest.raises(AssertionError,match='input'):planned_jobs(tmp_path,cases,diagnostics=True,selection=path)
