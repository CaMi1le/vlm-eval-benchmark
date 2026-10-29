"""Native thinking controls, natural-EOS budgets, and lossless answer boundaries."""
import hashlib
import re
from pathlib import Path
from .common import image_messages, sha


def messages_for(config, case, root, no_image=False):
    messages = image_messages(case, root, no_image)
    if config.get('thinking') and config.get('thinking_system_prompt'):
        messages.insert(0, {'role': 'system', 'content': config['thinking_system_prompt']})
    return messages


def template_kwargs(config):
    result = {'enable_thinking': bool(config.get('thinking', False))}
    if config.get('reasoning_effort') is not None:
        result['reasoning_effort'] = config['reasoning_effort']
    return result


def template_evidence(processor, config):
    case = {'media': [], 'prompt': 'What is 1 + 1?'}
    rendered = {}
    for thinking in (False, True):
        cfg = dict(config, thinking=thinking)
        rendered[str(thinking).lower()] = processor.apply_chat_template(
            messages_for(cfg, case, Path('.'), True), tokenize=False,
            add_generation_prompt=True, **template_kwargs(cfg))
    if config.get('thinking'):
        assert rendered['true'] != rendered['false'], 'Thinking flag does not change the actual prompt'
    template_files = {}
    for name in ('chat_template.jinja', 'chat_template.json', 'tokenizer_config.json'):
        path = Path(config['path']) / name
        if path.is_file():
            template_files[name] = sha(path)
    return {'templates': rendered, 'template_files_sha256': template_files,
            'native_control_changes_prompt': rendered['true'] != rendered['false']}


def output_budget(config, input_tokens):
    remaining = int(config['max_model_len']) - int(input_tokens)
    if remaining <= 0:
        raise ValueError(f'Input occupies context: {input_tokens}/{config["max_model_len"]}')
    requested = config.get('max_new_tokens')
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError('max_new_tokens must be positive or null (all remaining context)')
    return remaining if requested is None else min(requested, remaining)


def torch_generation_kwargs(config, input_tokens):
    sampling = dict(config.get('sampling', {'do_sample': False}))
    sampling['max_new_tokens'] = output_budget(config, input_tokens)
    # No hidden time limit. A caller can explicitly configure a positive one.
    if config.get('max_time_seconds') is not None:
        assert config['max_time_seconds'] > 0
        sampling['max_time'] = config['max_time_seconds']
    return sampling


def vllm_sampling_kwargs(config):
    sampling = dict(config.get('sampling', {'do_sample': False}))
    do_sample = sampling.pop('do_sample', False)
    if not do_sample:
        sampling = {'temperature': 0}
    sampling['max_tokens'] = config.get('max_new_tokens')
    # None instructs vLLM to use the remaining model context, never its default 16.
    return sampling


def response_fields(processor, ids, config, rendered_prompt):
    tokenizer = getattr(processor, 'tokenizer', processor)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    plain = tokenizer.decode(ids, skip_special_tokens=True)
    canonical = raw
    keep = {'<think>', '</think>', '<|begin_of_box|>', '<|end_of_box|>'}
    for special in sorted(set(tokenizer.all_special_tokens) - keep, key=len, reverse=True):
        canonical = canonical.replace(special, '')
    opened_in_prompt = rendered_prompt.rfind('<think>') > rendered_prompt.rfind('</think>')
    if opened_in_prompt:
        canonical = '<think>\n' + canonical
    blocks = list(re.finditer(r'<think>(.*?)</think>', canonical, re.S))
    incomplete = canonical.rfind('<think>') > canonical.rfind('</think>')
    final = re.sub(r'<think>.*?</think>', '', canonical, flags=re.S).strip()
    if incomplete:
        final = final.split('<think>')[0].strip()
    reasoning = '\n'.join(m.group(1).strip() for m in blocks)
    if incomplete:
        reasoning += canonical[canonical.rfind('<think>') + len('<think>'):]
    return {'answer': canonical.strip(), 'raw_generated_text': raw,
            'raw_generated_text_skip_special': plain, 'final_answer_text': final,
            'reasoning_text': reasoning.strip(), 'reasoning_incomplete': incomplete,
            'thinking_requested': bool(config.get('thinking', False)),
            'thinking_opened_in_prompt': opened_in_prompt,
            'rendered_prompt_sha256': hashlib.sha256(rendered_prompt.encode()).hexdigest(),
            'rendered_prompt_suffix': rendered_prompt[-256:]}
