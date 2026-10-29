"""Gold-independent final-answer extraction shared by every evaluated model."""
import re
import unicodedata

PROTOCOL = 'public-benchmark-final-answer-v3'


def clean(text):
    text = unicodedata.normalize('NFKC', str(text)).strip()
    # Some tokenizers do not mark their native EOS/next-role token as special.
    # Remove terminal protocol delimiters only; preserve the raw response.
    text = re.sub(r'(?:<\|(?:im_end|endoftext|user|assistant|end|eot_id)\|>\s*)+$', '', text).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I)
    if re.search(r'<think>', text, re.I):
        text = re.split(r'<think>', text, flags=re.I)[0]
    text = re.sub(r'```(?:text|plaintext|markdown)?\s*\n?(.*?)```', r'\1', text, flags=re.S | re.I)
    text = re.sub(r'\*\*|__', '', text)
    return text.strip()


def boxes(text):
    found = [(m.start(), m.end(), m.group(1)) for m in re.finditer(
        r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text, re.S)]
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end(); depth = 1
        for index in range(start, len(text)):
            if text[index] == '{': depth += 1
            elif text[index] == '}': depth -= 1
            if depth == 0:
                found.append((match.start(), index+1, text[start:index]))
                break
    return sorted(found)


def atom(text):
    value = clean(text)
    value = re.sub(r'\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}', r'\1', value)
    value = re.sub(r'\\(?:\[|\]|\(|\))', '', value)
    value=value.strip(' \t\n`*').strip()
    if len(value)>1 and value.startswith('$') and value.endswith('$'):value=value.strip('$').strip()
    return value


FINAL = re.compile(r'(?:\b(?:(?:the\s+)?(?:final|correct)\s+)?answer\s*(?:is\s*[:：]?\s*|[:：]\s*)'
                   r'|\b(?:the\s+)?(?:correct\s+)?(?:option|choice|order|relation|action(?:\s+to\s+take\s+next)?)\s+is\s*[:：]?\s*'
                   r'|(?:最终答案|正确答案|答案)\s*(?:是|为|[:：])\s*)', re.I)


def surfaces(answer):
    """Prioritize explicit final declarations and terminal answer boxes/lines."""
    text = clean(answer)
    marked = list(FINAL.finditer(text))
    if marked:
        yield atom(text[marked[-1].end():]), 'explicit_final'
    wrapped = boxes(text)
    if wrapped:
        # Multiple distinct boxes are ambiguous unless a final declaration selects one.
        values = {atom(value) for _, _, value in wrapped}
        if len(values) == 1 or not text[wrapped[-1][1]:].strip(' \n\t.$`*'):
            yield atom(wrapped[-1][2]), 'answer_box'
    lines = [atom(line) for line in text.splitlines() if atom(line)]
    if lines:
        yield lines[-1], 'last_line'
        yield lines[0], 'first_line'
    yield atom(text), 'whole_answer'


def choice_text(text):
    return re.sub(r'\s+', ' ', atom(text).strip(' \n\t.,:;!?"\'`')).casefold()


def parse_choice(answer, choices):
    allowed = {chr(65+i): value for i, value in enumerate(choices)}
    text=clean(answer)
    lines=[atom(line) for line in text.splitlines() if atom(line)]
    if lines:
        terminal=re.fullmatch(r'[\[(]?([A-Ha-h])[\])]?[\s.!。]*',lines[-1])
        if terminal and terminal.group(1).upper() in allowed:
            return {'parsed':terminal.group(1).upper(),'method':'terminal_letter','surface':lines[-1],'issue':None}
    assertions=list(re.finditer(r'\b([A-H])\s+is\s+the\s+(?:correct|right)\s+(?:answer|option|choice)\b',text))
    assertions+=list(re.finditer(r'\bis\s+([A-H])\s*\(the\s+(?:first|second|third|fourth|fifth|sixth)\s+image\)',text))
    assertions+=list(re.finditer(r'\b(?:correct\s+description|next\s+action)[^.!?\n]{0,100}\bis\s+([A-H])\b',text))
    if assertions:
        selected=max(assertions,key=lambda m:m.start()).group(1)
        if selected in allowed:return {'parsed':selected,'method':'explicit_selection_sentence','surface':text,'issue':None}
    for surface, method in surfaces(answer):
        wrapped = boxes(surface)
        if wrapped and len({atom(v) for _, _, v in wrapped}) == 1:
            surface = atom(wrapped[-1][2])
        if method=='explicit_final' and re.match(r'^[\[(]?[A-H][\])]?\s+(?:or|and)\s+[\[(]?[A-H]\b',surface):
            return {'parsed':None,'method':method,'surface':surface,'issue':'multiple_final_options'}
        # An explicit letter must be a whole token or a leading option label.
        single = re.fullmatch(r'[\[(]?([A-Ha-h])[\])]?[\s.!。]*', surface)
        label = re.match(r'^[\[(]?([A-H])[\])]?(?:[.:：、)]\s+|\s*\n)', surface)
        match = single or label
        if match and match.group(1).upper() in allowed:
            if re.search(r'\b[A-H]\s+(?:or|and)\s+[A-H]\b',surface):
                return {'parsed':None,'method':method,'surface':surface,'issue':'multiple_final_options'}
            if label and method=='whole_answer' and len(lines)>1:
                continue
            if label and method in ('first_line','last_line','whole_answer'):
                enumerated=re.findall(r'(?m)^\s*([A-H])[.)]\s+',text)
                if len(set(enumerated))>1:
                    continue  # A cut-off option-by-option analysis has no final selection.
                if (method=='first_line' or len(lines)==1) and re.search(r'\b(?:wait|reconsider|but actually)\b',text[match.end():],re.I):
                    continue  # An initial selection followed by self-retraction is not final.
            letter = match.group(1).upper()
            return {'parsed':letter,'method':method,'surface':surface,'issue':None}
        # Text matching uses only the supplied choices, never the reference label.
        matches = [key for key, value in allowed.items() if choice_text(surface)==choice_text(value)]
        if len(matches)==1:
            return {'parsed':matches[0],'method':method+'_choice_text','surface':surface,'issue':None}
        if method == 'explicit_final':
            first = atom(surface.splitlines()[0]) if surface.splitlines() else ''
            match = re.match(r'^[\[(]?([A-H])[\])]?(?=[\s.,:;!?。]|$)', first)
            if match and match.group(1) in allowed:
                return {'parsed':match.group(1),'method':method,'surface':first,'issue':None}
            # Do not fall back to an earlier answer when the explicit final contradicts it.
            return {'parsed':None,'method':method,'surface':surface,'issue':'unresolved_explicit_final'}
    return {'parsed':None,'method':'unresolved','surface':clean(answer),'issue':'no_unique_final_option'}


def parse_yesno(answer):
    for surface, method in surfaces(answer):
        wrapped=boxes(surface)
        if wrapped:surface=atom(wrapped[-1][2])
        match=re.match(r'^(yes|no)\b',surface,re.I)
        if match:
            return {'parsed':match.group(1).lower(),'method':method,'surface':surface,'issue':None}
        if method=='explicit_final':
            return {'parsed':None,'method':method,'surface':surface,'issue':'unresolved_explicit_final'}
    return {'parsed':None,'method':'unresolved','surface':clean(answer),'issue':'no_explicit_yesno'}


def parse_text(answer):
    text=clean(answer)
    wrapped=boxes(text)
    if wrapped:
        return {'parsed':atom(wrapped[-1][2]),'method':'answer_box','surface':wrapped[-1][2],'issue':None}
    marked=list(FINAL.finditer(text))
    if marked:
        text=text[marked[-1].end():].strip()
        method='explicit_final'
    else:method='whole_answer'
    value=atom(text)
    # Presentation quotes and math delimiters, without changing actual OCR characters.
    if len(value)>1 and (value[0],value[-1]) in [('"','"'),("'","'"),('“','”')]:value=value[1:-1]
    return {'parsed':value,'method':method,'surface':text,'issue':None if value else 'empty_final_answer'}


def parse_answer(case, answer):
    if case['task']=='mcq':return parse_choice(answer,case['choices'])
    if case['task']=='yesno':return parse_yesno(answer)
    result=parse_text(answer)
    if case['task']=='chart':
        from .chart_answers import chart_conclusion, terminal_answer
        if result['method']!='answer_box':
            terminal=terminal_answer(result['parsed'],case['question'],case.get('unit_context'))
            if terminal is not None:
                result={**result,'parsed':terminal,'method':'chart_terminal_line','surface':terminal,'issue':None}
        extracted=chart_conclusion(result['parsed'],case['question'],case.get('unit_context'))
        if extracted:
            method=(result['method']+'/' if result['method']=='chart_terminal_line' else '')+extracted[1]
            result={**result,'parsed':extracted[0],'method':method,'issue':None}
    return result
