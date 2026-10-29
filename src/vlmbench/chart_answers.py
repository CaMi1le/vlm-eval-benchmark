"""Extract unambiguous chart conclusions without access to reference answers."""
import re
from .quantities import NUMBER, SCALES, UNITS, quantity

AMBIGUOUS=re.compile(r'\b(?:maybe|perhaps|either|uncertain|unsure|versus|rather than|not|never)\b|\bor\b|\b\w+n[’\']t\b',re.I)
PREDICATE=re.compile(r'\b(?:is|are|was|were|equals?|equaled|generated|made|spent|reached|amounts?\s+to|amounted\s+to|stood\s+at)\s+',re.I)
YEAR_QUESTION=re.compile(r'\b(?:what|which)\s+(?:(?:is|was)\s+(?:the\s+)?)?year\b',re.I)
BOOL_QUESTION=re.compile(r'^\s*(?:is|are|was|were|does|do|did|has|have|had|can|will|would)\b',re.I)


def terminal_answer(value,question,context=None):
    """Select an explicit terminal surface, independently of reference answers.

    A separate short final paragraph or a standalone terminal quantity/list/yes-no
    outranks preceding explanation. Preserve unknown units and alternatives so
    the scorer can reject them; never take a numeric substring from the line.
    """
    text=str(value).strip();lines=[line.strip() for line in text.splitlines() if line.strip()]
    if len(lines)<2:return None
    tail=lines[-1]
    if len(tail)>300 or re.match(r'^(?:[-*•]\s|\d+[.)]\s|[|#>]|(?:source|note|reference|disclaimer)\s*:)',tail,re.I):return None
    if re.search(r'\n[ \t]*\n[ \t]*[^\n]+\s*$',text):return tail
    if re.fullmatch(r'(?:yes|no)[.!。]?',tail,re.I) or quantity(tail,context,question):return tail
    if re.match(r'^(?:[$€£]\s*)?(?:approximately\s+|about\s+|around\s+)?[+-]?(?:\d|\.\d)',tail,re.I):return tail
    number=r'(?:'+NUMBER+r')'
    if re.fullmatch(r'\[?\s*'+number+r'(?:\s*(?:,|;|and)\s*'+number+r')+\s*\]?',tail,re.I):return tail
    return None


def _phrase(text):
    text=text.strip().strip('"“”').strip()
    # Keep decimal points and U.S. currency spelling; remove following commentary.
    text=re.split(r'\.(?:\s+(?=[A-Z])|$)',text,maxsplit=1)[0]
    unit_words='|'.join(re.escape(v) for v in sorted(set(UNITS)|set(SCALES),key=len,reverse=True))
    if re.search(r'\bin\s+(?:'+unit_words+r')\b',text,re.I):
        return text.strip(' \t\n.。!')  # Do not erase declarations such as "100 in millions".
    split=re.search(r',\s+(?=[A-Za-z])|;|\s+(?:in|during|for|as of|on|according to|which|with)\s+',text,re.I)
    if split:
        head,tail=text[:split.start()],text[split.end():]
        # A bare number followed by an unknown unit declaration must stay unresolved.
        bare=bool(re.fullmatch(r'(?:approximately\s+|about\s+|around\s+)?(?:'+NUMBER+r')',head,re.I))
        temporal=bool(re.match(r'(?:the\s+)?(?:\d{4}\b|(?:19|20)\d{2}/\d{2}\b)',tail,re.I))
        if not bare or temporal or split.group().lstrip().startswith((',', ';')):text=head
    return text.strip(' \t\n.。!')


def _quantity_surface(text,question,context):
    if AMBIGUOUS.search(text):return None
    value=_phrase(text)
    parenthetic=re.fullmatch(r'(.+?)\s*\(([^()]+)\)',value)
    if parenthetic:
        head,comment=parenthetic.groups()
        if re.match(r'^(?:in\s+)',comment,re.I):
            # "14 (in millions)" declares a scale, not disposable commentary.
            value=head+' '+re.sub(r'^in\s+','',comment,flags=re.I)
        elif re.fullmatch(r'(?:19|20)\d{2}',comment) or re.match(r'^(?:for|during|as of)\s+',comment,re.I):
            value=head.strip()
    parsed=quantity(value,context,question)
    if parsed:
        # Counts may carry ordinary nouns without declaring a different scale.
        if not context and parsed['dimension']=='count' and not re.search(r'\b(?:thousands?|millions?|billions?|trillions?|[kmb])\b',value,re.I):
            match=re.match('('+NUMBER+')',value)
            if match:return match.group(1)
        return value
    match=re.fullmatch(r'(?P<number>'+NUMBER+r')\s+(?P<nouns>[A-Za-z][A-Za-z -]*)',value)
    if match and re.search(r'\bhow many\b',question,re.I):
        nouns=set(re.findall(r'[a-z]+',match['nouns'].lower()))
        question_words=set(re.findall(r'[a-z]+',question.lower()))
        # Never strip an explicit scale/currency or physical unit to fit a number.
        reserved=set(SCALES)|set(word for unit in UNITS if UNITS[unit][0]!='count' for word in unit.split())
        if nouns<=question_words and not nouns & reserved:return match['number']
    return None


def chart_conclusion(value,question,context=None):
    """Return a presentation-normalized conclusion, or None when unresolved."""
    text=str(value).strip()
    if not text:return None
    if re.fullmatch(r'(?:yes|no)[.!。]?',text,re.I):return text.rstrip('.!。').lower(),'chart_yesno'
    if BOOL_QUESTION.search(question):
        match=re.search(r'(?:^|[.!?]\s+|\b(?:so|therefore|thus)[,:]?\s+)(yes|no)[.!。]?\s*$',text,re.I)
        if match:return match.group(1).lower(),'chart_terminal_yesno'
        return None
    if AMBIGUOUS.search(text):return None
    if YEAR_QUESTION.search(question):
        if re.search(r'\b\d{4}\s*[-/–]\s*\d{2,4}\b',text):return None
        years=set(re.findall(r'(?<![\d.])(?:18|19|20|21)\d{2}(?!\d|\.\d)',text))
        if len(years)==1:return years.pop(),'chart_year_statement'
        return None
    # Bare values already handled by the standard numeric/unit scorer stay intact.
    if quantity(text,context,question):return None
    candidates=[]
    if re.match(r'^(?:[$€£]\s*)?(?:approximately\s+|about\s+|around\s+)?[+-]?\d',text,re.I):
        value=_quantity_surface(text,question,context)
        if value:candidates.append(value)
    for predicate in PREDICATE.finditer(text):
        value=_quantity_surface(text[predicate.end():],question,context)
        if value:candidates.append(value)
    unique=set(candidates)
    if len(unique)==1:return candidates[0],'chart_quantity_statement'
    if len(unique)>1:return None
    # Quoted entities and a single copular conclusion are independent of gold text.
    quoted=re.findall(r'["“]([^"“”]+)["”]',text)
    if len(set(quoted))==1 and quoted and len(list(PREDICATE.finditer(text)))==1:
        return quoted[0].strip(),'chart_quoted_entity'
    copulas=list(re.finditer(r'\b(?:is|was|are|were)\s+',text,re.I))
    if len(copulas)==1:
        tail=_phrase(text[copulas[0].end():]).strip('"“”\'')
        if tail and len(tail.split())<=12 and not re.search(r'\b(?:not|because|however|but)\b',tail,re.I):
            return tail,'chart_copular_statement'
    return None
