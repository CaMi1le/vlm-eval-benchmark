import pytest
from vlmbench.answer_parser import parse_answer
from vlmbench.benchmark_score import score_answer


def case(question='What is the value?',context=None):
    return {'task':'chart','category':'human','question':question,'unit_context':context}


@pytest.mark.parametrize('question,answer,target',[
 ('What percentage?','The labeled bar is 5.1%.  \n\n\n5.1','5.1'),
 ('What is the difference?','Subtract the two values: 96.1 - 95.6 = 0.5.\n\n0.5','0.5'),
 ('Which marital group?','Married has 633 men and 524 women.\n\nMarried','Married'),
 ('Which country?','Since 39.6 is greater than 36.7, Netherlands has higher taxes.\n\nNetherlands','Netherlands'),
 ('What is the ratio?','The changes are 15 and 16.\n\n15/16','0.9375'),
 ('What two years?','The highest rates are 3.6%.\n\n2003 and 2009','[2003,2009]'),
 ('What percentage?','The value is 6.2%.\n6.2%','6.2'),
 ('Is it greater?','The intermediate sum is 2010.\n\nNo.','No'),
])
def test_explicit_terminal_conclusions_override_explanation(question,answer,target):
    result=score_answer(case(question),{'answers':[target]},answer)
    assert result['correct'],result
    assert result['method'].startswith('chart_terminal_line')


@pytest.mark.parametrize('tail',[
 '1055 billion U.S. dollars','1055 million GBP','1055 (in billion U.S. dollars)',
 '1055 quadrillion unknowncoin','1055 or 2000','I cannot determine the value.',
])
def test_terminal_surface_is_not_stripped_to_match_a_reference(tail):
    value=case('What is the revenue?',{'dimension':'USD','reference_scale':1e6})
    result=score_answer(value,{'answers':['1055']},'The earlier estimate was 1055 million U.S. dollars.\n\n'+tail)
    assert not result['correct'],result


def test_unknown_numeric_terminal_units_are_preserved_without_a_blank_line():
    value=case('What is the revenue?',{'dimension':'USD','reference_scale':1e6})
    text='The earlier estimate was 1055 million U.S. dollars. A caption follows.\n1055 unknowncoin'
    result=score_answer(value,{'answers':['1055']},text)
    assert result['parsed']=='1055 unknowncoin' and not result['correct']


def test_terminal_selection_is_independent_of_gold_and_never_uses_open_thoughts():
    text='<think>The estimate is 10.</think>\nThe final value was corrected.\n\n20'
    a=score_answer(case(),{'answers':['10']},text)
    b=score_answer(case(),{'answers':['20']},text)
    assert a['parsed']==b['parsed']=='20' and not a['correct'] and b['correct']
    assert not score_answer(case(),{'answers':['20']},'<think>Intermediate 10.\n\n20')['correct']


def test_boxes_ocr_and_multiline_lists_keep_their_existing_scope():
    text='\\boxed{10}\n\n20'
    assert parse_answer(case(),text)['parsed']=='10'
    assert parse_answer({'task':'ocr'},'First line\n\nSecond line')['parsed']=='First line\n\nSecond line'
    assert not score_answer(case('What two years?'),{'answers':['[2003,2009]']},'The values are shown.\n\n- 2009')['correct']
