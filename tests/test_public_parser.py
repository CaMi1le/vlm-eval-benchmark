import json
from pathlib import Path
import pytest
from vlmbench.answer_parser import parse_answer, parse_choice, parse_text
from vlmbench.benchmark_score import chart_correct, score_answer

@pytest.mark.parametrize('answer,expected',[
    ('The correct answer is **B**.', 'B'),
    ('A is a label on the diagram. Final answer: B','B'),
    ('<think>Answer: A</think> **B**','B'),
    ('<think>Answer: A',None),
    ('The correct answer is No.\n\n**B**','B'),
    ('A. red\nB. blue\nI cannot choose between A or B.',None),
    ('B. blue\nWait, I need to reconsider the image.',None),
    ('The correct option is:\nC. green.','C'),
    ('The final answer is A or B.',None),
    ('I see point A, point B, and group C in the diagram.',None),
    ('\\boxed{\\text{B}}','B'),
    ('The answer is <|begin_of_box|>C<|end_of_box|>.','C'),
    ('Based on the image, **B** is the correct answer. Box A is wrong.','B'),
    ('The missing part is **B** (the third image).','B'),
    ('The correct description that matches the action in the video is **A**.','A'),
    ('The word "Answer:" is quoted, but I have not selected an option.',None),
])
def test_explicit_final_answers_and_reasoning_boundaries(answer,expected):
    assert parse_choice(answer,['red','blue','green','yellow'])['parsed']==expected


def test_same_parser_never_uses_gold_or_reasoning_field():
    case={'task':'mcq','choices':['red','blue']}
    for gold in ['A','B']:
        reference={'answers':[gold]}
        result=score_answer(case,reference,'Final answer: B')
        assert result['parsed']=='B' and result['correct']==(gold=='B')
    assert parse_text('The answer is <|begin_of_box|>5.1%<|end_of_box|>.')['parsed']=='5.1%'


def test_chart_tolerance_lists_units_and_no_reference_fishing():
    assert chart_correct('What percentage?', '5.1%', '5.1')
    assert not chart_correct('What percentage?', '0.051%', '5.1')
    assert chart_correct('What fraction?', '5.1%', '0.051')
    assert chart_correct('How many?', '1,055', '1055')
    assert chart_correct('How many?', '104', '100')
    assert not chart_correct('How many?', '106', '100')
    assert chart_correct('How many?', '0.0', '0')
    assert not chart_correct('How many?', '0.1', '0')
    assert chart_correct('What 2 years?', '2009 and 2003', '[2003,2009]')
    assert not chart_correct('What 2 years?', '2003', '[2003,2009]')
    assert not chart_correct('What year?', '2009', '2003')  # Year question: exact, no 5% tolerance.
    assert not chart_correct('How many?', 'Maybe 100 or 200', '100')


def test_ocr_published_substring_formula_case_and_format_separation():
    case={'task':'ocr','category':'Regular Text Recognition'}
    assert score_answer(case,{'answers':['play']},'playing')['correct']
    case['category']='Handwritten Mathematical Expression Recognition'
    assert score_answer(case,{'answers':['P A']},'PA')['correct']
    assert not score_answer(case,{'answers':['PA']},'pa')['correct']
    result=score_answer({'task':'mcq','choices':['red','blue']},{'answers':['B']},'**B**')
    assert result['correct'] and result['format_ok'] is False


def test_physical_units_reject_wrong_currency_and_magnitude():
    usd={'dimension':'USD','reference_scale':1e6,'bare_scales':[1e6,1]}
    assert chart_correct('Company value?', '1,055 million U.S. dollars','1055',usd)
    assert chart_correct('Company value?', '$1.055 billion','1055',usd)
    assert not chart_correct('Company value?', '1055 billion U.S. dollars','1055',usd)
    assert not chart_correct('Company value?', '1055 million GBP','1055',usd)
    assert not chart_correct('Company value?', 'Maybe 1055 million or 2000 million','1055',usd)
    assert chart_correct('Company value?', '1055000000','1055',usd)
    eur={'dimension':'EUR','reference_scale':1000,'bare_scales':[1000,1]}
    assert chart_correct('Revenue?', 'Around 506,000 thousand euros (approximately 506 million euros)','509175',eur)
    assert not chart_correct('Revenue?', '506 thousand euros (or 506 million euros)','509175',eur)


def test_chart_quantity_presentations_and_ratio_conversion():
    assert chart_correct('How many cases?', '133 370','133370')
    assert chart_correct('What is the ratio?', '15:16','0.9375')
    assert not chart_correct('What is the ratio?', '1:0','0.9375')
    percent={'dimension':'ratio','reference_scale':.01}
    assert chart_correct('What is the largest bar?', '53%','53',percent)
    assert not chart_correct('What is the largest bar?', '0.53%','53',percent)
    yield_unit={'dimension':'yield_tonne_ha','reference_scale':1}
    assert chart_correct('Crop yield?', '600 kg per hectare','0.6',yield_unit)
    assert not chart_correct('Crop yield?', '600 tonnes per hectare','0.6',yield_unit)
    rate={'dimension':'deaths_per_birth','reference_scale':.0001}
    assert chart_correct('Mortality?', '0.4 deaths per 10,000 live births','0.4',rate)
    assert not chart_correct('Mortality?', '0.4 deaths per 1,000 live births','0.4',rate)
