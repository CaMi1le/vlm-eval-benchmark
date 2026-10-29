import pytest
from vlmbench.answer_parser import parse_answer
from vlmbench.benchmark_score import score_answer


def case(question,context=None):
    return {'task':'chart','question':question,'category':'human','unit_context':context}


@pytest.mark.parametrize('question,answer,target,context',[
 ('How many scheduled departures were there in 2019?','The number of departures in 2019 was approximately 9,500,000.','9464482',{'dimension':'count','reference_scale':1}),
 ('How much revenue did this cinema generate?','The cinema generated 98.96 million yuan in box office revenue.','98.96',{'dimension':'CNY','reference_scale':1e6}),
 ('What was GDP in the EU in 2020?','The GDP in the EU in 2020 was 13.31 trillion euros.','13.31',{'dimension':'EUR','reference_scale':1e12}),
 ('How many coronavirus cases were there in Qatar as of November 4, 2020?','As of November 4, 2020, there were 133,370 coronavirus cases in Qatar.','133370',None),
 ('How much money did the team make in 2019/20?','The team made 120 million U.S. dollars in the 2019/20 season.','120',{'dimension':'USD','reference_scale':1e6}),
 ('In what year did Belize score 0.7?','Belize scored 0.7 in 2021.','2021',None),
 ('Which marital group is highest?','The highest marital group is "Married", with 633 men.','Married',None),
 ('What number is the median bar?','The median bar is 16.7.','16.7',None),
 ('Is the product greater than 2000?','The product is 1023. So the answer is no.','No',None),
 ('What was the pre-COVID-19 visitor estimate?','The pre-COVID-19 estimate of visitors was 51 million.','51',{'dimension':'count','reference_scale':1e6}),
 ('How many users?','The number of users is 14 (in millions).','14',{'dimension':'count','reference_scale':1e6}),
])
def test_final_sentence_conclusions(question,answer,target,context):
    result=score_answer(case(question,context),{'answers':[target]},answer)
    assert result['correct'],result


@pytest.mark.parametrize('answer',[
 'The value is 1055 million GBP.',
 'The value is 1055 billion U.S. dollars.',
 'The value is 1055 million U.S. dollars or 2000 million U.S. dollars.',
 'The value is 1055 quadrillion unknowncoin.',
 'The value is not 1055 million U.S. dollars.',
 'The value is 1055 in billion U.S. dollars.',
 'The value is 1055 in quadrillions of unknowncoin.',
 'The value is 1055 (in billions of U.S. dollars).',
 'The value is 1055 (billion U.S. dollars).',
 'The value is 1055 (in quadrillions of unknowncoin).',
 'The value was 1055 million U.S. dollars. The value is 2000 million U.S. dollars.',
 '<think>The value is 1055 million U.S. dollars.',
])
def test_no_unit_erasure_ambiguity_or_thought_reference_fishing(answer):
    result=score_answer(case('What is the revenue?',{'dimension':'USD','reference_scale':1e6}),{'answers':['1055']},answer)
    assert not result['correct'],result


def test_extraction_does_not_depend_on_reference():
    value=case('What was GDP in 2020?',{'dimension':'EUR','reference_scale':1e12})
    text='The GDP in 2020 was 13.31 trillion euros.'
    a=score_answer(value,{'answers':['13.31']},text)
    b=score_answer(value,{'answers':['19.0']},text)
    assert a['parsed']==b['parsed']=='13.31 trillion euros'
    assert a['correct'] and not b['correct']


def test_ocr_text_and_multiple_years_are_not_rewritten():
    assert parse_answer({'task':'ocr'},'The value is 120.')['parsed']=='The value is 120.'
    assert not score_answer(case('Which year?'),{'answers':['2021']},'It was 2020 or 2021.')['correct']
    assert not score_answer(case('Which year?'),{'answers':['2021']},'0.2021')['correct']
    assert not score_answer(case('Which year?'),{'answers':['2021']},'It was not 2021.')['correct']
    assert not score_answer(case('Which year?'),{'answers':['2021']},"It wasn't 2021.")['correct']
    assert not score_answer(case('Which marital group is highest?'),{'answers':['Married']},'The highest group is not "Married".')['correct']
    assert score_answer(case('Which year?'),{'answers':['2019/20']},'2019/20')['correct']
