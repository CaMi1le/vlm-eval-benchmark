import pytest
from vlmbench.reviewed import equivalent,grade,quantity
from vlmbench.score import aggregate

def number(value,unit):return {'value':value,'unit':unit}

def test_units_preserve_dimensions_and_magnitude():
    case={'kind':'number','unit':'million_USD','relative_tolerance':.05}
    assert equivalent(case,number(1.055,'billion_USD'),number(1055,'million_USD'))
    assert not equivalent(case,number(1055,'billion_USD'),number(1055,'million_USD'))
    assert not equivalent(case,number(1055,'million_GBP'),number(1055,'million_USD'))
    assert quantity(number(600,'kg_per_hectare'),'tonne_per_hectare')==.6

def test_percentage_scale_and_zero():
    case={'kind':'number','unit':'percent','relative_tolerance':.05}
    gold=number(30,'percent')
    assert equivalent(case,number(.3,'ratio'),gold)
    assert equivalent(case,number(30,'percent'),gold)
    assert not equivalent(case,number(.3,'percent'),gold)
    assert not equivalent(case,number(30,'ratio'),gold)
    assert equivalent(case,number(0.0,'percent'),number(0,'percent'))
    assert not equivalent(case,number(.001,'percent'),number(0,'percent'))

def test_lists_and_ocr_keep_actual_content():
    case={'kind':'list'}
    assert equivalent(case,[2010,2003],[2003,2010])
    assert not equivalent(case,[2003],[2003,2010])
    assert not equivalent(case,[2003,2009],[2003,2010])
    assert equivalent({'kind':'text'},' RAINBOW ','rainbow')
    assert not equivalent({'kind':'text'},'spring','springs')
    assert not equivalent({'kind':'text'},'1231','1931')
    assert not equivalent({'kind':'text'},'No, but rainbow','rainbow')
    assert equivalent({'kind':'formula'},'$$y = -\\frac {19}{2}$$','y=-\\frac{19}{2}')
    assert not equivalent({'kind':'formula'},'pa','PA')

def test_final_conclusion_and_errors_not_reference_mentions():
    case={'kind':'choice','answers':['B'],'decision':'retain','reason':'Image marker A is option B.'}
    judged={'canonical':'A','reason':'Explicit final option A, despite mentioning B in reasoning.','format_issue':True}
    assert not grade(case,judged)['correct']
    judged['canonical']=None
    assert not grade(case,judged)['correct']
    judged['canonical']='B'
    assert grade(case,judged)['correct'] and not grade(case,judged,'error')['correct']

def test_exclusion_keeps_full_runtime_and_failure_denominator():
    base={'task':'mcq','status':'ok','format_ok':True,'parsed':'A','total_seconds':1,'correct':True,'eligible':True}
    rows=[base,{**base,'eligible':False,'correct':None,'total_seconds':100},{**base,'status':'error','correct':False,'total_seconds':None}]
    result=aggregate(rows)
    assert (result['correct'],result['n'],result['run_n'],result['failed'])==(1,2,3,1)
    assert result['total_seconds']['n']==2 and result['total_seconds']['median']==50.5


def test_restored_chart_death_rate_keeps_units_and_final_conclusion():
    case={'kind':'number','unit':'per_100000','relative_tolerance':.05,
          'answers':[number(47.68,'per_100000')],'decision':'retain','reason':'50.18 - 2.5 = 47.68.'}
    judgment={'canonical':number(47.68,'per_100000'),'reason':'Explicit arithmetic conclusion.','format_issue':True}
    assert grade(case,judgment)['correct']
    assert equivalent(case,number(.04768,'percent'),case['answers'][0])
    judgment['canonical']=number(64,'year')
    assert not grade(case,judgment)['correct']  # Correct rate only offered as a conditional alternative.
    assert not equivalent(case,number(45,'per_100000'),case['answers'][0])
