import copy
import hashlib
import json
import pytest
from vlmbench.dashboard import AXES,build_dashboard,collect,speed_score

def summary(model='local-fixture',backend='vllm_multimodal'):
    groups={name:{'n':24-(i%2),'run_n':24,'correct':12,'accuracy':12/(24-i%2)} for i,(name,_,_) in enumerate(AXES)}
    n=sum(v['n'] for v in groups.values())
    return {'model':model,'suite_sha256':'frozen-suite','reviewed_case_set_sha256':'reviewed-cases','effective_case_ids':[str(i) for i in range(n)],
            'outputs_sha256':'unchanged-answers','execution_context':{'backend':backend,'thinking':False},
            'overall':{'n':n,'run_n':144,'correct':72,'ok':144,'total_seconds':{'n':144,'median':1.,'p90':2.}},'datasets':groups}

def scores(tmp_path,rows,name='scores'):
    directory=tmp_path/name;directory.mkdir();p=directory/'comparison.json';p.write_text(json.dumps(rows));return p

def test_speed_scale_and_missing_values():
    assert speed_score(1)==50 and speed_score(.5)>50>speed_score(2)
    for missing in [None,0,-1,float('nan'),True]:assert speed_score(missing) is None
    for bad in [0,-1,float('nan'),True]:
        with pytest.raises(ValueError):speed_score(1,bad)

def test_one_score_dynamic_denominators_and_full_run_latency(tmp_path):
    rows=[summary(),summary('api-fixture','openai_compatible_api')]
    path=scores(tmp_path,rows);before=path.read_bytes();data=build_dashboard([path],tmp_path/'dashboard')
    assert data['core_items']==141 and data['run_items']==144
    assert data['models'][0]['metrics']['accuracy']['BLINK']['n']==23
    assert data['models'][0]['metrics']['speed']['timed_requests']==144
    assert [m['group'] for m in data['models']]==['local','api']
    assert path.read_bytes()==before and data['sources'][0]['sha256']==hashlib.sha256(before).hexdigest()
    html=(tmp_path/'dashboard/dashboard.html').read_text()
    assert 'id="protocol"' not in html and '<script src=' not in html and 'DASHBOARD_PAYLOAD' not in html
    assert (tmp_path/'dashboard/source-summaries/01-comparison.json').read_bytes()==before

def test_failed_fragments_do_not_gain_speed(tmp_path):
    row=summary();row['overall']['ok']=2;row['overall']['total_seconds'].update(n=2,median=.01)
    data=collect([scores(tmp_path,[row])]);metric=data['models'][0]['metrics']
    assert metric['speed']['score'] is None and metric['core']['failed']==142

def test_reject_missing_review_different_case_sets_and_duplicate_models(tmp_path):
    row=summary();missing=copy.deepcopy(row);missing.pop('reviewed_case_set_sha256')
    with pytest.raises(ValueError,match='review evidence'):collect([scores(tmp_path,[missing],'missing')])
    other=copy.deepcopy(row);other['model']='second';other['effective_case_ids'][0]='changed'
    with pytest.raises(ValueError,match='Effective case sets'):collect([scores(tmp_path,[row,other],'mismatch')])
    with pytest.raises(ValueError,match='one reviewed result'):collect([scores(tmp_path,[row,row],'duplicate')])
    bad=copy.deepcopy(row);bad['overall']['run_n']=24
    with pytest.raises(ValueError,match='completed core'):collect([scores(tmp_path,[bad],'partial')])

def test_sources_protected_fixed_scale_and_script_escape(tmp_path):
    row=summary('</script><script>alert(1)</script>');path=scores(tmp_path,[row])
    initial=collect([path])['models'][0]['metrics']['speed']['score']
    slow=summary('slow');slow['overall']['total_seconds']['median']=300
    assert collect([scores(tmp_path,[row,slow],'more')])['models'][0]['metrics']['speed']['score']==initial
    with pytest.raises(ValueError,match='separate'):build_dashboard([path],path.parent/'output')
    build_dashboard([path],tmp_path/'output')
    html=(tmp_path/'output/dashboard.html').read_text()
    assert '</script><script>alert(1)' not in html and '\\u003c/script\\u003e' in html
    with pytest.raises(FileExistsError):build_dashboard([path],tmp_path/'output')
