import json
from vlmbench.batch import smoke_gate,supervisor_command

def test_expanded_suite_and_deadline_reach_both_stages(tmp_path):
    for view in ['smoke','core']:
        args=supervisor_command(tmp_path,'small-model','round-v2',view,'public-vlm-v2',14400)
        assert args[args.index('--suite')+1]=='public-vlm-v2'
        assert args[args.index('--deadline')+1]=='14400'
        assert args[args.index('--view')+1]==view
        assert '--diagnostics' in args

def test_smoke_gate_uses_technical_completion_not_accuracy(tmp_path):
    planned=[{'id':'a','condition':'image'}]
    (tmp_path/'planned.json').write_text(json.dumps(planned))
    row={'id':'a','condition':'image','status':'ok','answer':'B','correct':False,'length_limit_reached':True}
    (tmp_path/'outputs.jsonl').write_text(json.dumps(row)+'\n')
    assert smoke_gate(tmp_path)['passed']
    row['answer']='';(tmp_path/'outputs.jsonl').write_text(json.dumps(row)+'\n')
    assert not smoke_gate(tmp_path)['passed']

def test_smoke_gate_rejects_missing_or_duplicate_requests(tmp_path):
    (tmp_path/'planned.json').write_text(json.dumps([{'id':'a','condition':'image'},{'id':'b','condition':'image'}]))
    row={'id':'a','condition':'image','status':'ok','answer':'A'}
    (tmp_path/'outputs.jsonl').write_text((json.dumps(row)+'\n')*2)
    assert not smoke_gate(tmp_path)['passed']
