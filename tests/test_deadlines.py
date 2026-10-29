import pytest
from vlmbench.time_policy import enforce_deadline
from vlmbench.bounded_batch import completed_prefix


@pytest.mark.parametrize('elapsed,expected',[(899.9,'ok'),(900,'ok'),(900.01,'timeout'),(1587.8,'timeout')])
def test_uniform_deadline_discards_late_eos_without_changing_source(elapsed,expected):
    row={'id':'case','condition':'image','status':'ok','answer':'A','total_seconds':elapsed}
    result=enforce_deadline(row)
    assert result['status']==expected
    assert row['answer']=='A' and row['status']=='ok' and row['total_seconds']==elapsed
    if expected=='timeout':
        assert result['answer']=='' and result['model_case_abandoned']
        assert result['observed_total_seconds']==elapsed and result['total_seconds']==900


def test_chunk_timeout_advances_to_remaining_cases_without_retrying_failed_case():
    rows=[{'id':'done','condition':'image','status':'ok','answer':'B','total_seconds':3},
          {'id':'slow','condition':'image','status':'not_run'},
          {'id':'next','condition':'image','status':'not_run'}]
    timeout={'id':'slow','condition':'image','observed_elapsed_seconds':900.2}
    outcomes=completed_prefix(rows,timeout,900)
    assert [(r['id'],r['status']) for r in outcomes]==[('done','ok'),('slow','timeout')]
    remaining=[r['id'] for r in rows if r['id'] not in {o['id'] for o in outcomes}]
    assert remaining==['next']
    assert outcomes[0]==rows[0]


def test_prior_shorter_timeout_is_not_retried_or_reclassified_as_a_900_second_wait():
    prior={'id':'case31','condition':'image','status':'timeout','answer':'','total_seconds':600,'retry_discarded':True}
    assert enforce_deadline(prior)==prior
    assert completed_prefix([prior],None,900)==[prior]


def test_thinking_uses_same_budget_without_erasing_native_thought_boundaries():
    row={'id':'thinking','condition':'image','status':'ok','answer':'<think>reconsider</think>\nB',
         'reasoning_text':'reconsider','thinking_requested':True,'total_seconds':1200}
    result=enforce_deadline(row,900,phase='thinking')
    assert result['status']=='timeout' and result['answer']==''
    assert result['timeout_policy']['schema']=='thinking-900s-20261020'
    assert row['answer'].endswith('B') and result['thinking_requested']


def test_nonthinking_historical_policy_remains_byte_equivalent_after_thinking_support():
    row={'id':'old','condition':'image','status':'ok','answer':'A','total_seconds':1000}
    result=enforce_deadline(row)
    assert result['error']=='Nonthinking response exceeded the declared deadline; this model abandons this case.'
    assert result['timeout_policy']['schema']=='nonthinking-900s-20261020'


def test_no_timeout_silently_hides_an_unfinished_placeholder():
    assert completed_prefix([{'id':'pending','condition':'image','status':'not_run'}],None,900)==[]


def test_timeout_targets_exact_model_case_condition():
    row={'id':'same','condition':'no_image','status':'not_run'}
    timeout={'id':'same','condition':'image','observed_elapsed_seconds':901}
    assert enforce_deadline(row,900,timeout)==row


def test_bounded_batch_restarts_engine_for_only_unfinished_cases(tmp_path,monkeypatch):
    import json,sys
    from pathlib import Path
    from vlmbench import bounded_batch
    from vlmbench.common import write,sha
    root=tmp_path/'root';suite=root/'data/public-vlm-v2';suite.mkdir(parents=True)
    (suite/'freeze.json').write_text('{}')
    write(suite/'views.json',{'core':['slow','next'],'smoke':[],'no_image_diagnostic':[]})
    cases=[{'id':i,'input_sha256':i+'-hash'} for i in ['slow','next']]
    selection=root/'provenance/selected.json';write(selection,{'suite_sha256':sha(suite/'freeze.json'),'requests':[dict(c,condition='image') for c in cases]})
    cfg={'thinking':False};config=root/'configs/test.json';write(config,{'model':cfg})
    monkeypatch.setattr(bounded_batch,'verify_suite',lambda _:cases)
    selections=[]
    class Child:
        def __init__(self,command,**kwargs):
            runid=command[command.index('--run-id')+1]
            requests=json.loads(Path(command[command.index('--selection')+1]).read_text())['requests']
            selections.append([r['id'] for r in requests]);self.pid=1000+len(selections)
            d=root/'runs'/runid/'model';result=d/'result';result.mkdir(parents=True)
            first=len(selections)==1;self.code=1 if first else 0
            records=[{**r,'model':'model','status':'not_run' if first else 'ok','answer':'' if first else 'B','total_seconds':1} for r in requests]
            (result/'outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            write(result/'run.json',{'model':'model','config':cfg})
            write(d/'gpu-release.json',{'worker_gone':True,'container_gone':True,'target_gpus_idle':True,'before':{},'after':{},
                'request_timeout':{'id':'slow','condition':'image','observed_elapsed_seconds':900.1} if first else None})
        def wait(self):return self.code
    monkeypatch.setattr(bounded_batch.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['bounded','--root',str(root),'--run-id','bounded-test','--model','model','--models-config',str(config),'--selection',str(selection)])
    bounded_batch.main()
    assert selections==[['slow','next'],['next']]
    result=root/'runs/bounded-test/model'
    records=[json.loads(l) for l in (result/'result/outputs.jsonl').read_text().splitlines()]
    assert [(r['id'],r['status']) for r in records]==[('slow','timeout'),('next','ok')]
    assert json.loads((result/'bounded-state.json').read_text())['finished']
    assert json.loads((result/'gpu-release.json').read_text())['worker_gone']
