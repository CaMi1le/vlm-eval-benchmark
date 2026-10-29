import copy
import json
from pathlib import Path
import pytest
from test_api import api_server,reply,suite_fixture
from vlmbench.api import run_api
from vlmbench.benchmark_score import compare,score_run
from vlmbench.common import jsonl,json_hash,sha,write
from vlmbench.dashboard import collect
from test_dashboard import summary,scores


def prepared(tmp_path):
    suite=tmp_path/'suite';suite_fixture(suite)
    run=tmp_path/'run'
    with api_server(tmp_path,[reply()]) as (credentials,received):
        run_api(suite,credentials,run,concurrency=4)
    return suite,run


def test_public_labels_need_no_manual_review_and_keep_raw_bytes(tmp_path):
    suite,run=prepared(tmp_path);before=(run/'outputs.jsonl').read_bytes()
    summaries,rows=compare(suite,[run],tmp_path/'scores')
    assert summaries[0]['overall']['n']==summaries[0]['overall']['correct']==5
    assert summaries[0]['manual_content_review'] is False
    assert (run/'outputs.jsonl').read_bytes()==before
    assert len(rows['test-high'])==6
    assert summaries[0]['api_token_usage']['requests_with_usage']==6


def test_failures_and_recovery_timing_remain_accounted(tmp_path):
    suite,run=prepared(tmp_path);raw=jsonl(run/'outputs.jsonl')
    raw[0].update(status='error',answer='',total_seconds=300)
    raw[1]['recovery']={'first_attempt_seconds':300}
    (run/'outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
    result,rows=score_run(suite,run,tmp_path/'scores')
    assert result['overall']['n']==5 and result['overall']['correct']==4
    assert result['overall']['failed']==1
    assert rows[1]['timing_seconds']==rows[1]['total_seconds']+300
    assert result['timeout_recovery']['requests']==1
    assert 'cumulative' in result['execution_context']['latency_kind']


def test_wrong_input_or_duplicate_requests_rejected(tmp_path):
    suite,run=prepared(tmp_path);path=run/'outputs.jsonl';raw=jsonl(path)
    changed=copy.deepcopy(raw);changed[0]['input_sha256']='wrong'
    path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(AssertionError,match='Changed input'):score_run(suite,run,tmp_path/'out')
    path.write_text(''.join(json.dumps(r)+'\n' for r in raw+[raw[0]]))
    with pytest.raises(AssertionError,match='Duplicate'):score_run(suite,run,tmp_path/'out')


def test_public_dashboard_requires_shared_protocol_and_policy(tmp_path):
    row=summary();row.pop('reviewed_case_set_sha256')
    row.update(scoring_protocol='vlm-public-label-scoring-1',case_set_sha256='shared',
               scoring_policy_sha256='policy',manual_content_review=False)
    data=collect([scores(tmp_path,[row])])
    assert data['schema']=='vlm-seven-axis-2' and '公开标签' in data['grading_note']
    other=copy.deepcopy(row);other['model']='second';other['scoring_policy_sha256']='changed'
    with pytest.raises(ValueError,match='policies differ'):collect([scores(tmp_path,[row,other],'changed')])
    with pytest.raises(ValueError,match='protocols differ'):collect([scores(tmp_path,[row,summary('reviewed')],'mixed')])


def test_unit_metadata_must_bind_frozen_inputs_and_references(tmp_path,monkeypatch):
    from vlmbench import benchmark_score
    suite,run=prepared(tmp_path);cases=jsonl(suite/'cases.jsonl');refs=jsonl(suite/'references.jsonl')
    path=tmp_path/'units.json';context={'dimension':'count','reference_scale':1,
        'input_sha256':cases[0]['input_sha256'],'reference_sha256':json_hash(refs[0]),
        'media_sha256':[m['sha256'] for m in cases[0]['media']]}
    write(path,{'suite_sha256':sha(suite/'freeze.json'),'cases':{cases[0]['id']:context}})
    monkeypatch.setattr(benchmark_score,'UNIT_FILE',path)
    assert benchmark_score.score_run(suite,run,tmp_path/'correct')[0]['overall']['correct']==5
    context['reference_sha256']='wrong'
    write(path,{'suite_sha256':sha(suite/'freeze.json'),'cases':{cases[0]['id']:context}})
    with pytest.raises(AssertionError,match='Unit metadata reference'):benchmark_score.score_run(suite,run,tmp_path/'wrong')
