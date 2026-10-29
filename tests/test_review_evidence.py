import json
import pytest
from vlmbench.common import jsonl,sha,write
from vlmbench.score import score_run
from test_integrity import fixture_suite
from review_fixture import review_fixture

def setup(tmp_path):
    suite=tmp_path/'suite';suite.mkdir();case=fixture_suite(suite);run=tmp_path/'run';run.mkdir()
    write(run/'run.json',{'model':'test','suite_sha256':sha(suite/'freeze.json')})
    write(run/'planned.json',[{'id':case['id'],'condition':'image'}])
    (run/'outputs.jsonl').write_text(json.dumps({**case,'condition':'image','status':'ok','answer':'<|begin_of_box|>A<|end_of_box|>'})+'\n')
    manifest=review_fixture(suite,[run],tmp_path/'review')
    return suite,run,manifest

def test_review_preserves_raw_answer_and_single_content_score(tmp_path):
    suite,run,manifest=setup(tmp_path)
    result=score_run(suite,run,tmp_path/'score',manifest);row=jsonl(tmp_path/'score/scored.jsonl')[0]
    assert result['overall']['correct']==1 and row['answer']=='<|begin_of_box|>A<|end_of_box|>'
    assert row['parsed']=='A' and row['format_issue'] and not row['format_ok']
    assert 'raw_correct' not in row and 'answer_protocol' not in result

def test_reject_changed_raw_output_and_incomplete_or_tampered_review(tmp_path):
    suite,run,manifest=setup(tmp_path)
    evidence=manifest.parent/'answers.jsonl';original=evidence.read_bytes()
    evidence.write_text('')
    with pytest.raises(ValueError,match='evidence hash'):score_run(suite,run,tmp_path/'score',manifest)
    m=json.loads(manifest.read_text());m['files']['answers.jsonl']=sha(evidence);write(manifest,m)
    with pytest.raises(ValueError,match='answer count'):score_run(suite,run,tmp_path/'score',manifest)
    evidence.write_bytes(original);m['files']['answers.jsonl']=sha(evidence);write(manifest,m)
    row=jsonl(run/'outputs.jsonl')[0];row['answer']='B';(run/'outputs.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='run hash'):score_run(suite,run,tmp_path/'score',manifest)
    assert not (tmp_path/'score').exists()
