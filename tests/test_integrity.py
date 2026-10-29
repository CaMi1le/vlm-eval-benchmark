import base64
import json
import shutil
from pathlib import Path
import pytest
from vlmbench.common import json_hash, sha, verify_suite, write, image_messages
from vlmbench.supervise import repair_missing
from review_fixture import review_fixture

def fixture_suite(root):
    case={'id':'fixture:1','dataset':'fixture','category':'x','question':'Which?','choices':['cat','dog'],'media':[],'task':'mcq','prompt':'Which? A. cat B. dog'}
    case['input_sha256']=json_hash({k:case[k] for k in ['question','choices','media','task','prompt']})
    (root/'cases.jsonl').write_text(json.dumps(case)+'\n');(root/'references.jsonl').write_text(json.dumps({'id':case['id'],'answers':['A']})+'\n')
    write(root/'freeze.json',{'case_count':1,'files':[{'file':n,'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in ['cases.jsonl','references.jsonl']]})
    return case

def test_input_integrity_and_no_reference_in_prompt(tmp_path):
    case=fixture_suite(tmp_path);assert verify_suite(tmp_path)==[case]
    assert 'answers' not in json.dumps(image_messages(case,tmp_path))
    (tmp_path/'cases.jsonl').write_text('tampered')
    with pytest.raises(AssertionError):verify_suite(tmp_path)

def test_repair_killed_worker_retains_denominator(tmp_path):
    case=fixture_suite(tmp_path);output=tmp_path/'result';output.mkdir()
    (output/'outputs.jsonl').write_bytes(b'{"id":')
    repair_missing(output,[case],[(case['id'],'image'),(case['id'],'no_image')],'test')
    rows=[json.loads(l) for l in (output/'outputs.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(r['status']=='not_run' for r in rows)
    assert (output/'interrupted-fragment.bin').read_bytes()==b'{"id":'
    repair_missing(output,[case],[(case['id'],'image'),(case['id'],'no_image')],'test')
    assert len((output/'outputs.jsonl').read_text().splitlines())==2

def test_score_and_offline_gallery_keep_failures_and_escape_html(tmp_path):
    from vlmbench.score import compare
    suite=tmp_path/'suite';suite.mkdir();case=fixture_suite(suite)
    runs=[]
    for model,status,answer in [('one','ok','A'),('two','error','</script><script>alert(1)</script>')]:
        run=tmp_path/model;run.mkdir();runs.append(run)
        write(run/'run.json',{'model':model,'suite_sha256':sha(suite/'freeze.json')})
        write(run/'planned.json',[{'id':case['id'],'condition':'image'}])
        row={**case,'model':model,'condition':'image','status':status,'answer':answer}
        (run/'outputs.jsonl').write_text(json.dumps(row)+'\n')
    result=compare(suite,runs,tmp_path/'scores',review_fixture(suite,runs,tmp_path/'review'))
    assert result[0]['overall']['correct']==1 and result[1]['overall']['n']==1 and result[1]['overall']['failed']==1
    html=(tmp_path/'scores/review.html').read_text()
    assert '<script>alert(1)' not in html and '\\u003cscript' in html

def test_score_rejects_duplicate_and_wrong_input(tmp_path):
    from vlmbench.score import score_run
    suite=tmp_path/'suite';suite.mkdir();case=fixture_suite(suite);run=tmp_path/'run';run.mkdir()
    write(run/'run.json',{'model':'test','suite_sha256':sha(suite/'freeze.json')});write(run/'planned.json',[{'id':case['id'],'condition':'image'}])
    row={**case,'condition':'image','status':'ok','answer':'A'}
    (run/'outputs.jsonl').write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='duplicate'):score_run(suite,run,tmp_path/'out',tmp_path/'missing-review')
    row['input_sha256']='wrong';(run/'outputs.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Input mismatch'):score_run(suite,run,tmp_path/'out',tmp_path/'missing-review')

def test_gallery_embeds_original_images_once_and_survives_copy(tmp_path):
    from PIL import Image
    from vlmbench.report import gallery
    suite=tmp_path/'suite';suite.mkdir();case=fixture_suite(suite)
    image=suite/'original.png';Image.new('RGB',(5,3),(10,20,30)).save(image)
    original=image.read_bytes();digest=sha(image)
    case['media']=[{'path':'original.png','sha256':digest,'width':5,'height':3,'timestamp_seconds':t} for t in (1.0,2.0)]
    (suite/'cases.jsonl').write_text(json.dumps(case)+'\n')
    scores=tmp_path/'scores';(scores/'one').mkdir(parents=True)
    (scores/'one/scored.jsonl').write_text(json.dumps({'id':case['id'],'condition':'image','answer':'A'})+'\n')
    policy={'schema':'fixture','description':'<script> remains text'}
    write(scores/'review-policy.json',policy)
    gallery(suite,scores,['one'])
    portable=tmp_path/'portable.html';shutil.copyfile(scores/'review.html',portable)
    shutil.rmtree(suite);shutil.rmtree(scores)
    html=portable.read_text()
    data=json.loads(html.split('<script id="data" type="application/json">',1)[1].split('</script>',1)[0])
    assert data['review_policy']==policy
    assert len(data['cases'][0]['media'])==2 and list(data['assets'])==[digest]
    prefix,encoded=data['assets'][digest].split(',',1)
    assert prefix=='data:image/png;base64' and base64.b64decode(encoded,validate=True)==original
    assert 'href="review-policy.json"' not in html and '../suite/' not in html
    assert '<script> remains text' not in html

def test_gallery_rejects_tampered_original_image(tmp_path):
    from vlmbench.report import gallery
    suite=tmp_path/'suite';suite.mkdir();case=fixture_suite(suite)
    image=suite/'original.png';image.write_bytes(b'original')
    case['media']=[{'path':'original.png','sha256':sha(image),'width':1,'height':1}]
    (suite/'cases.jsonl').write_text(json.dumps(case)+'\n');image.write_bytes(b'tampered')
    scores=tmp_path/'scores';(scores/'one').mkdir(parents=True)
    (scores/'one/scored.jsonl').write_text(json.dumps({'id':case['id'],'condition':'image'})+'\n')
    with pytest.raises(AssertionError,match='Image hash mismatch'):gallery(suite,scores,['one'])
