from collections import Counter
import io
import json
from pathlib import Path
import zipfile

import av
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vlmbench.common import json_hash, jsonl, prompt, sha, verify_suite, write
from vlmbench.expand import balanced_fill, clusters, fingerprint, frame_indices, prepare_expanded


def test_balancing_retains_anchors_and_rejects_bad_candidates():
    rows = [{'_file': 'f', '_row': i, '_subtask': 'a' if i < 8 else 'b'} for i in range(16)]
    accepted = []
    def accept(row):
        if row['_row'] == 9:return False
        accepted.append(row['_row']);return True
    result = balanced_fill(rows, {('f', 0), ('f', 1)}, 8, 19, 'MVBench', accept)
    assert {0, 1} <= {r['_row'] for r in result}
    assert Counter(r['_subtask'] for r in result) == {'a': 4, 'b': 4}
    assert 9 not in accepted and len(result) == 8
    again = balanced_fill(rows, {('f', 0), ('f', 1)}, 8, 19, 'MVBench', lambda r: r['_row'] != 9)
    assert result == again
    with pytest.raises(ValueError):balanced_fill(rows, {('f', 99)}, 8, 19, 'MVBench', accept)


def test_sparse_frames_explicitly_allow_repeats_and_require_media():
    assert frame_indices(8) == list(range(8))
    assert len(frame_indices(3)) == 8 and set(frame_indices(3)) == {0, 1, 2}
    with pytest.raises(ValueError):frame_indices(0)


def test_shared_images_and_video_sources_form_clusters():
    cases = [{'id': str(i), 'dataset': 'BLINK', 'media': [{'sha256': image}]} for i, image in enumerate(['a', 'a', 'b'])]
    cases += [{'id': 'v' + str(i), 'dataset': 'MVBench', 'media': [{'sha256': str(i)}]} for i in range(2)]
    provenance = {c['id']: {'video_sha256': 'same-source-video'} for c in cases}
    groups = {r['id']: r for r in clusters(cases, provenance)}
    assert groups['0']['cluster_id'] == groups['1']['cluster_id'] != groups['2']['cluster_id']
    assert groups['v0']['cluster_id'] == groups['v1']['cluster_id'] and groups['v0']['cluster_size'] == 2


def test_input_duplicates_ignore_ids_and_whitespace_but_not_images():
    case = {'id': 'one', 'task': 'mcq', 'question': 'Which  one?', 'choices': ['a', 'b'], 'media': [{'sha256': 'a'}]}
    other = {**case, 'id': 'two', 'question': 'Which one?'}
    assert fingerprint(case) == fingerprint(other)
    other['media'] = [{'sha256': 'b'}]
    assert fingerprint(case) != fingerprint(other)


def fixture(tmp_path):
    cache = tmp_path/'cache';base = tmp_path/'base';(base/'media').mkdir(parents=True);(base/'videos').mkdir()
    image = io.BytesIO();Image.new('RGB', (32, 32), 'blue').save(image, format='PNG');raw = image.getvalue()
    image_path = 'media/'+__import__('hashlib').sha256(raw).hexdigest()+'.png';(base/image_path).write_bytes(raw)
    media = [{'path': image_path, 'sha256': sha(base/image_path), 'width':32, 'height':32}]
    video = tmp_path/'clip.mp4'
    with av.open(str(video),'w') as container:
        stream = container.add_stream('mpeg4',rate=8);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
        for i in range(8):
            frame=av.VideoFrame.from_image(Image.new('RGB',(32,32),(i*20,0,255)))
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    vpath='videos/'+sha(video)+'.mp4';(base/vpath).write_bytes(video.read_bytes())
    source_rows = {
      'MMStar': [{'index':i,'question':f'MMStar question {i}\nOptions: A: a, B: b','answer':'A','category':'coarse','l2_category':'fine','image':{'bytes':raw}} for i in range(3)],
      'BLINK': [{'idx':str(i),'question':f'BLINK question {i}','sub_task':'Counting','choices':['a','b'],'answer':'(A)','image_1':{'bytes':raw}} for i in range(3)],
      'OCRBench': [{'question':f'OCR question {i}','question_type':'Regular','dataset':'test','answer':['a'],'image':{'bytes':raw}} for i in range(3)],
      'ChartQA': [{'query':f'Chart question {i}','label':['1'],'human_or_machine':i%2,'image':{'bytes':raw}} for i in range(3)],
      'POPE': [{'id':str(i),'question':f'POPE question {i}','answer':'yes' if i%2==0 else 'no','category':'random','image':{'bytes':raw}} for i in range(3)]}
    configs={};files=[];cases=[];references=[];provenance=[]
    for name, rows in source_rows.items():
        path=cache/name/'data.parquet';path.parent.mkdir(parents=True);pq.write_table(pa.Table.from_pylist(rows),path)
        configs[name]={'repo':'fixture/'+name,'revision':'fixture','files':['data.parquet']}
        if name=='MMStar':question,choices,task,answer,category='MMStar question 0',['a','b'],'mcq',['A'],'coarse'
        elif name=='BLINK':question,choices,task,answer,category='BLINK question 0',['a','b'],'mcq',['A'],'Counting'
        elif name=='OCRBench':question,choices,task,answer,category='OCR question 0',[],'ocr',['a'],'Regular'
        elif name=='ChartQA':question,choices,task,answer,category='Chart question 0',[],'chart',['1'],'human_or_machine=0'
        else:question,choices,task,answer,category='POPE question 0',[],'yesno',['yes'],'random'
        case={'id':name+':0','dataset':name,'category':category,'question':question,'choices':choices,'task':task,'media':media,'prompt':prompt(question,choices,task)}
        case['input_sha256']=json_hash({k:case[k] for k in ['question','choices','media','task','prompt']});cases.append(case)
        references.append({'id':case['id'],'answers':answer});provenance.append({'id':case['id'],'file':'data.parquet','row':0})
    mv=cache/'MVBench';(mv/'json').mkdir(parents=True);(mv/'video').mkdir()
    row={'video':'clip.mp4','question':'Video question','candidates':['a','b'],'answer':'a'}
    write(mv/'json/action_antonym.json',[row]);write(mv/'json/action_sequence.json',[{**row,'question':'Bounded video','start':.25,'end':.625}])
    for archive,prefix in [('ssv2_video.zip','ssv2_video/'),('star.zip','star/Charades_v1_480/')]:
        with zipfile.ZipFile(mv/'video'/archive,'w') as z:z.writestr(prefix+'clip.mp4',video.read_bytes())
    configs['MVBench']={'repo':'fixture/MVBench','revision':'fixture','files':['json/action_antonym.json','json/action_sequence.json','video/ssv2_video.zip','video/star.zip'],'subtasks':['action_antonym','action_sequence']}
    sequence=[{**media[0],'timestamp_seconds':i/8,'frame_index':i,'video_sha256':sha(video)} for i in range(8)]
    c={'id':'MVBench:action_antonym:0','dataset':'MVBench','category':'action_antonym','question':'Video question','choices':['a','b'],'task':'mcq','media':sequence,'prompt':prompt('Video question',['a','b'],'mcq')}
    c['input_sha256']=json_hash({k:c[k] for k in ['question','choices','media','task','prompt']});cases.append(c)
    references.append({'id':c['id'],'answers':['A']});provenance.append({'id':c['id'],'file':'json/action_antonym.json','row':0,'video_path':vpath,'video_sha256':sha(video)})
    for name,records in [('cases',cases),('references',references),('provenance',provenance)]:(base/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(base/'views.json',{'smoke':[c['id'] for c in cases],'core':[c['id'] for c in cases],'no_image_diagnostic':[cases[0]['id']]})
    write(base/'freeze.json',{'suite':'fixture-v1','case_count':6,'files':[{'file':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in base.rglob('*') if p.is_file()]})
    for name,src in configs.items():
        for file in src['files']:
            p=cache/name/file;files.append({'dataset':name,'file':file,'bytes':p.stat().st_size,'sha256':sha(p)})
    write(cache/'source-lock.json',{'sources':{name:{'repo':s['repo'],'revision':s['revision']} for name,s in configs.items()},'files':files})
    config=tmp_path/'config.json';write(config,{'suite':'fixture-v2','seed':19,'legacy_case_count':6,'core_per_dataset':2,'video_frames':8,'sources':configs})
    return config,cache,base


def test_full_preparation_keeps_legacy_and_honors_video_bounds(tmp_path):
    config,cache,base=fixture(tmp_path);old=verify_suite(base);old_freeze=sha(base/'freeze.json');out=tmp_path/'expanded'
    report=prepare_expanded(config,cache,base,out);cases=verify_suite(out);byid={c['id']:c for c in cases}
    assert len(cases)==12 and all(byid[c['id']]==c for c in old)
    assert sha(base/'freeze.json')==old_freeze and report['legacy_inputs_and_labels_identical']
    views=json.loads((out/'views.json').read_text());assert len(views['regression'])==len(views['delta'])==6 and len(views['core'])==12
    sequence=next(c for c in cases if c['category']=='action_sequence')
    assert len(sequence['media'])==8 and all(.25<=m['timestamp_seconds']<=.625 for m in sequence['media'])
    assert len({m['frame_index'] for m in sequence['media']})==4
    assert len(jsonl(out/'groups.jsonl'))==12 and report['model_inference_performed'] is False
    import importlib.util
    spec=importlib.util.spec_from_file_location('bank_audit',Path(__file__).resolve().parents[1]/'scripts/audit_expanded_suite.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audited=module.audit_suite(out,base)
    assert audited['all_checks_passed'] and audited['legacy_count']==6 and audited['new_exact_duplicates']==0
    with pytest.raises(FileExistsError):prepare_expanded(config,cache,base,out)
    (out/sequence['media'][0]['path']).write_bytes(b'tampered')
    with pytest.raises(AssertionError):verify_suite(out)


def test_repeated_inputs_with_conflicting_labels_are_logged_before_freeze(tmp_path):
    from vlmbench.prepare import stable_rank
    config,cache,base=fixture(tmp_path)
    path=cache/'OCRBench/data.parquet';rows=pq.read_table(path).to_pylist()
    bad=min([1,2],key=lambda i:stable_rank(19,'OCRBench',repr(('data.parquet',i))))
    rows[bad]['question']=rows[0]['question'];rows[bad]['answer']=['conflicting label']
    pq.write_table(pa.Table.from_pylist(rows),path)
    lock=json.loads((cache/'source-lock.json').read_text())
    record=next(r for r in lock['files'] if r['dataset']=='OCRBench')
    record.update(bytes=path.stat().st_size,sha256=sha(path));write(cache/'source-lock.json',lock)
    report=prepare_expanded(config,cache,base,tmp_path/'out')
    excluded=[r for r in report['exclusions'] if r['dataset']=='OCRBench']
    assert len(excluded)==1 and excluded[0]['conflicting_reference'] is True
    assert len(verify_suite(tmp_path/'out'))==12
