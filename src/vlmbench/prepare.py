"""Stratified immutable samples, official labels, and deterministic video frames."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
import av
from PIL import Image
import pyarrow.parquet as pq
from .common import json_hash,now,prompt,sha,write

def stable_rank(seed,dataset,identity):
    return hashlib.sha256(f'{seed}:{dataset}:{identity}'.encode()).hexdigest()

def sample(rows,n,seed,name,stratum):
    groups=defaultdict(list)
    for r in rows:groups[str(stratum(r))].append(r)
    for key in groups:groups[key].sort(key=lambda r:stable_rank(seed,name,r['_id']))
    result=[];keys=sorted(groups)
    while len(result)<n:
        progress=False
        for key in keys:
            if groups[key] and len(result)<n:result.append(groups[key].pop(0));progress=True
        if not progress:raise ValueError(f'Insufficient samples: {name}')
    return result

def split_mmstar(text):
    if '\nOptions:' in text:
        q,opts=re.split(r'\nOptions:\s*',text,maxsplit=1)
        matches=list(re.finditer(r'\b([A-H]):\s*',opts))
    else:
        q,opts=re.split(r'\nChoices:\s*',text,maxsplit=1)
        matches=list(re.finditer(r'(?:^|\n)\(([A-H])\)\s*',opts))
    letters=[m.group(1) for m in matches]
    assert letters==[chr(65+i) for i in range(len(letters))] and len(letters)>=2
    choices=[opts[m.end():matches[i+1].start() if i+1<len(matches) else len(opts)].strip().rstrip(',') for i,m in enumerate(matches)]
    return q,choices

def prepare(config,cache,destination):
    config=Path(config);cfg=json.loads(config.read_text());cache=Path(cache);out=Path(destination)
    assert not out.exists(),f'Refuse to overwrite suite {out}'
    source_lock=json.loads((cache/'source-lock.json').read_text())
    for f in source_lock['files']:
        p=cache/f['dataset']/f['file'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    out.mkdir(parents=True);(out/'media').mkdir();(out/'videos').mkdir()
    cases=[];references=[];provenance=[]
    def save_image(raw,extra=None):
        raw=raw['bytes'] if isinstance(raw,dict) else raw
        image=Image.open(io.BytesIO(raw));image.load()
        ext={'JPEG':'jpg','PNG':'png','WEBP':'webp','GIF':'gif'}.get(image.format)
        assert ext is not None,image.format
        h=hashlib.sha256(raw).hexdigest();p=out/'media'/f'{h}.{ext}'
        if not p.exists():p.write_bytes(raw)
        return {'path':str(p.relative_to(out)),'sha256':h,'width':image.width,'height':image.height,**(extra or {})}
    def add(name,row,question,choices,task,answers,media,category,source):
        cid=name+':'+str(row['_id'])
        c={'id':cid,'dataset':name,'category':str(category),'question':question,'choices':choices,'task':task,'media':media,'prompt':prompt(question,choices,task)}
        assert media and all(m['width']>0 and m['height']>0 for m in media)
        c['input_sha256']=json_hash({k:c[k] for k in ['question','choices','media','task','prompt']})
        cases.append(c);references.append({'id':cid,'answers':answers,'category':str(category),'task':task,'origin':'published dataset labels; not model-authored'})
        provenance.append({'id':cid,'dataset_source':source_lock['sources'][name],**source})
    for name,src in cfg['sources'].items():
        if name=='MVBench':continue
        rows=[]
        for filename in src['files']:
            data=pq.read_table(cache/name/filename).to_pylist()
            for i,r in enumerate(data):
                r['_file']=filename;r['_row']=i
                r['_id']=str(r.get('idx',r.get('id',r.get('index',i))))
                r['_subtask']=filename.split('/')[0]
                rows.append(r)
        strata={'MMStar':lambda r:r['category'],'BLINK':lambda r:r['sub_task'],'OCRBench':lambda r:r['question_type'],
                'ChartQA':lambda r:r['human_or_machine'],'POPE':lambda r:(r['category'],r['answer'])}
        selected=sample(rows,cfg['core_per_dataset'],cfg['seed'],name,strata[name])
        for r in selected:
            media=[save_image(r[k]) for k in sorted(r) if (k=='image' or re.fullmatch(r'image_\d',k)) and r[k] and (isinstance(r[k],bytes) or r[k].get('bytes'))]
            if name=='MMStar':q,ch=split_mmstar(r['question']);task='mcq';answers=[r['answer']];cat=r['category']
            elif name=='BLINK':q=r['question'];ch=r['choices'];task='mcq';answers=[r['answer'].strip('() ')];cat=r['sub_task']
            elif name=='OCRBench':q=r['question'];ch=[];task='ocr';answers=r['answer'];cat=r['question_type']
            elif name=='ChartQA':q=r['query'];ch=[];task='chart';answers=r['label'];cat='human_or_machine='+str(r['human_or_machine'])
            else:q=r['question'];ch=[];task='yesno';answers=[r['answer'].lower()];cat=r['category']
            assert isinstance(answers,list) and answers
            if task=='mcq':assert answers[0] in [chr(65+i) for i in range(len(ch))]
            add(name,r,q,ch,task,answers,media,cat,{'file':r['_file'],'row':r['_row'],'original_id':r['_id']})
        print('PREPARED',name,len(selected),flush=True)
    # MVBench v1 deliberately covers three public subtasks, not all twenty.
    src=cfg['sources']['MVBench'];rows=[]
    mapping={'action_antonym':('ssv2_video.zip','ssv2_video/'),
             'fine_grained_action':('Moments_in_Time_Raw.zip','Moments_in_Time_Raw/videos/'),
             'egocentric_navigation':('vlnqa.zip','vlnqa/')}
    for subtask in src['subtasks']:
        for i,r in enumerate(json.loads((cache/'MVBench/json'/f'{subtask}.json').read_text())):
            r.update(_id=f'{subtask}:{i}',_subtask=subtask,_row=i);rows.append(r)
    selected=sample(rows,cfg['core_per_dataset'],cfg['seed'],'MVBench',lambda r:r['_subtask'])
    for r in selected:
        archive,prefix=mapping[r['_subtask']]
        with zipfile.ZipFile(cache/'MVBench/video'/archive) as z:
            member=prefix+r['video']
            if member not in z.namelist():
                matches=[n for n in z.namelist() if n.endswith('/'+r['video'])]
                assert len(matches)==1,(member,matches)
                member=matches[0]
            raw=z.read(member)
        vh=hashlib.sha256(raw).hexdigest();vp=out/'videos'/(vh+Path(r['video']).suffix);vp.write_bytes(raw)
        with av.open(str(vp)) as container:
            frames=list(container.decode(video=0))
        assert len(frames)>=cfg['video_frames']
        indices=[min(len(frames)-1,int((i+.5)*len(frames)/cfg['video_frames'])) for i in range(cfg['video_frames'])]
        media=[]
        for i in indices:
            frame=frames[i];assert frame.pts is not None
            b=io.BytesIO();frame.to_image().convert('RGB').save(b,format='PNG')
            media.append(save_image(b.getvalue(),{'frame_index':i,'timestamp_seconds':float(frame.pts*frame.time_base),'video_sha256':vh}))
        answer=chr(65+r['candidates'].index(r['answer']))
        add('MVBench',r,r['question'],r['candidates'],'mcq',[answer],media,r['_subtask'],{'file':f'json/{r["_subtask"]}.json','row':r['_row'],
            'archive':archive,'archive_member':member,'original_video':r['video'],'video_path':str(vp.relative_to(out)),'video_sha256':vh,'decoded_frames':len(frames),'sampled_indices':indices,'sampling':'8 equal-segment center frame indices; original PTS; lossless RGB PNG; input as chronological multi-image sequence'})
        print('PREPARED_VIDEO',r['_id'],flush=True)
    assert Counter(c['dataset'] for c in cases)=={k:cfg['core_per_dataset'] for k in cfg['sources']}
    # Smoke cases are a view into the frozen core, not a second test set.
    quotas={'MMStar':6,'BLINK':4,'OCRBench':4,'ChartQA':2,'POPE':6,'MVBench':2}
    smoke=[]
    for name,n in quotas.items():smoke.extend(c['id'] for c in [x for x in cases if x['dataset']==name][:n])
    no_image=[c['id'] for c in cases if c['dataset']=='MMStar'][:6]
    for name,rows in [('cases',cases),('references',references),('provenance',provenance)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write(out/'views.json',{'smoke':smoke,'core':[c['id'] for c in cases],'no_image_diagnostic':no_image})
    shutil.copy2(config,out/'dataset-config.json');write(out/'source-lock.json',source_lock)
    files=[{'file':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'freeze.json',{'suite':cfg['suite'],'frozen_utc':now(),'seed':cfg['seed'],'case_count':len(cases),'smoke_count':len(smoke),
        'files':files,'sampling':'Stable SHA256 ordering within strata, round-robin allocation; independent of model outputs.',
        'scope':'Internal stratified regression subset, not full benchmark leaderboard scores. Official labels retained unchanged.'})
    print('FROZEN',len(cases),'core;',len(smoke),'smoke; 6 no-image diagnostics',flush=True)
