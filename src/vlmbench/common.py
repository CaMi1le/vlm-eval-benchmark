from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def json_hash(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jsonl(path): return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def append(stream,x):
    import os
    stream.write(json.dumps(x,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def verify_suite(root):
    root=Path(root);freeze=json.loads((root/'freeze.json').read_text())
    for f in freeze['files']:
        p=(root/f['file']).resolve()
        assert p.is_relative_to(root.resolve()),'Path escapes suite'
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['file']
    cases=jsonl(root/'cases.jsonl');refs=jsonl(root/'references.jsonl')
    assert len(cases)==len(refs)==freeze['case_count']
    assert len({c['id'] for c in cases})==freeze['case_count']
    assert {c['id'] for c in cases}=={r['id'] for r in refs}
    locked={f['file']:f for f in freeze['files']}
    for c in cases:
        assert c['input_sha256']==json_hash({k:c[k] for k in ['question','choices','media','task','prompt']})
        for m in c['media']:
            assert m['path'] in locked and m['sha256']==locked[m['path']]['sha256'],'Media is not locked'
    return cases

def prompt(question,choices,task):
    if task=='mcq':
        opts='\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(choices))
        return question+ ('\n'+opts if opts else '')+'\nAnswer with only the option letter.'
    if task=='yesno':return question+'\nAnswer with only yes or no.'
    return question+'\nAnswer briefly with only the requested text or value.'

def image_messages(case,root,no_image=False):
    content=[]
    if not no_image:
        for m in case['media']:
            if 'timestamp_seconds' in m:content.append({'type':'text','text':f'Frame at {m["timestamp_seconds"]:.6f} seconds:'})
            content.append({'type':'image','url':str((Path(root)/m['path']).resolve())})
    content.append({'type':'text','text':case['prompt']})
    return [{'role':'user','content':content}]
