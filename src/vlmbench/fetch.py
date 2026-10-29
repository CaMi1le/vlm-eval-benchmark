"""Download only explicit pinned files; verify Hub LFS SHA256 when supplied."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from .common import sha,write,now

def fetch(config,cache):
    config=json.loads(Path(config).read_text());cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    api=HfApi();jobs=[];sources={}
    for name,src in config['sources'].items():
        info=api.dataset_info(src['repo'],revision=src['revision'],files_metadata=True)
        assert info.sha==src['revision']
        card=info.card_data.to_dict() if info.card_data else {}
        sources[name]={'repo':src['repo'],'revision':info.sha,'license_metadata':card.get('license'),'origin':src['origin'],'split':src['split']}
        meta={f.rfilename:f for f in info.siblings}
        for f in src['files']:jobs.append((name,src,f,meta[f]))
    def one(job):
        name,src,file,meta=job
        p=Path(hf_hub_download(src['repo'],filename=file,repo_type='dataset',revision=src['revision'],local_dir=cache/name))
        h=sha(p)
        assert meta.size is None or meta.size==p.stat().st_size
        expected=meta.lfs.sha256 if meta.lfs else None
        assert expected is None or h==expected
        print('FETCHED',name,file,p.stat().st_size,flush=True)
        return {'dataset':name,'file':file,'bytes':p.stat().st_size,'sha256':h,'hub_lfs_sha256':expected,'hub_blob_id':meta.blob_id}
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(one,jobs))
    write(cache/'source-lock.json',{'fetched_utc':now(),'sources':sources,'files':files})
