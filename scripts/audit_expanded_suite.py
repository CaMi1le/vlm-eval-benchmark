#!/usr/bin/env python3
"""Read-only expanded-bank audit, including exact legacy inclusion; no inference."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from vlmbench.common import jsonl, now, sha, verify_suite
from vlmbench.expand import clusters, fingerprint


def audit_suite(root, base):
    root, base = Path(root), Path(base)
    cases, inherited = verify_suite(root), verify_suite(base)
    byid = {c['id']: c for c in cases}; refs = {r['id']: r for r in jsonl(root/'references.jsonl')}
    old_refs = {r['id']: r for r in jsonl(base/'references.jsonl')}; ids = set(byid); anchors = {c['id'] for c in inherited}
    assert all(byid[c['id']] == c and refs[c['id']] == old_refs[c['id']] for c in inherited)
    cfg = json.loads((root/'dataset-config.json').read_text()); n = cfg['core_per_dataset']
    assert Counter(c['dataset'] for c in cases) == {name:n for name in cfg['sources']}
    views = json.loads((root/'views.json').read_text()); base_views = json.loads((base/'views.json').read_text())
    assert len(views['core']) == len(set(views['core'])) == len(cases) and set(views['core']) == ids
    assert set(views['regression']) == anchors and len(views['regression']) == len(inherited)
    assert set(views['delta']) == ids-anchors and not (set(views['regression']) & set(views['delta']))
    assert views['smoke'] == base_views['smoke'] and views['no_image_diagnostic'] == base_views['no_image_diagnostic']
    assert set(views['smoke']) <= anchors and set(views['no_image_diagnostic']) <= set(views['smoke'])
    provenance = {r['id']:r for r in jsonl(root/'provenance.jsonl')};groups = jsonl(root/'groups.jsonl')
    assert set(provenance) == {g['id'] for g in groups} == ids
    assert groups == clusters(cases,provenance)
    seen = {}; duplicate_legacy = []; checked_media = set(); nominal_frame_cases = 0; repeat_frame_cases = 0
    for c in cases:
        assert set(c) == {'id','dataset','category','question','choices','task','media','prompt','input_sha256'}
        reference = refs[c['id']]['answers'];assert isinstance(reference,list) and reference
        if c['task']=='mcq':assert len(reference)==1 and reference[0] in [chr(65+i) for i in range(len(c['choices']))]
        if c['task']=='yesno':assert reference in [['yes'],['no']]
        fp=fingerprint(c)
        if fp in seen:
            assert c['id'] in anchors and seen[fp] in anchors,'New duplicate question'
            duplicate_legacy.append([seen[fp],c['id']])
        seen[fp]=c['id']
        for media in c['media']:
            if media['path'] in checked_media:continue
            with Image.open(root/media['path']) as image:
                image.load();assert image.size==(media['width'],media['height'])
            checked_media.add(media['path'])
        if c['dataset']=='MVBench':
            assert len(c['media'])==cfg['video_frames']
            stamps=[m['timestamp_seconds'] for m in c['media']];assert stamps==sorted(stamps)
            detail=provenance[c['id']]
            if detail.get('time_window_seconds'):
                start,end=detail['time_window_seconds'];assert all(start<=t<=end for t in stamps)
            if any('timestamp_origin' in m for m in c['media']):nominal_frame_cases+=1
            if len(set(stamps))<len(stamps):repeat_frame_cases+=1
    coverage=json.loads((root/'preparation-report.json').read_text())
    assert coverage['model_inference_performed'] is False
    assert coverage['base_freeze_sha256']==sha(base/'freeze.json')
    assert len(coverage['coverage']['BLINK']['primary_strata'])==len(cfg['sources']['BLINK']['files'])
    assert set(coverage['coverage']['MVBench']['primary_strata'])==set(cfg['sources']['MVBench']['subtasks'])
    labels={name:dict(Counter(str(refs[c['id']]['answers']) for c in cases if c['dataset']==name and c['task'] in ['mcq','yesno'])) for name in cfg['sources']}
    return {'checked_utc':now(),'suite':cfg['suite'],'freeze_sha256':sha(root/'freeze.json'),'core_count':len(cases),
            'legacy_count':len(inherited),'delta_count':len(cases)-len(inherited),'legacy_inputs_labels_and_media_identical':True,
            'all_views_valid':True,'new_exact_duplicates':0,'inherited_exact_duplicate_pairs':duplicate_legacy,
            'decoded_images_verified':len(checked_media),'source_frame_cases_with_nominal_timestamps':nominal_frame_cases,
            'video_cases_with_repeated_sampled_frames':repeat_frame_cases,'group_records_verified':len(groups),
            'label_distribution':labels,'model_inference_performed':False,'all_checks_passed':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',type=Path,required=True);parser.add_argument('--base-suite',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit_suite(args.suite,args.base_suite),ensure_ascii=False,indent=2))
