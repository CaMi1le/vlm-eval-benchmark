"""Prepare a larger frozen bank, inheriting v1 verbatim; CPU and data only."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import shutil
import zipfile

import av
from PIL import Image
import pyarrow.parquet as pq

from .common import json_hash, jsonl, now, prompt, sha, verify_suite, write
from .prepare import split_mmstar, stable_rank

# Official MVBench notebook source paths and bound flags, without licensed NTU.
VIDEO = {
    'action_sequence': ('star.zip', 'star/Charades_v1_480/', True),
    'action_prediction': ('star.zip', 'star/Charades_v1_480/', True),
    'action_antonym': ('ssv2_video.zip', 'ssv2_video/', False),
    'fine_grained_action': ('Moments_in_Time_Raw.zip', 'Moments_in_Time_Raw/videos/', False),
    'unexpected_action': ('FunQA_test.zip', 'FunQA_test/test/', False),
    'object_existence': ('clevrer.zip', 'clevrer/video_validation/', False),
    'object_interaction': ('star.zip', 'star/Charades_v1_480/', True),
    'object_shuffle': ('perception.zip', 'perception/videos/', False),
    'moving_direction': ('clevrer.zip', 'clevrer/video_validation/', False),
    'action_localization': ('sta.zip', 'sta/sta_video/', True),
    'scene_transition': ('scene_qa.zip', 'scene_qa/video/', False),
    'action_count': ('perception.zip', 'perception/videos/', False),
    'moving_count': ('clevrer.zip', 'clevrer/video_validation/', False),
    'moving_attribute': ('clevrer.zip', 'clevrer/video_validation/', False),
    'state_change': ('perception.zip', 'perception/videos/', False),
    'character_order': ('perception.zip', 'perception/videos/', False),
    'egocentric_navigation': ('vlnqa.zip', 'vlnqa/', False),
    'episodic_reasoning': ('tvqa.zip', 'tvqa/frames_fps3_hq/', True),
    'counterfactual_inference': ('clevrer.zip', 'clevrer/video_validation/', False),
}


def source_key(row):
    return row['_file'], row['_row']


def strata(name, row):
    if name == 'MMStar':return str(row['category']), str(row['l2_category'])
    if name == 'BLINK':return row['_file'].split('/')[0], str(row['sub_task'])
    if name == 'OCRBench':return str(row['question_type']), str(row['dataset'])
    if name == 'ChartQA':return str(row['human_or_machine']), ''
    if name == 'POPE':return str(row['category']) + '/' + str(row['answer']).lower(), ''
    return row['_subtask'], ''


def balanced_fill(rows, required, target, seed, name, accept):
    """Balance primary then secondary strata, count inherited anchors first."""
    bykey = {source_key(row): row for row in rows}
    if len(bykey) != len(rows) or not set(required) <= set(bykey):
        raise ValueError('Duplicate source keys or inherited rows missing: ' + name)
    if len(required) > target:raise ValueError('Target cannot exclude inherited cases')
    pools = defaultdict(lambda: defaultdict(list)); counts = Counter(); fine = Counter()
    for row in rows:
        a, b = strata(name, row)
        if source_key(row) in required:counts[a] += 1; fine[a, b] += 1
        else:pools[a][b].append(row)
    for group in pools.values():
        for queue in group.values():queue.sort(key=lambda r: stable_rank(seed, name, repr(source_key(r))), reverse=True)
    selected = [bykey[key] for key in sorted(required)]
    while len(selected) < target:
        live = [a for a in pools if any(pools[a].values())]
        if not live:raise ValueError('Insufficient valid candidates for ' + name)
        a = min(live, key=lambda key: (counts[key], key))
        b = min((b for b in pools[a] if pools[a][b]), key=lambda key: (fine[a, key], key))
        row = pools[a][b].pop()
        if accept(row):
            selected.append(row); counts[a] += 1; fine[a, b] += 1
    return selected


def frame_indices(count, requested=8):
    if count < 1:raise ValueError('No frames within the published time window')
    return [min(count - 1, int((i + .5) * count / requested)) for i in range(requested)]


def fingerprint(case):
    return json_hash({'task': case['task'], 'question': ' '.join(case['question'].split()),
                      'choices': [' '.join(str(c).split()) for c in case['choices']],
                      'images': [m['sha256'] for m in case['media']],
                      'times': [m.get('timestamp_seconds') for m in case['media']]})


def clusters(cases, provenance):
    """Conservative shared-image / same-source-video components for later CIs."""
    parent = {c['id']: c['id'] for c in cases}; owner = {}; resources = {}
    def find(key):
        while parent[key] != key:parent[key] = parent[parent[key]]; key = parent[key]
        return key
    for c in cases:
        p = provenance[c['id']]
        if c['dataset'] == 'MVBench':
            keys = ['video:' + p['video_sha256']] if p.get('video_sha256') else ['frames:' + p['archive'] + ':' + p['original_video']]
        else:keys = ['image:' + m['sha256'] for m in c['media']]
        resources[c['id']] = keys
        for key in keys:
            if key in owner:parent[find(c['id'])] = find(owner[key])
            else:owner[key] = c['id']
    members = defaultdict(list)
    for c in cases:members[find(c['id'])].append(c['id'])
    labels = {key: json_hash(sorted(ids)) for key, ids in members.items()}
    return [{'id': c['id'], 'dataset': c['dataset'], 'cluster_id': labels[find(c['id'])],
             'cluster_size': len(members[find(c['id'])]), 'resources': resources[c['id']]} for c in cases]


def prepare_expanded(config, cache, base_suite, destination):
    cfg = json.loads(Path(config).read_text()); cache = Path(cache); base = Path(base_suite); out = Path(destination)
    if out.exists():raise FileExistsError('Refuse to overwrite suite: ' + str(out))
    inherited = verify_suite(base); expected = cfg.get('legacy_case_count', 144)
    assert len(inherited) == expected
    lock = json.loads((cache / 'source-lock.json').read_text())
    locked = {(f['dataset'], f['file']): f for f in lock['files']}
    for name, src in cfg['sources'].items():
        assert src['repo'] == lock['sources'][name]['repo'] and src['revision'] == lock['sources'][name]['revision']
        for filename in src['files']:
            item = locked[name, filename]; path = cache / name / filename
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
    print('SOURCES_VERIFIED', len(locked), flush=True)
    out.mkdir(parents=True); (out / 'media').mkdir(); (out / 'videos').mkdir()
    cases = list(inherited); refs = jsonl(base / 'references.jsonl'); provenance = jsonl(base / 'provenance.jsonl')
    base_refs = {r['id']: r for r in refs}; base_prov = {p['id']: p for p in provenance}
    old_freeze = sha(base / 'freeze.json'); seen = {}; exclusions = []; inventory = {}; coverage = {}
    for case in inherited:
        seen[fingerprint(case)] = case['id']
        for m in case['media']:
            target = out / m['path']
            if not target.exists():shutil.copyfile(base / m['path'], target)
        p = base_prov[case['id']]
        if p.get('video_path') and not (out / p['video_path']).exists():shutil.copyfile(base / p['video_path'], out / p['video_path'])
        p['inherited_from'] = {'suite': json.loads((base / 'freeze.json').read_text())['suite'], 'freeze_sha256': old_freeze}

    def save_image(raw, extra=None):
        raw = raw['bytes'] if isinstance(raw, dict) else raw
        with Image.open(io.BytesIO(raw)) as image:
            image.load(); width, height = image.size; ext = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'GIF': 'gif'}[image.format]
        h = hashlib.sha256(raw).hexdigest(); path = out / 'media' / (h + '.' + ext)
        if not path.exists():path.write_bytes(raw)
        return {'path': str(path.relative_to(out)), 'sha256': h, 'width': width, 'height': height, **(extra or {})}

    with ExitStack() as stack:
        zips = {}; row_tables = {}
        def archive(name):
            if name not in zips:zips[name] = stack.enter_context(zipfile.ZipFile(cache / 'MVBench/video' / name))
            return zips[name]
        def member(zipname, prefix, video):
            z = archive(zipname); path = prefix + video
            if path in z.namelist():return zipname, path
            matches = [n for n in z.namelist() if n.endswith('/' + video)]
            if len(matches) == 1:return zipname, matches[0]
            repair = archive('data0613.zip'); matches = [n for n in repair.namelist() if n.endswith('/' + video)]
            if len(matches) != 1:raise ValueError('Missing/ambiguous video: ' + zipname + '/' + video)
            return 'data0613.zip', matches[0]
        def video_media(row):
            zipname, prefix, bounded = VIDEO[row['_subtask']]
            bound = (float(row['start']), float(row['end'])) if bounded else None
            detail = {'file': row['_file'], 'row': row['_row'], 'archive': zipname, 'original_video': row['video'], 'time_window_seconds': bound}
            if row['_subtask'] == 'episodic_reasoning':
                z = archive(zipname); names = [n for n in z.namelist() if n.startswith(prefix + row['video'].rstrip('/') + '/') and n.endswith('.jpg')]
                if not names:names = [n for n in z.namelist() if ('/' + row['video'].strip('/') + '/') in n and n.endswith('.jpg')]
                ordered = sorted((int(Path(n).stem), n) for n in names)
                eligible = [(i, n) for i, n in ordered if not bound or bound[0] <= (i-1)/3 <= bound[1]]
                selected = [eligible[i] for i in frame_indices(len(eligible), cfg['video_frames'])]
                media = [save_image(z.read(n), {'source_frame_index': i, 'timestamp_seconds': (i-1)/3,
                          'timestamp_origin': 'nominal 3 FPS source frames; original video PTS unavailable'}) for i, n in selected]
                detail.update(frame_members=[n for i, n in selected], nominal_fps=3, unique_sampled_frames=len(set(i for i, n in selected)), sampling='8 equal-segment center source-frame indices inside published bounds; original JPEG bytes; nominal timestamps')
                return media, detail
            zipname, path = member(zipname, prefix, row['video']); raw = archive(zipname).read(path)
            vh = hashlib.sha256(raw).hexdigest(); vp = out / 'videos' / (vh + Path(row['video']).suffix)
            if not vp.exists():vp.write_bytes(raw)
            stamps = []
            with av.open(str(vp)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if frame.pts is None:raise ValueError('Source frame has no PTS')
                    t = float(frame.pts * frame.time_base)
                    if not bound or bound[0] <= t <= bound[1]:stamps.append((i, t))
            selected = [stamps[i] for i in frame_indices(len(stamps), cfg['video_frames'])]; wanted = {i for i, t in selected}; saved = {}
            with av.open(str(vp)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in wanted:
                        buf = io.BytesIO(); frame.to_image().convert('RGB').save(buf, format='PNG')
                        saved[i] = save_image(buf.getvalue(), {'frame_index': i, 'timestamp_seconds': float(frame.pts * frame.time_base), 'video_sha256': vh})
                    if len(saved) == len(wanted):break
            detail.update(archive=zipname, archive_member=path, video_path=str(vp.relative_to(out)), video_sha256=vh,
                          eligible_frames=len(stamps), sampled_indices=[i for i, t in selected], unique_sampled_frames=len(wanted),
                          sampling='8 equal-segment center frame indices inside published bounds; original PTS; lossless RGB PNG')
            return [saved[i] for i, t in selected], detail

        for name, src in cfg['sources'].items():
            rows = []; row_tables.clear()
            for filename in src['files']:
                if name == 'MVBench':
                    if not filename.startswith('json/'):continue
                    data = json.loads((cache / name / filename).read_text())
                else:
                    parquet = pq.ParquetFile(cache / name / filename)
                    columns = [c for c in parquet.schema_arrow.names if not re.fullmatch(r'image(?:_\d+)?', c)]
                    data = parquet.read(columns=columns).to_pylist()
                for i, row in enumerate(data):
                    row.update(_file=filename, _row=i, _subtask=Path(filename).stem)
                    row['_id'] = str(row.get('idx', row.get('id', row.get('index', i))))
                    rows.append(row)
            required = {(base_prov[c['id']]['file'], base_prov[c['id']]['row']) for c in inherited if c['dataset'] == name}
            inventory[name] = {'rows': len(rows), 'primary_strata': dict(Counter(strata(name, r)[0] for r in rows)),
                               'secondary_strata': dict(Counter('/'.join(strata(name, r)) for r in rows))}
            def accept(row):
                try:
                    detail = {'file': row['_file'], 'row': row['_row'], 'original_id': row['_id'], 'strata': list(strata(name, row)),
                              'dataset_source': lock['sources'][name]}
                    if name == 'MVBench':
                        media, video_detail = video_media(row); detail.update(video_detail)
                        q, ch, task, answers, cat = row['question'], row['candidates'], 'mcq', [chr(65 + row['candidates'].index(row['answer']))], row['_subtask']
                        cid = 'MVBench:' + row['_subtask'] + ':' + str(row['_row'])
                    else:
                        if row['_file'] not in row_tables:row_tables[row['_file']] = pq.read_table(cache / name / row['_file']).to_pylist()
                        full = row_tables[row['_file']][row['_row']]
                        media = [save_image(full[k]) for k in sorted(full) if re.fullmatch(r'image(?:_\d+)?', k) and full[k] and (isinstance(full[k], bytes) or full[k].get('bytes'))]
                        if name == 'MMStar':q, ch = split_mmstar(row['question']); task, answers, cat = 'mcq', [row['answer']], row['category']
                        elif name == 'BLINK':q, ch, task, answers, cat = row['question'], row['choices'], 'mcq', [row['answer'].strip('() ')], row['sub_task']
                        elif name == 'OCRBench':q, ch, task, answers, cat = row['question'], [], 'ocr', row['answer'], row['question_type']
                        elif name == 'ChartQA':q, ch, task, answers, cat = row['query'], [], 'chart', row['label'], 'human_or_machine=' + str(row['human_or_machine'])
                        else:q, ch, task, answers, cat = row['question'], [], 'yesno', [row['answer'].lower()], row['category']
                        cid = name + ':' + (str(cat) + ':' if name == 'POPE' else '') + row['_id']
                    if not media or not answers:raise ValueError('Empty image or reference')
                    if task == 'mcq' and answers[0] not in [chr(65 + i) for i in range(len(ch))]:raise ValueError('Invalid option label')
                    case = {'id': cid, 'dataset': name, 'category': str(cat), 'question': q, 'choices': ch, 'task': task,
                            'media': media, 'prompt': prompt(q, ch, task)}
                    fp = fingerprint(case)
                    if fp in seen:
                        kept = next(r for r in refs if r['id'] == seen[fp])
                        exclusions.append({'dataset': name, 'file': row['_file'], 'row': row['_row'],
                                           'reason': 'duplicate question and visual input', 'duplicates': seen[fp],
                                           'conflicting_reference': kept['answers'] != answers,
                                           'candidate_answers': answers, 'kept_answers': kept['answers']}); return False
                    case['input_sha256'] = json_hash({k: case[k] for k in ['question', 'choices', 'media', 'task', 'prompt']})
                    seen[fp] = cid; cases.append(case)
                    refs.append({'id': cid, 'answers': answers, 'category': str(cat), 'task': task, 'origin': 'published dataset labels; not model-authored'})
                    provenance.append({'id': cid, **detail})
                    return True
                except (ValueError, KeyError, OSError, av.error.FFmpegError) as error:
                    exclusions.append({'dataset': name, 'file': row['_file'], 'row': row['_row'], 'reason': type(error).__name__ + ': ' + str(error)})
                    return False
            selected = balanced_fill(rows, required, cfg['core_per_dataset'], cfg['seed'], name, accept)
            coverage[name] = {'n': len(selected), 'inherited': len(required), 'primary_strata': dict(Counter(strata(name, r)[0] for r in selected)),
                              'secondary_strata': dict(Counter('/'.join(strata(name, r)) for r in selected))}
            print('EXPANDED', name, len(selected), 'strata', len(coverage[name]['primary_strata']), flush=True)
    assert len(cases) == len({c['id'] for c in cases}) == cfg['core_per_dataset'] * len(cfg['sources'])
    byid = {c['id']: c for c in cases}; references = {r['id']: r for r in refs}
    assert all(byid[c['id']] == c and references[c['id']] == base_refs[c['id']] for c in inherited)
    prov = {p['id']: p for p in provenance}; group_rows = clusters(cases, prov)
    old_views = json.loads((base / 'views.json').read_text()); legacy_ids = [c['id'] for c in inherited]
    order = sorted(byid, key=lambda identity: stable_rank(cfg['seed'], cfg['suite'], identity))
    views = {'smoke': old_views['smoke'], 'regression': legacy_ids, 'core': order,
             'delta': [i for i in order if i not in set(legacy_ids)], 'no_image_diagnostic': old_views['no_image_diagnostic']}
    report = {'suite': cfg['suite'], 'prepared_utc': now(), 'core_count': len(cases), 'inherited_count': len(inherited),
              'delta_count': len(cases) - len(inherited), 'base_freeze_sha256': old_freeze, 'legacy_inputs_and_labels_identical': True,
              'coverage': coverage, 'candidate_inventory': inventory, 'exclusions': exclusions,
              'excluded_subtasks': cfg['sources']['MVBench'].get('excluded_subtasks', {}),
              'media_groups': {name: {'distinct_images': len({m['sha256'] for c in cases if c['dataset'] == name for m in c['media']}),
                                      'clusters': len({g['cluster_id'] for g in group_rows if g['dataset'] == name}),
                                      'largest_cluster': max(g['cluster_size'] for g in group_rows if g['dataset'] == name)} for name in cfg['sources']},
              'scope': 'Larger frozen internal subsets; not full official leaderboard scores. Shared-media clusters retained for later paired/group-aware uncertainty analysis.',
              'model_inference_performed': False, 'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['av', 'pillow', 'pyarrow']}}}
    # Remove candidate media rejected during deterministic, model-independent validation.
    keep = {m['path'] for c in cases for m in c['media']} | {p['video_path'] for p in provenance if p.get('video_path')}
    for folder in ['media', 'videos']:
        for path in (out / folder).iterdir():
            if str(path.relative_to(out)) not in keep:path.unlink()
    for name, records in [('cases', cases), ('references', refs), ('provenance', provenance), ('groups', group_rows)]:
        (out / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    write(out / 'views.json', views); write(out / 'preparation-report.json', report)
    shutil.copyfile(config, out / 'dataset-config.json'); write(out / 'source-lock.json', lock)
    files = [{'file': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    write(out / 'freeze.json', {'suite': cfg['suite'], 'frozen_utc': now(), 'seed': cfg['seed'], 'case_count': len(cases), 'smoke_count': len(views['smoke']),
          'inherited_suite_sha256': old_freeze, 'inherited_case_count': len(inherited), 'files': files,
          'sampling': 'All v1 anchors retained verbatim; least-filled primary/secondary strata, stable SHA256 candidate ordering; invalid media and exact repeated inputs excluded before model runs.',
          'scope': report['scope']})
    verify_suite(out); print('FROZEN', cfg['suite'], len(cases), flush=True)
    return report
