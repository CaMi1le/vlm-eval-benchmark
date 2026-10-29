"""Hash-bound request subsets; selections never depend on answer correctness."""
import json
from pathlib import Path
from .common import sha


def planned_jobs(suite, cases, view='core', diagnostics=False, selection=None):
    views = json.loads((Path(suite) / 'views.json').read_text())
    ids = views[view]
    order = [i for i in views['smoke'] if i in ids] + [i for i in ids if i not in views['smoke']]
    jobs = [(i, 'image') for i in order]
    if diagnostics:
        jobs += [(i, 'no_image') for i in views['no_image_diagnostic'] if i in ids]
    if selection is None:
        return jobs
    document = json.loads(Path(selection).read_text())
    assert document['suite_sha256'] == sha(Path(suite) / 'freeze.json'), 'Selection suite mismatch'
    selected = document['requests']
    keys = {(r['id'], r['condition']) for r in selected}
    assert len(keys) == len(selected) and keys <= set(jobs), 'Invalid or duplicate selection'
    byid = {c['id']: c for c in cases}
    assert all(r['input_sha256'] == byid[r['id']]['input_sha256'] for r in selected), 'Selection input mismatch'
    return [job for job in jobs if job in keys]
