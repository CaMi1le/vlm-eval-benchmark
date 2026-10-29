import argparse
import json
from pathlib import Path

import pytest

from vlmbench.common import jsonl, verify_suite, write
from vlmbench.demo import demo
from vlmbench.public_workflow import run_local


def test_demo_scores_gallery_and_portable_radar(tmp_path):
    root = tmp_path / 'demo'
    result = demo(root)
    assert result['synthetic'] and result['model_calls'] == 0
    assert len(verify_suite(root / 'suite')) == 144
    scores = json.loads((root / 'report/scores/comparison.json').read_text())
    assert [s['overall']['correct'] for s in scores] == [144, 108]
    html = (root / 'report/review.html').read_text()
    assert 'SCRIPTED SYNTHETIC DEMO' in html
    assert '回归 24 题' in html and '增量 120 题' in html
    radar = root / 'report/dashboard'
    assert str(tmp_path) not in (radar / 'dashboard.html').read_text()
    assert (radar / 'source-summaries/01-comparison.json').read_bytes() == (root / 'report/scores/comparison.json').read_bytes()
    with pytest.raises(FileExistsError):
        demo(root)


def test_run_snapshot_and_request_selection(tmp_path, monkeypatch):
    from vlmbench import public_workflow
    root = tmp_path / 'demo'
    demo(root)
    model = tmp_path / 'model'
    model.mkdir()
    config = tmp_path / 'models.json'
    write(config, {'fixture': {'host': 'auto', 'path': str(model), 'gpus': [0],
                               'backend': 'fixture', 'thinking': False}})
    args = argparse.Namespace(workspace=tmp_path / 'work', suite=root / 'suite', models=config,
                              model='fixture', run_id='check', seconds=900, view='smoke', warmup_case_id=None)
    def child(command, env):
        work = Path(command[command.index('--root') + 1])
        selection = json.loads((work / 'provenance/selection.json').read_text())['requests']
        assert len(selection) == 30
        assert command[command.index('--warmup-case-id') + 1] == 'MMStar:demo-0'
        outputs = work / 'runs/check/fixture/result'
        outputs.mkdir(parents=True)
        (outputs / 'outputs.jsonl').write_text(''.join(json.dumps({**r, 'status': 'ok'}) + '\n' for r in selection))
        return 0
    monkeypatch.setattr(public_workflow.subprocess, 'call', child)
    assert run_local(args) == 0
    original = root / 'suite/media/MMStar-0.png'
    snapshot = args.workspace / 'data/suite/media/MMStar-0.png'
    before = snapshot.read_bytes()
    original.write_bytes(b'edited source')
    assert snapshot.read_bytes() == before
    with pytest.raises(FileExistsError):
        run_local(args)


def test_run_rejects_workspace_inside_suite_before_creating_it(tmp_path):
    args = argparse.Namespace(workspace=tmp_path/'suite/nested', suite=tmp_path/'suite')
    with pytest.raises(ValueError, match='outside'):
        run_local(args)
    assert not args.workspace.exists()


def test_processor_normalization_preserves_tensors(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('prepare_minicpm47', Path(__file__).parents[1]/'scripts/prepare_minicpm47.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path/'source'
    source.mkdir()
    write(source/'processor_config.json', {'image_processor_type': 'MiniCPMV4_7ImageProcessor', 'auto_map': {'x': 'y'}})
    monkeypatch.setattr(module, 'EXPECTED', {'processor_config.json': module.digest(source/'processor_config.json')})
    (source/'part.safetensors').write_bytes(b'synthetic tensor bytes')
    output = tmp_path/'runtime'
    module.prepare(source, output)
    assert (output/'part.safetensors').read_bytes() == (source/'part.safetensors').read_bytes()
    assert json.loads((output/'processor_config.json').read_text()) == {'image_processor_type': 'MiniCPMV4_6ImageProcessor'}
    assert 'auto_map' in json.loads((source/'processor_config.json').read_text())
    assert len(json.loads((output/'runtime-view-manifest.json').read_text())['files']) == 2
    with pytest.raises(FileExistsError):
        module.prepare(source, output)


def test_report_rejects_escaping_model_identifier(tmp_path):
    from vlmbench.benchmark_score import compare
    root = tmp_path/'demo'
    demo(root)
    run = root/'runs/demo-scripted-perfect'
    meta = json.loads((run/'run.json').read_text())
    meta['model'] = '../../outside'
    write(run/'run.json', meta)
    with pytest.raises(ValueError, match='escapes'):
        compare(root/'suite', [run], tmp_path/'report')
    assert not (tmp_path.parent/'outside').exists()
