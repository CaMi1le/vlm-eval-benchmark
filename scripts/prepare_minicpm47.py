#!/usr/bin/env python3
"""Create a separate native-processor view; never change downloaded tensor bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ALIASES = {'MiniCPMV4_7ImageProcessor': 'MiniCPMV4_6ImageProcessor',
           'MiniCPMV4_7VideoProcessor': 'MiniCPMV4_6VideoProcessor'}
EXPECTED = {
    'preprocessor_config.json': '7a300ccbcf796ebfcd773404a04c72c733e4880612bab1075014d41130fe6dbc',
    'processor_config.json': 'a8a03bfcf67379b092ba086750418292dbca2d9a42880555d5366a0946b4bbcb',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k != 'auto_map'}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return ALIASES.get(value, value) if isinstance(value, str) else value


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError('Runtime view must be outside the downloaded directory')
    if output.exists():
        raise FileExistsError(output)
    for name, expected in EXPECTED.items():
        if digest(source / name) != expected:
            raise ValueError(f'Unexpected processor revision: {name}')
    files = [p for p in sorted(source.iterdir()) if p.is_file()]
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Download the complete checkpoint first')
    output.mkdir(parents=True)
    records = []
    for src in files:
        target = output / src.name
        before = digest(src)
        if src.name in {*EXPECTED, 'tokenizer_config.json'}:
            target.write_text(json.dumps(normalize(json.loads(src.read_text())), indent=2) + '\n')
        else:
            shutil.copyfile(src, target)
        after = digest(target)
        if src.name.endswith('.safetensors') and before != after:
            raise ValueError('Tensor copy checksum mismatch')
        records.append({'file': src.name, 'source_sha256': before, 'runtime_sha256': after,
                        'bytes': target.stat().st_size})
    (output / 'runtime-view-manifest.json').write_text(json.dumps({
        'source_repo': 'openbmb/MiniCPM-V-4.7-35B-A3B-AWQ',
        'revision': '9832dea29707a60ba3c2001b6ad5533212d150cd',
        'aliases': ALIASES, 'files': records}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
