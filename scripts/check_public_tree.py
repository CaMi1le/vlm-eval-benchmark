#!/usr/bin/env python3
"""Check tracked release contents; print filenames only, never matching secret text."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Obvious access credentials and runtime/model files must not enter a release.
PATTERNS = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
            rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{40,}',
            rb'hf_[A-Za-z0-9]{30,}', rb'sk-[A-Za-z0-9_-]{30,}']
FORBIDDEN_SUFFIXES = {'.safetensors', '.gguf', '.onnx', '.engine', '.pem', '.key', '.zip', '.tar', '.gz'}


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    for name in filter(None, names):
        path = ROOT / name
        if path.is_symlink() or path.suffix in FORBIDDEN_SUFFIXES or name.endswith('.local.json') or path.name.startswith('.env'):
            errors.append(name)
            continue
        data = path.read_bytes()
        if len(data) > 2_000_000 or any(re.search(pattern, data) for pattern in PATTERNS):
            errors.append(name)
    if errors:
        raise SystemExit('Release content check failed: ' + ', '.join(errors))
    print(f'Public tree checked: {len([n for n in names if n])} files')


if __name__ == '__main__':
    main()
