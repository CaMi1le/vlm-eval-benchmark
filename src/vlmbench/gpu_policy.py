"""Validate physical GPU assignments before waiting for or launching a worker."""
import json
from pathlib import Path
from .common import sha


def enforce_gpu_policy(root, config):
    path = Path(root) / 'configs/gpu-resource-policy.json'
    if not path.exists():
        return None  # Historical frozen campaigns predate explicit host policies.
    policy = json.loads(path.read_text())
    assert policy['schema'] == 'vlm-gpu-resource-policy-1'
    host = config['host']
    rule = policy['hosts'].get(host)
    if rule is None:
        return None
    allowed, reserved = set(rule['allowed_gpus']), set(rule['reserved_gpus'])
    assert allowed and allowed.isdisjoint(reserved), 'Invalid GPU resource policy'
    requested = config['gpus']
    if not requested or any(type(gpu) is not int or gpu not in allowed for gpu in requested):
        raise ValueError(f'GPU policy rejects {host} assignment {requested}; allowed physical GPUs: {sorted(allowed)}')
    return {'file': str(path.relative_to(root)), 'sha256': sha(path), 'host': host,
            'allowed_gpus': sorted(allowed), 'assigned_gpus': requested}
