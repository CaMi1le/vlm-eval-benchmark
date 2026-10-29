"""Explicit user pauses apply only to thinking, never to nonthinking results."""
import json
from pathlib import Path


def paused_thinking_models(root):
    root = Path(root)
    for path in (root / 'provenance/thinking-paused-models.json',
                 root / 'configs/paused-thinking-models.json'):
        if path.exists():
            policy = json.loads(path.read_text())
            assert policy['schema'] == 'vlm-thinking-pause-1'
            assert policy['phase'] == 'thinking'
            assert isinstance(policy['models'], list) and all(isinstance(m, str) for m in policy['models'])
            return set(policy['models'])
    return set()


def enforce_thinking_pause(root, model, config):
    if config.get('thinking') and model in paused_thinking_models(root):
        raise RuntimeError(f'Thinking evaluation paused by user: {model}; explicit resume required')
