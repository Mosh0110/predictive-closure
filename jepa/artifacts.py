import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(config):
    return [Path(config[key]) for key in ('output_json', 'output_checkpoint') if key in config]


def record_completion(path, config):
    result = json.loads(Path(config['output_json']).read_text())
    assert result['config'] == config
    with Path(path).open('x') as handle:
        json.dump({p.name:sha256(p) for p in output_paths(config)}, handle, indent=2)


def verify_completion(path, config):
    assert json.loads(Path(path).read_text()) == {p.name:sha256(p) for p in output_paths(config)}
