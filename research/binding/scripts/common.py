"""Small artifact helpers for an immutable, locally executed experiment."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(canonical(value) + '\n')
    os.replace(temporary, path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Immutable artifact mismatch: ' + path.name)
    else:
        save(path, value)


def save_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as handle:
        os.fchmod(handle.fileno(), 0o600)
        for value in values:
            handle.write(canonical(value) + '\n')
