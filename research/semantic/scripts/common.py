import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT.parent

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(canonical(value) + '\n')
    os.replace(temporary, path)

def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Frozen artifact changed: ' + path.name)
    else:
        save(path, value)
