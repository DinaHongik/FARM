#!/usr/bin/env python3
"""Verify release integrity without model calls or private data access."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from farm_release.cli import load_examples

PATTERNS = [r'\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{24,}',
            r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
            r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
            r'AKIA[0-9A-Z]{16}']
# The recovered privacy regression tests intentionally contain one alphabetic
# dummy token. Only this exact fixture hash is exempt, never real credentials.
DUMMY_TOKEN_SHA256 = '82be8a4d9cdebab78235e6d0618fdea34065fcb6f498073283ff4ec7376e551a'
# Public examples plus the reviewed documentation/source files already on GitHub.
# Additional dataset files still require an explicit release decision.
PUBLIC_DATA_FILES = {
    'data/ifttt_examples.json', 'data/manifest.json',
    'data/README.md', 'data/category_analyze.py', 'data/split_data.py',
    'data/test/README.md',
}
MANUSCRIPT_EXTENSIONS = {
    '.tex', '.latex', '.bib', '.cls', '.sty', '.bst', '.aux', '.bbl', '.blg',
    '.pdf', '.doc', '.docx', '.odt',
}


def is_manuscript_artifact(path):
    return (path.suffix.lower() in MANUSCRIPT_EXTENSIONS
            or any(part.lower() in {'paper', 'manuscript', 'appendix'} for part in path.parts))


def main():
    errors=[]
    if (ROOT/'.git').is_dir():
        tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.split('\0')
        for name in filter(None,tracked):
            path=Path(name)
            if is_manuscript_artifact(path):
                errors.append('Manuscript artifact is not allowed in this release: '+name)
            if (any(part in {'private','outputs','checkpoints','models'} for part in path.parts)
                    or path.name.startswith('.env') and path.name!='.env.example'
                    or path.suffix in {'.pem','.key','.p12','.pfx','.pt','.pth','.safetensors'}
                    or name.startswith('data/') and name not in PUBLIC_DATA_FILES):
                errors.append('Private or unexpected tracked artifact: '+name)
    examples=load_examples(ROOT/'data/ifttt_examples.json')
    if len(examples)!=100:errors.append('Expected exactly 100 examples')
    data_manifest=json.loads((ROOT/'data/manifest.json').read_text())
    if hashlib.sha256((ROOT/'data/ifttt_examples.json').read_bytes()).hexdigest()!=data_manifest['examples_sha256']:
        errors.append('Example checksum mismatch')
    manifest=json.loads((ROOT/'docs/SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path=ROOT/entry['path']
        expected=entry.get('release_sha256',entry['source_sha256'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            errors.append('Source checksum mismatch: '+entry['path'])
    skip={'.git','.venv','__pycache__','outputs','.pytest_cache','build','dist'}
    scanned=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in skip or p.endswith('.egg-info') for p in path.relative_to(ROOT).parts):continue
        if path.name.startswith('.env') and path.name!='.env.example':
            # Local ignored credentials are allowed, but cannot be staged.
            continue
        if is_manuscript_artifact(path.relative_to(ROOT)):
            errors.append('Manuscript artifact is not allowed in this release: '+str(path.relative_to(ROOT)))
            continue
        if path.suffix in {'.pem','.key','.p12','.pfx','.safetensors','.pt','.pth'}:
            errors.append('Forbidden release file: '+str(path.relative_to(ROOT)))
        scanned+=1
        text=path.read_text(errors='replace')
        for line,value in enumerate(text.splitlines(),1):
            matches=[m.group(0) for pattern in PATTERNS for m in re.finditer(pattern,value)]
            if any(hashlib.sha256(m.encode()).hexdigest()!=DUMMY_TOKEN_SHA256 for m in matches):
                errors.append(f'Possible credential: {path.relative_to(ROOT)}:{line}')
    if errors:
        print('\n'.join(errors),file=sys.stderr);return 1
    print(f'PASS: 100 examples, no manuscript artifacts, {len(manifest["files"])} recovered source checksums, {scanned} text files scanned')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
