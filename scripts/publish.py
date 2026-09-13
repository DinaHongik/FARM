#!/usr/bin/env python3
"""Publish this committed release to DinaHongik/FARM without force-pushing.

Fetches the existing default-branch history before creating a normal merge.
Existing remote-only files are retained; overlapping files use this release.
No credentials are accepted as script arguments or written to the repository.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'https://github.com/DinaHongik/FARM.git', 'https://github.com/DinaHongik/FARM',
           'git@github.com:DinaHongik/FARM.git'}
ENV = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}


def git(*args, capture=False, check=True):
    return subprocess.run(['git', *args],cwd=ROOT,env=ENV,text=True,
                          stdout=subprocess.PIPE if capture else None,check=check)


def main():
    if git('status','--porcelain',capture=True).stdout.strip():
        raise RuntimeError('Commit or safely set aside local changes before publishing')
    for option in ['--get-url', '--push']:
        args=['remote','get-url']+(['--push'] if option=='--push' else [])+['--all','origin']
        urls=git(*args,capture=True).stdout.splitlines()
        if len(urls)!=1 or urls[0] not in ALLOWED:
            raise RuntimeError('origin must point only to DinaHongik/FARM; embedded credentials are not allowed')
    subprocess.run([sys.executable,'scripts/verify_release.py'],cwd=ROOT,check=True)
    refs=git('ls-remote','--symref','origin','HEAD',capture=True).stdout
    default=next((line.split()[1].removeprefix('refs/heads/') for line in refs.splitlines()
                  if line.startswith('ref: refs/heads/')), 'main')
    if refs.strip():
        git('fetch','origin',f'refs/heads/{default}:refs/remotes/origin/{default}')
        ancestor=git('merge-base','--is-ancestor',f'origin/{default}','HEAD',check=False)
        if ancestor.returncode not in {0,1}:
            raise RuntimeError('Cannot verify remote ancestry')
        if ancestor.returncode:
            git('merge','--allow-unrelated-histories','--no-edit','-X','ours',f'origin/{default}')
    # Recheck after integration, including files that existed only on GitHub.
    subprocess.run([sys.executable,'scripts/verify_release.py'],cwd=ROOT,check=True)
    env={**ENV,'PYTHONPATH':str(ROOT/'src')}
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env=env,check=True)
    git('push','-u','origin',f'HEAD:refs/heads/{default}')
    print(f'Published to https://github.com/DinaHongik/FARM/tree/{default}')


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'Publishing stopped: {exc}',file=sys.stderr)
        print('If authentication failed, sign in with Git credentials that have access to DinaHongik/FARM, then repeat this command. No force push is used.',file=sys.stderr)
        raise SystemExit(1)
