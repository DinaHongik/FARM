#!/usr/bin/env python3
"""Promote validated first-50-step checkpoints, without extra training exposure."""
import json
from pathlib import Path
import subprocess
import time
from prepare import read,write,sha


def main():
    root=Path(__file__).resolve().parent
    assert not (root/'arms').exists(),'refusing to overwrite a full run'
    assert read(root/'data_validation.json')['status']=='passed'
    assert read(root/'runtime_validation.json')['status']=='passed'
    records=[]
    for arm in ('s1','f1','f2','f3','f4'):
        run=root/'smoke'/arm
        assert read(run/'reload_checks.json')['status']=='passed'
        assert not (run/'failure.json').exists()
        assert subprocess.run(['tmux','has-session','-t','farm-five-'+arm+'-smoke'],capture_output=True).returncode!=0
        phases=('shared',) if arm=='s1' else ('joint',) if arm=='f4' else ('trigger','action')
        for phase in phases:
            identity=read(run/phase/'identity.json')
            assert identity['data_manifest']==sha(root/'derived'/'manifest.json')
            for name,digest in identity['code'].items():assert sha(root/name)==digest,name
            complete=read(run/phase/'checkpoint-000050'/'complete.json')
            assert complete['step']==50
            assert read(run/phase/'smoke.done.json')['fingerprint']==complete['fingerprint']
            records.append({'arm':arm,'phase':phase,'resume_step':50,'fingerprint':complete['fingerprint']})
    (root/'smoke').rename(root/'arms')
    write(root/'promotion.json',{'promoted_at_unix':time.time(),'phases':records,
          'policy':'first 50 optimizer steps retained; full runs resume at 50 of 1506, not extra epochs'})
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
