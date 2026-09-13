#!/usr/bin/env python3
"""Launch five durable tmux jobs with explicit, separate GPU placement."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from prepare import write

GPUS={'s1':1,'f1':0,'f2':5,'f3':6,'f4':7}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-steps',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    assert (root/'derived'/'manifest.json').is_file()
    entries=[]
    for arm,gpu in GPUS.items():
        name='farm-five-'+arm+('-smoke' if args.smoke else '')
        exists=subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0
        if exists:
            raise RuntimeError('session already exists: '+name)
        log=root/(arm+('.smoke.log' if args.smoke else '.log'))
        cmd=['bash',str(root/'run_job.sh'),arm,str(gpu)]
        if args.smoke:cmd+=['--smoke','--max-steps',str(args.max_steps)]
        command=shlex.join(cmd)+' >> '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',name,command],check=True)
        entries.append({'arm':arm,'gpu':gpu,'session':name,'log':str(log)})
    write(root/('smoke_launch.json' if args.smoke else 'launch.json'),
          {'launched_at_unix':time.time(),'jobs':entries,'detached':True,
           'dependency':'DGX power/network only; no client PC dependency'})
    print(json.dumps(entries,indent=2))


if __name__=='__main__':main()
