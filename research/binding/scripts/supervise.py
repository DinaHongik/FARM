"""Detached DGX job: checks, gated cloud smoke, full experiment, replay, scores."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from common import ROOT,save,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--env-file',type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    lock=(ROOT/'private/supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    def status(phase,state='running',**extra):
        save(ROOT/'STATUS.json',{'state':state,'phase':phase,'pid':os.getpid(),'host':socket.gethostname(),
             'n':150,'workflows':3,'expected_outputs':450,'started_unix':started,'updated_unix':time.time(),
             'runner_location':'DGX','pc_must_remain_on':False,'backend':'Ollama Cloud deepseek-v4-flash:0731',
             'max_generated_tokens_per_call':8192,'run_id':'binding_8192_direct','thinking':False,**extra})
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase,state,flush=True)
    steps=[
      ('unit_validation',[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v']),
      ('smoke_generation',[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--inputs',str(ROOT/'private/smoke_inputs.jsonl'),
                           '--output',str(ROOT/'private/smoke_run'),'--env-file',str(args.env_file),'--smoke']),
      ('smoke_replay',[sys.executable,str(ROOT/'scripts/validate_run.py'),'--directory',str(ROOT/'private/smoke_run'),'--smoke']),
      ('smoke_scoring',[sys.executable,str(ROOT/'scripts/summarize.py'),'--directory',str(ROOT/'private/smoke_run'),'--smoke']),
      ('full_generation',[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--inputs',str(ROOT/'private/inputs.jsonl'),
                          '--output',str(ROOT/'private/full_run'),'--env-file',str(args.env_file)]),
      ('full_replay',[sys.executable,str(ROOT/'scripts/validate_run.py')]),
      ('scoring',[sys.executable,str(ROOT/'scripts/summarize.py')]),
    ]
    for phase,command in steps:
        status(phase)
        code=subprocess.call(command,cwd=ROOT)
        if code:
            status(phase,'paused_error',exit_code=code)
            return code
    status('complete','complete',results='results/BINDING_150_RESULTS.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
