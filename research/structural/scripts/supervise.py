"""Detached supervisor: complete the frozen run, then build its aggregate report."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env-file',type=Path,required=True)
    a=p.parse_args()
    status=ROOT/'STATUS.json'
    save(status,{'state':'running','n':150,'arms':3,'condition':'oracle_endpoints',
                 'backend':'Ollama Cloud deepseek-v4-flash:0731','launched_unix':time.time(),
                 'runner_location':'user workstation','pc_must_remain_on':True})
    command=[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--env-file',str(a.env_file.resolve())]
    code=subprocess.call(command)
    if code==0:
        code=subprocess.call([sys.executable,str(ROOT/'scripts/summarize.py')])
    save(status,{'state':'complete' if code==0 else 'paused_error','n':150,'arms':3,
                 'condition':'oracle_endpoints','backend':'Ollama Cloud deepseek-v4-flash:0731',
                 'updated_unix':time.time(),'exit_code':code,
                 'results':'results/AGENTIC_150_RESULTS.json' if code==0 else None})
    raise SystemExit(code)
