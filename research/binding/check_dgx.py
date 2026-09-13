"""Fetch a DGX status snapshot; optionally collect aggregate results (no private data)."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REMOTE='/raid/session/aicontents/farm_binding_benchmark_8192_direct'
CODE='''import json,pathlib
r=pathlib.Path('/raid/session/aicontents/farm_binding_benchmark_8192_direct')
s=json.loads((r/'STATUS.json').read_text())
p=r/'private/full_run/status.json'
if p.exists():s['progress']=json.loads(p.read_text())
elif (r/'private/smoke_run/status.json').exists():s['smoke_progress']=json.loads((r/'private/smoke_run/status.json').read_text())
s['supervisor_alive']=pathlib.Path('/proc/'+str(s['pid'])).exists()
for name in ['SMOKE_VALIDATION.json','RUN_VALIDATION.json','SMOKE_OUTPUT_VALIDATION.json','OUTPUT_VALIDATION.json']:
 p=r/'results'/name
 if p.exists():s[name]=json.loads(p.read_text())
print(json.dumps(s))
'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');args=p.parse_args()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','aicontents',
                           'python3 -c '+shlex.quote(CODE)],capture_output=True,text=True,check=True)
    status=json.loads(result.stdout);status['collected_at_unix']=time.time()
    (ROOT/'STATUS.json').write_text(json.dumps(status,sort_keys=True)+'\n')
    if args.collect:
        subprocess.run(['scp','-q','-o','BatchMode=yes','-r','aicontents:'+REMOTE+'/results/.',str(ROOT/'results')],check=True)
    print(json.dumps(status,indent=2))
