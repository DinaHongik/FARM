"""Explicit six-GPU durable launch after validated preflight."""
import argparse,json,shlex,subprocess,time
from pathlib import Path
from experiment import RUNS
from prepare import read,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-steps',type=int,default=2)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    assert (root/'DATA_READY.json').exists()
    if not args.smoke:
        gate=read(root/'PREFLIGHT_PASSED.json')
        assert gate['status']=='passed'
        for name,digest in gate['code'].items():assert sha(root/name)==digest,name
    entries=[]
    # Validate every slot before launching any job.
    for run_id,(_,_,_,gpu) in RUNS.items():
        name='farm-six-'+run_id+('-smoke' if args.smoke else '')
        assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0,name
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>100:raise RuntimeError(f'GPU {gpu} busy: {used} MiB')
        entries.append(dict(run_id=run_id,gpu=gpu,session=name))
    for entry in entries:
        logfile=root/(entry['run_id']+('.smoke.log' if args.smoke else '.log'))
        cmd=['bash',str(root/'run_job.sh'),entry['run_id'],str(entry['gpu'])]
        if args.smoke:cmd+=['--smoke','--max-steps',str(args.max_steps)]
        subprocess.run(['tmux','new-session','-d','-s',entry['session'],shlex.join(cmd)+' >> '+shlex.quote(str(logfile))+' 2>&1'],check=True)
        entry['log']=str(logfile)
    write(root/('smoke_launch.json' if args.smoke else 'launch.json'),dict(jobs=entries,detached=True,
          launched_at_unix=time.time(),dependency='DGX only; client PC may disconnect'))
    print(json.dumps(entries,indent=2))


if __name__=='__main__':main()
