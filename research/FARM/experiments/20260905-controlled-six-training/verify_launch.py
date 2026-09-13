"""Verify real full-data optimizer progress and physical GPU placement."""
import csv,io,json,os,subprocess,time
from pathlib import Path
from experiment import RUNS
from prepare import read,write,sha


def main():
    root=Path(__file__).resolve().parent
    launch=read(root/'launch.json');gate=read(root/'PREFLIGHT_PASSED.json')
    for name,digest in gate['code'].items():assert sha(root/name)==digest,name
    device_rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True))))
    devices={row[1].strip():int(row[0]) for row in device_rows}
    process_rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True))))
    processes={int(row[1]):dict(gpu=devices[row[0].strip()],memory_mib=int(row[2])) for row in process_rows}
    checks=[]
    for entry in launch['jobs']:
        run=root/'arms'/entry['run_id'];progress=read(run/'progress.json')
        assert not (run/'failure.json').exists(),entry['run_id']
        assert progress['status']=='training' and progress['step']>=10,progress
        assert progress['total_steps']==1506
        pid=progress['pid'];os.kill(pid,0)
        assert processes[pid]['gpu']==entry['gpu']==RUNS[entry['run_id']][3]
        assert subprocess.run(['tmux','has-session','-t',entry['session']],capture_output=True).returncode==0
        identity=read(run/progress['phase']/'identity.json')
        assert identity['data_manifest']==sha(root/'derived'/'manifest.json')
        assert (run/progress['phase']/'checkpoint-000005'/'complete.json').exists()
        checks.append(dict(run_id=entry['run_id'],phase=progress['phase'],gpu=entry['gpu'],pid=pid,
               step=progress['step'],target_steps_per_role=1506,loss=progress['loss'],
               memory_mib=processes[pid]['memory_mib'],session=entry['session'],checkpoint_5_complete=True))
    for session in ('farm-six-monitor','farm-six-previous-window-correction'):
        assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0,session
    write(root/'LAUNCH_VERIFIED.json',dict(status='all_six_running',verified_at_unix=time.time(),
        training_groups=8018,jobs=checks,detached=True,client_pc_required=False,
        evaluation='automatic after training; previous-window reevaluation queued after S1 releases GPU 7'))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
