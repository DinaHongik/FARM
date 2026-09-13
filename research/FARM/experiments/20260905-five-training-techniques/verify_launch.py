#!/usr/bin/env python3
"""Verify real optimizer progress, detached sessions and five separate GPUs."""
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import subprocess
import time
from launch import GPUS
from prepare import read,write,sha


def main():
    root=Path(__file__).resolve().parent
    assert read(root/'runtime_validation.json')['status']=='passed'
    gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    gpu_index={uuid.strip():int(index) for index,uuid in (line.split(',') for line in gpu_lines)}
    compute_lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                           '--format=csv,noheader,nounits'],text=True).splitlines()
    running={int(pid.strip()):(gpu_index[uuid.strip()],int(memory.strip()))
             for uuid,pid,memory in (line.split(',') for line in compute_lines)}
    jobs=[]
    for arm,gpu in GPUS.items():
        session='farm-five-'+arm
        subprocess.run(['tmux','has-session','-t',session],check=True,capture_output=True)
        record=read(root/'arms'/arm/'progress.json')
        assert record['status']=='training'
        assert record['step']>50,'waiting for an update after checkpoint recovery'
        assert record['gpu']==gpu
        assert math.isfinite(record['loss']) and math.isfinite(record['gradient_norm'])
        assert time.time()-record['updated_at_unix']<120
        pid=record['pid']
        assert running[pid][0]==gpu and running[pid][1]>1000
        cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert b'--arm' in cmdline and arm.encode() in cmdline
        assert b'--smoke' not in cmdline
        env=dict(entry.split(b'=',1) for entry in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in entry)
        assert env[b'CUDA_VISIBLE_DEVICES']==str(gpu).encode()
        assert b'PYTHONPATH' not in env
        assert not (root/'arms'/arm/'failure.json').exists()
        jobs.append({**record,'session':session,'gpu_process_memory_mib':running[pid][1],
                     'log':str(root/(arm+'.log'))})
    assert len({job['gpu'] for job in jobs})==5
    result={'status':'verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'verified_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),
            'client_pc_can_disconnect':True,'remote_root':str(root),'jobs':jobs,
            'source_hashes':{p.name:sha(p) for p in sorted(root.glob('*.py'))},
            'data_manifest_sha256':sha(root/'derived'/'manifest.json'),
            'automatic_development_evaluation':True}
    write(root/'LAUNCH_VERIFIED.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
