"""Twenty actual optimizer steps on the full training stream, isolated output."""
import argparse
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['legacy','deterministic'],required=True)
p.add_argument('--replica',required=True)
p.add_argument('--steps',type=int,default=20)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
if a.mode=='deterministic':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(a.mode=='deterministic')
source=Path('/raid/session/aicontents/farm/experiments/20260905-controlled-six-training')
sys.path.insert(0,str(source))
from train import train_phase
from experiment import configuration
from prepare import write
root=source.parent/'20260905-strong-validation'
run=root/'trajectory_probes'/f'{a.mode}_{a.replica}'
if run.exists() and not a.resume:raise RuntimeError('Refusing to overwrite an existing probe')
cfg=configuration('f3_fixed_seed42')
write(run/f'probe-{a.steps}.json',dict(mode=a.mode,replica=a.replica,steps=a.steps,resume=a.resume,scope='training-only; original group order and optimizer schedule'))
train_phase(run,source/'derived','trigger',cfg,max_steps=a.steps)
write(run/f'complete-{a.steps}.json',dict(status='complete',mode=a.mode,replica=a.replica,steps=a.steps))
