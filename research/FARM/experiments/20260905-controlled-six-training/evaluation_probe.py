"""Exercise final evaluation paths using only the smoke checkpoints/training rows."""
import argparse
from pathlib import Path
import torch
from evaluate import evaluate
from experiment import RUNS,configuration


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',choices=tuple(RUNS),action='append',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    for run_id in args.run_id:
        evaluate(root/'smoke'/run_id,root/'derived',configuration(run_id),validation_only=True)
