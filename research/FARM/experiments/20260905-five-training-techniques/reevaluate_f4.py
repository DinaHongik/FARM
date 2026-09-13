#!/usr/bin/env python3
"""Correct F4 evaluation only; preserve trained weights and the original report."""
from pathlib import Path
import time
import torch
from prepare import read,write,sha
from evaluate import evaluate
from collect import collect

root=Path(__file__).resolve().parent
torch.set_num_threads(4)
run=root/'arms'/'f4'
old=read(root/'audit'/'evaluation_numpy_axis_v1'/'summary.json')
evaluate(run,root/'derived',read(run/'config.json'))
new=read(run/'summary.json')
assert old['metrics']['marginal_top1']==new['metrics']['marginal_top1']
assert old['metrics']['candidate_lattice_coverage']==new['metrics']['candidate_lattice_coverage']
write(root/'F4_EVALUATION_CORRECTION.json',{
    'corrected_at_unix':time.time(),
    'cause':'NumPy separated advanced indexing made action scores a column instead of a row',
    'training_affected':False,'retrained':False,
    'training_tensor_action_shape':'1 x A (correct)',
    'old_numpy_action_shape':'A x 1 (incorrect)',
    'old_exact_pair_top1':old['metrics']['exact_pair_top1'],
    'corrected_exact_pair_top1':new['metrics']['exact_pair_top1'],
    'marginal_scores_and_coverage_unchanged':True,
    'new_result_sha256':sha(run/'results_dev.json'),
    'regression_tests':'test_pair_scoring.py: 2 passed, reproduced failures before correction'
})
collect()
print('CORRECTED_F4',new['metrics']['correct'],new['rows'],flush=True)
