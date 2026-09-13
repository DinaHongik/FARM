"""Actual FARM cached-gradient step repeated with identical weights and RNG."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['legacy', 'deterministic'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.mode == 'deterministic':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(args.mode == 'deterministic')
source = Path('/raid/session/aicontents/farm/experiments/20260905-controlled-six-training')
sys.path.insert(0, str(source))
from train import load_model, function_step
from experiment import configuration
from objectives import ServiceHead
from prepare import read

config = configuration('f3_fixed_seed42')
derived = source / 'derived'
groups = read(derived / 'function/train.json')
corpus = read(derived / 'function/corpus.json')
pools = {s: read(derived / f'function/negatives_{s}.json') for s in ['trigger', 'action']}
order = list(range(len(groups)))
random.Random(42).shuffle(order)
indices = order[:16]
examples = [groups[i] for i in indices]
torch.manual_seed(42)
random.seed(42)
model = load_model(config=config)
prototypes = torch.load(derived/'service_only/prototypes.pt', map_location='cpu', weights_only=True)
heads = torch.nn.ModuleDict({'trigger': ServiceHead(prototypes['trigger'], config['temperature'])}).cuda().train()
states = (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state())
records = []
reference = None
try:
    for repeat in range(args.repeats):
        random.setstate(states[0]); torch.set_rng_state(states[1]); torch.cuda.set_rng_state(states[2])
        model.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
        loss, metrics = function_step({'trigger': model}, heads, examples, indices, corpus, pools, 'trigger', 0, config)
        gradients = {k: p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
        gradients.update({'head.'+k: p.grad.detach().cpu().clone() for k,p in heads.named_parameters() if p.grad is not None})
        if reference is None:
            reference = gradients
            records.append({'repeat': repeat, 'loss': loss, 'gradient_tensors': len(gradients)})
        else:
            unequal = [k for k in gradients if not torch.equal(gradients[k],reference[k])]
            maximum = max(float((gradients[k]-reference[k]).abs().max()) for k in gradients)
            records.append({'repeat': repeat, 'loss': loss, 'unequal_tensors': len(unequal), 'max_absolute_gradient_difference': maximum, 'first_unequal_tensors': unequal[:5]})
            del gradients
        print(json.dumps(records[-1]), flush=True)
        gc.collect()
    result = {'status': 'passed' if all(r.get('unequal_tensors',0)==0 for r in records) else 'failed',
              'scope': 'repeat actual first batch without optimizer updates; identical model and RNG',
              'mode': args.mode, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'attention_implementation': model[0].auto_model.config._attn_implementation,
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'records': records}
except Exception as exc:
    result = {'status': 'error', 'mode': args.mode, 'error': repr(exc), 'records': records}
    raise
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
