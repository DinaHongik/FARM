"""Preserved diagnostic for the epoch-resume mismatch; no production training."""
from pathlib import Path
import json,random
import torch
from train import load_model,prepare_features
from prepare import read
from boundary_probe import compare
from runtime import inference_state


root=Path(__file__).resolve().parent
torch.set_num_threads(4)
a=root/'boundary_probe'/'resumed'/'trigger'/'epoch-01'
b=root/'boundary_probe'/'uninterrupted'/'trigger'/'epoch-01'
print('EPOCH_ONE',compare(a/'trigger',b/'trigger'),flush=True)
for suffix in ('next_negatives.json',):
    x,y=read(a/suffix),read(b/suffix)
    print('POOLS_EQUAL',x['pools']==y['pools'],flush=True)
base=load_model();loaded=load_model(a/'trigger')
n1,n2=list(dict(base.named_parameters())),list(dict(loaded.named_parameters()))
print('PARAMETER_ORDER',n1==n2,flush=True)
if n1!=n2:
    print('ORDER_DIFFERENCES',[(i,x,y) for i,(x,y) in enumerate(zip(n1,n2)) if x!=y][:20],flush=True)
groups=read(root/'boundary_probe'/'derived'/'function'/'train.json')
corpus=read(root/'derived'/'function'/'corpus.json')['trigger']
for role,texts in [('query',[g['query'] for g in groups]),('document',[r['text'] for r in corpus])]:
    x,y=prepare_features(base,texts,role),prepare_features(loaded,texts,role)
    for key in ('input_ids','attention_mask'):
        same=x[key].shape==y[key].shape and torch.equal(x[key],y[key])
        print('TOKENIZATION',role,key,same,tuple(x[key].shape),tuple(y[key].shape),flush=True)
print('BASE_MODEL_CONFIG',base[0].auto_model.config.to_dict(),flush=True)
print('LOADED_MODEL_CONFIG',loaded[0].auto_model.config.to_dict(),flush=True)
print('TRAINING_FLAGS',[(n,m.training) for n,m in loaded.named_modules() if not m.training],flush=True)
