"""Real optimizer/epoch/miner resume test on 16 training queries; no final test."""
from pathlib import Path
import gc, json, shutil
import torch
from safetensors.torch import load_file
from prepare import read,write,sha
from experiment import configuration
from train import train_phase


def compare(a,b):
    checks=0;maximum=0.
    for f in sorted(a.rglob('*.safetensors')):
        counterpart=b/f.relative_to(a)
        x,y=load_file(str(f)),load_file(str(counterpart))
        assert x.keys()==y.keys()
        for key in x:
            torch.testing.assert_close(x[key],y[key],rtol=1e-5,atol=1e-6)
            maximum=max(maximum,float((x[key]-y[key]).abs().max()));checks+=1
        del x,y;gc.collect()
    assert checks
    x=torch.load(a.parent/'state.pt',map_location='cpu',weights_only=False)
    y=torch.load(b.parent/'state.pt',map_location='cpu',weights_only=False)
    for key in x['heads']:
        torch.testing.assert_close(x['heads'][key],y['heads'][key],rtol=1e-5,atol=1e-6)
    assert x['scheduler']==y['scheduler']
    for index,state in x['optimizer']['state'].items():
        for key,val in state.items():
            other=y['optimizer']['state'][index][key]
            if torch.is_tensor(val):torch.testing.assert_close(val,other,rtol=1e-5,atol=1e-6)
            else:assert val==other
    return dict(parameter_tensors=checks,max_absolute_difference=maximum,optimizer_and_scheduler_match=True)


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    probe=root/'boundary_probe'
    if probe.exists():raise RuntimeError('do not overwrite boundary probe')
    derived=probe/'derived';shutil.copytree(root/'derived',derived)
    for view in ('function','service_only'):
        for name in ('train','auxiliary_validation'):
            p=derived/view/(name+'.json');write(p,read(p)[:16])
    for side in ('trigger','action'):
        p=derived/'function'/f'negatives_{side}.json';write(p,read(p)[:16])
    manifest=read(derived/'manifest.json')
    manifest['artifacts']={str(p.relative_to(derived)):sha(p) for p in derived.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(derived/'manifest.json',manifest)
    cfg=configuration('f3_refresh_seed42')
    # Three one-batch epochs exercise both declared refresh boundaries.
    resumed=probe/'resumed';uninterrupted=probe/'uninterrupted';control=probe/'fixed'
    train_phase(resumed,derived,'trigger',cfg,max_steps=1)
    first_pool=sha(resumed/'trigger'/'epoch-01'/'next_negatives.json')
    train_phase(resumed,derived,'trigger',cfg)
    assert sha(resumed/'trigger'/'epoch-01'/'next_negatives.json')==first_pool
    train_phase(uninterrupted,derived,'trigger',cfg)
    result=compare(resumed/'trigger'/'epoch-03'/'trigger',uninterrupted/'trigger'/'epoch-03'/'trigger')
    fixed=configuration('f3_fixed_seed42')
    train_phase(control,derived,'trigger',fixed,max_steps=1)
    initial=compare(control/'trigger'/'epoch-01'/'trigger',resumed/'trigger'/'epoch-01'/'trigger')
    write(root/'BOUNDARY_PROBE.json',dict(status='passed',training_queries=16,epochs=3,
          full_role_catalog=True,exact_pool_reused=True,resume_equivalence=result,
          matched_first_epoch=initial,scope='isolated training-only software test'))
    print('BOUNDARY_PROBE_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
