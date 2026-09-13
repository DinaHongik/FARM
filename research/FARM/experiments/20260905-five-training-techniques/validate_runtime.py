#!/usr/bin/env python3
"""Exercise all inference branches and the post-warmup curriculum on training inputs."""
import gc
from pathlib import Path
import torch
from torch import nn
from prepare import read,write,SIDES
from train import load_model,config_for,function_step,validate_service_inputs
from evaluate import evaluate,checkpoint_path


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    derived=root/'derived'
    initial=root/'smoke_initial_microbatches'
    for arm in ('s1','f1','f4'):
        evaluate(initial/arm,derived,read(initial/arm/'config.json'),validation_only=True)
        gc.collect();torch.cuda.empty_cache()
    service=read(derived/'service_only'/'train.json')
    service_corpus=read(derived/'service_only'/'corpus.json')
    validate_service_inputs(service,service_corpus)
    contaminated=[{**service[0],'function_name':'must_be_rejected'}]
    try:
        validate_service_inputs(contaminated,service_corpus)
    except ValueError:
        pass
    else:
        raise AssertionError('service-only feature isolation failed')
    corpus=read(derived/'function'/'corpus.json')
    groups=read(derived/'function'/'train.json')
    pools={s:read(derived/'function'/('negatives_'+s+'.json')) for s in SIDES}
    selected=[i for i,pool in enumerate(pools['trigger']) if pool['global']!=pool['within']][:16]
    model=load_model(checkpoint_path(initial/'f2','trigger')/'trigger')
    loss,metrics=function_step({'trigger':model},nn.ModuleDict().cuda(),[groups[i] for i in selected],
                              selected,corpus,pools,'trigger',1,config_for('f2'))
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    assert norm>0
    write(root/'runtime_validation.json',{'status':'passed','inference_branches':['s1','f1','f4'],
          'curriculum_epoch2_backward':'passed','service_contamination_rejected':True,
          'curriculum_training_loss':loss,'gradient_norm':float(norm)})
    print('RUNTIME_VALIDATION_PASSED',flush=True)


if __name__=='__main__':main()
