#!/usr/bin/env python3
"""Actual pretrained EmbeddingGemma gradients: ordinary graph versus cached replay."""
import gc,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path('/raid/session/aicontents/farm/experiments/20260905-five-training-techniques')
sys.path.insert(0,str(ROOT))
from prepare import read,write
from train import load_model,prepare_features,mask_for
from objectives import positive_loss,cache_embeddings,replay_embeddings,embed_features


def chunks(model,features,micro=2):
    n=len(features['input_ids'])
    outputs=[]
    for start in range(0,n,micro):
        piece={k:v[start:start+micro] if torch.is_tensor(v) and v.ndim and len(v)==n else v
               for k,v in features.items()}
        outputs.append(embed_features(model,piece))
    return torch.cat(outputs)


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    data=ROOT/'derived'/'function'
    groups=read(data/'train.json');corpus=read(data/'corpus.json')['trigger']
    pools=read(data/'negatives_trigger.json')
    multi=next(i for i,g in enumerate(groups) if len({p[0] for p in g['valid_pairs']})>1)
    indices=[0,multi];chosen=[groups[i] for i in indices]
    labels=[sorted({p[0] for p in g['valid_pairs']}) for g in chosen]
    ids=sorted(set().union(*(set(labels[k])|set(pools[i]['global']) for k,i in enumerate(indices))))
    model=load_model()
    qf=prepare_features(model,[g['query'] for g in chosen],'query')
    df=prepare_features(model,[corpus[i]['text'] for i in ids],'document')
    mask=mask_for(labels,ids,'cuda:0')
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    direct_q=chunks(model,qf);direct_d=chunks(model,df)
    direct_loss=positive_loss(20*direct_q@direct_d.T,mask)
    direct_loss.backward()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    direct_value=float(direct_loss.detach());del direct_q,direct_d,direct_loss
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    q,qc=cache_embeddings(model,qf,2);d,dc=cache_embeddings(model,df,2)
    cached_loss=positive_loss(20*q@d.T,mask)
    cached_loss.backward();replay_embeddings(model,q,qc);replay_embeddings(model,d,dc)
    maximum_absolute=0.;error_sq=0.;ref_sq=0.;bad=[]
    for name,param in model.named_parameters():
        if name not in gradients:
            if param.grad is not None:bad.append(name)
            continue
        assert param.grad is not None,name
        ref=gradients[name];diff=param.grad-ref
        maximum_absolute=max(maximum_absolute,float(diff.abs().max()))
        error_sq+=float(diff.double().square().sum());ref_sq+=float(ref.double().square().sum())
    relative=(error_sq/ref_sq)**.5
    report={'direct_loss':direct_value,'cached_loss':float(cached_loss.detach()),
            'max_absolute_gradient_difference':maximum_absolute,'relative_global_gradient_error':relative,
            'parameter_tensors_compared':len(gradients),'unexpected_gradient_tensors':bad,
            'multi_positive_example':True,'model':'pristine EmbeddingGemma-300m',
            'pooling_include_prompt':getattr(model[1],'include_prompt',None),
            'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}
    assert abs(report['direct_loss']-report['cached_loss'])<1e-4
    assert relative<1e-4 and not bad,report
    report['status']='passed'
    write(Path(__file__).with_name('REAL_GRADIENT_PROBE.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
