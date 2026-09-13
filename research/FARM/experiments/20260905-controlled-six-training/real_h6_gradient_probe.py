#!/usr/bin/env python3
"""Actual pretrained EmbeddingGemma gradients: ordinary graph versus cached replay."""
import gc,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from prepare import read,write
from train import load_model,prepare_features,mask_for
from objectives import ServiceHead,cache_embeddings,replay_embeddings,embed_features
from experiment import hierarchical_loss


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
    from collections import Counter
    sizes=Counter(r['service'] for r in corpus)
    first=next(i for i,g in enumerate(groups) if any(sizes[corpus[p[0]]['service']]>len({v[0] for v in g['valid_pairs']}) for p in g['valid_pairs']))
    indices=[first,multi];chosen=[groups[i] for i in indices]
    labels=[sorted({p[0] for p in g['valid_pairs']}) for g in chosen]
    services={corpus[p]['service'] for values in labels for p in values}
    ids=[i for i,r in enumerate(corpus) if r['service'] in services]
    doc_services=[corpus[i]['service'] for i in ids]
    model=load_model()
    prototypes=torch.load(ROOT/'derived'/'service_only'/'prototypes.pt',weights_only=True)
    head=ServiceHead(prototypes['trigger']).cuda()
    qf=prepare_features(model,[g['query'] for g in chosen],'query')
    df=prepare_features(model,[corpus[i]['text'] for i in ids],'document')
    mask=mask_for(labels,ids,'cuda:0')
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    direct_q=chunks(model,qf);direct_d=chunks(model,df)
    direct_loss=hierarchical_loss(20*direct_q@direct_d.T,head(direct_q),doc_services,mask)
    direct_loss.backward()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    head_gradients={name:p.grad.detach().clone() for name,p in head.named_parameters()}
    direct_value=float(direct_loss.detach());del direct_q,direct_d,direct_loss
    model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    q,qc=cache_embeddings(model,qf,2);d,dc=cache_embeddings(model,df,2)
    cached_loss=hierarchical_loss(20*q@d.T,head(q),doc_services,mask)
    cached_loss.backward();replay_embeddings(model,q,qc);replay_embeddings(model,d,dc)
    assert d.grad.norm().item()>0, 'probe needs a genuine within-service document gradient'
    maximum_absolute=0.;error_sq=0.;ref_sq=0.;bad=[]
    for name,param in model.named_parameters():
        if name not in gradients:
            if param.grad is not None:bad.append(name)
            continue
        assert param.grad is not None,name
        ref=gradients[name];diff=param.grad-ref
        maximum_absolute=max(maximum_absolute,float(diff.abs().max()))
        error_sq+=float(diff.double().square().sum());ref_sq+=float(ref.double().square().sum())
    for name,param in head.named_parameters():
        torch.testing.assert_close(param.grad,head_gradients[name],rtol=1e-4,atol=1e-5)
    relative=(error_sq/ref_sq)**.5
    report={'direct_loss':direct_value,'cached_loss':float(cached_loss.detach()),
            'max_absolute_gradient_difference':maximum_absolute,'relative_global_gradient_error':relative,
            'parameter_tensors_compared':len(gradients),'unexpected_gradient_tensors':bad,
            'multi_positive_example':True,'model':'pristine EmbeddingGemma-300m; H6 exact conditional loss', 'head_gradient_checked':True, 'documents':len(ids), 'document_gradient_norm':float(d.grad.norm()),
            'pooling_include_prompt':getattr(model[1],'include_prompt',None),
            'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}
    assert abs(report['direct_loss']-report['cached_loss'])<1e-4
    assert relative<1e-4 and not bad,report
    report['status']='passed'
    write(Path(__file__).with_name('REAL_H6_GRADIENT_PROBE.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
