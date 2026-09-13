"""Automatic full-dev evaluation, paired comparisons and checkpoint reload checks."""
from collections import Counter
import gc
import math
from pathlib import Path
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from prepare import SIDES, read, write, sha
from objectives import ServiceHead, PairInteraction, embed_features
from pair_scoring import joint_pair_logits


def checkpoint_path(run, phase):
    roots=sorted(p for p in (run/phase).glob('checkpoint-*') if (p/'complete.json').is_file())
    if not roots:
        raise RuntimeError('no complete checkpoint for '+phase)
    return roots[-1]


def load(path):
    model=SentenceTransformer(str(path),device='cuda:0',
                 model_kwargs={'torch_dtype':torch.float32},local_files_only=True,
                 processor_kwargs={'fix_mistral_regex':False})
    model.max_seq_length=512
    model.eval()
    return model


def smoke_reload(run, derived, config, phases):
    checks=[]
    for phase in phases:
        root=checkpoint_path(run,phase)
        roles=('shared',) if phase=='shared' else SIDES if phase=='joint' else (phase,)
        for role in roles:
            model=load(root/role)
            query='When a new item appears, archive it'
            actual=model.encode_query([query],normalize_embeddings=True,convert_to_tensor=True).cpu()
            expected=torch.load(root/(role+'.sentinel.pt'),weights_only=True)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
            checks.append({'phase':phase,'role':role,'max_absolute_difference':float((actual-expected).abs().max())})
            del model
            gc.collect();torch.cuda.empty_cache()
    write(run/'reload_checks.json',{'status':'passed','checks':checks})


def encoded(model,texts,role):
    fn=model.encode_query if role=='query' else model.encode_document
    values=fn(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    if not np.isfinite(values).all():
        raise RuntimeError('non-finite evaluation embeddings')
    return values


def paired_interval(values, families, samples=2000):
    """Paired family-cluster bootstrap; samples resample clusters, retain their cases."""
    keys=sorted(set(families));index={key:i for i,key in enumerate(keys)}
    sums=np.zeros(len(keys));counts=np.zeros(len(keys))
    for value,family in zip(values,families):
        sums[index[family]]+=value;counts[index[family]]+=1
    rng=np.random.default_rng(42)
    boot=[]
    for _ in range(samples):
        selected=rng.integers(0,len(keys),size=len(keys))
        boot.append(float(sums[selected].sum()/counts[selected].sum()))
    return {'difference':float(np.mean(values)),'paired_family_bootstrap_95_percent':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_samples':samples,'families':len(keys)}


def paired_comparisons(run, groups, hits):
    project=run.parents[3]
    old=project/'experiments'/'20260831T171222Z-farm-four-pipelines'/'results'
    paths={'e0_service':old/'e0_dev.json'} if run.name=='s1' else {'e3_function':old/'e3_dev.json'}
    if run.name in ('f2','f3','f4'):
        paths['f1_function']=run.parent/'f1'/'results_dev.json'
    result={}
    for label,path in paths.items():
        if not path.is_file():
            result[label]={'status':'reference_not_finished'};continue
        reference=read(path)
        by_id={r['group_id']:r for r in reference['per_sample']}
        if set(by_id)!={g['group_id'] for g in groups}:
            raise RuntimeError('paired evaluation group mismatch')
        old_hits=[int(by_id[g['group_id']]['hits']['R@1']) for g in groups]
        differences=np.array(hits)-np.array(old_hits)
        result[label]={'status':'complete','reference_sha256':sha(path),
                       'rescues':int(sum(n and not o for n,o in zip(hits,old_hits))),
                       'regressions':int(sum(o and not n for n,o in zip(hits,old_hits))),
                       **paired_interval(differences,[g['family_id'] for g in groups])}
    return result


def evaluate(run, derived, config, validation_only=False):
    started=time.monotonic();service=config['arm']=='s1'
    view=derived/('service_only' if service else 'function')
    groups,train,corpus=read(view/'dev.json'),read(view/'train.json'),read(view/'corpus.json')
    if validation_only:
        groups=train[:16]
    manifest=read(derived/'manifest.json')
    assert sha(view/'dev.json')==manifest['artifacts'][str((view/'dev.json').relative_to(derived))]
    prototypes=torch.load(derived/'service_only'/'prototypes.pt',weights_only=True)
    vectors,scores,rankings={}, {}, {}
    if service:
        checkpoint=checkpoint_path(run,'shared') if validation_only else run/'shared'/'final'
        model=load(checkpoint/'shared')
        query=encoded(model,[g['query'] for g in groups],'query')
        from torch import nn
        heads=nn.ModuleDict({s:ServiceHead(prototypes[s]) for s in SIDES})
        head_state=(torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)['heads']
                    if validation_only else torch.load(checkpoint/'heads.pt',map_location='cpu',weights_only=True))
        heads.load_state_dict(head_state)
        with torch.no_grad():
            for side in SIDES:
                scores[side]=heads[side](torch.from_numpy(query)).numpy()
        del model;gc.collect();torch.cuda.empty_cache()
    else:
        for side in SIDES:
            phase='joint' if config['arm']=='f4' else side
            path=(checkpoint_path(run,phase) if validation_only else run/phase/'final')/side
            model=load(path)
            q=encoded(model,[g['query'] for g in groups],'query')
            d=encoded(model,[r['text'] for r in corpus[side]],'document')
            vectors[side]=(q,d);scores[side]=20.*q@d.T
            del model;gc.collect();torch.cuda.empty_cache()
    for side in SIDES:
        ids=np.array([c['id'] for c in corpus[side]])
        rankings[side]=np.stack([np.lexsort((ids,-row)) for row in scores[side]])
    pair_head=None
    if config['arm']=='f4':
        pair_head=PairInteraction(prototypes['trigger'].shape[1],config['interaction_rank']).eval()
        state=(torch.load(checkpoint_path(run,'joint')/'state.pt',map_location='cpu',weights_only=False)['heads']
               if validation_only else torch.load(run/'joint'/'final'/'heads.pt',map_location='cpu',weights_only=True))
        pair_head.load_state_dict({k.removeprefix('pair.'):v for k,v in state.items()})
    counts=Counter();side_hits={s:{k:[] for k in (1,5,10)} for s in SIDES}
    class_counts={s:{'tp':Counter(),'fp':Counter(),'fn':Counter()} for s in SIDES}
    side_rr={s:[] for s in SIDES}
    records=[];gold_train={tuple(p) for g in train for p in g['valid_pairs']}
    service_names=corpus if service else read(derived/'service_only'/'corpus.json')
    freq={s:Counter() for s in SIDES}
    for g in train:
        for i,s in enumerate(SIDES):
            freq[s].update({p[i] for p in g['valid_pairs']})
    strata={}
    for index,group in enumerate(groups):
        valid={tuple(p) for p in group['valid_pairs']}
        torder,aorder=rankings['trigger'][index],rankings['action'][index]
        trank,arank=np.empty(len(torder),dtype=int),np.empty(len(aorder),dtype=int)
        trank[torder]=np.arange(len(torder));arank[aorder]=np.arange(len(aorder))
        marginal=(int(torder[0]),int(aorder[0]));predicted=marginal
        pair_order=None
        if pair_head is not None:
            ti,ai=torder[:10],aorder[:10]
            qt,dt=vectors['trigger'];qa,da=vectors['action']
            with torch.no_grad():
                residual=pair_head(torch.from_numpy(qt[index]),torch.from_numpy(qa[index]),
                                  torch.from_numpy(dt[ti]),torch.from_numpy(da[ai])).numpy()
            joint=joint_pair_logits(scores['trigger'],scores['action'],index,ti,ai,residual)
            flat_order=np.argsort(-joint.reshape(-1),kind='stable')
            pair_order=[(int(ti[int(n)//len(ai)]),int(ai[int(n)%len(ai)])) for n in flat_order]
            predicted=pair_order[0]
        hit=int(predicted in valid)
        counts['correct']+=hit;counts['marginal_correct']+=int(marginal in valid)
        coverage={k:int(any(trank[t]<k and arank[a]<k for t,a in valid)) for k in (1,5,10)}
        for k in (1,5,10):counts['coverage_'+str(k)]+=coverage[k]
        for i,s in enumerate(SIDES):
            ranks=trank if i==0 else arank
            gold_labels={p[i] for p in valid}
            predicted_label=predicted[i]
            cc=class_counts[s]
            if predicted_label in gold_labels:cc['tp'][predicted_label]+=1
            else:cc['fp'][predicted_label]+=1
            cc['fn'].update(gold_labels-{predicted_label})
            side_rr[s].append(1./(1+min(ranks[p] for p in gold_labels)))
            for k in (1,5,10):side_hits[s][k].append(int(any(ranks[p[i]]<k for p in valid)))
        if pair_order:
            for k in (1,5,10):counts['ranked_pair_'+str(k)]+=int(bool(set(pair_order[:k]) & valid))
        projected_correct=hit
        if not service:
            valid_services={(corpus['trigger'][t]['service'],corpus['action'][a]['service']) for t,a in valid}
            predicted_service=(corpus['trigger'][predicted[0]]['service'],corpus['action'][predicted[1]]['service'])
            projected_correct=int(predicted_service in valid_services)
            counts['projected_service_correct']+=projected_correct
            counts['correct_services_wrong_functions']+=int(projected_correct and not hit)
        tags=['all_pairs_unseen' if not valid.intersection(gold_train) else 'some_pair_seen',
              'multi_gold' if len(valid)>1 else 'single_gold']
        for i,s in enumerate(SIDES):
            frequencies=[freq[s][p[i]] for p in valid]
            tag='all_unseen' if max(frequencies)==0 else 'has_singleton' if 1 in frequencies else 'other_seen'
            tags.append(s+'_'+tag)
        if service:
            explicit=[any(corpus[s][p[i]]['name'].casefold() in group['query'].casefold()
                          for p in valid) for i,s in enumerate(SIDES)]
        else:
            explicit=[any(service_names[s][corpus[s][p[i]]['service']]['name'].casefold()
                          in group['query'].casefold() for p in valid) for i,s in enumerate(SIDES)]
        tags.append('both_service_names_explicit' if all(explicit) else
                    'neither_service_name_explicit' if not any(explicit) else 'one_service_name_explicit')
        for tag in tags:
            counter=strata.setdefault(tag,Counter());counter['rows']+=1;counter['correct']+=hit
        records.append({'group_id':group['group_id'],'family_id':group['family_id'],
          'valid_pairs':[[corpus['trigger'][t]['id'],corpus['action'][a]['id']] for t,a in sorted(valid)],
          'prediction':[corpus[s][predicted[i]]['id'] for i,s in enumerate(SIDES)],
          'top_trigger_ids':[corpus['trigger'][int(i)]['id'] for i in torder[:10]],
          'top_action_ids':[corpus['action'][int(i)]['id'] for i in aorder[:10]],
          'hits':{'R@1':hit},'candidate_lattice_coverage':coverage,
          'marginal_top1_correct':int(marginal in valid),'projected_service_correct':projected_correct})
    n=len(groups);hits=[r['hits']['R@1'] for r in records]
    metrics={'exact_pair_top1':counts['correct']/n,'correct':counts['correct'],
             'marginal_top1':counts['marginal_correct']/n,
             'candidate_lattice_coverage':{str(k):counts['coverage_'+str(k)]/n for k in (1,5,10)},
             'sides':{s:{'R@'+str(k):float(np.mean(side_hits[s][k])) for k in (1,5,10)} for s in SIDES},
             'strata':{k:{**v,'accuracy':v['correct']/v['rows']} for k,v in strata.items()}}
    for side in SIDES:
        cc=class_counts[side];classes=set(cc['tp'])|set(cc['fp'])|set(cc['fn'])
        f1=[2*cc['tp'][c]/max(1,2*cc['tp'][c]+cc['fp'][c]+cc['fn'][c]) for c in classes]
        metrics['sides'][side]['macro_F1_multilabel_gold']=float(np.mean(f1))
        metrics['sides'][side]['MRR']=float(np.mean(side_rr[side]))
    metrics['macro_F1_policy']='each observed side label is a positive; one predicted label; macro over gold/predicted union'
    if not service:
        metrics['projected_service_pair_accuracy']=counts['projected_service_correct']/n
        metrics['correct_services_wrong_functions']=counts['correct_services_wrong_functions']
    if pair_head is not None:
        metrics['ranked_pair_recall']={str(k):counts['ranked_pair_'+str(k)]/n for k in (1,5,10)}
        metrics['selection_accuracy_given_lattice_coverage']=counts['correct']/max(1,counts['coverage_10'])
        metrics['interaction_rescues']=sum(r['hits']['R@1'] and not r['marginal_top1_correct'] for r in records)
        metrics['interaction_regressions']=sum(r['marginal_top1_correct'] and not r['hits']['R@1'] for r in records)
    result={'status':'complete','arm':config['arm'],'task_level':'service' if service else 'function',
            'split':'encoder_train' if validation_only else 'dev',
            'evaluation_scope':'training-only software check' if validation_only else 'previously exposed development set; no final-test evaluation',
            'rows':n,'metrics':metrics,'paired_comparisons':{} if validation_only else paired_comparisons(run,groups,hits),
            'evaluation_seconds':time.monotonic()-started,'per_sample':records,
            'completed_at_unix':time.time(),'config':config,'data_manifest_sha256':sha(derived/'manifest.json')}
    if validation_only:
        assert 0 <= metrics['exact_pair_top1'] <= 1
        assert n==16
        write(run/'evaluation_validation.json',{'status':'passed','rows':n,'branch':config['arm'],
                                              'evaluation_seconds':result['evaluation_seconds']})
        print('EVALUATION_VALIDATION_PASSED',config['arm'],flush=True)
        return
    write(run/'results_dev.json',result)
    write(run/'summary.json',{k:v for k,v in result.items() if k!='per_sample'})
    write(run/'progress.json',{'status':'complete','arm':config['arm'],'rows':n,'exact_pair_top1':metrics['exact_pair_top1'],
                              'completed_at_unix':time.time()})
    print('EVALUATION_COMPLETE',config['arm'],counts['correct'],n,flush=True)
