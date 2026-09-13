"""Epoch retrieval evidence and checkpoint-bound negative refresh."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from prepare import read, write, sha, SIDES
from experiment import inference_state, mine_from_scores, hierarchical_log_probs


def encode(model, texts, role):
    fn = model.encode_query if role=='query' else model.encode_document
    values = fn(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite embeddings')
    return values


def source_identity(checkpoint, model_side):
    files = sorted(p for p in (checkpoint/model_side).rglob('*') if p.is_file())
    return {str(p.relative_to(checkpoint)): sha(p) for p in files}


def refresh_negatives(model, checkpoint, groups, corpus, side, config, data_sha):
    path = checkpoint/'next_negatives.json'
    identity = dict(model=source_identity(checkpoint, side), data_manifest_sha256=data_sha,
                    phase=side, count=config['negatives_per_query'], ceiling=config['cosine_ceiling'],
                    policy='stable top legal; full positives, aliases and identical texts excluded')
    if path.exists():
        previous = read(path)
        if previous['identity'] != identity:
            raise RuntimeError('mining checkpoint/data identity mismatch')
        return previous['pools']
    started=time.monotonic()
    with inference_state(model):
        q=encode(model,[g['query'] for g in groups],'query')
        d=encode(model,[r['text'] for r in corpus],'document')
        scores=q@d.T
        pools=mine_from_scores(scores,groups,corpus,SIDES.index(side),
                               config['negatives_per_query'],config['cosine_ceiling'])
    write(path,dict(identity=identity,pools=pools,seconds=time.monotonic()-started,
                    groups=len(groups),created_at_unix=time.time()))
    print(json.dumps({'event':'negative_refresh_complete','side':side,'path':str(path),
                      'groups':len(groups),'seconds':time.monotonic()-started}),flush=True)
    return pools


def epoch_validation(model, heads, side, checkpoint, derived, config):
    path=checkpoint/'auxiliary_validation.json'
    if path.exists():
        return
    service=config['arm']=='s1'
    view=derived/('service_only' if service else 'function')
    groups,corpus=read(view/'auxiliary_validation.json'),read(view/'corpus.json')
    started=time.monotonic();results={};rankings={}
    with inference_state(model):
        q=encode(model,[g['query'] for g in groups],'query')
        for role in (SIDES if service else (side,)):
            if service:
                scores=heads[role](torch.from_numpy(q).to(next(model.parameters()).device)).cpu().numpy()
            else:
                d=encode(model,[r['text'] for r in corpus[role]],'document')
                scores=(1./config['temperature'])*q@d.T
                if config['arm']=='h6':
                    service_scores=heads[role](torch.from_numpy(q).to(next(model.parameters()).device)).cpu()
                    scores=hierarchical_log_probs(torch.from_numpy(scores),service_scores,
                            [r['service'] for r in corpus[role]]).numpy()
            ids=np.array([r['id'] for r in corpus[role]])
            order=np.stack([np.lexsort((ids,-row))[:10] for row in scores])
            rankings[role]=order.tolist()
            ix=SIDES.index(role)
            results[role]={f'R@{k}':sum(any(p[ix] in order[i,:k] for p in g['valid_pairs'])
                              for i,g in enumerate(groups))/len(groups) for k in (1,5,10)}
    write(path,dict(split='reranker_train',scope='historically used auxiliary validation',
                   rows=len(groups),group_ids=[g['group_id'] for g in groups],rankings=rankings,
                   metrics=results,seconds=time.monotonic()-started))
    print(json.dumps({'event':'epoch_validation','phase':side,'checkpoint':str(checkpoint),
                      'metrics':results}),flush=True)


def joint_learning_curves(run, derived, config):
    service=config['arm']=='s1'
    groups=read(derived/('service_only' if service else 'function')/'auxiliary_validation.json')
    phases=('shared',) if service else SIDES
    curves=[]
    for epoch in range(1,config['epochs']+1):
        roles={}
        for phase in phases:
            value=read(run/phase/f'epoch-{epoch:02d}'/'auxiliary_validation.json')
            assert value['group_ids']==[g['group_id'] for g in groups]
            roles.update(value['rankings'])
        metrics={f'coverage@{k}':sum(any(p[0] in roles['trigger'][i][:k] and
                           p[1] in roles['action'][i][:k] for p in g['valid_pairs'])
                           for i,g in enumerate(groups))/len(groups) for k in (1,5,10)}
        curves.append({'epoch':epoch,**metrics})
    selected=max(curves,key=lambda row:(row['coverage@1'],-row['epoch']))['epoch']
    write(run/'learning_curves.json',dict(split='reranker_train',rows=len(groups),curves=curves,
          primary_epoch=config['primary_epoch'],secondary_selected_epoch=selected,
          policy='matched role epochs; earliest tie; development does not select checkpoints'))


def candidate_metrics(cache, corpus):
    with torch.no_grad():
        scores=cache['scores'].masked_fill(~cache['allowed'],-torch.inf)
        positive=cache['positive']
        negative=~positive & cache['allowed']
        mass=torch.softmax(scores,dim=-1)
        services=torch.tensor([corpus[i]['service'] for i in cache['candidates']],device=scores.device)
        sibling=torch.zeros_like(positive)
        for row, labels in enumerate(cache['labels']):
            for service in {corpus[i]['service'] for i in labels}:
                sibling[row] |= services==service
        sibling &= negative
        margin=(scores.masked_fill(~positive,-torch.inf).max(-1).values-
                scores.masked_fill(~negative,-torch.inf).max(-1).values)
        finite=margin[torch.isfinite(margin)]
        return dict(candidate_count=len(cache['candidates']),
                    mean_candidate_margin=float(finite.mean()) if len(finite) else None,
                    mean_sibling_softmax_mass=float((mass*sibling).sum(-1).mean()))
