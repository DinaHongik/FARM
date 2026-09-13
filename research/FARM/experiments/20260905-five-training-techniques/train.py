#!/usr/bin/env python3
"""Five isolated FARM trainers, resumable at exact optimizer steps."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

from prepare import BASE, REVISION, SIDES, read, write, sha
from objectives import (ServiceHead, PairInteraction, positive_loss,
                        cache_embeddings, replay_embeddings, embed_features)


def config_for(arm):
    return dict(arm=arm, base_model=BASE, revision=REVISION, epochs=3, batch_size=16,
                learning_rate=2e-5, head_learning_rate=1e-3, weight_decay=.01,
                warmup_ratio=.1, seed=42, max_seq_length=512, temperature=.05,
                microbatch=4 if arm == 's1' else 8, service_weight=.2,
                pair_weight=.2, interaction_rank=32, checkpoint_steps=250,
                full_backbone_trainable=True, dtype='float32')


def prepare_features(model, texts, role):
    return model.preprocess(texts, prompt=model.prompts[role])


def load_model(path=None):
    kwargs = {'revision': REVISION} if path is None else {}
    if path is not None:
        kwargs['processor_kwargs']={'fix_mistral_regex':False}
    model = SentenceTransformer(str(path) if path else BASE, **kwargs,
             device='cuda:0', model_kwargs={'torch_dtype': torch.float32}, local_files_only=True)
    model.max_seq_length = 512
    assert model.prompts.get('query') and model.prompts.get('document')
    model.train()
    return model


def side_labels(groups, side_index):
    return [sorted({pair[side_index] for pair in g['valid_pairs']}) for g in groups]


def mask_for(labels, candidates, device):
    index = {value: i for i, value in enumerate(candidates)}
    mask = torch.zeros((len(labels),len(candidates)), dtype=torch.bool, device=device)
    for i, values in enumerate(labels):
        for value in values:
            mask[i,index[value]] = True
    return mask


def side_batch(model, groups, indices, corpus, pools, side, epoch, config):
    side_index = SIDES.index(side)
    labels = side_labels(groups, side_index)
    strategy = 'within' if config['arm'] == 'f2' and epoch >= 1 else 'global'
    local = [sorted(set(positive + pools[idx][strategy]))
             for positive, idx in zip(labels, indices)]
    candidates = sorted(set().union(*(set(x) for x in local)))
    positive = mask_for(labels, candidates, 'cuda:0')
    allowed = torch.ones_like(positive)
    for row, values in enumerate(labels):
        texts = {corpus[i]['text'] for i in values}
        aliases = set().union(*(set(corpus[i]['equivalent_indices']) for i in values))
        for column, idx in enumerate(candidates):
            if idx not in values and (idx in aliases or corpus[idx]['text'] in texts):
                allowed[row,column] = False
    q, qc = cache_embeddings(model, prepare_features(model, [g['query'] for g in groups], 'query'),
                             config['microbatch'])
    d, dc = cache_embeddings(model, prepare_features(model, [corpus[i]['text'] for i in candidates], 'document'),
                             config['microbatch'])
    scores = 20. * q @ d.T
    loss = positive_loss(scores, positive, allowed)
    return dict(q=q, d=d, qc=qc, dc=dc, scores=scores, loss=loss,
                candidates=candidates, local=local, labels=labels)


def function_step(models, heads, groups, indices, corpus, pools, phase, epoch, config):
    sides = SIDES if phase == 'joint' else (phase,)
    cache = {s: side_batch(models[s], groups, indices, corpus[s], pools[s], s, epoch, config)
             for s in sides}
    loss = sum(c['loss'] for c in cache.values())
    metrics = {'marginal_loss': float(loss.detach())}
    if config['arm'] == 'f3':
        c = cache[phase]
        labels = [sorted({corpus[phase][i]['service'] for i in values}) for values in c['labels']]
        logits = heads[phase](c['q'])
        service_loss = positive_loss(logits, mask_for(labels, list(range(logits.shape[1])), logits.device))
        loss = loss + config['service_weight'] * service_loss
        metrics['service_loss'] = float(service_loss.detach())
    if phase == 'joint':
        t, a = cache['trigger'], cache['action']
        tmap = {idx:i for i,idx in enumerate(t['candidates'])}
        amap = {idx:i for i,idx in enumerate(a['candidates'])}
        pair_losses = []
        for row, group in enumerate(groups):
            tids, aids = t['local'][row], a['local'][row]
            ti, ai = [tmap[i] for i in tids], [amap[i] for i in aids]
            logits = t['scores'][row,ti,None] + a['scores'][row,None,ai]
            logits = logits + heads['pair'](t['q'][row], a['q'][row], t['d'][ti], a['d'][ai])
            positives = torch.zeros_like(logits, dtype=torch.bool)
            for pt, pa in group['valid_pairs']:
                positives[tids.index(pt),aids.index(pa)] = True
            pair_losses.append(positive_loss(logits.flatten()[None], positives.flatten()[None]))
        pair_loss = torch.stack(pair_losses).mean()
        loss = loss + config['pair_weight'] * pair_loss
        metrics['pair_loss'] = float(pair_loss.detach())
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite training loss')
    loss.backward()
    for side in sides:
        c = cache[side]
        replay_embeddings(models[side], c['q'], c['qc'])
        replay_embeddings(models[side], c['d'], c['dc'])
    return float(loss.detach()), metrics


def service_step(model, heads, groups, config):
    q, chunks = cache_embeddings(model, prepare_features(model,[g['query'] for g in groups],'query'),
                                config['microbatch'])
    losses = []
    for i, side in enumerate(SIDES):
        logits = heads[side](q)
        mask = mask_for(side_labels(groups,i), list(range(logits.shape[1])),logits.device)
        losses.append(positive_loss(logits,mask))
    loss = sum(losses)
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite service loss')
    loss.backward()
    replay_embeddings(model,q,chunks)
    return float(loss.detach()), {'service_loss': float(loss.detach())}


def checkpoint(root, models, heads, optimizer, scheduler, step, fingerprint):
    temporary = root / ('checkpoint-%06d.tmp' % step)
    destination = root / ('checkpoint-%06d' % step)
    if destination.exists():
        return
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for side, model in models.items():
        model.save_pretrained(str(temporary / side))
        model.eval()
        with torch.no_grad():
            sentinel = embed_features(model,prepare_features(model,['When a new item appears, archive it'],'query')).cpu()
        torch.save(sentinel,temporary/(side+'.sentinel.pt'))
        model.train()
    torch.save({'heads': heads.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'python_rng': random.getstate(), 'fingerprint': fingerprint}, temporary/'state.pt')
    write(temporary/'complete.json', {'step':step,'fingerprint':fingerprint})
    os.replace(temporary,destination)
    # Keep two complete resumable checkpoints, never delete a pre-existing run.
    complete = sorted(p for p in root.glob('checkpoint-*') if (p/'complete.json').is_file())
    for old in complete[:-2]:
        shutil.rmtree(old)


def latest_checkpoint(root, fingerprint):
    options = sorted(p for p in root.glob('checkpoint-*') if (p/'complete.json').is_file())
    if not options:
        return None
    path = options[-1]
    if read(path/'complete.json')['fingerprint'] != fingerprint:
        raise RuntimeError('checkpoint configuration/data/code identity changed')
    return path


def validate_service_inputs(groups, corpus):
    for group in groups:
        if set(group) != {'group_id','family_id','query','valid_pairs'}:
            raise ValueError('service loader received extra features')
        for pair in group['valid_pairs']:
            if len(pair)!=2 or not all(isinstance(i,int) for i in pair):
                raise ValueError('service pairs must be integer service indices')
    for side in SIDES:
        for row in corpus[side]:
            if set(row) != {'id','name'}:
                raise ValueError('service catalog received function metadata')


def train_phase(run, derived, phase, config, max_steps=0):
    root = run / phase
    root.mkdir(parents=True,exist_ok=True)
    service = config['arm']=='s1'
    view = derived / ('service_only' if service else 'function')
    groups, corpus = read(view/'train.json'), read(view/'corpus.json')
    if service:
        validate_service_inputs(groups,corpus)
    pools = {} if service else {s:read(view/('negatives_'+s+'.json')) for s in SIDES}
    manifest = read(derived/'manifest.json')
    expected_files = [view/'train.json',view/'corpus.json',derived/'service_only'/'prototypes.pt']
    if not service:
        expected_files += [view/('negatives_'+s+'.json') for s in SIDES]
    for path in expected_files:
        assert sha(path)==manifest['artifacts'][str(path.relative_to(derived))],path.name
    prototypes = torch.load(derived/'service_only'/'prototypes.pt',map_location='cpu',weights_only=True)
    steps_per_epoch = math.ceil(len(groups)/config['batch_size'])
    total_steps = steps_per_epoch*config['epochs']
    target_steps = min(total_steps,max_steps) if max_steps else total_steps
    code = {name:sha(Path(__file__).with_name(name)) for name in ('train.py','objectives.py','prepare.py')}
    identity = {'config':config,'data_manifest':sha(derived/'manifest.json'),'code':code,'phase':phase}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (root/'done.json').exists() and read(root/'done.json').get('fingerprint')==fingerprint:
        return
    write(root/'identity.json',identity)
    saved = latest_checkpoint(root,fingerprint)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    model_sides = ('shared',) if service else SIDES if phase=='joint' else (phase,)
    models = {s:load_model(saved/s if saved else None) for s in model_sides}
    heads = nn.ModuleDict()
    if service:
        for s in SIDES:
            heads[s] = ServiceHead(prototypes[s])
    elif config['arm']=='f3':
        heads[phase] = ServiceHead(prototypes[phase])
    elif phase=='joint':
        heads['pair'] = PairInteraction(prototypes['trigger'].shape[1],config['interaction_rank'])
    heads.cuda().train()
    params = [p for m in models.values() for p in m.parameters() if p.requires_grad]
    optimizer_groups = [{'params':params,'lr':config['learning_rate']}]
    head_params = list(heads.parameters())
    if head_params:
        optimizer_groups.append({'params':head_params,'lr':config['head_learning_rate']})
    optimizer = torch.optim.AdamW(optimizer_groups,weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,round(total_steps*.1),total_steps)
    step = 0
    if saved:
        state = torch.load(saved/'state.pt',map_location='cpu',weights_only=False)
        heads.load_state_dict(state['heads'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        del state
    first_step = step
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'event':'train_start','phase':phase,'arm':config['arm'],
                      'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                      'step':step,'target_steps':target_steps,'training_groups':len(groups)}),flush=True)
    cached_epoch, order = -1,None
    while step < target_steps:
        epoch,batch = divmod(step,steps_per_epoch)
        if epoch != cached_epoch:
            order = list(range(len(groups)))
            random.Random(config['seed']+epoch).shuffle(order)
            cached_epoch = epoch
        indices = order[batch*config['batch_size']:(batch+1)*config['batch_size']]
        examples = [groups[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        if service:
            loss,extra = service_step(models['shared'],heads,examples,config)
        else:
            loss,extra = function_step(models,heads,examples,indices,corpus,pools,phase,epoch,config)
        gradient_norm = nn.utils.clip_grad_norm_(params+head_params,1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        if step==first_step+1 or step%5==0 or step==target_steps:
            elapsed = time.monotonic()-started
            record = {'status':'training','arm':config['arm'],'phase':phase,'pid':os.getpid(),
                      'gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'step':step,'total_steps':total_steps,
                      'epoch':epoch+batch/steps_per_epoch,'loss':loss,**extra,
                      'gradient_norm':float(gradient_norm),'elapsed_seconds':elapsed,
                      'seconds_per_step':elapsed/(step-first_step),
                      'remaining_phase_seconds':elapsed/(step-first_step)*(total_steps-step),
                      'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                      'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                      'learning_rate':scheduler.get_last_lr()[0], 'updated_at_unix':time.time()}
            write(root/'progress.json',record)
            write(run/'progress.json',record)
            print(json.dumps(record),flush=True)
        if step in (5,50) or step%config['checkpoint_steps']==0 or step==target_steps:
            checkpoint(root,models,heads,optimizer,scheduler,step,fingerprint)
    if step==total_steps:
        final = root/'final'
        final.mkdir(exist_ok=True)
        for side,model in models.items():
            model.save_pretrained(str(final/side))
        torch.save(heads.state_dict(),final/'heads.pt')
        write(root/'done.json',{'status':'trained','step':step,'fingerprint':fingerprint,
                              'training_seconds_this_process':time.monotonic()-started,
                              'checkpoint':str(final)})
    else:
        write(root/'smoke.done.json',{'status':'smoke_passed','step':step,'fingerprint':fingerprint})
    del models,heads,optimizer,scheduler,params,head_params,optimizer_groups
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('s1','f1','f2','f3','f4'),required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--derived',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--phase',choices=('shared','trigger','action','joint'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.umask(0o077)
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or torch.cuda.device_count()!=1:
        raise RuntimeError('exactly one allocated GPU required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    config=config_for(args.arm)
    run=args.run_root/('smoke' if args.smoke else 'arms')/args.arm
    run.mkdir(parents=True,exist_ok=True)
    write(run/'config.json',config)
    phases=('shared',) if args.arm=='s1' else ('joint',) if args.arm=='f4' else SIDES
    if args.phase:
        assert args.phase in phases
        phases=(args.phase,)
    try:
        for phase in phases:
            train_phase(run,args.derived,phase,config,args.max_steps)
        if not args.max_steps and not args.phase:
            write(run/'progress.json',{'status':'evaluating','arm':args.arm,'pid':os.getpid(),
                                      'gpu':int(os.environ['CUDA_VISIBLE_DEVICES'])})
            from evaluate import evaluate
            evaluate(run,args.derived,config)
        elif args.smoke:
            from evaluate import smoke_reload
            smoke_reload(run,args.derived,config,phases)
    except Exception as exc:
        write(run/'failure.json',{'status':'failed','arm':args.arm,'error':repr(exc),
                                  'pid':os.getpid(),'updated_at_unix':time.time()})
        raise


if __name__=='__main__':
    main()
