#!/usr/bin/env python3
"""Read-only final-model retrieval diagnostics; no optimizer or benchmark selection."""
import argparse
from collections import Counter
import gc,json,random,sys,time
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/raid/session/aicontents/farm/experiments/20260905-five-training-techniques')
sys.path.insert(0,str(ROOT))
from prepare import SIDES,read,write
from evaluate import load,encoded


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('f1','f3'),required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(__file__).resolve().parent/args.arm;out.mkdir(exist_ok=True)
    derived=ROOT/'derived'/'function'
    train,dev,corpus=read(derived/'train.json'),read(derived/'dev.json'),read(derived/'corpus.json')
    report={'arm':args.arm,'scope':'final checkpoint diagnostics; existing train/dev only','sides':{}}
    retrieved={}
    order=list(range(len(train)));random.Random(44).shuffle(order)
    for side_index,side in enumerate(SIDES):
        model=load(ROOT/'arms'/args.arm/side/'final'/side)
        d=encoded(model,[r['text'] for r in corpus[side]],'document')
        qt=encoded(model,[r['query'] for r in train],'query')
        qd=encoded(model,[r['query'] for r in dev],'query')
        np.savez_compressed(out/(side+'_embeddings.npz'),documents=d,train_queries=qt,dev_queries=qd)
        scores=qt@d.T;devscores=qd@d.T
        pools=read(derived/('negatives_'+side+'.json'))
        positives=[{p[side_index] for p in g['valid_pairs']} for g in train]
        text_indices={}
        for j,row in enumerate(corpus[side]):text_indices.setdefault(row['text'],set()).add(j)
        blocked=[set().union(*(set(corpus[side][p]['equivalent_indices']) |
                               text_indices[corpus[side][p]['text']] for p in pos)) for pos in positives]
        count=Counter();sampled_pool_margins=[];full_margins=[];inbatch_margins=[]
        sampled_neg_ranks=[];top_confuser_in_saved_four=0
        for group_idx in range(len(train)):
            pos=positives[group_idx]
            gold=max(scores[group_idx,p] for p in pos)
            ranked=np.argsort(-scores[group_idx],kind='stable').tolist()
            wrong=[j for j in ranked if j not in blocked[group_idx]]
            first_wrong=wrong[0]
            negative=pools[group_idx]['global']
            sample_max=max(scores[group_idx,n] for n in negative)
            full_max=scores[group_idx,first_wrong]
            sampled_pool_margins.append(float(gold-sample_max));full_margins.append(float(gold-full_max))
            count['beats_saved_four']+=int(gold>sample_max)
            count['beats_saved_four_but_loses_full_catalog']+=int(gold>sample_max and gold<full_max)
            top_confuser_in_saved_four+=int(first_wrong in negative)
            ranks={n:i+1 for i,n in enumerate(ranked)}
            sampled_neg_ranks.extend(ranks[n] for n in negative)
        for start in range(0,len(order),16):
            batch=order[start:start+16]
            candidates=set().union(*(positives[i]|set(pools[i]['global']) for i in batch))
            for idx in batch:
                negatives=candidates-blocked[idx]
                gold=max(scores[idx,p] for p in positives[idx])
                maximum=max(scores[idx,n] for n in negatives)
                inbatch_margins.append(float(gold-maximum))
                count['beats_final_epoch_candidate_pool']+=int(gold>maximum)
                count['beats_final_epoch_pool_but_loses_full_catalog']+=int(gold>maximum and full_margins[idx]<0)
        metrics={}
        for split,groups,values in [('train',train,scores),('dev',dev,devscores)]:
            rank=np.argsort(-values,axis=1,kind='stable')
            hits={k:sum(any(p[side_index] in set(rank[i,:k]) for p in g['valid_pairs']) for i,g in enumerate(groups))
                  for k in (1,5,10)}
            metrics[split]={'rows':len(groups),**{f'R@{k}':hits[k]/len(groups) for k in hits}}
            retrieved[(side,split)]=rank[:,:10]
        tokenizer=model.tokenizer
        lengths=tokenizer([model.prompts['document']+r['text'] for r in corpus[side]],
                          truncation=False,padding=False,return_length=True)['length']
        report['sides'][side]={**metrics,'training_diagnostics':dict(count),
          'saved_four_contains_current_hardest_confuser':top_confuser_in_saved_four,
          'mean_saved_four_margin':float(np.mean(sampled_pool_margins)),
          'median_saved_negative_rank':float(np.median(sampled_neg_ranks)),
          'mean_full_catalog_margin':float(np.mean(full_margins)),
          'mean_final_epoch_candidate_margin':float(np.mean(inbatch_margins)),
          'documents_over_512_tokens':sum(n>512 for n in lengths),'max_document_tokens':max(lengths)}
        write(out/'diagnostics.json',report)
        print(args.arm,side,json.dumps(report['sides'][side]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    report['joint']={}
    for split,groups in [('train',train),('dev',dev)]:
        counts={k:sum(any(p[0] in set(retrieved[('trigger',split)][i,:k]) and
                         p[1] in set(retrieved[('action',split)][i,:k]) for p in g['valid_pairs'])
                     for i,g in enumerate(groups)) for k in (1,5,10)}
        report['joint'][split]={'rows':len(groups),**{f'coverage@{k}':counts[k]/len(groups) for k in counts}}
    write(out/'diagnostics.json',report)
    print('DONE',args.arm,json.dumps(report['joint']),flush=True)


if __name__=='__main__':main()
