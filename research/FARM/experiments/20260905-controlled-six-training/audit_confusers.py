"""Small deterministic training-only inspection packet; private artifacts stay on DGX."""
from pathlib import Path
import os,random
import numpy as np
from prepare import read,write,SIDES
from experiment import mine_from_scores


def main():
    os.umask(0o077)
    root=Path(__file__).resolve().parent
    groups=read(root/'derived'/'function'/'train.json')
    corpus=read(root/'derived'/'function'/'corpus.json')
    audit=root.parent/'20260905-training-root-cause-audit'/'f3'
    selected=random.Random(20260905).sample(range(len(groups)),8)
    packet=[]
    for ix,side in enumerate(SIDES):
        data=np.load(audit/(side+'_embeddings.npz'))
        scores=data['train_queries'][selected]@data['documents'].T
        sample=[groups[i] for i in selected]
        pools=mine_from_scores(scores,sample,corpus[side],ix)
        for row,g in enumerate(sample):
            gold=sorted({p[ix] for p in g['valid_pairs']});negative=pools[row]['global'][0]
            packet.append(dict(group_id=g['group_id'],side=side,query=g['query'],
                positives=[corpus[side][p] for p in gold],negative=corpus[side][negative],
                top_negative_score=float(scores[row,negative]),
                best_positive_score=max(float(scores[row,p]) for p in gold)))
    write(root/'CONFUSER_AUDIT_PRIVATE.json',packet)
    for r in packet:
        print(r['side'],r['query'])
        print('GOLD',r['positives'][0]['text'][:450])
        print('CONFUSER',r['negative']['text'][:450])
    print('No labels changed; review packet before the main launch.')


if __name__=='__main__':main()
