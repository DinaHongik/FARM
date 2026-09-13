"""Validate against recorded development outcomes, then score frozen holdout."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from retrieval_metrics import score_rankings

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,1337],required=True);args=p.parse_args()
base=Path(__file__).resolve().parents[3];source=base/'experiments/20260905-controlled-six-training'
root=base/'experiments/20260905-strong-validation'
sys.path.insert(0,str(source))
from evaluate import load,encoded
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
corpus=read(source/'derived/function/corpus.json')
dev=read(source/'derived/function/dev.json')
holdout_path=root/'prepared/holdout.json'
audit=read(root/'results/HOLDOUT_AUDIT.json')
assert sha(holdout_path)==audit['holdout_sha256']
holdout=read(holdout_path)
for policy in ['fixed','refresh']:
    run_id=f'f3_{policy}_seed{args.seed}'
    run=source/'arms'/run_id
    output=root/'evaluations'/run_id;output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise RuntimeError('Frozen evaluation already exists; no silent overwrite')
    config=read(run/'config.json');expected=read(run/'results_dev.json')
    rankings={split:{} for split in ['dev','holdout']};identities={};started=time.monotonic();documents={}
    for side in ['trigger','action']:
        path=run/side/'final'/side
        identities[side]={str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file()}
        model=load(path);model.max_seq_length=512
        d=encoded(model,[r['text'] for r in corpus[side]],'document')
        ids=np.array([r['id'] for r in corpus[side]])
        documents[side]=d
        q=encoded(model,[g['query'] for g in dev],'query')
        scores=(1./config['temperature'])*q@d.T
        rankings['dev'][side]=np.stack([np.lexsort((ids,-row))[:10] for row in scores])
        del model;gc.collect();torch.cuda.empty_cache()
    dev_metrics,dev_records=score_rankings(dev,corpus,rankings['dev'])
    old={r['group_id']:r for r in expected['per_sample']}
    assert dev_metrics['correct']==expected['metrics']['correct']
    assert all(r['correct']==old[r['group_id']]['hits']['R@1'] and r['prediction']==old[r['group_id']]['prediction'] for r in dev_records)
    for side in ['trigger','action']:
        model=load(run/side/'final'/side);model.max_seq_length=512
        q=encoded(model,[g['query'] for g in holdout],'query')
        scores=(1./config['temperature'])*q@documents[side].T
        ids=np.array([r['id'] for r in corpus[side]])
        rankings['holdout'][side]=np.stack([np.lexsort((ids,-row))[:10] for row in scores])
        del model;gc.collect();torch.cuda.empty_cache()
    metrics,records=score_rankings(holdout,corpus,rankings['holdout'])
    summary={'status':'complete','run_id':run_id,'seed':args.seed,'policy':policy,'checkpoint_epoch':3,
             'split':'remaining_test_holdout','metrics':metrics,'development_reproduction':{'rows':len(dev),'all_predictions_match':True,'correct':dev_metrics['correct']},
             'protocol_sha256':audit['protocol_sha256'],'holdout_sha256':audit['holdout_sha256'],
             'checkpoint_files_sha256':identities,'evaluation_seconds':time.monotonic()-started,
             'script_sha256':sha(Path(__file__)),'metrics_script_sha256':sha(Path(__file__).with_name('retrieval_metrics.py'))}
    (output/'predictions.json').write_text(json.dumps(records,sort_keys=True)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'run_id':run_id,'metrics':metrics,'dev_reproduced':True}),flush=True)
