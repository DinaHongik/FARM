from pathlib import Path
import hashlib,json
from retrieval_metrics import paired_comparison

root=Path('/raid/session/aicontents/farm/experiments/20260905-strong-validation')
def read(p):return json.loads(p.read_text())
arms={};pairs={}
for seed in [42,1337]:
    records={}
    for policy in ['fixed','refresh']:
        run=f'f3_{policy}_seed{seed}';p=root/'evaluations'/run
        arms[run]=read(p/'summary.json')
        records[policy]=read(p/'predictions.json')
    pairs[str(seed)]=paired_comparison(records['fixed'],records['refresh'])
result={'status':'complete','audit':read(root/'results/HOLDOUT_AUDIT.json'),
        'primary_seed':1337,'replication_seed':42,'arms':arms,'paired_comparisons':pairs,
        'scope':'frozen epoch-three checkpoints; no retraining or tuning on this subset',
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'results/HOLDOUT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'complete','rows':result['audit']['remaining_rows'],'pairs':pairs},indent=2))
