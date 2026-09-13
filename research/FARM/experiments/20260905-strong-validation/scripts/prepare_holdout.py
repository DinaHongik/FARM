"""Identifier/family exclusions before evaluating the frozen checkpoints."""
from pathlib import Path
import hashlib,json,sys

base=Path(__file__).resolve().parents[3]
source=base/'experiments/20260905-controlled-six-training'
root=base/'experiments/20260905-strong-validation'
data=base/'data/v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(data/'manifest.json')
test_path=data/'splits/test.json'
assert sha(test_path)==manifest['artifacts']['splits/test.json']['sha256']
exposure_path=base/'experiments/20260904T170552Z-farm-round9-agentic-benchmarks/manifests/samples/farm_v2_test.json'
sample=read(exposure_path)
ids=sample.get('ordered_case_ids')
if ids is None:
    # Prepared cases are already-exposed records; read identifiers only into the exclusion set.
    path=exposure_path.parents[2]/'prepared/farm_v2_test/cases.jsonl'
    ids=[json.loads(line)['case_id'] for line in path.read_text().splitlines() if line.strip()]
assert len(ids)==150 and len(set(ids))==150
exposed={i.removeprefix('farm:') for i in ids}
rows=read(test_path)
assert exposed.issubset({g['group_id'] for g in rows})
families={g['semantic_family_id'] for g in rows if g['group_id'] in exposed}
overlap={}
for split in ['encoder_train','dev','reranker_train']:
    p=data/'splits'/f'{split}.json'
    other=read(p)
    overlap[split]={g['semantic_family_id'] for g in other}&{g['semantic_family_id'] for g in rows}
for values in overlap.values():families.update(values)
kept=[g for g in rows if g['semantic_family_id'] not in families]
assert kept
corpus=read(source/'derived/function/corpus.json')
lookup={s:{} for s in ['trigger','action']}
for s in lookup:
    for i,row in enumerate(corpus[s]):
        lookup[s][row['id']]=i
    for i,row in enumerate(corpus[s]):
        for alias in row['aliases']:lookup[s].setdefault(alias,i)
groups=[]
for g in kept:
    pairs=sorted({tuple(lookup[s][p[s+'_url']] for s in ['trigger','action']) for p in g['valid_pairs']})
    groups.append({'group_id':g['group_id'],'family_id':g['semantic_family_id'],'query':g['query'],'valid_pairs':pairs})
target=root/'prepared';target.mkdir(exist_ok=True)
p=target/'holdout.json'
if p.exists():raise RuntimeError('Holdout already prepared; do not silently redefine population')
p.write_text(json.dumps(groups,sort_keys=True)+'\n')
audit={'status':'prepared','test_total':len(rows),'previously_exposed_cases':len(exposed),
       'excluded_by_family_or_other_splits':len(rows)-len(kept),'remaining_rows':len(kept),
       'remaining_families':len({g['family_id'] for g in groups}),
       'family_overlap_with_other_splits':{k:len(v) for k,v in overlap.items()},
       'test_source_sha256':sha(test_path),'exposure_manifest_sha256':sha(exposure_path),
       'holdout_sha256':sha(p),'catalog_sha256':sha(source/'derived/function/corpus.json'),
       'protocol_sha256':sha(root/'VALIDATION_PROTOCOL.md'),
       'ordered_group_ids_sha256':hashlib.sha256('\n'.join(g['group_id'] for g in groups).encode()).hexdigest(),
       'scope':'remaining existing test subset; excludes documented prior-exposure families; no claim of globally pristine data'}
(root/'results/HOLDOUT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
