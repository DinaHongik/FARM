"""Reuse immutable prior data and add a family-separated auxiliary validation view."""
from pathlib import Path
from collections import Counter
import json, random, shutil
from prepare import read, write, sha, SIDES
from experiment import RUNS, configuration


def main():
    root=Path(__file__).resolve().parent
    project=root.parents[1]
    previous=root.parent/'20260905-five-training-techniques'/'derived'
    out=root/'derived'
    if out.exists():
        raise RuntimeError('derived data already prepared; do not overwrite')
    shutil.copytree(previous,out)
    corpus=read(out/'function'/'corpus.json')
    groups=read(out/'function'/'train.json')
    lookups={s:{r['id']:i for i,r in enumerate(corpus[s])} for s in SIDES}
    for s in SIDES:
        for i,r in enumerate(corpus[s]):
            for alias in r['aliases']:lookups[s].setdefault(alias,i)
    original=project/'data'/'v2'
    source=original/'splits'/'reranker_train.json'
    expected=read(original/'manifest.json')['artifacts']['splits/reranker_train.json']['sha256']
    assert sha(source)==expected
    validation=[];service_validation=[]
    for g in read(source):
        pairs=sorted({tuple(lookups[s][p[s+'_url']] for s in SIDES) for p in g['valid_pairs']})
        base=dict(group_id=g['group_id'],family_id=g['semantic_family_id'],query=g['query'])
        validation.append(dict(**base,valid_pairs=pairs))
        service_validation.append(dict(**base,valid_pairs=sorted({tuple(corpus[s][p[i]]['service'] for i,s in enumerate(SIDES)) for p in pairs})))
    sets={name:{g['family_id'] for g in data} for name,data in [('train',groups),('aux',validation),('dev',read(out/'function'/'dev.json'))]}
    assert not sets['train']&sets['aux'] and not sets['aux']&sets['dev']
    write(out/'function'/'auxiliary_validation.json',validation)
    write(out/'service_only'/'auxiliary_validation.json',service_validation)
    manifest=read(out/'manifest.json')
    manifest['previous_manifest_sha256']=sha(previous/'manifest.json')
    manifest['auxiliary_validation']={'source_sha256':sha(source),'rows':len(validation),'family_overlap':0,'history':'existing reranker_train split; not untouched confirmation'}
    manifest['artifacts']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',manifest)
    write(root/'configs.json',{name:configuration(name) for name in RUNS})
    # Exact H6 document-gradient workload, not an assumed full-catalog batch cost.
    order=list(range(len(groups)));random.Random(42).shuffle(order)
    workload={}
    for ix,s in enumerate(SIDES):
        sizes=Counter(r['service'] for r in corpus[s]);counts=[]
        for start in range(0,len(order),16):
            services={corpus[s][p[ix]]['service'] for i in order[start:start+16] for p in groups[i]['valid_pairs']}
            counts.append(sum(sizes[t] for t in services))
        workload[s]={'mean_documents_per_batch':sum(counts)/len(counts),'max':max(counts)}
    write(root/'DATA_READY.json',dict(training_groups=len(groups),auxiliary_rows=len(validation),
                                    family_overlap=0,h6_workload=workload,manifest_sha256=sha(out/'manifest.json')))
    print('DATA_READY',len(groups),len(validation),json.dumps(workload))


if __name__=='__main__':main()
