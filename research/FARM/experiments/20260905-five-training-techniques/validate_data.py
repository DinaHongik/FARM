#!/usr/bin/env python3
"""Exhaustive feature-isolation, label, split and mined-negative validation."""
from pathlib import Path
from prepare import SIDES,read,write,sha


def main():
    root=Path(__file__).resolve().parent
    derived=root/'derived';manifest=read(derived/'manifest.json')
    for rel,digest in manifest['artifacts'].items():
        assert sha(derived/rel)==digest,rel
    services=read(derived/'service_only'/'corpus.json')
    corpus=read(derived/'function'/'corpus.json')
    for side in SIDES:
        assert all(set(s)=={'id','name'} for s in services[side])
    function=read(derived/'function'/'train.json')
    service=read(derived/'service_only'/'train.json')
    assert len(function)==len(service)==8018
    for f,s in zip(function,service):
        assert set(s)=={'group_id','family_id','query','valid_pairs'}
        assert (f['group_id'],f['query'],f['family_id'])==(s['group_id'],s['query'],s['family_id'])
        assert {tuple(p) for p in s['valid_pairs']}=={
            tuple(corpus[side][p[i]]['service'] for i,side in enumerate(SIDES)) for p in f['valid_pairs']}
    train_families={g['family_id'] for g in function}
    assert not train_families.intersection(g['family_id'] for g in read(derived/'function'/'dev.json'))
    result={'status':'passed','training_groups':len(function),'feature_allowlist':'passed',
            'family_separation':'passed','hash_validation':'passed','sides':{}}
    for i,side in enumerate(SIDES):
        pools=read(derived/'function'/('negatives_'+side+'.json'))
        assert len(pools)==len(function)
        changing=0
        for group,pool in zip(function,pools):
            positives={p[i] for p in group['valid_pairs']}
            aliases=set().union(*(set(corpus[side][p]['equivalent_indices']) for p in positives))
            texts={corpus[side][p]['text'] for p in positives}
            for strategy in ('global','within'):
                assert len(set(pool[strategy]))==len(pool[strategy])==4
                assert not set(pool[strategy]).intersection(aliases)
                assert not texts.intersection(corpus[side][n]['text'] for n in pool[strategy])
            changing+=pool['global']!=pool['within']
            positive_services={corpus[side][p]['service'] for p in positives}
            assert all(corpus[side][n]['service'] in positive_services for n in pool['within'][:pool['sibling_count']])
        result['sides'][side]={'negative_pools':len(pools),'curriculum_changes_pool':changing}
    write(root/'data_validation.json',result)
    print(result)


if __name__=='__main__':main()
