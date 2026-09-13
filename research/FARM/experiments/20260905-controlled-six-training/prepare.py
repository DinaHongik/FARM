#!/usr/bin/env python3
"""Prepare group-preserving service/function inputs and frozen training-only negatives."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

BASE = 'google/embeddinggemma-300m'
REVISION = '57c266a740f537b4dc058e1b0cda161fd15afa75'
SIDES = ('trigger', 'action')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(data, out):
    manifest = read(data / 'manifest.json')
    selected = ['splits/encoder_train.json', 'splits/dev.json',
                'corpus/triggers.json', 'corpus/actions.json']
    for name in selected:
        assert sha(data / name) == manifest['artifacts'][name]['sha256'], name
    source = {s: read(data / 'corpus' / (s + 's.json')) for s in SIDES}
    services, corpus, lookup = {}, {}, {}
    for side in SIDES:
        names = {}
        for row in source[side]:
            names.setdefault(row['channel'], []).append(row['channel_display'])
        services[side] = [{'id': key, 'name': sorted(Counter(values),
                            key=lambda n: (-values.count(n), n))[0]}
                          for key, values in sorted(names.items())]
        service_index = {row['id']: i for i, row in enumerate(services[side])}
        corpus[side] = [{'id': row['url'], 'text': row['text_schema'],
                         'service': service_index[row['channel']],
                         'aliases': sorted(set(row['equivalent_urls'] + [row['url']]))}
                        for row in source[side]]
        lookup[side] = {row['id']: i for i, row in enumerate(corpus[side])}
        alias_members = {}
        for i, row in enumerate(corpus[side]):
            for alias in row['aliases']:
                alias_members.setdefault(alias, set()).add(i)
                if alias not in lookup[side]:
                    lookup[side][alias] = i
        for row in corpus[side]:
            row['equivalent_indices'] = sorted(set().union(*(alias_members[a] for a in row['aliases'])))
    splits = {}
    for split in ('encoder_train', 'dev'):
        groups, service_groups = [], []
        for group in read(data / 'splits' / (split + '.json')):
            pairs = sorted({tuple(lookup[s][p[s + '_url']] for s in SIDES)
                            for p in group['valid_pairs']})
            assert pairs
            base = {'group_id': group['group_id'], 'family_id': group['semantic_family_id'],
                    'query': group['query']}
            groups.append({**base, 'valid_pairs': pairs})
            service_pairs = sorted({tuple(corpus[s][pair[i]]['service']
                                           for i, s in enumerate(SIDES)) for pair in pairs})
            service_groups.append({**base, 'valid_pairs': service_pairs})
        name = 'train' if split == 'encoder_train' else split
        write(out / 'function' / (name + '.json'), groups)
        write(out / 'service_only' / (name + '.json'), service_groups)
        splits[name] = groups
    assert not ({g['family_id'] for g in splits['train']} &
                {g['family_id'] for g in splits['dev']})
    write(out / 'function' / 'corpus.json', corpus)
    write(out / 'service_only' / 'corpus.json', services)
    return corpus, services, splits['train'], manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    corpus, services, groups, source_manifest = export(args.data_root, args.out)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(BASE, revision=REVISION, device='cuda:0',
                               model_kwargs={'torch_dtype': torch.float32}, local_files_only=True)
    model.max_seq_length = 512
    model.eval()
    assert set(('query','document')).issubset(model.prompts)
    prototypes = {}
    queries = model.encode_query([g['query'] for g in groups], batch_size=64,
                                 normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True)
    for side_index, side in enumerate(SIDES):
        prototypes[side] = model.encode_document([s['name'] for s in services[side]],
                           batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                           convert_to_tensor=True).cpu()
        docs = model.encode_document([r['text'] for r in corpus[side]], batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True)
        assert np.isfinite(docs).all() and np.isfinite(queries).all()
        pools, fallback_groups = [], 0
        for start in range(0, len(groups), 128):
            scores = queries[start:start+128] @ docs.T
            for j, group in enumerate(groups[start:start+128]):
                positive = {p[side_index] for p in group['valid_pairs']}
                forbidden = set().union(*(set(corpus[side][p]['equivalent_indices']) for p in positive))
                positive_text = {corpus[side][p]['text'] for p in positive}
                positive_services = {corpus[side][p]['service'] for p in positive}
                order = np.argsort(-scores[j], kind='stable').tolist()
                legal = [i for i in order if i not in forbidden and
                         corpus[side][i]['text'] not in positive_text and scores[j,i] <= .98]
                assert len(legal) >= 4
                global_ids = legal[:4]
                siblings = [i for i in legal if corpus[side][i]['service'] in positive_services][:2]
                fallback_groups += len(siblings) < 2
                mixed = siblings + [i for i in legal if i not in siblings][:4-len(siblings)]
                pools.append({'global': global_ids, 'within': mixed,
                              'sibling_count': len(siblings)})
        write(args.out / 'function' / ('negatives_' + side + '.json'), pools)
        print(json.dumps({'phase': 'mined', 'side': side, 'groups': len(pools),
                          'sibling_fallback_groups': fallback_groups}), flush=True)
    torch.save(prototypes, args.out / 'service_only' / 'prototypes.pt')
    artifacts = {str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*')
                 if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')}
    write(args.out / 'manifest.json', {'base_model': BASE, 'revision': REVISION,
          'source_manifest_sha256': sha(args.data_root / 'manifest.json'),
          'dataset_id': source_manifest['dataset_id'], 'artifacts': artifacts,
          'prompts': model.prompts, 'groups': len(groups), 'negative_count': 4,
          'miner': 'pristine pretrained; encoder_train queries only; fixed .98 ceiling',
          'service_feature_policy': 'raw query and canonical service names/indices only'})
    print(json.dumps({'phase': 'prepared', 'groups': len(groups), 'artifacts': len(artifacts)}), flush=True)


if __name__ == '__main__':
    main()
