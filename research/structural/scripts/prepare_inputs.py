"""Freeze the previously exposed 150 cases, separating private scoring gold."""
import argparse
from pathlib import Path
from common import ROOT, freeze, read, rows, save_rows, sha


def prepare(farm, output):
    old = farm / 'experiments/20260904T170552Z-farm-round9-agentic-benchmarks'
    case_path = old / 'prepared/farm_v2_test/cases.jsonl'
    sample_path = old / 'manifests/samples/farm_v2_test.json'
    cases, sample = rows(case_path), read(sample_path)
    assert len(cases) == sample['sample_size'] == 150
    assert [c['case_id'] for c in cases] == sample['ordered_case_ids']
    split = read(farm / 'data/v2/splits/test.json')
    by_id = {g['group_id']: g for g in split}
    assert all(c['case_id'] == 'farm:' + c['audit']['group_id'] for c in cases)
    assert all(c['audit']['group_id'] in by_id for c in cases)
    inputs = [{'case_id': c['case_id'], 'query': c['input']['query']} for c in cases]
    gold = [{'case_id': c['case_id'], 'family_id': by_id[c['audit']['group_id']]['semantic_family_id'],
             'gold': c['private_gold'], 'binding_reference': None} for c in cases]
    identity = {'n': 150, 'source_cases_sha256': sha(case_path),
                'source_sample_sha256': sha(sample_path), 'source_split_sha256': sha(farm/'data/v2/splits/test.json'),
                'protocol_sha256': sha(ROOT/'PROTOCOL.md'),
                'population': 'previously exposed Round 9 FARM-v2 cases',
                'semantic_binding_gold_available': False}
    freeze(output/'input_manifest.json', identity)
    for name, values in [('inputs.jsonl', inputs), ('gold.jsonl', gold)]:
        path = output/name
        if path.exists():
            assert rows(path) == values
        else:
            save_rows(path, values)
    print('Frozen 150 inputs; scoring gold stored separately. No model calls.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--farm-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'private')
    args = p.parse_args()
    prepare(args.farm_root.resolve(), args.output.resolve())
