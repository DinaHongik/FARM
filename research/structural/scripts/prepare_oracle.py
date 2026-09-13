"""Prepare the explicit oracle-endpoint configuration condition, never retrieval."""
import argparse
from pathlib import Path
from common import ROOT, freeze, read, rows, save_rows, sha


def prepare(farm, output):
    old = farm/'experiments/20260904T170552Z-farm-round9-agentic-benchmarks'
    cases = rows(old/'prepared/farm_v2_test/cases.jsonl')
    assert len(cases) == 150
    sidecars = farm/'experiments/20260905T0848KST-farm-architecture-research/results/schema_sidecar_v1'
    catalogs, evidence = {}, {}
    for side in ['trigger', 'action']:
        catalogs[side] = {}
        for item in read(farm/f'data/v2/corpus/{side}s.json'):
            for url in [item['url'], *item.get('equivalent_urls', [])]:
                catalogs[side][url] = item
        evidence[side] = {v['endpoint_id']: v for v in rows(sidecars/f'{side}s.jsonl')}
    inputs = []
    for case in cases:
        pairs = case['private_gold']['valid_pairs']
        pair = min(pairs, key=lambda v: (v['trigger_url'], v['action_url']))
        endpoints = {}
        for side in ['trigger', 'action']:
            item = catalogs[side][pair[side+'_url']]
            meta = evidence[side][item['url']]
            endpoints[side] = {
                'endpoint_id': item['url'], 'side': side,
                'service': item['channel_display'], 'function': item['function_name'],
                'description': item['description'], 'schema_revision': sha(sidecars/f'{side}s.jsonl'),
                'schema_unreconciled': bool(meta.get('source_only_input_fields')),
                'fields': [{'slug': f['slug'], 'label': f.get('label', ''),
                            'help_text': f.get('help_text', ''),
                            'required': f['required'] if f['requiredness_state']=='known' else None,
                            'value_type': f['type'] if f['type_state']=='known' else None,
                            'bindable': False, 'binding_capability_known': False}
                           for f in meta['input_fields']],
                'ingredients': [{'slug': i['slug'], 'label': i.get('label', ''),
                                 'value_type': i['type'] if i['type_state']=='known' else None}
                                for i in meta['ingredients']],
            }
        inputs.append({'case_id': case['case_id'], 'query': case['input']['query'], 'endpoints': endpoints})
    manifest = {'n': 150, 'condition': 'oracle_endpoints',
                'case_source_sha256': sha(old/'prepared/farm_v2_test/cases.jsonl'),
                'sidecar_sha256': {s: sha(sidecars/f'{s}s.jsonl') for s in ['trigger', 'action']},
                'selection_rule': 'lexicographically first annotated valid pair',
                'protocol_sha256': sha(ROOT/'PROTOCOL.md'),
                'binding_gold_available': False}
    freeze(output/'oracle_manifest.json', manifest)
    path = output/'oracle_inputs.jsonl'
    if path.exists():
        assert rows(path) == inputs
    else:
        save_rows(path, inputs)
    print('Prepared 150 oracle-endpoint inputs, with no binding answers or source examples in prompts.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--farm-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'private')
    args = p.parse_args()
    prepare(args.farm_root.resolve(), args.output.resolve())
