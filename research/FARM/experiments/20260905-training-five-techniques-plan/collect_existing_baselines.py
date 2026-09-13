#!/usr/bin/env python3
"""Read completed DGX dev predictions; export aggregates only, no inference."""
import json
import subprocess
from pathlib import Path

REMOTE_CODE = r'''
import collections, datetime, hashlib, json, pathlib, urllib.parse
root = pathlib.Path('/raid/session/aicontents/farm/experiments/20260831T171222Z-farm-four-pipelines')
out = {'scope': 'existing full development predictions only; no new inference',
       'collected_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'baselines': {}}
def service(endpoint):
    parts = urllib.parse.urlparse(endpoint).path.strip('/').split('/')
    assert len(parts) >= 3, endpoint
    return parts[0]
for arm in ('e0','e1','e2','e3'):
    path = root / 'results' / f'{arm}_dev.json'
    raw = path.read_bytes()
    result = json.loads(raw)
    joint = result['metrics']['joint']
    out['baselines'][arm] = {
        'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'task_level': result['task_level'], 'rows': result['rows'],
        'joint_exact_top1': joint['R@1'],
        'joint_exact_top1_count': sum(row['hits']['R@1'] for row in result['per_sample']),
        'valid_pair_in_top5_per_side': joint['R@5'],
        'valid_pair_in_top10_per_side': joint['R@10'],
    }
    if arm != 'e3':
        continue
    counts = collections.Counter()
    for row in result['per_sample']:
        pairs = {tuple(pair) for pair in row['valid_pairs']}
        predicted = (row['top_trigger_ids'][0], row['top_action_ids'][0])
        service_pairs = {(service(t), service(a)) for t,a in pairs}
        exact = predicted in pairs
        service_correct = tuple(map(service, predicted)) in service_pairs
        counts['function_pair_correct' if exact else 'function_pair_wrong'] += 1
        counts['projected_service_pair_correct' if service_correct else 'projected_service_pair_wrong'] += 1
        counts['service_pair_correct_but_function_pair_wrong'] += int(service_correct and not exact)
        counts['wrong_top1_with_valid_pair_in_top10_lattice'] += int(not exact and row['hits']['R@10'])
        counts['valid_pair_missing_from_top10_lattice'] += int(not row['hits']['R@10'])
        for i, side in enumerate(('trigger','action')):
            valid_ids = {p[i] for p in pairs}
            side_correct = predicted[i] in valid_ids
            svc_correct = service(predicted[i]) in {service(x) for x in valid_ids}
            counts[f'{side}_function_wrong'] += int(not side_correct)
            counts[f'{side}_service_correct_but_function_wrong'] += int(svc_correct and not side_correct)
    assert counts['function_pair_correct'] + counts['function_pair_wrong'] == result['rows']
    assert counts['function_pair_wrong'] == counts['wrong_top1_with_valid_pair_in_top10_lattice'] + counts['valid_pair_missing_from_top10_lattice']
    out['e3_error_decomposition'] = dict(counts)
print(json.dumps(out, indent=2, sort_keys=True))
'''

if __name__ == '__main__':
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'aicontents', 'python3', '-'],
        input=REMOTE_CODE, text=True, capture_output=True, check=True,
    )
    result = json.loads(completed.stdout)
    destination = Path(__file__).with_name('EXISTING_BASELINES.json')
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
