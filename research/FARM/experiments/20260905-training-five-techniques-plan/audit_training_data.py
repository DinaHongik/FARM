"""Read-only training-data audit; writes aggregates, never private examples."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List


FARM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FARM))
from farm.dataset_v2 import query_key
from farm.render import parse_url


def read(relative):
    return json.loads((FARM / relative).read_text())


def collisions(rows, batch_size, epochs=3):
    """Counterfactual random-batch label collisions, NOT actual E3 sampler logs."""
    rng = random.Random(42)
    affected_batches = affected_rows = batches = exposures = 0
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [rows[i] for i in order[start:start + batch_size]]
            labels = [r['label_url'] for r in batch]
            affected = sum(any(j != i and label in set(row['valid_label_urls'])
                               for j, label in enumerate(labels))
                           for i, row in enumerate(batch))
            affected_rows += affected
            affected_batches += int(affected > 0)
            batches += 1
            exposures += len(batch)
    return {'batches': batches, 'batches_with_known_positive_collision': affected_batches,
            'row_exposures': exposures, 'row_exposures_with_collision': affected_rows,
            'scope': 'hypothetical shuffled MNRL batches; current NO_DUPLICATES is not simulated'}


def main():
    os.umask(0o077)
    manifest = read('data/v2/manifest.json')
    raw = read('data/iftttt_dataset_full_trigger_action.json')
    original = read('data/server_copy/train_applets.json')
    train = read('data/v2/splits/encoder_train.json')
    counts = manifest['counts']
    summary = {'scope': 'raw source and encoder-training data only; no final-test payloads opened',
               'source_sha256': hashlib.sha256((FARM / 'data/iftttt_dataset_full_trigger_action.json').read_bytes()).hexdigest(),
               'v2_manifest_sha256': hashlib.sha256((FARM / 'data/v2/manifest.json').read_bytes()).hexdigest(),
               'v1': {'raw_service_records': len(raw), 'training_rows': len(original)},
               'v2': {'query_groups': counts['query_groups'], 'train_groups': len(train),
                      'split_group_counts_from_manifest': counts['split_groups'],
                      'multi_gold_train_groups': sum(len(r['valid_pairs']) > 1 for r in train)},
               'sides': {}}
    original_queries = defaultdict(set)
    applets = Counter()
    for row in original:
        original_queries[query_key(row.get('query', ''))].add((row['trigger']['url'], row['action']['url']))
        applets[row.get('applet_url', '')] += 1
    summary['v1'].update({'unique_normalized_training_queries': len(original_queries),
                          'normalized_queries_with_multiple_observed_pairs': sum(len(v) > 1 for v in original_queries.values()),
                          'unique_applet_urls': len(applets),
                          'duplicate_applet_row_excess': sum(v - 1 for v in applets.values())})

    # Load only pure legacy render functions; no model/index/network imports.
    source = ast.parse((FARM / 'rag/indexer.py').read_text())
    names = {'extract_channel_from_filter_code', 'extract_trigger_text', 'extract_action_text'}
    module = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {'Dict': Dict, 'Any': Any, 'List': List}
    exec(compile(ast.fix_missing_locations(module), '<legacy-renderers>', 'exec'), namespace)

    for side, filename in [('trigger', 'triggers'), ('action', 'actions')]:
        legacy_catalog = read(f'data/{filename}_rag.json')
        renderer = namespace[f'extract_{side}_text']
        indexed = {renderer(row) for row in legacy_catalog}
        corpus = read(f'data/v2/corpus/{filename}.json')
        by_url = {r['url']: r for r in corpus}
        pairs = read(f'data/v2/pairs/{side}_encoder_train_schema.json')
        freq = Counter()
        service_freq = Counter()
        same_service_options = []
        service_sizes = Counter(r['channel'] for r in corpus)
        for group in train:
            valid = {p[f'{side}_url'] for p in group['valid_pairs']}
            freq.update(valid)
            services = {by_url[url]['channel'] for url in valid}
            service_freq.update(services)
            alternatives = {r['url'] for r in corpus if r['channel'] in services} - valid
            same_service_options.append(len(alternatives))
        alias_multi = sum(len(r.get('equivalent_urls', [])) > 1 for r in corpus)
        summary['sides'][side] = {
            'legacy_catalog_rows': len(legacy_catalog),
            'legacy_positive_matches_any_indexed_text': sum(renderer(r[side]) in indexed for r in original),
            'legacy_positive_total': len(original),
            'legacy_components_missing_category': sum('category' not in r[side] for r in original),
            'corpus_functions': len(corpus), 'corpus_services': len(service_sizes),
            'seen_training_functions': len(freq), 'seen_training_services': len(service_freq),
            'singleton_training_functions': sum(v == 1 for v in freq.values()),
            'training_function_memberships': sum(freq.values()),
            'top10_function_memberships': sum(v for _, v in freq.most_common(10)),
            'v2_positive_rows': len(pairs),
            'v2_byte_identical_positive_rows': sum(r['positive'] == by_url[r['label_url']]['text_schema'] for r in pairs),
            'rows_with_multiple_valid_side_labels': sum(len(r['valid_label_urls']) > 1 for r in pairs),
            'groups_with_same_service_alternatives': sum(v > 0 for v in same_service_options),
            'median_same_service_alternatives': sorted(same_service_options)[len(same_service_options)//2],
            'functions_with_multiple_declared_equivalent_urls': alias_multi,
            'random_batch16_collision_diagnostic': collisions(pairs, 16),
        }

    # Audit label shape, not public examples or predictions.
    public_train = read('Targe/data/dataset/train_recipe.json')
    parsed, objects, keys, invalid = 0, 0, Counter(), 0
    for row in public_train:
        output = row.get('output')
        try:
            value = json.loads(output) if isinstance(output, str) else output
            parsed += 1
            if isinstance(value, dict):
                objects += 1
                keys.update(value.keys())
        except (json.JSONDecodeError, TypeError):
            invalid += 1
    summary['public_train_label_shape'] = {'rows': len(public_train), 'json_parseable_outputs': parsed,
                                          'json_object_outputs': objects, 'top_level_key_counts': dict(keys),
                                          'non_json_outputs': invalid}
    output = Path(sys.argv[1])
    with output.open('x') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
