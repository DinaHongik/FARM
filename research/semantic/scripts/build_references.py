"""Build blinded consensus references and retain every disagreement for review."""
from collections import Counter
from common import ROOT, read, save, freeze, digest, canonical
from reference import validate_reference
from scoring import validate_scoring_reference
from evidence import normalize_quotes

def option_key(field):
    return canonical({'status': field.get('status'),
                      'acceptable': sorted(field.get('acceptable', []), key=canonical)})

if __name__ == '__main__':
    cases = read(ROOT / 'private/ANNOTATION_INPUTS.json')
    references = []; packets = []; counts = Counter()
    for case in cases:
        proposals = {}
        for label in ['A', 'B']:
            path = ROOT / 'private/proposals' / label / (digest(case['case_id']) + '.json')
            if not path.exists():
                raise RuntimeError('Both reference passes must finish before consensus scoring')
            item = read(path)
            assert item['input_sha256'] == digest(case)
            normalized, changes = normalize_quotes(case, item['reference'])
            item['original_validation_errors'] = item['validation_errors']
            item['reference'] = normalized
            item['verified_excerpt_normalizations'] = changes
            item['validation_errors'] = validate_scoring_reference(case, normalized)
            counts['formatting_normalizations'] += len(changes)
            proposals[label] = item
        refs = [proposals[label]['reference'] for label in ['A', 'B']]
        valid = [not validate_scoring_reference(case, ref) for ref in refs]
        maps = [{(f['side'], f['field']): f for f in ref.get('fields', []) if isinstance(f, dict) and f.get('side') in ['trigger','action'] and isinstance(f.get('field'),str)}
                if isinstance(ref, dict) else {} for ref in refs]
        fields = []
        for side, endpoint in case['endpoints'].items():
            for f in endpoint['fields']:
                key = side, f['slug']
                same = all(valid) and key in maps[0] and key in maps[1] and option_key(maps[0][key]) == option_key(maps[1][key])
                if same:
                    value = dict(maps[0][key])
                    value['rationale'] = 'Two blinded AI proposals agree. Human review remains required. ' + value.get('rationale', '')
                    counts['agreed_fields'] += 1
                    if value['status'] in ['closed', 'missing_context']: counts['provisionally_scorable_fields'] += 1
                else:
                    value = {'side': side, 'field': f['slug'], 'status': 'unjudgeable', 'acceptable': [],
                             'rationale': 'Annotator disagreement or invalid reference; resolve during review.', 'evidence_quotes': []}
                    counts['fields_requiring_resolution'] += 1
                fields.append(value)
        same_sets = (all(valid) and all(f['status'] in ['closed','missing_context'] for f in fields if f['side']=='action')
                     and refs[0].get('binding_sets_exhaustive') is True
                     and refs[1].get('binding_sets_exhaustive') is True
                     and canonical(sorted([sorted(s, key=canonical) for s in refs[0]['binding_sets']], key=canonical))
                     == canonical(sorted([sorted(s, key=canonical) for s in refs[1]['binding_sets']], key=canonical)))
        consensus = {'fields': fields, 'binding_sets': refs[0]['binding_sets'] if same_sets else [],
                     'binding_sets_exhaustive': bool(same_sets),
                     'cross_field_constraints': sorted({v if isinstance(v,str) else canonical(v) for ref in refs if isinstance(ref, dict)
                                                        for v in ref.get('cross_field_constraints', [])}),
                     'notes': 'AI consensus subset only. Agreement does not establish correctness.'}
        if not all(valid): counts['cases_with_invalid_proposal'] += 1
        if same_sets: counts['agreed_exhaustive_binding_cases'] += 1
        item = {'case_id': case['case_id'], 'case_number': case['case_number'],
                'reference_state': 'ai_consensus_provisional', 'human_review': None,
                'input_sha256': digest(case), 'reference': consensus}
        assert not validate_scoring_reference(case, consensus)
        references.append(item)
        packets.append({'case': case, 'proposals': proposals, 'consensus': item})
    freeze(ROOT / 'private/AI_CONSENSUS.json', references)
    save(ROOT / 'private/REVIEW_PACKETS.json', packets)
    summary = {'requests': len(cases), 'fields': sum(len(x['reference']['fields']) for x in references),
               **dict(counts), 'human_reviewed_cases': 0, 'mode': 'ai_assisted_provisional'}
    save(ROOT / 'results/REFERENCE_PREPARATION.json', summary)
    print(canonical(summary))
