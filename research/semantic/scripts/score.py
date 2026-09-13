"""Score frozen predictions only after reference creation; default requires human review."""
import argparse
from pathlib import Path
import numpy as np
from common import ROOT, REVISION, read, save, sha, digest, canonical
from reference import validate_reference
from scoring import score_case, validate_scoring_reference

ARMS = ['single_agent', 'farm_feedback', 'farm_no_feedback']

def aggregate(records):
    fields = sum(r['scorable_fields'] for r in records)
    correct = sum(r['correct_fields'] for r in records)
    judged = [r for r in records if r['scorable_fields']]
    binding = [r['binding'] for r in records if r['binding'] is not None]
    tp, fp, fn = [sum(b[k] for b in binding) for k in ['tp', 'fp', 'fn']]
    decisions = [d for r in records for d in r['field_results']]
    whole_n = sum(r['whole_decision_eligible'] for r in records)
    whole_correct = sum(r['whole_decision_correct'] for r in records)
    bound_n = sum(r['fully_bound_eligible'] for r in records)
    bound_correct = sum(r['fully_bound_correct'] for r in records)
    def ratio(n,d): return n/d if d else None
    return {'requests': len(records), 'requests_with_scorable_fields': len(judged),
            'scorable_fields': fields, 'correct_field_decisions': correct,
            'field_decision_agreement_micro': ratio(correct, fields),
            'field_decision_agreement_macro': float(np.mean([r['field_accuracy'] for r in judged])) if judged else None,
            'binding_reference_cases': len(binding), 'binding_cases_with_edges': sum(b['has_edges'] for b in binding),
            'binding_tp': tp, 'binding_fp': fp, 'binding_fn': fn,
            'binding_precision': ratio(tp, tp+fp), 'binding_recall': ratio(tp,tp+fn),
            'binding_f1': ratio(2*tp,2*tp+fp+fn),
            'whole_decision_cases': whole_n, 'whole_decision_correct': whole_correct,
            'whole_decision_agreement': ratio(whole_correct, whole_n),
            'fully_bound_reference_cases': bound_n, 'fully_bound_correct': bound_correct,
            'fully_bound_agreement': ratio(bound_correct, bound_n),
            'required_missing_decisions': sum(d['expected_missing'] for d in decisions),
            'correct_missing_decisions': sum(d['expected_missing'] and d['correct'] for d in decisions),
            'unnecessary_missing_decisions': sum(d['unnecessary_missing'] for d in decisions),
            'invalid_draft_shapes': sum(not r['draft_shape_valid'] for r in records)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['human_reviewed','ai_pilot'], default='human_reviewed')
    parser.add_argument('--references', type=Path)
    args = parser.parse_args()
    path = args.references or ROOT / 'private' / ('HUMAN_REVIEWED.json' if args.mode=='human_reviewed' else 'AI_CONSENSUS.json')
    if not path.exists():
        raise SystemExit('No reviewed reference file yet. Complete and export the blinded review first; AI pilot scoring must be explicitly requested with --mode ai_pilot.')
    references = read(path)
    cases = read(ROOT / 'private/ANNOTATION_INPUTS.json')
    by_id = {r['case_id']: r for r in references}
    assert len(by_id) == len(references) == len(cases) == 150
    assert set(by_id) == {c['case_id'] for c in cases}
    for case in cases:
        item = by_id[case['case_id']]
        assert item['input_sha256'] == digest(case)
        errors = validate_scoring_reference(case, item['reference'])
        if errors: raise RuntimeError('Invalid reference; review before scoring: ' + str(case['case_number']))
        if args.mode == 'human_reviewed':
            review = item.get('human_review') or {}
            if (item.get('reference_state') != 'human_reviewed' or not review.get('reviewer_name','').strip()
                or review.get('confirmed_all_fields') is not True or not review.get('reviewed_at_utc')):
                raise SystemExit('Human review is incomplete. Model proposals cannot be reported as human-reviewed gold.')
    frozen = read(ROOT / 'private/PREDICTION_FREEZE.json')
    predictions = REVISION / 'agentic_validation/private/full_run/records'
    assert len(frozen) == 450
    for relative, checksum in frozen.items():
        assert sha(predictions / relative) == checksum, 'Frozen prediction changed'
    per_arm = {arm: [] for arm in ARMS}; initial = {arm: [] for arm in ARMS}
    for case in cases:
        ref = by_id[case['case_id']]['reference']
        for arm in ARMS:
            record = read(predictions / arm / (digest(case['case_id']) + '.json'))
            per_arm[arm].append(score_case(case, ref, record['final']))
            initial[arm].append(score_case(case, ref, record['initial']))
    results = {'status': 'complete', 'mode': args.mode, 'requests': 150, 'population_fields': 558,
               'condition': 'supplied_correct_endpoints', 'reference_sha256': sha(path),
               'prediction_freeze_sha256': sha(ROOT / 'private/PREDICTION_FREEZE.json'),
               'human_reviewed_cases': 150 if args.mode == 'human_reviewed' else 0,
               'semantic_binding_accuracy_claim_permitted': args.mode == 'human_reviewed',
               'execution_success': None, 'arms': {}, 'paired': {}}
    for arm in ARMS:
        results['arms'][arm] = aggregate(per_arm[arm])
        before, after = initial[arm], per_arm[arm]
        results['arms'][arm]['repair'] = {
            'field_decision_net_gain': sum(a['correct_fields']-b['correct_fields'] for b,a in zip(before,after)),
            'whole_decision_rescues': sum(b['whole_decision_eligible'] and not b['whole_decision_correct'] and a['whole_decision_correct'] for b,a in zip(before,after)),
            'whole_decision_regressions': sum(b['whole_decision_correct'] and not a['whole_decision_correct'] for b,a in zip(before,after)),
        }
    counts = np.array([r['scorable_fields'] for r in per_arm['farm_feedback']], dtype=float)
    for other in ['single_agent', 'farm_no_feedback']:
        delta = np.array([a['correct_fields']-b['correct_fields'] for a,b in zip(per_arm['farm_feedback'], per_arm[other])], dtype=float)
        rng = np.random.default_rng(20260906)
        samples = []
        if counts.sum():
            for _ in range(10000):
                indices = rng.integers(0,150,150)
                denominator = counts[indices].sum()
                if denominator: samples.append(100*delta[indices].sum()/denominator)
        results['paired']['farm_feedback_vs_'+other] = {
            'field_agreement_delta_pp': 100*float(delta.sum()/counts.sum()) if counts.sum() else None,
            'case_bootstrap_95_pp': np.quantile(samples,[.025,.975]).tolist() if samples else None,
            'scope': 'Fixed reference set; does not capture annotation error or judge bias.',
        }
    results['interpretation'] = ('Agreement with a blinded AI-consensus subset, not human-verified semantic accuracy. '
        'Judgeable fields are a selected subset. Unresolved/open-ended targets remain excluded and counted.'
        if args.mode=='ai_pilot' else 'Agreement with human-reviewed AI-assisted references. '
        'Not an independently double-human-annotated dataset; execution is unmeasured.')
    stem = 'AI_PILOT_RESULTS' if args.mode=='ai_pilot' else 'HUMAN_REVIEWED_RESULTS'
    save(ROOT / 'results' / (stem+'.json'), results)
    save(ROOT / 'private' / (stem+'_case_scores.json'), {'final':per_arm,'initial':initial})
    print(canonical(results))
