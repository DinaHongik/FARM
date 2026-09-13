"""Exact semantic-reference scoring. No LLM judge and no checker-as-gold."""
import math
from common import canonical

SCORABLE = {'closed', 'missing_context'}

def validate_scoring_reference(case, reference):
    from reference import validate_reference
    errors = validate_reference(case, reference)
    if errors: return errors
    for field in reference['fields']:
        options = [token(o) for o in field.get('acceptable', [])]
        if any(o is None for o in options): errors.append('unscorable_literal_or_option')
        if field['status'] == 'missing_context' and set(options) != {('needs_input',)}:
            errors.append('missing_context_must_identify_missing_input')
    if reference['binding_sets_exhaustive']:
        action = [f for f in reference['fields'] if f['side']=='action']
        if not reference['binding_sets'] or any(f['status'] not in SCORABLE for f in action):
            errors.append('binding_exhaustiveness_conflicts_with_field_scope')
        for alternative in reference['binding_sets']:
            mapping = {e['field']: e['ingredient_slug'] for e in alternative}
            for f in action:
                options = {token(o) for o in f.get('acceptable', [])}
                if f['field'] in mapping:
                    if ('trigger_output', mapping[f['field']]) not in options:
                        errors.append('binding_set_disagrees_with_field_options')
                elif options and all(o and o[0]=='trigger_output' for o in options):
                    errors.append('binding_set_omits_required_dynamic_choice')
    return sorted(set(errors))

def normalized_value(source, query):
    a, b = source.get('start'), source.get('end')
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(query):
        return None
    text = query[a:b]
    if text != source.get('text'):
        return None
    transform = source.get('transform')
    try:
        if transform == 'identity': value = text
        elif transform == 'parse_integer':
            import re
            if not re.fullmatch(r'[+-]?\d+', text.strip()): return None
            value = int(text.strip())
        elif transform == 'parse_number':
            import re
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', text.strip()): return None
            value = float(text.strip())
            if not math.isfinite(value): return None
        elif transform == 'parse_boolean':
            value = {'true': True, 'false': False}[text.strip().lower()]
        else: return None
    except (ValueError, OverflowError, KeyError):
        return None
    return {'kind': 'literal', 'value': value}

def token(option):
    if not isinstance(option, dict): return None
    kind = option.get('kind')
    if kind == 'literal' and 'value' in option:
        value = option['value']
        # Numeric int/float representations can agree; booleans and strings do
        # not silently become numbers or case-insensitive labels.
        if type(value) in (int, float):
            if not math.isfinite(value): return None
            return ('literal_number', value)
        if type(value) not in (str, bool): return None
        return ('literal', type(value).__name__, value)
    if kind == 'trigger_output' and isinstance(option.get('ingredient_slug'), str):
        return ('trigger_output', option['ingredient_slug'])
    if kind in {'omit', 'needs_input'}:
        return (kind,)
    return None

def decision_option(source, query):
    if not isinstance(source, dict): return None
    if source.get('kind') == 'query_span':
        return normalized_value(source, query)
    if source.get('kind') == 'needs_input' and not isinstance(source.get('question'), str):
        return None
    return source if source.get('kind') in {'trigger_output', 'omit', 'needs_input'} else None

def edges(draft):
    result = set()
    if not isinstance(draft, dict) or not isinstance(draft.get('action_fields'), list): return result
    for d in draft['action_fields']:
        if not isinstance(d, dict) or not isinstance(d.get('source'), dict): continue
        s = d['source']
        if s.get('kind') == 'trigger_output' and isinstance(d.get('field'), str) and isinstance(s.get('ingredient_slug'), str):
            result.add((d['field'], s['ingredient_slug']))
    return result

def binding_counts(predicted, alternatives):
    """Select one coherent reference, never a union of incompatible sets."""
    best = None
    for alternative in alternatives:
        target = {(e['field'], e['ingredient_slug']) for e in alternative}
        tp = len(predicted & target); fp = len(predicted - target); fn = len(target - predicted)
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else None
        candidate = {'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1,
                     'exact': predicted == target, 'has_edges': bool(predicted or target)}
        # Identical empty sets are exact but have no F1 contribution.
        key = (1.0 if f1 is None else f1, tp, -fp, -fn, canonical(alternative))
        if best is None or key > best[0]: best = key, candidate
    return best[1] if best else None

def score_case(case, reference, draft):
    wanted = {(s, f['slug']) for s, e in case['endpoints'].items() for f in e['fields']}
    parsed = isinstance(draft, dict) and all(isinstance(draft.get(s+'_fields'), list) for s in ['trigger', 'action'])
    assigned = {}; duplicates = set(); malformed = False
    if parsed:
        for side in ['trigger', 'action']:
            for d in draft[side+'_fields']:
                if not isinstance(d, dict) or not isinstance(d.get('field'), str):
                    malformed = True; continue
                key = side, d['field']
                if key in assigned: duplicates.add(key)
                assigned[key] = decision_option(d.get('source'), case['query'])
    field_results = []
    for f in reference['fields']:
        key = f['side'], f['field']
        if f['status'] not in SCORABLE: continue
        options = {token(v) for v in f['acceptable']}
        options.discard(None)
        observed = token(assigned.get(key))
        correct = parsed and key not in duplicates and observed is not None and observed in options
        expected_missing = options == {('needs_input',)}
        unnecessary_missing = observed == ('needs_input',) and ('needs_input',) not in options
        field_results.append({'key': ':'.join(key), 'correct': bool(correct),
                              'expected_missing': expected_missing,
                              'predicted_missing': observed == ('needs_input',),
                              'unnecessary_missing': unnecessary_missing})
    scorable = len(field_results)
    correct_n = sum(x['correct'] for x in field_results)
    dynamic_options = any(o.get('kind') == 'trigger_output'
                          for f in reference['fields'] for o in f.get('acceptable', []))
    whole_eligible = (len(wanted) > 0 and scorable == len(wanted)
                      and not reference.get('cross_field_constraints')
                      and (not dynamic_options or reference.get('binding_sets_exhaustive') is True))
    predicted_edges = edges(draft) if parsed else set()
    binding = None
    if reference.get('binding_sets_exhaustive') and reference.get('binding_sets'):
        binding = binding_counts(predicted_edges, reference['binding_sets'])
        # A malformed draft cannot be counted as an exact empty binding success.
        if binding is not None and not parsed:
            binding['exact'] = False
    whole_correct = (whole_eligible and parsed and not malformed and not duplicates
                     and set(assigned) == wanted and correct_n == scorable)
    if whole_correct and binding is not None:
        whole_correct = binding['exact']
    unresolved_reference = any(x['expected_missing'] for x in field_results)
    fully_bound_eligible = whole_eligible and not unresolved_reference
    return {'scorable_fields': scorable, 'correct_fields': correct_n,
            'field_results': field_results,
            'field_accuracy': correct_n/scorable if scorable else None,
            'whole_decision_eligible': whole_eligible,
            'whole_decision_correct': bool(whole_correct),
            'fully_bound_eligible': fully_bound_eligible,
            'fully_bound_correct': bool(fully_bound_eligible and whole_correct),
            'binding': binding, 'draft_shape_valid': parsed,
            'extra_fields': len(set(assigned) - wanted), 'duplicate_fields': len(duplicates)}
