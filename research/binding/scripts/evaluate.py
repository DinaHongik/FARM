"""Independent controlled-reference evaluator; never used for generation/repair."""
import math
import re
from common import canonical


def query_value(query, source):
    a,b=source.get('start'),source.get('end')
    if type(a) is not int or type(b) is not int or not 0<=a<b<=len(query):
        raise ValueError('invalid_span')
    text=query[a:b]
    if text!=source.get('text'):raise ValueError('ungrounded_span')
    transform=source.get('transform')
    if transform=='identity':return text
    if transform=='parse_integer' and re.fullmatch(r'[+-]?\d+',text.strip()):return int(text)
    if transform=='parse_number' and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',text.strip()):
        number=float(text)
        if math.isfinite(number):return number
    if transform=='parse_boolean' and text.strip().lower() in ['true','false']:
        return text.strip().lower()=='true'
    raise ValueError('invalid_transform')


def source_option(case, side, field, source):
    if not isinstance(source,dict):raise ValueError('malformed_source')
    shapes={'query_span':{'kind','start','end','text','transform'},
            'trigger_output':{'kind','ingredient_slug'},'omit':{'kind'},'needs_input':{'kind','question'}}
    kind=source.get('kind')
    if not isinstance(kind,str) or kind not in shapes or set(source)!=shapes[kind]:
        raise ValueError('malformed_source')
    if kind=='query_span':
        value=query_value(case['query'],source)
        # Current fixture schema contains documented text controls only.
        if field['value_type']=='string' and type(value) is not str:raise ValueError('type_mismatch')
        return {'kind':'literal','value':value}
    if kind=='trigger_output':
        slug=source.get('ingredient_slug')
        if not isinstance(slug,str) or side!='action' or not field['bindable']:
            raise ValueError('unavailable_ingredient')
        if slug not in {i['slug'] for i in case['endpoints']['trigger']['ingredients']}:
            raise ValueError('unknown_ingredient')
        return {'kind':kind,'ingredient_slug':slug}
    if kind=='omit' and field['required'] is not False:raise ValueError('required_omitted')
    if kind=='needs_input' and (not isinstance(source.get('question'),str) or not source['question'].strip()):
        raise ValueError('empty_clarification')
    return {'kind':kind}


def inspect(case, draft):
    errors=[]; options={}; sources={}; duplicates=set()
    if not isinstance(draft,dict) or any(not isinstance(draft.get(s+'_fields'),list) for s in ['trigger','action']):
        return {},{},['invalid_draft_shape']
    for side,endpoint in case['endpoints'].items():
        fields={f['slug']:f for f in endpoint['fields']};seen=set()
        for decision in draft[side+'_fields']:
            if not isinstance(decision,dict) or set(decision)!={'field','source'} or not isinstance(decision.get('field'),str):
                errors.append(side+':malformed_decision');continue
            slug=decision['field'];key=(side,slug)
            if slug in seen:
                errors.append(side+':duplicate_field:'+slug);duplicates.add(key)
            seen.add(slug)
            if slug not in fields:
                errors.append(side+':unknown_field:'+slug);continue
            sources[key]=decision['source']
            try:options[key]=source_option(case,side,fields[slug],decision['source'])
            except (ValueError,TypeError,OverflowError) as exc:
                errors.append(side+':'+str(exc)+':'+slug)
        for slug in fields.keys()-seen:errors.append(side+':missing_field:'+slug)
    for key in duplicates:options.pop(key,None)
    return options,sources,sorted(set(errors))


def materialize(case, draft, event):
    options,_,errors=inspect(case,draft)
    if errors:raise ValueError('invalid_configuration')
    result={'trigger':{},'action':{}}
    for (side,slug),option in options.items():
        if option['kind']=='needs_input':raise ValueError('missing_information')
        if option['kind']=='omit':continue
        if option['kind']=='literal':value=option['value']
        else:
            ingredient=option['ingredient_slug']
            if ingredient not in event:raise ValueError('missing_event_ingredient')
            value=event[ingredient]
            if type(value) is not str:raise ValueError('invalid_event_type')
        result[side][slug]=value
    return result


def predicted_edges(draft):
    edges=set()
    if not isinstance(draft,dict) or not isinstance(draft.get('action_fields'),list):return edges
    for item in draft['action_fields']:
        if not isinstance(item,dict):continue
        source=item.get('source')
        if (isinstance(source,dict) and source.get('kind')=='trigger_output'
            and isinstance(item.get('field'),str) and isinstance(source.get('ingredient_slug'),str)):
            edges.add((item['field'],source['ingredient_slug']))
    return edges


def score(case,gold,draft):
    options,_,errors=inspect(case,draft)
    details=[]
    for reference in gold['fields']:
        key=reference['side'],reference['field'];actual=options.get(key)
        correct=actual is not None and canonical(actual) in {canonical(o) for o in reference['acceptable']}
        expected_missing=all(o['kind']=='needs_input' for o in reference['acceptable'])
        static=all(o['kind']=='literal' for o in reference['acceptable'])
        has_value=actual is not None and actual['kind'] in ['literal','trigger_output']
        details.append({'key':':'.join(key),'correct':correct,'required':reference['required'],
                        'static':static,'expected_missing':expected_missing,'has_value':has_value,
                        'unnecessary_abstention':actual is not None and actual['kind']=='needs_input' and not expected_missing})
    target={(r['field'],o['ingredient_slug']) for r in gold['fields'] if r['side']=='action'
            for o in r['acceptable'] if o['kind']=='trigger_output'}
    # Builder guarantees exactly one choice per field. Do not union alternatives.
    assert all(len(r['acceptable'])==1 for r in gold['fields'])
    proposed=predicted_edges(draft)
    tp=len(target&proposed);fp=len(proposed-target);fn=len(target-proposed)
    fixture_passes=0
    for fixture in gold['fixtures']:
        try:passed=canonical(materialize(case,draft,fixture['event']))==canonical(fixture['expected_arguments'])
        except (ValueError,KeyError,TypeError):passed=False
        fixture_passes+=passed
    whole=not errors and all(d['correct'] for d in details)
    all_bound=not errors and all(o['kind']!='needs_input' for o in options.values())
    return {'case_id':case['case_id'],'family_id':gold['family_id'],'category':gold['category'],
            'whole_correct':whole,'fields':len(details),'correct_fields':sum(d['correct'] for d in details),
            'binding_applicable':bool(target),'binding_tp':tp,'binding_fp':fp,'binding_fn':fn,
            'exact_binding_set':bool(target) and target==proposed and not errors,
            'static_fields':sum(d['static'] for d in details),
            'correct_static_fields':sum(d['static'] and d['correct'] for d in details),
            'required_fields':sum(d['required'] for d in details),
            'required_with_value':sum(d['required'] and d['has_value'] for d in details),
            'required_with_correct_value':sum(d['required'] and d['has_value'] and d['correct'] for d in details),
            'resolvable_required_fields':sum(d['required'] and not d['expected_missing'] for d in details),
            'resolvable_required_with_value':sum(d['required'] and not d['expected_missing'] and d['has_value'] for d in details),
            'expected_missing_fields':sum(d['expected_missing'] for d in details),
            'correct_missing_fields':sum(d['expected_missing'] and d['correct'] for d in details),
            'unnecessary_abstentions':sum(d['unnecessary_abstention'] for d in details),
            'declared_schema_valid_fully_bound':all_bound,
            'fully_specified_reference':gold['fully_specified'],
            'fixture_count':len(gold['fixtures']),'fixtures_passed':fixture_passes,
            'materialization_pass':bool(gold['fixtures']) and fixture_passes==len(gold['fixtures']),
            'errors':errors,'field_details':details}


def reference_draft(case,gold):
    """Test helper only; never imported by generation/repair code."""
    draft={'trigger_fields':[],'action_fields':[],'preview':'Test reference'}
    for reference in gold['fields']:
        option=reference['acceptable'][0]
        if option['kind']=='literal':
            value=option['value'];start=case['query'].index(value)
            source={'kind':'query_span','start':start,'end':start+len(value),'text':value,'transform':'identity'}
        elif option['kind']=='needs_input':source={'kind':'needs_input','question':'What is the missing setting?'}
        else:source=dict(option)
        draft[reference['side']+'_fields'].append({'field':reference['field'],'source':source})
    return draft
