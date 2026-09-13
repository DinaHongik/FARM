"""Build controlled requests and explicit hidden references before inference."""
import copy
import random
import re
from collections import Counter
from common import ROOT, read, freeze, sha, digest, save_rows

VERSION = 'controlled-binding-v1'
SEED = 20260906
BARE_TRIGGERS = [
    'space/new_astronomy_picture_nasa', 'flickr/any_new_public_photo',
    'instagram/any_new_photo_by_you', 'facebook/new_status_message_by_you',
    'pinboard/new_bookmark_pb', 'youtube/new_liked_video',
    'android_messages/received_a_message', 'do_note/do_note_new_command_common',
    'medium/post_published_by_you', 'giphy/trending',
]
SETUP_TRIGGERS = [
    ('instagram/new_photo_by_you_tagged', 'research'),
    ('android_device/connect_to_wifi_network_with_ssid', 'Study WiFi'),
    ('soundcloud/new_track_from_search', 'instrumental piano'),
    ('twitter/new_tweet_by_user', 'research_updates'),
    ('finance/price_at_close_stocks', 'MSFT'),
    ('date_and_time/every_day_at', '09:30'),
    ('google_assistant_v2/activate_scene', 'study time'),
    ('filtrete/barcode_filter_life_threshold', '14'),
    ('feed/new_feed_item', 'https://example.org/research/feed.xml'),
    ('google_drive/any_new_file', 'Research/Incoming'),
]
# All chosen action fields have documented text controls. No unknown account IDs.
ACTIONS = [
    'email/send_me_email', 'google_docs/create_google_doc',
    'google_sheets/append_to_google_spreadsheet', 'dropbox/append_to_text_file_db',
    'android_wear/send_notification_to_android_wear', 'twitter/post_new_tweet',
    'sms/send_me_text', 'phone_call/call_my_phone',
]


def url(name, side):
    service, function = name.split('/', 1)
    return f'https://ifttt.com/{service}/{side}s/{function}'


def documented_endpoint(catalog, name, side):
    record = catalog[url(name, side)]
    old, doc = record['endpoint'], record['documentation']
    assert doc['status'] == 'available', name
    assert not doc.get('frozen_field_slugs_absent_from_page'), name
    assert not old['schema_unreconciled'], name
    text = doc['text']
    marker = side.capitalize() + ' fields\n'
    section = text.split(marker, 1)[1].split('\nIngredients\n', 1)[0] if marker in text else ''
    matches = list(re.finditer(r'\nSlug\n([^\n]+)\nRequired\n(true|false)(?:\n|$)', '\n'+section))
    assert {m.group(1) for m in matches} == {f['slug'] for f in old['fields']}, name
    fields = []
    for m in matches:
        prefix = ('\n'+section)[:m.start()]
        controls = list(re.finditer(r'\n(Text input[^\n]*)\nLabel\n([^\n]+)', prefix))
        assert controls, (name, m.group(1), 'unsupported control')
        control = controls[-1]
        prior = next(f for f in old['fields'] if f['slug'] == m.group(1))
        required = m.group(2) == 'true'
        assert prior['required'] is None or prior['required'] == required, name
        # Text control is a string representation, including numeric-looking text.
        fields.append({'slug':m.group(1), 'label':control.group(2),
                       'help_text':prior.get('help_text',''), 'required':required,
                       'value_type':'string', 'bindable':side=='action',
                       'binding_capability_known':True, 'documented_control':control.group(1)})
    ingredients = []
    section = text.split('\nIngredients\n', 1)[1] if '\nIngredients\n' in text else ''
    parts = re.split(r'\nSlug\n', section)
    for k, part in enumerate(parts[1:]):
        slug = part.split('\n')[0]
        before = parts[k]
        if k:
            before = before.split('\nType\n',1)[-1].split('\n',1)[-1]
        lines = before.strip().splitlines()
        declared_type = re.search(r'\nType\n([^\n]+)', '\n'+part)
        ingredients.append({'slug':slug, 'label':lines[0] if lines else slug,
                            'description':' '.join(lines[1:]),
                            'value_type':'string',
                            'documented_semantic_type':declared_type.group(1) if declared_type else None})
    assert {i['slug'] for i in ingredients} == {i['slug'] for i in old['ingredients']}, name
    assert ingredients or side == 'action', name
    return {'endpoint_id':url(name,side), 'side':side, 'service':old['service'],
            'function':old['function'], 'description':old['description'],
            'schema_revision':digest(doc), 'schema_unreconciled':False,
            'fields':sorted(fields,key=lambda x:x['slug']), 'ingredients':ingredients}


def choose_ingredients(trigger, variant):
    # Prefer content over timestamps for interpretable examples; retain all distractors.
    candidates = [i for i in trigger['ingredients'] if i['documented_semantic_type'] == 'String']
    candidates = candidates or trigger['ingredients']
    ranked = sorted(candidates, key=lambda i:(
        not any(x in i['slug'].lower() for x in ['title','text','message','caption','name','description','content']),
        i['slug']))
    return ranked[variant % len(ranked):] + ranked[:variant % len(ranked)]


def static_value(field, family, variant):
    if field['slug'] in ['path']: return f'Research/Batch{family:02d}/Set{variant+1}'
    if any(x in field['slug'] for x in ['url']): return f'https://example.org/images/sample-{family}-{variant}.jpg'
    if field['slug'] == 'formatted_row': return f'Experiment {family} ||| Trial {variant+1} ||| 27'
    if field['slug'] in ['filename','subject','title']: return f'Research batch {family:02d} version {variant+1}'
    return ['The reading is 27.5 degrees.', 'Review item 42 tomorrow.', 'Status: true; next check at 09:30.'][variant]


def setup_value(name, base, variant):
    # Every variation is fixed before inference and stays within its documented representation.
    if name.startswith('finance/'): return ['MSFT','AAPL','IBM'][variant]
    if name.startswith('date_and_time/'): return ['09:30','14:15','18:45'][variant]
    if name.startswith('filtrete/'): return ['14','21','7'][variant]
    if name.startswith('feed/'): return f'https://example.org/research/feed{variant+1}.xml'
    if name.startswith('instagram/'): return ['research','fieldnotes','observations'][variant]
    if name.startswith('twitter/'): return ['research_updates','field_notes','science_reports'][variant]
    return base + [' A',' B',' C'][variant]


def event_value(ingredient, family, variant, fixture, ordinal):
    slug = ingredient['slug']; low = slug.lower(); kind = ingredient.get('documented_semantic_type') or ''
    if 'url' in low or 'url' in kind.lower() or low in ['link','shareurl']:
        return f'https://example.org/events/{family}/{variant}/{fixture}/{slug}.jpg'
    if 'date' in kind.lower() or any(x in low for x in ['created','published','occurred','date','time']):
        return f'2026-09-{6+fixture:02d}T{8+variant:02d}:{family%60:02d}:{ordinal:02d}Z'
    if any(x in low for x in ['latitude','longitude','price','percentage','temp','days','threshold']):
        return f'{10+family}.{ordinal:02d}{variant}{fixture}'
    return f'{ingredient["label"]} ({slug}) sample {family}-{variant}-{fixture}'


def make_case(catalog, family, category, trigger_name, action_name, variant, setup=None, smoke=False):
    trigger = documented_endpoint(catalog,trigger_name,'trigger')
    action = documented_endpoint(catalog,action_name,'action')
    if category in ['dynamic','static']: assert not trigger['fields']
    family_id = f'{VERSION}-{"smoke" if smoke else "test"}-family-{family:02d}'
    case_id = f'{family_id}-variant-{variant+1}'
    instructions=[]; target=[]
    candidates = choose_ingredients(trigger,variant)
    missing_side = 'trigger' if category=='missing' and family%2 else 'action'
    missing_field = None
    if category=='missing':
        endpoint = trigger if missing_side=='trigger' else action
        missing_field = next(f['slug'] for f in endpoint['fields'] if f['required'])
    for side, endpoint in [('trigger',trigger),('action',action)]:
        for index, field in enumerate(endpoint['fields']):
            label=field['label']; slug=field['slug']; choice=None
            if side==missing_side and slug==missing_field:
                choice={'kind':'needs_input'}
                instructions.append(f'I have not decided the {side} setting "{label}" yet; ask me for it.')
            elif not field['required'] and slug not in ['body']:
                choice={'kind':'omit'}
                instructions.append(f'Leave the optional {side} setting "{label}" unset.')
            elif side=='trigger':
                value=setup_value(trigger_name,setup,variant)
                choice={'kind':'literal','value':value}
                instructions.append(f'For the trigger setting "{label}", use exactly "{value}".')
            elif category=='static' or (category=='mixed' and slug in ['subject','filename']):
                value=static_value(field,family,variant)
                choice={'kind':'literal','value':value}
                instructions.append(f'Set the action setting "{label}" to the literal text "{value}".')
            else:
                ingredient=candidates[index%len(candidates)]
                choice={'kind':'trigger_output','ingredient_slug':ingredient['slug']}
                instructions.append([
                    f'Use the event\'s "{ingredient["label"]}" directly as the action setting "{label}".',
                    f'The action setting "{label}" must contain only the event\'s "{ingredient["label"]}", with no added text.',
                    f'Copy the event\'s "{ingredient["label"]}" into the action setting "{label}" without changing it.'
                ][variant])
            target.append({'side':side,'field':slug,'required':field['required'],'acceptable':[choice]})
    # Instruction order differs from schema order, while meaning remains explicit.
    random.Random(SEED+family*10+variant).shuffle(instructions)
    query=(f'When "{trigger["function"]}" happens in {trigger["service"]}, '
           f'use "{action["function"]}" in {action["service"]}. '+' '.join(instructions))
    case={'case_id':case_id,'query':query,'endpoints':{'trigger':trigger,'action':action}}
    expected_fixtures=[]
    if category!='missing':
        # Base, unrelated-distractor change, then selected-source change.
        events=[{i['slug']:event_value(i,family,variant,k,n) for n,i in enumerate(trigger['ingredients'])} for k in range(3)]
        selected={r['acceptable'][0]['ingredient_slug'] for r in target if r['acceptable'][0]['kind']=='trigger_output'}
        for slug in selected:events[1][slug]=events[0][slug]
        for i in trigger['ingredients']:
            if i['slug'] not in selected:events[2][i['slug']]=events[1][i['slug']]
        for event in events:
            # Constructor-level direct expected values, independent of model interpreter.
            args={'trigger':{},'action':{}}
            for r in target:
                o=r['acceptable'][0]
                if o['kind']=='literal':args[r['side']][r['field']]=o['value']
                elif o['kind']=='trigger_output':args[r['side']][r['field']]=event[o['ingredient_slug']]
            expected_fixtures.append({'event':event,'expected_arguments':args})
    gold={'case_id':case_id,'family_id':family_id,'category':category,'variant':variant+1,
          'input_sha256':digest(case),'reference_state':'known_by_construction',
          'fields':target,'fixtures':expected_fixtures,'fully_specified':category!='missing'}
    return case,gold


def build():
    catalog=read(ROOT/'private/evidence/catalog_snapshot.json')
    cases=[]; references=[]
    family=0
    for category, count in [('dynamic',20),('static',10),('mixed',10),('missing',10)]:
        for k in range(count):
            family+=1
            trigger, setup=(BARE_TRIGGERS[k%10],None) if category in ['dynamic','static'] else SETUP_TRIGGERS[k]
            action=ACTIONS[(k+(k//10)*3)%len(ACTIONS)]
            if category=='mixed': action=ACTIONS[k%4] # enough fields for dynamic + literal
            for v in range(3):
                c,g=make_case(catalog,family,category,trigger,action,v,setup)
                cases.append(c);references.append(g)
    order=list(range(150)); random.Random(SEED).shuffle(order)
    cases=[cases[i] for i in order];references=[references[i] for i in order]
    smoke=[]; smoke_gold=[]
    for k, cat in enumerate(['dynamic','static','mixed','missing']):
        t,setup=(BARE_TRIGGERS[(k+3)%10],None) if k<2 else SETUP_TRIGGERS[k+2]
        c,g=make_case(catalog,90+k,cat,t,ACTIONS[k],2,setup,True)
        smoke.append(c);smoke_gold.append(g)
    # Freeze requires equality on resume; no outcome-dependent rebuilding.
    for name,values in [('inputs',cases),('smoke_inputs',smoke)]:
        path=ROOT/f'private/{name}.jsonl'
        if path.exists():
            from common import rows
            assert rows(path)==values
        else:save_rows(path,values)
    freeze(ROOT/'private/references.json',references)
    freeze(ROOT/'private/smoke_references.json',smoke_gold)
    endpoints={e['endpoint_id']:e for c in cases for e in c['endpoints'].values()}
    result={'version':VERSION,'n':150,'families':50,'seed':SEED,
            'categories':dict(Counter(g['category'] for g in references)),
            'fully_specified':sum(g['fully_specified'] for g in references),
            'fixture_events':sum(len(g['fixtures']) for g in references),
            'endpoints':len(endpoints),'triggers':len({c['endpoints']['trigger']['endpoint_id'] for c in cases}),
            'actions':len({c['endpoints']['action']['endpoint_id'] for c in cases}),
            'input_sha256':sha(ROOT/'private/inputs.jsonl'),'reference_sha256':sha(ROOT/'private/references.json'),
            'catalog_snapshot_sha256':sha(ROOT/'private/evidence/catalog_snapshot.json'),
            'builder_sha256':sha(ROOT/'scripts/build_cases.py'),
            'representation':'documented text controls and string-valued ingredients; no account lookup',
            'scope':'controlled requests, supplied endpoints, local argument materialization, no platform execution'}
    freeze(ROOT/'results/DATASET_MANIFEST.json',result)
    print(result)


if __name__=='__main__':build()
