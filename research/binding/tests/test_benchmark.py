import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import read, rows, digest
from evaluate import score, reference_draft, materialize, inspect, query_value
from workflows import envelope, check, run, initial_farm


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={c['case_id']:c for c in rows(ROOT/'private/inputs.jsonl')}
        cls.gold=read(ROOT/'private/references.json')

    def dynamic(self):
        gold=next(g for g in self.gold if g['category']=='dynamic' and len(self.cases[g['case_id']]['endpoints']['trigger']['ingredients'])>2)
        case=self.cases[gold['case_id']]
        return case,gold,reference_draft(case,gold)

    def test_all_150_constructed_answers_and_fixtures(self):
        for gold in self.gold:
            case=self.cases[gold['case_id']]
            self.assertEqual(digest(case),gold['input_sha256'])
            draft=reference_draft(case,gold);result=score(case,gold,draft)
            self.assertTrue(result['whole_correct'],gold['case_id'])
            self.assertEqual(result['materialization_pass'],gold['fully_specified'])
            report=check(case,draft)
            self.assertFalse(report['errors'],(gold['case_id'],report))
            self.assertEqual(report['locally_valid'],gold['fully_specified'])
        self.assertEqual(len(self.gold),150)
        self.assertEqual(len({g['family_id'] for g in self.gold}),50)

    def test_wrong_same_type_ingredient_fails(self):
        c,g,d=self.dynamic()
        s=next(x['source'] for x in d['action_fields'] if x['source']['kind']=='trigger_output')
        s['ingredient_slug']=next(i['slug'] for i in c['endpoints']['trigger']['ingredients'] if i['slug']!=s['ingredient_slug'])
        r=score(c,g,d)
        self.assertFalse(r['whole_correct']);self.assertFalse(r['materialization_pass'])
        self.assertGreater(r['binding_fp'],0);self.assertGreater(r['binding_fn'],0)
        self.assertTrue(check(c,d)['locally_valid']) # schema conformity is not semantic correctness

    def test_swapped_fields_fail(self):
        g=next(g for g in self.gold if g['category']=='dynamic' and len({r['acceptable'][0].get('ingredient_slug') for r in g['fields'] if r['acceptable'][0]['kind']=='trigger_output'})>=2)
        c=self.cases[g['case_id']];d=reference_draft(c,g)
        ds=[x for x in d['action_fields'] if x['source']['kind']=='trigger_output']
        ds[0]['source'],ds[1]['source']=ds[1]['source'],ds[0]['source']
        self.assertFalse(score(c,g,d)['whole_correct']);self.assertFalse(score(c,g,d)['materialization_pass'])

    def test_wrong_literal_and_bad_span_fail(self):
        g=next(g for g in self.gold if g['category']=='static');c=self.cases[g['case_id']];d=reference_draft(c,g)
        s=next(x['source'] for x in d['action_fields'] if x['source']['kind']=='query_span')
        s.update(start=0,end=4,text=c['query'][:4])
        self.assertFalse(score(c,g,d)['whole_correct'])
        s['text']='fabricated'
        self.assertTrue(inspect(c,d)[2])

    def test_required_omission_and_unnecessary_abstention_fail(self):
        c,g,d=self.dynamic();f=next(f['slug'] for f in c['endpoints']['action']['fields'] if f['required'])
        item=next(x for x in d['action_fields'] if x['field']==f)
        item['source']={'kind':'omit'}
        self.assertFalse(score(c,g,d)['whole_correct'])
        item['source']={'kind':'needs_input','question':'What value?'}
        r=score(c,g,d);self.assertFalse(r['whole_correct']);self.assertGreater(r['unnecessary_abstentions'],0)

    def test_missing_information_is_not_execution(self):
        g=next(g for g in self.gold if g['category']=='missing');c=self.cases[g['case_id']];d=reference_draft(c,g)
        r=score(c,g,d)
        self.assertTrue(r['whole_correct']);self.assertFalse(r['materialization_pass'])
        self.assertFalse(r['declared_schema_valid_fully_bound']);self.assertEqual(r['fixture_count'],0)
        item=next(x for s in ['trigger','action'] for x in d[s+'_fields'] if x['source']['kind']=='needs_input')
        item['source']={'kind':'query_span','start':0,'end':4,'text':c['query'][:4],'transform':'identity'}
        self.assertFalse(score(c,g,d)['whole_correct'])

    def test_malformed_extra_missing_duplicate_are_failures(self):
        c,g,base=self.dynamic()
        for d in [None,[],{}, {'trigger_fields':[],'action_fields':'bad'}]:self.assertFalse(score(c,g,d)['whole_correct'])
        d=copy.deepcopy(base);d['action_fields'].append(copy.deepcopy(d['action_fields'][0]))
        self.assertFalse(score(c,g,d)['whole_correct'])
        d=copy.deepcopy(base);d['action_fields'].pop();self.assertFalse(score(c,g,d)['whole_correct'])
        d=copy.deepcopy(base);d['action_fields'].append({'field':'hallucinated','source':{'kind':'omit'}})
        self.assertFalse(score(c,g,d)['whole_correct'])

    def test_future_trigger_output_and_fabricated_resource_fail(self):
        g=next(g for g in self.gold if g['category']=='mixed');c=self.cases[g['case_id']];d=reference_draft(c,g)
        d['trigger_fields'][0]['source']={'kind':'trigger_output','ingredient_slug':c['endpoints']['trigger']['ingredients'][0]['slug']}
        self.assertFalse(score(c,g,d)['whole_correct'])
        d['trigger_fields'][0]['source']={'kind':'resource_ref','resource_id':'invented','observation_id':'invented'}
        self.assertFalse(score(c,g,d)['whole_correct'])

    def test_event_dependency_and_distractor_invariance(self):
        c,g,d=self.dynamic()
        outcomes=[materialize(c,d,f['event']) for f in g['fixtures']]
        self.assertEqual(outcomes[0],outcomes[1]);self.assertNotEqual(outcomes[1],outcomes[2])
        # A constant implementation could match fixture 0, but not the changed-source fixture.
        self.assertNotEqual(g['fixtures'][0]['expected_arguments'],g['fixtures'][2]['expected_arguments'])

    def test_static_invariance_and_empty_binding_not_perfect_f1(self):
        g=next(g for g in self.gold if g['category']=='static');c=self.cases[g['case_id']];d=reference_draft(c,g)
        r=score(c,g,d)
        self.assertFalse(r['binding_applicable']);self.assertEqual(r['binding_tp'],0)
        self.assertTrue(r['materialization_pass'])
        self.assertEqual(g['fixtures'][0]['expected_arguments'],g['fixtures'][2]['expected_arguments'])

    def test_each_fixture_distinguishes_every_ingredient(self):
        for gold in self.gold:
            for fixture in gold['fixtures']:
                values=list(fixture['event'].values())
                self.assertEqual(len(values),len(set(values)),gold['case_id'])

    def test_gold_and_fixture_values_are_not_in_prompt_envelope(self):
        for g in self.gold:
            c=self.cases[g['case_id']];payload=envelope(c)
            self.assertEqual(set(payload),{'query','selected_endpoints'})
            text=json.dumps(payload)
            for key in ['expected_arguments','acceptable','reference_state','family_id','fixtures','case_id']:
                self.assertNotIn('"'+key+'"',text)

    def test_numeric_looking_strings_do_not_silently_coerce(self):
        query='27.5 true'
        self.assertEqual(query_value(query,{'start':0,'end':4,'text':'27.5','transform':'identity'}),'27.5')
        with self.assertRaises(ValueError):query_value('1e3',{'start':0,'end':3,'text':'1e3','transform':'parse_number'})

    def test_smoke_families_are_disjoint(self):
        smoke=read(ROOT/'private/smoke_references.json')
        self.assertFalse({g['family_id'] for g in smoke}&{g['family_id'] for g in self.gold})

    def test_workflow_ablation_and_budget(self):
        c,g,d=self.dynamic()
        class Client:
            def __init__(self):self.payloads=[]
            def chat(self,identity,system,payload):
                self.payloads.append((identity,payload))
                parsed={'repair_needed':False} if '/review-' in identity else copy.deepcopy(d)
                return {'parsed':parsed}
        client=Client();shared=initial_farm(client,c)
        no=run(client,c,'farm_no_feedback',shared)
        yes=run(client,c,'farm_feedback',shared)
        single=run(client,c,'single_agent')
        self.assertEqual(no['initial'],yes['initial']);self.assertEqual(no['initial'],no['final'])
        self.assertEqual(len(no['calls']),3)
        for r in [no,yes,single]:self.assertLessEqual(len(r['calls']),7)
        self.assertFalse(any('checker' in p for i,p in client.payloads if '/shared/' in i))
        self.assertTrue(any('checker' in p for i,p in client.payloads if '/single_agent/revision-' in i))


if __name__=='__main__':unittest.main()
