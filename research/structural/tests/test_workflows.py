import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from workflows import check, envelope, initial_farm, run


def case():
    endpoint={'endpoint_id':'test','schema_revision':'fixture-v1','service':'Fixture',
              'function':'Fixture function','description':'Synthetic runner test',
              'fields':[],'ingredients':[],'schema_unreconciled':False}
    trigger={**endpoint,'side':'trigger'}
    action={**endpoint,'side':'action','fields':[{'slug':'name','label':'Name','help_text':'',
        'required':True,'value_type':'string','bindable':False,'binding_capability_known':True}]}
    return {'case_id':'smoke','query':'save Alice','endpoints':{'trigger':trigger,'action':action},
            'private_gold':{'never_send':'GOLD_SENTINEL'}}


def draft(text='Alice',start=5,end=10):
    return {'trigger_fields':[],'action_fields':[{'field':'name','source':{
        'kind':'query_span','start':start,'end':end,'text':text,'transform':'identity'}}],'preview':'Save name'}


class Fake:
    def __init__(self, responses):self.responses=list(responses);self.requests=[]
    def chat(self,identity,system,payload):
        self.requests.append((identity,system,payload))
        return {'parsed':self.responses.pop(0),'prompt_tokens':1,'completion_tokens':1,'provider_seconds':0}


class WorkflowTests(unittest.TestCase):
    def test_prompt_allowlist(self):
        self.assertNotIn('GOLD_SENTINEL',str(envelope(case())))
        self.assertNotIn('private_gold',str(envelope(case())))

    def test_unknown_type_does_not_hide_fabricated_span(self):
        c=case();c['endpoints']['action']['fields'][0]['value_type']=None
        r=check(c,draft('Bob',5,8))
        self.assertEqual(r['unsupported_assignments'],1)
        self.assertFalse(r['no_provenance_violation'])
        self.assertTrue(r['unresolved'])

    def test_valid_provenance_is_not_semantic_gold(self):
        r=check(case(),draft())
        self.assertTrue(r['complete']);self.assertTrue(r['locally_valid'])
        self.assertIsNone(r['semantic_binding_accuracy'])

    def test_unknown_type_does_not_hide_invalid_conversion(self):
        c=case();c['endpoints']['action']['fields'][0]['value_type']=None
        d=draft();d['action_fields'][0]['source']['transform']='parse_integer'
        self.assertEqual(check(c,d)['unsupported_assignments'],1)

    def test_ablation_gets_no_feedback(self):
        client=Fake([{}, {'trigger_fields':[]},draft('Bob',5,8)])
        r=run(client,case(),'farm_no_feedback')
        self.assertEqual(len(client.requests),3)
        self.assertTrue(all('checker' not in p for _,_,p in client.requests))
        self.assertEqual(r['initial'],r['final'])
        self.assertEqual(r['final_checks']['unsupported_assignments'],1)

    def test_shared_initial_is_identical_and_repair_is_measured(self):
        bad=draft('Bob',5,8)
        shared=(bad,[{'stage':'shared'}]*3)
        client=Fake([{'repair_needed':True,'issues':['wrong span']},draft(),{'repair_needed':False,'issues':[]}])
        good=run(client,case(),'farm_feedback',shared)
        off=run(Fake([]),case(),'farm_no_feedback',shared)
        self.assertEqual(good['initial'],off['initial'])
        self.assertEqual(good['initial_checks']['unsupported_assignments'],1)
        self.assertEqual(good['final_checks']['unsupported_assignments'],0)
        self.assertLessEqual(len(good['calls']),7)

    def test_strong_single_agent_can_repair(self):
        client=Fake([draft('Bob',5,8),draft()])
        r=run(client,case(),'single_agent')
        self.assertEqual(r['final_checks']['unsupported_assignments'],0)
        self.assertIn('checker',client.requests[1][2])

    def test_malformed_output_remains_failure(self):
        self.assertFalse(check(case(),None)['protocol_valid'])
        self.assertFalse(check(case(),{})['complete'])


if __name__=='__main__':unittest.main()
