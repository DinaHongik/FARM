import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scoring import score_case, binding_counts, normalized_value, token, validate_scoring_reference
from reference import validate_reference
from evidence import exact_excerpt, normalize_quotes

def fixture():
    case={'query':'Save 5 files','endpoints':{
        'trigger':{'fields':[],'ingredients':[{'slug':'Body'},{'slug':'Title'}]},
        'action':{'fields':[{'slug':'content','required':True}]}}}
    reference={'fields':[{'side':'action','field':'content','status':'closed',
                         'acceptable':[{'kind':'trigger_output','ingredient_slug':'Body'}],
                         'evidence_quotes':['Body'],'rationale':'Use the declared body'}],
               'binding_sets':[[{'field':'content','ingredient_slug':'Body'}]],
               'binding_sets_exhaustive':True,'cross_field_constraints':[]}
    draft={'trigger_fields':[],'action_fields':[{'field':'content','source':{'kind':'trigger_output','ingredient_slug':'Body'}}]}
    return case,reference,draft

class SemanticsTests(unittest.TestCase):
    def test_correct_binding_and_wrong_same_type_ingredient(self):
        c,r,d=fixture();self.assertTrue(score_case(c,r,d)['fully_bound_correct'])
        d['action_fields'][0]['source']['ingredient_slug']='Title'
        got=score_case(c,r,d)
        self.assertEqual(got['correct_fields'],0);self.assertEqual(got['binding']['f1'],0)

    def test_alternatives_are_not_unioned(self):
        refs=[[{'field':'a','ingredient_slug':'X'},{'field':'b','ingredient_slug':'Y'}],
              [{'field':'a','ingredient_slug':'Y'},{'field':'b','ingredient_slug':'X'}]]
        got=binding_counts({('a','X'),('b','X')},refs)
        self.assertEqual(got['f1'],.5);self.assertFalse(got['exact'])

    def test_missing_context_is_not_fully_bound(self):
        c,r,d=fixture();r['fields'][0]['status']='missing_context'
        r['fields'][0]['acceptable']=[{'kind':'needs_input'}]
        r['binding_sets']=[[]]
        d['action_fields'][0]['source']={'kind':'needs_input','question':'Which content?'}
        got=score_case(c,r,d)
        self.assertTrue(got['whole_decision_correct']);self.assertFalse(got['fully_bound_eligible'])
        self.assertIsNone(got['binding']['f1'])

    def test_unnecessary_abstention_is_incorrect(self):
        c,r,d=fixture();d['action_fields'][0]['source']={'kind':'needs_input','question':'Content?'}
        got=score_case(c,r,d)
        self.assertEqual(got['correct_fields'],0);self.assertTrue(got['field_results'][0]['unnecessary_missing'])

    def test_open_ended_not_exact_mismatch(self):
        c,r,d=fixture();r['fields'][0]['status']='open_ended';r['binding_sets_exhaustive']=False
        got=score_case(c,r,d);self.assertEqual(got['scorable_fields'],0)
        self.assertFalse(got['whole_decision_eligible']);self.assertIsNone(got['binding'])

    def test_invalid_and_duplicate_predictions_fail(self):
        c,r,d=fixture()
        got=score_case(c,r,None);self.assertEqual(got['scorable_fields'],1);self.assertEqual(got['correct_fields'],0)
        d['action_fields'].append(d['action_fields'][0])
        got=score_case(c,r,d);self.assertFalse(got['whole_decision_correct']);self.assertEqual(got['correct_fields'],0)

    def test_extra_field_blocks_whole_configuration(self):
        c,r,d=fixture();d['action_fields'].append({'field':'invented','source':{'kind':'omit'}})
        got=score_case(c,r,d);self.assertEqual(got['correct_fields'],1);self.assertFalse(got['whole_decision_correct'])

    def test_query_provenance_and_typed_normalization(self):
        src={'kind':'query_span','start':5,'end':6,'text':'5','transform':'parse_integer'}
        self.assertEqual(normalized_value(src,'Save 5 files'),{'kind':'literal','value':5})
        src['text']='7';self.assertIsNone(normalized_value(src,'Save 5 files'))
        self.assertNotEqual(token({'kind':'literal','value':True}),token({'kind':'literal','value':1}))
        self.assertNotEqual(token({'kind':'literal','value':'5'}),token({'kind':'literal','value':5}))

    def test_unknown_requiredness_cannot_be_omitted(self):
        c,r,d=fixture();c['endpoints']['action']['fields'][0]['required']=None
        r['fields'][0]['acceptable']=[{'kind':'omit'}]
        self.assertIn('invalid_omission:action:content',validate_reference(c,r))

    def test_unknown_binding_combinations_defer_whole_score(self):
        c,r,d=fixture();r['binding_sets_exhaustive']=False
        self.assertFalse(score_case(c,r,d)['whole_decision_eligible'])

    def test_inconsistent_binding_reference_is_rejected(self):
        c,r,d=fixture();r['binding_sets']=[[]]
        self.assertIn('binding_set_omits_required_dynamic_choice',validate_scoring_reference(c,r))

    def test_number_parser_preserves_the_declared_transform(self):
        self.assertIsNone(normalized_value({'start':0,'end':3,'text':'1e3','transform':'parse_number'},'1e3'))

    def test_evidence_normalization_preserves_words_and_numbers(self):
        text='Required\ntrue\nUse “|||” to separate cells\nLimit 2000'
        self.assertEqual(exact_excerpt('Required: true',[text]),'Required\ntrue')
        self.assertEqual(exact_excerpt('Use "|||" to separate cells',[text]),'Use “|||” to separate cells')
        self.assertIsNone(exact_excerpt('Limit 1000',[text]))
        self.assertIsNone(exact_excerpt('Cells are optional',[text]))
        self.assertEqual(exact_excerpt('"Required\\ntrue"',[text]),'Required\ntrue')

if __name__=='__main__':unittest.main()
