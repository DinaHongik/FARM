import unittest
import numpy as np
from retrieval_metrics import score_rankings,paired_comparison

class Metrics(unittest.TestCase):
    def test_marginal_positives_do_not_imply_valid_pair(self):
        g=[{'group_id':'g','family_id':'f','valid_pairs':[[0,1],[1,0]]}]
        c={s:[{'id':str(i),'service':0} for i in range(2)] for s in ['trigger','action']}
        m,r=score_rankings(g,c,{s:np.array([[0,1]]) for s in c})
        self.assertEqual(m['correct'],0)
        self.assertEqual(m['candidate_product_coverage']['1'],0)
        self.assertEqual(m['candidate_product_coverage']['5'],1)
        self.assertEqual(m['correct_service_wrong_function'],1)

    def test_valid_alternative_pair_is_correct(self):
        g=[{'group_id':'g','family_id':'f','valid_pairs':[[0,1],[1,0]]}]
        c={s:[{'id':str(i),'service':i} for i in range(2)] for s in ['trigger','action']}
        m,_=score_rankings(g,c,{'trigger':np.array([[1,0]]),'action':np.array([[0,1]])})
        self.assertEqual(m['correct'],1)

    def test_family_bootstrap_and_exact_sign_test(self):
        a=[{'group_id':str(i),'family_id':str(i//2),'correct':0} for i in range(4)]
        b=[dict(r,correct=1) for r in a]
        p=paired_comparison(a,b,samples=100)
        self.assertEqual(p['families'],2)
        self.assertEqual(p['difference_percentage_points'],100)
        self.assertEqual(p['paired_family_bootstrap_95_percentage_points'],[100,100])
        self.assertEqual(p['exact_mcnemar_two_sided_p'],.125)

    def test_pairing_rejects_different_populations(self):
        with self.assertRaises(AssertionError):
            paired_comparison([{'group_id':'a','family_id':'x','correct':1}],[])

if __name__=='__main__':unittest.main()
