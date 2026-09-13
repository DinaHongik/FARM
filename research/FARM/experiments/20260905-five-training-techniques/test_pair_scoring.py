import unittest
import numpy as np
from pair_scoring import joint_pair_logits


class PairScoringTest(unittest.TestCase):
    def test_every_cell_uses_its_own_action_score(self):
        trigger=np.array([[3.,1.,2.]])
        action=np.array([[1.,4.,2.]])
        ids=np.array([0,1,2]);residual=np.zeros((3,3))
        actual=joint_pair_logits(trigger,action,0,ids,ids,residual)
        expected=np.array([[4.,7.,5.],[2.,5.,3.],[3.,6.,4.]])
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(np.unravel_index(actual.argmax(),actual.shape),(0,1))

    def test_unequal_candidate_counts_and_permuted_ids(self):
        trigger=np.array([[3.,1.,2.]])
        action=np.array([[1.,4.,2.,7.]])
        ti=np.array([2,0]);ai=np.array([3,1,0])
        residual=np.arange(6).reshape(2,3)/10
        actual=joint_pair_logits(trigger,action,0,ti,ai,residual)
        expected=np.array([[9.,6.,3.],[10.,7.,4.]])+residual
        np.testing.assert_allclose(actual,expected)


if __name__=='__main__':unittest.main()
