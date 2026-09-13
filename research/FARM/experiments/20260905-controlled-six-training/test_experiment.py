import copy, json, random, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from experiment import hierarchical_log_probs, hierarchical_loss, inference_state, mine_from_scores
from objectives import ServiceHead
from runtime import refresh_negatives


class ExperimentTest(unittest.TestCase):
    def test_saved_attention_window(self):
        from checkpoint_loading import test_serialized_window_override_avoids_second_halving
        test_serialized_window_override_avoids_second_halving()

    def test_probability_normalization_and_service_marginals(self):
        f=torch.tensor([[1.,2.,4.,3.,0.],[0.,2.,1.,4.,3.]],requires_grad=True)
        s=torch.tensor([[2.,1.],[-1.,1.]],requires_grad=True)
        services=[0,0,1,1,1]
        p=hierarchical_log_probs(f,s,services).exp()
        torch.testing.assert_close(p.sum(-1),torch.ones(2))
        torch.testing.assert_close(p[:,:2].sum(-1),s.softmax(-1)[:,0])
        torch.testing.assert_close(p[:,2:].sum(-1),s.softmax(-1)[:,1])
        # Within-service additive shifts must not change the distribution.
        shifted=f.detach()+torch.tensor([[13.,13.,-7.,-7.,-7.]])
        torch.testing.assert_close(p,hierarchical_log_probs(shifted,s,services).exp())

    def test_loss_matches_independent_two_stage_calculation_and_gradients(self):
        f=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
        s=torch.tensor([[2.,1.]],requires_grad=True)
        positive=torch.tensor([[True,False,False,True]])
        actual=hierarchical_loss(f,s,[0,0,1,1],positive)
        manual=-(s.log_softmax(-1)[0,0]+f[:,:2].log_softmax(-1)[0,0]+
                 s.log_softmax(-1)[0,1]+f[:,2:].log_softmax(-1)[0,1])/2
        torch.testing.assert_close(actual,manual)
        for a,b in zip(torch.autograd.grad(actual,(f,s),retain_graph=True),torch.autograd.grad(manual,(f,s))):
            torch.testing.assert_close(a,b)

    def test_masked_aliases_and_empty_non_gold_service_have_finite_gradients(self):
        f=torch.tensor([[0.,1000.,2.,3.]],requires_grad=True)
        s=torch.tensor([[0.,0.]],requires_grad=True)
        positive=torch.tensor([[True,False,False,False]])
        allowed=torch.tensor([[True,False,False,False]])
        loss=hierarchical_loss(f,s,[0,0,1,1],positive,allowed)
        loss.backward()
        torch.testing.assert_close(loss,torch.tensor(2.).log())
        self.assertTrue(torch.isfinite(f.grad).all())
        torch.testing.assert_close(f.grad,torch.zeros_like(f))
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_soft_selection_can_recover_from_top_service_gate(self):
        # The favored service spreads .6 over two functions; the other has .4 on one.
        s=torch.tensor([[.6,.4]]).log()
        f=torch.zeros(1,3)
        p=hierarchical_log_probs(f,s,[0,0,1]).exp()
        self.assertEqual(int(p.argmax()),2)
        self.assertEqual(int(s.argmax()),0)

    def test_temperature_changes_execution(self):
        torch.manual_seed(7)
        head=ServiceHead(torch.eye(3),temperature=.05)
        query=torch.tensor([[1.,0.,0.]])
        a=head(query);head.temperature=.1
        torch.testing.assert_close(head(query),a/2)

    def test_inference_restores_rng_and_mixed_modes_even_after_failure(self):
        model=nn.Sequential(nn.Linear(3,3),nn.Dropout()).train();model[0].eval()
        cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state()
        with self.assertRaises(RuntimeError):
            with inference_state(model):
                self.assertFalse(model.training)
                torch.rand(4);random.random();np.random.rand(4)
                raise RuntimeError('simulated interrupted mining')
        self.assertTrue(model.training);self.assertFalse(model[0].training)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertEqual(python,random.getstate())
        self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]))

    def test_miner_masks_all_positives_aliases_text_duplicates_and_ceiling(self):
        corpus=[dict(text=str(i),equivalent_indices=[i]) for i in range(8)]
        corpus[0]['equivalent_indices']=[0,1];corpus[2]['text']='0'
        groups=[{'valid_pairs':[[0,0],[3,0]]}]
        scores=np.array([[.9,.95,.94,.92,.999,.8,.7,.6]])
        pools=mine_from_scores(scores,groups,corpus,0,count=2)
        self.assertEqual(pools[0]['global'],[5,6])

    def test_boundary_resume_reuses_exact_pool_and_rejects_changed_checkpoint(self):
        model=nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp);(checkpoint/'trigger').mkdir()
            (checkpoint/'trigger'/'weights').write_text('fixed')
            corpus=[dict(text=str(i),equivalent_indices=[i]) for i in range(3)]
            groups=[dict(query='request',valid_pairs=[[0,0]])]
            config=dict(negatives_per_query=2,cosine_ceiling=.98)
            with patch('runtime.encode',side_effect=[np.array([[1.,0.]]),np.array([[1.,0.],[.8,.2],[.4,.6]])]):
                first=refresh_negatives(model,checkpoint,groups,corpus,'trigger',config,'datahash')
            with patch('runtime.encode',side_effect=AssertionError('must not recompute')):
                second=refresh_negatives(model,checkpoint,groups,corpus,'trigger',config,'datahash')
            self.assertEqual(first,second)
            (checkpoint/'trigger'/'weights').write_text('different')
            with self.assertRaises(RuntimeError):
                refresh_negatives(model,checkpoint,groups,corpus,'trigger',config,'datahash')


if __name__=='__main__':unittest.main()
