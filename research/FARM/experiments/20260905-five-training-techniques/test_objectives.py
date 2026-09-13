"""Numerical checks for scientific training behavior, independent of model weights."""
import copy
import unittest
import torch
from torch import nn
from objectives import (positive_loss, PairInteraction, ServiceHead,
                        cache_embeddings, replay_embeddings, embed_features)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(30,8)
        self.dropout=nn.Dropout(.3)
        self.linear=nn.Linear(8,8)
    def forward(self,features):
        x=self.embedding(features['input_ids']).mean(1)
        return {'sentence_embedding':self.linear(self.dropout(x))}


class ObjectivesTest(unittest.TestCase):
    def test_alternative_positives_and_masked_alias_gradient(self):
        logits=torch.zeros(1,4,requires_grad=True)
        loss=positive_loss(logits,torch.tensor([[1,1,0,0]],dtype=torch.bool),
                           torch.tensor([[1,1,1,0]],dtype=torch.bool))
        loss.backward()
        torch.testing.assert_close(loss,torch.tensor(3.).log())
        torch.testing.assert_close(logits.grad,torch.tensor([[-1/6,-1/6,1/3,0.]]))

    def test_missing_positive_rejected(self):
        with self.assertRaises(ValueError):
            positive_loss(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))

    def test_gradient_cache_matches_direct_graph_with_dropout(self):
        torch.manual_seed(6)
        direct=Toy().train(); cached=copy.deepcopy(direct)
        features={'input_ids':torch.arange(24).reshape(6,4)}
        target=torch.randn(6,8)
        rng=torch.get_rng_state()
        pieces=[embed_features(direct,{'input_ids':features['input_ids'][i:i+2]}) for i in range(0,6,2)]
        ((torch.cat(pieces)-target)**2).mean().backward()
        torch.set_rng_state(rng)
        leaf,chunks=cache_embeddings(cached,features,2)
        ((leaf-target)**2).mean().backward()
        replay_embeddings(cached,leaf,chunks)
        for first,second in zip(direct.parameters(),cached.parameters()):
            torch.testing.assert_close(first.grad,second.grad,rtol=2e-5,atol=2e-7)

    def test_pair_interaction_initial_identity_and_learnable_coupling(self):
        torch.manual_seed(3)
        pair=PairInteraction(8,4)
        q1,q2=torch.randn(8),torch.randn(8)
        t,a=torch.randn(2,8),torch.randn(2,8)
        initial=pair(q1,q2,t,a)
        torch.testing.assert_close(initial,torch.zeros(2,2))
        positive_loss(initial.reshape(1,4),torch.tensor([[1,0,0,0]],dtype=torch.bool)).backward()
        self.assertGreater(pair.query.weight.grad.norm().item(),0)
        with torch.no_grad():
            pair.query.weight-=.1*pair.query.weight.grad
        score=pair(q1,q2,t,a)
        cross_difference=score[0,0]+score[1,1]-score[0,1]-score[1,0]
        self.assertGreater(abs(cross_difference.item()),1e-7)

    def test_additive_pair_loss_factorizes(self):
        t,a=torch.tensor([.2,1.]),torch.tensor([-.1,.5,1.2])
        joint=(t[:,None]+a[None,:]).reshape(1,-1)
        target=torch.zeros_like(joint,dtype=torch.bool);target[0,5]=True
        loss=positive_loss(joint,target)
        expected=-t.log_softmax(0)[1]-a.log_softmax(0)[2]
        torch.testing.assert_close(loss,expected)

    def test_service_initially_scores_only_name_prototypes(self):
        prototypes=torch.nn.functional.normalize(torch.randn(5,8),dim=-1)
        q=torch.nn.functional.normalize(torch.randn(2,8),dim=-1)
        head=ServiceHead(prototypes)
        torch.testing.assert_close(head(q),20*q@prototypes.T)

    def test_optimizer_resume_is_same_next_update(self):
        torch.manual_seed(8)
        model=nn.Linear(2,2); optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.ones(3,2)
        model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
        restored=copy.deepcopy(model)
        restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001)
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for m,o in ((model,optimizer),(restored,restored_optimizer)):
            m(x).square().mean().backward();o.step()
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
