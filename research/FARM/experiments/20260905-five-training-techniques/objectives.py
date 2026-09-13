"""Multi-positive losses, nonseparable pair scores, and exact gradient replay."""
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def positive_loss(logits, positive_mask, allowed_mask=None):
    """Uniform cross-entropy over every known positive, averaged over groups."""
    positive_mask = positive_mask.bool()
    if logits.shape != positive_mask.shape or not positive_mask.any(dim=-1).all():
        raise ValueError('each query needs at least one positive in a matching mask')
    if allowed_mask is not None:
        if (positive_mask & ~allowed_mask).any():
            raise ValueError('a positive was masked out')
        logits = logits.masked_fill(~allowed_mask, float('-inf'))
    log_prob = F.log_softmax(logits, dim=-1)
    # where avoids 0 * -inf in masked columns.
    selected = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    return -(selected.sum(-1) / positive_mask.sum(-1)).mean()


class ServiceHead(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        dim = prototypes.shape[1]
        self.projection = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.projection.weight)
        self.register_buffer('prototypes', prototypes.clone())

    def forward(self, queries):
        return 20. * F.normalize(self.projection(queries), dim=-1) @ self.prototypes.T


class PairInteraction(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.query = nn.Linear(2 * dim, rank, bias=False)
        self.trigger = nn.Linear(dim, rank, bias=False)
        self.action = nn.Linear(dim, rank, bias=False)
        # Nonzero endpoint factors let query projection receive a first-step gradient.
        nn.init.zeros_(self.query.weight)

    def forward(self, query_trigger, query_action, triggers, actions):
        """One query, endpoint lists -> nonseparable [T,A] residual logits."""
        u = self.query(torch.cat((query_trigger, query_action), dim=-1))
        v = F.normalize(self.trigger(triggers), dim=-1)
        w = F.normalize(self.action(actions), dim=-1)
        return torch.einsum('r,tr,ar->ta', u, v, w)


def model_device(model):
    return next(model.parameters()).device


def embed_features(model, features):
    device = model_device(model)
    features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
    return F.normalize(model(features)['sentence_embedding'], dim=-1)


@dataclass
class ReplayChunk:
    features: dict
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor | None
    size: int


def cache_embeddings(model, features, microbatch):
    chunks, outputs = [], []
    size = features['input_ids'].shape[0]
    cuda = model_device(model).type == 'cuda'
    for start in range(0, size, microbatch):
        chunk = {k: v[start:start+microbatch] if torch.is_tensor(v) and v.ndim and v.shape[0] == size else v
                 for k, v in features.items()}
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if cuda else None
        with torch.no_grad():
            output = embed_features(model, chunk)
        chunks.append(ReplayChunk(chunk, cpu_rng, cuda_rng, len(output)))
        outputs.append(output)
    leaf = torch.cat(outputs).detach().requires_grad_(True)
    return leaf, chunks


def replay_embeddings(model, leaf, chunks):
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError('missing or non-finite cached embedding gradient')
    cuda = model_device(model).type == 'cuda'
    start = 0
    for chunk in chunks:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else []):
            torch.set_rng_state(chunk.cpu_rng)
            if cuda:
                torch.cuda.set_rng_state(chunk.cuda_rng)
            output = embed_features(model, chunk.features)
            gradient = leaf.grad[start:start+chunk.size]
            torch.sum(output * gradient).backward()
        start += chunk.size
