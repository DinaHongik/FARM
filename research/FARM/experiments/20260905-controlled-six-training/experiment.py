"""Frozen six-run design and service/function probability factorization."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch.nn import functional as F

RUNS = {
    'f3_fixed_seed42': ('f3', 42, False, 0),
    'f3_refresh_seed42': ('f3', 42, True, 1),
    'f3_fixed_seed1337': ('f3', 1337, False, 5),
    'f3_refresh_seed1337': ('f3', 1337, True, 6),
    's1_seed1337': ('s1', 1337, False, 7),
    'h6_seed42': ('h6', 42, False, 2),
}


def configuration(run_id):
    arm, seed, refresh, gpu = RUNS[run_id]
    return dict(run_id=run_id, arm=arm, seed=seed, gpu=gpu, negative_refresh=refresh,
                base_model='google/embeddinggemma-300m',
                revision='57c266a740f537b4dc058e1b0cda161fd15afa75',
                epochs=3, batch_size=16, learning_rate=2e-5, head_learning_rate=1e-3,
                weight_decay=.01, warmup_ratio=.1, max_seq_length=512, temperature=.05,
                microbatch=4 if arm=='s1' else 8, service_weight=.2,
                pair_weight=.2, interaction_rank=32, checkpoint_steps=250,
                full_backbone_trainable=True, dtype='float32', gradient_clip_norm=1.,
                refresh_after_epochs=[1,2] if refresh else [], negatives_per_query=4,
                cosine_ceiling=.98, primary_epoch=3,
                h6_objective='uniform positive negative log p(service|query) p(function|service,query)'
                if arm=='h6' else None)


@contextmanager
def inference_state(model):
    """Mining/validation cannot advance the training RNG or change model modes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


def hierarchical_log_probs(function_scores, service_scores, document_services, allowed=None):
    """Exact p(s|q)*p(f|s,q); each represented service needs its full function set.

    Training supplies all functions of gold services (plus aliases masked per row).
    Inference supplies the entire catalog and represents every service.
    """
    if allowed is None:
        allowed = torch.ones_like(function_scores, dtype=torch.bool)
    if function_scores.shape != allowed.shape:
        raise ValueError('function mask shape mismatch')
    service_lp = F.log_softmax(service_scores, dim=-1)
    columns = torch.as_tensor(document_services, device=function_scores.device, dtype=torch.long)
    output = torch.full_like(function_scores, -torch.inf)
    for service in columns.unique().tolist():
        take = columns == service
        values = function_scores[:, take].masked_fill(~allowed[:, take], -torch.inf)
        # Avoid differentiating logsumexp(-inf,...,-inf) for excluded services.
        has_any = allowed[:, take].any(-1, keepdim=True)
        safe_values = torch.where(has_any, values, torch.zeros_like(values))
        normalizer = torch.logsumexp(safe_values, dim=-1, keepdim=True)
        output[:, take] = values-normalizer+service_lp[:, service, None]
    return output


def hierarchical_loss(function_scores, service_scores, document_services, positive, allowed=None):
    if not positive.any(-1).all() or (allowed is not None and (positive & ~allowed).any()):
        raise ValueError('every query needs unmasked positives')
    lp = hierarchical_log_probs(function_scores, service_scores, document_services, allowed)
    return -(torch.where(positive, lp, torch.zeros_like(lp)).sum(-1)/positive.sum(-1)).mean()


def mine_from_scores(scores, groups, corpus, side_index, count=4, ceiling=.98):
    pools = []
    if scores.shape != (len(groups), len(corpus)) or not np.isfinite(scores).all():
        raise ValueError('invalid mining scores')
    for row, group in enumerate(groups):
        positives = {p[side_index] for p in group['valid_pairs']}
        forbidden = positives | set().union(*(set(corpus[p]['equivalent_indices']) for p in positives))
        texts = {corpus[p]['text'] for p in positives}
        order = np.argsort(-scores[row], kind='stable')
        legal = [int(i) for i in order if i not in forbidden and corpus[i]['text'] not in texts
                 and scores[row, i] <= ceiling]
        if len(legal) < count:
            raise ValueError('insufficient legal negatives')
        pools.append({'global': legal[:count], 'within': legal[:count]})
    return pools
