"""Cartesian pair logits used by the final joint evaluator."""
def joint_pair_logits(trigger_scores, action_scores, query_index, trigger_ids, action_ids, residual):
    trigger = trigger_scores[query_index, trigger_ids]
    action = action_scores[query_index, action_ids]
    if residual.shape != (len(trigger), len(action)):
        raise ValueError('pair residual shape must match the candidate Cartesian product')
    # Insert axes after selecting vectors: NumPy moves separated advanced-index
    # dimensions to the front, unlike the training tensor indexing expression.
    return trigger[:, None] + action[None, :] + residual
