"""
Evaluation Script for FARM Encoders

Computes retrieval metrics:
- Recall@K (K=1, 5, 10)
- MRR (Mean Reciprocal Rank)

Compares pretrained vs fine-tuned models.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexer import extract_trigger_text, extract_action_text
from train.config import config


@dataclass
class EvalResult:
    """Evaluation result container."""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_3: float
    model_name: str
    dataset_type: str  # "trigger" or "action"

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "dataset": self.dataset_type,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr@3": round(self.mrr_at_3, 4),
        }


def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """Compute Recall@K: fraction of queries with correct result in top-K."""
    return sum(1 for r in ranks if r <= k) / len(ranks)


def compute_mrr_at_k(ranks: List[int], k: int = 3) -> float:
    """
    Compute Mean Reciprocal Rank @ K.

    Only considers reciprocal rank if correct answer is in top-K,
    otherwise contributes 0.
    """
    rr_sum = 0.0
    for r in ranks:
        if r <= k:
            rr_sum += 1.0 / r
        # else: contributes 0
    return rr_sum / len(ranks)


def evaluate_retrieval(
    model: SentenceTransformer,
    queries: List[str],
    corpus: List[str],
    model_name: str = "model",
    dataset_type: str = "unknown",
    batch_size: int = 32,
) -> EvalResult:
    """
    Evaluate retrieval performance.

    Args:
        model: SentenceTransformer model
        queries: List of query texts (descriptions)
        corpus: List of corpus texts (schema-enriched API texts)
        model_name: Name for logging
        dataset_type: "trigger" or "action"
        batch_size: Batch size for encoding

    Returns:
        EvalResult with metrics
    """
    print(f"\nEvaluating {model_name} on {dataset_type}...")
    print(f"  Queries: {len(queries)}")
    print(f"  Corpus:  {len(corpus)}")

    # Encode queries and corpus
    print("  Encoding queries...")
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("  Encoding corpus...")
    corpus_embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Compute similarity matrix (queries x corpus)
    print("  Computing similarities...")
    similarities = np.dot(query_embeddings, corpus_embeddings.T)

    # For each query i, the correct answer is corpus[i] (same index)
    # Find rank of correct answer
    ranks = []
    for i in range(len(queries)):
        # Get similarity scores for this query
        scores = similarities[i]
        # Rank all corpus items by similarity (descending)
        sorted_indices = np.argsort(-scores)
        # Find rank of correct item (i)
        rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed
        ranks.append(rank)

    # Compute metrics
    recall_1 = compute_recall_at_k(ranks, k=1)
    recall_3 = compute_recall_at_k(ranks, k=3)
    recall_5 = compute_recall_at_k(ranks, k=5)
    recall_10 = compute_recall_at_k(ranks, k=10)
    mrr_3 = compute_mrr_at_k(ranks, k=3)

    result = EvalResult(
        recall_at_1=recall_1,
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr_at_3=mrr_3,
        model_name=model_name,
        dataset_type=dataset_type
    )

    print(f"  Results:")
    print(f"    Recall@1:  {recall_1:.4f}")
    print(f"    Recall@3:  {recall_3:.4f}")
    print(f"    Recall@5:  {recall_5:.4f}")
    print(f"    Recall@10: {recall_10:.4f}")
    print(f"    MRR@3:     {mrr_3:.4f}")

    return result


def load_trigger_data() -> Tuple[List[str], List[str]]:
    """Load trigger queries and corpus."""
    with open(config.triggers_path, 'r') as f:
        triggers = json.load(f)

    queries = [t["description"] for t in triggers if t.get("description")]
    corpus = [extract_trigger_text(t) for t in triggers if t.get("description")]

    return queries, corpus


def load_action_data() -> Tuple[List[str], List[str]]:
    """Load action queries and corpus."""
    with open(config.actions_path, 'r') as f:
        actions = json.load(f)

    queries = [a["description"] for a in actions if a.get("description")]
    corpus = [extract_action_text(a) for a in actions if a.get("description")]

    return queries, corpus


def evaluate_triggers(
    pretrained_model: Optional[str] = None,
    finetuned_model: Optional[str] = None
) -> List[EvalResult]:
    """Evaluate trigger retrieval."""
    queries, corpus = load_trigger_data()
    results = []

    # Evaluate pretrained
    if pretrained_model:
        model = SentenceTransformer(pretrained_model)
        result = evaluate_retrieval(
            model, queries, corpus,
            model_name="pretrained",
            dataset_type="trigger"
        )
        results.append(result)

    # Evaluate fine-tuned
    if finetuned_model:
        model = SentenceTransformer(finetuned_model)
        result = evaluate_retrieval(
            model, queries, corpus,
            model_name="fine-tuned",
            dataset_type="trigger"
        )
        results.append(result)

    return results


def evaluate_actions(
    pretrained_model: Optional[str] = None,
    finetuned_model: Optional[str] = None
) -> List[EvalResult]:
    """Evaluate action retrieval."""
    queries, corpus = load_action_data()
    results = []

    # Evaluate pretrained
    if pretrained_model:
        model = SentenceTransformer(pretrained_model)
        result = evaluate_retrieval(
            model, queries, corpus,
            model_name="pretrained",
            dataset_type="action"
        )
        results.append(result)

    # Evaluate fine-tuned
    if finetuned_model:
        model = SentenceTransformer(finetuned_model)
        result = evaluate_retrieval(
            model, queries, corpus,
            model_name="fine-tuned",
            dataset_type="action"
        )
        results.append(result)

    return results


def run_full_evaluation(save_path: Optional[Path] = None):
    """
    Run full evaluation comparing pretrained vs fine-tuned.

    Args:
        save_path: Optional path to save results JSON
    """
    print("=" * 60)
    print("FARM Encoder Evaluation")
    print("=" * 60)

    all_results = []

    # Check if fine-tuned models exist
    trigger_ft = config.project_root / config.trigger_output_dir / "final"
    action_ft = config.project_root / config.action_output_dir / "final"

    # Evaluate triggers
    print("\n" + "-" * 60)
    print("TRIGGER EVALUATION")
    print("-" * 60)
    trigger_results = evaluate_triggers(
        pretrained_model=config.base_model,
        finetuned_model=str(trigger_ft) if trigger_ft.exists() else None
    )
    all_results.extend(trigger_results)

    # Evaluate actions
    print("\n" + "-" * 60)
    print("ACTION EVALUATION")
    print("-" * 60)
    action_results = evaluate_actions(
        pretrained_model=config.base_model,
        finetuned_model=str(action_ft) if action_ft.exists() else None
    )
    all_results.extend(action_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Dataset':<10} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'MRR@3':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.model_name:<15} {r.dataset_type:<10} {r.recall_at_1:<8.4f} {r.recall_at_3:<8.4f} {r.recall_at_5:<8.4f} {r.recall_at_10:<8.4f} {r.mrr_at_3:<8.4f}")

    # Save results
    if save_path is None:
        save_path = config.project_root / "logs" / "evaluation_results.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\nResults saved to: {save_path}")

    return all_results


if __name__ == "__main__":
    run_full_evaluation()
