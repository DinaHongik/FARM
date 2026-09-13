"""
FARM Evaluation Framework

Two evaluation scripts:
1. evaluate_rag.py - RAG retrieval performance (fast, no LLM)
2. evaluate_e2e.py - End-to-end with agents (slow, requires LLM)

Uses well-established metrics from recognized benchmarks:
- Retrieval: Recall@K, MRR, NDCG (TREC, MS MARCO, BEIR)
- Service Selection: Accuracy, Joint Accuracy (Intent Classification)
- Slot Filling: JGA, Slot F1 (MultiWOZ, SGD)
- Binding: Accuracy, F1 (Spider Text-to-SQL)
- End-to-End: Success Rate, Pass@K (HumanEval)

Test sets (in data/test/):
- gold.json: Clear descriptions (500 samples)
- noisy.json: Vague descriptions (500 samples)
- oneshot.json: Rare APIs (200 samples)
"""

from .evaluate_e2e import (
    FARMEvaluator,
    EvaluationResult,
    RetrievalMetrics,
    ServiceSelectionMetrics,
    SlotFillingMetrics,
    BindingMetrics,
    EndToEndMetrics,
)

from .evaluate_rag import (
    RAGEvaluator,
    RAGMetrics,
    TestSetResults,
)

__all__ = [
    # E2E evaluation
    "FARMEvaluator",
    "EvaluationResult",
    "RetrievalMetrics",
    "ServiceSelectionMetrics",
    "SlotFillingMetrics",
    "BindingMetrics",
    "EndToEndMetrics",
    # RAG evaluation
    "RAGEvaluator",
    "RAGMetrics",
    "TestSetResults",
]
