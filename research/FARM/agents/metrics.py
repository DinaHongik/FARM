"""
Metrics Tracking for FARM Agentic System
========================================

Tracks and aggregates metrics for evaluation and analysis.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class NegotiationMetrics:
    """
    Metrics for a single negotiation session.

    Tracks performance, quality, and debugging information.
    Includes reasoning fields for RAGAS-based evaluation.
    """
    # Identification
    query: str = ""
    session_id: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Total negotiation duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    # Negotiation metrics
    pairs_attempted: int = 0
    first_pair_success: bool = False
    final_status: str = ""  # "success", "failed", "error"

    # Quality metrics
    verifier_score: float = 0.0
    is_executable: bool = False

    # Candidates info
    num_trigger_candidates: int = 0
    num_action_candidates: int = 0

    # Selection info
    selected_trigger_idx: int = -1
    selected_action_idx: int = -1
    selected_trigger_name: str = ""
    selected_action_name: str = ""

    # Bindings
    num_bindings: int = 0
    num_ingredient_bindings: int = 0
    num_static_bindings: int = 0

    # =========================================================================
    # REASONING FIELDS (NEW - for RAGAS evaluation)
    # =========================================================================
    planner_reasoning: str = ""       # Chain-of-thought from planner
    trigger_reasoning: str = ""       # Reasoning for trigger selection
    action_reasoning: str = ""        # Reasoning for action selection
    verifier_critique: str = ""       # Verifier's critique/reflection

    # Reasoning quality metrics
    reasoning_word_count: int = 0     # Total words in reasoning
    reasoning_has_thinking: bool = False  # Contains "THINKING:" marker

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["duration_seconds"] = self.duration_seconds
        return d


class MetricsTracker:
    """
    Tracks metrics across multiple negotiation sessions.

    Provides aggregation and export functionality.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self._sessions: List[NegotiationMetrics] = []

    def start_session(self, query: str, session_id: str = "") -> NegotiationMetrics:
        """
        Start tracking a new session.

        Args:
            query: User query
            session_id: Optional session identifier

        Returns:
            NegotiationMetrics instance for this session
        """
        metrics = NegotiationMetrics(
            query=query,
            session_id=session_id,
            start_time=time.time(),
        )
        self._sessions.append(metrics)
        return metrics

    def end_session(self, metrics: NegotiationMetrics, state: Dict[str, Any]):
        """
        End a session and extract final metrics from state.

        Args:
            metrics: Metrics instance to update
            state: Final negotiation state
        """
        metrics.end_time = time.time()

        # Extract from state
        metrics.pairs_attempted = state.get("pair_attempt", 0) + 1
        metrics.first_pair_success = metrics.pairs_attempted == 1 and \
            state.get("negotiation_status") == "accepted"

        status = state.get("negotiation_status", "")
        if status == "accepted" or state.get("final_applet"):
            metrics.final_status = "success"
        elif status == "failed":
            metrics.final_status = "failed"
        else:
            metrics.final_status = "error"

        # Quality
        metrics.verifier_score = state.get("verifier_score", 0.0)
        metrics.is_executable = state.get("is_executable", False)

        # Candidates
        metrics.num_trigger_candidates = len(state.get("trigger_candidates", []))
        metrics.num_action_candidates = len(state.get("action_candidates", []))

        # Selection
        metrics.selected_trigger_idx = state.get("current_trigger_idx", -1)
        metrics.selected_action_idx = state.get("current_action_idx", -1)

        if state.get("trigger_candidates") and metrics.selected_trigger_idx >= 0:
            try:
                metrics.selected_trigger_name = \
                    state["trigger_candidates"][metrics.selected_trigger_idx].get("service_name", "")
            except (IndexError, KeyError):
                pass

        if state.get("action_candidates") and metrics.selected_action_idx >= 0:
            try:
                metrics.selected_action_name = \
                    state["action_candidates"][metrics.selected_action_idx].get("service_name", "")
            except (IndexError, KeyError):
                pass

        # Bindings
        bindings = state.get("binding_map") or []
        metrics.num_bindings = len(bindings)
        metrics.num_ingredient_bindings = sum(
            1 for b in bindings if b.get("source_type") == "ingredient"
        )
        metrics.num_static_bindings = sum(
            1 for b in bindings if b.get("source_type") == "static"
        )

        # Errors
        if state.get("error"):
            metrics.errors.append(state["error"])

        # =====================================================================
        # REASONING FIELDS (NEW - for RAGAS evaluation)
        # =====================================================================
        metrics.planner_reasoning = state.get("planner_reasoning", "")
        metrics.trigger_reasoning = state.get("trigger_reasoning", "")
        metrics.action_reasoning = state.get("action_reasoning", "")
        metrics.verifier_critique = state.get("verifier_critique", "")

        # Reasoning quality metrics
        all_reasoning = " ".join([
            metrics.planner_reasoning,
            metrics.trigger_reasoning,
            metrics.action_reasoning,
            metrics.verifier_critique,
        ])
        metrics.reasoning_word_count = len(all_reasoning.split())
        metrics.reasoning_has_thinking = "THINKING" in all_reasoning.upper()

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all sessions.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self._sessions:
            return {"total_sessions": 0}

        total = len(self._sessions)
        successful = sum(1 for m in self._sessions if m.final_status == "success")
        failed = sum(1 for m in self._sessions if m.final_status == "failed")
        errors = sum(1 for m in self._sessions if m.final_status == "error")

        first_pair = sum(1 for m in self._sessions if m.first_pair_success)

        durations = [m.duration_seconds for m in self._sessions if m.duration_seconds > 0]
        scores = [m.verifier_score for m in self._sessions if m.verifier_score > 0]
        pairs = [m.pairs_attempted for m in self._sessions]

        return {
            "total_sessions": total,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "success_rate": successful / total if total > 0 else 0,
            "first_pair_success_rate": first_pair / total if total > 0 else 0,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "avg_verifier_score": sum(scores) / len(scores) if scores else 0,
            "avg_pairs_attempted": sum(pairs) / len(pairs) if pairs else 0,
            "executable_rate": sum(1 for m in self._sessions if m.is_executable) / total if total > 0 else 0,
        }

    def export_to_json(self, path: str):
        """
        Export all metrics to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "aggregate": self.get_aggregate_metrics(),
            "sessions": [m.to_dict() for m in self._sessions],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print a summary of metrics to console."""
        agg = self.get_aggregate_metrics()

        print("\n" + "=" * 60)
        print("NEGOTIATION METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Sessions:        {agg['total_sessions']}")
        print(f"Successful:            {agg['successful']}")
        print(f"Failed:                {agg['failed']}")
        print(f"Errors:                {agg['errors']}")
        print(f"Success Rate:          {agg['success_rate']:.1%}")
        print(f"First-Pair Success:    {agg['first_pair_success_rate']:.1%}")
        print(f"Avg Duration:          {agg['avg_duration_seconds']:.2f}s")
        print(f"Avg Verifier Score:    {agg['avg_verifier_score']:.2f}")
        print(f"Avg Pairs Attempted:   {agg['avg_pairs_attempted']:.1f}")
        print(f"Executable Rate:       {agg['executable_rate']:.1%}")
        print("=" * 60)


# Global tracker instance
_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Get the global metrics tracker.

    Returns:
        MetricsTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker


def reset_metrics():
    """Reset the global metrics tracker."""
    global _tracker
    _tracker = None
