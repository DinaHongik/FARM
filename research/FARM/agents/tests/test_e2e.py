"""
End-to-End Tests for FARM Agentic System
========================================

Integration tests for the complete negotiation workflow.
"""

import pytest
from agents.graph import (
    create_negotiation_graph,
    compile_graph,
    run_negotiation,
    quick_test,
)
from agents.state import create_initial_state
from agents.metrics import MetricsTracker


# =============================================================================
# Graph Creation Tests
# =============================================================================

class TestGraphCreation:
    """Tests for graph creation and compilation."""

    def test_create_graph(self):
        """Should create a valid graph."""
        graph = create_negotiation_graph(use_mock=True)
        assert graph is not None

    def test_compile_graph_without_memory(self):
        """Should compile graph without memory."""
        graph = create_negotiation_graph(use_mock=True)
        compiled = compile_graph(graph, use_memory=False)
        assert compiled is not None

    def test_compile_graph_with_memory(self):
        """Should compile graph with memory."""
        graph = create_negotiation_graph(use_mock=True)
        compiled = compile_graph(graph, use_memory=True)
        assert compiled is not None


# =============================================================================
# Mock Negotiation Tests
# =============================================================================

class TestMockNegotiation:
    """Tests using mock data (no RAG or LLM required)."""

    def test_quick_test_runs(self):
        """Quick test should complete without errors."""
        result = quick_test()
        assert result is not None

    def test_mock_negotiation_produces_applet(self):
        """Mock negotiation should produce an applet."""
        result = run_negotiation(
            query="When darkness detected, log to spreadsheet",
            use_mock=True,
            use_simple=True,
            use_memory=False,
        )

        # Should have final applet or rejection history
        assert result.get("final_applet") is not None or \
               len(result.get("rejection_history", [])) > 0

    def test_mock_negotiation_has_trigger_action(self):
        """Mock negotiation applet should have trigger and action."""
        result = run_negotiation(
            query="When darkness detected, log to spreadsheet",
            use_mock=True,
            use_simple=True,
            use_memory=False,
        )

        if result.get("final_applet"):
            applet = result["final_applet"]
            assert "trigger" in applet
            assert "action" in applet
            assert applet["trigger"]["service_name"] is not None
            assert applet["action"]["service_name"] is not None

    def test_mock_negotiation_has_bindings(self):
        """Mock negotiation applet should have bindings."""
        result = run_negotiation(
            query="When darkness detected, log to spreadsheet",
            use_mock=True,
            use_simple=True,
            use_memory=False,
        )

        if result.get("final_applet"):
            applet = result["final_applet"]
            assert "bindings" in applet
            # Should have at least some bindings
            assert len(applet.get("bindings", [])) > 0 or \
                   len(applet.get("action", {}).get("field_values", {})) > 0


# =============================================================================
# State Flow Tests
# =============================================================================

class TestStateFlow:
    """Tests for state transitions during negotiation."""

    def test_state_starts_pending(self):
        """Initial state should be pending."""
        state = create_initial_state("Test")
        assert state["negotiation_status"] == "pending"

    def test_candidates_loaded(self):
        """Candidates should be loaded after first node."""
        graph = create_negotiation_graph(use_mock=True, use_simple=True)
        compiled = compile_graph(graph, use_memory=False)

        # Run one step
        initial = create_initial_state("Test query")
        events = list(compiled.stream(initial))

        # First event should be load_candidates
        assert len(events) > 0
        first_event = events[0]
        # Check candidates were loaded
        found_candidates = False
        for node_output in first_event.values():
            if node_output.get("trigger_candidates"):
                found_candidates = True
                break
        assert found_candidates


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for metrics tracking."""

    def test_metrics_tracker_creation(self):
        """Should create metrics tracker."""
        tracker = MetricsTracker()
        assert tracker is not None

    def test_metrics_session_tracking(self):
        """Should track session metrics."""
        tracker = MetricsTracker()
        metrics = tracker.start_session("Test query", "test_session")

        assert metrics.query == "Test query"
        assert metrics.session_id == "test_session"

    def test_metrics_aggregation(self):
        """Should aggregate metrics across sessions."""
        tracker = MetricsTracker()

        # Add some sessions
        for i in range(3):
            metrics = tracker.start_session(f"Query {i}")
            metrics.final_status = "success" if i < 2 else "failed"
            metrics.verifier_score = 0.8 + (i * 0.05)

        agg = tracker.get_aggregate_metrics()

        assert agg["total_sessions"] == 3
        assert agg["successful"] == 2
        assert agg["failed"] == 1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_query_handled(self):
        """Should handle empty query gracefully."""
        result = run_negotiation(
            query="",
            use_mock=True,
            use_simple=True,
            use_memory=False,
        )
        # Should complete without crashing
        assert result is not None

    def test_no_candidates_handled(self):
        """Should handle case with no candidates."""
        graph = create_negotiation_graph(use_mock=False, use_simple=True)
        compiled = compile_graph(graph, use_memory=False)

        # This will fail to load candidates (no RAG), but should not crash
        try:
            result = compiled.invoke(create_initial_state("Test"))
            # Either error state or failed negotiation
            assert result.get("error") is not None or \
                   result.get("negotiation_status") == "failed"
        except Exception:
            # Exception is also acceptable for missing RAG
            pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
