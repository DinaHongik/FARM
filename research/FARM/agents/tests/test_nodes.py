"""
Tests for Agent Nodes
=====================

Unit tests for individual LangGraph nodes.
"""

import pytest
from agents.state import NegotiationState, create_initial_state
from agents.nodes.load_candidates import load_candidates_mock
from agents.nodes.trigger_node import trigger_agent_simple
from agents.nodes.action_node import _semantic_match
from agents.nodes.fallback import fallback_node, should_continue, get_current_pair_info
from agents.nodes.output_parser import (
    extract_json_from_response,
    parse_json_response,
    safe_parse,
)
from agents.errors import ParsingError


# =============================================================================
# Output Parser Tests
# =============================================================================

class TestOutputParser:
    """Tests for LLM output parsing."""

    def test_parse_raw_json(self):
        """Should parse raw JSON."""
        response = '{"key": "value", "number": 42}'
        result = parse_json_response(response)

        assert result["key"] == "value"
        assert result["number"] == 42

    def test_parse_json_in_code_block(self):
        """Should extract JSON from code blocks."""
        response = """Here is the response:
```json
{"key": "value"}
```
That's the output."""

        result = parse_json_response(response)
        assert result["key"] == "value"

    def test_parse_json_with_leading_text(self):
        """Should extract JSON with leading text."""
        response = 'Some explanation: {"key": "value"}'
        result = parse_json_response(response)
        assert result["key"] == "value"

    def test_parse_json_with_trailing_text(self):
        """Should extract JSON with trailing text."""
        response = '{"key": "value"} That was the JSON.'
        result = parse_json_response(response)
        assert result["key"] == "value"

    def test_parse_nested_json(self):
        """Should handle nested JSON."""
        response = '{"outer": {"inner": "value"}}'
        result = parse_json_response(response)
        assert result["outer"]["inner"] == "value"

    def test_parse_array_json(self):
        """Should handle JSON arrays."""
        response = '[{"a": 1}, {"b": 2}]'
        result = parse_json_response(response)
        assert len(result) == 2

    def test_parse_invalid_json_raises(self):
        """Should raise ParsingError for invalid JSON."""
        response = "This is not JSON"
        with pytest.raises(ParsingError):
            parse_json_response(response)

    def test_safe_parse_returns_default(self):
        """Safe parse should return default on failure."""
        response = "Not JSON"
        result = safe_parse(response, default={"fallback": True})
        assert result["fallback"] is True


# =============================================================================
# Load Candidates Tests
# =============================================================================

class TestLoadCandidates:
    """Tests for candidate loading."""

    def test_mock_loader_returns_candidates(self):
        """Mock loader should return candidates."""
        state = create_initial_state("Test query")
        updates = load_candidates_mock(state)

        assert "trigger_candidates" in updates
        assert "action_candidates" in updates
        assert len(updates["trigger_candidates"]) > 0
        assert len(updates["action_candidates"]) > 0

    def test_mock_candidates_have_required_fields(self):
        """Mock candidates should have all required fields."""
        state = create_initial_state("Test query")
        updates = load_candidates_mock(state)

        for trigger in updates["trigger_candidates"]:
            assert "service_name" in trigger
            assert "category" in trigger
            assert "api_info" in trigger

        for action in updates["action_candidates"]:
            assert "service_name" in action
            assert "category" in action
            assert "api_info" in action


# =============================================================================
# Trigger Node Tests
# =============================================================================

class TestTriggerNode:
    """Tests for Trigger Agent node."""

    def test_simple_trigger_extracts_offer(self):
        """Simple trigger should extract OFFER from candidate."""
        state = create_initial_state("Test query")
        state = {**state, **load_candidates_mock(state)}

        updates = trigger_agent_simple(state)

        assert "current_offer" in updates
        assert updates["current_offer"]["service_name"] is not None
        assert "ingredients" in updates["current_offer"]

    def test_trigger_handles_empty_candidates(self):
        """Trigger should handle empty candidates gracefully."""
        state = create_initial_state("Test query")
        state["trigger_candidates"] = []

        updates = trigger_agent_simple(state)

        assert updates.get("negotiation_status") == "failed"
        assert "error" in updates


# =============================================================================
# Action Node Tests
# =============================================================================

class TestSemanticMatch:
    """Tests for semantic matching heuristics."""

    def test_title_matches_name(self):
        """Title should match name-related fields."""
        assert _semantic_match("entrytitle", "name")
        assert _semantic_match("title", "subject")

    def test_content_matches_message(self):
        """Content should match message-related fields."""
        assert _semantic_match("entrycontent", "message")
        assert _semantic_match("content", "body")

    def test_time_matches_date(self):
        """Time should match date-related fields."""
        assert _semantic_match("eventtime", "timestamp")
        assert _semantic_match("publishedtime", "date")

    def test_no_match_for_unrelated(self):
        """Should not match unrelated terms."""
        assert not _semantic_match("devicename", "photourl")
        assert not _semantic_match("temperature", "username")


# =============================================================================
# Fallback Node Tests
# =============================================================================

class TestFallbackNode:
    """Tests for fallback logic."""

    def test_fallback_increments_pair(self):
        """Fallback should move to next pair."""
        state = create_initial_state("Test query")
        state = {**state, **load_candidates_mock(state)}
        state["pair_attempt"] = 0
        state["current_trigger_idx"] = 0
        state["current_action_idx"] = 0

        updates = fallback_node(state)

        assert updates["pair_attempt"] == 1
        # Next pair is (0, 1)
        assert updates["current_trigger_idx"] == 0
        assert updates["current_action_idx"] == 1

    def test_fallback_fails_after_all_pairs(self):
        """Fallback should fail after all pairs exhausted."""
        state = create_initial_state("Test query")
        state = {**state, **load_candidates_mock(state)}
        state["pair_attempt"] = 8  # Last pair

        updates = fallback_node(state)

        assert updates["negotiation_status"] == "failed"

    def test_should_continue_verify_on_accept(self):
        """Should route to verify on accept."""
        state = create_initial_state("Test")
        state["negotiation_status"] = "accepted"

        result = should_continue(state)
        assert result == "verify"

    def test_should_continue_on_reject(self):
        """Should route to continue on reject (if pairs available)."""
        state = create_initial_state("Test")
        state["negotiation_status"] = "rejected"
        state["pair_attempt"] = 0
        state["trigger_candidates"] = [1, 2, 3]
        state["action_candidates"] = [1, 2, 3]

        result = should_continue(state)
        assert result == "continue"

    def test_should_end_on_failed(self):
        """Should route to end on failed."""
        state = create_initial_state("Test")
        state["negotiation_status"] = "failed"

        result = should_continue(state)
        assert result == "end"


# =============================================================================
# State Tests
# =============================================================================

class TestState:
    """Tests for state management."""

    def test_create_initial_state(self):
        """Initial state should have all required fields."""
        state = create_initial_state("Test query")

        assert state["query"] == "Test query"
        assert state["trigger_candidates"] == []
        assert state["action_candidates"] == []
        assert state["current_trigger_idx"] == 0
        assert state["current_action_idx"] == 0
        assert state["negotiation_status"] == "pending"
        assert state["rejection_history"] == []

    def test_get_current_pair_info(self):
        """Should get current pair info."""
        state = create_initial_state("Test")
        state = {**state, **load_candidates_mock(state)}

        info = get_current_pair_info(state)

        assert "pair_attempt" in info
        assert "trigger_name" in info
        assert "action_name" in info


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
