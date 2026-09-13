"""
Tests for Prompt Templates
==========================

Unit tests for prompt formatting and template generation.
"""

import pytest
from agents.prompts.trigger_agent import (
    TRIGGER_AGENT_SYSTEM_PROMPT,
    format_trigger_user_prompt,
    extract_ingredients_from_api,
)
from agents.prompts.action_agent import (
    ACTION_AGENT_SYSTEM_PROMPT,
    format_action_user_prompt,
    extract_fields_from_api,
    get_required_fields,
)
from agents.prompts.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    format_verifier_user_prompt,
    calculate_rule_based_score,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_TRIGGER = {
    "service_name": "Darkness detected",
    "category": "Smart home & IoT",
    "description": "This trigger fires when darkness is detected.",
    "api_info": {
        "Ingredients": {
            "DetectedAt": {
                "Slug": "detected_at",
                "Type": "String",
                "Example": "January 15, 2024 at 06:30PM"
            },
            "DeviceName": {
                "Slug": "device_name",
                "Type": "String",
                "Example": "Living Room Sensor"
            }
        }
    }
}

SAMPLE_ACTION = {
    "service_name": "Add row to spreadsheet",
    "category": "Popular services",
    "description": "This action adds a row to a spreadsheet.",
    "api_info": {
        "Action fields": {
            "Spreadsheet name": {
                "Label": "Spreadsheet name",
                "Slug": "filename",
                "Required": "true"
            },
            "Formatted row": {
                "Label": "Formatted row",
                "Slug": "formatted_row",
                "Required": "true"
            },
            "Drive folder path": {
                "Label": "Drive folder path",
                "Slug": "folder_path",
                "Required": "false"
            }
        }
    }
}

SAMPLE_OFFER = {
    "service_name": "Darkness detected",
    "category": "Smart home & IoT",
    "ingredients": [
        {"name": "DetectedAt", "slug": "detected_at", "type": "String", "example": "..."},
        {"name": "DeviceName", "slug": "device_name", "type": "String", "example": "..."},
    ]
}


# =============================================================================
# Trigger Agent Tests
# =============================================================================

class TestTriggerAgentPrompts:
    """Tests for Trigger Agent prompts."""

    def test_system_prompt_not_empty(self):
        """System prompt should be defined."""
        assert TRIGGER_AGENT_SYSTEM_PROMPT
        assert len(TRIGGER_AGENT_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_key_instructions(self):
        """System prompt should contain key instructions."""
        assert "OFFER" in TRIGGER_AGENT_SYSTEM_PROMPT
        assert "ingredients" in TRIGGER_AGENT_SYSTEM_PROMPT
        assert "JSON" in TRIGGER_AGENT_SYSTEM_PROMPT

    def test_format_user_prompt(self):
        """User prompt should format correctly."""
        query = "When darkness detected, log to spreadsheet"
        prompt = format_trigger_user_prompt(query, SAMPLE_TRIGGER)

        assert query in prompt
        assert SAMPLE_TRIGGER["service_name"] in prompt
        assert SAMPLE_TRIGGER["category"] in prompt
        assert "DetectedAt" in prompt

    def test_extract_ingredients(self):
        """Should extract ingredients from API info."""
        ingredients = extract_ingredients_from_api(SAMPLE_TRIGGER["api_info"])

        assert len(ingredients) == 2
        assert ingredients[0]["name"] == "DetectedAt"
        assert ingredients[1]["name"] == "DeviceName"

    def test_extract_ingredients_empty(self):
        """Should handle empty API info."""
        ingredients = extract_ingredients_from_api({})
        assert ingredients == []


# =============================================================================
# Action Agent Tests
# =============================================================================

class TestActionAgentPrompts:
    """Tests for Action Agent prompts."""

    def test_system_prompt_not_empty(self):
        """System prompt should be defined."""
        assert ACTION_AGENT_SYSTEM_PROMPT
        assert len(ACTION_AGENT_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_key_instructions(self):
        """System prompt should contain key instructions."""
        assert "ACCEPT" in ACTION_AGENT_SYSTEM_PROMPT
        assert "REJECT" in ACTION_AGENT_SYSTEM_PROMPT
        assert "bindings" in ACTION_AGENT_SYSTEM_PROMPT

    def test_format_user_prompt(self):
        """User prompt should format correctly."""
        query = "When darkness detected, log to spreadsheet"
        prompt = format_action_user_prompt(query, SAMPLE_ACTION, SAMPLE_OFFER)

        assert query in prompt
        assert SAMPLE_ACTION["service_name"] in prompt
        assert "Spreadsheet name" in prompt
        assert SAMPLE_OFFER["service_name"] in prompt

    def test_extract_fields(self):
        """Should extract fields from API info."""
        fields = extract_fields_from_api(SAMPLE_ACTION["api_info"])

        assert len(fields) == 3
        assert fields[0]["name"] == "Spreadsheet name"
        assert fields[0]["required"] is True
        assert fields[2]["required"] is False

    def test_get_required_fields(self):
        """Should get only required fields."""
        required = get_required_fields(SAMPLE_ACTION["api_info"])

        assert len(required) == 2
        assert "Spreadsheet name" in required
        assert "Formatted row" in required
        assert "Drive folder path" not in required


# =============================================================================
# Verifier Tests
# =============================================================================

class TestVerifierPrompts:
    """Tests for Contract Verifier prompts."""

    def test_system_prompt_not_empty(self):
        """System prompt should be defined."""
        assert VERIFIER_SYSTEM_PROMPT
        assert len(VERIFIER_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_scoring_criteria(self):
        """System prompt should contain scoring criteria."""
        assert "score" in VERIFIER_SYSTEM_PROMPT.lower()
        assert "critique" in VERIFIER_SYSTEM_PROMPT.lower()

    def test_format_user_prompt(self):
        """User prompt should format correctly."""
        query = "When darkness detected, log to spreadsheet"
        bindings = [
            {"action_field": "Spreadsheet name", "source_type": "static", "static_value": "Log"},
            {"action_field": "Formatted row", "source_type": "ingredient", "ingredient_name": "DetectedAt"},
        ]

        prompt = format_verifier_user_prompt(query, SAMPLE_TRIGGER, SAMPLE_ACTION, bindings)

        assert query in prompt
        assert SAMPLE_TRIGGER["service_name"] in prompt
        assert SAMPLE_ACTION["service_name"] in prompt

    def test_rule_based_score_complete(self):
        """Rule-based score for complete applet."""
        bindings = [
            {"action_field": "Spreadsheet name", "source_type": "static", "static_value": "Log"},
            {"action_field": "Formatted row", "source_type": "ingredient", "ingredient_name": "DetectedAt"},
        ]

        result = calculate_rule_based_score(SAMPLE_TRIGGER, SAMPLE_ACTION, bindings)

        assert result["score"] >= 0.8
        assert result["is_executable"] is True
        assert len(result["missing_fields"]) == 0

    def test_rule_based_score_incomplete(self):
        """Rule-based score for incomplete applet."""
        bindings = [
            {"action_field": "Spreadsheet name", "source_type": "static", "static_value": "Log"},
            # Missing "Formatted row"
        ]

        result = calculate_rule_based_score(SAMPLE_TRIGGER, SAMPLE_ACTION, bindings)

        assert result["score"] < 0.8
        assert "Formatted row" in result["missing_fields"]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
