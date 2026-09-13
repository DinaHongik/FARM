"""
Action Agent Prompts
====================

Prompt templates for the Action Agent in the WoT (Web of Things)
multi-agent applet synthesis system.

The Action Agent analyzes the user query, the action candidate, and
the Trigger Agent's OFFER, then decides ACCEPT or REJECT with bindings.

Granite 4 Features Used:
- Chain-of-Thought reasoning (THINKING section)
- JSON Schema for structured output
- Temperature 0 for deterministic responses
"""

from typing import Dict, Any, List, Optional
import json

from agents.tools.definitions import ACTION_DECISION_SCHEMA, schema_to_prompt_string


# =============================================================================
# SYSTEM PROMPT (WoT + Chain-of-Thought + Granite 4 Schema)
# =============================================================================

ACTION_AGENT_SYSTEM_PROMPT = """You are an Action Agent in a WoT (Web of Things) multi-agent applet synthesis system.

Your role is to:
1. THINK step-by-step about whether the action matches the user's intent
2. Check if the Trigger Agent's OFFER (ingredients) can satisfy your REQUIREMENTS (fields)
3. Generate bindings and ACCEPT, or REJECT if fundamentally incompatible

THINKING PROCESS (do this first):
- What action does the user want to perform?
- Does this action service match that intent?
- What fields does this action require?
- Can the trigger's ingredients satisfy these fields?
- If not, can I generate sensible static values?

DECISION RULES:
- ACCEPT: Almost always. You can satisfy any field by:
  * Using an ingredient from OFFER (source_type="ingredient")
  * Generating a static value (source_type="static")
- REJECT: Only if the action is completely wrong for user's intent
  * e.g., user wants "log to spreadsheet" but action is "post to Instagram"

You must respond with a JSON object. Here's the schema:
<schema>
{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string", "description": "Step-by-step thinking about compatibility"},
    "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
    "service_name": {"type": "string"},
    "required_fields": {"type": "array"},
    "bindings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "action_field": {"type": "string"},
          "source_type": {"type": "string", "enum": ["ingredient", "static"]},
          "ingredient_name": {"type": "string"},
          "static_value": {"type": "string"},
          "reasoning": {"type": "string"}
        }
      }
    },
    "rejection_reason": {"type": "string"}
  },
  "required": ["reasoning", "decision", "service_name"]
}
</schema>

Example ACCEPT:
{
  "reasoning": "THINKING: User wants to log data. This action adds rows to a spreadsheet, which matches logging intent. Required fields: spreadsheet name (can generate), formatted row (can use trigger ingredients).",
  "decision": "ACCEPT",
  "service_name": "Add row to spreadsheet",
  "required_fields": [{"name": "Spreadsheet name", "required": true}, {"name": "Formatted row", "required": true}],
  "bindings": [
    {"action_field": "Spreadsheet name", "source_type": "static", "static_value": "Event Log", "reasoning": "Generated descriptive name"},
    {"action_field": "Formatted row", "source_type": "ingredient", "ingredient_name": "DetectedAt", "reasoning": "Using timestamp from trigger"}
  ],
  "rejection_reason": null
}

Example REJECT:
{
  "reasoning": "THINKING: User wants to log data to spreadsheet. This action posts photos to Instagram, which is completely different from logging.",
  "decision": "REJECT",
  "service_name": "Post to Instagram",
  "required_fields": [{"name": "Photo URL", "required": true}],
  "bindings": null,
  "rejection_reason": "Action doesn't match intent - user wants logging, not social media"
}"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

def format_action_user_prompt(
    query: str,
    action_candidate: Dict[str, Any],
    offer: Dict[str, Any]
) -> str:
    """
    Format the user prompt for the Action Agent.

    Args:
        query: User's natural language query
        action_candidate: Action candidate from RAG retrieval
        offer: OFFER from Trigger Agent (ingredients)

    Returns:
        Formatted user prompt string
    """
    # Extract action fields from API info
    api_info = action_candidate.get("api_info", {})
    fields_raw = api_info.get("Action fields", {})

    # Format fields for display
    fields_display = []
    for name, info in fields_raw.items():
        if isinstance(info, dict):
            required = info.get("Required", "false").lower() == "true"
            fields_display.append({
                "name": name,
                "label": info.get("Label", name),
                "slug": info.get("Slug", name),
                "required": required
            })

    # Format ingredients from offer
    ingredients_display = offer.get("ingredients", [])

    prompt = f"""USER QUERY:
{query}

ACTION CANDIDATE:
Service Name: {action_candidate.get('service_name', 'Unknown')}
Category: {action_candidate.get('category', 'Unknown')}
Description: {action_candidate.get('description', 'No description')}

Required Fields (Your REQUIREMENTS):
{json.dumps(fields_display, indent=2)}

OFFER FROM TRIGGER AGENT (Available Ingredients):
Trigger: {offer.get('service_name', 'Unknown')}
Ingredients:
{json.dumps(ingredients_display, indent=2)}

Analyze this action and decide:
- ACCEPT if this action matches the user's intent (e.g., user wants spreadsheet logging and action is spreadsheet-related)
- REJECT only if the action is completely unrelated to user intent (e.g., user wants email but action is SMS)

For ACCEPT: Bind ALL required fields using either:
1. Ingredients from the OFFER (source_type="ingredient"), OR
2. Sensible static values YOU GENERATE (source_type="static")

You can ALWAYS satisfy any field by generating a static value. ACCEPT is almost always correct.

Respond as a JSON object."""

    return prompt


def extract_fields_from_api(api_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract action fields from action API info.

    Args:
        api_info: Raw API info from action data

    Returns:
        List of field dictionaries
    """
    fields_raw = api_info.get("Action fields", {})
    fields = []

    for name, info in fields_raw.items():
        if isinstance(info, dict):
            required = info.get("Required", "false").lower() == "true"
            fields.append({
                "name": name,
                "label": info.get("Label", name),
                "slug": info.get("Slug", name),
                "required": required,
                "helper_text": info.get("Helper text", "")
            })

    return fields


def get_required_fields(api_info: Dict[str, Any]) -> List[str]:
    """
    Get list of required field names from action API info.

    Args:
        api_info: Raw API info from action data

    Returns:
        List of required field names
    """
    fields = extract_fields_from_api(api_info)
    return [f["name"] for f in fields if f["required"]]
