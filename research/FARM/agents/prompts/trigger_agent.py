"""
Trigger Agent Prompts
=====================

Prompt templates for the Trigger Agent in the WoT (Web of Things)
multi-agent applet synthesis system.

The Trigger Agent analyzes the user query and the retrieved trigger
candidate, then produces an OFFER containing available ingredients.

Granite 4 Features Used:
- Chain-of-Thought reasoning (THINKING section)
- JSON Schema for structured output
- Temperature 0 for deterministic responses
"""

from typing import Dict, Any, List
import json

from agents.tools.definitions import TRIGGER_OFFER_SCHEMA, schema_to_prompt_string


# =============================================================================
# SYSTEM PROMPT (WoT + Chain-of-Thought + Granite 4 Schema)
# =============================================================================

TRIGGER_AGENT_SYSTEM_PROMPT = """You are a Trigger Agent in a WoT (Web of Things) multi-agent applet synthesis system.

Your role is to:
1. THINK step-by-step about whether the trigger matches the user's intent
2. Analyze the trigger's API schema and available ingredients
3. Make an OFFER by listing all available ingredients (data fields)

THINKING PROCESS (do this first):
- What event is the user trying to detect?
- Does this trigger fire on that event?
- What data (ingredients) does this trigger provide?
- Are these ingredients useful for the user's goal?

You must respond with a JSON object. Here's the schema you must follow:
<schema>
{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string", "description": "Step-by-step reasoning about trigger match"},
    "service_name": {"type": "string"},
    "category": {"type": "string"},
    "ingredients": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "slug": {"type": "string"},
          "type": {"type": "string"},
          "example": {"type": "string"}
        }
      }
    }
  },
  "required": ["reasoning", "service_name", "ingredients"]
}
</schema>

Example response:
{
  "reasoning": "THINKING: The user wants to detect darkness. This trigger fires when a light sensor detects low light levels, which matches the intent. It provides timestamp and device name as ingredients.",
  "service_name": "Darkness detected",
  "category": "Smart home & sensors",
  "ingredients": [
    {"name": "DetectedAt", "slug": "detected_at", "type": "String", "example": "2024-01-15 18:30:00"},
    {"name": "DeviceName", "slug": "device_name", "type": "String", "example": "Living Room Sensor"}
  ]
}"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

def format_trigger_user_prompt(
    query: str,
    trigger_candidate: Dict[str, Any]
) -> str:
    """
    Format the user prompt for the Trigger Agent.

    Args:
        query: User's natural language query
        trigger_candidate: Trigger candidate from RAG retrieval

    Returns:
        Formatted user prompt string
    """
    # Extract ingredients from API info
    api_info = trigger_candidate.get("api_info", {})
    ingredients_raw = api_info.get("Ingredients", {})

    # Format ingredients for display
    ingredients_display = []
    for name, info in ingredients_raw.items():
        if isinstance(info, dict):
            ingredients_display.append({
                "name": name,
                "slug": info.get("Slug", name),
                "type": info.get("Type", "String"),
                "example": info.get("Example", "N/A")
            })

    prompt = f"""USER QUERY:
{query}

TRIGGER CANDIDATE:
Service Name: {trigger_candidate.get('service_name', 'Unknown')}
Category: {trigger_candidate.get('category', 'Unknown')}
Description: {trigger_candidate.get('description', 'No description')}

Available Ingredients (Data this trigger provides):
{json.dumps(ingredients_display, indent=2)}

Analyze this trigger and provide your response as a JSON object.
Explain why this trigger matches or does not match the user's intent.
List all available ingredients as your OFFER."""

    return prompt


def extract_ingredients_from_api(api_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract ingredient list from trigger API info.

    Args:
        api_info: Raw API info from trigger data

    Returns:
        List of ingredient dictionaries
    """
    ingredients_raw = api_info.get("Ingredients", {})
    ingredients = []

    for name, info in ingredients_raw.items():
        if isinstance(info, dict):
            ingredients.append({
                "name": name,
                "slug": info.get("Slug", name),
                "type": info.get("Type", "String"),
                "example": info.get("Example", "")
            })

    return ingredients
