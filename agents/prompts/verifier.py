"""
Contract Verifier Prompts
=========================

Prompt templates for the Contract Verifier in the WoT (Web of Things)
multi-agent applet synthesis system.

The Verifier evaluates the complete applet with REFLECTION:
1. Reflects on each component (trigger, action, bindings)
2. Checks alignment with user intent
3. Scores quality and executability

Granite 4 Features Used:
- Chain-of-Thought with explicit REFLECTION
- JSON Schema for structured output
- LLM-as-a-Judge pattern
"""

from typing import Dict, Any, List
import json

from agents.tools.definitions import VERIFIER_SCHEMA, schema_to_prompt_string


# =============================================================================
# SYSTEM PROMPT (WoT + Reflection + Granite 4 Schema)
# =============================================================================

VERIFIER_SYSTEM_PROMPT = """You are a Contract Verifier for WoT (Web of Things) applet synthesis.

IMPORTANT: The trigger and action have already been selected by a high-accuracy semantic matching system.
Your role is to evaluate BINDINGS and EXECUTABILITY only - NOT whether trigger/action match the intent.

FOCUS AREAS (Bindings & Execution):
1. Are the BINDINGS semantically meaningful? (ingredients connected to appropriate fields)
2. Are all REQUIRED action fields bound to something?
3. Can this applet EXECUTE without errors?
4. Are there any type mismatches in bindings?

DO NOT re-evaluate trigger/action selection - that was already done with high confidence.

You must respond with a JSON object. Here's the schema:
<schema>
{
  "type": "object",
  "properties": {
    "reflection": {"type": "string", "description": "Brief reflection on bindings and executability"},
    "bindings_valid": {"type": "boolean", "description": "Are bindings semantically appropriate?"},
    "score": {"type": "number", "description": "Quality score 0.0 to 1.0 (for bindings/execution)"},
    "critique": {"type": "string"},
    "is_executable": {"type": "boolean"},
    "missing_fields": {"type": "array", "items": {"type": "string"}},
    "suggestions": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["reflection", "score", "is_executable"]
}
</schema>

SCORING CRITERIA (Bindings & Execution ONLY):
- Binding Quality (0-0.4): Are bindings semantically appropriate?
- Completeness (0-0.3): Are all required fields bound?
- Executability (0-0.3): Will this run without errors?

SCORE INTERPRETATION:
- 0.8-1.0: Excellent - Bindings complete and appropriate
- 0.6-0.8: Good - Minor binding issues, functional
- 0.4-0.6: Fair - Some bindings missing or questionable
- 0.0-0.4: Poor - Major binding issues or not executable

Example response:
{
  "reflection": "BINDINGS CHECK: Required fields 'Spreadsheet name' and 'Formatted row' are bound. 'Formatted row' uses trigger timestamp ingredient - semantically meaningful for logging. No type mismatches.",
  "bindings_valid": true,
  "score": 0.85,
  "critique": "Bindings are complete and appropriate. All required fields bound.",
  "is_executable": true,
  "missing_fields": [],
  "suggestions": ["Consider adding device name to the log row"]
}"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

def format_verifier_user_prompt(
    query: str,
    trigger: Dict[str, Any],
    action: Dict[str, Any],
    bindings: List[Dict[str, Any]]
) -> str:
    """
    Format the user prompt for the Contract Verifier.

    Args:
        query: Original user query
        trigger: Trigger configuration
        action: Action configuration
        bindings: List of bindings

    Returns:
        Formatted user prompt string
    """
    # Extract trigger ingredients and action fields for binding context
    trigger_api = trigger.get('api_info', {})
    trigger_ingredients = trigger_api.get('Ingredients', {})
    action_api = action.get('api_info', {})
    action_fields = action_api.get('Action fields', {})

    # Find required fields
    required_fields = [
        name for name, info in action_fields.items()
        if isinstance(info, dict) and info.get('Required', 'false').lower() == 'true'
    ]

    prompt = f"""USER QUERY (for context):
{query}

TRIGGER: {trigger.get('service_name', 'Unknown')}
Available Ingredients: {list(trigger_ingredients.keys()) if trigger_ingredients else 'None'}

ACTION: {action.get('service_name', 'Unknown')}
Required Fields: {required_fields if required_fields else 'None'}
All Fields: {list(action_fields.keys()) if action_fields else 'None'}

BINDINGS:
{json.dumps(bindings, indent=2)}

Evaluate the BINDINGS and EXECUTABILITY:
1. Are all required fields bound?
2. Are bindings semantically appropriate?
3. Can this execute without errors?

Respond as a JSON object."""

    return prompt


def calculate_rule_based_score(
    trigger: Dict[str, Any],
    action: Dict[str, Any],
    bindings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate a rule-based score as fallback.

    This provides a deterministic baseline score when LLM verification
    fails or for comparison purposes.

    Args:
        trigger: Trigger configuration
        action: Action configuration
        bindings: List of bindings

    Returns:
        Dictionary with score and details
    """
    score = 0.0
    issues = []

    # Check trigger exists
    if trigger.get("service_name"):
        score += 0.2
    else:
        issues.append("Missing trigger service")

    # Check action exists
    if action.get("service_name"):
        score += 0.2
    else:
        issues.append("Missing action service")

    # Check bindings exist
    if bindings and len(bindings) > 0:
        score += 0.2

        # Check binding quality
        ingredient_bindings = [
            b for b in bindings
            if b.get("source_type") == "ingredient"
        ]
        if ingredient_bindings:
            score += 0.2  # Has dynamic bindings
    else:
        issues.append("No bindings defined")

    # Check required fields coverage
    action_api = action.get("api_info", {})
    action_fields = action_api.get("Action fields", {})
    required_fields = [
        name for name, info in action_fields.items()
        if isinstance(info, dict) and info.get("Required", "false").lower() == "true"
    ]

    bound_fields = [b.get("action_field") for b in (bindings or [])]
    missing = [f for f in required_fields if f not in bound_fields]

    if not missing:
        score += 0.2
    else:
        issues.append(f"Missing required fields: {missing}")

    return {
        "score": min(score, 1.0),
        "issues": issues,
        "missing_fields": missing,
        "is_executable": len(missing) == 0 and score >= 0.6
    }
