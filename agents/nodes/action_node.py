"""
Action Agent Node
=================

LangGraph node implementing the Action Agent logic.
Analyzes query, action candidate, and OFFER to produce ACCEPT/REJECT
decision with bindings.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Add parent directory to path for module imports
NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import ResolutionState
from agents.config import get_llm, config
from agents.errors import ParsingError, ResolutionError
from agents.nodes.output_parser import parse_json_response
from agents.prompts.action_agent import (
    ACTION_AGENT_SYSTEM_PROMPT,
    format_action_user_prompt,
    extract_fields_from_api,
    get_required_fields,
)


def action_agent_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Execute the Action Agent.

    This node:
    1. Gets current action candidate and OFFER from state
    2. Calls LLM to reason about action fit and compatibility
    3. Decides ACCEPT or REJECT
    4. If ACCEPT, generates bindings

    Args:
        state: Current resolution state

    Returns:
        State updates with decision and bindings
    """
    # Get current action candidate
    action_idx = state["current_action_idx"]
    action_candidates = state["action_candidates"]

    if not action_candidates or action_idx >= len(action_candidates):
        return {
            "error": "No action candidates available",
            "resolution_status": "failed",
        }

    action = action_candidates[action_idx]
    query = state["query"]
    offer = state["current_offer"]

    # Debug: show which action is being evaluated
    if config.verbose:
        print(f"\n    [DEBUG] Evaluating action #{action_idx}: {action.get('service_name', 'Unknown')}")

    if not offer:
        return {
            "error": "No OFFER available from Trigger Agent",
            "resolution_status": "failed",
        }

    # Format prompt
    user_prompt = format_action_user_prompt(query, action, offer)

    # Call LLM
    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": ACTION_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Debug: print raw LLM response
        if config.verbose:
            print(f"\n    [DEBUG] Action Agent LLM response:")
            print(f"    {response_text[:500]}...")

        # Parse response
        parsed = parse_json_response(response_text)

        # Extract decision
        decision = parsed.get("decision", "REJECT").upper()

        # Update messages
        new_messages = [
            HumanMessage(content=f"[Action Agent] {user_prompt}"),
            AIMessage(content=response_text),
        ]

        if decision == "ACCEPT":
            bindings = parsed.get("bindings", [])

            return {
                "action_reasoning": parsed.get("reasoning", ""),
                "current_requirements": {
                    "service_name": action["service_name"],
                    "category": action["category"],
                    "required_fields": parsed.get("required_fields", []),
                },
                "resolution_status": "accepted",
                "binding_map": bindings,
                "messages": new_messages,
            }
        else:
            # REJECT - record and prepare for fallback
            rejection_reason = parsed.get(
                "rejection_reason",
                "Action agent rejected the pair"
            )

            rejection_record = {
                "trigger_idx": state["current_trigger_idx"],
                "action_idx": action_idx,
                "reason": rejection_reason,
            }

            return {
                "action_reasoning": parsed.get("reasoning", ""),
                "current_requirements": {
                    "service_name": action["service_name"],
                    "category": action["category"],
                    "required_fields": parsed.get("required_fields", []),
                },
                "resolution_status": "rejected",
                "rejection_history": state["rejection_history"] + [rejection_record],
                "messages": new_messages,
            }

    except ParsingError as e:
        # Parsing failed - try rule-based fallback
        return _action_agent_fallback(state, action, offer, str(e))

    except Exception as e:
        return {
            "error": f"Action agent error: {str(e)}",
            "resolution_status": "failed",
        }


def _action_agent_fallback(
    state: ResolutionState,
    action: Dict[str, Any],
    offer: Dict[str, Any],
    error_msg: str
) -> Dict[str, Any]:
    """
    Rule-based fallback when LLM parsing fails.

    Attempts to create bindings based on field/ingredient matching.
    Uses smart heuristics for common action types (spreadsheet, notification).

    Args:
        state: Current state
        action: Action candidate
        offer: Trigger OFFER
        error_msg: Error message from parsing failure

    Returns:
        State updates
    """
    # Get required fields and ingredients
    api_info = action.get("api_info", {})
    required_fields = get_required_fields(api_info)
    ingredients = offer.get("ingredients", [])
    action_name = action.get("service_name", "").lower()

    # Try to create bindings
    bindings = []
    used_ingredients = set()

    for field in required_fields:
        field_lower = field.lower()
        field_slug = api_info.get("Action fields", {}).get(field, {}).get("Slug", field_lower)

        # Try to find matching ingredient
        matched = False
        for ing in ingredients:
            ing_name = ing.get("name", "")
            ing_name_lower = ing_name.lower()

            # Simple heuristic matching
            if (
                ing_name_lower in field_lower or
                field_lower in ing_name_lower or
                _semantic_match(ing_name_lower, field_lower)
            ):
                bindings.append({
                    "action_field": field,
                    "source_type": "ingredient",
                    "ingredient_name": ing_name,
                    "static_value": None,
                    "reasoning": f"Semantic match: {ing_name} -> {field}"
                })
                used_ingredients.add(ing_name)
                matched = True
                break

        if not matched:
            # Smart defaults for common action types
            binding = _get_smart_binding(
                field, field_slug, action_name, ingredients, used_ingredients, offer
            )
            bindings.append(binding)
            if binding.get("source_type") == "ingredient":
                used_ingredients.add(binding.get("ingredient_name", ""))

    # Accept with whatever bindings we have
    return {
        "action_reasoning": f"[Fallback] Generated bindings for {action['service_name']}",
        "current_requirements": {
            "service_name": action["service_name"],
            "category": action["category"],
            "required_fields": required_fields,
        },
        "resolution_status": "accepted",
        "binding_map": bindings,
    }


def _get_smart_binding(
    field: str,
    field_slug: str,
    action_name: str,
    ingredients: list,
    used_ingredients: set,
    offer: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate smart bindings for common action types.

    Handles spreadsheet rows, notifications, etc. with intelligent defaults.

    Args:
        field: Field name
        field_slug: Field slug
        action_name: Action service name (lowercase)
        ingredients: Available ingredients from trigger
        used_ingredients: Set of already used ingredient names
        offer: Full offer dict

    Returns:
        Binding dictionary
    """
    field_lower = field.lower()

    # Spreadsheet actions: combine all ingredients into formatted row
    if "spreadsheet" in action_name or "row" in action_name:
        if "formatted" in field_lower or "row" in field_lower:
            # Combine all ingredients with ||| separator
            ing_names = [ing.get("name") for ing in ingredients]
            formatted = "|||".join([f"{{{{{name}}}}}" for name in ing_names])
            return {
                "action_field": field,
                "source_type": "static",
                "ingredient_name": None,
                "static_value": formatted,
                "reasoning": f"Combined all {len(ing_names)} trigger ingredients into formatted row"
            }
        elif "name" in field_lower or "filename" in field_slug:
            # Use trigger service name for spreadsheet name
            trigger_name = offer.get("service_name", "Events")
            return {
                "action_field": field,
                "source_type": "static",
                "ingredient_name": None,
                "static_value": f"{trigger_name} Log",
                "reasoning": f"Auto-generated spreadsheet name from trigger"
            }

    # Notification/message actions: use first content-like ingredient or all
    if "notification" in action_name or "message" in field_lower:
        # Try to find a content/title ingredient
        for ing in ingredients:
            ing_name = ing.get("name", "")
            ing_lower = ing_name.lower()
            if any(x in ing_lower for x in ["title", "content", "message", "text", "name"]):
                if ing_name not in used_ingredients:
                    return {
                        "action_field": field,
                        "source_type": "ingredient",
                        "ingredient_name": ing_name,
                        "static_value": None,
                        "reasoning": f"Using {ing_name} for message content"
                    }

        # Combine all ingredients if no specific content field
        ing_names = [ing.get("name") for ing in ingredients]
        combined = " - ".join([f"{{{{{name}}}}}" for name in ing_names])
        return {
            "action_field": field,
            "source_type": "static",
            "ingredient_name": None,
            "static_value": combined,
            "reasoning": f"Combined {len(ing_names)} trigger ingredients for message"
        }

    # Title/subject fields: use first title-like ingredient
    if any(x in field_lower for x in ["title", "subject", "name"]):
        for ing in ingredients:
            ing_name = ing.get("name", "")
            ing_lower = ing_name.lower()
            if any(x in ing_lower for x in ["title", "name", "subject"]):
                if ing_name not in used_ingredients:
                    return {
                        "action_field": field,
                        "source_type": "ingredient",
                        "ingredient_name": ing_name,
                        "static_value": None,
                        "reasoning": f"Using {ing_name} for title/name field"
                    }

    # URL fields: use first URL ingredient
    if "url" in field_lower or "link" in field_lower:
        for ing in ingredients:
            ing_name = ing.get("name", "")
            if "url" in ing_name.lower() or "link" in ing_name.lower():
                if ing_name not in used_ingredients:
                    return {
                        "action_field": field,
                        "source_type": "ingredient",
                        "ingredient_name": ing_name,
                        "static_value": None,
                        "reasoning": f"Using {ing_name} for URL field"
                    }

    # Default: use placeholder
    return {
        "action_field": field,
        "source_type": "static",
        "ingredient_name": None,
        "static_value": f"[{field}]",
        "reasoning": f"No matching ingredient for {field}"
    }


def _semantic_match(ingredient: str, field: str) -> bool:
    """
    Simple semantic matching heuristics.

    Args:
        ingredient: Ingredient name (lowercase)
        field: Field name (lowercase)

    Returns:
        True if semantically related
    """
    # Common semantic mappings
    mappings = {
        "title": ["name", "subject", "header"],
        "content": ["message", "body", "text", "description"],
        "url": ["link", "address"],
        "time": ["date", "timestamp", "when"],
        "author": ["user", "creator", "by"],
        "image": ["photo", "picture", "media"],
    }

    for key, synonyms in mappings.items():
        if key in ingredient:
            if any(s in field for s in synonyms) or key in field:
                return True
        if key in field:
            if any(s in ingredient for s in synonyms) or key in ingredient:
                return True

    return False


def action_agent_simple(state: ResolutionState) -> Dict[str, Any]:
    """
    Simplified Action Agent without LLM call.

    Uses rule-based binding generation for testing and fallback.

    Args:
        state: Current resolution state

    Returns:
        State updates with decision and bindings
    """
    action_idx = state["current_action_idx"]
    action_candidates = state["action_candidates"]

    if not action_candidates or action_idx >= len(action_candidates):
        return {
            "error": "No action candidates available",
            "resolution_status": "failed",
        }

    action = action_candidates[action_idx]
    offer = state["current_offer"]

    if not offer:
        return {
            "error": "No OFFER available from Trigger Agent",
            "resolution_status": "failed",
        }

    # Use rule-based binding
    return _action_agent_fallback(state, action, offer, "simple_mode")
