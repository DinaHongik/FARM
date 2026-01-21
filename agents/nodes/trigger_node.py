"""
Trigger Agent Node
==================

LangGraph node implementing the Trigger Agent logic.
Analyzes query and trigger candidate, produces OFFER with ingredients.
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
from agents.errors import ParsingError, LLMError
from agents.nodes.output_parser import parse_json_response
from agents.prompts.trigger_agent import (
    TRIGGER_AGENT_SYSTEM_PROMPT,
    format_trigger_user_prompt,
    extract_ingredients_from_api,
)


def trigger_agent_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Execute the Trigger Agent.

    This node:
    1. Gets current trigger candidate from state
    2. Calls LLM to reason about trigger fit
    3. Extracts OFFER (ingredients) from response
    4. Updates state with offer and reasoning

    Args:
        state: Current resolution state

    Returns:
        State updates with trigger reasoning and offer
    """
    # Get current trigger candidate
    trigger_idx = state["current_trigger_idx"]
    trigger_candidates = state["trigger_candidates"]

    if not trigger_candidates or trigger_idx >= len(trigger_candidates):
        return {
            "error": "No trigger candidates available",
            "resolution_status": "failed",
        }

    trigger = trigger_candidates[trigger_idx]
    query = state["query"]

    # Format prompt
    user_prompt = format_trigger_user_prompt(query, trigger)

    # Call LLM
    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": TRIGGER_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Parse response
        parsed = parse_json_response(response_text)

        # Extract ingredients for the OFFER
        ingredients = parsed.get("ingredients", [])

        # If LLM didn't extract ingredients, use operation config directly
        if not ingredients:
            ingredients = extract_ingredients_from_api(trigger.get("api_info", {}))

        # Build OFFER
        offer = {
            "service_name": trigger["service_name"],
            "category": trigger["category"],
            "ingredients": ingredients,
        }

        # Update messages for conversation history
        new_messages = [
            HumanMessage(content=f"[Trigger Agent] {user_prompt}"),
            AIMessage(content=response_text),
        ]

        return {
            "current_offer": offer,
            "trigger_reasoning": parsed.get("reasoning", ""),
            "messages": new_messages,
        }

    except ParsingError as e:
        # Parsing failed - try to extract ingredients from operation config directly
        ingredients = extract_ingredients_from_api(trigger.get("api_info", {}))

        offer = {
            "service_name": trigger["service_name"],
            "category": trigger["category"],
            "ingredients": ingredients,
        }

        return {
            "current_offer": offer,
            "trigger_reasoning": f"[Parsing fallback] Selected {trigger['service_name']}",
            "error": f"Trigger agent parse error: {e.message}",
        }

    except Exception as e:
        return {
            "error": f"Trigger agent error: {str(e)}",
            "resolution_status": "failed",
        }


def trigger_agent_simple(state: ResolutionState) -> Dict[str, Any]:
    """
    Simplified Trigger Agent without LLM call.

    Uses rule-based extraction for testing and fallback.

    Args:
        state: Current resolution state

    Returns:
        State updates with trigger info
    """
    trigger_idx = state["current_trigger_idx"]
    trigger_candidates = state["trigger_candidates"]

    if not trigger_candidates or trigger_idx >= len(trigger_candidates):
        return {
            "error": "No trigger candidates available",
            "resolution_status": "failed",
        }

    trigger = trigger_candidates[trigger_idx]

    # Extract ingredients directly from operation config
    ingredients = extract_ingredients_from_api(trigger.get("api_info", {}))

    offer = {
        "service_name": trigger["service_name"],
        "category": trigger["category"],
        "ingredients": ingredients,
    }

    return {
        "current_offer": offer,
        "trigger_reasoning": f"Selected top-{trigger_idx + 1} trigger: {trigger['service_name']}",
    }
