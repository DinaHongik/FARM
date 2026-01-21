"""
Contract Verifier Node
======================

LangGraph node implementing the Contract Verifier.
Scores and critiques the complete applet configuration.
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
from agents.errors import ParsingError, VerifierError
from agents.nodes.output_parser import parse_json_response
from agents.prompts.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    format_verifier_user_prompt,
    calculate_rule_based_score,
)


def verifier_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Execute the Contract Verifier.

    This node:
    1. Takes the accepted trigger-action pair with bindings
    2. Calls LLM to score and critique the applet
    3. Updates state with verification results
    4. Builds the final executable applet

    Args:
        state: Current resolution state (must have accepted pair)

    Returns:
        State updates with verifier output and final applet
    """
    # Verify we have an accepted pair
    if state["resolution_status"] != "accepted":
        return {
            "error": "Cannot verify - no accepted pair",
        }

    # Get trigger and action
    trigger_idx = state["current_trigger_idx"]
    action_idx = state["current_action_idx"]
    trigger = state["trigger_candidates"][trigger_idx]
    action = state["action_candidates"][action_idx]
    bindings = state["binding_map"] or []

    query = state["query"]

    # Format prompt
    user_prompt = format_verifier_user_prompt(query, trigger, action, bindings)

    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Parse response
        parsed = parse_json_response(response_text)

        # Extract verifier output
        score = parsed.get("score", 0.5)
        critique = parsed.get("critique", "")
        is_executable = parsed.get("is_executable", score >= config.verifier_threshold)
        missing_fields = parsed.get("missing_fields", [])
        suggestions = parsed.get("suggestions", [])

        # Build final applet
        final_applet = _build_final_applet(
            query=query,
            trigger=trigger,
            action=action,
            bindings=bindings,
            score=score,
            critique=critique,
            pair_attempt=state["pair_attempt"] + 1,
        )

        # Update messages
        new_messages = [
            HumanMessage(content=f"[Verifier] {user_prompt}"),
            AIMessage(content=response_text),
        ]

        return {
            "verifier_score": score,
            "verifier_critique": critique,
            "is_executable": is_executable,
            "final_applet": final_applet,
            "messages": new_messages,
        }

    except ParsingError as e:
        # Use rule-based scoring as fallback
        rule_result = calculate_rule_based_score(trigger, action, bindings)

        final_applet = _build_final_applet(
            query=query,
            trigger=trigger,
            action=action,
            bindings=bindings,
            score=rule_result["score"],
            critique=f"[Rule-based] Issues: {rule_result['issues']}",
            pair_attempt=state["pair_attempt"] + 1,
        )

        return {
            "verifier_score": rule_result["score"],
            "verifier_critique": f"[Fallback] {rule_result['issues']}",
            "is_executable": rule_result["is_executable"],
            "final_applet": final_applet,
            "error": f"Verifier parse error: {e.message}",
        }

    except Exception as e:
        # Still build applet even on error
        final_applet = _build_final_applet(
            query=query,
            trigger=trigger,
            action=action,
            bindings=bindings,
            score=0.5,
            critique=f"Verification failed: {str(e)}",
            pair_attempt=state["pair_attempt"] + 1,
        )

        return {
            "verifier_score": 0.5,
            "verifier_critique": f"Verification error: {str(e)}",
            "is_executable": False,
            "final_applet": final_applet,
            "error": f"Verifier error: {str(e)}",
        }


def _build_final_applet(
    query: str,
    trigger: Dict[str, Any],
    action: Dict[str, Any],
    bindings: list,
    score: float,
    critique: str,
    pair_attempt: int,
) -> Dict[str, Any]:
    """
    Build the final executable applet structure.

    Args:
        query: Original user query
        trigger: Trigger configuration
        action: Action configuration
        bindings: Field bindings
        score: Verifier score
        critique: Verifier critique
        pair_attempt: Which pair attempt succeeded

    Returns:
        Final applet dictionary
    """
    # Build field values from bindings
    field_values = {}
    for binding in bindings:
        field = binding.get("action_field", "")
        if binding.get("source_type") == "ingredient":
            ing_name = binding.get("ingredient_name", "")
            field_values[field] = f"{{{{{ing_name}}}}}"  # {{IngredientName}}
        else:
            field_values[field] = binding.get("static_value", "")

    return {
        "query": query,
        "trigger": {
            "service_name": trigger.get("service_name"),
            "category": trigger.get("category"),
            "description": trigger.get("description"),
            "api_info": trigger.get("api_info", {}),  # Operation-level configuration for display
            "ingredients": list(trigger.get("api_info", {}).get("Ingredients", {}).keys()),
        },
        "action": {
            "service_name": action.get("service_name"),
            "category": action.get("category"),
            "description": action.get("description"),
            "api_info": action.get("api_info", {}),  # Operation-level configuration for display
            "field_values": field_values,
        },
        "bindings": bindings,
        "metadata": {
            "verifier_score": score,
            "verifier_critique": critique,
            "resolution_rounds": pair_attempt,
            "is_executable": score >= config.verifier_threshold,
        },
    }


def verifier_simple(state: ResolutionState) -> Dict[str, Any]:
    """
    Simplified verifier without LLM call.

    Uses rule-based scoring for testing and fallback.

    Args:
        state: Current resolution state

    Returns:
        State updates with verification results
    """
    trigger_idx = state["current_trigger_idx"]
    action_idx = state["current_action_idx"]
    trigger = state["trigger_candidates"][trigger_idx]
    action = state["action_candidates"][action_idx]
    bindings = state["binding_map"] or []

    # Rule-based scoring
    result = calculate_rule_based_score(trigger, action, bindings)

    final_applet = _build_final_applet(
        query=state["query"],
        trigger=trigger,
        action=action,
        bindings=bindings,
        score=result["score"],
        critique=f"Rule-based: {result['issues']}",
        pair_attempt=state["pair_attempt"] + 1,
    )

    return {
        "verifier_score": result["score"],
        "verifier_critique": str(result["issues"]),
        "is_executable": result["is_executable"],
        "final_applet": final_applet,
    }
