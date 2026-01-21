"""
Fallback Logic for Agent Resolution
====================================

Handles pair iteration when Action Agent rejects.
Implements the 3x3 grid search through trigger-action combinations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Literal

# Add parent directory to path for module imports
NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import ResolutionState
from agents.config import config

# Local PAIR_ORDER for backward compatibility with old resolution graph
# NOTE: New selector graph uses RAG score ranking instead
PAIR_ORDER = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
]


def fallback_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Handle rejection and move to next pair.

    This node is called when the Action Agent rejects the current
    trigger-action pair. It advances to the next pair in the
    iteration order.

    Args:
        state: Current resolution state

    Returns:
        State updates with next pair indices
    """
    current_attempt = state["pair_attempt"]
    next_attempt = current_attempt + 1

    # Check if we've exhausted all pairs
    if next_attempt >= len(PAIR_ORDER):
        return {
            "resolution_status": "failed",
            "error": f"All {len(PAIR_ORDER)} pairs exhausted",
        }

    # Get next pair
    next_trigger_idx, next_action_idx = PAIR_ORDER[next_attempt]

    # Validate indices against available candidates
    num_triggers = len(state["trigger_candidates"])
    num_actions = len(state["action_candidates"])

    if next_trigger_idx >= num_triggers or next_action_idx >= num_actions:
        # Skip invalid pairs and try next
        return fallback_node({
            **state,
            "pair_attempt": next_attempt,
        })

    return {
        "current_trigger_idx": next_trigger_idx,
        "current_action_idx": next_action_idx,
        "pair_attempt": next_attempt,
        "resolution_status": "negotiating",
        # Reset current resolution state
        "current_offer": None,
        "trigger_reasoning": None,
        "current_requirements": None,
        "action_reasoning": None,
        "binding_map": None,
    }


def should_continue(state: ResolutionState) -> Literal["continue", "verify", "end"]:
    """
    Routing function for the resolution graph.

    Determines the next step based on resolution status:
    - "continue": Go to fallback and try next pair
    - "verify": Proceed to Contract Verifier
    - "end": Resolution complete (success or failure)

    Args:
        state: Current resolution state

    Returns:
        Next node routing decision
    """
    status = state["resolution_status"]

    if status == "accepted":
        return "verify"
    elif status == "rejected":
        # Check if more pairs available
        next_attempt = state["pair_attempt"] + 1
        if next_attempt < min(len(PAIR_ORDER), config.max_pairs):
            return "continue"
        else:
            return "end"
    elif status == "failed":
        return "end"
    elif status in ("pending", "negotiating"):
        # Should not reach here normally
        return "continue"
    else:
        return "end"


def get_current_pair_info(state: ResolutionState) -> Dict[str, Any]:
    """
    Get information about the current trigger-action pair.

    Utility function for debugging and logging.

    Args:
        state: Current resolution state

    Returns:
        Dictionary with current pair details
    """
    trigger_idx = state["current_trigger_idx"]
    action_idx = state["current_action_idx"]

    trigger_name = "N/A"
    action_name = "N/A"

    if state["trigger_candidates"] and trigger_idx < len(state["trigger_candidates"]):
        trigger_name = state["trigger_candidates"][trigger_idx].get("service_name", "N/A")

    if state["action_candidates"] and action_idx < len(state["action_candidates"]):
        action_name = state["action_candidates"][action_idx].get("service_name", "N/A")

    return {
        "pair_attempt": state["pair_attempt"] + 1,
        "total_pairs": len(PAIR_ORDER),
        "trigger_idx": trigger_idx,
        "action_idx": action_idx,
        "trigger_name": trigger_name,
        "action_name": action_name,
        "status": state["resolution_status"],
    }


def format_rejection_summary(state: ResolutionState) -> str:
    """
    Format a summary of all rejections for debugging.

    Args:
        state: Current resolution state

    Returns:
        Formatted rejection summary string
    """
    history = state.get("rejection_history", [])

    if not history:
        return "No rejections recorded"

    lines = ["Rejection History:"]
    for i, record in enumerate(history, 1):
        lines.append(
            f"  {i}. T{record['trigger_idx']}-A{record['action_idx']}: {record['reason']}"
        )

    return "\n".join(lines)
