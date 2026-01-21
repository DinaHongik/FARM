"""
Action Selector Node (Multi-Agent Offer-Verify Design)
======================================================

MULTI-AGENT PATTERN:
1. Trigger Agent makes an OFFER (ingredients it provides)
2. Action Agent VERIFIES if offer satisfies its needs (required fields)
3. If verified -> generate bindings
4. If rejected -> try next pair

This is true multi-agent: two agents with different roles communicating!
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher

NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import ResolutionState
from agents.config import config, get_llm
from agents.prompts.action_agent import get_required_fields

# Whether to use LLM verification for offer-verify pattern
# TEMPORARILY DISABLED: Was rejecting valid pairs, need to debug
USE_OFFER_VERIFY = False


def action_selector_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Action Agent: Verify trigger's offer and select action.

    Multi-Agent Offer-Verify Pattern:
    1. Receive OFFER from Trigger Agent (ingredients)
    2. VERIFY if offer can satisfy action's required fields
    3. If verified -> accept and generate bindings
    4. If rejected -> signal to try next pair

    Args:
        state: Current resolution state

    Returns:
        State updates with verification result and bindings
    """
    action_candidates = state["action_candidates"]
    offer = state["current_offer"]
    query = state.get("query", "")
    action_idx = state.get("suggested_action_idx", 0)  # Suggestion from trigger selector

    if not action_candidates:
        return {
            "error": "No action candidates available",
            "resolution_status": "failed",
        }

    if not offer:
        return {
            "error": "No trigger offer available",
            "resolution_status": "failed",
        }

    max_actions = min(len(action_candidates), getattr(config, "top_k_candidates", 5))

    if action_idx >= max_actions:
        action_idx = 0

    # LLM re-rank with AGREEMENT-BASED selection (not unconditional override)
    # Only override RAG if LLM pick has similar RAG score
    rag_top_idx = 0
    rag_top_score = action_candidates[0].get("score", 0) if action_candidates else 0

    if len(action_candidates) >= 2:
        trigger_name = offer.get("service_name", "")
        trigger_desc = offer.get("description", "")
        trigger_ingredients = [ing.get("name", "") for ing in offer.get("ingredients", [])]

        llm_pick_idx = _llm_rerank_actions(
            query=query,
            trigger_name=trigger_name,
            trigger_description=trigger_desc,
            trigger_ingredients=trigger_ingredients,
            actions=action_candidates[:max_actions],
        )

        # Agreement-based selection
        if llm_pick_idx is None:
            action_idx = rag_top_idx
            selection_method = "RAG (LLM unavailable)"
        elif llm_pick_idx == rag_top_idx:
            action_idx = llm_pick_idx
            selection_method = "Agreement (RAG + LLM)"
        else:
            llm_pick_score = action_candidates[llm_pick_idx].get("score", 0)
            score_ratio = llm_pick_score / rag_top_score if rag_top_score > 0 else 0
            # Threshold 0.80: Trust LLM if its pick has at least 80% of RAG top score
            # This balances trusting LLM reasoning while still respecting RAG rankings
            if score_ratio >= 0.80:
                action_idx = llm_pick_idx
                selection_method = f"LLM (score ratio: {score_ratio:.2f})"
            else:
                action_idx = rag_top_idx
                selection_method = f"RAG (LLM pick too low: {score_ratio:.2f})"

        if config.verbose:
            print(f"    [Selection] {selection_method}")

    # Track if LLM overrode RAG (for verifier fallback)
    llm_overrode_rag = (action_idx != rag_top_idx)

    action = action_candidates[action_idx]

    # DEBUG: Show action candidates with scores
    if config.verbose:
        print(f"\n    [Action Agent] Candidates from RAG:")
        for i, a in enumerate(action_candidates[:max_actions]):
            score = a.get("score", 0)
            name = a.get("service_name", "?")
            marker = " <- RAG top" if i == 0 else ""
            if i == action_idx:
                marker = " <- selected"
            print(f"      A{i}: {name} (score={score:.3f}){marker}")

        # Show what trigger offers vs what action needs
        ing_names = [ing.get("name", "") for ing in offer.get("ingredients", [])]
        required = get_required_fields(action.get("api_info", {}))
        print(f"\n    [Action Agent] Trigger OFFERS: {ing_names}")
        print(f"    [Action Agent] Action REQUIRES: {required}")

        # Show matching analysis
        if required:
            print(f"    [Action Agent] Matching:")
            for field in required:
                matched_ing = None
                for ing in ing_names:
                    if ing and field and (ing.lower() in field.lower() or field.lower() in ing.lower()):
                        matched_ing = ing
                        break
                if matched_ing:
                    print(f"      {field} <- {matched_ing} [matched]")
                else:
                    print(f"      {field} <- [static]")

    # OFFER-VERIFY: Action Agent verifies if trigger's offer is acceptable
    verification_passed = True
    verification_reason = "Pair ranking"

    if USE_OFFER_VERIFY:
        if config.verbose:
            print(f"\n    [Action Agent] Verifying offer from Trigger Agent...")

        verified, reason = _verify_offer(
            query=query,
            offer=offer,
            action=action,
        )

        if verified:
            verification_passed = True
            verification_reason = f"Verified: {reason}"
            if config.verbose:
                print(f"    [Action Agent] ACCEPTED: {reason}")
        else:
            verification_passed = False
            verification_reason = f"Rejected: {reason}"
            if config.verbose:
                print(f"    [Action Agent] REJECTED: {reason}")

    # If verification failed, signal rejection and track attempt
    if not verification_passed:
        # Get current attempt count and tried pairs
        pair_attempt = state.get("pair_attempt", 0) + 1
        tried_pairs = list(state.get("tried_pairs") or [])
        trigger_idx = state.get("current_trigger_idx", 0)
        tried_pairs.append((trigger_idx, action_idx))

        return {
            "current_action_idx": action_idx,
            "action_reasoning": verification_reason,
            "resolution_status": "rejected",  # Signal to try next pair
            "pair_attempt": pair_attempt,  # Track retry count
            "tried_pairs": tried_pairs,  # Mark this pair as tried
            "llm_overrode_rag": llm_overrode_rag,  # For verifier fallback
            "current_requirements": {
                "service_name": action["service_name"],
                "category": action["category"],
                "required_fields": get_required_fields(action.get("api_info", {})),
            },
        }

    # Verification passed - generate bindings
    bindings = _generate_bindings(action, offer)

    if config.verbose:
        print(f"    [Action Agent] Selected A{action_idx}: {action['service_name']}")
        print(f"    Generated {len(bindings)} bindings")

    return {
        "current_action_idx": action_idx,
        "action_reasoning": verification_reason,
        "llm_overrode_rag": llm_overrode_rag,  # For verifier fallback
        "current_requirements": {
            "service_name": action["service_name"],
            "category": action["category"],
            "required_fields": get_required_fields(action.get("api_info", {})),
        },
        "resolution_status": "accepted",
        "binding_map": bindings,
    }


def _generate_bindings(action: Dict, offer: Dict) -> List[Dict]:
    """Generate bindings between trigger ingredients and action fields."""
    api_info = action.get("api_info", {})
    required_fields = get_required_fields(api_info)
    ingredients = offer.get("ingredients", [])

    bindings = []
    used_ingredients = set()

    for field in required_fields:
        field_lower = field.lower()

        # Try to find matching ingredient
        matched = False
        for ing in ingredients:
            ing_name = ing.get("name", "")
            ing_lower = ing_name.lower()

            if ing_name in used_ingredients:
                continue

            # Matching heuristics
            if (ing_lower in field_lower or field_lower in ing_lower or
                _semantic_match(ing_lower, field_lower)):
                bindings.append({
                    "action_field": field,
                    "source_type": "ingredient",
                    "ingredient_name": ing_name,
                    "static_value": None,
                })
                used_ingredients.add(ing_name)
                matched = True
                break

        if not matched:
            # Use placeholder for unmatched fields
            bindings.append({
                "action_field": field,
                "source_type": "static",
                "ingredient_name": None,
                "static_value": f"[{field}]",
            })

    return bindings


def _semantic_match(ingredient: str, field: str) -> bool:
    """Fuzzy semantic matching using string similarity."""
    ratio = SequenceMatcher(None, ingredient, field).ratio()
    if ratio >= 0.6:
        return True

    if len(ingredient) >= 3 and len(field) >= 3:
        if ingredient in field or field in ingredient:
            return True

    ing_words = set(ingredient.replace("_", " ").split())
    field_words = set(field.replace("_", " ").split())

    if ing_words & field_words:
        return True

    return False


def _verify_offer(
    query: str,
    offer: Dict[str, Any],
    action: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Action Agent verifies if trigger's offer can satisfy its needs.

    This is the core of the Offer-Verify multi-agent pattern.
    Uses LLM to judge semantic compatibility - NO HARDCODING.

    Args:
        query: User query for context
        offer: Trigger's offer containing ingredients
        action: Action candidate with required fields

    Returns:
        Tuple of (verified: bool, reason: str)
    """
    # Extract ingredients from offer
    ingredients = offer.get("ingredients", [])
    ingredient_names = [ing.get("name", "") for ing in ingredients if ing.get("name")]

    # Extract required fields from action
    required_fields = get_required_fields(action.get("api_info", {}))

    # If no required fields, auto-accept
    if not required_fields:
        return True, "Action has no required fields"

    # If no ingredients, check if action can work without data
    if not ingredient_names:
        return True, "Trigger provides no data, action may use static values"

    # Format for LLM - NO HARDCODING, just schema info
    ingredients_text = "\n".join(f"  - {name}" for name in ingredient_names)
    fields_text = "\n".join(f"  - {field}" for field in required_fields)

    prompt = f"""You are an Action Agent verifying if a trigger can provide useful data for an action.

USER QUERY: {query}

TRIGGER OFFERS these ingredients (data it provides):
{ingredients_text}

ACTION REQUIRES these fields (data it needs):
{fields_text}

VERIFICATION TASK:
Can the trigger's ingredients reasonably provide data for the action's required fields?

Consider:
1. Semantic compatibility (e.g., "temperature" can fill "value", "message" can fill "body")
2. Data type compatibility (e.g., text can fill text fields, numbers can fill number fields)
3. User intent - does this combination make sense for the query?

Answer format:
VERDICT: YES or NO
REASON: (brief explanation)

Answer:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Parse response
        lines = answer.split("\n")
        verdict = "YES"
        reason = "Ingredients can satisfy required fields"

        for line in lines:
            line_upper = line.upper()
            if "VERDICT:" in line_upper:
                verdict = "YES" if "YES" in line_upper else "NO"
            elif "REASON:" in line_upper:
                reason = line.split(":", 1)[-1].strip()

        return verdict == "YES", reason

    except Exception as e:
        # On error, default to accept (fail open)
        if config.verbose:
            print(f"    [Action Agent] Verification error: {e}, defaulting to accept")
        return True, f"Verification error: {e}"


def _llm_rerank_actions(
    query: str,
    trigger_name: str,
    trigger_description: str,
    trigger_ingredients: List[str],
    actions: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Use LLM to pick the best action from top-5 candidates.

    Shares trigger context with action selection (no hardcoding).

    Args:
        query: User query
        trigger_name: Selected trigger name
        trigger_description: Trigger description (what it detects/fires on)
        trigger_ingredients: Data the trigger provides
        actions: List of action candidates

    Returns:
        Index of the best action, or None on error
    """
    options = ["A", "B", "C", "D", "E"]
    options_text = ""

    for i, action in enumerate(actions[:5]):
        name = action.get('service_name', 'Unknown')
        category = action.get('category', '')
        # Use FULL description - don't truncate! The details matter for disambiguation
        desc = action.get('description', 'No description')
        cat_str = f" [{category}]" if category else ""
        options_text += f"\n{options[i]}: {name}{cat_str}\n   {desc}\n"

    valid_options = options[:len(actions)]

    # Build trigger context from shared state
    trigger_context = f"TRIGGER: {trigger_name}"
    if trigger_description:
        trigger_context += f"\nTRIGGER DESCRIPTION: {trigger_description}"
    if trigger_ingredients:
        trigger_context += f"\nTRIGGER PROVIDES: {', '.join(trigger_ingredients)}"

    prompt = f"""Select the best action for this automation query.

QUERY: {query}

{trigger_context}

ACTION OPTIONS:{options_text}

Think step-by-step:
1. What functionality does the user need? (focus on the verb/action, not brand names)
2. Read each action's DESCRIPTION carefully - what does it actually do?
3. Which action's description best matches the required functionality?

Based on your reasoning, answer with one letter ({'/'.join(valid_options)}):"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Show full LLM reasoning in verbose mode
        if config.verbose:
            print(f"    [LLM Re-ranker] Full response:")
            for line in answer.split('\n'):
                print(f"      {line}")

        answer_upper = answer.upper()

        # Parse response - find the answer letter more carefully
        # The naive "if opt in answer" matches A in "Aqara", "Action", etc.
        chosen_idx = None

        # Strategy 1: Check first line - LLM often starts with just the letter
        first_line = answer_upper.split('\n')[0].strip()
        for i, opt in enumerate(valid_options):
            if first_line == opt or first_line.startswith(f"{opt}:") or first_line.startswith(f"{opt} "):
                chosen_idx = i
                break

        # Strategy 2: Look for patterns like "is C:" or "Answer: C" or "Option C"
        if chosen_idx is None:
            for i, opt in enumerate(valid_options):
                patterns = [
                    rf'\bis\s+{opt}\b',           # "is C"
                    rf'answer[:\s]+{opt}\b',       # "Answer: C" or "Answer C"
                    rf'option\s+{opt}\b',          # "Option C"
                    rf'\b{opt}:\s+\w',             # "C: Switching"
                    rf'select(?:ed|ing)?\s+{opt}\b', # "select C" or "selected C"
                ]
                for pattern in patterns:
                    if re.search(pattern, answer_upper, re.IGNORECASE):
                        chosen_idx = i
                        break
                if chosen_idx is not None:
                    break

        if chosen_idx is not None:
            if config.verbose:
                opt = valid_options[chosen_idx]
                print(f"    [LLM Re-ranker] Picked {opt}: A{chosen_idx} ({actions[chosen_idx].get('service_name', 'Unknown')})")
            return chosen_idx

        if config.verbose:
            print(f"    [LLM Re-ranker] Unclear response - no valid option found")
        return None

    except Exception as e:
        if config.verbose:
            print(f"    [LLM Re-ranker] Error: {e}")
        return None


def action_selector_simple(state: ResolutionState) -> Dict[str, Any]:
    """Simple version (same as main now)."""
    return action_selector_node(state)
