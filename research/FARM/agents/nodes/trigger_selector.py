"""
Trigger Selector Node (Multi-Agent Offer-Verify Design)
=======================================================

MULTI-AGENT PATTERN:
1. Trigger Agent selects trigger and creates OFFER (ingredients it provides)
2. Action Agent receives offer and VERIFIES if it satisfies its needs
3. If rejected → try next pair

This is the TRIGGER AGENT - it makes offers to the Action Agent.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import NegotiationState
from agents.config import config, get_llm
from agents.prompts.trigger_agent import extract_ingredients_from_api

# Whether to use LLM re-ranking (recommended: True)
USE_LLM_RERANKER = True


def trigger_selector_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Select trigger using two-stage retrieval: RAG + LLM re-ranking.

    Stage 1: RAG retrieves top-5 candidates (fast, 90%+ R@5)
    Stage 2: LLM picks best from top-5 (accurate, handles ambiguity)

    Args:
        state: Current negotiation state

    Returns:
        State updates with selected trigger
    """
    trigger_candidates = state["trigger_candidates"]
    action_candidates = state.get("action_candidates", [])
    tried_pairs = list(state.get("tried_pairs") or [])
    pair_ranking = state.get("pair_ranking")  # Pre-computed ranking

    if not trigger_candidates:
        return {
            "error": "No trigger candidates available",
            "negotiation_status": "failed",
        }

    max_triggers = min(len(trigger_candidates), getattr(config, "top_k_candidates", 5))
    max_actions = min(len(action_candidates), getattr(config, "top_k_candidates", 5))

    query = state.get("query", "")
    used_reranker = False

    # Compute pair ranking on first call (use RAG scores)
    if pair_ranking is None:
        pair_ranking = _compute_pair_ranking(
            trigger_candidates[:max_triggers],
            action_candidates[:max_actions]
        )
        if config.verbose:
            print(f"\n    [Trigger Selector] RAG pair ranking ({len(pair_ranking)} pairs)")
            for i, (t, a, score) in enumerate(pair_ranking[:5]):
                print(f"      #{i+1}: T{t}+A{a} (score={score:.3f})")

        # Stage 2: LLM re-ranking of top-5 triggers
        if USE_LLM_RERANKER and len(trigger_candidates) >= 2:
            if config.verbose:
                print(f"\n    [LLM Re-ranker] Comparing top-{max_triggers} triggers...")

            # Get unique triggers from top pairs
            unique_triggers = []
            seen = set()
            for t_idx, _, _ in pair_ranking:
                if t_idx not in seen and len(unique_triggers) < max_triggers:
                    unique_triggers.append(t_idx)
                    seen.add(t_idx)

            # Ask LLM to pick best trigger
            best_trigger = _llm_rerank_triggers(
                query=query,
                triggers=[trigger_candidates[i] for i in unique_triggers],
                trigger_indices=unique_triggers,
            )

            if best_trigger is not None:
                # Agreement-based selection (not unconditional override)
                rag_best = pair_ranking[0][0]
                rag_best_score = trigger_candidates[rag_best].get("score", 0)
                llm_pick_score = trigger_candidates[best_trigger].get("score", 0)

                if best_trigger == rag_best:
                    if config.verbose:
                        print(f"    [LLM Re-ranker] Agreement: T{rag_best}")
                    used_reranker = False
                else:
                    score_ratio = llm_pick_score / rag_best_score if rag_best_score > 0 else 0
                    if score_ratio >= 0.95:
                        if config.verbose:
                            print(f"    [LLM Re-ranker] Override: T{rag_best} → T{best_trigger} (ratio={score_ratio:.2f})")
                        pair_ranking = _reorder_ranking(pair_ranking, best_trigger)
                        used_reranker = True
                    else:
                        if config.verbose:
                            print(f"    [LLM Re-ranker] Trust RAG: T{rag_best} (LLM pick ratio={score_ratio:.2f})")
                        used_reranker = False

    # Find best untried pair
    best_pair = None
    for t_idx, a_idx, score in pair_ranking:
        if (t_idx, a_idx) not in tried_pairs:
            best_pair = (t_idx, a_idx, score)
            break

    if best_pair is None:
        return {
            "error": "All pairs exhausted",
            "negotiation_status": "failed",
        }

    t_idx, a_idx, score = best_pair
    trigger = trigger_candidates[t_idx]
    ingredients = extract_ingredients_from_api(trigger.get("api_info", {}))

    offer = {
        "service_name": trigger["service_name"],
        "category": trigger["category"],
        "description": trigger.get("description", ""),
        "ingredients": ingredients,
    }

    reasoning = f"Pair rank score={score:.3f}"
    if used_reranker:
        reasoning += " (LLM re-ranked)"

    if config.verbose:
        print(f"\n    [Trigger Agent] Selected T{t_idx}: {trigger['service_name']}")
        ingredient_names = [ing.get("name", "") for ing in ingredients]
        print(f"    [Trigger Agent] OFFER: {len(ingredients)} ingredients → {ingredient_names[:3]}{'...' if len(ingredients) > 3 else ''}")

    return {
        "current_trigger_idx": t_idx,
        "suggested_action_idx": a_idx,  # Suggestion only - action_selector decides
        "current_offer": offer,
        "trigger_reasoning": reasoning,
        "pair_ranking": pair_ranking,  # Store for reuse
    }


def _compute_pair_ranking(
    triggers: List[Dict],
    actions: List[Dict]
) -> List[Tuple[int, int, float]]:
    """
    Compute ranking of all trigger-action pairs using RAG scores.

    Combined score = trigger_rag_score + action_rag_score

    Args:
        triggers: List of trigger candidates with 'score' field
        actions: List of action candidates with 'score' field

    Returns:
        List of (trigger_idx, action_idx, combined_score) sorted by score descending
    """
    pairs = []

    for t_idx, trigger in enumerate(triggers):
        t_score = trigger.get("score", 0.5)  # RAG score
        for a_idx, action in enumerate(actions):
            a_score = action.get("score", 0.5)  # RAG score
            combined = t_score + a_score
            pairs.append((t_idx, a_idx, combined))

    # Sort by combined score (best first)
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs


def _llm_rerank_triggers(
    query: str,
    triggers: List[Dict[str, Any]],
    trigger_indices: List[int],
) -> Optional[int]:
    """
    Use LLM to pick the best trigger from top-5 candidates.

    This is the key innovation: RAG retrieves candidates, LLM re-ranks.
    Industry standard two-stage retrieval pattern.

    Args:
        query: User query
        triggers: List of trigger candidates (up to 5)
        trigger_indices: Original indices of triggers

    Returns:
        Index of the best trigger, or None on error
    """
    # Build options string
    options = ["A", "B", "C", "D", "E"]
    options_text = ""
    for i, (trigger, idx) in enumerate(zip(triggers, trigger_indices)):
        if i >= 5:
            break
        name = trigger.get('service_name', 'Unknown')
        category = trigger.get('category', '')
        # Use FULL description - don't truncate! The details matter for disambiguation
        desc = trigger.get('description', 'No description')
        cat_str = f" [{category}]" if category else ""
        options_text += f"\n{options[i]}: {name}{cat_str}\n   {desc}\n"

    num_options = min(len(triggers), 5)
    valid_options = options[:num_options]

    prompt = f"""Select the best trigger for this automation query.

QUERY: {query}

OPTIONS:{options_text}

Read each option's DESCRIPTION carefully - it explains exactly what the trigger detects.

DIRECTION MEANINGS:
- "rises above" / "increases" = value goes UP past threshold
- "drops below" / "decreases" = value goes DOWN past threshold
- "changes" = any change in either direction

Match the query's intent to the trigger whose description fits best.

Answer with one letter ({'/'.join(valid_options)}):"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content.strip().upper()

        # Parse answer
        for i, opt in enumerate(valid_options):
            if opt in answer:
                chosen_idx = trigger_indices[i]
                if config.verbose:
                    print(f"    [LLM Re-ranker] Picked {opt}: T{chosen_idx} ({triggers[i].get('service_name', 'Unknown')})")
                return chosen_idx

        if config.verbose:
            print(f"    [LLM Re-ranker] Unclear response: {answer}")
        return None

    except Exception as e:
        if config.verbose:
            print(f"    [LLM Re-ranker] Error: {e}")
        return None


def _reorder_ranking(
    pair_ranking: List[Tuple[int, int, float]],
    preferred_trigger: int
) -> List[Tuple[int, int, float]]:
    """
    Reorder pair ranking to prioritize pairs with the preferred trigger.

    Args:
        pair_ranking: Current ranking
        preferred_trigger: Trigger index to prioritize

    Returns:
        New ranking with preferred trigger pairs first
    """
    preferred = []
    others = []

    for t_idx, a_idx, score in pair_ranking:
        if t_idx == preferred_trigger:
            preferred.append((t_idx, a_idx, score))
        else:
            others.append((t_idx, a_idx, score))

    return preferred + others


def trigger_selector_simple(state: NegotiationState) -> Dict[str, Any]:
    """Simple version without LLM tiebreaker."""
    # For simple mode, just use RAG ranking without tiebreaker
    trigger_candidates = state["trigger_candidates"]
    action_candidates = state.get("action_candidates", [])
    tried_pairs = list(state.get("tried_pairs") or [])
    pair_ranking = state.get("pair_ranking")

    if not trigger_candidates:
        return {
            "error": "No trigger candidates available",
            "negotiation_status": "failed",
        }

    max_triggers = min(len(trigger_candidates), getattr(config, "top_k_candidates", 5))
    max_actions = min(len(action_candidates), getattr(config, "top_k_candidates", 5))

    if pair_ranking is None:
        pair_ranking = _compute_pair_ranking(
            trigger_candidates[:max_triggers],
            action_candidates[:max_actions]
        )

    # Find best untried pair (no tiebreaker)
    best_pair = None
    for t_idx, a_idx, score in pair_ranking:
        if (t_idx, a_idx) not in tried_pairs:
            best_pair = (t_idx, a_idx, score)
            break

    if best_pair is None:
        return {
            "error": "All pairs exhausted",
            "negotiation_status": "failed",
        }

    t_idx, a_idx, score = best_pair
    trigger = trigger_candidates[t_idx]
    ingredients = extract_ingredients_from_api(trigger.get("api_info", {}))

    return {
        "current_trigger_idx": t_idx,
        "suggested_action_idx": a_idx,  # Suggestion only - action_selector decides
        "current_offer": {
            "service_name": trigger["service_name"],
            "category": trigger["category"],
            "description": trigger.get("description", ""),
            "ingredients": ingredients,
        },
        "trigger_reasoning": f"Pair rank score={score:.3f}",
        "pair_ranking": pair_ranking,
    }
