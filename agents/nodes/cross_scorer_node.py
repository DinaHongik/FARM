"""
Cross-Scorer Node
=================

LangGraph node that scores all trigger-action pairs for compatibility.
This enables best-first search through the candidate space.

Handles the asymmetric position problem:
- Trigger answer might be at R@1, Action at R@5
- Cross-scoring finds compatible pairs regardless of RAG rank

Three scoring modes:
1. Schema-based (fast, deterministic, no LLM)
2. LLM-based (accurate, slower)
3. Hybrid (schema first, LLM for top candidates)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import ResolutionState
from agents.config import config


@dataclass
class PairScore:
    """Score for a trigger-action pair."""
    trigger_idx: int
    action_idx: int
    score: float
    coverage: float  # Ingredient-field coverage
    rag_score: float  # Combined RAG score
    reasoning: str


def cross_scorer_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Score all trigger-action pairs and create priority queue.

    This node:
    1. Gets all trigger and action candidates from state
    2. Computes compatibility score for each pair
    3. Creates sorted priority queue for resolution

    Args:
        state: Current resolution state with candidates

    Returns:
        State updates with pair_scores and priority_queue
    """
    trigger_candidates = state.get("trigger_candidates", [])
    action_candidates = state.get("action_candidates", [])

    if not trigger_candidates or not action_candidates:
        return {
            "error": "No candidates available for cross-scoring",
            "pair_scores": [],
            "priority_queue": [],
        }

    if config.verbose:
        print(f"\n[CROSS-SCORER] Scoring {len(trigger_candidates)}x{len(action_candidates)} pairs")

    # Score all pairs
    pair_scores = []

    for i, trigger in enumerate(trigger_candidates):
        for j, action in enumerate(action_candidates):
            score = compute_pair_score(trigger, action, i, j)
            pair_scores.append(score)

    # Sort by score descending
    pair_scores.sort(key=lambda x: x.score, reverse=True)

    # Create priority queue (list of (trigger_idx, action_idx) tuples)
    priority_queue = [(ps.trigger_idx, ps.action_idx) for ps in pair_scores]

    if config.verbose:
        print(f"[CROSS-SCORER] Top 5 pairs:")
        for ps in pair_scores[:5]:
            print(f"  (T{ps.trigger_idx}, A{ps.action_idx}): {ps.score:.3f} - {ps.reasoning[:50]}")

    # Convert to serializable format
    pair_scores_dict = [
        {
            "trigger_idx": ps.trigger_idx,
            "action_idx": ps.action_idx,
            "score": ps.score,
            "coverage": ps.coverage,
            "rag_score": ps.rag_score,
            "reasoning": ps.reasoning,
        }
        for ps in pair_scores
    ]

    return {
        "pair_scores": pair_scores_dict,
        "priority_queue": priority_queue,
        "current_pair_idx": 0,  # Start with best pair
    }


def compute_pair_score(
    trigger: Dict[str, Any],
    action: Dict[str, Any],
    trigger_idx: int,
    action_idx: int
) -> PairScore:
    """
    Compute compatibility score for a trigger-action pair.

    Uses schema-based scoring (Option A - fast, deterministic).

    Args:
        trigger: Trigger candidate
        action: Action candidate
        trigger_idx: Index in candidate list
        action_idx: Index in candidate list

    Returns:
        PairScore with compatibility metrics
    """
    # Extract ingredients from trigger
    trigger_api = trigger.get("api_info", {})
    ingredients_raw = trigger_api.get("Ingredients", {})

    ingredient_names = set()
    ingredient_types = {}

    for name, info in ingredients_raw.items():
        name_lower = name.lower()
        ingredient_names.add(name_lower)
        if isinstance(info, dict):
            slug = info.get("Slug", "").lower()
            if slug:
                ingredient_names.add(slug)
            ingredient_types[name_lower] = info.get("Type", "String")

    # Extract required fields from action
    action_api = action.get("api_info", {})
    fields_raw = action_api.get("Action fields", {})

    required_fields = []
    all_fields = []

    for name, info in fields_raw.items():
        name_lower = name.lower()
        all_fields.append(name_lower)
        if isinstance(info, dict):
            slug = info.get("Slug", "").lower()
            required = info.get("Required", "false").lower() == "true"
            if required:
                required_fields.append(name_lower)
                if slug:
                    required_fields.append(slug)

    # Remove duplicates
    required_fields = list(set(required_fields))

    # Calculate coverage score
    covered = 0
    coverage_details = []

    for field in required_fields:
        # Check direct match
        if field in ingredient_names:
            covered += 1
            coverage_details.append(f"{field}=direct")
            continue

        # Check partial match
        matched = False
        for ing in ingredient_names:
            if ing in field or field in ing:
                covered += 1
                coverage_details.append(f"{field}~{ing}")
                matched = True
                break

            # Semantic matching for common pairs
            if semantic_match(ing, field):
                covered += 0.8  # Partial credit for semantic match
                coverage_details.append(f"{field}~={ing}")
                matched = True
                break

        if not matched:
            coverage_details.append(f"{field}=MISSING")

    # Calculate coverage ratio
    total_required = len(set(required_fields))
    if total_required == 0:
        coverage = 1.0  # No required fields = compatible
    else:
        coverage = covered / total_required

    # Combine with RAG scores
    trigger_rag = trigger.get("score", 0.5)
    action_rag = action.get("score", 0.5)
    rag_score = (trigger_rag + action_rag) / 2

    # Final score: weighted combination
    # Coverage matters more than RAG rank for compatibility
    final_score = 0.7 * coverage + 0.3 * rag_score

    # Build reasoning
    trigger_name = trigger.get("service_name", f"Trigger[{trigger_idx}]")
    action_name = action.get("service_name", f"Action[{action_idx}]")

    reasoning = f"{trigger_name} -> {action_name}: "
    reasoning += f"coverage={coverage:.2f} ({covered}/{total_required} fields), "
    reasoning += f"rag={rag_score:.2f}"

    return PairScore(
        trigger_idx=trigger_idx,
        action_idx=action_idx,
        score=round(final_score, 4),
        coverage=round(coverage, 4),
        rag_score=round(rag_score, 4),
        reasoning=reasoning,
    )


def semantic_match(ingredient: str, field: str) -> bool:
    """
    Check for semantic similarity between ingredient and field names.

    Args:
        ingredient: Ingredient name (lowercase)
        field: Field name (lowercase)

    Returns:
        True if semantically related
    """
    # Common semantic mappings for WoT services
    mappings = {
        # Time-related
        "time": ["date", "timestamp", "when", "datetime", "created"],
        "date": ["time", "timestamp", "when", "datetime"],
        "timestamp": ["time", "date", "when", "created"],

        # Content-related
        "title": ["name", "subject", "header", "label"],
        "name": ["title", "subject", "label", "filename"],
        "content": ["message", "body", "text", "description", "data"],
        "message": ["content", "body", "text", "notification"],
        "body": ["content", "message", "text"],
        "text": ["content", "message", "body", "description"],
        "description": ["content", "text", "body", "summary"],

        # Link-related
        "url": ["link", "address", "href", "source"],
        "link": ["url", "address", "href"],

        # Media-related
        "image": ["photo", "picture", "media", "attachment"],
        "photo": ["image", "picture", "media"],

        # Person-related
        "author": ["user", "creator", "by", "from", "sender"],
        "user": ["author", "creator", "owner", "from"],
        "email": ["address", "mail", "sender", "recipient"],

        # Location-related
        "location": ["place", "address", "where", "position"],
        "address": ["location", "place", "street"],

        # Value-related
        "value": ["data", "amount", "number", "reading"],
        "data": ["value", "content", "reading", "row"],
        "row": ["data", "line", "entry", "formatted"],
    }

    # Check direct mapping
    if ingredient in mappings:
        if any(syn in field for syn in mappings[ingredient]):
            return True
        if field in mappings[ingredient]:
            return True

    if field in mappings:
        if any(syn in ingredient for syn in mappings[field]):
            return True
        if ingredient in mappings[field]:
            return True

    return False


def get_next_pair(state: ResolutionState) -> Optional[Tuple[int, int]]:
    """
    Get the next pair to try from priority queue.

    Args:
        state: Current state

    Returns:
        (trigger_idx, action_idx) tuple or None if exhausted
    """
    priority_queue = state.get("priority_queue", [])
    current_idx = state.get("current_pair_idx", 0)
    attempted = state.get("attempted_pairs", [])

    # Find next unattempted pair
    while current_idx < len(priority_queue):
        pair = priority_queue[current_idx]
        if pair not in attempted:
            return tuple(pair)
        current_idx += 1

    return None


def cross_scorer_simple(state: ResolutionState) -> Dict[str, Any]:
    """
    Simplified cross-scorer using only RAG scores.

    Fallback when schema information is incomplete.

    Args:
        state: Current state

    Returns:
        State updates with priority queue
    """
    trigger_candidates = state.get("trigger_candidates", [])
    action_candidates = state.get("action_candidates", [])

    # Simple scoring: just combine RAG scores
    pairs = []
    for i, t in enumerate(trigger_candidates):
        for j, a in enumerate(action_candidates):
            t_score = t.get("score", 0.5)
            a_score = a.get("score", 0.5)
            combined = (t_score + a_score) / 2
            pairs.append((i, j, combined))

    # Sort by score
    pairs.sort(key=lambda x: x[2], reverse=True)

    priority_queue = [(p[0], p[1]) for p in pairs]

    return {
        "priority_queue": priority_queue,
        "current_pair_idx": 0,
    }
