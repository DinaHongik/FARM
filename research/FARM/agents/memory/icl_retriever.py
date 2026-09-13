"""
In-Context Learning (ICL) Example Retriever
==========================================

Retrieves similar training examples for few-shot prompting.
Uses the existing fine-tuned embeddings for similarity search.

This module does NOT modify the RAG system - it only uses
the existing embeddings for example retrieval.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ICLExample:
    """An in-context learning example."""
    query: str
    trigger_service: str
    action_service: str
    trigger_ingredients: List[str]
    action_fields: List[str]
    bindings: List[Dict[str, Any]]
    similarity: float = 0.0


class ICLRetriever:
    """
    Retrieves similar examples from training data for ICL.

    Uses simple keyword matching as baseline.
    Can be enhanced to use embedding similarity.
    """

    def __init__(self, examples_path: Optional[str] = None):
        """
        Initialize the ICL retriever.

        Args:
            examples_path: Path to training examples JSON file
        """
        self.examples: List[Dict[str, Any]] = []
        self._load_examples(examples_path)

    def _load_examples(self, path: Optional[str] = None):
        """Load training examples from file."""
        if path is None:
            # Default path to training data
            path = PROJECT_ROOT / "data" / "train_applets.json"

        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        self.examples = data[:1000]  # Limit for performance
                    elif isinstance(data, dict):
                        self.examples = list(data.values())[:1000]
            except Exception as e:
                print(f"Warning: Could not load ICL examples: {e}")
                self.examples = []
        else:
            # Try alternative path
            alt_path = PROJECT_ROOT / "data" / "iftttt_dataset_full_trigger_action.json"
            if Path(alt_path).exists():
                try:
                    with open(alt_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.examples = data[:1000]
                except Exception:
                    self.examples = []

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        trigger_hint: Optional[str] = None,
        action_hint: Optional[str] = None
    ) -> List[ICLExample]:
        """
        Retrieve similar examples for the given query.

        Args:
            query: User query
            top_k: Number of examples to retrieve
            trigger_hint: Optional hint about trigger type
            action_hint: Optional hint about action type

        Returns:
            List of ICLExample objects
        """
        if not self.examples:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_examples = []

        for ex in self.examples:
            score = self._compute_similarity(
                query_words,
                query_lower,
                ex,
                trigger_hint,
                action_hint
            )
            if score > 0:
                scored_examples.append((ex, score))

        # Sort by score descending
        scored_examples.sort(key=lambda x: x[1], reverse=True)

        # Convert to ICLExample objects
        results = []
        for ex, score in scored_examples[:top_k]:
            icl_example = self._convert_to_icl_example(ex, score)
            if icl_example:
                results.append(icl_example)

        return results

    def _compute_similarity(
        self,
        query_words: set,
        query_lower: str,
        example: Dict[str, Any],
        trigger_hint: Optional[str],
        action_hint: Optional[str]
    ) -> float:
        """
        Compute similarity score between query and example.

        Uses keyword overlap as baseline.
        """
        score = 0.0

        # Get example description
        desc = example.get("description", "").lower()
        desc_words = set(desc.split())

        # Keyword overlap
        overlap = len(query_words & desc_words)
        score += overlap * 0.1

        # Trigger service match
        trigger_service = example.get("trigger_service", "").lower()
        if trigger_hint and trigger_hint.lower() in trigger_service:
            score += 0.5
        elif any(word in trigger_service for word in query_words):
            score += 0.2

        # Action service match
        action_service = example.get("action_service", "").lower()
        if action_hint and action_hint.lower() in action_service:
            score += 0.5
        elif any(word in action_service for word in query_words):
            score += 0.2

        # Semantic keyword matching
        semantic_matches = {
            "log": ["spreadsheet", "sheet", "record", "save"],
            "notify": ["notification", "alert", "message", "send"],
            "dark": ["darkness", "light", "sensor", "brightness"],
            "email": ["mail", "message", "send"],
            "location": ["place", "geo", "area", "zone"],
        }

        for keyword, synonyms in semantic_matches.items():
            if keyword in query_lower:
                for syn in synonyms:
                    if syn in desc or syn in trigger_service or syn in action_service:
                        score += 0.3
                        break

        return score

    def _convert_to_icl_example(
        self,
        example: Dict[str, Any],
        similarity: float
    ) -> Optional[ICLExample]:
        """Convert raw example to ICLExample object."""
        try:
            # Extract trigger info
            trigger_service = example.get("trigger_service", "")
            trigger_api = example.get("trigger_api", {})
            trigger_ingredients = []

            if isinstance(trigger_api, dict):
                ingredients = trigger_api.get("Ingredients", {})
                if isinstance(ingredients, dict):
                    trigger_ingredients = list(ingredients.keys())

            # Extract action info
            action_service = example.get("action_service", "")
            action_api = example.get("action_api", {})
            action_fields = []

            if isinstance(action_api, dict):
                fields = action_api.get("Action fields", {})
                if isinstance(fields, dict):
                    action_fields = list(fields.keys())

            # Build example bindings (simplified)
            bindings = []
            for i, field in enumerate(action_fields[:3]):
                if i < len(trigger_ingredients):
                    bindings.append({
                        "action_field": field,
                        "source_type": "ingredient",
                        "ingredient_name": trigger_ingredients[i]
                    })

            return ICLExample(
                query=example.get("description", ""),
                trigger_service=trigger_service,
                action_service=action_service,
                trigger_ingredients=trigger_ingredients,
                action_fields=action_fields,
                bindings=bindings,
                similarity=similarity
            )

        except Exception:
            return None

    def format_for_prompt(self, examples: List[ICLExample]) -> str:
        """
        Format ICL examples for inclusion in prompts.

        Args:
            examples: List of ICL examples

        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""

        lines = ["Here are similar examples for reference:\n"]

        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Query: \"{ex.query}\"")
            lines.append(f"  Trigger: {ex.trigger_service}")
            lines.append(f"  Action: {ex.action_service}")
            if ex.bindings:
                lines.append(f"  Bindings: {len(ex.bindings)} fields bound")
            lines.append("")

        return "\n".join(lines)


def get_icl_examples(
    query: str,
    top_k: int = 3,
    trigger_hint: Optional[str] = None,
    action_hint: Optional[str] = None
) -> List[ICLExample]:
    """
    Convenience function to get ICL examples.

    Args:
        query: User query
        top_k: Number of examples
        trigger_hint: Optional trigger hint
        action_hint: Optional action hint

    Returns:
        List of ICL examples
    """
    retriever = ICLRetriever()
    return retriever.retrieve(query, top_k, trigger_hint, action_hint)
