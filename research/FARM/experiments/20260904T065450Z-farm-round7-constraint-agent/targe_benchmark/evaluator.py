"""Strict, deterministic endpoint-pair ranking metrics.

The evaluator is independent of model/framework code.  References and ranked
predictions are structured endpoint identifiers; no fuzzy matching, LLM judge,
or error-path credit is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _sequence(value: Any, *, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{location} must be an array")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, location: str) -> None:
    actual = {str(key) for key in value}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys mismatch; missing={missing}, extra={extra}")


def _text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Endpoint:
    channel: str
    function: str

    def __post_init__(self) -> None:
        _text(self.channel, location="Endpoint.channel")
        _text(self.function, location="Endpoint.function")

    @classmethod
    def from_value(cls, value: Any, *, location: str) -> "Endpoint":
        obj = _mapping(value, location=location)
        _strict_keys(obj, {"channel", "function"}, location=location)
        return cls(
            channel=_text(obj["channel"], location=f"{location}.channel"),
            function=_text(obj["function"], location=f"{location}.function"),
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    trigger: Endpoint
    action: Endpoint

    def __post_init__(self) -> None:
        if not isinstance(self.trigger, Endpoint) or not isinstance(self.action, Endpoint):
            raise ValueError("Recipe.trigger and Recipe.action must be Endpoint instances")

    @classmethod
    def from_value(cls, value: Any, *, location: str) -> "Recipe":
        obj = _mapping(value, location=location)
        _strict_keys(obj, {"trigger", "action"}, location=location)
        return cls(
            trigger=Endpoint.from_value(obj["trigger"], location=f"{location}.trigger"),
            action=Endpoint.from_value(obj["action"], location=f"{location}.action"),
        )


def _cutoffs(values: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("cutoffs must be an array of positive integers")
    cutoffs: set[int] = set()
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"cutoffs[{index}] must be a positive integer")
        cutoffs.add(value)
    if not cutoffs:
        raise ValueError("at least one cutoff is required")
    return tuple(sorted(cutoffs))


T = TypeVar("T")


def _first_rank(ranking: Sequence[Recipe], target: T, project: Callable[[Recipe], T]) -> int | None:
    for rank, candidate in enumerate(ranking, start=1):
        if project(candidate) == target:
            return rank
    return None


def _rate(numerator: int, total: int) -> dict[str, int | float]:
    return {"correct": numerator, "total": total, "score": numerator / total}


def _mean(total_value: float, total: int) -> dict[str, int | float]:
    return {"sum": total_value, "total": total, "score": total_value / total}


def evaluate_rankings(
    references: Sequence[Recipe],
    rankings: Sequence[Sequence[Recipe]],
    *,
    cutoffs: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate aligned references/rankings with exact, case-sensitive equality.

    Empty rankings are valid failed predictions and receive zero for every
    endpoint and recipe metric.  Malformed types and alignment mismatches raise
    ``ValueError`` instead of being silently skipped.
    """

    reference_values = _sequence(references, location="references")
    ranking_values = _sequence(rankings, location="rankings")
    if len(reference_values) != len(ranking_values):
        raise ValueError(
            f"references/rankings length mismatch: {len(reference_values)} != {len(ranking_values)}"
        )
    if not reference_values:
        raise ValueError("references and rankings must contain at least one case")

    parsed_references: list[Recipe] = []
    parsed_rankings: list[tuple[Recipe, ...]] = []
    for index, reference in enumerate(reference_values):
        if not isinstance(reference, Recipe):
            raise ValueError(f"references[{index}] must be a Recipe")
        parsed_references.append(reference)
    for case_index, ranking in enumerate(ranking_values):
        candidates = _sequence(ranking, location=f"rankings[{case_index}]")
        parsed: list[Recipe] = []
        for rank_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Recipe):
                raise ValueError(f"rankings[{case_index}][{rank_index}] must be a Recipe")
            parsed.append(candidate)
        parsed_rankings.append(tuple(parsed))

    ks = _cutoffs(cutoffs)
    total = len(parsed_references)
    first_ranks: list[dict[str, int | None]] = []
    top1_counts = {"trigger_em": 0, "action_em": 0, "recipe_em": 0}

    for reference, ranking in zip(parsed_references, parsed_rankings, strict=True):
        trigger_rank = _first_rank(ranking, reference.trigger, lambda recipe: recipe.trigger)
        action_rank = _first_rank(ranking, reference.action, lambda recipe: recipe.action)
        recipe_rank = _first_rank(ranking, reference, lambda recipe: recipe)
        first_ranks.append(
            {"trigger": trigger_rank, "action": action_rank, "recipe": recipe_rank}
        )
        if trigger_rank == 1:
            top1_counts["trigger_em"] += 1
        if action_rank == 1:
            top1_counts["action_em"] += 1
        if recipe_rank == 1:
            top1_counts["recipe_em"] += 1

    at_k: dict[str, dict[str, dict[str, int | float]]] = {}
    for cutoff in ks:
        cutoff_metrics: dict[str, dict[str, int | float]] = {}
        for component in ("trigger", "action", "recipe"):
            ranks = [item[component] for item in first_ranks]
            hits = sum(rank is not None and rank <= cutoff for rank in ranks)
            reciprocal_sum = sum(
                1.0 / rank for rank in ranks if rank is not None and rank <= cutoff
            )
            cutoff_metrics[f"{component}_recall"] = _rate(hits, total)
            cutoff_metrics[f"{component}_mrr"] = _mean(reciprocal_sum, total)
        at_k[str(cutoff)] = cutoff_metrics

    full_ranking: dict[str, dict[str, int | float]] = {}
    for component in ("trigger", "action", "recipe"):
        reciprocal_sum = sum(
            1.0 / rank
            for rank in (item[component] for item in first_ranks)
            if rank is not None
        )
        full_ranking[f"{component}_mrr"] = _mean(reciprocal_sum, total)

    lengths = [len(ranking) for ranking in parsed_rankings]
    return {
        "schema_version": 1,
        "comparison": "case_sensitive_exact_endpoint_identifiers",
        "cases": total,
        "empty_rankings": sum(length == 0 for length in lengths),
        "ranking_lengths": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / total,
        },
        "top1": {
            metric: _rate(count, total) for metric, count in top1_counts.items()
        },
        "at_k": at_k,
        "full_ranking": full_ranking,
    }


def evaluate_payload(payload: Any, *, cutoffs: Sequence[int] = (1, 3, 5)) -> dict[str, Any]:
    """Parse and evaluate the strict JSON-compatible CLI payload schema."""

    obj = _mapping(payload, location="payload")
    _strict_keys(obj, {"references", "rankings"}, location="payload")
    references_raw = _sequence(obj["references"], location="payload.references")
    rankings_raw = _sequence(obj["rankings"], location="payload.rankings")
    if len(references_raw) != len(rankings_raw):
        raise ValueError(
            "payload.references/payload.rankings length mismatch: "
            f"{len(references_raw)} != {len(rankings_raw)}"
        )

    references = [
        Recipe.from_value(value, location=f"payload.references[{index}]")
        for index, value in enumerate(references_raw)
    ]
    rankings = [
        [
            Recipe.from_value(candidate, location=f"payload.rankings[{case_index}][{rank_index}]")
            for rank_index, candidate in enumerate(
                _sequence(ranking, location=f"payload.rankings[{case_index}]")
            )
        ]
        for case_index, ranking in enumerate(rankings_raw)
    ]
    return evaluate_rankings(references, rankings, cutoffs=cutoffs)
