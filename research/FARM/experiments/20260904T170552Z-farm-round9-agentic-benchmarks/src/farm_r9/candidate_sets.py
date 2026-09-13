"""Candidate fusion, opaque aliases, and deterministic retrieval metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from farm_r9.sampling import stable_order_key


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], *, constant: int = 60) -> list[tuple[str, float]]:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    normalized = sorted(tuple(ranking) for ranking in rankings)
    if len(normalized) < 2:
        raise ValueError("RRF requires at least two rankings")
    scores: dict[str, float] = defaultdict(float)
    for ranking in normalized:
        if len(ranking) != len(set(ranking)):
            raise ValueError("a source ranking contains duplicate identifiers")
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] += 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def opaque_candidate_view(
    candidates: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    side: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if side not in {"trigger", "action"}:
        raise ValueError("side must be trigger or action")
    if not 1 <= len(candidates) <= 99:
        raise ValueError("candidate view must contain 1..99 items")
    canonical_ids = [str(candidate["canonical_id"]) for candidate in candidates]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("candidate canonical IDs must be unique")
    prefix = "T" if side == "trigger" else "A"
    shuffled = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (stable_order_key(seed, case_id, side, item["canonical_id"]), item["canonical_id"]),
    )
    public, alias_map = [], {}
    forbidden = {"canonical_id", "rank", "score", "seed_ranks", "url"}
    for index, candidate in enumerate(shuffled, start=1):
        alias = f"{prefix}{index:02d}"
        alias_map[alias] = candidate["canonical_id"]
        public.append({"alias": alias, **{key: value for key, value in candidate.items() if key not in forbidden}})
    return public, alias_map


def rank_of_any(ranking: Sequence[str], gold: Iterable[str]) -> int | None:
    gold_set = set(gold)
    return next((rank for rank, identifier in enumerate(ranking, start=1) if identifier in gold_set), None)


def independent_recall(rows: Sequence[Mapping[str, Any]], *, cutoffs: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot score empty candidates")
    metrics: dict[str, Any] = {"n": len(rows)}
    for cutoff in cutoffs:
        trigger = sum(rank_of_any(row["trigger_ranking"], row["gold_trigger_ids"]) is not None and rank_of_any(row["trigger_ranking"], row["gold_trigger_ids"]) <= cutoff for row in rows)
        action = sum(rank_of_any(row["action_ranking"], row["gold_action_ids"]) is not None and rank_of_any(row["action_ranking"], row["gold_action_ids"]) <= cutoff for row in rows)
        joint = sum(
            any(
                pair["trigger_id"] in row["trigger_ranking"][:cutoff]
                and pair["action_id"] in row["action_ranking"][:cutoff]
                for pair in row["gold_pairs"]
            )
            for row in rows
        )
        for label, count in (("trigger", trigger), ("action", action), ("joint_independent", joint)):
            metrics[f"{label}_R@{cutoff}"] = {"correct": count, "n": len(rows), "value": count / len(rows)}
    return metrics
