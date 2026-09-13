"""Deterministic, hash-bound stratified sampling."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence, TypeVar


T = TypeVar("T", bound=Mapping[str, Any])


def stable_order_key(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` proportionally with deterministic largest remainder."""
    positive = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    available = sum(positive.values())
    if total < 1 or total > available:
        raise ValueError(f"sample size must be in [1, {available}]")
    ideals = {key: total * count / available for key, count in positive.items()}
    quotas = {key: min(count, math.floor(ideals[key])) for key, count in positive.items()}

    # When possible, retain every prespecified stratum in the audit sample.
    if total >= len(positive):
        for key in sorted(positive):
            if quotas[key] == 0:
                quotas[key] = 1

    while sum(quotas.values()) > total:
        candidates = [key for key in positive if quotas[key] > 1]
        if not candidates:
            raise AssertionError("unable to reduce proportional quotas")
        key = min(candidates, key=lambda item: (ideals[item] - quotas[item], item))
        quotas[key] -= 1
    while sum(quotas.values()) < total:
        candidates = [key for key in positive if quotas[key] < positive[key]]
        if not candidates:
            raise AssertionError("unable to fill proportional quotas")
        key = max(candidates, key=lambda item: (ideals[item] - quotas[item], -ord(item[0]) if item else 0, item))
        quotas[key] += 1
    return dict(sorted(quotas.items()))


def stratified_sample(
    rows: Sequence[T],
    *,
    size: int,
    seed: int,
    benchmark: str,
    id_of: Callable[[T], str],
    stratum_of: Callable[[T], str],
    quotas: Mapping[str, int] | None = None,
) -> tuple[list[T], dict[str, Any]]:
    if len({id_of(row) for row in rows}) != len(rows):
        raise ValueError("sampling IDs must be unique")
    buckets: dict[str, list[T]] = defaultdict(list)
    for row in rows:
        buckets[str(stratum_of(row))].append(row)
    counts = {key: len(value) for key, value in buckets.items()}
    chosen_quotas = proportional_quotas(counts, size) if quotas is None else {
        str(key): int(value) for key, value in quotas.items()
    }
    if sum(chosen_quotas.values()) != size:
        raise ValueError("explicit quotas do not sum to requested size")
    if set(chosen_quotas) - set(buckets):
        raise ValueError("quota references an unknown stratum")
    if any(chosen_quotas[key] > len(buckets[key]) for key in chosen_quotas):
        raise ValueError("quota exceeds available rows")

    selected: list[T] = []
    for stratum in sorted(chosen_quotas):
        ordered = sorted(
            buckets[stratum],
            key=lambda row: (stable_order_key(seed, benchmark, stratum, id_of(row)), id_of(row)),
        )
        selected.extend(ordered[: chosen_quotas[stratum]])
    selected.sort(key=lambda row: (stable_order_key(seed, benchmark, "final", id_of(row)), id_of(row)))
    realized = Counter(stratum_of(row) for row in selected)
    manifest = {
        "benchmark": benchmark,
        "seed": seed,
        "sampling": "deterministic-stratified-sha256-v1",
        "population_size": len(rows),
        "sample_size": len(selected),
        "population_strata": dict(sorted(counts.items())),
        "sample_strata": dict(sorted((str(key), int(value)) for key, value in realized.items())),
        "quotas": chosen_quotas,
        "ordered_case_ids": [id_of(row) for row in selected],
    }
    return selected, manifest
