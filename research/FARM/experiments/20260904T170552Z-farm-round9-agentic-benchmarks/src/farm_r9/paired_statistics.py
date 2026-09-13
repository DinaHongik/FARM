"""Deterministic paired uncertainty and multiplicity utilities.

The bootstrap functions in this module accept only already-scored Boolean
case outcomes.  They never accept prompts, model responses, tool arguments,
or other content-bearing records, and their return values are aggregate-only.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 9_052_026
CONFIDENCE_LEVEL = 0.95


class PairedStatisticsError(ValueError):
    """Raised when outcomes cannot support the declared paired analysis."""


def _strict_probability(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise PairedStatisticsError(f"{name} must be a finite probability")
    return float(value)


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated sample quantile (Hyndman--Fan type 7)."""

    if not sorted_values:
        raise PairedStatisticsError("a bootstrap distribution cannot be empty")
    probability = _strict_probability(probability, name="quantile probability")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - weight)
        + float(sorted_values[upper]) * weight
    )


def paired_delta_bootstrap(
    current: Sequence[bool],
    reference: Sequence[bool],
    *,
    strata: Sequence[str] | None = None,
    expected_strata: Mapping[str, int] | None = None,
    expected_n: int | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the paired accuracy delta in percentage points.

    Each case, rather than each endpoint component or model call, is the
    resampling unit.  When ``strata`` is supplied, sampling occurs separately
    within every stratum while preserving all observed stratum sizes.
    """

    current_values = tuple(current)
    reference_values = tuple(reference)
    n = len(current_values)
    if n == 0 or len(reference_values) != n:
        raise PairedStatisticsError("paired outcomes must have equal nonzero length")
    if expected_n is not None and (
        isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n != n
    ):
        raise PairedStatisticsError("paired outcomes do not match the frozen sample size")
    if any(type(value) is not bool for value in (*current_values, *reference_values)):
        raise PairedStatisticsError("paired outcomes must be strict booleans")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples != BOOTSTRAP_RESAMPLES
    ):
        raise PairedStatisticsError("the reviewer protocol requires exactly 10000 resamples")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != BOOTSTRAP_SEED
    ):
        raise PairedStatisticsError("bootstrap seed differs from the frozen protocol")

    if strata is None:
        if expected_strata is not None:
            raise PairedStatisticsError("expected strata were supplied without case strata")
        groups = (tuple(range(n)),)
        method = "paired_case_bootstrap_percentile"
        stratification: dict[str, Any] = {
            "stratified": False,
            "stratum_counts": None,
        }
    else:
        stratum_values = tuple(strata)
        if len(stratum_values) != n or any(
            not isinstance(value, str) or not value for value in stratum_values
        ):
            raise PairedStatisticsError("every paired case must have one nonempty stratum")
        observed_counts = Counter(stratum_values)
        if expected_strata is None:
            raise PairedStatisticsError("stratified bootstrap requires frozen stratum quotas")
        if any(
            not isinstance(label, str)
            or not label
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for label, count in expected_strata.items()
        ):
            raise PairedStatisticsError("frozen stratum quotas are malformed")
        if dict(observed_counts) != dict(expected_strata):
            raise PairedStatisticsError("paired cases do not match the frozen stratum quotas")
        ordered_labels = tuple(sorted(expected_strata))
        groups = tuple(
            tuple(index for index, label in enumerate(stratum_values) if label == target)
            for target in ordered_labels
        )
        method = "stratified_paired_case_bootstrap_percentile"
        stratification = {
            "stratified": True,
            "stratification_variable": "frozen_ambiguity_stratum",
            "stratum_counts": {
                label: int(expected_strata[label]) for label in ordered_labels
            },
        }

    differences = tuple(
        int(current_value) - int(reference_value)
        for current_value, reference_value in zip(current_values, reference_values, strict=True)
    )
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        difference_sum = 0
        for group in groups:
            group_size = len(group)
            for _ in range(group_size):
                difference_sum += differences[group[rng.randrange(group_size)]]
        distribution.append(100.0 * difference_sum / n)
    distribution.sort()
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower = _type7_quantile(distribution, tail)
    upper = _type7_quantile(distribution, 1.0 - tail)
    observed = 100.0 * sum(differences) / n
    return {
        "method": method,
        "confidence_level": CONFIDENCE_LEVEL,
        "resamples": resamples,
        "seed": seed,
        "rng": "python_random_mt19937_v2",
        "quantile_method": "hyndman_fan_type_7",
        "experimental_unit": "paired_case",
        **stratification,
        "observed_delta_percentage_points": round(observed, 6),
        "low_percentage_points": round(lower, 6),
        "high_percentage_points": round(upper, 6),
    }


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    family_label: str,
    declared_family_size: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return deterministic Holm-adjusted p-values for one complete family.

    Callers must pass the *complete, predeclared* comparison family.  This
    helper deliberately cannot infer a family from whichever result files are
    currently convenient or complete.
    """

    if not isinstance(p_values, Mapping) or not p_values:
        raise PairedStatisticsError("Holm adjustment requires a nonempty complete family")
    if (
        not isinstance(family_label, str)
        or not family_label
        or len(family_label) > 128
        or any(character in family_label for character in "\r\n\t")
    ):
        raise PairedStatisticsError("Holm family label must be a short public label")
    if (
        isinstance(declared_family_size, bool)
        or not isinstance(declared_family_size, int)
        or declared_family_size <= 0
        or declared_family_size != len(p_values)
    ):
        raise PairedStatisticsError(
            "Holm inputs do not match the explicitly declared complete family size"
        )
    alpha_value = _strict_probability(alpha, name="alpha")
    if alpha_value in {0.0, 1.0}:
        raise PairedStatisticsError("alpha must be strictly between zero and one")
    validated: dict[str, float] = {}
    for label, value in p_values.items():
        if not isinstance(label, str) or not label:
            raise PairedStatisticsError("Holm family labels must be nonempty strings")
        validated[label] = _strict_probability(value, name="Holm p-value")

    ordered = sorted(validated.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_max = 0.0
    adjusted: dict[str, float] = {}
    for index, (label, p_value) in enumerate(ordered):
        running_max = max(running_max, (family_size - index) * p_value)
        adjusted[label] = min(1.0, running_max)
    adjusted_by_label = {label: adjusted[label] for label in sorted(adjusted)}
    return {
        "method": "holm_bonferroni",
        "family_label": family_label,
        "alpha": alpha_value,
        "family_size": family_size,
        "complete_family_asserted": True,
        "adjusted_p_values": adjusted_by_label,
        "reject_null": {
            label: adjusted_by_label[label] <= alpha_value
            for label in adjusted_by_label
        },
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONFIDENCE_LEVEL",
    "PairedStatisticsError",
    "holm_bonferroni",
    "paired_delta_bootstrap",
]
