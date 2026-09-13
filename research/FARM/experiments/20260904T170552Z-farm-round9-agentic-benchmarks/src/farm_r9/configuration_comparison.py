"""Privacy-safe paired comparison of completed FARM configuration arms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from farm_r9.artifact_io import read_jsonl, sha256_file


EXPECTED_N = 150
ONE_SHOT_ARM = "single_shot_configurator"
BOUNDED_ARM = "bounded_configuration_agent"
METRICS = (
    "compiler_valid",
    "structurally_complete",
    "executable_ready_under_supplied_evidence",
)
_Z_95 = 1.959963984540054


class ComparisonInputError(ValueError):
    """Raised when a private ledger cannot safely support paired inference."""


def _load_completed_ledger(
    path: Path,
    *,
    expected_arm: str,
) -> dict[str, dict[str, bool]]:
    if not path.is_file():
        raise ComparisonInputError("configuration ledger is missing")
    try:
        records = read_jsonl(path)
    except (OSError, ValueError) as error:
        raise ComparisonInputError("configuration ledger is unreadable or malformed") from error
    if len(records) != EXPECTED_N:
        raise ComparisonInputError("configuration ledger must contain exactly 150 records")

    paired_values: dict[str, dict[str, bool]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ComparisonInputError("configuration ledger has a missing case ID")
        if case_id in paired_values:
            raise ComparisonInputError("configuration ledger has a duplicate case ID")
        if record.get("arm") != expected_arm:
            raise ComparisonInputError("configuration ledger has the wrong or mixed arm")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ComparisonInputError("configuration ledger has missing metrics")
        values: dict[str, bool] = {}
        for metric in METRICS:
            value = metrics.get(metric)
            if not isinstance(value, bool):
                raise ComparisonInputError("configuration ledger has a missing or non-boolean outcome")
            values[metric] = value
        paired_values[case_id] = values
    return paired_values


def _percentage(count: int, n: int = EXPECTED_N) -> float:
    return round(100.0 * count / n, 6)


def _wilson_95(successes: int, n: int = EXPECTED_N) -> dict[str, float | str]:
    proportion = successes / n
    denominator = 1 + _Z_95 * _Z_95 / n
    centre = (proportion + _Z_95 * _Z_95 / (2 * n)) / denominator
    radius = _Z_95 * math.sqrt(
        (proportion * (1 - proportion) + _Z_95 * _Z_95 / (4 * n)) / n
    ) / denominator
    return {
        "method": "wilson_95",
        "low": round(100.0 * max(0.0, centre - radius), 6),
        "high": round(100.0 * min(1.0, centre + radius), 6),
        "level": 95.0,
    }


def _exact_mcnemar(bounded_only: int, one_shot_only: int) -> float:
    discordant = bounded_only + one_shot_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, index)
        for index in range(min(bounded_only, one_shot_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * lower_tail)


def compare_configuration_ledgers(
    one_shot_records: Path,
    bounded_records: Path,
) -> dict:
    """Compare paired outcomes without returning IDs, content, paths, or traces."""

    one_shot = _load_completed_ledger(one_shot_records, expected_arm=ONE_SHOT_ARM)
    bounded = _load_completed_ledger(bounded_records, expected_arm=BOUNDED_ARM)
    if set(one_shot) != set(bounded):
        raise ComparisonInputError("paired configuration ledgers have different case-ID sets")

    metric_results = {}
    for metric in METRICS:
        cells = {
            "both_correct": 0,
            "bounded_only_rescue": 0,
            "one_shot_only_regression": 0,
            "neither": 0,
        }
        for case_id, one_shot_values in one_shot.items():
            one_shot_ok = one_shot_values[metric]
            bounded_ok = bounded[case_id][metric]
            if one_shot_ok and bounded_ok:
                cells["both_correct"] += 1
            elif bounded_ok:
                cells["bounded_only_rescue"] += 1
            elif one_shot_ok:
                cells["one_shot_only_regression"] += 1
            else:
                cells["neither"] += 1

        one_shot_correct = cells["both_correct"] + cells["one_shot_only_regression"]
        bounded_correct = cells["both_correct"] + cells["bounded_only_rescue"]
        raw_counts = {
            "one_shot_correct": one_shot_correct,
            "bounded_correct": bounded_correct,
            **cells,
        }
        metric_results[metric] = {
            "raw_counts": raw_counts,
            "percentages": {
                "one_shot": _percentage(one_shot_correct),
                "bounded": _percentage(bounded_correct),
                **{name: _percentage(count) for name, count in cells.items()},
            },
            "delta_percentage_points": round(
                100.0 * (bounded_correct - one_shot_correct) / EXPECTED_N,
                6,
            ),
            "confidence_intervals": {
                "one_shot": _wilson_95(one_shot_correct),
                "bounded": _wilson_95(bounded_correct),
            },
            "mcnemar_exact_two_sided_p": _exact_mcnemar(
                cells["bounded_only_rescue"],
                cells["one_shot_only_regression"],
            ),
        }

    return {
        "schema_version": "round9-configuration-paired-comparison-v1",
        "benchmark_label": "farm_v2_test",
        "n": EXPECTED_N,
        "arms": {
            "one_shot": ONE_SHOT_ARM,
            "bounded": BOUNDED_ARM,
        },
        "metrics": metric_results,
        "hashes": {
            "one_shot_records_sha256": sha256_file(one_shot_records),
            "bounded_records_sha256": sha256_file(bounded_records),
        },
    }


__all__ = ["ComparisonInputError", "compare_configuration_ledgers"]
