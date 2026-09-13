"""Strict aggregate-only public reporting for the frozen Yao experiment.

The private run aggregate is already free of model transcripts, but its nested
Yao-specific fields do not fit the generic FARM aggregate exporter.  This
module validates every accepted field and every reported arithmetic identity
before constructing a small, detached public report.  Unexpected fields are
rejected rather than silently discarded.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from farm_r9.paired_statistics import PairedStatisticsError, paired_delta_bootstrap
from farm_r9.yao_agent import COMPONENTS, exact_mcnemar, wilson_interval


EXPECTED_N = 150
EXPECTED_SAMPLE_SHA256 = (
    "5543d5e331033147a960392c474f5bb2b1560e2d9a7ca1942e8de8d06aa345ff"
)
METRICS = (*COMPONENTS, "joint")
EXPECTED_STRATA = {"CI": 28, "VI-1": 18, "VI-2": 32, "VI-3": 10, "VI-4": 62}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "benchmark_label",
    "arm",
    "n",
    "raw_numerators",
    "percentages",
    "confidence_intervals",
    "terminal_status_counts",
    "failure_counts",
    "questions",
    "usage",
    "by_stratum",
    "hashes",
    "model_metadata",
    "protocol_metadata",
    "cost",
    "paired_outcomes",
}
_USAGE_INTEGER_KEYS = (
    "semantic_calls",
    "prompt_tokens",
    "completion_tokens",
    "physical_attempts",
    "strict_json_outputs",
    "cache_hits",
)
_USAGE_NUMBER_KEYS = ("provider_latency_ms", "queue_wait_ms")
_ASK_POLICY_KEYS = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
)
_PAIR_KEYS = (
    "both_correct",
    "current_only",
    "reference_only",
    "both_wrong",
    "n",
    "delta_percentage_points",
    "mcnemar_exact_two_sided_p",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+@()-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROVIDER_SECRET = re.compile(
    r"(?:^sk-[A-Za-z0-9_-]{16,}$|^[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{10,})"
)
_FAILURE_CODE_SLUGS = {
    "no_commit_within_budget": "no_commit_within_budget",
    "non_strict_json_envelope": "non_strict_json_envelope",
    "strict action schema validation failed": "strict_action_schema_validation_failed",
    "model action is not allowed by the frozen arm": "model_action_not_allowed",
    "model output must be one JSON object": "model_output_not_object",
    "no valid JSON object in model output": "model_output_invalid_json",
    "repeated_component_question": "repeated_component_question",
    "question_budget_exhausted": "question_budget_exhausted",
    "frozen_answer_binding_changed": "frozen_answer_binding_changed",
    **{
        f"prediction is outside canonical {component} vocabulary": f"prediction_outside_{component}_vocabulary"
        for component in COMPONENTS
    },
}


class YaoPublicReportError(ValueError):
    """Raised when an input cannot satisfy the public aggregate contract."""


@dataclass(frozen=True)
class _TerminalProjection:
    stratum: str
    scores: dict[str, bool]
    terminal_status: str
    public_failure_code: str | None


def _object(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise YaoPublicReportError(f"{path} must be an object with string keys")
    return value


def _exact_fields(
    value: Any,
    *,
    expected: set[str] | frozenset[str],
    path: str,
    optional: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    value = _object(value, path=path)
    if set(value) - expected or (expected - optional) - set(value):
        raise YaoPublicReportError(f"{path} has unexpected or missing fields")
    return value


def _count(value: Any, *, path: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise YaoPublicReportError(f"{path} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise YaoPublicReportError(f"{path} exceeds its aggregate denominator")
    return value


def _number(value: Any, *, path: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise YaoPublicReportError(f"{path} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise YaoPublicReportError(f"{path} is below its permitted range")
    return result


def _same(actual: float, expected: float, *, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise YaoPublicReportError(f"{path} is arithmetically inconsistent")


def _identifier(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or ".." in value
        or _PROVIDER_SECRET.search(value)
    ):
        raise YaoPublicReportError(f"{path} must be a safe public identifier")
    return value


def _fixed(value: Any, expected: str, *, path: str) -> str:
    if value != expected:
        raise YaoPublicReportError(f"{path} does not match the frozen protocol")
    return expected


def _metric_report(
    *,
    raw: Any,
    percentages: Any,
    intervals: Any,
    n: int,
    path: str,
) -> tuple[dict[str, int], dict[str, float], dict[str, dict[str, Any]]]:
    raw = _exact_fields(raw, expected=set(METRICS), path=f"{path}.raw_numerators")
    percentages = _exact_fields(
        percentages, expected=set(METRICS), path=f"{path}.percentages"
    )
    intervals = _exact_fields(
        intervals, expected=set(METRICS), path=f"{path}.confidence_intervals"
    )
    public_raw: dict[str, int] = {}
    public_percentages: dict[str, float] = {}
    public_intervals: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        correct = _count(raw[metric], path=f"{path}.raw_numerators.{metric}", maximum=n)
        percentage = _number(percentages[metric], path=f"{path}.percentages.{metric}")
        if not 0 <= percentage <= 100:
            raise YaoPublicReportError(f"{path}.percentages.{metric} is outside 0..100")
        _same(percentage, correct * 100 / n, path=f"{path}.percentages.{metric}")
        interval = _exact_fields(
            intervals[metric],
            expected={"method", "percent"},
            path=f"{path}.confidence_intervals.{metric}",
        )
        _fixed(
            interval["method"],
            "wilson_95",
            path=f"{path}.confidence_intervals.{metric}.method",
        )
        bounds = interval["percent"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise YaoPublicReportError(
                f"{path}.confidence_intervals.{metric}.percent must contain two bounds"
            )
        low = _number(bounds[0], path=f"{path}.confidence_intervals.{metric}.low")
        high = _number(bounds[1], path=f"{path}.confidence_intervals.{metric}.high")
        if not 0 <= low <= high <= 100:
            raise YaoPublicReportError(
                f"{path}.confidence_intervals.{metric} has invalid bounds"
            )
        expected_low, expected_high = wilson_interval(correct, n)
        _same(low, expected_low, path=f"{path}.confidence_intervals.{metric}.low")
        _same(high, expected_high, path=f"{path}.confidence_intervals.{metric}.high")
        public_raw[metric] = correct
        public_percentages[metric] = percentage
        public_intervals[metric] = {
            "method": "wilson_95",
            "low_percent": low,
            "high_percent": high,
        }
    return public_raw, public_percentages, public_intervals


def _questions(value: Any, *, n: int) -> dict[str, Any]:
    fields = {
        "total",
        "mean_per_case",
        "zero_question_cases",
        "count_distribution",
        "by_component",
        "ask_policy_confusion",
        "ask_policy_precision",
        "ask_policy_recall",
    }
    value = _exact_fields(value, expected=fields, path="$.questions")
    total = _count(value["total"], path="$.questions.total", maximum=4 * n)
    mean = _number(value["mean_per_case"], path="$.questions.mean_per_case", minimum=0)
    _same(mean, total / n, path="$.questions.mean_per_case")
    zero = _count(
        value["zero_question_cases"], path="$.questions.zero_question_cases", maximum=n
    )

    distribution = _object(
        value["count_distribution"], path="$.questions.count_distribution"
    )
    if set(distribution) - {str(index) for index in range(5)}:
        raise YaoPublicReportError("$.questions.count_distribution has an invalid bin")
    public_distribution = {
        str(index): _count(
            distribution.get(str(index), 0),
            path=f"$.questions.count_distribution.{index}",
            maximum=n,
        )
        for index in range(5)
    }
    if sum(public_distribution.values()) != n:
        raise YaoPublicReportError("$.questions.count_distribution does not sum to n")
    if sum(int(key) * count for key, count in public_distribution.items()) != total:
        raise YaoPublicReportError(
            "$.questions.count_distribution disagrees with total"
        )
    if public_distribution["0"] != zero:
        raise YaoPublicReportError(
            "$.questions.zero_question_cases disagrees with distribution"
        )

    by_component = _exact_fields(
        value["by_component"], expected=set(COMPONENTS), path="$.questions.by_component"
    )
    public_by_component = {
        component: _count(
            by_component[component],
            path=f"$.questions.by_component.{component}",
            maximum=n,
        )
        for component in COMPONENTS
    }
    if sum(public_by_component.values()) != total:
        raise YaoPublicReportError("$.questions.by_component disagrees with total")

    confusion = _exact_fields(
        value["ask_policy_confusion"],
        expected=set(_ASK_POLICY_KEYS),
        path="$.questions.ask_policy_confusion",
    )
    public_confusion = {
        key: _count(
            confusion[key],
            path=f"$.questions.ask_policy_confusion.{key}",
            maximum=4 * n,
        )
        for key in _ASK_POLICY_KEYS
    }
    if sum(public_confusion.values()) != 4 * n:
        raise YaoPublicReportError(
            "$.questions.ask_policy_confusion does not cover 4*n decisions"
        )
    if public_confusion["true_positive"] + public_confusion["false_positive"] != total:
        raise YaoPublicReportError(
            "$.questions.ask_policy_confusion disagrees with asks"
        )

    precision_denominator = (
        public_confusion["true_positive"] + public_confusion["false_positive"]
    )
    recall_denominator = (
        public_confusion["true_positive"] + public_confusion["false_negative"]
    )
    precision = value["ask_policy_precision"]
    recall = value["ask_policy_recall"]
    expected_precision = (
        public_confusion["true_positive"] / precision_denominator
        if precision_denominator
        else None
    )
    expected_recall = (
        public_confusion["true_positive"] / recall_denominator
        if recall_denominator
        else None
    )
    public_precision = _nullable_ratio(
        precision, expected_precision, path="$.questions.ask_policy_precision"
    )
    public_recall = _nullable_ratio(
        recall, expected_recall, path="$.questions.ask_policy_recall"
    )
    return {
        "total": total,
        "mean_per_case": mean,
        "zero_question_cases": zero,
        "count_distribution": public_distribution,
        "by_component": public_by_component,
        "ask_policy_confusion": public_confusion,
        "ask_policy_precision": public_precision,
        "ask_policy_recall": public_recall,
    }


def _nullable_ratio(value: Any, expected: float | None, *, path: str) -> float | None:
    if expected is None:
        if value is not None:
            raise YaoPublicReportError(f"{path} must be null for a zero denominator")
        return None
    actual = _number(value, path=path)
    if not 0 <= actual <= 1:
        raise YaoPublicReportError(f"{path} is outside 0..1")
    _same(actual, expected, path=path)
    return actual


def _usage(value: Any, *, n: int) -> dict[str, Any]:
    value = _exact_fields(value, expected={"totals", "means_per_case"}, path="$.usage")
    expected_keys = set(_USAGE_INTEGER_KEYS) | set(_USAGE_NUMBER_KEYS)
    totals = _exact_fields(
        value["totals"], expected=expected_keys, path="$.usage.totals"
    )
    means = _exact_fields(
        value["means_per_case"], expected=expected_keys, path="$.usage.means_per_case"
    )
    public_totals: dict[str, int | float] = {}
    public_means: dict[str, float] = {}
    for key in _USAGE_INTEGER_KEYS:
        total = _count(totals[key], path=f"$.usage.totals.{key}")
        mean = _number(means[key], path=f"$.usage.means_per_case.{key}", minimum=0)
        _same(mean, total / n, path=f"$.usage.means_per_case.{key}")
        public_totals[key] = total
        public_means[key] = mean
    for key in _USAGE_NUMBER_KEYS:
        total = _number(totals[key], path=f"$.usage.totals.{key}", minimum=0)
        mean = _number(means[key], path=f"$.usage.means_per_case.{key}", minimum=0)
        _same(mean, total / n, path=f"$.usage.means_per_case.{key}")
        public_totals[key] = total
        public_means[key] = mean
    if public_totals["cache_hits"] > public_totals["semantic_calls"]:
        raise YaoPublicReportError("$.usage cache hits exceed semantic calls")
    if public_totals["strict_json_outputs"] > public_totals["semantic_calls"]:
        raise YaoPublicReportError("$.usage strict outputs exceed semantic calls")
    total_tokens = int(public_totals["prompt_tokens"]) + int(
        public_totals["completion_tokens"]
    )
    public_totals["total_tokens"] = total_tokens
    public_means["total_tokens"] = total_tokens / n
    return {"totals": public_totals, "means_per_case": public_means}


def _cost(value: Any, *, usage: Mapping[str, Any]) -> dict[str, Any]:
    value = _exact_fields(
        value,
        expected={
            "currency",
            "prompt_cost_per_million",
            "completion_cost_per_million",
            "estimated_total",
        },
        path="$.cost",
    )
    _fixed(value["currency"], "USD", path="$.cost.currency")
    prompt_rate = _nullable_nonnegative(
        value["prompt_cost_per_million"], path="$.cost.prompt_cost_per_million"
    )
    completion_rate = _nullable_nonnegative(
        value["completion_cost_per_million"], path="$.cost.completion_cost_per_million"
    )
    if (prompt_rate is None) != (completion_rate is None):
        raise YaoPublicReportError(
            "$.cost pricing must be wholly available or wholly unavailable"
        )
    estimated = value["estimated_total"]
    if prompt_rate is None:
        if estimated is not None:
            raise YaoPublicReportError(
                "$.cost.estimated_total must be null without pricing"
            )
        public_estimated = None
    else:
        public_estimated = _number(estimated, path="$.cost.estimated_total", minimum=0)
        totals = usage["totals"]
        expected = (
            totals["prompt_tokens"] * prompt_rate
            + totals["completion_tokens"] * completion_rate
        ) / 1_000_000
        _same(public_estimated, expected, path="$.cost.estimated_total")
    return {
        "currency": "USD",
        "available": prompt_rate is not None,
        "prompt_cost_per_million": prompt_rate,
        "completion_cost_per_million": completion_rate,
        "estimated_total": public_estimated,
    }


def _nullable_nonnegative(value: Any, *, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path=path, minimum=0)


def _terminal_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    path: str,
) -> dict[str, _TerminalProjection]:
    """Validate only the score/stratum projection needed for paired reporting."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise YaoPublicReportError(f"{path} must be a terminal-record sequence")
    if len(records) != EXPECTED_N:
        raise YaoPublicReportError(f"{path} must contain exactly 150 terminal records")
    projected: dict[str, _TerminalProjection] = {}
    strata: Counter[str] = Counter()
    for record_value in records:
        record = _object(record_value, path=f"{path}.<record>")
        if record.get("schema_version") != "round9-yao-case-result-v1":
            raise YaoPublicReportError(f"{path} has an invalid case-result schema")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise YaoPublicReportError(f"{path} has a missing case ID")
        if case_id in projected:
            raise YaoPublicReportError(f"{path} has a duplicate case ID")
        if record.get("arm") != arm:
            raise YaoPublicReportError(f"{path} has a record from the wrong arm")
        if record.get("terminal") is not True:
            raise YaoPublicReportError(f"{path} has a nonterminal record")
        stratum = record.get("stratum")
        if stratum not in EXPECTED_STRATA:
            raise YaoPublicReportError(f"{path} has an unknown ambiguity stratum")
        scores_value = _exact_fields(
            record.get("scores"), expected=set(METRICS), path=f"{path}.<record>.scores"
        )
        if any(type(scores_value[metric]) is not bool for metric in METRICS):
            raise YaoPublicReportError(f"{path} scores must be JSON booleans")
        scores = {metric: scores_value[metric] for metric in METRICS}
        if scores["joint"] != all(scores[component] for component in COMPONENTS):
            raise YaoPublicReportError(f"{path} has an inconsistent joint score")
        terminal_status = record.get("terminal_status")
        if terminal_status not in {"committed", "protocol_failure"}:
            raise YaoPublicReportError(f"{path} has an invalid terminal status")
        failure_code = record.get("failure_code")
        if terminal_status == "committed":
            if failure_code is not None:
                raise YaoPublicReportError(
                    f"{path} has a failure code on a committed record"
                )
            public_failure_code = None
        else:
            if (
                not isinstance(failure_code, str)
                or failure_code not in _FAILURE_CODE_SLUGS
            ):
                raise YaoPublicReportError(
                    f"{path} has an invalid protocol-failure category"
                )
            if any(scores.values()):
                raise YaoPublicReportError(
                    f"{path} counts a protocol failure as correct"
                )
            public_failure_code = _FAILURE_CODE_SLUGS[failure_code]
        projected[case_id] = _TerminalProjection(
            stratum=stratum,
            scores=scores,
            terminal_status=terminal_status,
            public_failure_code=public_failure_code,
        )
        strata[stratum] += 1
    if dict(strata) != EXPECTED_STRATA:
        raise YaoPublicReportError(f"{path} does not match the frozen stratum quotas")
    return projected


def _paired(
    value: Any,
    *,
    n: int,
    arm: str,
    current_records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    current_raw: Mapping[str, int],
    current_by_stratum: Mapping[str, Any],
    current_statuses: Mapping[str, int],
    current_failures: Mapping[str, int],
) -> dict[str, Any]:
    if arm != "bounded_clarification_agent":
        raise YaoPublicReportError(
            "$.paired_outcomes can name rescue/regression only for agent versus one-shot"
        )
    value = _exact_fields(value, expected=set(METRICS), path="$.paired_outcomes")
    current = _terminal_ledger(
        current_records,
        arm="bounded_clarification_agent",
        path="$.current_records",
    )
    reference = _terminal_ledger(
        reference_records,
        arm="same_model_one_shot",
        path="$.reference_records",
    )
    if set(current) != set(reference):
        raise YaoPublicReportError("paired Yao ledgers have different case-ID sets")
    ordered_case_ids = tuple(sorted(current))
    if any(
        current[case_id].stratum != reference[case_id].stratum
        for case_id in ordered_case_ids
    ):
        raise YaoPublicReportError("paired Yao ledgers disagree on ambiguity strata")
    ledger_statuses = Counter(
        current[case_id].terminal_status for case_id in ordered_case_ids
    )
    if {
        status: ledger_statuses[status] for status in ("committed", "protocol_failure")
    } != dict(current_statuses):
        raise YaoPublicReportError(
            "current Yao ledger disagrees with terminal statuses"
        )
    ledger_failures = Counter(
        projection.public_failure_code
        for projection in current.values()
        if projection.public_failure_code is not None
    )
    if dict(sorted(ledger_failures.items())) != dict(current_failures):
        raise YaoPublicReportError("current Yao ledger disagrees with failure counts")
    for metric in METRICS:
        if (
            sum(current[case_id].scores[metric] for case_id in ordered_case_ids)
            != current_raw[metric]
        ):
            raise YaoPublicReportError(
                "current Yao ledger disagrees with its aggregate"
            )
        for stratum, expected_n in EXPECTED_STRATA.items():
            stratum_values = [
                current[case_id].scores[metric]
                for case_id in ordered_case_ids
                if current[case_id].stratum == stratum
            ]
            if (
                len(stratum_values) != expected_n
                or sum(stratum_values)
                != (current_by_stratum[stratum]["raw_numerators"][metric])
            ):
                raise YaoPublicReportError(
                    "current Yao ledger disagrees with its stratum aggregate"
                )
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        cells = _exact_fields(
            value[metric], expected=set(_PAIR_KEYS), path=f"$.paired_outcomes.{metric}"
        )
        both_correct = _count(
            cells["both_correct"],
            path=f"$.paired_outcomes.{metric}.both_correct",
            maximum=n,
        )
        rescue = _count(
            cells["current_only"],
            path=f"$.paired_outcomes.{metric}.current_only",
            maximum=n,
        )
        regression = _count(
            cells["reference_only"],
            path=f"$.paired_outcomes.{metric}.reference_only",
            maximum=n,
        )
        both_wrong = _count(
            cells["both_wrong"],
            path=f"$.paired_outcomes.{metric}.both_wrong",
            maximum=n,
        )
        cell_n = _count(cells["n"], path=f"$.paired_outcomes.{metric}.n")
        if cell_n != n or both_correct + rescue + regression + both_wrong != n:
            raise YaoPublicReportError(
                f"$.paired_outcomes.{metric} cells do not sum to n"
            )
        delta = _number(
            cells["delta_percentage_points"],
            path=f"$.paired_outcomes.{metric}.delta_percentage_points",
        )
        _same(
            delta,
            (rescue - regression) * 100 / n,
            path=f"$.paired_outcomes.{metric}.delta_percentage_points",
        )
        p_value = _number(
            cells["mcnemar_exact_two_sided_p"],
            path=f"$.paired_outcomes.{metric}.mcnemar_exact_two_sided_p",
        )
        if not 0 <= p_value <= 1:
            raise YaoPublicReportError(
                f"$.paired_outcomes.{metric} McNemar p is outside 0..1"
            )
        _same(
            p_value,
            exact_mcnemar(rescue, regression),
            path=f"$.paired_outcomes.{metric}.mcnemar_exact_two_sided_p",
        )
        current_values = tuple(
            current[case_id].scores[metric] for case_id in ordered_case_ids
        )
        reference_values = tuple(
            reference[case_id].scores[metric] for case_id in ordered_case_ids
        )
        recomputed_cells = {
            "both_correct": sum(
                current_value and reference_value
                for current_value, reference_value in zip(
                    current_values, reference_values, strict=True
                )
            ),
            "current_only": sum(
                current_value and not reference_value
                for current_value, reference_value in zip(
                    current_values, reference_values, strict=True
                )
            ),
            "reference_only": sum(
                not current_value and reference_value
                for current_value, reference_value in zip(
                    current_values, reference_values, strict=True
                )
            ),
            "both_wrong": sum(
                not current_value and not reference_value
                for current_value, reference_value in zip(
                    current_values, reference_values, strict=True
                )
            ),
        }
        if recomputed_cells != {
            "both_correct": both_correct,
            "current_only": rescue,
            "reference_only": regression,
            "both_wrong": both_wrong,
        }:
            raise YaoPublicReportError(
                "paired Yao ledgers disagree with the aggregate contingency table"
            )
        try:
            uncertainty = paired_delta_bootstrap(
                current_values,
                reference_values,
                strata=tuple(current[case_id].stratum for case_id in ordered_case_ids),
                expected_strata=EXPECTED_STRATA,
                expected_n=n,
            )
        except PairedStatisticsError as error:
            raise YaoPublicReportError(
                "paired Yao ledgers cannot satisfy the uncertainty protocol"
            ) from error
        _same(
            uncertainty["observed_delta_percentage_points"],
            round(delta, 6),
            path=f"$.paired_outcomes.{metric}.bootstrap_observed_delta",
        )
        metrics[metric] = {
            "n": n,
            "both_correct_count": both_correct,
            "rescue_count": rescue,
            "regression_count": regression,
            "both_wrong_count": both_wrong,
            "rescue_percent": rescue * 100 / n,
            "regression_percent": regression * 100 / n,
            "delta_percentage_points": delta,
            "delta_confidence_interval_95": uncertainty,
            "mcnemar_exact_two_sided_p": p_value,
        }
    return {
        "reference_arm": "same_model_one_shot",
        "metrics": metrics,
        "multiplicity_policy": {
            "confirmatory_metric": "joint",
            "family_unit": "predeclared_agent_or_model_comparison",
            "rule": "holm_bonferroni_if_complete_family_has_more_than_one_comparison",
            "status": "family_not_inferred_from_a_single_comparison_artifact",
        },
    }


def _statuses(value: Any, *, n: int) -> dict[str, int]:
    value = _object(value, path="$.terminal_status_counts")
    if set(value) - {"committed", "protocol_failure"}:
        raise YaoPublicReportError("$.terminal_status_counts has an unknown status")
    result = {
        key: _count(
            value.get(key, 0), path=f"$.terminal_status_counts.{key}", maximum=n
        )
        for key in ("committed", "protocol_failure")
    }
    if sum(result.values()) != n:
        raise YaoPublicReportError("$.terminal_status_counts does not sum to n")
    return result


def _failures(value: Any, *, expected_total: int) -> dict[str, int]:
    value = _object(value, path="$.failure_counts")
    result: dict[str, int] = {}
    for private_code, count_value in value.items():
        if private_code not in _FAILURE_CODE_SLUGS:
            raise YaoPublicReportError(
                "$.failure_counts has an unrecognized failure category"
            )
        public_code = _FAILURE_CODE_SLUGS[private_code]
        result[public_code] = result.get(public_code, 0) + _count(
            count_value, path="$.failure_counts.<category>", maximum=EXPECTED_N
        )
    if sum(result.values()) != expected_total:
        raise YaoPublicReportError("$.failure_counts disagrees with protocol failures")
    return dict(sorted(result.items()))


def _metadata(value: Any, *, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _exact_fields(
        value[0], expected={"name", "provider"}, path="$.model_metadata"
    )
    public_model = {
        "name": _identifier(model["name"], path="$.model_metadata.name"),
        "provider": _identifier(model["provider"], path="$.model_metadata.provider"),
    }
    if public_model["provider"] not in {"ollama_cloud", "ollama_local"}:
        raise YaoPublicReportError("$.model_metadata.provider is not a frozen provider")
    protocol = _exact_fields(
        value[1],
        expected={
            "version",
            "arm",
            "temperature",
            "seed",
            "questions_max",
            "semantic_calls_max",
            "repairs_max",
            "failure_policy",
            "answer_policy",
        },
        path="$.protocol_metadata",
    )
    expected_questions = 0 if arm == "same_model_one_shot" else 4
    expected_calls = 1 if arm == "same_model_one_shot" else 5
    public_protocol = {
        "version": _fixed(
            protocol["version"], "round9-yao-v1", path="$.protocol_metadata.version"
        ),
        "arm": _fixed(protocol["arm"], arm, path="$.protocol_metadata.arm"),
        "temperature": _number(
            protocol["temperature"], path="$.protocol_metadata.temperature"
        ),
        "seed": _count(protocol["seed"], path="$.protocol_metadata.seed"),
        "questions_max": _count(
            protocol["questions_max"], path="$.protocol_metadata.questions_max"
        ),
        "semantic_calls_max": _count(
            protocol["semantic_calls_max"],
            path="$.protocol_metadata.semantic_calls_max",
        ),
        "repairs_max": _count(
            protocol["repairs_max"], path="$.protocol_metadata.repairs_max"
        ),
        "failure_policy": _fixed(
            protocol["failure_policy"],
            "terminal_incorrect_no_fallback",
            path="$.protocol_metadata.failure_policy",
        ),
        "answer_policy": _fixed(
            protocol["answer_policy"],
            "frozen_official_pool_sha256_v1",
            path="$.protocol_metadata.answer_policy",
        ),
    }
    if (
        public_protocol["temperature"] != 0
        or public_protocol["seed"] != 42
        or public_protocol["questions_max"] != expected_questions
        or public_protocol["semantic_calls_max"] != expected_calls
        or public_protocol["repairs_max"] != 0
    ):
        raise YaoPublicReportError("$.protocol_metadata changed a frozen run parameter")
    return public_model, public_protocol


def _hashes(value: Any) -> dict[str, str]:
    value = _exact_fields(
        value,
        expected={"sample_sha256", "catalog_sha256", "prompt_sha256"},
        path="$.hashes",
    )
    result: dict[str, str] = {}
    for key in ("sample_sha256", "catalog_sha256", "prompt_sha256"):
        digest = value[key]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise YaoPublicReportError(f"$.hashes.{key} must be a full SHA-256")
        result[key] = digest
    if result["sample_sha256"] != EXPECTED_SAMPLE_SHA256:
        raise YaoPublicReportError(
            "$.hashes.sample_sha256 differs from the frozen sample"
        )
    return result


def export_public_yao_aggregate(
    aggregate: Mapping[str, Any],
    *,
    current_records: Sequence[Mapping[str, Any]] | None = None,
    reference_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and return an aggregate-only Yao report.

    This accepts only the exact current private aggregate schema.  In
    particular, case IDs, prompts, predictions, events, traces, and free-form
    text have no accepted location in the resulting tree.
    """
    aggregate = _exact_fields(
        aggregate,
        expected=_TOP_LEVEL_FIELDS,
        optional={"paired_outcomes"},
        path="$",
    )
    _fixed(
        aggregate["schema_version"], "round9-yao-aggregate-v1", path="$.schema_version"
    )
    _fixed(aggregate["benchmark_label"], "interactive_ifttt", path="$.benchmark_label")
    arm = aggregate["arm"]
    if arm not in {"same_model_one_shot", "bounded_clarification_agent"}:
        raise YaoPublicReportError("$.arm does not match a frozen Yao arm")
    n = _count(aggregate["n"], path="$.n")
    if n != EXPECTED_N:
        raise YaoPublicReportError("$.n does not match the frozen 150-case protocol")

    raw, percentages, intervals = _metric_report(
        raw=aggregate["raw_numerators"],
        percentages=aggregate["percentages"],
        intervals=aggregate["confidence_intervals"],
        n=n,
        path="$",
    )
    strata = _exact_fields(
        aggregate["by_stratum"], expected=set(EXPECTED_STRATA), path="$.by_stratum"
    )
    public_strata: dict[str, Any] = {}
    stratum_metric_sums = {metric: 0 for metric in METRICS}
    for stratum, expected_n in EXPECTED_STRATA.items():
        entry = _exact_fields(
            strata[stratum],
            expected={"n", "raw_numerators", "percentages", "confidence_intervals"},
            path=f"$.by_stratum.{stratum}",
        )
        stratum_n = _count(entry["n"], path=f"$.by_stratum.{stratum}.n")
        if stratum_n != expected_n:
            raise YaoPublicReportError(
                f"$.by_stratum.{stratum}.n changed its frozen quota"
            )
        s_raw, s_percentages, s_intervals = _metric_report(
            raw=entry["raw_numerators"],
            percentages=entry["percentages"],
            intervals=entry["confidence_intervals"],
            n=stratum_n,
            path=f"$.by_stratum.{stratum}",
        )
        for metric in METRICS:
            stratum_metric_sums[metric] += s_raw[metric]
        public_strata[stratum] = {
            "n": stratum_n,
            "raw_numerators": s_raw,
            "percentages": s_percentages,
            "confidence_intervals": s_intervals,
        }
    if stratum_metric_sums != raw:
        raise YaoPublicReportError(
            "$.by_stratum raw counts disagree with overall counts"
        )

    statuses = _statuses(aggregate["terminal_status_counts"], n=n)
    failures = _failures(
        aggregate["failure_counts"], expected_total=statuses["protocol_failure"]
    )
    if any(correct > statuses["committed"] for correct in raw.values()):
        raise YaoPublicReportError(
            "$.raw_numerators can count a protocol failure as correct"
        )
    questions = _questions(aggregate["questions"], n=n)
    usage = _usage(aggregate["usage"], n=n)
    cost = _cost(aggregate["cost"], usage=usage)
    model, protocol = _metadata(
        (aggregate["model_metadata"], aggregate["protocol_metadata"]), arm=arm
    )
    question_limit = protocol["questions_max"] * n
    if questions["total"] > question_limit or any(
        int(count) > protocol["questions_max"] and frequency
        for count, frequency in questions["count_distribution"].items()
    ):
        raise YaoPublicReportError("$.questions exceeds the frozen arm budget")
    semantic_calls = usage["totals"]["semantic_calls"]
    if not n <= semantic_calls <= protocol["semantic_calls_max"] * n:
        raise YaoPublicReportError(
            "$.usage.semantic_calls exceeds the frozen arm budget"
        )
    if usage["totals"]["physical_attempts"] < (
        semantic_calls - usage["totals"]["cache_hits"]
    ):
        raise YaoPublicReportError(
            "$.usage physical attempts are internally inconsistent"
        )

    public: dict[str, Any] = {
        "schema_version": "round9-yao-public-aggregate-v3",
        "benchmark_label": "interactive_ifttt",
        "evaluation_scope": "partial:150_of_3870",
        "arm": arm,
        "n": n,
        "raw_numerators": raw,
        "raw_denominators": {metric: n for metric in METRICS},
        "percentages": percentages,
        "confidence_intervals": intervals,
        "terminal_status_counts": statuses,
        "failure_counts": failures,
        "failure_denominator": n,
        "questions": questions,
        "usage": usage,
        "cost": cost,
        "by_stratum": public_strata,
        "model_metadata": model,
        "protocol_metadata": protocol,
        "hashes": _hashes(aggregate["hashes"]),
    }
    if "paired_outcomes" in aggregate:
        if current_records is None or reference_records is None:
            raise YaoPublicReportError(
                "paired public export requires both terminal record ledgers"
            )
        public["paired_outcomes"] = _paired(
            aggregate["paired_outcomes"],
            n=n,
            arm=arm,
            current_records=current_records,
            reference_records=reference_records,
            current_raw=raw,
            current_by_stratum=public_strata,
            current_statuses=statuses,
            current_failures=failures,
        )
    elif current_records is not None or reference_records is not None:
        raise YaoPublicReportError(
            "terminal record ledgers were supplied without a paired aggregate"
        )
    return copy.deepcopy(public)
