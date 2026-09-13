"""Narrow, auditable RAGAS-secondary diagnostics for Round 9.

Endpoint, field binding, compilation, and task success are scored elsewhere by
exact or benchmark-native evaluators.  This module intentionally exposes only:

* deterministic set-based context-ID precision/recall; and
* optional RAGAS 0.4.3 ``Faithfulness`` for a natural-language preview.

The deterministic implementation mirrors RAGAS 0.4.3's ID metrics but does not
import RAGAS or invoke a judge.  This keeps the primary retrieval diagnostic
reproducible and makes the optional dependency boundary explicit.
"""

from __future__ import annotations

import importlib.metadata
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Sequence

from farm_r9.privacy import DataClassification, PrivacyViolation


PINNED_RAGAS_VERSION = "0.4.3"
DEFAULT_EXPECTED_N = 150


class RagasSecondaryError(RuntimeError):
    """Protocol error with a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class JudgeRoute(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True)
class IDMetricRecord:
    """One retrieval result; IDs are opaque identifiers, never text contexts."""

    case_id: str
    retrieved_context_ids: tuple[str, ...]
    reference_context_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "IDMetricRecord":
        return cls(
            case_id=_nonempty_string(row.get("case_id"), "case_id"),
            retrieved_context_ids=_id_tuple(
                row.get("retrieved_context_ids"), "retrieved_context_ids"
            ),
            reference_context_ids=_id_tuple(
                row.get("reference_context_ids"), "reference_context_ids"
            ),
        )


@dataclass(frozen=True)
class PreviewRecord:
    """Inputs needed by RAGAS Faithfulness for one preview."""

    case_id: str
    user_input: str
    response: str
    retrieved_contexts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PreviewRecord":
        contexts = row.get("retrieved_contexts")
        if not isinstance(contexts, Sequence) or isinstance(
            contexts, (str, bytes, bytearray)
        ):
            raise RagasSecondaryError(
                "INVALID_PREVIEW_RECORD", "retrieved_contexts must be a list"
            )
        parsed_contexts = tuple(
            _nonempty_string(value, "retrieved_contexts[]") for value in contexts
        )
        if not parsed_contexts:
            raise RagasSecondaryError(
                "INVALID_PREVIEW_RECORD", "retrieved_contexts cannot be empty"
            )
        return cls(
            case_id=_nonempty_string(row.get("case_id"), "case_id"),
            user_input=_nonempty_string(row.get("user_input"), "user_input"),
            response=_nonempty_string(row.get("response"), "response"),
            retrieved_contexts=parsed_contexts,
        )


def require_ragas_043() -> str:
    """Require the one audited RAGAS release for judge-based diagnostics."""

    try:
        installed = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RagasSecondaryError(
            "RAGAS_NOT_INSTALLED",
            f"faithfulness requires ragas=={PINNED_RAGAS_VERSION}",
        ) from exc
    if installed != PINNED_RAGAS_VERSION:
        raise RagasSecondaryError(
            "RAGAS_VERSION_MISMATCH",
            f"expected ragas=={PINNED_RAGAS_VERSION}, found {installed}",
        )
    return installed


def score_id_record(record: IDMetricRecord) -> dict[str, Any]:
    """Mirror RAGAS 0.4.3 ID precision/recall using unique string IDs."""

    retrieved = set(record.retrieved_context_ids)
    references = set(record.reference_context_ids)
    if not retrieved:
        raise RagasSecondaryError(
            "EMPTY_RETRIEVED_IDS", f"{record.case_id} has no retrieved context IDs"
        )
    if not references:
        raise RagasSecondaryError(
            "EMPTY_REFERENCE_IDS", f"{record.case_id} has no reference context IDs"
        )
    hits = len(retrieved & references)
    precision = hits / len(retrieved)
    recall = hits / len(references)
    return {
        "case_id": record.case_id,
        "retrieved_unique_n": len(retrieved),
        "reference_unique_n": len(references),
        "intersection_n": hits,
        "id_context_precision": precision,
        "id_context_recall": recall,
        "precision_perfect": precision == 1.0,
        "recall_perfect": recall == 1.0,
        "both_perfect": precision == 1.0 and recall == 1.0,
    }


def score_id_records(
    records: Sequence[IDMetricRecord], *, expected_n: int = DEFAULT_EXPECTED_N
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score a frozen sample and return private rows plus aggregate statistics."""

    _require_exact_n(records, expected_n)
    if len({record.case_id for record in records}) != len(records):
        raise RagasSecondaryError("DUPLICATE_CASE_ID", "case IDs must be unique")
    rows = [score_id_record(record) for record in records]
    n = len(rows)
    intersection_n = sum(int(row["intersection_n"]) for row in rows)
    retrieved_n = sum(int(row["retrieved_unique_n"]) for row in rows)
    reference_n = sum(int(row["reference_unique_n"]) for row in rows)
    precision_perfect = sum(bool(row["precision_perfect"]) for row in rows)
    recall_perfect = sum(bool(row["recall_perfect"]) for row in rows)
    both_perfect = sum(bool(row["both_perfect"]) for row in rows)
    precision_values = [float(row["id_context_precision"]) for row in rows]
    recall_values = [float(row["id_context_recall"]) for row in rows]
    aggregate = {
        "n": n,
        "macro_id_context_precision": sum(precision_values) / n,
        "macro_id_context_recall": sum(recall_values) / n,
        "micro_id_context_precision": intersection_n / retrieved_n,
        "micro_id_context_recall": intersection_n / reference_n,
        "intersection_n": intersection_n,
        "retrieved_unique_n": retrieved_n,
        "reference_unique_n": reference_n,
        "precision_perfect_n": precision_perfect,
        "recall_perfect_n": recall_perfect,
        "both_perfect_n": both_perfect,
        "confidence_intervals_95": {
            "macro_id_context_precision": _bootstrap_mean_interval(precision_values),
            "macro_id_context_recall": _bootstrap_mean_interval(recall_values),
            "id_precision_perfect": _wilson_interval(precision_perfect, n),
            "id_recall_perfect": _wilson_interval(recall_perfect, n),
            "id_both_perfect": _wilson_interval(both_perfect, n),
        },
    }
    return rows, aggregate


async def score_preview_faithfulness(
    records: Sequence[PreviewRecord],
    *,
    evaluator_llm: Any,
    classification: DataClassification,
    judge_route: JudgeRoute,
    expected_n: int = DEFAULT_EXPECTED_N,
    metric_factory: Callable[[Any], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run optional RAGAS Faithfulness with an already configured judge.

    ``evaluator_llm`` must be a RAGAS-compatible evaluator.  Callers are
    responsible for applying the Round 9 global Cloud limiter to every physical
    request.  Confidential records are rejected for a Cloud judge before the
    optional RAGAS import or any model call.
    """

    if not isinstance(classification, DataClassification):
        raise PrivacyViolation("an explicit DataClassification value is required")
    if not isinstance(judge_route, JudgeRoute):
        raise PrivacyViolation("an explicit JudgeRoute value is required")
    if (
        classification is DataClassification.CONFIDENTIAL
        and judge_route is not JudgeRoute.LOCAL
    ):
        raise PrivacyViolation(
            "confidential preview faithfulness requires a local judge"
        )
    _require_exact_n(records, expected_n)
    if len({record.case_id for record in records}) != len(records):
        raise RagasSecondaryError("DUPLICATE_CASE_ID", "case IDs must be unique")
    require_ragas_043()

    if metric_factory is None:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness

        metric = Faithfulness(llm=evaluator_llm)

        async def evaluate(record: PreviewRecord) -> float:
            sample = SingleTurnSample(
                user_input=record.user_input,
                response=record.response,
                retrieved_contexts=list(record.retrieved_contexts),
            )
            return float(await metric.single_turn_ascore(sample))

    else:
        evaluator = metric_factory(evaluator_llm)

        async def evaluate(record: PreviewRecord) -> float:
            result = evaluator(record)
            if isinstance(result, Awaitable):
                result = await result
            return float(result)

    rows: list[dict[str, Any]] = []
    failure_n = 0
    for record in records:
        try:
            score = await evaluate(record)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("score must be finite and between zero and one")
            rows.append(
                {
                    "case_id": record.case_id,
                    "faithfulness": score,
                    "status": "scored",
                }
            )
        except Exception as exc:  # protocol failures are counted, never coerced
            failure_n += 1
            rows.append(
                {
                    "case_id": record.case_id,
                    "faithfulness": None,
                    "status": "protocol_failure",
                    "failure_type": type(exc).__name__,
                }
            )
    scored = [float(row["faithfulness"]) for row in rows if row["status"] == "scored"]
    aggregate = {
        "n": len(rows),
        "faithfulness_scored_n": len(scored),
        "faithfulness_protocol_failure_n": failure_n,
        "faithfulness_mean": sum(scored) / len(scored) if scored else None,
        "faithfulness_coverage": len(scored) / len(rows),
        "ragas_version": PINNED_RAGAS_VERSION,
        "judge_route": judge_route.value,
    }
    return rows, aggregate


def public_secondary_aggregate(
    *,
    benchmark_label: str,
    id_aggregate: Mapping[str, Any],
    faithfulness_aggregate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build only the aggregate allowlisted for public disclosure."""

    n = int(id_aggregate["n"])
    raw_numerators = {
        "id_precision_perfect": int(id_aggregate["precision_perfect_n"]),
        "id_recall_perfect": int(id_aggregate["recall_perfect_n"]),
        "id_both_perfect": int(id_aggregate["both_perfect_n"]),
    }
    raw_denominators = {name: n for name in raw_numerators}
    percentages = {
        "macro_id_context_precision": _public_percentage(
            float(id_aggregate["macro_id_context_precision"])
        ),
        "macro_id_context_recall": _public_percentage(
            float(id_aggregate["macro_id_context_recall"])
        ),
        "id_precision_perfect": _public_percentage(
            int(id_aggregate["precision_perfect_n"]) / n
        ),
        "id_recall_perfect": _public_percentage(
            int(id_aggregate["recall_perfect_n"]) / n
        ),
        "id_both_perfect": _public_percentage(int(id_aggregate["both_perfect_n"]) / n),
    }
    failure_counts: dict[str, int] = {}
    if faithfulness_aggregate is not None:
        if int(faithfulness_aggregate["n"]) != n:
            raise RagasSecondaryError(
                "FAITHFULNESS_DENOMINATOR_MISMATCH",
                "ID and faithfulness diagnostics must use the same frozen cases",
            )
        scored_n = int(faithfulness_aggregate["faithfulness_scored_n"])
        failure_n = int(faithfulness_aggregate["faithfulness_protocol_failure_n"])
        if scored_n + failure_n != n:
            raise RagasSecondaryError(
                "INVALID_FAITHFULNESS_COUNTS", "scored plus failures must equal n"
            )
        raw_numerators["faithfulness_scored"] = scored_n
        raw_denominators["faithfulness_scored"] = n
        failure_counts["faithfulness_protocol_failure"] = failure_n
        percentages["faithfulness_coverage"] = _public_percentage(scored_n / n)
        mean = faithfulness_aggregate.get("faithfulness_mean")
        if mean is not None:
            value = float(mean)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RagasSecondaryError(
                    "INVALID_FAITHFULNESS_MEAN",
                    "faithfulness mean must be finite and between zero and one",
                )
            percentages["faithfulness_mean_scored_cases"] = _public_percentage(value)
    result: dict[str, Any] = {
        "benchmark_label": benchmark_label,
        "n": n,
        "raw_numerators": raw_numerators,
        "raw_denominators": raw_denominators,
        "percentages": percentages,
        "confidence_intervals": {
            name: {
                "low": _public_percentage(float(interval["low"])),
                "high": _public_percentage(float(interval["high"])),
                "level": 95,
            }
            for name, interval in id_aggregate["confidence_intervals_95"].items()
        },
        "protocol_metadata": {
            "name": "ragas_secondary",
            "version": PINNED_RAGAS_VERSION,
            "role": "secondary_diagnostic",
            "format": "deterministic_id_set_metrics",
        },
    }
    if failure_counts:
        result["failure_counts"] = failure_counts
    return result


def _require_exact_n(records: Sequence[Any], expected_n: int) -> None:
    if (
        isinstance(expected_n, bool)
        or not isinstance(expected_n, int)
        or expected_n <= 0
    ):
        raise RagasSecondaryError("INVALID_EXPECTED_N", "expected_n must be positive")
    if len(records) != expected_n:
        raise RagasSecondaryError(
            "CASE_COUNT_MISMATCH", f"expected {expected_n} cases, found {len(records)}"
        )


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagasSecondaryError(
            "INVALID_RECORD", f"{field} must be a nonempty string"
        )
    return value


def _id_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RagasSecondaryError("INVALID_RECORD", f"{field} must be a list")
    parsed = tuple(_nonempty_string(str(item), f"{field}[]") for item in value)
    if not parsed:
        raise RagasSecondaryError("INVALID_RECORD", f"{field} cannot be empty")
    return parsed


def _wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> dict[str, float]:
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == n else min(1.0, center + radius)
    return {"low": low, "high": high}


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 42, resamples: int = 10_000
) -> dict[str, float]:
    """Fixed-seed case bootstrap percentile interval for a macro mean."""

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    low_index = math.floor(0.025 * (resamples - 1))
    high_index = math.ceil(0.975 * (resamples - 1))
    return {"low": means[low_index], "high": means[high_index]}


def _public_percentage(proportion: float) -> float:
    """Stable table-ready percentage without binary floating-point noise."""

    return round(100.0 * proportion, 10)
