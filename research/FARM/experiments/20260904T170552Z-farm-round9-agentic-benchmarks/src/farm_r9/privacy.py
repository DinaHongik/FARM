"""Fail-closed confidentiality boundaries for Round 9 artifacts and routing."""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn


class DataClassification(str, Enum):
    """The only data classes accepted at an inference boundary."""

    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


class DataSource(str, Enum):
    """Registered dataset lineage; display benchmark names are not authority."""

    FARM_V2 = "farm_v2"
    RECIPEGEN = "recipegen"
    INTERACTIVE_IFTTT = "interactive_ifttt"
    BFCL_V4 = "bfcl_v4"
    TARGE = "targe"


_SOURCE_CLASSIFICATION = {
    DataSource.FARM_V2: DataClassification.CONFIDENTIAL,
    DataSource.RECIPEGEN: DataClassification.PUBLIC,
    DataSource.INTERACTIVE_IFTTT: DataClassification.PUBLIC,
    DataSource.BFCL_V4: DataClassification.PUBLIC,
    DataSource.TARGE: DataClassification.PUBLIC,
}


@dataclass(frozen=True)
class CloudRoutePermit:
    """Credential-free proof that a registered source passed local policy."""

    source: DataSource
    classification: DataClassification


class PrivacyViolation(ValueError):
    """Raised when data is unsafe for the requested disclosure boundary."""


class CloudRoutingDenied(PermissionError):
    """Raised when a payload cannot be sent to an external Cloud model."""


_PUBLIC_AGGREGATE_FIELDS = {
    "benchmark_label",
    "n",
    "raw_numerators",
    "raw_denominators",
    "percentages",
    "confidence_intervals",
    "failure_counts",
    "model_metadata",
    "protocol_metadata",
    "operational_metrics",
    "hashes",
}
_CONFIDENTIAL_PUBLIC_AGGREGATE_FIELDS = {
    "n",
    "raw_numerators",
    "raw_denominators",
    "percentages",
    "confidence_intervals",
    "failure_counts",
    "hashes",
}
_PUBLIC_HASH_FIELDS = {
    "aggregate_input_sha256",
    "artifact_sha256",
    "code_sha256",
    "manifest_sha256",
    "model_sha256",
    "sample_manifest_sha256",
    "source_artifact_sha256",
}
_CONFIDENTIAL_PUBLIC_HASH_FIELDS = {
    "aggregate_input_sha256",
    "artifact_sha256",
    "code_artifact_sha256",
    "endpoint_records_artifact_sha256",
    "one_shot_records_artifact_sha256",
    "bounded_records_artifact_sha256",
    "sample_artifact_sha256",
    "source_artifact_sha256",
}
_REQUIRED_AGGREGATE_FIELDS = {"benchmark_label", "n"}
_REQUIRED_CONFIDENTIAL_AGGREGATE_FIELDS = {
    "n",
    "raw_numerators",
    "raw_denominators",
    "percentages",
    "confidence_intervals",
    "hashes",
}
_MODEL_METADATA_FIELDS = {
    "id",
    "name",
    "provider",
    "version",
    "revision",
    "snapshot",
    "digest",
    "digest_sha256",
    "digest_prefix",
    "expected_digest_prefix",
    "role",
    "architecture",
    "parameter_count",
    "quantization",
}
_PROTOCOL_METADATA_FIELDS = {
    "id",
    "name",
    "version",
    "arm",
    "role",
    "format",
    "api_style",
    "semantic_calls_max",
    "questions_max",
    "repairs_max",
    "tool_budget",
    "temperature",
    "seed",
    "max_tokens",
    "thinking_mode",
}
_OPERATIONAL_INTEGER_FIELDS = {
    "semantic_calls_total",
    "physical_attempts_total",
    "cache_hits_total",
    "input_tokens_total",
    "completion_tokens_total",
    "tokens_total",
    "questions_total",
    "answered_questions_total",
    "repair_calls_total",
}
_OPERATIONAL_NUMBER_FIELDS = {
    "semantic_calls_mean_per_case",
    "semantic_calls_p50",
    "semantic_calls_p95",
    "provider_latency_ms_total",
    "provider_latency_ms_mean_per_call",
    "provider_latency_ms_p50",
    "provider_latency_ms_p95",
    "queue_wait_ms_total",
    "questions_mean_per_case",
}
_OPERATIONAL_NULLABLE_NUMBER_FIELDS = {
    "cost_usd_total",
    "cost_usd_mean_per_case",
}
_OPERATIONAL_BOOLEAN_FIELDS = {"cost_available"}
_FORBIDDEN_FIELD_PREFIXES = (
    "query",
    "schema",
    "candidate",
    "message",
    "prompt",
    "response",
    "transcript",
    "toolresult",
    "toolcallresult",
    "rawcaseid",
    "caseid",
    "rawid",
)
_PUBLIC_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_PUBLIC_METADATA_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+@()-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_CREDENTIAL_TEXT = re.compile(
    r"(?:bearer\s+\S+|(?:api[_-]?key|password|passwd|access[_-]?token|"
    r"refresh[_-]?token|authorization)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
_SENSITIVE_FIELD_MARKERS = (
    "password",
    "passwd",
    "pwd",
    "apikey",
    "token",
    "authorization",
    "auth",
    "authheader",
    "bearer",
    "clientsecret",
    "privatekey",
    "credential",
    "cookie",
    "session",
    "secret",
    "secretref",
)
_SAFE_TOKEN_ACCOUNTING_FIELDS = {
    "completiontokens",
    "inputtokens",
    "outputtokens",
    "prompttokens",
    "reasoningtokens",
    "tokencount",
    "totaltokens",
}
_NAME_VALUE_MARKERS = {"name", "field", "key", "parameter", "argument"}
_NAME_VALUE_PAYLOADS = {
    "value",
    "content",
    "default",
    "example",
    "mockvalue",
    "literal",
}
_INLINE_BEARER = re.compile(r"\bbearer\s+[^\s,;]+", re.IGNORECASE)
_INLINE_ASSIGNMENT = re.compile(
    r"[\"']?(api[_-]?key|password|passwd|pwd|token|access[_-]?tokens?|"
    r"refresh[_-]?tokens?|authorization|auth|client[_-]?secret|private[_-]?key)"
    r"[\"']?\s*[:=]\s*(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
    r"[^\s,;}\]]+)",
    re.IGNORECASE,
)
_STANDALONE_PROVIDER_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_STANDALONE_DOTTED_TOKEN = re.compile(
    r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{8,})?\b"
)
REDACTED = "[REDACTED]"


def _deny_cloud(reason: str) -> NoReturn:
    raise CloudRoutingDenied(f"Ollama Cloud routing denied: {reason}")


def require_cloud_route(
    classification: DataClassification,
    *,
    benchmark_label: str,
    source: DataSource | None = None,
) -> CloudRoutePermit:
    """Authorize Cloud use only for explicitly typed, non-FARM public data.

    A string that happens to contain ``"public"`` is deliberately insufficient.
    Benchmark labels never upgrade a confidential classification.  FARM labels
    are denied as a second, fail-safe guard against accidental misclassification.
    """

    if not isinstance(classification, DataClassification):
        _deny_cloud("an explicit DataClassification value is required")
    if not isinstance(source, DataSource):
        _deny_cloud("a registered DataSource lineage is required")
    registered_classification = _SOURCE_CLASSIFICATION[source]
    if classification is not registered_classification:
        _deny_cloud("classification conflicts with registered source lineage")
    if registered_classification is not DataClassification.PUBLIC:
        _deny_cloud("registered confidential data is local-only")
    if not isinstance(benchmark_label, str) or not benchmark_label.strip():
        _deny_cloud("a nonempty benchmark label is required")

    normalized_label = re.sub(r"[^a-z0-9]", "", benchmark_label.casefold())
    if "farm" in normalized_label:
        _deny_cloud("FARM data is confidential and local-only")
    return CloudRoutePermit(source=source, classification=registered_classification)


def export_public_aggregate(
    aggregate: Mapping[str, Any],
    *,
    source_classification: DataClassification,
) -> dict[str, Any]:
    """Validate and copy one record safe to publish from any data class.

    Confidential fields are rejected rather than silently dropped.  This makes
    a producer explicitly construct the small aggregate contract and prevents a
    later schema expansion from becoming an accidental disclosure.
    """

    if not isinstance(source_classification, DataClassification):
        raise PrivacyViolation(
            "source_classification must be an explicit DataClassification value"
        )
    if not isinstance(aggregate, Mapping):
        raise PrivacyViolation("public aggregate must be a JSON object")

    _reject_forbidden_fields(aggregate)
    confidential = source_classification is DataClassification.CONFIDENTIAL
    allowed_fields = (
        _CONFIDENTIAL_PUBLIC_AGGREGATE_FIELDS
        if confidential
        else _PUBLIC_AGGREGATE_FIELDS
    )
    required_fields = (
        _REQUIRED_CONFIDENTIAL_AGGREGATE_FIELDS
        if confidential
        else _REQUIRED_AGGREGATE_FIELDS
    )
    unknown = set(aggregate) - allowed_fields
    if unknown:
        raise PrivacyViolation(
            "aggregate contains a field that is not public-exportable"
        )
    missing = required_fields - set(aggregate)
    if missing:
        names = ", ".join(sorted(missing))
        raise PrivacyViolation(f"public aggregate is missing required field: {names}")

    n = _nonnegative_integer(aggregate["n"], "$.n")
    if confidential and n == 0:
        raise PrivacyViolation("a confidential paper-table aggregate requires n > 0")
    exported: dict[str, Any] = {"n": n}
    if not confidential:
        exported["benchmark_label"] = _public_label(
            aggregate["benchmark_label"], "$.benchmark_label"
        )

    if "raw_numerators" in aggregate:
        exported["raw_numerators"] = _count_mapping(
            aggregate["raw_numerators"],
            path="$.raw_numerators",
            maximum=n,
        )
    if "raw_denominators" in aggregate:
        exported["raw_denominators"] = _count_mapping(
            aggregate["raw_denominators"],
            path="$.raw_denominators",
            maximum=n,
        )
    if "percentages" in aggregate:
        exported["percentages"] = _percentage_mapping(
            aggregate["percentages"],
            path="$.percentages",
        )
    if "confidence_intervals" in aggregate:
        exported["confidence_intervals"] = _confidence_intervals(
            aggregate["confidence_intervals"],
            path="$.confidence_intervals",
        )
    if "failure_counts" in aggregate:
        exported["failure_counts"] = _count_mapping(
            aggregate["failure_counts"],
            path="$.failure_counts",
            maximum=n,
        )
    if not confidential and "model_metadata" in aggregate:
        exported["model_metadata"] = _metadata(
            aggregate["model_metadata"],
            allowed_fields=_MODEL_METADATA_FIELDS,
            path="$.model_metadata",
        )
    if not confidential and "protocol_metadata" in aggregate:
        exported["protocol_metadata"] = _metadata(
            aggregate["protocol_metadata"],
            allowed_fields=_PROTOCOL_METADATA_FIELDS,
            path="$.protocol_metadata",
        )
    if not confidential and "operational_metrics" in aggregate:
        exported["operational_metrics"] = _operational_metrics(
            aggregate["operational_metrics"],
            n=n,
            path="$.operational_metrics",
        )
    if "hashes" in aggregate:
        exported["hashes"] = _hash_mapping(
            aggregate["hashes"],
            path="$.hashes",
            allowed_fields=(
                _CONFIDENTIAL_PUBLIC_HASH_FIELDS
                if confidential
                else _PUBLIC_HASH_FIELDS
            ),
        )

    if confidential:
        _validate_confidential_metric_table(exported)

    # Return a detached tree even when callers supplied custom Mapping objects.
    return copy.deepcopy(exported)


def redact_public_trace(
    trace: Any,
    *,
    source_classification: DataClassification,
    source: DataSource | None = None,
) -> Any:
    """Redact mock credential material from an upstream public trace.

    This is not a declassification mechanism: confidential traces are refused
    in full and must only use :func:`export_public_aggregate`.
    """

    if (
        not isinstance(source_classification, DataClassification)
        or not isinstance(source, DataSource)
        or _SOURCE_CLASSIFICATION[source] is not DataClassification.PUBLIC
        or source_classification is not _SOURCE_CLASSIFICATION[source]
    ):
        raise PrivacyViolation(
            "trace redaction requires registered public upstream lineage"
        )
    return _redact_trace_value(trace, path="$")


def _normalize_field_name(key: str) -> str:
    normalized = unicodedata.normalize("NFKC", key).casefold()
    return re.sub(r"[^a-z0-9]", "", normalized)


def _is_sensitive_field(key: str) -> bool:
    normalized = _normalize_field_name(key)
    # Plural token-accounting fields (input_tokens, token_count) are metrics,
    # while a singular token suffix denotes credential material.
    if normalized in _SAFE_TOKEN_ACCOUNTING_FIELDS:
        return False
    return any(marker in normalized for marker in _SENSITIVE_FIELD_MARKERS)


def _redact_trace_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise PrivacyViolation(f"non-string trace field forbidden at {path}")

        marker_is_sensitive = any(
            _normalize_field_name(key) in _NAME_VALUE_MARKERS
            and isinstance(child, str)
            and _is_sensitive_field(child)
            for key, child in value.items()
        )
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = _normalize_field_name(key)
            child_path = f"{path}.{key}"
            if _is_sensitive_field(key):
                result[key] = REDACTED
            elif marker_is_sensitive and normalized in _NAME_VALUE_PAYLOADS:
                result[key] = REDACTED
            else:
                result[key] = _redact_trace_value(child, path=child_path)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _redact_trace_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        value = _INLINE_BEARER.sub(REDACTED, value)
        value = _INLINE_ASSIGNMENT.sub(REDACTED, value)
        value = _STANDALONE_PROVIDER_TOKEN.sub(REDACTED, value)
        return _STANDALONE_DOTTED_TOKEN.sub(REDACTED, value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PrivacyViolation(f"non-finite trace number forbidden at {path}")
        return value
    raise PrivacyViolation(f"non-JSON trace value forbidden at {path}")


def _reject_forbidden_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PrivacyViolation(f"non-string field name forbidden at {path}")
            normalized = _normalize_field_name(key)
            if any(prefix in normalized for prefix in _FORBIDDEN_FIELD_PREFIXES):
                raise PrivacyViolation("forbidden confidential field detected")
            _reject_forbidden_fields(child, path=f"{path}.*")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=f"{path}[{index}]")


def _public_label(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _PUBLIC_LABEL.fullmatch(value):
        raise PrivacyViolation(f"{path} must be a short public identifier")
    if (
        _CREDENTIAL_TEXT.search(value)
        or _STANDALONE_PROVIDER_TOKEN.search(value)
        or _STANDALONE_DOTTED_TOKEN.search(value)
    ):
        raise PrivacyViolation(f"credential-shaped text forbidden at {path}")
    return value


def _nonnegative_integer(value: Any, path: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrivacyViolation(f"{path} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise PrivacyViolation(f"{path} cannot exceed n={maximum}")
    return value


def _finite_number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivacyViolation(f"{path} must be numeric")
    if not math.isfinite(value):
        raise PrivacyViolation(f"{path} must be finite")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrivacyViolation(f"{path} must be an object")
    return value


def _count_mapping(value: Any, *, path: str, maximum: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, count in _mapping(value, path).items():
        label = _public_label(label, path)
        result[label] = _nonnegative_integer(count, f"{path}.<entry>", maximum=maximum)
    return result


def _percentage_mapping(value: Any, *, path: str) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for label, percentage in _mapping(value, path).items():
        label = _public_label(label, path)
        number = _finite_number(percentage, f"{path}.<entry>")
        if not 0 <= number <= 100:
            raise PrivacyViolation(f"{path}.<entry> must be between 0 and 100")
        result[label] = number
    return result


def _confidence_intervals(
    value: Any, *, path: str
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for label, interval in _mapping(value, path).items():
        label = _public_label(label, path)
        interval_path = f"{path}.<entry>"
        interval = _mapping(interval, interval_path)
        unknown = set(interval) - {"low", "high", "level"}
        if unknown or "low" not in interval or "high" not in interval:
            raise PrivacyViolation(
                f"{interval_path} must contain only low, high, and optional level"
            )
        low = _finite_number(interval["low"], f"{interval_path}.low")
        high = _finite_number(interval["high"], f"{interval_path}.high")
        if not 0 <= low <= high <= 100:
            raise PrivacyViolation(
                f"{interval_path} must satisfy 0 <= low <= high <= 100"
            )
        public_interval: dict[str, int | float] = {"low": low, "high": high}
        if "level" in interval:
            level = _finite_number(interval["level"], f"{interval_path}.level")
            if not 0 < level <= 100:
                raise PrivacyViolation(
                    f"{interval_path}.level must be between 0 and 100"
                )
            public_interval["level"] = level
        result[label] = public_interval
    return result


def _validate_confidential_metric_table(exported: Mapping[str, Any]) -> None:
    numerators = exported["raw_numerators"]
    denominators = exported["raw_denominators"]
    percentages = exported["percentages"]
    intervals = exported["confidence_intervals"]
    metric_names = set(numerators)
    if not metric_names or any(
        set(values) != metric_names for values in (denominators, percentages, intervals)
    ):
        raise PrivacyViolation(
            "confidential metric numerators, denominators, percentages, and uncertainty must align"
        )
    for metric in metric_names:
        numerator = numerators[metric]
        denominator = denominators[metric]
        if denominator <= 0 or numerator > denominator:
            raise PrivacyViolation(
                "confidential metric counts have an invalid denominator"
            )
        expected_percentage = 100.0 * numerator / denominator
        if not math.isclose(
            percentages[metric],
            expected_percentage,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise PrivacyViolation(
                "confidential percentage is inconsistent with raw counts"
            )
        interval = intervals[metric]
        if not interval["low"] <= expected_percentage <= interval["high"]:
            raise PrivacyViolation(
                "confidential uncertainty excludes the observed percentage"
            )


def _metadata(
    value: Any,
    *,
    allowed_fields: set[str],
    path: str,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _metadata_record(
                item, allowed_fields=allowed_fields, path=f"{path}[{index}]"
            )
            for index, item in enumerate(value)
        ]
    return _metadata_record(value, allowed_fields=allowed_fields, path=path)


def _operational_metrics(value: Any, *, n: int, path: str) -> dict[str, Any]:
    value = _mapping(value, path)
    allowed = (
        _OPERATIONAL_INTEGER_FIELDS
        | _OPERATIONAL_NUMBER_FIELDS
        | _OPERATIONAL_NULLABLE_NUMBER_FIELDS
        | _OPERATIONAL_BOOLEAN_FIELDS
    )
    if set(value) - allowed:
        raise PrivacyViolation(f"{path} contains a non-public operational field")
    result: dict[str, Any] = {}
    for key, child in value.items():
        child_path = f"{path}.{key}"
        if key in _OPERATIONAL_INTEGER_FIELDS:
            result[key] = _nonnegative_integer(child, child_path)
        elif key in _OPERATIONAL_NUMBER_FIELDS:
            number = _finite_number(child, child_path)
            if number < 0:
                raise PrivacyViolation(f"{child_path} must be non-negative")
            result[key] = number
        elif key in _OPERATIONAL_NULLABLE_NUMBER_FIELDS:
            if child is None:
                result[key] = None
            else:
                number = _finite_number(child, child_path)
                if number < 0:
                    raise PrivacyViolation(f"{child_path} must be non-negative")
                result[key] = number
        elif key in _OPERATIONAL_BOOLEAN_FIELDS:
            if not isinstance(child, bool):
                raise PrivacyViolation(f"{child_path} must be boolean")
            result[key] = child

    if {"input_tokens_total", "completion_tokens_total", "tokens_total"} <= set(result):
        if (
            result["tokens_total"]
            != result["input_tokens_total"] + result["completion_tokens_total"]
        ):
            raise PrivacyViolation(f"{path} token totals are inconsistent")
    if "cache_hits_total" in result and "semantic_calls_total" in result:
        if result["cache_hits_total"] > result["semantic_calls_total"]:
            raise PrivacyViolation(
                f"{path}.cache_hits_total cannot exceed semantic calls"
            )
    if (
        "semantic_calls_mean_per_case" in result
        and "semantic_calls_total" in result
        and n
    ):
        expected = result["semantic_calls_total"] / n
        if not math.isclose(
            result["semantic_calls_mean_per_case"], expected, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise PrivacyViolation(f"{path} semantic-call mean is inconsistent with n")
    if "questions_mean_per_case" in result and "questions_total" in result and n:
        expected = result["questions_total"] / n
        if not math.isclose(
            result["questions_mean_per_case"], expected, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise PrivacyViolation(f"{path} question mean is inconsistent with n")
    if {"semantic_calls_p50", "semantic_calls_p95"} <= set(result):
        if result["semantic_calls_p50"] > result["semantic_calls_p95"]:
            raise PrivacyViolation(f"{path} semantic-call quantiles are inconsistent")
    if {"provider_latency_ms_p50", "provider_latency_ms_p95"} <= set(result):
        if result["provider_latency_ms_p50"] > result["provider_latency_ms_p95"]:
            raise PrivacyViolation(f"{path} latency quantiles are inconsistent")
    if "cost_available" in result:
        cost_values = [result.get(key) for key in _OPERATIONAL_NULLABLE_NUMBER_FIELDS]
        if result["cost_available"] and any(value is None for value in cost_values):
            raise PrivacyViolation(f"{path} available cost requires both cost totals")
        if not result["cost_available"] and any(
            value is not None for value in cost_values
        ):
            raise PrivacyViolation(f"{path} unavailable cost must remain null")
    return result


def _metadata_record(
    value: Any,
    *,
    allowed_fields: set[str],
    path: str,
) -> dict[str, Any]:
    value = _mapping(value, path)
    unknown = set(value) - allowed_fields
    if unknown:
        raise PrivacyViolation(f"metadata contains a non-public field at {path}")

    result: dict[str, Any] = {}
    for key, child in value.items():
        child_path = f"{path}.{key}"
        if child is None or isinstance(child, bool):
            result[key] = child
        elif isinstance(child, (int, float)) and not isinstance(child, bool):
            result[key] = _finite_number(child, child_path)
        elif isinstance(child, str):
            if (
                not _PUBLIC_METADATA_IDENTIFIER.fullmatch(child)
                or ".." in child
                or any(ord(character) < 32 for character in child)
            ):
                raise PrivacyViolation(
                    f"{child_path} must be a public metadata identifier"
                )
            if (
                _CREDENTIAL_TEXT.search(child)
                or _STANDALONE_PROVIDER_TOKEN.search(child)
                or _STANDALONE_DOTTED_TOKEN.search(child)
            ):
                raise PrivacyViolation(
                    f"credential-shaped text forbidden at {child_path}"
                )
            result[key] = child
        else:
            raise PrivacyViolation(f"{child_path} must be scalar public metadata")
    return result


def _hash_mapping(
    value: Any,
    *,
    path: str,
    allowed_fields: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, digest in _mapping(value, path).items():
        label = _public_label(label, path)
        if label not in allowed_fields:
            raise PrivacyViolation(f"{path} contains a non-public hash category")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise PrivacyViolation(f"{path} values must be full SHA-256 hashes")
        result[label] = digest.lower()
    return result


__all__ = [
    "CloudRoutingDenied",
    "CloudRoutePermit",
    "DataClassification",
    "DataSource",
    "PrivacyViolation",
    "REDACTED",
    "export_public_aggregate",
    "redact_public_trace",
    "require_cloud_route",
]
