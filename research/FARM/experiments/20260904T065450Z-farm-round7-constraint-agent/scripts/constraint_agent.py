#!/usr/bin/env python3
"""Factorized, constraint-guided FARM endpoint selection.

This module deliberately stops at endpoint-pair selection.  It does not claim
to compile an executable applet because the current evaluation cases do not
contain gold field bindings.  Instead it exposes three real, deterministic
tools (catalog search, endpoint-schema lookup, and compatibility validation),
records every invocation, and lets a one-shot chooser alter either side of the
retrieval baseline.  The conservative default accepts a non-baseline proposal
only after schema validation and an order-reversed unanimous verification;
audited ``none`` and ``single`` verifier settings support controlled ablations.

The inference boundary is intentionally strict: evaluation labels are rejected
before any tool or model call, endpoint identities/ranks are replaced by opaque
aliases in model requests, and model choices cannot leave the independently
retrieved top-ten trigger/action sets.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, Sequence


KEEP = "KEEP"
ABSTAIN = "ABSTAIN"


class Side(str, Enum):
    TRIGGER = "trigger"
    ACTION = "action"


class EditDecision(str, Enum):
    KEEP = "KEEP"
    CHANGE_TRIGGER = "CHANGE_TRIGGER"
    CHANGE_ACTION = "CHANGE_ACTION"
    CHANGE_BOTH = "CHANGE_BOTH"
    ABSTAIN = "ABSTAIN"


class ValidationStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class StatelessChooser(Protocol):
    """One-shot model boundary.

    Implementations must answer solely from the supplied request.  The core
    sends a fresh deep-copied request for every invocation and never forwards
    earlier messages or hidden state.
    """

    def select(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


ToolName = Literal["search_catalog", "read_endpoint_schema", "validate_pair"]


@dataclass(frozen=True)
class Policy:
    candidate_depth: Literal[10] = 10
    evidence_view: Literal["plain", "schema"] = "schema"
    verification: Literal["none", "single", "dual_unanimous"] = "dual_unanimous"
    max_repairs: Literal[0, 1] = 1
    presentation_seed: int = 42

    def __post_init__(self) -> None:
        if self.candidate_depth != 10:
            raise ValueError("candidate_depth is frozen at ten")
        if self.evidence_view not in {"plain", "schema"}:
            raise ValueError("unsupported evidence view")
        if self.verification not in {"none", "single", "dual_unanimous"}:
            raise ValueError("verification must be none, single, or dual_unanimous")
        if (
            not isinstance(self.max_repairs, int)
            or isinstance(self.max_repairs, bool)
            or self.max_repairs not in {0, 1}
        ):
            raise ValueError("max_repairs must be zero or one")
        if isinstance(self.presentation_seed, bool) or not isinstance(
            self.presentation_seed, int
        ):
            raise ValueError("presentation_seed must be an integer")


_REFERENCE_KEYS = {
    "correct_pair",
    "expected_pair",
    "gold",
    "gold_pair",
    "gold_pairs",
    "gold_trigger_urls",
    "gold_action_urls",
    "gold_channel_pairs",
    "gold_service_pairs",
    "ground_truth",
    "is_correct",
    "observed_pairs",
    "pair_rank",
    "reference",
    "references",
    "reference_answer",
    "valid_pairs",
}
_ROOT_REFERENCE_KEYS = {"answer", "answers", "label", "labels", "target", "targets"}
_REFERENCE_PREFIXES = ("gold_", "reference_", "ground_truth_", "target_")


def reference_paths(value: Any, path: str = "case") -> list[str]:
    """Return paths of fields that may encode evaluation answers."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold()
            child_path = f"{path}.{raw_key}"
            if (
                key in _REFERENCE_KEYS
                or key.startswith(_REFERENCE_PREFIXES)
                or (path == "case" and key in _ROOT_REFERENCE_KEYS)
            ):
                found.append(child_path)
            found.extend(reference_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(reference_paths(child, f"{path}[{index}]"))
    return found


def assert_inference_safe(value: Any, path: str = "case") -> None:
    leaked = reference_paths(value, path)
    if leaked:
        raise ValueError(f"reference fields crossed inference seam: {sorted(leaked)}")


def _text(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    raise ValueError(f"missing non-empty text field; tried {keys}")


def _optional_text(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _normal_type(value: str) -> str:
    normalized = value.strip().casefold()
    aliases = {
        "str": "string",
        "text": "string",
        "uri": "string",
        "url": "string",
        "int": "integer",
        "float": "number",
        "double": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }
    return aliases.get(normalized, normalized or "unknown")


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "unknown"
    required: bool | None = None
    aliases: tuple[str, ...] = ()
    source: Literal["trigger", "constant", "context", "user", "unknown"] = "unknown"
    bindable: bool | None = None
    can_have_default: bool | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("field name must be non-empty")
        if self.required is not None and not isinstance(self.required, bool):
            raise ValueError("field required must be boolean or null")
        if self.source not in {"trigger", "constant", "context", "user", "unknown"}:
            raise ValueError(f"unsupported field source: {self.source}")
        if self.bindable is not None and not isinstance(self.bindable, bool):
            raise ValueError("field bindable must be boolean or null")
        if self.can_have_default is not None and not isinstance(
            self.can_have_default, bool
        ):
            raise ValueError("field can_have_default must be boolean or null")

    @property
    def normalized_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in (
                _normal_name(self.name),
                *(_normal_name(alias) for alias in self.aliases),
            )
            if name
        )

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": _normal_type(self.type),
            "required": self.required,
            "source": self.source,
            "bindable": self.bindable,
            "can_have_default": self.can_have_default,
        }


def _optional_bool(
    raw: Mapping[str, Any], key: str, default: bool | None
) -> bool | None:
    value = raw.get(key, default)
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"schema field {key} must be boolean or null")


def _one_field(
    name: str,
    raw: Any,
    required_names: set[str],
    *,
    extra_aliases: Sequence[str] = (),
) -> FieldSpec:
    required_default: bool | None = True if name in required_names else False
    if isinstance(raw, str):
        return FieldSpec(
            name=name,
            type=raw,
            required=required_default,
            aliases=tuple(extra_aliases),
        )
    if not isinstance(raw, Mapping):
        return FieldSpec(
            name=name, required=required_default, aliases=tuple(extra_aliases)
        )
    aliases_raw = raw.get("aliases", ())
    if isinstance(aliases_raw, str):
        aliases = (aliases_raw,)
    elif isinstance(aliases_raw, Sequence) and not isinstance(
        aliases_raw, (bytes, bytearray)
    ):
        aliases = tuple(str(item).strip() for item in aliases_raw if str(item).strip())
    else:
        aliases = ()
    aliases = tuple(
        dict.fromkeys(
            alias
            for alias in (*extra_aliases, *aliases)
            if alias and _normal_name(alias) != _normal_name(name)
        )
    )

    # Dataset-v2's `bindable` flag means that IFTTT permits an ingredient in
    # this slot; it is not evidence that this particular recipe must bind one.
    # Therefore native fields stay `unknown` unless a source is explicit.
    source = str(raw.get("source", "unknown")).strip().casefold()
    if source not in {"trigger", "constant", "context", "user", "unknown"}:
        source = "unknown"
    return FieldSpec(
        name=name,
        type=str(raw.get("type", "unknown")),
        required=_optional_bool(raw, "required", required_default),
        aliases=aliases,
        source=source,  # type: ignore[arg-type]
        bindable=_optional_bool(raw, "bindable", None),
        can_have_default=_optional_bool(raw, "can_have_default", None),
    )


def _fields(raw: Any) -> tuple[FieldSpec, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping) and isinstance(raw.get("properties"), Mapping):
        required_raw = raw.get("required", ())
        required_names = (
            {str(item) for item in required_raw}
            if isinstance(required_raw, Sequence)
            and not isinstance(required_raw, (str, bytes))
            else set()
        )
        return tuple(
            _one_field(str(name), spec, required_names)
            for name, spec in sorted(
                raw["properties"].items(), key=lambda item: str(item[0])
            )
        )
    if isinstance(raw, Mapping):
        return tuple(
            _one_field(str(name), spec, set())
            for name, spec in sorted(raw.items(), key=lambda item: str(item[0]))
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parsed: list[FieldSpec] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("schema field list entries must be mappings")
            name = _text(item, "slug", "name", "field", "id", "label")
            alternate_names = tuple(
                item[key].strip()
                for key in ("label", "slug", "name", "field", "id")
                if isinstance(item.get(key), str) and item[key].strip()
            )
            parsed.append(_one_field(name, item, set(), extra_aliases=alternate_names))
        return tuple(sorted(parsed, key=lambda field: field.name))
    raise ValueError("schema fields must be a mapping or sequence")


@dataclass(frozen=True)
class EndpointSchema:
    role: Literal["trigger", "action", "unknown"] = "unknown"
    inputs: tuple[FieldSpec, ...] = ()
    outputs: tuple[FieldSpec, ...] = ()

    @classmethod
    def from_value(cls, value: Any, expected_side: Side) -> "EndpointSchema":
        if value is None:
            return cls(role=expected_side.value)
        if not isinstance(value, Mapping):
            raise ValueError("endpoint schema must be a mapping")
        role = (
            str(value.get("role", value.get("side", expected_side.value)))
            .strip()
            .casefold()
        )
        if role not in {"trigger", "action", "unknown"}:
            role = expected_side.value
        inputs = value.get("inputs", value.get("parameters", value.get("input_schema")))
        outputs = value.get("outputs", value.get("output_schema"))
        return cls(role=role, inputs=_fields(inputs), outputs=_fields(outputs))  # type: ignore[arg-type]

    @classmethod
    def from_candidate(
        cls, candidate: Mapping[str, Any], expected_side: Side
    ) -> "EndpointSchema":
        """Normalize either a nested schema or a native Dataset-v2 corpus row."""

        nested_raw = candidate.get("schema")
        if nested_raw is None:
            nested: Mapping[str, Any] = {}
        elif isinstance(nested_raw, Mapping):
            nested = nested_raw
        else:
            raise ValueError("endpoint schema must be a mapping")

        role = (
            str(
                nested.get(
                    "role",
                    nested.get(
                        "side",
                        candidate.get(
                            "kind",
                            candidate.get(
                                "side", candidate.get("role", expected_side.value)
                            ),
                        ),
                    ),
                )
            )
            .strip()
            .casefold()
        )
        if role not in {"trigger", "action", "unknown"}:
            role = expected_side.value

        if "inputs" in nested:
            inputs = nested["inputs"]
        elif "parameters" in nested:
            inputs = nested["parameters"]
        elif "input_schema" in nested:
            inputs = nested["input_schema"]
        else:
            inputs = candidate.get("input_fields")

        if "outputs" in nested:
            outputs = nested["outputs"]
        elif "output_schema" in nested:
            outputs = nested["output_schema"]
        else:
            outputs = candidate.get("ingredients")

        return cls(  # type: ignore[arg-type]
            role=role,
            inputs=_fields(inputs),
            outputs=_fields(outputs),
        )

    def public(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "inputs": [field.public() for field in self.inputs],
            "outputs": [field.public() for field in self.outputs],
        }


@dataclass(frozen=True)
class EndpointCandidate:
    identity: str
    side: Side
    retrieval_rank: int
    service_name: str
    function_name: str
    evidence: str
    schema: EndpointSchema

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], side: Side, fallback_rank: int, evidence_view: str
    ) -> "EndpointCandidate":
        identity = _text(raw, "identity", "url", "id")
        declared_side = _optional_text(raw, "kind", "side", "role")
        if declared_side is not None and declared_side.casefold() in {
            "trigger",
            "action",
        }:
            if declared_side.casefold() != side.value:
                raise ValueError(f"candidate {identity!r} is on the wrong side")
        rank_raw = raw.get("retrieval_rank", raw.get("rank", fallback_rank))
        if isinstance(rank_raw, bool) or not isinstance(rank_raw, int) or rank_raw < 1:
            raise ValueError("retrieval rank must be a positive integer")
        if evidence_view == "schema":
            evidence = _text(
                raw,
                "text_schema",
                "schema_text",
                "text_plain",
                "text",
                "description",
                "evidence",
            )
        else:
            evidence = _text(
                raw, "text_plain", "text", "description", "evidence", "text_schema"
            )
        return cls(
            identity=identity,
            side=side,
            retrieval_rank=rank_raw,
            service_name=_text(
                raw,
                "service_name",
                "channel_display",
                "service_display",
                "channel",
                "service",
            ),
            function_name=_text(raw, "function_name", "name", "display"),
            evidence=evidence,
            schema=EndpointSchema.from_candidate(raw, side),
        )

    def public(self, alias: str, *, include_schema: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_id": alias,
            "service_name": self.service_name,
            "function_name": self.function_name,
            "evidence": self.evidence,
        }
        if include_schema:
            result["schema"] = self.schema.public()
        return result


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Literal["error", "warning"]
    field: str | None
    expected_type: str | None = None
    observed_type: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "field": self.field,
            "expected_type": self.expected_type,
            "observed_type": self.observed_type,
        }


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]
    bindings: tuple[tuple[str, str], ...] = ()
    criterion: str = "endpoint_schema_compatibility_proxy"

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "criterion": self.criterion,
            "issues": [issue.public() for issue in self.issues],
            "bindings": [
                {"action_input": action_input, "trigger_output": trigger_output}
                for action_input, trigger_output in self.bindings
            ],
        }


@dataclass(frozen=True)
class ToolEvent:
    sequence: int
    tool: ToolName
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tool": self.tool,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "result": copy.deepcopy(dict(self.result)),
            "ok": self.ok,
        }


def _types_compatible(producer: str, consumer: str) -> bool:
    left, right = _normal_type(producer), _normal_type(consumer)
    if "unknown" in {left, right} or "any" in {left, right}:
        return True
    if left == right:
        return True
    return {left, right} <= {"integer", "number"}


def validate_compatibility(
    trigger: EndpointCandidate, action: EndpointCandidate
) -> ValidationResult:
    """Validate only schema facts observable at inference time.

    Required action inputs explicitly marked ``source=trigger`` must match a
    trigger output by normalized name/alias and compatible type.  Other
    required inputs are reported as unverified warnings because this dataset
    does not contain gold static/configuration bindings.
    """

    issues: list[ValidationIssue] = []
    bindings: list[tuple[str, str]] = []
    if trigger.schema.role == "action":
        issues.append(ValidationIssue("TRIGGER_ROLE_MISMATCH", "error", None))
    if action.schema.role == "trigger":
        issues.append(ValidationIssue("ACTION_ROLE_MISMATCH", "error", None))

    outputs = trigger.schema.outputs
    for field in action.schema.inputs:
        if not field.required:
            continue
        if field.source in {"constant", "context", "user"}:
            continue
        if field.source == "unknown":
            issues.append(
                ValidationIssue(
                    "UNVERIFIED_REQUIRED_INPUT",
                    "warning",
                    field.name,
                    expected_type=_normal_type(field.type),
                )
            )
            continue
        matches = [
            output
            for output in outputs
            if field.normalized_names.intersection(output.normalized_names)
        ]
        if not matches:
            issues.append(
                ValidationIssue(
                    "REQUIRED_TRIGGER_OUTPUT_MISSING",
                    "error",
                    field.name,
                    expected_type=_normal_type(field.type),
                )
            )
            continue
        compatible = next(
            (
                output
                for output in matches
                if _types_compatible(output.type, field.type)
            ),
            None,
        )
        if compatible is None:
            observed = ",".join(
                sorted({_normal_type(output.type) for output in matches})
            )
            issues.append(
                ValidationIssue(
                    "TRIGGER_ACTION_TYPE_MISMATCH",
                    "error",
                    field.name,
                    expected_type=_normal_type(field.type),
                    observed_type=observed,
                )
            )
            continue
        bindings.append((field.name, compatible.name))

    if any(issue.severity == "error" for issue in issues):
        status = ValidationStatus.INCOMPATIBLE
    elif issues or not trigger.schema.outputs or not action.schema.inputs:
        status = ValidationStatus.UNKNOWN
    else:
        status = ValidationStatus.COMPATIBLE
    return ValidationResult(
        status=status, issues=tuple(issues), bindings=tuple(bindings)
    )


class CatalogTools:
    """Deterministic tools backed by the hydrated inference candidate lattice."""

    def __init__(self, candidates: Mapping[Side, Sequence[EndpointCandidate]]) -> None:
        self._candidates = {side: tuple(values) for side, values in candidates.items()}
        self._by_identity = {
            candidate.identity: candidate
            for values in self._candidates.values()
            for candidate in values
        }
        if len(self._by_identity) != sum(
            len(values) for values in self._candidates.values()
        ):
            raise ValueError(
                "endpoint identities must be unique across both candidate lists"
            )
        self._retrieved_identities: set[str] = set()
        self.events: list[ToolEvent] = []

    def _record(
        self, tool: ToolName, arguments: Mapping[str, Any], result: Mapping[str, Any]
    ) -> None:
        self.events.append(
            ToolEvent(
                sequence=len(self.events) + 1,
                tool=tool,
                arguments=copy.deepcopy(dict(arguments)),
                result=copy.deepcopy(dict(result)),
            )
        )

    def search_catalog(
        self, query: str, side: Side, depth: int
    ) -> tuple[EndpointCandidate, ...]:
        if depth != 10:
            raise ValueError("search depth must be ten")
        ordered = tuple(
            sorted(
                self._candidates[side],
                key=lambda item: (item.retrieval_rank, item.identity),
            )[:depth]
        )
        if len(ordered) != depth:
            raise ValueError(
                f"{side.value} catalog search requires at least ten candidates"
            )
        self._retrieved_identities.update(item.identity for item in ordered)
        self._record(
            "search_catalog",
            {
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "side": side.value,
                "depth": depth,
            },
            {
                "count": len(ordered),
                "endpoint_identities": [item.identity for item in ordered],
            },
        )
        return ordered

    def read_endpoint_schema(self, identity: str) -> EndpointSchema:
        candidate = self._by_identity.get(identity)
        if candidate is None or identity not in self._retrieved_identities:
            raise ValueError("schema lookup escaped retrieved catalog")
        result = candidate.schema.public()
        self._record("read_endpoint_schema", {"endpoint_identity": identity}, result)
        return candidate.schema

    def validate_pair(
        self, trigger: EndpointCandidate, action: EndpointCandidate
    ) -> ValidationResult:
        if (
            trigger.identity not in self._retrieved_identities
            or action.identity not in self._retrieved_identities
        ):
            raise ValueError("validation escaped retrieved catalog")
        result = validate_compatibility(trigger, action)
        self._record(
            "validate_pair",
            {"trigger_identity": trigger.identity, "action_identity": action.identity},
            result.public(),
        )
        return result


@dataclass(frozen=True)
class FactorizedChoice:
    decision: EditDecision
    trigger_choice: str
    action_choice: str


@dataclass(frozen=True)
class Pair:
    trigger: EndpointCandidate
    action: EndpointCandidate

    @property
    def identities(self) -> dict[str, str]:
        return {
            "trigger_url": self.trigger.identity,
            "action_url": self.action.identity,
        }


@dataclass(frozen=True)
class CandidateDeck:
    trigger_by_alias: Mapping[str, EndpointCandidate]
    action_by_alias: Mapping[str, EndpointCandidate]
    alias_by_identity: Mapping[str, str]

    def by_side(self, side: Side) -> Mapping[str, EndpointCandidate]:
        return self.trigger_by_alias if side is Side.TRIGGER else self.action_by_alias


def _hydrate_case(
    case: Mapping[str, Any], policy: Policy
) -> tuple[str, str, dict[Side, tuple[EndpointCandidate, ...]]]:
    assert_inference_safe(case)
    case_id = _text(case, "group_id", "case_id", "id")
    query = _text(case, "query", "recipe", "text")
    hydrated: dict[Side, tuple[EndpointCandidate, ...]] = {}
    for side in Side:
        raw = case.get(f"{side.value}_candidates")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError(f"{side.value}_candidates must be a sequence")
        candidates = tuple(
            EndpointCandidate.from_mapping(item, side, index + 1, policy.evidence_view)
            for index, item in enumerate(raw)
            if isinstance(item, Mapping)
        )
        if len(candidates) != len(raw):
            raise ValueError(f"every {side.value} candidate must be a mapping")
        if len({candidate.identity for candidate in candidates}) != len(candidates):
            raise ValueError(f"duplicate {side.value} candidate identity")
        hydrated[side] = candidates
    return case_id, query, hydrated


def _opaque_deck(
    case_id: str,
    searched: Mapping[Side, Sequence[EndpointCandidate]],
    policy: Policy,
) -> CandidateDeck:
    by_side: dict[Side, dict[str, EndpointCandidate]] = {}
    alias_by_identity: dict[str, str] = {}
    for side in Side:
        presentation = sorted(
            searched[side],
            key=lambda item: hashlib.sha256(
                f"farm-r7\0{policy.presentation_seed}\0{case_id}\0{side.value}\0{item.identity}".encode()
            ).hexdigest(),
        )
        prefix = "T" if side is Side.TRIGGER else "A"
        aliases = {
            f"{prefix}{index + 1:02d}": item for index, item in enumerate(presentation)
        }
        by_side[side] = aliases
        alias_by_identity.update(
            {item.identity: alias for alias, item in aliases.items()}
        )
    return CandidateDeck(
        trigger_by_alias=by_side[Side.TRIGGER],
        action_by_alias=by_side[Side.ACTION],
        alias_by_identity=alias_by_identity,
    )


def _request_id(case_id: str, phase: str, ordinal: int) -> str:
    value = hashlib.sha256(
        f"farm-r7-request\0{case_id}\0{phase}\0{ordinal}".encode()
    ).hexdigest()
    return f"R{value[:16]}"


def _public_candidates(deck: CandidateDeck) -> dict[str, list[dict[str, Any]]]:
    return {
        side.value: [
            candidate.public(alias) for alias, candidate in deck.by_side(side).items()
        ]
        for side in Side
    }


def _factorized_request(
    *,
    case_id: str,
    query: str,
    deck: CandidateDeck,
    baseline: Pair,
    phase: Literal["factorized_selection", "validator_repair"],
    ordinal: int,
    feedback: ValidationResult | None = None,
    prior: Pair | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": "farm_r7_factorized_request_v1",
        "request_id": _request_id(case_id, phase, ordinal),
        "phase": phase,
        "query": query,
        "instruction": (
            "Choose trigger and action independently from the supplied endpoint lists. "
            "KEEP means retain that side of current_pair. Change a side only when visible "
            "evidence clearly supports the replacement. Return exactly the declared contract."
        ),
        "current_pair": {
            "trigger_choice": deck.alias_by_identity[baseline.trigger.identity],
            "action_choice": deck.alias_by_identity[baseline.action.identity],
        },
        "candidates": _public_candidates(deck),
        "output_contract": {
            "decision": [item.value for item in EditDecision],
            "trigger_choice": [KEEP, *deck.trigger_by_alias.keys()],
            "action_choice": [KEEP, *deck.action_by_alias.keys()],
            "additional_properties": False,
        },
    }
    if feedback is not None and prior is not None:
        request["instruction"] = (
            "The selected endpoint pair failed deterministic schema validation. Make at most "
            "one corrected factorized choice using only the supplied candidates and validator "
            "feedback. KEEP is relative to current_pair, not the rejected proposal."
        )
        request["rejected_pair"] = {
            "trigger_choice": deck.alias_by_identity[prior.trigger.identity],
            "action_choice": deck.alias_by_identity[prior.action.identity],
        }
        request["validator_feedback"] = feedback.public()
    assert_inference_safe(request, "model_request")
    return request


_COMMON_RESPONSE_FIELDS = {
    "ok",
    "api_attempts",
    "tool_calls",
    "usage",
    "error",
}


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _usage(value: Any) -> tuple[dict[str, int | float], bool]:
    result: dict[str, int | float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }
    if not isinstance(value, Mapping):
        return result, False
    complete = True
    for key in result:
        observed = value.get(key)
        if (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) >= 0
            and math.isfinite(float(observed))
        ):
            result[key] = observed
        else:
            complete = False
    return result, complete


def _call_raw(
    chooser: StatelessChooser,
    request: Mapping[str, Any],
    calls: list[dict[str, Any]],
) -> Mapping[str, Any]:
    clean_request = copy.deepcopy(dict(request))
    assert_inference_safe(clean_request, "model_request")
    try:
        raw = chooser.select(copy.deepcopy(clean_request))
    except Exception as exc:  # policy must fail closed at the model boundary
        raw = {
            "ok": False,
            "api_attempts": 0,
            "tool_calls": 0,
            "usage": {},
            "error": f"chooser_exception:{type(exc).__name__}",
        }
    result = dict(raw) if isinstance(raw, Mapping) else {}
    usage, usage_complete = _usage(result.get("usage"))
    attempts = result.get("api_attempts")
    tool_calls = result.get("tool_calls")
    counts_complete = _nonnegative_int(attempts) and _nonnegative_int(tool_calls)
    calls.append(
        {
            "phase": clean_request["phase"],
            "request_id": clean_request["request_id"],
            "request": clean_request,
            "api_attempts": attempts if _nonnegative_int(attempts) else 0,
            "model_tool_calls": tool_calls if _nonnegative_int(tool_calls) else 0,
            "usage": usage,
            "accounting_complete": counts_complete and usage_complete,
            "reported_ok": result.get("ok") is True,
            "error": None
            if result.get("ok") is True
            else str(result.get("error") or "adapter_failure")[:200],
        }
    )
    return result


def _parse_factorized(
    raw: Mapping[str, Any], deck: CandidateDeck, baseline: Pair
) -> tuple[FactorizedChoice | None, str | None]:
    expected = _COMMON_RESPONSE_FIELDS | {"decision", "trigger_choice", "action_choice"}
    if set(raw) != expected or raw.get("ok") is not True:
        return None, "factorized_response_contract_failure"
    if not _nonnegative_int(raw.get("api_attempts")) or not _nonnegative_int(
        raw.get("tool_calls")
    ):
        return None, "factorized_accounting_contract_failure"
    try:
        decision = EditDecision(raw.get("decision"))
    except (TypeError, ValueError):
        return None, "factorized_decision_invalid"
    trigger_choice, action_choice = raw.get("trigger_choice"), raw.get("action_choice")
    if not isinstance(trigger_choice, str) or not isinstance(action_choice, str):
        return None, "factorized_choice_type_invalid"
    baseline_trigger = deck.alias_by_identity[baseline.trigger.identity]
    baseline_action = deck.alias_by_identity[baseline.action.identity]
    trigger_changed = trigger_choice != KEEP
    action_changed = action_choice != KEEP
    expected_changes = {
        EditDecision.KEEP: (False, False),
        EditDecision.ABSTAIN: (False, False),
        EditDecision.CHANGE_TRIGGER: (True, False),
        EditDecision.CHANGE_ACTION: (False, True),
        EditDecision.CHANGE_BOTH: (True, True),
    }[decision]
    if (trigger_changed, action_changed) != expected_changes:
        return None, "factorized_decision_choice_mismatch"
    if trigger_changed:
        if (
            trigger_choice not in deck.trigger_by_alias
            or trigger_choice == baseline_trigger
        ):
            return None, "trigger_choice_outside_top10_or_not_changed"
    elif trigger_choice != KEEP:
        return None, "trigger_keep_contract_failure"
    if action_changed:
        if (
            action_choice not in deck.action_by_alias
            or action_choice == baseline_action
        ):
            return None, "action_choice_outside_top10_or_not_changed"
    elif action_choice != KEEP:
        return None, "action_keep_contract_failure"
    return FactorizedChoice(decision, trigger_choice, action_choice), None


def _materialize(choice: FactorizedChoice, deck: CandidateDeck, baseline: Pair) -> Pair:
    trigger = (
        baseline.trigger
        if choice.trigger_choice == KEEP
        else deck.trigger_by_alias[choice.trigger_choice]
    )
    action = (
        baseline.action
        if choice.action_choice == KEEP
        else deck.action_by_alias[choice.action_choice]
    )
    return Pair(trigger, action)


def _schema_validate(tools: CatalogTools, pair: Pair) -> ValidationResult:
    # These are real catalog reads.  No read count is inferred from model calls.
    tools.read_endpoint_schema(pair.trigger.identity)
    tools.read_endpoint_schema(pair.action.identity)
    return tools.validate_pair(pair.trigger, pair.action)


def _pair_aliases(
    case_id: str, baseline: Pair, proposal: Pair
) -> tuple[dict[str, Pair], list[str]]:
    pairs = [baseline, proposal]
    identity_order = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"farm-r7-pair-id\0{case_id}\0{pair.trigger.identity}\0{pair.action.identity}".encode()
        ).hexdigest(),
    )
    by_alias = {f"P{index + 1:02d}": pair for index, pair in enumerate(identity_order)}
    first_order = sorted(
        by_alias,
        key=lambda alias: hashlib.sha256(
            f"farm-r7-pair-order\0{case_id}\0{alias}".encode()
        ).hexdigest(),
    )
    return by_alias, first_order


def _public_pair(
    alias: str,
    pair: Pair,
    deck: CandidateDeck,
    validation: ValidationResult,
) -> dict[str, Any]:
    return {
        "pair_id": alias,
        "trigger": pair.trigger.public(
            deck.alias_by_identity[pair.trigger.identity], include_schema=True
        ),
        "action": pair.action.public(
            deck.alias_by_identity[pair.action.identity], include_schema=True
        ),
        "schema_validation": validation.public(),
    }


def _verifier_request(
    *,
    case_id: str,
    query: str,
    pair_by_alias: Mapping[str, Pair],
    presentation: Sequence[str],
    deck: CandidateDeck,
    validations: Mapping[tuple[str, str], ValidationResult],
    ordinal: int,
) -> dict[str, Any]:
    request = {
        "schema_version": "farm_r7_pair_verifier_request_v1",
        "request_id": _request_id(case_id, "pair_verification", ordinal),
        "phase": "pair_verification",
        "query": query,
        "instruction": (
            "Compare only the two visible endpoint pairs. Choose the pair whose trigger and "
            "action most exactly implement the request and whose schema evidence is sound. "
            "Use ABSTAIN unless one complete pair is clearly better."
        ),
        "pairs": [
            _public_pair(
                alias,
                pair_by_alias[alias],
                deck,
                validations[
                    (
                        pair_by_alias[alias].trigger.identity,
                        pair_by_alias[alias].action.identity,
                    )
                ],
            )
            for alias in presentation
        ],
        "output_contract": {
            "choice_id": [*pair_by_alias.keys(), ABSTAIN],
            "additional_properties": False,
        },
    }
    assert_inference_safe(request, "model_request")
    return request


def _parse_verifier(
    raw: Mapping[str, Any], allowed: set[str]
) -> tuple[str | None, str | None]:
    expected = _COMMON_RESPONSE_FIELDS | {"choice_id"}
    if set(raw) != expected or raw.get("ok") is not True:
        return None, "verifier_response_contract_failure"
    if not _nonnegative_int(raw.get("api_attempts")) or not _nonnegative_int(
        raw.get("tool_calls")
    ):
        return None, "verifier_accounting_contract_failure"
    choice = raw.get("choice_id")
    if not isinstance(choice, str) or choice not in allowed | {ABSTAIN}:
        return None, "verifier_choice_invalid"
    return choice, None


def dual_unanimity_accepts(votes: Sequence[str | None], challenger_id: str) -> bool:
    """The exhaustive dual-verifier acceptance rule used by the early exit."""

    return len(votes) == 2 and all(vote == challenger_id for vote in votes)


def _accounting(
    calls: Sequence[Mapping[str, Any]], events: Sequence[ToolEvent]
) -> dict[str, Any]:
    usage: dict[str, int | float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }
    for call in calls:
        for key in usage:
            usage[key] += call["usage"][key]
    tool_counts = {
        name: sum(event.tool == name for event in events)
        for name in ("search_catalog", "read_endpoint_schema", "validate_pair")
    }
    return {
        "logical_model_calls": len(calls),
        "api_attempts": sum(int(call["api_attempts"]) for call in calls),
        "model_tool_calls": sum(int(call["model_tool_calls"]) for call in calls),
        "deterministic_tool_calls": len(events),
        "tool_counts": tool_counts,
        "catalog_reads": tool_counts["read_endpoint_schema"],
        "complete": all(bool(call["accounting_complete"]) for call in calls),
        "usage": usage,
    }


def _final_trace(
    *,
    case_id: str,
    policy: Policy,
    baseline: Pair,
    proposal: Pair | None,
    final: Pair,
    decision: str,
    fallback_reason: str | None,
    repair_attempted: bool,
    validations: Sequence[tuple[Pair, ValidationResult]],
    calls: Sequence[Mapping[str, Any]],
    tools: CatalogTools,
    deck: CandidateDeck,
) -> dict[str, Any]:
    return {
        "schema_version": "farm_round7_constraint_agent_trace_v1",
        "case_id": case_id,
        "policy": {
            "candidate_depth_per_side": policy.candidate_depth,
            "evidence_view": policy.evidence_view,
            "verification": policy.verification,
            "max_repairs": policy.max_repairs,
            "safe_fallback": "retrieval_top1_pair",
        },
        "candidate_universe": {
            "trigger": [
                candidate.identity for candidate in deck.trigger_by_alias.values()
            ],
            "action": [
                candidate.identity for candidate in deck.action_by_alias.values()
            ],
        },
        "baseline_pair": baseline.identities,
        "proposal_pair": proposal.identities if proposal is not None else None,
        "final_pair": final.identities,
        "retained_baseline": final == baseline,
        "stable_decision": decision,
        "fallback_reason": fallback_reason,
        "repair_attempted": repair_attempted,
        "validations": [
            {"pair": pair.identities, **validation.public()}
            for pair, validation in validations
        ],
        "compiled": False,
        "compilation_reason": "gold_or_verified_field_bindings_not_available",
        "calls": copy.deepcopy(list(calls)),
        "tool_trace": [event.as_dict() for event in tools.events],
        "accounting": _accounting(calls, tools.events),
    }


def resolve(
    case: Mapping[str, Any], chooser: StatelessChooser, policy: Policy | None = None
) -> dict[str, Any]:
    """Resolve one hydrated inference case with safe top-one fallback."""

    policy = policy or Policy()
    case_id, query, hydrated = _hydrate_case(case, policy)
    tools = CatalogTools(hydrated)
    searched = {
        side: tools.search_catalog(query, side, policy.candidate_depth) for side in Side
    }
    deck = _opaque_deck(case_id, searched, policy)
    baseline = Pair(searched[Side.TRIGGER][0], searched[Side.ACTION][0])
    calls: list[dict[str, Any]] = []
    validations: list[tuple[Pair, ValidationResult]] = []
    request = _factorized_request(
        case_id=case_id,
        query=query,
        deck=deck,
        baseline=baseline,
        phase="factorized_selection",
        ordinal=1,
    )
    raw = _call_raw(chooser, request, calls)
    choice, failure = _parse_factorized(raw, deck, baseline)
    if choice is None:
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=None,
            final=baseline,
            decision="SELECTION_FAILURE",
            fallback_reason=failure,
            repair_attempted=False,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )
    if choice.decision is EditDecision.ABSTAIN:
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=None,
            final=baseline,
            decision="SELECTION_ABSTAIN",
            fallback_reason="selector_abstained",
            repair_attempted=False,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )
    proposal = _materialize(choice, deck, baseline)
    if proposal == baseline:
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=proposal,
            final=baseline,
            decision="KEEP_TOP1",
            fallback_reason=None,
            repair_attempted=False,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )

    proposal_validation = _schema_validate(tools, proposal)
    validations.append((proposal, proposal_validation))
    repair_attempted = False
    if proposal_validation.status is ValidationStatus.INCOMPATIBLE:
        if policy.max_repairs == 0:
            return _final_trace(
                case_id=case_id,
                policy=policy,
                baseline=baseline,
                proposal=proposal,
                final=baseline,
                decision="VALIDATION_REJECT",
                fallback_reason="schema_incompatible_repair_disabled",
                repair_attempted=False,
                validations=validations,
                calls=calls,
                tools=tools,
                deck=deck,
            )
        repair_attempted = True
        repair_request = _factorized_request(
            case_id=case_id,
            query=query,
            deck=deck,
            baseline=baseline,
            phase="validator_repair",
            ordinal=2,
            feedback=proposal_validation,
            prior=proposal,
        )
        repair_raw = _call_raw(chooser, repair_request, calls)
        repair_choice, repair_failure = _parse_factorized(repair_raw, deck, baseline)
        if repair_choice is None:
            return _final_trace(
                case_id=case_id,
                policy=policy,
                baseline=baseline,
                proposal=proposal,
                final=baseline,
                decision="REPAIR_FAILURE",
                fallback_reason=repair_failure,
                repair_attempted=True,
                validations=validations,
                calls=calls,
                tools=tools,
                deck=deck,
            )
        if repair_choice.decision is EditDecision.ABSTAIN:
            return _final_trace(
                case_id=case_id,
                policy=policy,
                baseline=baseline,
                proposal=proposal,
                final=baseline,
                decision="REPAIR_ABSTAIN",
                fallback_reason="repair_selector_abstained",
                repair_attempted=True,
                validations=validations,
                calls=calls,
                tools=tools,
                deck=deck,
            )
        repaired = _materialize(repair_choice, deck, baseline)
        if repaired == baseline:
            return _final_trace(
                case_id=case_id,
                policy=policy,
                baseline=baseline,
                proposal=proposal,
                final=baseline,
                decision="REPAIR_KEEP_TOP1",
                fallback_reason="validator_rejected_initial_proposal",
                repair_attempted=True,
                validations=validations,
                calls=calls,
                tools=tools,
                deck=deck,
            )
        proposal = repaired
        proposal_validation = _schema_validate(tools, proposal)
        validations.append((proposal, proposal_validation))
        if proposal_validation.status is ValidationStatus.INCOMPATIBLE:
            return _final_trace(
                case_id=case_id,
                policy=policy,
                baseline=baseline,
                proposal=proposal,
                final=baseline,
                decision="VALIDATION_REJECT",
                fallback_reason="repair_remained_schema_incompatible",
                repair_attempted=True,
                validations=validations,
                calls=calls,
                tools=tools,
                deck=deck,
            )

    if policy.verification == "none":
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=proposal,
            final=proposal,
            decision="ACCEPT_PROPOSAL",
            fallback_reason=None,
            repair_attempted=repair_attempted,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )

    # Read and validate the baseline through the same real tools before showing
    # either pair to the verifier.  This is evidence parity, not a fake count.
    baseline_validation = _schema_validate(tools, baseline)
    validations.append((baseline, baseline_validation))
    validation_map = {
        (pair.trigger.identity, pair.action.identity): validation
        for pair, validation in validations
    }
    pair_by_alias, first_order = _pair_aliases(case_id, baseline, proposal)
    challenger_id = next(
        alias for alias, pair in pair_by_alias.items() if pair == proposal
    )
    allowed = set(pair_by_alias)

    first_request = _verifier_request(
        case_id=case_id,
        query=query,
        pair_by_alias=pair_by_alias,
        presentation=first_order,
        deck=deck,
        validations=validation_map,
        ordinal=len(calls) + 1,
    )
    first_raw = _call_raw(chooser, first_request, calls)
    first_vote, first_failure = _parse_verifier(first_raw, allowed)
    if policy.verification == "single" and first_vote == challenger_id:
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=proposal,
            final=proposal,
            decision="ACCEPT_PROPOSAL",
            fallback_reason=None,
            repair_attempted=repair_attempted,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )
    # Semantics-preserving early exit: a unanimous pair of challenger votes is
    # already impossible whenever the first vote is not the challenger.  The
    # same branch is the conservative veto for the single-verifier ablation.
    if first_vote != challenger_id:
        if first_vote is None:
            stable, reason = "VERIFIER_FAILURE", first_failure
        elif first_vote == ABSTAIN:
            stable, reason = "VERIFIER_ABSTAIN", "first_verifier_abstained"
        else:
            stable, reason = "KEEP_TOP1", None
        return _final_trace(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            proposal=proposal,
            final=baseline,
            decision=stable,
            fallback_reason=reason,
            repair_attempted=repair_attempted,
            validations=validations,
            calls=calls,
            tools=tools,
            deck=deck,
        )

    second_request = _verifier_request(
        case_id=case_id,
        query=query,
        pair_by_alias=pair_by_alias,
        presentation=list(reversed(first_order)),
        deck=deck,
        validations=validation_map,
        ordinal=len(calls) + 1,
    )
    second_raw = _call_raw(chooser, second_request, calls)
    second_vote, second_failure = _parse_verifier(second_raw, allowed)
    if dual_unanimity_accepts((first_vote, second_vote), challenger_id):
        final, stable, reason = proposal, "ACCEPT_PROPOSAL", None
    elif second_vote is None:
        final, stable, reason = baseline, "VERIFIER_FAILURE", second_failure
    elif second_vote == ABSTAIN:
        final, stable, reason = (
            baseline,
            "VERIFIER_ABSTAIN",
            "second_verifier_abstained",
        )
    else:
        final, stable, reason = baseline, "VERIFIER_DISAGREE", "verifiers_not_unanimous"
    return _final_trace(
        case_id=case_id,
        policy=policy,
        baseline=baseline,
        proposal=proposal,
        final=final,
        decision=stable,
        fallback_reason=reason,
        repair_attempted=repair_attempted,
        validations=validations,
        calls=calls,
        tools=tools,
        deck=deck,
    )


__all__ = [
    "ABSTAIN",
    "KEEP",
    "EditDecision",
    "Policy",
    "ValidationStatus",
    "assert_inference_safe",
    "dual_unanimity_accepts",
    "reference_paths",
    "resolve",
    "validate_compatibility",
]
