#!/usr/bin/env python3
"""Typed, deterministic FARM applet compiler and connector sandbox.

This module is deliberately transport-free.  A planner may produce an
``AppletProgram`` through any model framework, but the program crosses a strict
Pydantic boundary before it can be compiled or executed.  Compilation resolves
only opaque IDs from the supplied retrieval set, checks endpoint roles, checks
every binding against hydrated Dataset-v2 schemas, and rejects invented fields.

The connector sandbox is pure and deterministic: it synthesizes a typed canary
trigger event and records the action invocation in memory.  It never invokes a
live service.  Consequently, successful execution is an operational compile/run
metric, not evidence that a field binding is semantically correct on IFTTT.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


Role = Literal["trigger", "action"]
ValueType = Literal[
    "any", "string", "integer", "number", "boolean", "array", "object", "date", "datetime"
]

_SAFE_SEGMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_CANDIDATE_ID = re.compile(r"^[TA][0-9]{2,4}$")
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
    "reference_bindings",
    "valid_pairs",
}
_ROOT_REFERENCE_KEYS = {"answer", "answers", "label", "labels", "target", "targets"}
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
_REFERENCE_PREFIXES = ("gold_", "reference_", "ground_truth_", "target_")


class FrozenModel(BaseModel):
    """Strict value object used at every executable applet boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _safe_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    parts = value.split(".")
    if any(
        not _SAFE_SEGMENT.fullmatch(part)
        or part.startswith("__")
        or part.casefold() in {"prototype", "constructor"}
        for part in parts
    ):
        raise ValueError(f"unsafe dotted path: {value!r}")
    return value


def _candidate_id(value: str) -> str:
    if not isinstance(value, str) or not _CANDIDATE_ID.fullmatch(value):
        raise ValueError("candidate_id must be an opaque retrieval alias")
    return value


def _normal_type(raw: Any) -> ValueType:
    value = str(raw or "").strip().casefold()
    if not value:
        return "any"
    if "date with time" in value or "iso8601" in value or value in {"datetime", "timestamp"}:
        return "datetime"
    if value == "date" or value.startswith("date ("):
        return "date"
    aliases: dict[str, ValueType] = {
        "any": "any",
        "unknown": "any",
        "str": "string",
        "string": "string",
        "text": "string",
        "url": "string",
        "uri": "string",
        "email": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "decimal": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "map": "object",
        "json": "object",
        "object": "object",
    }
    return aliases.get(value, "any")


class EndpointField(FrozenModel):
    path: str
    value_type: ValueType = "any"
    required: bool = True
    bindable: bool | None = None
    can_have_default: bool | None = None
    label: str | None = None

    _validate_path = field_validator("path")(_safe_path)


class EndpointContract(FrozenModel):
    identity: str = Field(min_length=1)
    role: Role
    service_name: str = Field(min_length=1)
    function_name: str = Field(min_length=1)
    description: str | None = None
    inputs: tuple[EndpointField, ...] = ()
    outputs: tuple[EndpointField, ...] = ()

    @model_validator(mode="after")
    def unique_fields(self) -> "EndpointContract":
        for collection_name, fields in (("inputs", self.inputs), ("outputs", self.outputs)):
            paths = [field.path for field in fields]
            if len(paths) != len(set(paths)):
                raise ValueError(f"duplicate {collection_name} field path")
        if self.role == "trigger" and not self.outputs:
            # Triggers without ingredients are valid; context/constants may still
            # satisfy their action.  No fabricated output is introduced.
            return self
        return self


def _first_text(raw: Mapping[str, Any], keys: Sequence[str], *, default: str = "") -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _native_fields(raw: Any, *, outputs: bool) -> tuple[EndpointField, ...]:
    if raw is None:
        return ()
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw, Mapping) and isinstance(raw.get("properties"), Mapping):
        required = raw.get("required", ())
        required_items = (
            required
            if isinstance(required, Sequence) and not isinstance(required, (str, bytes))
            else ()
        )
        required_names = {
            str(item)
            for item in required_items
        }
        for name, spec in raw["properties"].items():
            mapped = dict(spec) if isinstance(spec, Mapping) else {"type": spec}
            mapped.setdefault("required", str(name) in required_names)
            rows.append((str(name), mapped))
    elif isinstance(raw, Mapping):
        for name, spec in raw.items():
            rows.append((str(name), spec if isinstance(spec, Mapping) else {"type": spec}))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("schema field entries must be mappings")
            path = _first_text(item, ("slug", "path", "name", "field", "id"))
            if not path:
                raise ValueError("schema field is missing a stable path/slug")
            rows.append((path, item))
    else:
        raise ValueError("schema fields must be a mapping or sequence")

    fields: list[EndpointField] = []
    for path, spec in rows:
        required_raw = spec.get("required")
        required = True if outputs or required_raw is None else bool(required_raw)
        fields.append(
            EndpointField(
                path=path,
                value_type=_normal_type(spec.get("type")),
                required=required,
                bindable=spec.get("bindable") if isinstance(spec.get("bindable"), bool) else None,
                can_have_default=(
                    spec.get("can_have_default")
                    if isinstance(spec.get("can_have_default"), bool)
                    else None
                ),
                label=_first_text(spec, ("label", "name")) or None,
            )
        )
    return tuple(sorted(fields, key=lambda field: field.path))


def hydrate_native_endpoint(raw: Mapping[str, Any], expected_role: Role) -> EndpointContract:
    """Hydrate a native Dataset-v2 corpus row without consulting evaluation data."""

    declared = _first_text(raw, ("kind", "role", "side"), default=expected_role).casefold()
    if declared not in {"trigger", "action"}:
        declared = expected_role
    if declared != expected_role:
        raise ValueError(f"endpoint role {declared!r} does not match {expected_role!r}")

    nested = raw.get("schema")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError("nested endpoint schema must be a mapping")
    schema: Mapping[str, Any] = nested or {}
    inputs_raw = schema.get("inputs", schema.get("parameters", raw.get("input_fields")))
    outputs_raw = schema.get("outputs", raw.get("ingredients"))
    identity = _first_text(raw, ("identity", "url", "id"))
    if not identity:
        raise ValueError("endpoint is missing an identity")
    service = _first_text(
        raw,
        ("service_name", "channel_display", "service_display", "channel", "service"),
    )
    function = _first_text(raw, ("function_name", "name", "display", "function_slug"))
    if not service or not function:
        raise ValueError("endpoint is missing service/function display names")
    return EndpointContract(
        identity=identity,
        role=expected_role,
        service_name=service,
        function_name=function,
        description=_first_text(raw, ("description", "main_description")) or None,
        inputs=_native_fields(inputs_raw, outputs=False),
        outputs=_native_fields(outputs_raw, outputs=True),
    )


class RetrievedEndpoint(FrozenModel):
    candidate_id: str
    contract: EndpointContract

    _validate_candidate_id = field_validator("candidate_id")(_candidate_id)


class CandidateSet(FrozenModel):
    triggers: tuple[RetrievedEndpoint, ...]
    actions: tuple[RetrievedEndpoint, ...]

    @model_validator(mode="after")
    def validate_sets(self) -> "CandidateSet":
        ids = [item.candidate_id for item in (*self.triggers, *self.actions)]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate aliases must be unique")
        if any(item.contract.role != "trigger" for item in self.triggers):
            raise ValueError("trigger candidate set contains an action")
        if any(item.contract.role != "action" for item in self.actions):
            raise ValueError("action candidate set contains a trigger")
        return self

    @classmethod
    def from_native(
        cls,
        trigger_candidates: Sequence[Mapping[str, Any]],
        action_candidates: Sequence[Mapping[str, Any]],
    ) -> "CandidateSet":
        def hydrate(rows: Sequence[Mapping[str, Any]], role: Role, prefix: str) -> tuple[RetrievedEndpoint, ...]:
            ranked: list[tuple[int, str, EndpointContract]] = []
            for fallback_rank, row in enumerate(rows, start=1):
                contract = hydrate_native_endpoint(row, role)
                rank = row.get("retrieval_rank", row.get("rank", fallback_rank))
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                    raise ValueError("retrieval rank must be a positive integer")
                ranked.append((rank, contract.identity, contract))
            ranked.sort(key=lambda item: (item[0], item[1]))
            return tuple(
                RetrievedEndpoint(candidate_id=f"{prefix}{index:02d}", contract=contract)
                for index, (_, _, contract) in enumerate(ranked, start=1)
            )

        return cls(
            triggers=hydrate(trigger_candidates, "trigger", "T"),
            actions=hydrate(action_candidates, "action", "A"),
        )

    def resolve(self, candidate_id: str, role: Role) -> EndpointContract | None:
        collection = self.triggers if role == "trigger" else self.actions
        return next(
            (item.contract for item in collection if item.candidate_id == candidate_id),
            None,
        )


class TriggerEndpointRef(FrozenModel):
    candidate_id: str
    role: Literal["trigger"] = "trigger"

    _validate_candidate_id = field_validator("candidate_id")(_candidate_id)


class ActionEndpointRef(FrozenModel):
    candidate_id: str
    role: Literal["action"] = "action"

    _validate_candidate_id = field_validator("candidate_id")(_candidate_id)


class TriggerOutputSource(FrozenModel):
    kind: Literal["trigger_output"]
    path: str

    _validate_path = field_validator("path")(_safe_path)


class ContextSource(FrozenModel):
    kind: Literal["context"]
    path: str

    _validate_path = field_validator("path")(_safe_path)


class ConstantSource(FrozenModel):
    kind: Literal["constant"]
    value: JsonValue

    @field_validator("value")
    @classmethod
    def finite_json(cls, value: JsonValue) -> JsonValue:
        _canonical(value)
        return value


BindingSource = Annotated[
    Union[TriggerOutputSource, ContextSource, ConstantSource],
    Field(discriminator="kind"),
]


class FieldBinding(FrozenModel):
    target_path: str
    source: BindingSource

    _validate_target = field_validator("target_path")(_safe_path)


class AppletProgram(FrozenModel):
    version: Literal["farm.applet/v1"] = "farm.applet/v1"
    trigger: TriggerEndpointRef
    action: ActionEndpointRef
    bindings: tuple[FieldBinding, ...]


class ValidationIssue(FrozenModel):
    code: str = Field(min_length=1)
    location: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected: str | None = None
    observed: str | None = None


class StructuredAppletError(ValueError):
    """Base exception carrying machine-readable, immutable validation issues."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("structured error requires at least one issue")
        super().__init__("; ".join(f"{item.code}@{item.location}" for item in self.issues))

    def as_dict(self) -> dict[str, Any]:
        return {"issues": [item.model_dump(mode="json") for item in self.issues]}


class CompilationError(StructuredAppletError):
    pass


class SandboxExecutionError(StructuredAppletError):
    pass


class AuditEvent(FrozenModel):
    sequence: int = Field(ge=0)
    stage: Literal[
        "compile_started",
        "compile_succeeded",
        "trigger_fired",
        "binding_resolved",
        "action_invoked",
    ]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    details_json: str


class AuditTrace(FrozenModel):
    events: tuple[AuditEvent, ...] = ()

    def append(self, stage: AuditEvent.__annotations__["stage"], payload: Any, **details: JsonValue) -> "AuditTrace":
        assert_inference_safe(details, path="audit")
        event = AuditEvent(
            sequence=len(self.events),
            stage=stage,
            payload_sha256=_digest(payload),
            details_json=_canonical(details),
        )
        return AuditTrace(events=(*self.events, event))


class CompiledBinding(FrozenModel):
    target: EndpointField
    source: BindingSource
    source_type: ValueType


class CompiledApplet(FrozenModel):
    program: AppletProgram
    trigger_contract: EndpointContract
    action_contract: EndpointContract
    binding_plan: tuple[CompiledBinding, ...]
    audit_trace: AuditTrace


def _value_type(value: JsonValue) -> ValueType:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not JSON values")
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "any"


def _types_compatible(source: ValueType, target: ValueType) -> bool:
    return target == "any" or source == "any" or source == target or (source == "integer" and target == "number")


def _runtime_value_matches(value: JsonValue, target: ValueType) -> tuple[bool, str]:
    """Check a JSON runtime value, including typed ISO date encodings."""

    observed = _value_type(value)
    if target == "date" and isinstance(value, str):
        try:
            date.fromisoformat(value)
            return True, "date"
        except ValueError:
            return False, observed
    if target == "datetime" and isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True, "datetime"
        except ValueError:
            return False, observed
    return _types_compatible(observed, target), observed


class AppletCompiler:
    """Compile an applet against one immutable retrieval result."""

    def __init__(
        self,
        candidates: CandidateSet,
        *,
        context_fields: Sequence[EndpointField] = (),
    ) -> None:
        self._candidates = candidates
        self._context = {field.path: field for field in context_fields}

    def compile(self, program: AppletProgram) -> CompiledApplet:
        trace = AuditTrace().append(
            "compile_started",
            program,
            trigger_candidate=program.trigger.candidate_id,
            action_candidate=program.action.candidate_id,
        )
        issues: list[ValidationIssue] = []
        trigger = self._candidates.resolve(program.trigger.candidate_id, "trigger")
        action = self._candidates.resolve(program.action.candidate_id, "action")
        if trigger is None:
            issues.append(
                ValidationIssue(
                    code="candidate_not_retrieved",
                    location="trigger.candidate_id",
                    message="trigger endpoint is not in the retrieved trigger candidates",
                )
            )
        if action is None:
            issues.append(
                ValidationIssue(
                    code="candidate_not_retrieved",
                    location="action.candidate_id",
                    message="action endpoint is not in the retrieved action candidates",
                )
            )
        if issues:
            raise CompilationError(issues)
        assert trigger is not None and action is not None
        if trigger.role != "trigger":
            issues.append(
                ValidationIssue(code="wrong_endpoint_role", location="trigger", message="selected endpoint is not a trigger")
            )
        if action.role != "action":
            issues.append(
                ValidationIssue(code="wrong_endpoint_role", location="action", message="selected endpoint is not an action")
            )

        target_fields = {field.path: field for field in action.inputs}
        trigger_fields = {field.path: field for field in trigger.outputs}
        seen_targets: set[str] = set()
        plan: list[CompiledBinding] = []
        for index, binding in enumerate(program.bindings):
            location = f"bindings[{index}]"
            target = target_fields.get(binding.target_path)
            if target is None:
                issues.append(
                    ValidationIssue(
                        code="invented_action_field",
                        location=f"{location}.target_path",
                        message="binding target is not declared by the selected action",
                        observed=binding.target_path,
                    )
                )
            if binding.target_path in seen_targets:
                issues.append(
                    ValidationIssue(
                        code="duplicate_binding",
                        location=f"{location}.target_path",
                        message="an action field may be bound only once",
                        observed=binding.target_path,
                    )
                )
            seen_targets.add(binding.target_path)

            source_field: EndpointField | None = None
            if isinstance(binding.source, TriggerOutputSource):
                source_field = trigger_fields.get(binding.source.path)
                if source_field is None:
                    issues.append(
                        ValidationIssue(
                            code="invented_trigger_output",
                            location=f"{location}.source.path",
                            message="source is not declared by the selected trigger",
                            observed=binding.source.path,
                        )
                    )
                source_type: ValueType = source_field.value_type if source_field else "any"
            elif isinstance(binding.source, ContextSource):
                source_field = self._context.get(binding.source.path)
                if source_field is None:
                    issues.append(
                        ValidationIssue(
                            code="invented_context_field",
                            location=f"{location}.source.path",
                            message="source is not declared by the execution context",
                            observed=binding.source.path,
                        )
                    )
                source_type = source_field.value_type if source_field else "any"
            else:
                source_type = _value_type(binding.source.value)

            if target is not None and source_field is not None or target is not None and isinstance(binding.source, ConstantSource):
                if isinstance(binding.source, ConstantSource) and binding.source.value is None and target.required:
                    issues.append(
                        ValidationIssue(
                            code="null_required_value",
                            location=location,
                            message="a required action field cannot be bound to null",
                            expected=target.value_type,
                            observed="null",
                        )
                    )
                elif not _types_compatible(source_type, target.value_type):
                    issues.append(
                        ValidationIssue(
                            code="type_mismatch",
                            location=location,
                            message="binding source type cannot satisfy action field type",
                            expected=target.value_type,
                            observed=source_type,
                        )
                    )
                else:
                    plan.append(CompiledBinding(target=target, source=binding.source, source_type=source_type))

        for field in action.inputs:
            if field.required and field.path not in seen_targets:
                issues.append(
                    ValidationIssue(
                        code="missing_required_binding",
                        location=f"action.inputs.{field.path}",
                        message="required action input has no binding",
                        expected=field.value_type,
                    )
                )
        if issues:
            raise CompilationError(issues)

        plan.sort(key=lambda binding: binding.target.path)
        trace = trace.append(
            "compile_succeeded",
            [binding.model_dump(mode="json") for binding in plan],
            binding_count=len(plan),
        )
        return CompiledApplet(
            program=program,
            trigger_contract=trigger,
            action_contract=action,
            binding_plan=tuple(plan),
            audit_trace=trace,
        )


class RepairState(FrozenModel):
    program: AppletProgram
    repairs_used: int = Field(default=0, ge=0, le=1)
    max_repairs: Literal[0, 1] = 1
    last_issues: tuple[ValidationIssue, ...] = ()

    @model_validator(mode="after")
    def bounded(self) -> "RepairState":
        if self.repairs_used > self.max_repairs:
            raise ValueError("repair budget exceeded")
        return self

    def apply_repair(
        self,
        replacement: AppletProgram,
        issues: Sequence[ValidationIssue],
    ) -> "RepairState":
        if self.repairs_used >= self.max_repairs:
            raise ValueError("repair budget exhausted")
        return RepairState(
            program=replacement,
            repairs_used=self.repairs_used + 1,
            max_repairs=self.max_repairs,
            last_issues=tuple(issues),
        )


def _canary_value(value_type: ValueType, seed: str) -> JsonValue:
    number = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)
    if value_type in {"any", "string"}:
        return f"canary_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"
    if value_type == "integer":
        return number % 10_000 + 1
    if value_type == "number":
        return round((number % 1_000_000) / 100.0, 2)
    if value_type == "boolean":
        return bool(number % 2)
    if value_type == "array":
        return [f"canary_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:8]}"]
    if value_type == "object":
        return {"canary": hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]}
    epoch = date(2024, 1, 1) + timedelta(days=number % 365)
    if value_type == "date":
        return epoch.isoformat()
    moment = datetime(epoch.year, epoch.month, epoch.day, 12, 0, tzinfo=timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def _set_dotted(target: dict[str, JsonValue], path: str, value: JsonValue) -> None:
    parts = _safe_path(path).split(".")
    cursor: dict[str, JsonValue] = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            child: dict[str, JsonValue] = {}
            cursor[part] = child
            cursor = child
        elif isinstance(existing, dict):
            cursor = existing
        else:
            raise ValueError(f"overlapping schema paths at {path!r}")
    if parts[-1] in cursor:
        raise ValueError(f"duplicate schema path at {path!r}")
    cursor[parts[-1]] = value


def _get_dotted(source: Mapping[str, JsonValue], path: str) -> JsonValue:
    cursor: JsonValue = dict(source)
    for part in _safe_path(path).split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def deterministic_canary(contract: EndpointContract) -> dict[str, JsonValue]:
    if contract.role != "trigger":
        raise ValueError("canaries may only be generated for trigger contracts")
    payload: dict[str, JsonValue] = {}
    for field in sorted(contract.outputs, key=lambda item: item.path):
        _set_dotted(payload, field.path, _canary_value(field.value_type, f"{contract.identity}:{field.path}"))
    return payload


class ResolvedArgument(FrozenModel):
    path: str
    value: JsonValue

    _validate_path = field_validator("path")(_safe_path)


class ActionInvocation(FrozenModel):
    endpoint_identity: str
    arguments: tuple[ResolvedArgument, ...]
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionResult(FrozenModel):
    success: Literal[True] = True
    invocation: ActionInvocation
    audit_trace: AuditTrace


class SandboxConnectorRegistry:
    """A pure registry of contracts; it has no network or connector hooks."""

    def __init__(self, candidates: CandidateSet):
        self._contracts = {
            endpoint.contract.identity: endpoint.contract
            for endpoint in (*candidates.triggers, *candidates.actions)
        }

    def execute(
        self,
        compiled: CompiledApplet,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> ExecutionResult:
        registered_trigger = self._contracts.get(compiled.trigger_contract.identity)
        registered_action = self._contracts.get(compiled.action_contract.identity)
        if registered_trigger is None or registered_action is None:
            raise SandboxExecutionError(
                [
                    ValidationIssue(
                        code="connector_not_registered",
                        location="registry",
                        message="compiled endpoint is absent from this sandbox registry",
                    )
                ]
            )
        if registered_trigger != compiled.trigger_contract or registered_action != compiled.action_contract:
            raise SandboxExecutionError(
                [
                    ValidationIssue(
                        code="connector_contract_mismatch",
                        location="registry",
                        message="registered contract differs from the compiled contract",
                    )
                ]
            )
        trigger_payload = deterministic_canary(compiled.trigger_contract)
        trace = compiled.audit_trace.append(
            "trigger_fired",
            trigger_payload,
            endpoint_digest=_digest(compiled.trigger_contract.identity),
        )
        context_payload: Mapping[str, JsonValue] = context or {}
        issues: list[ValidationIssue] = []
        arguments: list[ResolvedArgument] = []
        for index, binding in enumerate(compiled.binding_plan):
            try:
                if isinstance(binding.source, TriggerOutputSource):
                    value = _get_dotted(trigger_payload, binding.source.path)
                elif isinstance(binding.source, ContextSource):
                    value = _get_dotted(context_payload, binding.source.path)
                else:
                    value = binding.source.value
            except KeyError:
                issues.append(
                    ValidationIssue(
                        code="missing_runtime_source",
                        location=f"bindings[{index}].source",
                        message="declared binding source is absent at runtime",
                    )
                )
                continue
            matches, observed = _runtime_value_matches(value, binding.target.value_type)
            if not matches:
                issues.append(
                    ValidationIssue(
                        code="runtime_type_mismatch",
                        location=f"bindings[{index}]",
                        message="runtime value does not satisfy the compiled action field",
                        expected=binding.target.value_type,
                        observed=observed,
                    )
                )
                continue
            argument = ResolvedArgument(path=binding.target.path, value=value)
            arguments.append(argument)
            trace = trace.append(
                "binding_resolved",
                argument,
                target=binding.target.path,
                source_kind=binding.source.kind,
            )
        if issues:
            raise SandboxExecutionError(issues)
        arguments.sort(key=lambda item: item.path)
        invocation_payload = {
            "endpoint": compiled.action_contract.identity,
            "arguments": [argument.model_dump(mode="json") for argument in arguments],
        }
        invocation = ActionInvocation(
            endpoint_identity=compiled.action_contract.identity,
            arguments=tuple(arguments),
            receipt_sha256=_digest(invocation_payload),
        )
        trace = trace.append(
            "action_invoked",
            invocation_payload,
            endpoint_digest=_digest(compiled.action_contract.identity),
            argument_count=len(arguments),
        )
        return ExecutionResult(invocation=invocation, audit_trace=trace)


class PublicField(FrozenModel):
    path: str
    value_type: ValueType
    required: bool


class InferenceCandidate(FrozenModel):
    candidate_id: str
    service_name: str
    function_name: str
    inputs: tuple[PublicField, ...]
    outputs: tuple[PublicField, ...]
    description: str | None = None

    _validate_candidate_id = field_validator("candidate_id")(_candidate_id)


class InferenceEnvelope(FrozenModel):
    request_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=16_000)
    trigger_candidates: tuple[InferenceCandidate, ...]
    action_candidates: tuple[InferenceCandidate, ...]


class InferenceLeakError(ValueError):
    def __init__(self, paths: Sequence[str]):
        self.paths = tuple(paths)
        super().__init__(f"forbidden fields crossed inference seam: {list(self.paths)}")


def forbidden_paths(value: Any, path: str = "case") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold()
            child_path = f"{path}.{raw_key}"
            if (
                key in _REFERENCE_KEYS
                or key in _SECRET_KEYS
                or key.startswith(_REFERENCE_PREFIXES)
                or (path == "case" and key in _ROOT_REFERENCE_KEYS)
            ):
                found.append(child_path)
            found.extend(forbidden_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{path}[{index}]"))
    return tuple(found)


def assert_inference_safe(value: Any, *, path: str = "case") -> None:
    paths = forbidden_paths(value, path)
    if paths:
        raise InferenceLeakError(paths)


def build_inference_envelope(
    case: Mapping[str, Any],
    candidates: CandidateSet,
) -> InferenceEnvelope:
    """Construct the only payload shape permitted at a planner/model seam."""

    assert_inference_safe(case)
    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("case requires a non-empty query")
    request_id = case.get("request_id", case.get("group_id", "request"))
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("case request_id/group_id must be a non-empty string")

    def public(items: Sequence[RetrievedEndpoint]) -> tuple[InferenceCandidate, ...]:
        return tuple(
            InferenceCandidate(
                candidate_id=item.candidate_id,
                service_name=item.contract.service_name,
                function_name=item.contract.function_name,
                description=item.contract.description,
                inputs=tuple(
                    PublicField(path=field.path, value_type=field.value_type, required=field.required)
                    for field in item.contract.inputs
                ),
                outputs=tuple(
                    PublicField(path=field.path, value_type=field.value_type, required=field.required)
                    for field in item.contract.outputs
                ),
            )
            for item in items
        )

    envelope = InferenceEnvelope(
        request_id=request_id.strip(),
        query=query.strip(),
        trigger_candidates=public(candidates.triggers),
        action_candidates=public(candidates.actions),
    )
    assert_inference_safe(envelope, path="inference")
    return envelope


__all__ = [
    "ActionEndpointRef",
    "ActionInvocation",
    "AppletCompiler",
    "AppletProgram",
    "AuditEvent",
    "AuditTrace",
    "CandidateSet",
    "CompilationError",
    "CompiledApplet",
    "ConstantSource",
    "ContextSource",
    "EndpointContract",
    "EndpointField",
    "ExecutionResult",
    "FieldBinding",
    "InferenceEnvelope",
    "InferenceLeakError",
    "RepairState",
    "SandboxConnectorRegistry",
    "SandboxExecutionError",
    "TriggerEndpointRef",
    "TriggerOutputSource",
    "ValidationIssue",
    "assert_inference_safe",
    "build_inference_envelope",
    "deterministic_canary",
    "forbidden_paths",
    "hydrate_native_endpoint",
    "run_applet_payload",
]


def run_applet_payload(payload: Mapping[str, Any]) -> ExecutionResult:
    """Validate, compile, and sandbox-run one JSON-compatible applet payload.

    This is the programmatic seam used by the small command-line entry point.
    It accepts no connector implementation and therefore cannot acquire live
    external side effects by configuration.
    """

    allowed = {
        "trigger_candidates",
        "action_candidates",
        "context_fields",
        "program",
        "context",
    }
    invented = sorted(str(key) for key in payload if key not in allowed)
    if invented:
        raise ValueError(f"unexpected execution payload fields: {invented}")
    trigger_rows = payload.get("trigger_candidates")
    action_rows = payload.get("action_candidates")
    if not isinstance(trigger_rows, list) or not all(isinstance(row, Mapping) for row in trigger_rows):
        raise ValueError("trigger_candidates must be a JSON array of objects")
    if not isinstance(action_rows, list) or not all(isinstance(row, Mapping) for row in action_rows):
        raise ValueError("action_candidates must be a JSON array of objects")
    raw_program = payload.get("program")
    if not isinstance(raw_program, Mapping):
        raise ValueError("program must be a JSON object")
    raw_context_fields = payload.get("context_fields", [])
    if not isinstance(raw_context_fields, list):
        raise ValueError("context_fields must be a JSON array")
    context_fields = tuple(
        EndpointField.model_validate_json(_canonical(field)) for field in raw_context_fields
    )
    program = AppletProgram.model_validate_json(_canonical(raw_program))
    context = payload.get("context", {})
    if not isinstance(context, Mapping):
        raise ValueError("context must be a JSON object")
    candidates = CandidateSet.from_native(trigger_rows, action_rows)
    compiled = AppletCompiler(candidates, context_fields=context_fields).compile(program)
    return SandboxConnectorRegistry(candidates).execute(compiled, context=context)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile and canary-run a typed FARM applet")
    parser.add_argument("input", help="JSON payload path, or '-' for stdin")
    arguments = parser.parse_args(argv)
    if arguments.input == "-":
        payload = json.load(sys.stdin)
    else:
        with open(arguments.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("top-level input must be a JSON object")
    result = run_applet_payload(payload)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
