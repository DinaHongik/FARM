"""Pure, evidence-gated configuration module; no model or connector I/O.

Local checks cover explicit primitive constraints and provenance, NOT semantic
binding gold or a connector's complete schema. Receipts must come from trusted
environment adapters, never from model output. This module cannot execute tools.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any


@dataclass(frozen=True)
class Field:
    slug: str
    required: bool | None
    value_type: str | None = None
    bindable: bool = False
    auth_like: bool = False
    enum: tuple[Any, ...] | None = None
    label: str = ""
    help_text: str = ""

    def __post_init__(self):
        if type(self.slug) is not str or not self.slug:
            raise ValueError("field slug must be a nonempty string")
        if self.required is not None and type(self.required) is not bool:
            raise ValueError("requiredness must be boolean or unknown")
        if type(self.bindable) is not bool or type(self.auth_like) is not bool:
            raise ValueError("field capabilities must be booleans")


@dataclass(frozen=True)
class Ingredient:
    slug: str
    value_type: str | None


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    side: str
    schema_revision: str
    fields: tuple[Field, ...] = ()
    ingredients: tuple[Ingredient, ...] = ()

    def __post_init__(self):
        if self.side not in ("trigger", "action") or not self.endpoint_id or not self.schema_revision:
            raise ValueError("endpoint requires side, identity and schema revision")
        if len({f.slug for f in self.fields}) != len(self.fields):
            raise ValueError("duplicate schema field")
        if len({i.slug for i in self.ingredients}) != len(self.ingredients):
            raise ValueError("duplicate ingredient")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    endpoint_id: str
    field_slug: str
    schema_revision: str
    resource_ids: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    side: str
    field_slug: str
    source: dict[str, Any]

    def __post_init__(self):
        if type(self.side) is not str or type(self.field_slug) is not str:
            raise ValueError("decision identity must use strings")


@dataclass(frozen=True)
class Receipt:
    kind: str  # backend_validation, dry_run, execution
    state_sha256: str
    passed: bool


@dataclass(frozen=True)
class Issue:
    code: str
    side: str
    field_slug: str
    severity: str = "error"


@dataclass(frozen=True)
class Report:
    status: str
    locally_valid: bool
    issues: tuple[Issue, ...]
    resolved: tuple[dict[str, Any], ...]
    state_sha256: str
    backend_validation_verified: bool = False
    dry_run_verified: bool = False
    execution_verified: bool = False
    checks_scope: str = "declared primitive/enum constraints and source provenance; not semantic correctness"


_TYPES = {"string": "string", "text": "string", "str": "string", "integer": "integer", "int": "integer",
          "number": "number", "float": "number", "double": "number", "boolean": "boolean", "bool": "boolean"}


def primitive_type(value_type):
    return _TYPES.get(str(value_type or "").casefold().strip())


def _value_matches(value, value_type):
    if value_type == "string":
        return type(value) is str
    if value_type == "integer":
        return type(value) is int
    if value_type == "number":
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if value_type == "boolean":
        return type(value) is bool
    return False


def _transform(text, operation):
    if operation == "identity":
        return text
    stripped = text.strip()
    if operation == "parse_integer" and re.fullmatch(r"[+-]?\d+", stripped):
        return int(stripped)
    if operation == "parse_number" and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped):
        value = float(stripped)
        if math.isfinite(value):
            return value
    if operation == "parse_boolean" and stripped.casefold() in ("true", "false"):
        return stripped.casefold() == "true"
    raise ValueError("unsupported_or_invalid_transformation")


class ConfigurationSession:
    """One selected endpoint pair and its trusted observations, isolated per case.

    The caller injects environment evidence. Model-proposed decisions cannot
    create observations, secret aliases, or execution receipts through this
    interface. The caller must create a new session when evidence changes.
    """
    def __init__(self, *, query: str, trigger: Endpoint, action: Endpoint,
                 observations: tuple[Observation, ...] = (), secret_aliases: tuple[str, ...] = ()):
        if trigger.side != "trigger" or action.side != "action":
            raise ValueError("endpoint sides reversed")
        if len({o.observation_id for o in observations}) != len(observations):
            raise ValueError("duplicate observation identity")
        self.query = query
        self.endpoints = {"trigger": trigger, "action": action}
        self.observations = {o.observation_id: o for o in observations}
        self.secret_aliases = tuple(secret_aliases)

    def compile(self, decisions: tuple[Decision, ...], *, receipts: tuple[Receipt, ...] = ()) -> Report:
        context = {"query": self.query, "endpoints": {k: asdict(v) for k, v in self.endpoints.items()},
            "observations": [asdict(self.observations[k]) for k in sorted(self.observations)],
            "secret_aliases": sorted(self.secret_aliases), "decisions": [asdict(d) for d in decisions]}
        fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        issues, resolved, selected = [], [], {}
        for decision in decisions:
            key = decision.side, decision.field_slug
            if key in selected:
                issues.append(Issue("duplicate_decision", *key))
            selected[key] = decision
            endpoint = self.endpoints.get(decision.side)
            if endpoint is None or decision.field_slug not in {f.slug for f in endpoint.fields}:
                issues.append(Issue("unknown_field", *key))
        for side, endpoint in self.endpoints.items():
            for field in endpoint.fields:
                key = side, field.slug
                decision = selected.get(key)
                if decision is None:
                    issues.append(Issue("missing_decision", *key, "unresolved"))
                    continue
                try:
                    value, kind = self._resolve(endpoint, field, decision.source)
                    resolved.append({"side": side, "field_slug": field.slug, "kind": kind, "value": value})
                except ValueError as error:
                    code = str(error)
                    severity = "unresolved" if code in {"needs_input", "unknown_requiredness", "unknown_value_type", "unknown_ingredient_type"} else "error"
                    issues.append(Issue(code, *key, severity))
        local_valid = not issues
        verified = {kind: False for kind in ("backend_validation", "dry_run", "execution")}
        receipt_groups = {kind: [] for kind in verified}
        for receipt in receipts:
            if type(receipt.kind) is not str or receipt.kind not in receipt_groups or type(receipt.passed) is not bool:
                issues.append(Issue("invalid_receipt", "environment", ""))
            elif receipt.state_sha256 != fingerprint:
                issues.append(Issue("stale_receipt", "environment", ""))
            else:
                receipt_groups[receipt.kind].append(receipt.passed)
        # Contradictory receipt evidence is not silently turned into a success.
        for kind, outcomes in receipt_groups.items():
            if outcomes and not all(outcomes):
                issues.append(Issue("failed_" + kind, "environment", ""))
        # Compute flags only after all evidence is checked; a later failed
        # receipt must not leave an earlier overall-verification flag set.
        for kind, outcomes in receipt_groups.items():
            verified[kind] = bool(outcomes) and all(outcomes) and not issues
        status = ("invalid" if any(i.severity == "error" for i in issues) else "needs_evidence" if issues
                  else "execution_verified" if verified["execution"] else "dry_run_verified" if verified["dry_run"]
                  else "backend_validated" if verified["backend_validation"] else "locally_checked")
        return Report(status, local_valid, tuple(issues), tuple(resolved), fingerprint,
            verified["backend_validation"], verified["dry_run"], verified["execution"])

    def _resolve(self, endpoint, field, source):
        if type(source) is not dict or type(source.get("kind")) is not str:
            raise ValueError("invalid_source_shape")
        kind = source.get("kind")
        shapes = {"omit": {"kind"}, "needs_input": {"kind", "question"},
            "query_span": {"kind", "start", "end", "text", "transform"},
            "trigger_output": {"kind", "ingredient_slug"},
            "resource_ref": {"kind", "observation_id", "resource_id"},
            "secret_ref": {"kind", "alias"}}
        if kind not in shapes or set(source) != shapes[kind]:
            raise ValueError("invalid_source_shape")
        for name in shapes[kind] - {"kind", "start", "end"}:
            if type(source[name]) is not str:
                raise ValueError("invalid_source_shape")
        if kind == "needs_input":
            raise ValueError("needs_input")
        if field.required is None:
            raise ValueError("unknown_requiredness")
        if kind == "omit":
            if field.required:
                raise ValueError("required_omitted")
            return None, kind
        if field.auth_like and kind != "secret_ref":
            raise ValueError("credential_must_use_secret_ref")
        target_type = primitive_type(field.value_type)
        if target_type is None:
            raise ValueError("unknown_value_type")
        if kind == "query_span":
            start, end, text = source["start"], source["end"], source["text"]
            if (type(start) is not int or type(end) is not int or type(text) is not str
                    or start < 0 or end <= start or end > len(self.query) or self.query[start:end] != text):
                raise ValueError("ungrounded_query_span")
            value = _transform(text, source["transform"])
        elif kind == "resource_ref":
            observation = self.observations.get(source["observation_id"])
            if observation is None or source["resource_id"] not in observation.resource_ids:
                raise ValueError("unobserved_resource")
            if (observation.endpoint_id != endpoint.endpoint_id or observation.field_slug != field.slug
                    or observation.schema_revision != endpoint.schema_revision):
                raise ValueError("resource_scope_mismatch")
            value = source["resource_id"]
        elif kind == "secret_ref":
            if not field.auth_like or source["alias"] not in self.secret_aliases:
                raise ValueError("invalid_secret_ref")
            return {"secret_alias": source["alias"]}, kind
        else:
            if endpoint.side != "action":
                raise ValueError("future_trigger_output_unavailable")
            ingredient = next((i for i in self.endpoints["trigger"].ingredients if i.slug == source["ingredient_slug"]), None)
            if ingredient is None or not field.bindable:
                raise ValueError("invalid_trigger_binding")
            source_type = primitive_type(ingredient.value_type)
            if source_type is None:
                raise ValueError("unknown_ingredient_type")
            if source_type != target_type and not (source_type == "integer" and target_type == "number"):
                raise ValueError("type_mismatch")
            if field.enum is not None:
                raise ValueError("runtime_enum_constraint_unverified")
            return {"trigger_ingredient": ingredient.slug}, kind
        if not _value_matches(value, target_type):
            raise ValueError("type_mismatch")
        if field.enum is not None and not any(type(value) is type(option) and value == option for option in field.enum):
            raise ValueError("enum_mismatch")
        return value, kind
