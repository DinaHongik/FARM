#!/usr/bin/env python3
"""Run one resumable Round8 executable-agent arm on the frozen dev screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from executable_applet import (
    ActionEndpointRef,
    AppletCompiler,
    AppletProgram,
    CandidateSet,
    CompilationError,
    ContextSource,
    EndpointField,
    FieldBinding,
    SandboxConnectorRegistry,
    SandboxExecutionError,
    TriggerEndpointRef,
    TriggerOutputSource,
    ValidationIssue,
)
from dspy_executable_agent import (
    BinderCriticResponse,
    BoundedDSPyApplet,
    EndpointProposal,
    FactorizedDSPyApplet,
    PlannerResponse,
    PortUsage,
    SpecialistResponse,
    TriggerConsensusDSPyApplet,
    TriggerConsensusResponse,
)
from frozen_router import routing_score as frozen_routing_score
from frozen_router import validate_router


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
SCREEN_HASH = "087c4f5d87c357952a3a94c02abe359710d06be6927859b1644da092cbf87f14"
ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
    "trigger_consensus_executable",
)
CONSENSUS_ROUTING_THRESHOLD = 0.40
OUTCOMES = (
    "function_trigger", "function_action", "function_joint",
    "service_trigger", "service_action", "service_joint",
)
_URL = re.compile(r"\b(?:[a-z][a-z0-9+.-]*://|www\.)\S+", re.I)
_WORDS = re.compile(r"[a-z0-9]+")


def validate_local_ollama_host(value: str) -> str:
    """Accept only an explicit local Ollama HTTP endpoint with a port."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise ValueError("invalid local Ollama host") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("local Ollama host must be http://127.0.0.1:PORT")
    return f"http://{parsed.hostname}:{port}"


def output_root_for_mode(
    run_root: Path,
    *,
    synthetic: bool,
    local: bool,
    smoke: int | None,
    model_digest: str | None = None,
) -> Path:
    if synthetic and local:
        raise ValueError("synthetic and local Ollama modes are mutually exclusive")
    if synthetic:
        return run_root / "synthetic"
    if local:
        if model_digest is None or re.fullmatch(r"[0-9a-f]{64}", model_digest) is None:
            raise ValueError("local output requires the exact 64-character model digest")
        base = run_root / "local_granite" / model_digest
        return base / f"smoke-{smoke}" if smoke is not None else base / "full"
    return run_root / "smoke" if smoke is not None else run_root


def transport_metadata(*, local: bool) -> dict[str, Any]:
    return {
        "transport": "local_loopback" if local else "ollama_cloud",
        "data_left_dgx": not local,
    }


def select_smoke_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    count: int,
    consensus_router: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Keep ordinary prefix smokes; consensus smokes use the routed subsequence."""

    if count < 1:
        raise ValueError("smoke count must be positive")
    if arm != "trigger_consensus_executable":
        return [dict(row) for row in rows[:count]]
    if consensus_router is None:
        raise ValueError("consensus smoke requires the frozen router")
    selected = [
        dict(row)
        for row in rows
        if frozen_routing_score(row, consensus_router)
        >= CONSENSUS_ROUTING_THRESHOLD
    ][:count]
    if not selected:
        raise RuntimeError("consensus smoke has no routed cases")
    return selected


def _json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any, *, secret: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if secret and secret in payload:
        raise RuntimeError("secret persistence gate failed")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Mapping[str, Any], *, secret: str = "") -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if secret and secret in payload:
        raise RuntimeError("secret persistence gate failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        group = row.get("group_id") if isinstance(row, Mapping) else None
        if not isinstance(group, str) or not group or group in seen:
            raise ValueError(f"invalid or duplicate record at JSONL line {number}")
        seen.add(group)
        rows.append(dict(row))
    return rows


def _validate_record_prefix(
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
) -> None:
    """Require append-only records to be the exact canonical selection prefix."""
    if len(records) > len(selected):
        raise RuntimeError("resume records exceed the canonical selection")
    for index, record in enumerate(records):
        expected = selected[index]
        if record.get("schema_version") != "farm_round8_executable_record_v1":
            raise RuntimeError(f"resume record {index} has an invalid schema")
        if record.get("arm_id") != arm:
            raise RuntimeError(f"resume record {index} belongs to a different arm")
        if record.get("group_id") != expected.get("group_id"):
            raise RuntimeError("resume records are not the canonical selected prefix")
        if "semantic_family_id" in expected and record.get("semantic_family_id") != expected.get("semantic_family_id"):
            raise RuntimeError(f"resume record {index} has a mismatched semantic family")


def load_resume_state(
    records_path: Path,
    attempts_path: Path,
    progress_path: Path,
    binding: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[dict[str, Any]]:
    """Validate an interrupted append-only run before issuing another request."""
    partial_exists = any(path.exists() for path in (records_path, attempts_path, progress_path))
    if not partial_exists:
        return []
    if not progress_path.exists():
        raise RuntimeError("partial state exists without its progress binding")
    progress = read_json(progress_path)
    if not isinstance(progress, Mapping):
        raise RuntimeError("resume progress must be a JSON object")
    if progress.get("phase") != "running":
        raise RuntimeError("resume progress is not in the running phase")
    if progress.get("binding") != binding:
        raise RuntimeError("resume binding changed")
    records = load_jsonl(records_path)
    _validate_record_prefix(records, selected, arm)
    if progress.get("completed_rows") != len(records):
        raise RuntimeError("resume completed row count does not match records")
    if progress.get("target_rows") != len(selected):
        raise RuntimeError("resume target row count changed")
    return records


def validate_completed_state(
    result_path: Path,
    progress_path: Path,
    records_path: Path,
    binding: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    """Authenticate all persisted artifacts before treating a run as complete."""
    if not progress_path.exists() or not records_path.exists():
        raise RuntimeError("completed result lacks progress or canonical records")
    result = read_json(result_path)
    if not isinstance(result, Mapping):
        raise RuntimeError("completed result must be a JSON object")
    if result.get("status") != "completed":
        raise RuntimeError("result is not completed")
    if result.get("binding") != binding:
        raise RuntimeError("completed result binding changed")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("rows") != len(selected):
        raise RuntimeError("completed result row count changed")

    records = load_jsonl(records_path)
    _validate_record_prefix(records, selected, arm)
    if len(records) != len(selected):
        raise RuntimeError("completed records row count changed")

    progress = read_json(progress_path)
    if not isinstance(progress, Mapping):
        raise RuntimeError("completed progress must be a JSON object")
    if progress.get("phase") != "complete":
        raise RuntimeError("completed progress phase changed")
    if progress.get("binding") != binding:
        raise RuntimeError("completed progress binding changed")
    if progress.get("completed_rows") != len(selected) or progress.get("target_rows") != len(selected):
        raise RuntimeError("completed progress row count changed")
    output = progress.get("output")
    if not isinstance(output, str) or Path(output).resolve() != result_path.resolve():
        raise RuntimeError("completed progress output path changed")
    if progress.get("output_sha256") != sha256_file(result_path):
        raise RuntimeError("completed result hash changed")
    return dict(result)


def ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([row["group_id"] for row in rows]).encode()).hexdigest()


def hash_order(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (hashlib.sha256(str(row["group_id"]).encode()).hexdigest(), str(row["group_id"])))


def select_discovery_rows(
    discovery_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the already-consumed, discovery-only Round6 screen.

    Round8 deliberately has no code path to the 1,145-row development files.
    Its sole evaluation input is a hash-pinned 231-row Round6 record artifact,
    which predates this architecture and contains no reserved-confirmation row.
    Only the frozen input fields are projected from that artifact.
    """

    allowed = (
        "group_id",
        "semantic_family_id",
        "query",
        "trigger_candidates",
        "action_candidates",
        "valid_pairs",
    )
    selected: list[dict[str, Any]] = []
    for index, source in enumerate(discovery_records):
        if not isinstance(source, Mapping):
            raise ValueError(f"discovery row {index} is not an object")
        missing = [key for key in allowed if key not in source]
        if missing:
            raise ValueError(f"discovery row {index} lacks fields: {missing}")
        row = {key: source[key] for key in allowed}
        if not isinstance(row["group_id"], str) or not row["group_id"]:
            raise ValueError(f"invalid discovery group id at row {index}")
        if not isinstance(row["semantic_family_id"], str) or not row["semantic_family_id"]:
            raise ValueError(f"invalid discovery family id at row {index}")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"invalid discovery query at row {index}")
        for side in ("trigger", "action"):
            candidates = row[f"{side}_candidates"]
            if not isinstance(candidates, list) or len(candidates) < 10:
                raise ValueError(f"discovery row {index} lacks ten {side} candidates")
        if not isinstance(row["valid_pairs"], list) or not row["valid_pairs"]:
            raise ValueError(f"discovery row {index} lacks valid pairs")
        selected.append(row)
    if len(selected) != 231:
        raise ValueError("Round8 expects exactly 231 discovery-only rows")
    if len({row["group_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate discovery group id")
    if len({row["semantic_family_id"] for row in selected}) != len(selected):
        raise ValueError("discovery screen is not family-purged")
    if ids_hash(selected) != SCREEN_HASH:
        raise ValueError("discovery screen identity changed")
    return selected, {
        "source_rows": 231,
        "source_scope": "already_consumed_discovery_only",
        "raw_window": None,
        "selected_rows": 231,
        "selected_group_ids_sha256": SCREEN_HASH,
        "reserved_confirmation_rows_read": 0,
        "reserved_payload_sent_to_model": False,
    }


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DirectSelection(StrictModel):
    trigger_choice: str = Field(min_length=1, pattern=r"^(DEFER_TO_RETRIEVER|T\d{2,4})$")
    action_choice: str = Field(min_length=1, pattern=r"^(DEFER_TO_RETRIEVER|A\d{2,4})$")


class SpecialistSelection(StrictModel):
    choice: str = Field(min_length=1, pattern=r"^(DEFER_TO_RETRIEVER|[TA]\d{2,4})$")
    confidence: float = Field(ge=0.0, le=1.0)


class ConsensusSelection(StrictModel):
    """Model-visible choice contract; confidence is deliberately absent."""

    choice: str = Field(
        min_length=1,
        pattern=r"^(DEFER_TO_RETRIEVER|[SF]\d{2,4})$",
    )


class PlanSubmission(StrictModel):
    program: AppletProgram


class BinderSubmission(StrictModel):
    approved: bool
    program: AppletProgram | None = None
    rejection_code: str | None = None

    @model_validator(mode="after")
    def complete(self) -> "BinderSubmission":
        if self.approved != (self.program is not None) or self.approved == (self.rejection_code is not None):
            raise ValueError("binder verdict fields are inconsistent")
        return self


def _sanitize(value: Any, *, limit: int = 1600) -> str:
    text = " ".join(str(value or "").split())
    return _URL.sub("[redacted]", text)[:limit]


def _set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _canary(value_type: str, seed: str) -> Any:
    number = int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16)
    if value_type in {"any", "string", "date", "datetime"}:
        return f"canary_{number:012x}" if value_type in {"any", "string"} else ("2024-01-01" if value_type == "date" else "2024-01-01T12:00:00Z")
    if value_type == "integer": return number % 10000 + 1
    if value_type == "number": return (number % 100000) / 100.0
    if value_type == "boolean": return bool(number % 2)
    if value_type == "array": return [f"canary_{number:08x}"]
    if value_type == "object": return {"canary": f"{number:08x}"}
    return f"canary_{number:012x}"


@dataclass(frozen=True)
class TriggerCandidateView:
    name: Literal["schema_m5", "fused_m10"]
    public_candidates: tuple[Mapping[str, Any], ...]
    alias_to_primary: Mapping[str, str]
    primary_candidate_ids: tuple[str, ...]
    presentation_sha256: str


@dataclass(frozen=True)
class PreparedCase:
    group_id: str
    semantic_family_id: str
    request_id: str
    query: str
    candidates: CandidateSet
    descriptions: Mapping[str, str]
    baseline_trigger_alias: str
    baseline_action_alias: str
    alias_to_identity: Mapping[str, str]
    context_fields: tuple[EndpointField, ...]
    context: Mapping[str, Any]
    trigger_seed_support: Mapping[str, int]
    consensus_views: Mapping[str, TriggerCandidateView]
    consensus_routing_score: float | None


def _build_consensus_trigger_view(
    *,
    group_id: str,
    row: Mapping[str, Any],
    candidates: CandidateSet,
    identity_to_primary: Mapping[str, str],
    corpora: Mapping[str, Mapping[str, Mapping[str, Any]]],
    name: Literal["schema_m5", "fused_m10"],
    depth: Literal[5, 10],
) -> TriggerCandidateView:
    ranked = row.get("trigger_candidates")
    if not isinstance(ranked, list) or len(ranked) < depth:
        raise ValueError(f"{name} requires {depth} ranked triggers")
    entries: list[tuple[str, dict[str, Any]]] = []
    evidence_key = "text_schema" if name == "schema_m5" else "text_plain"
    for raw in ranked[:depth]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} trigger candidate is not an object")
        identity = raw.get("url")
        if not isinstance(identity, str) or identity not in identity_to_primary:
            raise ValueError(f"{name} trigger escaped frozen candidates")
        primary_id = identity_to_primary[identity]
        candidate = candidates.resolve(primary_id, "trigger")
        if candidate is None:
            raise ValueError(f"{name} trigger escaped typed candidates")
        native = corpora["trigger"][identity]
        evidence = raw.get(evidence_key)
        if not isinstance(evidence, str) or not evidence.strip():
            evidence = native.get("description") or native.get("title") or candidate.function_name
        public: dict[str, Any] = {
            "service_name": _sanitize(candidate.service_name, limit=240),
            "function_name": _sanitize(candidate.function_name, limit=400),
            "evidence": _sanitize(evidence),
        }
        if name == "schema_m5":
            public["inputs"] = [
                {
                    "path": field.path,
                    "value_type": field.value_type,
                    "required": field.required,
                }
                for field in candidate.inputs
            ]
            public["outputs"] = [
                {
                    "path": field.path,
                    "value_type": field.value_type,
                    "required": field.required,
                }
                for field in candidate.outputs
            ]
        entries.append((primary_id, public))
    generator = random.Random(
        int(
            hashlib.sha256(
                f"round8:trigger-consensus:{group_id}:{name}".encode()
            ).hexdigest(),
            16,
        )
    )
    generator.shuffle(entries)
    prefix = "S" if name == "schema_m5" else "F"
    public_candidates: list[dict[str, Any]] = []
    alias_to_primary: dict[str, str] = {}
    for index, (primary_id, public) in enumerate(entries, 1):
        alias = f"{prefix}{index:02d}"
        alias_to_primary[alias] = primary_id
        public_candidates.append({"candidate_id": alias, **public})
    public_payload = tuple(public_candidates)
    assert_model_safe(public_payload)
    return TriggerCandidateView(
        name=name,
        public_candidates=public_payload,
        alias_to_primary=alias_to_primary,
        primary_candidate_ids=tuple(alias_to_primary.values()),
        presentation_sha256=_hash(
            {
                "name": name,
                "alias_to_primary": alias_to_primary,
                "public_candidates": public_payload,
            }
        ),
    )


def prepare_case(
    row: Mapping[str, Any],
    corpora: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    consensus_routing_score: float | None = None,
    require_consensus_metadata: bool = False,
) -> PreparedCase:
    group = str(row.get("group_id") or "")
    family = str(row.get("semantic_family_id") or group)
    query = row.get("query")
    if not group or not isinstance(query, str) or not query.strip():
        raise ValueError("screen row lacks group/query")
    raw_by_side: dict[str, list[dict[str, Any]]] = {}
    baseline: dict[str, str] = {}
    for side in ("trigger", "action"):
        ranked = row.get(f"{side}_candidates")
        if not isinstance(ranked, list) or len(ranked) < 10:
            raise ValueError("each side requires ten independent candidates")
        baseline[side] = str(ranked[0]["url"])
        hydrated = []
        for candidate in ranked[:10]:
            identity = candidate.get("url")
            if not isinstance(identity, str) or identity not in corpora[side]:
                raise ValueError("candidate absent from frozen corpus")
            native = dict(corpora[side][identity])
            for key in ("rank", "retrieval_rank", "retrieval_score", "seed_ranks", "text_plain", "text_schema"):
                native.pop(key, None)
            hydrated.append(native)
        random.Random(int(hashlib.sha256(f"round8:{group}:{side}".encode()).hexdigest(), 16)).shuffle(hydrated)
        raw_by_side[side] = hydrated
    candidates = CandidateSet.from_native(raw_by_side["trigger"], raw_by_side["action"])
    descriptions: dict[str, str] = {}
    alias_to_identity: dict[str, str] = {}
    native_maps = {side: {str(x["url"]): x for x in raw_by_side[side]} for side in ("trigger", "action")}
    for item in (*candidates.triggers, *candidates.actions):
        alias_to_identity[item.candidate_id] = item.contract.identity
        native = native_maps[item.contract.role][item.contract.identity]
        descriptions[item.candidate_id] = _sanitize(native.get("description") or native.get("title") or item.contract.function_name)
    inverse = {identity: alias for alias, identity in alias_to_identity.items()}
    seed_support: dict[str, int] = {}
    ranked_triggers = row.get("trigger_candidates")
    assert isinstance(ranked_triggers, list)
    for raw in ranked_triggers[:10]:
        identity = raw.get("url") if isinstance(raw, Mapping) else None
        seeds = raw.get("seed_ranks") if isinstance(raw, Mapping) else None
        if require_consensus_metadata and not isinstance(seeds, Mapping):
            raise ValueError("consensus trigger candidate lacks seed_ranks")
        if isinstance(identity, str) and identity in inverse:
            seed_support[inverse[identity]] = (
                sum(value is not None for value in seeds.values())
                if isinstance(seeds, Mapping)
                else 0
            )
    consensus_views = {
        "schema_m5": _build_consensus_trigger_view(
            group_id=group,
            row=row,
            candidates=candidates,
            identity_to_primary=inverse,
            corpora=corpora,
            name="schema_m5",
            depth=5,
        ),
        "fused_m10": _build_consensus_trigger_view(
            group_id=group,
            row=row,
            candidates=candidates,
            identity_to_primary=inverse,
            corpora=corpora,
            name="fused_m10",
            depth=10,
        ),
    }
    fields: list[EndpointField] = []
    context: dict[str, Any] = {}
    for item in candidates.actions:
        for target in item.contract.inputs:
            if not target.required:
                continue
            path = f"config.{item.candidate_id}.{target.path}"
            fields.append(EndpointField(path=path, value_type=target.value_type, required=True))
            _set_dotted(context, path, _canary(target.value_type, f"{item.candidate_id}:{target.path}"))
    return PreparedCase(
        group_id=group, semantic_family_id=family,
        request_id="R" + hashlib.sha256(f"round8:{group}".encode()).hexdigest()[:24],
        query=query.strip(), candidates=candidates, descriptions=descriptions,
        baseline_trigger_alias=inverse[baseline["trigger"]],
        baseline_action_alias=inverse[baseline["action"]],
        alias_to_identity=alias_to_identity, context_fields=tuple(fields), context=context,
        trigger_seed_support=seed_support,
        consensus_views=consensus_views,
        consensus_routing_score=consensus_routing_score,
    )


def _candidate_public(item: Any, description: str) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id,
        "service_name": _sanitize(item.contract.service_name, limit=240),
        "function_name": _sanitize(item.contract.function_name, limit=400),
        "description": description,
        "inputs": [{"path": x.path, "value_type": x.value_type, "required": x.required} for x in item.contract.inputs],
        "outputs": [{"path": x.path, "value_type": x.value_type, "required": x.required} for x in item.contract.outputs],
    }


def public_case_payload(case: PreparedCase) -> dict[str, Any]:
    payload = {
        "query": _sanitize(case.query, limit=16000),
        "trigger_candidates": [_candidate_public(x, case.descriptions[x.candidate_id]) for x in case.candidates.triggers],
        "action_candidates": [_candidate_public(x, case.descriptions[x.candidate_id]) for x in case.candidates.actions],
        "available_context_fields": [x.model_dump(mode="json") for x in case.context_fields],
    }
    assert_model_safe(payload)
    return payload


def assert_model_safe(value: Any, path: str = "request") -> None:
    forbidden = {"group_id", "semantic_family_id", "url", "identity", "rank", "retrieval_rank", "retrieval_score", "score", "valid_pairs", "gold", "labels", "target", "api_key", "authorization", "password", "secret", "token"}
    if isinstance(value, BaseModel): value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in forbidden or normalized.startswith(("gold_", "reference_", "ground_truth_")):
                raise ValueError(f"forbidden model field: {path}.{key}")
            assert_model_safe(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value): assert_model_safe(child, f"{path}[{index}]")
    elif isinstance(value, str) and _URL.search(value):
        raise ValueError(f"URL crossed model boundary: {path}")


def deterministic_autowire(case: PreparedCase, trigger_alias: str, action_alias: str) -> AppletProgram:
    trigger = case.candidates.resolve(trigger_alias, "trigger")
    action = case.candidates.resolve(action_alias, "action")
    if trigger is None or action is None:
        raise ValueError("selection escaped retrieved candidates")
    bindings = []
    outputs = {(field.path, field.value_type): field for field in trigger.outputs}
    contexts = {field.path for field in case.context_fields}
    for target in action.inputs:
        if not target.required: continue
        match = outputs.get((target.path, target.value_type))
        if match is not None:
            source = TriggerOutputSource(kind="trigger_output", path=match.path)
        else:
            path = f"config.{action_alias}.{target.path}"
            if path not in contexts: raise ValueError("autowire context contract missing")
            source = ContextSource(kind="context", path=path)
        bindings.append(FieldBinding(target_path=target.path, source=source))
    return AppletProgram(
        trigger=TriggerEndpointRef(candidate_id=trigger_alias),
        action=ActionEndpointRef(candidate_id=action_alias), bindings=tuple(bindings),
    )


@dataclass(frozen=True)
class ToolCall:
    role: str
    ok: bool
    value: BaseModel | None
    transport_attempts: int
    model_tool_calls: int
    provider_requests: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    error_code: str | None
    schema_valid: bool


class NativeOllamaToolClient:
    def __init__(self, host: str, api_key: str | None, model: str, journal_path: Path, *, client: Any | None = None, timeout: float = 240, retry_delay: float = 0, seed: int = 42, think: str | None = "low"):
        if not host or not model: raise ValueError("host and model are required")
        if api_key is not None and not api_key: raise ValueError("api_key must be non-empty or None")
        if api_key is None:
            host = validate_local_ollama_host(host)
        if think not in {None, "low", "medium", "high"}: raise ValueError("unsupported think setting")
        base = host.rstrip("/")
        self.endpoint = base + ("/chat" if base.endswith("/api") else "/api/chat")
        self.api_key, self.model, self.journal_path = api_key, model, journal_path
        self.timeout, self.seed, self.think = timeout, seed, think
        self._lock = threading.Lock()
        self._owns = client is None
        if client is None:
            import httpx
            client = httpx.Client(timeout=timeout, trust_env=api_key is not None)
        self.client = client

    def close(self) -> None:
        if self._owns: self.client.close()

    def _journal(self, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True)
        if self.api_key and self.api_key in payload: raise RuntimeError("attempt journal secret gate failed")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n"); handle.flush(); os.fsync(handle.fileno())

    def call_tool(self, *, semantic_id: str, role: str, public_request: Mapping[str, Any], tool_name: str, description: str, output_model: type[BaseModel], parameter_schema: Mapping[str, Any] | None = None) -> ToolCall:
        from jsonschema import Draft202012Validator

        assert_model_safe(public_request)
        schema = dict(parameter_schema or output_model.model_json_schema())
        Draft202012Validator.check_schema(schema)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Use only supplied evidence. Never invent a candidate. Call the terminal tool exactly once and emit no prose."},
                {"role": "user", "content": json.dumps(public_request, ensure_ascii=False, sort_keys=True)},
            ],
            "tools": [{"type": "function", "function": {"name": tool_name, "description": description, "parameters": schema}}],
            "stream": False, "options": {"temperature": 0, "seed": self.seed},
        }
        if self.think is not None:
            payload["think"] = self.think
        serialized = _json(payload)
        if self.api_key and self.api_key in serialized: raise RuntimeError("request secret gate failed")
        request_hash = hashlib.sha256(serialized.encode()).hexdigest()
        self._journal({"event": "request_started", "semantic_id": semantic_id, "role": role, "request_hash": request_hash, "attempt": 1})
        started = time.monotonic(); response = None; error = None
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key is not None:
                headers["Authorization"] = "Bearer " + self.api_key
            response = self.client.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        except Exception:
            error = "transport_error"
        latency = max(0.0, time.monotonic() - started)
        status = getattr(response, "status_code", None)
        value = None
        raw = b""
        if response is not None:
            raw = getattr(response, "content", b"") or b""
            try: value = response.json()
            except Exception: error = "invalid_response_json"
        prompt = int(value.get("prompt_eval_count") or 0) if isinstance(value, Mapping) else 0
        completion = int(value.get("eval_count") or 0) if isinstance(value, Mapping) else 0
        calls_count = 0; parsed = None; schema_valid = False
        diagnostics: list[dict[str, Any]] = []
        if error is None and (not isinstance(status, int) or not 200 <= status < 300): error = f"http_status_{status}"
        if error is None and not isinstance(value, Mapping):
            error = "response_schema_invalid"
            diagnostics.append({"loc": ["response"], "type": "mapping_required"})
        if error is None and value.get("model") != self.model:
            error = "response_model_mismatch"
            diagnostics.append({"loc": ["response", "model"], "type": "literal_mismatch"})
        if error is None and value.get("done") is not True:
            error = "response_not_done"
            diagnostics.append({"loc": ["response", "done"], "type": "literal_mismatch"})
        if error is None and value.get("done_reason") != "stop":
            error = "nonterminal_done_reason"
            diagnostics.append({"loc": ["response", "done_reason"], "type": "literal_mismatch"})
        if error is None:
            calls = (value.get("message") or {}).get("tool_calls") if isinstance(value, Mapping) else None
            if not isinstance(calls, list) or len(calls) != 1: error = "invalid_tool_call_count"
            else:
                calls_count = len(calls); function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
                if not isinstance(function, Mapping) or function.get("name") != tool_name: error = "wrong_tool_name"
                else:
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try: arguments = json.loads(arguments)
                        except Exception:
                            arguments = None
                            diagnostics.append({"loc": ["arguments"], "type": "json_parse"})
                    schema_errors = sorted(
                        Draft202012Validator(schema).iter_errors(arguments),
                        key=lambda item: tuple(str(part) for part in item.absolute_path),
                    )
                    if schema_errors:
                        error = "invalid_tool_schema"
                        diagnostics.extend({
                            "loc": ["arguments", *(str(part) for part in item.absolute_path)],
                            "type": f"jsonschema.{item.validator}",
                        } for item in schema_errors)
                    else:
                        schema_valid = True
                        try:
                            parsed = output_model.model_validate_json(json.dumps(arguments), strict=True)
                        except ValidationError as validation:
                            error = "invalid_tool_arguments"
                            diagnostics.extend({
                                "loc": ["arguments", *(str(part) for part in item["loc"])],
                                "type": f"pydantic.{item['type']}",
                            } for item in validation.errors(include_url=False, include_input=False))
                        except (ValueError, TypeError):
                            error = "invalid_tool_arguments"
                            diagnostics.append({"loc": ["arguments"], "type": "pydantic.value_error"})
        ok = error is None and parsed is not None
        self._journal({
            "event": "request_finished", "semantic_id": semantic_id, "role": role, "request_hash": request_hash,
            "attempt": 1, "http_status": status, "response_hash": hashlib.sha256(raw).hexdigest() if raw else None,
            "outcome": "valid_tool_call" if ok else error, "model_tool_calls": calls_count,
            "prompt_tokens": prompt, "completion_tokens": completion, "latency_seconds": latency,
            **({"validation_errors": diagnostics} if diagnostics else {}),
        })
        return ToolCall(role, ok, parsed, 1, calls_count, 1, prompt, completion, latency, error, schema_valid)


def _tool_schema(model: type[BaseModel], enums: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    schema = model.model_json_schema()
    for field_name, choices in enums.items(): schema["properties"][field_name]["enum"] = list(choices)
    return schema


def plan_tool_schema(
    case: PreparedCase,
    *,
    fixed_trigger: str | None = None,
    fixed_action: str | None = None,
    binder: bool = False,
) -> dict[str, Any]:
    """Build the exact model-visible program grammar for one retrieval set."""
    trigger_ids = [item.candidate_id for item in case.candidates.triggers]
    action_ids = [item.candidate_id for item in case.candidates.actions]
    if fixed_trigger is not None and fixed_trigger not in trigger_ids:
        raise ValueError("fixed trigger escaped candidates")
    if fixed_action is not None and fixed_action not in action_ids:
        raise ValueError("fixed action escaped candidates")
    triggers = (
        [item for item in case.candidates.triggers if item.candidate_id == fixed_trigger]
        if fixed_trigger is not None else list(case.candidates.triggers)
    )
    actions = (
        [item for item in case.candidates.actions if item.candidate_id == fixed_action]
        if fixed_action is not None else list(case.candidates.actions)
    )
    target_paths = sorted({field.path for item in actions for field in item.contract.inputs})
    trigger_paths = sorted({field.path for item in triggers for field in item.contract.outputs})
    allowed_action_ids = [fixed_action] if fixed_action is not None else action_ids
    context_paths = sorted(
        field.path for field in case.context_fields
        if fixed_action is None or field.path.startswith(f"config.{fixed_action}.")
    )

    def endpoint(role: str, choices: Sequence[str], fixed: str | None) -> dict[str, Any]:
        candidate: dict[str, Any] = {"type": "string"}
        if fixed is None: candidate["enum"] = list(choices)
        else: candidate["const"] = fixed
        return {
            "type": "object", "additionalProperties": False,
            "properties": {
                "candidate_id": candidate,
                "role": {"type": "string", "const": role, "default": role},
            },
            "required": ["candidate_id"],
        }

    sources = []
    if trigger_paths:
        sources.append({
            "type": "object", "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "const": "trigger_output"},
                "path": {"type": "string", "enum": trigger_paths},
            }, "required": ["kind", "path"],
        })
    if context_paths:
        sources.append({
            "type": "object", "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "const": "context"},
                "path": {"type": "string", "enum": context_paths},
            }, "required": ["kind", "path"],
        })
    bindings: dict[str, Any] = {"type": "array"}
    if not target_paths or not sources:
        bindings["maxItems"] = 0
    else:
        bindings["items"] = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "target_path": {"type": "string", "enum": target_paths},
                "source": {"oneOf": sources},
            }, "required": ["target_path", "source"],
        }
    program = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "const": "farm.applet/v1", "default": "farm.applet/v1"},
            "trigger": endpoint("trigger", trigger_ids, fixed_trigger),
            "action": endpoint("action", allowed_action_ids, fixed_action),
            "bindings": bindings,
        }, "required": ["trigger", "action", "bindings"],
    }
    if not binder:
        return {
            "type": "object", "additionalProperties": False,
            "properties": {"program": program}, "required": ["program"],
        }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "approved": {"type": "boolean"},
            "program": {"anyOf": [program, {"type": "null"}]},
            "rejection_code": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        },
        "required": ["approved", "program", "rejection_code"],
    }


class SyntheticToolClient:
    """Offline typed port used only by --synthetic and unit tests."""
    def __init__(self, model: str, journal_path: Path):
        self.model, self.journal_path, self._lock = model, journal_path, threading.Lock()

    def close(self) -> None: return None

    def _journal(self, value: Mapping[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())

    @staticmethod
    def _choose(query: str, candidates: Sequence[Mapping[str, Any]], *, allow_defer: bool) -> str:
        terms = set(_WORDS.findall(query.casefold()))
        scored = []
        for item in candidates:
            evidence = " ".join(str(item.get(k, "")) for k in ("service_name", "function_name", "description"))
            scored.append((len(terms & set(_WORDS.findall(evidence.casefold()))), str(item["candidate_id"])))
        score, alias = max(scored, key=lambda item: (item[0], item[1]))
        return alias if score or not allow_defer else "DEFER_TO_RETRIEVER"

    @staticmethod
    def _program(request: Mapping[str, Any], trigger: str | None = None, action: str | None = None) -> dict[str, Any]:
        trigger = trigger or SyntheticToolClient._choose(str(request["query"]), request["trigger_candidates"], allow_defer=False)
        action = action or SyntheticToolClient._choose(str(request["query"]), request["action_candidates"], allow_defer=False)
        trigger_row = next(x for x in request["trigger_candidates"] if x["candidate_id"] == trigger)
        action_row = next(x for x in request["action_candidates"] if x["candidate_id"] == action)
        outputs = {(x["path"], x["value_type"]) for x in trigger_row["outputs"]}
        context_paths = {x["path"] for x in request["available_context_fields"]}
        bindings = []
        for target in action_row["inputs"]:
            if not target["required"]: continue
            if (target["path"], target["value_type"]) in outputs:
                source = {"kind": "trigger_output", "path": target["path"]}
            else:
                path = f"config.{action}.{target['path']}"
                if path not in context_paths: raise ValueError("synthetic context is incomplete")
                source = {"kind": "context", "path": path}
            bindings.append({"target_path": target["path"], "source": source})
        return {"version": "farm.applet/v1", "trigger": {"candidate_id": trigger, "role": "trigger"}, "action": {"candidate_id": action, "role": "action"}, "bindings": bindings}

    def call_tool(self, *, semantic_id: str, role: str, public_request: Mapping[str, Any], tool_name: str, description: str, output_model: type[BaseModel], parameter_schema: Mapping[str, Any] | None = None) -> ToolCall:
        assert_model_safe(public_request)
        self._journal({"event": "request_started", "semantic_id": semantic_id, "role": role, "request_hash": _hash(public_request), "attempt": 0})
        if tool_name == "submit_selection":
            raw = {"trigger_choice": self._choose(str(public_request["query"]), public_request["trigger_candidates"], allow_defer=True), "action_choice": self._choose(str(public_request["query"]), public_request["action_candidates"], allow_defer=True)}
        elif tool_name in {"submit_trigger", "submit_action"}:
            side = "trigger" if tool_name == "submit_trigger" else "action"
            raw = {"choice": self._choose(str(public_request["query"]), public_request[f"{side}_candidates"], allow_defer=True), "confidence": 0.9}
        elif tool_name in {
            "submit_trigger_schema_m5",
            "submit_trigger_fused_m10",
        }:
            raw = {
                "choice": self._choose(
                    str(public_request["query"]),
                    public_request["trigger_candidates"],
                    allow_defer=True,
                )
            }
        elif tool_name in {"submit_plan", "repair_plan"}:
            raw = {"program": self._program(public_request)}
        elif tool_name == "submit_bound_plan":
            raw = {"approved": True, "program": self._program(public_request, str(public_request["trigger_proposal"]), str(public_request["action_proposal"])), "rejection_code": None}
        else: raise ValueError("unknown synthetic terminal tool")
        parsed = output_model.model_validate_json(json.dumps(raw), strict=True)
        self._journal({"event": "request_finished", "semantic_id": semantic_id, "role": role, "request_hash": _hash(public_request), "attempt": 0, "http_status": None, "response_hash": _hash(raw), "outcome": "valid_tool_call", "model_tool_calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": 0.0})
        return ToolCall(role, True, parsed, 0, 1, 0, 0, 0, 0.0, None, True)


@dataclass
class Ledger:
    calls: list[ToolCall] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, call: ToolCall) -> ToolCall:
        with self.lock: self.calls.append(call)
        return call


def _usage(call: ToolCall, model: str) -> PortUsage:
    return PortUsage(provider_requests=call.provider_requests, provider_model=model if call.provider_requests else None, prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens, latency_ms=call.latency_seconds * 1000)


class Ports:
    def __init__(self, case: PreparedCase, client: Any, model: str, ledger: Ledger):
        self.case, self.client, self.model, self.ledger = case, client, model, ledger
        self.trigger_proposal: str | None = None
        self.action_proposal: str | None = None
        self.consensus_schema_proposal: str | None = None
        self.consensus_fused_proposal: str | None = None

    def _call(self, role: str, request: Mapping[str, Any], tool: str, output: type[BaseModel], schema: Mapping[str, Any] | None = None) -> ToolCall:
        call = self.client.call_tool(semantic_id=f"{self.case.request_id}:{role}", role=role, public_request=request, tool_name=tool, description=f"Submit the typed {role} terminal result.", output_model=output, parameter_schema=schema)
        self.ledger.add(call)
        if not call.ok or call.value is None: raise RuntimeError(call.error_code or "model_call_failed")
        return call

    def planner(self, _request: Any) -> PlannerResponse:
        public = public_case_payload(self.case) | {"instruction": "Select one trigger and action, then bind every required action input only to an exact typed trigger output or supplied context field."}
        call = self._call("planner", public, "submit_plan", PlanSubmission, plan_tool_schema(self.case))
        assert isinstance(call.value, PlanSubmission)
        return PlannerResponse(program=call.value.program, usage=_usage(call, self.model))

    def repairer(self, request: Any) -> PlannerResponse:
        public = public_case_payload(self.case) | {
            "instruction": "Repair once using only these deterministic violation codes.",
            "failed_stage": request.failed_stage,
            "failed_program": request.failed_program.model_dump(mode="json") if request.failed_program else None,
            "violations": [{"code": x.code, "location": x.location, "expected": x.expected, "observed": x.observed} for x in request.issues],
        }
        call = self._call("repair", public, "repair_plan", PlanSubmission, plan_tool_schema(self.case))
        assert isinstance(call.value, PlanSubmission)
        return PlannerResponse(program=call.value.program, usage=_usage(call, self.model))

    def _specialist(self, side: Literal["trigger", "action"], _request: Any) -> SpecialistResponse:
        base = public_case_payload(self.case)
        public = {"query": base["query"], f"{side}_candidates": base[f"{side}_candidates"], "instruction": f"Choose only the {side}. DEFER_TO_RETRIEVER is cascade abstention when evidence is insufficient."}
        aliases = [x["candidate_id"] for x in public[f"{side}_candidates"]]
        call = self._call(f"{side}_specialist", public, f"submit_{side}", SpecialistSelection, _tool_schema(SpecialistSelection, {"choice": ["DEFER_TO_RETRIEVER", *aliases]}))
        assert isinstance(call.value, SpecialistSelection)
        choice = (self.case.baseline_trigger_alias if side == "trigger" else self.case.baseline_action_alias) if call.value.choice == "DEFER_TO_RETRIEVER" else call.value.choice
        allowed = {x.candidate_id for x in getattr(self.case.candidates, f"{side}s")}
        if choice not in allowed: raise ValueError("specialist escaped candidates")
        if side == "trigger": self.trigger_proposal = choice
        else: self.action_proposal = choice
        return SpecialistResponse(proposal=EndpointProposal(side=side, candidate_id=choice, confidence=call.value.confidence), usage=_usage(call, self.model))

    def trigger_specialist(self, request: Any) -> SpecialistResponse: return self._specialist("trigger", request)
    def action_specialist(self, request: Any) -> SpecialistResponse: return self._specialist("action", request)

    def _trigger_consensus_view(
        self,
        name: Literal["schema_m5", "fused_m10"],
        request: Any,
    ) -> TriggerConsensusResponse:
        view = self.case.consensus_views[name]
        if request.view != name:
            raise ValueError("consensus port received the wrong view")
        if set(request.candidate_ids) != set(view.primary_candidate_ids):
            raise ValueError("consensus port candidate boundary changed")
        public = {
            "query": _sanitize(self.case.query, limit=16000),
            "trigger_candidates": list(view.public_candidates),
            "instruction": (
                "Choose the single trigger function that best matches the request. "
                "Use DEFER_TO_RETRIEVER when the evidence does not justify an override."
            ),
        }
        assert_model_safe(public)
        aliases = list(view.alias_to_primary)
        role = f"trigger_{name}"
        call = self._call(
            role,
            public,
            f"submit_{role}",
            ConsensusSelection,
            _tool_schema(
                ConsensusSelection,
                {"choice": ["DEFER_TO_RETRIEVER", *aliases]},
            ),
        )
        assert isinstance(call.value, ConsensusSelection)
        if call.value.choice == "DEFER_TO_RETRIEVER":
            return TriggerConsensusResponse(
                deferred=True,
                candidate_id=None,
                usage=_usage(call, self.model),
            )
        primary = view.alias_to_primary.get(call.value.choice)
        if primary is None:
            raise ValueError("consensus choice escaped its dynamic aliases")
        if name == "schema_m5":
            self.consensus_schema_proposal = primary
        else:
            self.consensus_fused_proposal = primary
        return TriggerConsensusResponse(
            deferred=False,
            candidate_id=primary,
            usage=_usage(call, self.model),
        )

    def trigger_schema_m5(self, request: Any) -> TriggerConsensusResponse:
        return self._trigger_consensus_view("schema_m5", request)

    def trigger_fused_m10(self, request: Any) -> TriggerConsensusResponse:
        return self._trigger_consensus_view("fused_m10", request)

    def binder(self, request: Any) -> BinderCriticResponse:
        public = public_case_payload(self.case) | {"trigger_proposal": request.trigger.candidate_id, "action_proposal": request.action.candidate_id, "instruction": "Critique the two specialist choices, then construct a complete typed program or reject it."}
        call = self._call("binder_critic", public, "submit_bound_plan", BinderSubmission, plan_tool_schema(self.case, fixed_trigger=request.trigger.candidate_id, fixed_action=request.action.candidate_id, binder=True))
        assert isinstance(call.value, BinderSubmission)
        return BinderCriticResponse(approved=call.value.approved, program=call.value.program, rejection_code=call.value.rejection_code, usage=_usage(call, self.model))


@dataclass(frozen=True)
class CaseOutcome:
    terminal_status: str
    final_trigger_alias: str
    final_action_alias: str
    proposal_trigger_alias: str | None
    proposal_action_alias: str | None
    program: AppletProgram | None
    attempted_programs: tuple[AppletProgram, ...]
    issues: tuple[ValidationIssue, ...]
    calls: tuple[ToolCall, ...]
    semantic_calls: int
    compilation_attempts: int
    execution_attempts: int
    repairs_attempted: int
    repair_success: bool
    compiled: bool
    sandbox_run: bool
    receipt: str | None
    deterministic_tools: int
    policy_trace: Mapping[str, Any] | None = None


def _issue(code: str, location: str = "model") -> ValidationIssue:
    return ValidationIssue(code=code, location=location, message="typed experiment stage failed")


def run_case(arm: str, case: PreparedCase, client: Any, model: str) -> CaseOutcome:
    ledger = Ledger(); ports = Ports(case, client, model, ledger)
    if arm == "typed_direct":
        public = public_case_payload(case) | {"instruction": "Choose trigger and action independently. DEFER_TO_RETRIEVER is cascade abstention when evidence is insufficient."}
        aliases_t = [x.candidate_id for x in case.candidates.triggers]; aliases_a = [x.candidate_id for x in case.candidates.actions]
        call = client.call_tool(semantic_id=f"{case.request_id}:direct", role="joint_endpoint_selector", public_request=public, tool_name="submit_selection", description="Submit one trigger and action selection.", output_model=DirectSelection, parameter_schema=_tool_schema(DirectSelection, {"trigger_choice": ["DEFER_TO_RETRIEVER", *aliases_t], "action_choice": ["DEFER_TO_RETRIEVER", *aliases_a]}))
        ledger.add(call); issues: list[ValidationIssue] = []; program = None; compiled = run = False; receipt = None; comp_n = exec_n = 0
        trigger = case.baseline_trigger_alias; action = case.baseline_action_alias
        if call.ok and isinstance(call.value, DirectSelection):
            trigger = case.baseline_trigger_alias if call.value.trigger_choice == "DEFER_TO_RETRIEVER" else call.value.trigger_choice
            action = case.baseline_action_alias if call.value.action_choice == "DEFER_TO_RETRIEVER" else call.value.action_choice
            if trigger not in aliases_t or action not in aliases_a: issues.append(_issue("selection_outside_candidates"))
            else:
                try:
                    program = deterministic_autowire(case, trigger, action); comp_n = 1
                    compiled_program = AppletCompiler(case.candidates, context_fields=case.context_fields).compile(program); compiled = True; exec_n = 1
                    execution = SandboxConnectorRegistry(case.candidates).execute(compiled_program, context=case.context); run = True; receipt = execution.invocation.receipt_sha256
                except (CompilationError, SandboxExecutionError) as error: issues.extend(error.issues)
                except Exception: issues.append(_issue("direct_execution_failed", "deterministic_harness"))
        else: issues.append(_issue(call.error_code or "direct_call_failed"))
        if not run: trigger, action = case.baseline_trigger_alias, case.baseline_action_alias
        return CaseOutcome("executed" if run else "safe_fallback", trigger, action, (program.trigger.candidate_id if program else None), (program.action.candidate_id if program else None), program if run else None, (program,) if program else (), tuple(issues), tuple(ledger.calls), 1, comp_n, exec_n, 0, False, compiled, run, receipt, 1 + comp_n + exec_n)
    ports = Ports(case, client, model, ledger)
    if arm in {"typed_schema_plan", "typed_execute_repair"}:
        module = BoundedDSPyApplet(ports.planner, repairer=ports.repairer if arm == "typed_execute_repair" else None, max_repairs=1 if arm == "typed_execute_repair" else 0)
    elif arm == "dspy_factorized":
        module = FactorizedDSPyApplet(ports.trigger_specialist, ports.action_specialist, ports.binder, parallel_specialists=True)
    elif arm == "trigger_consensus_executable":
        if case.consensus_routing_score is None:
            raise ValueError("consensus arm requires a frozen routing score")
        module = TriggerConsensusDSPyApplet(
            ports.trigger_schema_m5,
            ports.trigger_fused_m10,
            routing_threshold=CONSENSUS_ROUTING_THRESHOLD,
        )
    else: raise ValueError("unknown arm")
    module_inputs: dict[str, Any] = {
        "case": {"request_id": case.request_id, "query": case.query},
        "candidates": case.candidates,
        "context_fields": case.context_fields,
        "context": case.context,
        "baseline_trigger_id": case.baseline_trigger_alias,
        "baseline_action_id": case.baseline_action_alias,
    }
    if arm == "trigger_consensus_executable":
        module_inputs.update(
            {
                "schema_m5_candidate_ids": case.consensus_views[
                    "schema_m5"
                ].primary_candidate_ids,
                "fused_m10_candidate_ids": case.consensus_views[
                    "fused_m10"
                ].primary_candidate_ids,
                "trigger_seed_support": case.trigger_seed_support,
                "routing_score": case.consensus_routing_score,
            }
        )
    result = module(**module_inputs).result
    executed = result.terminal_status == "executed"
    program = result.program if executed else None
    final_t = program.trigger.candidate_id if program else case.baseline_trigger_alias
    final_a = program.action.candidate_id if program else case.baseline_action_alias
    proposal = result.attempted_programs[-1] if result.attempted_programs else None
    if arm == "trigger_consensus_executable":
        proposal_t = ports.consensus_schema_proposal
        proposal_a = case.baseline_action_alias if proposal_t is not None else None
    else:
        proposal_t = proposal.trigger.candidate_id if proposal else ports.trigger_proposal
        proposal_a = proposal.action.candidate_id if proposal else ports.action_proposal
    compiled = result.execution_attempts > 0
    receipt = result.execution.invocation.receipt_sha256 if result.execution else None
    trace = (
        result.trigger_consensus_trace.model_dump(mode="json")
        if result.trigger_consensus_trace is not None
        else None
    )
    return CaseOutcome(result.terminal_status, final_t, final_a, proposal_t, proposal_a, program, result.attempted_programs, result.issues, tuple(ledger.calls), result.call_totals.logical_calls, result.compilation_attempts, result.execution_attempts, result.repairs_attempted, bool(result.repairs_attempted and executed), compiled, executed, receipt, result.compilation_attempts + result.execution_attempts, trace)


def _pair(identity_t: str, identity_a: str, corpora: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, str]:
    if identity_t not in corpora["trigger"] or identity_a not in corpora["action"]: raise ValueError("evaluation endpoint absent from corpus")
    return {"trigger_url": identity_t, "action_url": identity_a, "trigger_service": str(corpora["trigger"][identity_t]["channel"]), "action_service": str(corpora["action"][identity_a]["channel"])}


def _score(prediction: Mapping[str, str], gold: Sequence[Mapping[str, str]]) -> dict[str, bool]:
    fp = {(x["trigger_url"], x["action_url"]) for x in gold}; sp = {(x["trigger_service"], x["action_service"]) for x in gold}
    return {
        "function_trigger": prediction["trigger_url"] in {x[0] for x in fp},
        "function_action": prediction["action_url"] in {x[1] for x in fp},
        "function_joint": (prediction["trigger_url"], prediction["action_url"]) in fp,
        "service_trigger": prediction["trigger_service"] in {x[0] for x in sp},
        "service_action": prediction["action_service"] in {x[1] for x in sp},
        "service_joint": (prediction["trigger_service"], prediction["action_service"]) in sp,
    }


def evaluation_record(row: Mapping[str, Any], case: PreparedCase, outcome: CaseOutcome, corpora: Mapping[str, Mapping[str, Mapping[str, Any]]], arm: str) -> dict[str, Any]:
    gold = [_pair(str(x["trigger_url"]), str(x["action_url"]), corpora) for x in row["valid_pairs"]]
    base = _pair(str(row["trigger_candidates"][0]["url"]), str(row["action_candidates"][0]["url"]), corpora)
    final = _pair(case.alias_to_identity[outcome.final_trigger_alias], case.alias_to_identity[outcome.final_action_alias], corpora)
    proposal = None
    if outcome.proposal_trigger_alias in case.alias_to_identity and outcome.proposal_action_alias in case.alias_to_identity:
        proposal = _pair(case.alias_to_identity[str(outcome.proposal_trigger_alias)], case.alias_to_identity[str(outcome.proposal_action_alias)], corpora)
    trigger_ids = [str(x["url"]) for x in row["trigger_candidates"][:10]]; action_ids = [str(x["url"]) for x in row["action_candidates"][:10]]
    fp = {(x["trigger_url"], x["action_url"]) for x in gold}; sp = {(x["trigger_service"], x["action_service"]) for x in gold}
    ts = [str(corpora["trigger"][x]["channel"]) for x in trigger_ids]; acs = [str(corpora["action"][x]["channel"]) for x in action_ids]
    oracle = {
        "function_trigger": any(t in trigger_ids for t, _ in fp), "function_action": any(a in action_ids for _, a in fp), "function_joint": any(t in trigger_ids and a in action_ids for t, a in fp),
        "service_trigger": any(t in ts for t, _ in sp), "service_action": any(a in acs for _, a in sp), "service_joint": any(t in ts and a in acs for t, a in sp),
    }
    ranks_t = {x: i for i, x in enumerate(trigger_ids, 1)}; ranks_a = {x: i for i, x in enumerate(action_ids, 1)}
    rank = min((max(ranks_t[t], ranks_a[a]) for t, a in fp if t in ranks_t and a in ranks_a), default=None)
    bucket = "rank1" if rank == 1 else "rank2_5" if rank is not None and rank <= 5 else "rank6_10" if rank is not None and rank <= 10 else "outside10"
    calls = [{"role": x.role, "ok": x.ok, "schema_valid": x.schema_valid, "transport_attempts": x.transport_attempts, "model_tool_calls": x.model_tool_calls, "provider_requests": x.provider_requests, "prompt_tokens": x.prompt_tokens, "completion_tokens": x.completion_tokens, "latency_seconds": x.latency_seconds, "error_code": x.error_code} for x in outcome.calls]
    prompt = sum(x.prompt_tokens for x in outcome.calls); completion = sum(x.completion_tokens for x in outcome.calls)
    account = {"semantic_calls": outcome.semantic_calls, "transport_attempts": sum(x.transport_attempts for x in outcome.calls), "model_tool_calls": sum(x.model_tool_calls for x in outcome.calls), "provider_requests": sum(x.provider_requests for x in outcome.calls), "deterministic_tool_calls": outcome.deterministic_tools, "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, "latency_seconds": sum(x.latency_seconds for x in outcome.calls), "failures": sum(not x.ok for x in outcome.calls)}
    if arm == "trigger_consensus_executable":
        trace = outcome.policy_trace or {}
        expected = int(bool(trace.get("schema_m5_called"))) + int(
            bool(trace.get("fused_m10_called"))
        )
        protocol_valid = (
            len(outcome.calls) == expected
            and expected <= 2
            and all(x.ok and x.schema_valid for x in outcome.calls)
        )
    else:
        expected = 3 if arm == "dspy_factorized" else 1
        protocol_valid = len(outcome.calls) >= expected and len(outcome.calls) <= (2 if arm == "typed_execute_repair" else expected) and all(x.ok and x.schema_valid for x in outcome.calls)
    policy_trace = dict(outcome.policy_trace) if outcome.policy_trace else None
    if policy_trace is not None:
        policy_trace["view_presentations"] = {
            name: {
                "candidate_count": len(view.public_candidates),
                "presentation_sha256": view.presentation_sha256,
            }
            for name, view in case.consensus_views.items()
        }
    return {
        "schema_version": "farm_round8_executable_record_v1", "group_id": row["group_id"], "semantic_family_id": row["semantic_family_id"], "query": row["query"], "arm_id": arm,
        "baseline_pair": base, "proposal_pair": proposal, "final_pair": final, "valid_pairs": gold,
        "baseline_score": _score(base, gold), "proposal_score": _score(proposal, gold) if proposal else None, "final_score": _score(final, gold), "candidate_oracle": oracle, "rank_bucket": bucket, "gold_joint_rank": rank,
        "side_edit": {"trigger": final["trigger_url"] != base["trigger_url"], "action": final["action_url"] != base["action_url"]},
        "retained_baseline": final == base, "fallback_used": outcome.terminal_status != "executed", "terminal_status": outcome.terminal_status,
        "program": outcome.program.model_dump(mode="json") if outcome.program else None,
        "attempted_programs": [x.model_dump(mode="json") for x in outcome.attempted_programs],
        "issues": [x.model_dump(mode="json") for x in outcome.issues], "calls": calls, "accounting": account, "protocol_valid": protocol_valid,
        "policy_trace": policy_trace,
        "execution": {"strict_parse": bool(outcome.calls) and all(x.ok and x.schema_valid for x in outcome.calls), "compilation_attempts": outcome.compilation_attempts, "compiled": outcome.compiled, "sandbox_attempts": outcome.execution_attempts, "sandbox_run": outcome.sandbox_run, "repair_attempted": bool(outcome.repairs_attempted), "repair_success": outcome.repair_success, "receipt_sha256": outcome.receipt, "binding_claim_allowed": False},
    }


def exact_mcnemar(before: Sequence[int], after: Sequence[int]) -> dict[str, Any]:
    rescues = sum(x == 0 and y == 1 for x, y in zip(before, after)); regressions = sum(x == 1 and y == 0 for x, y in zip(before, after)); discordant = rescues + regressions
    tail = 1.0 if not discordant else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(rescues, regressions) + 1)) / 2**discordant)
    return {"rescues": rescues, "regressions": regressions, "discordant": discordant, "net": rescues - regressions, "exact_two_sided_p": tail}


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records: raise ValueError("cannot summarize empty records")
    result: dict[str, Any] = {"rows": len(records)}
    for name in OUTCOMES:
        before = [int(x["baseline_score"][name]) for x in records]; after = [int(x["final_score"][name]) for x in records]
        result[name] = {"baseline_hits": sum(before), "baseline_rate": sum(before) / len(records), "agent_hits": sum(after), "agent_rate": sum(after) / len(records), "absolute_delta": (sum(after) - sum(before)) / len(records), **exact_mcnemar(before, after)}
    result["candidate_oracle"] = {name: {"hits": sum(bool(x["candidate_oracle"][name]) for x in records), "rate": sum(bool(x["candidate_oracle"][name]) for x in records) / len(records)} for name in OUTCOMES}
    result["oracle_conditioned_selection"] = {}
    for name in OUTCOMES:
        eligible = [x for x in records if x["candidate_oracle"][name]]; proposals = [x for x in eligible if isinstance(x.get("proposal_score"), Mapping)]
        result["oracle_conditioned_selection"][name] = {"eligible": len(eligible), "proposal_available": len(proposals), "proposal_hits": sum(bool(x["proposal_score"][name]) for x in proposals), "proposal_accuracy": sum(bool(x["proposal_score"][name]) for x in proposals) / len(proposals) if proposals else None, "final_hits": sum(bool(x["final_score"][name]) for x in eligible), "final_accuracy": sum(bool(x["final_score"][name]) for x in eligible) / len(eligible) if eligible else None}
    changed = [x for x in records if not x["retained_baseline"]]; rescues = result["function_joint"]["rescues"]; regressions = result["function_joint"]["regressions"]
    result["safety"] = {"top1_rescues": rescues, "top1_regressions": regressions, "changed_cases": len(changed), "override_precision_all_changes": sum(bool(x["final_score"]["function_joint"]) for x in changed) / len(changed) if changed else None, "discordant_rescue_precision": rescues / (rescues + regressions) if rescues + regressions else None, "baseline_correct_retention": 1 - regressions / result["function_joint"]["baseline_hits"] if result["function_joint"]["baseline_hits"] else None, "fallback_rate": sum(bool(x["fallback_used"]) for x in records) / len(records)}
    result["execution"] = {"strict_parse_rate": sum(bool(x["execution"]["strict_parse"]) for x in records) / len(records), "compile_rate": sum(bool(x["execution"]["compiled"]) for x in records) / len(records), "sandbox_run_rate": sum(bool(x["execution"]["sandbox_run"]) for x in records) / len(records), "repair_attempt_rate": sum(bool(x["execution"]["repair_attempted"]) for x in records) / len(records), "repair_success_rate": sum(bool(x["execution"]["repair_success"]) for x in records) / max(1, sum(bool(x["execution"]["repair_attempted"]) for x in records)), "binding_claim_allowed": False}
    cost_keys = ("semantic_calls", "transport_attempts", "model_tool_calls", "provider_requests", "deterministic_tool_calls", "prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds", "failures")
    result["cost"] = {key: sum(float(x["accounting"][key]) for x in records) for key in cost_keys}; result["cost"]["semantic_calls_per_case"] = result["cost"]["semantic_calls"] / len(records); result["cost"]["tokens_per_case"] = result["cost"]["total_tokens"] / len(records)
    result["rank_bucket_selection"] = {bucket: {"rows": len(rows), "proposal_hits": sum(bool(x.get("proposal_score") and x["proposal_score"]["function_joint"]) for x in rows), "final_hits": sum(bool(x["final_score"]["function_joint"]) for x in rows)} for bucket in ("rank1", "rank2_5", "rank6_10", "outside10") for rows in [[x for x in records if x["rank_bucket"] == bucket]]}
    result["failures"] = dict(Counter(issue["code"] for x in records for issue in x["issues"]))
    result["per_service"] = {}
    for side in ("trigger", "action"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in records: groups[str(row["baseline_pair"][f"{side}_service"])].append(row)
        result["per_service"][side] = {service: {"cases": len(rows), "function_joint_hits": sum(bool(x["final_score"]["function_joint"]) for x in rows), "function_joint_rate": sum(bool(x["final_score"]["function_joint"]) for x in rows) / len(rows), "semantic_calls": sum(int(x["accounting"]["semantic_calls"]) for x in rows), "total_tokens": sum(int(x["accounting"]["total_tokens"]) for x in rows), "sandbox_runs": sum(bool(x["execution"]["sandbox_run"]) for x in rows)} for service, rows in sorted(groups.items())}
    traced = [
        row["policy_trace"]
        for row in records
        if isinstance(row.get("policy_trace"), Mapping)
    ]
    if traced:
        result["trigger_consensus"] = {
            "routing_threshold": CONSENSUS_ROUTING_THRESHOLD,
            "routed_cases": sum(bool(x["routed"]) for x in traced),
            "schema_m5_calls": sum(bool(x["schema_m5_called"]) for x in traced),
            "fused_m10_calls": sum(bool(x["fused_m10_called"]) for x in traced),
            "exact_consensus_cases": sum(
                bool(x["exact_trigger_consensus"]) for x in traced
            ),
            "executed_consensus_cases": sum(
                row["terminal_status"] == "executed" for row in records
            ),
            "baseline_action_violations": sum(
                bool(row["side_edit"]["action"]) for row in records
            ),
            "decision_reasons": dict(
                Counter(str(x["decision_reason"]) for x in traced)
            ),
        }
    return result


def synthetic_metric_record(*, before: bool, after: bool, oracle: bool, service: str) -> dict[str, Any]:
    scores_before = {x: before for x in OUTCOMES}; scores_after = {x: after for x in OUTCOMES}
    return {"group_id": _hash((before, after, oracle, service)), "semantic_family_id": service, "baseline_pair": {"trigger_service": service, "action_service": service}, "baseline_score": scores_before, "proposal_score": scores_after, "final_score": scores_after, "candidate_oracle": {x: oracle for x in OUTCOMES}, "rank_bucket": "rank2_5", "retained_baseline": before == after, "fallback_used": False, "execution": {"strict_parse": True, "compiled": True, "sandbox_run": True, "repair_attempted": False, "repair_success": False}, "accounting": {"semantic_calls": 1, "transport_attempts": 1, "model_tool_calls": 1, "provider_requests": 1, "deterministic_tool_calls": 2, "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "latency_seconds": 0.1, "failures": 0}, "issues": []}


def _corpus_map(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows: raise ValueError("frozen corpus is empty")
    result = {}
    for row in rows:
        identity = row.get("url") if isinstance(row, Mapping) else None
        if not isinstance(identity, str) or not identity or identity in result: raise ValueError("invalid corpus identity")
        result[identity] = dict(row)
    return result


def load_consensus_router(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Load the unchanged hash-pinned Round3/Round6 router weights."""

    sources = matrix["sources"]
    path = Path(sources["recoverability_router"]).resolve()
    if sha256_file(path) != sources["recoverability_router_sha256"]:
        raise ValueError("matrix-pinned recoverability router changed")
    router = validate_router(
        read_json(path),
        candidate_sha256=str(sources["router_candidate_sha256"]),
    )
    policy = matrix.get("trigger_consensus_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("matrix lacks trigger_consensus_policy")
    if router.get("dataset_id") != DATASET_ID:
        raise ValueError("recoverability router dataset binding changed")
    if abs(float(router["threshold"]) - float(policy["source_threshold"])) > 1e-15:
        raise ValueError("source router threshold changed")
    if abs(float(policy["routing_threshold"]) - CONSENSUS_ROUTING_THRESHOLD) > 1e-15:
        raise ValueError("consensus routing threshold changed")
    return router


def load_real_inputs(matrix: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    sources = matrix["sources"]
    discovery_path = Path(sources["discovery_screen"]).resolve()
    if sha256_file(discovery_path) != sources["discovery_screen_sha256"]:
        raise ValueError("matrix-pinned discovery screen changed")
    data_root = Path(sources["data_root"]).resolve(); data_manifest_path = data_root / "manifest.json"
    if sha256_file(data_manifest_path) != sources["data_manifest_sha256"]: raise ValueError("data manifest changed")
    data_manifest = read_json(data_manifest_path); corpora = {}
    for side, relative in (("trigger", "corpus/triggers.json"), ("action", "corpus/actions.json")):
        path = data_root / relative
        if sha256_file(path) != data_manifest["artifacts"][relative]["sha256"]: raise ValueError(f"{side} corpus changed")
        corpora[side] = _corpus_map(read_json(path))
    # This JSONL is the already-consumed Round6 discovery screen.  No active
    # source points at the 1,145-row dev files or the reserved confirmation.
    selected, selection = select_discovery_rows(load_jsonl(discovery_path))
    return selected, corpora, selection


def synthetic_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    corpora: dict[str, dict[str, dict[str, Any]]] = {"trigger": {}, "action": {}}
    rows = []
    concepts = (("mail", "chat", "message"), ("calendar", "notes", "event"), ("weather", "alerts", "forecast"))
    for case_index, (trigger_service, action_service, noun) in enumerate(concepts):
        triggers = []; actions = []
        for index in range(10):
            trigger_id = f"trigger://synthetic-{case_index}-{index}"; action_id = f"action://synthetic-{case_index}-{index}"
            trigger = {"url": trigger_id, "kind": "trigger", "channel": trigger_service if index == 0 else f"other-trigger-{case_index}-{index}", "channel_display": trigger_service if index == 0 else f"Other trigger {index}", "function_name": f"New {noun}" if index == 0 else f"Unrelated trigger {index}", "description": f"Fires when a new {noun} arrives" if index == 0 else "Unrelated synthetic event", "input_fields": [], "ingredients": [{"slug": "body", "type": "String", "required": True}]}
            action = {"url": action_id, "kind": "action", "channel": action_service if index == 0 else f"other-action-{case_index}-{index}", "channel_display": action_service if index == 0 else f"Other action {index}", "function_name": f"Post {noun}" if index == 0 else f"Unrelated action {index}", "description": f"Posts the {noun}" if index == 0 else "Unrelated synthetic operation", "ingredients": [], "input_fields": [{"slug": "body", "type": "String", "required": True}, {"slug": "destination", "type": "String", "required": True}]}
            corpora["trigger"][trigger_id] = trigger; corpora["action"][action_id] = action
            triggers.append({
                "url": trigger_id,
                "retrieval_rank": index + 1,
                "retrieval_score": 1.0 / (index + 1),
                "channel": trigger["channel"],
                "seed_ranks": {
                    "seed42": index + 1,
                    "seed1337": index + 1,
                    "seed2025": index + 1,
                },
                "text_plain": trigger["description"],
                "text_schema": trigger["description"],
            }); actions.append({
                "url": action_id,
                "retrieval_rank": index + 1,
                "retrieval_score": 1.0 / (index + 1),
                "channel": action["channel"],
                "seed_ranks": {
                    "seed42": index + 1,
                    "seed1337": index + 1,
                    "seed2025": index + 1,
                },
                "text_plain": action["description"],
                "text_schema": action["description"],
            })
        rows.append({"group_id": f"synthetic-{case_index}", "semantic_family_id": f"synthetic-family-{case_index}", "query": f"When a new {noun} arrives in {trigger_service}, post the {noun} to {action_service}", "trigger_candidates": triggers, "action_candidates": actions, "valid_pairs": [{"trigger_url": triggers[0]["url"], "action_url": actions[0]["url"]}]})
    return rows, corpora, {"source_rows": len(rows), "raw_window": None, "selected_rows": len(rows), "selected_group_ids_sha256": ids_hash(rows), "reserved_payload_sent_to_model": False}


def load_environment(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1); value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'": value = value[1:-1]
        values[key.strip()] = value
    return values


def _api_endpoint(host: str, suffix: str) -> str:
    base = host.rstrip("/")
    return base + (f"/{suffix}" if base.endswith("/api") else f"/api/{suffix}")


def verify_model_digest(host: str, key: str | None, model: str, digest: str, *, exact: bool = False) -> None:
    import httpx
    if key is None:
        host = validate_local_ollama_host(host)
    headers = {"Authorization": "Bearer " + key} if key is not None else {}
    with httpx.Client(timeout=240, trust_env=key is not None) as client:
        response = client.get(_api_endpoint(host, "tags"), headers=headers)
        response.raise_for_status(); payload = response.json()
    record = next((x for x in payload.get("models", []) if (x.get("model") or x.get("name")) == model), None)
    observed = record.get("digest") if isinstance(record, Mapping) else None
    matches = observed == digest if exact else isinstance(observed, str) and (observed == digest or observed.startswith(digest))
    if not isinstance(observed, str) or not matches: raise RuntimeError("Ollama model digest unavailable or changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--env-file", type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int); parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--local-ollama-host")
    parser.add_argument("--model")
    parser.add_argument("--expected-digest")
    args = parser.parse_args(argv)
    if args.smoke is not None and args.smoke < 1: parser.error("--smoke must be positive")
    local = args.local_ollama_host is not None
    if args.synthetic and local: parser.error("--synthetic and --local-ollama-host are mutually exclusive")
    if local:
        try: args.local_ollama_host = validate_local_ollama_host(args.local_ollama_host)
        except ValueError as error: parser.error(str(error))
        if not args.model or not args.expected_digest: parser.error("local Ollama requires --model and --expected-digest")
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_digest): parser.error("--expected-digest must be the exact 64-character lowercase SHA-256 digest")
    elif args.model is not None or args.expected_digest is not None:
        parser.error("--model and --expected-digest are local-Ollama-only controls")
    if not args.synthetic and not local and args.env_file is None: parser.error("--env-file is required outside --synthetic/local Ollama")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    script_root = Path(__file__).resolve().parent
    matrix_path = run_root / "EXPERIMENT_MATRIX.json"
    matrix = read_json(matrix_path)
    if matrix.get("run_id") != RUN_ID or matrix.get("dataset_id") != DATASET_ID:
        raise ValueError("run root is not bound to Round8 matrix")
    arm_spec = next((x for x in matrix.get("arms", []) if x.get("id") == args.arm), None)
    if arm_spec is None:
        raise ValueError("arm absent from matrix")

    local = args.local_ollama_host is not None
    if args.synthetic:
        selected, corpora, selection = synthetic_inputs()
        key = ""
        host: str | None = None
        model_name = str(matrix["model"]["name"])
        model_digest = str(matrix["model"]["digest"])
    elif local:
        selected, corpora, selection = load_real_inputs(matrix)
        key = ""
        host = str(args.local_ollama_host)
        model_name = str(args.model)
        model_digest = str(args.expected_digest)
    else:
        selected, corpora, selection = load_real_inputs(matrix)
        environment = load_environment(args.env_file.resolve())
        model_spec = matrix["model"]
        key = os.environ.get(model_spec["key_env"]) or environment.get(model_spec["key_env"], "")
        host = os.environ.get(model_spec["host_env"]) or environment.get(model_spec["host_env"], "")
        if not key or not host:
            raise RuntimeError("Ollama environment slots are unconfigured")
        model_name = str(model_spec["name"])
        model_digest = str(model_spec["digest"])

    consensus_router: Mapping[str, Any] | None = None
    if args.arm == "trigger_consensus_executable" and not args.synthetic:
        consensus_router = load_consensus_router(matrix)

    if args.smoke is not None:
        if args.arm == "trigger_consensus_executable" and args.synthetic:
            selected = selected[:args.smoke]
        else:
            selected = select_smoke_rows(
                selected,
                arm=args.arm,
                count=args.smoke,
                consensus_router=consensus_router,
            )
    output = output_root_for_mode(
        run_root,
        synthetic=args.synthetic,
        local=local,
        smoke=args.smoke,
        model_digest=model_digest if local else None,
    )
    records_path = output / "records" / f"{args.arm}.jsonl"
    attempts_path = output / "attempts" / f"{args.arm}.jsonl"
    progress_path = output / "progress" / f"{args.arm}.json"
    result_path = output / "results" / f"{args.arm}.json"

    mode = "synthetic" if args.synthetic else "local_granite_smoke" if local and args.smoke else "local_granite" if local else "smoke" if args.smoke else "family_purged_agent_discovery"
    transport = (
        {"transport": "synthetic_offline", "data_left_dgx": False}
        if args.synthetic
        else transport_metadata(local=local)
    )
    binding = {
        "run_id": RUN_ID,
        "mode": mode,
        "arm": args.arm,
        "dataset_id": DATASET_ID,
        "matrix_sha256": sha256_file(matrix_path),
        "runner_sha256": sha256_file(script_root / "run_round8.py"),
        "dspy_executable_agent_sha256": sha256_file(script_root / "dspy_executable_agent.py"),
        "executable_applet_sha256": sha256_file(script_root / "executable_applet.py"),
        "model": model_name,
        "model_digest": model_digest,
        "data_left_dgx": transport["data_left_dgx"],
        "selection": selection | {
            "executed_rows": len(selected),
            "executed_group_ids_sha256": ids_hash(selected),
        },
        "protocol": {
            "native_api": "/api/chat",
            **transport,
            "structured_format": False,
            "terminal_tool_calls": True,
            "transport_attempts_per_semantic_call": 1,
            "fresh_state_per_case": True,
            "candidate_aliases_hide_rank": True,
            "baseline_aliases_model_visible": False,
            "cascade_abstention_sentinel": "DEFER_TO_RETRIEVER",
            "dynamic_schema_candidate_boundary": True,
            "client_side_schema_validation": True,
            "gold_at_inference_boundary": False,
            "fallback": "private_retrieval_top1_after_abstention_or_failure",
            "live_connectors": False,
            "think": None if local or args.synthetic else "low",
        },
    }
    if args.arm == "trigger_consensus_executable":
        binding["trigger_consensus_policy"] = {
            "routing_threshold": CONSENSUS_ROUTING_THRESHOLD,
            "router_sha256": matrix["sources"]["recoverability_router_sha256"],
            "router_runtime_sha256": sha256_file(
                script_root / "frozen_router.py"
            ),
            "first_view": "schema_m5",
            "second_view": "fused_m10",
            "minimum_seed_support": 2,
            "sequential_early_exit": True,
            "confidence_used": False,
            "baseline_action_retained": True,
        }

    if result_path.exists():
        validate_completed_state(
            result_path, progress_path, records_path, binding, selected, args.arm
        )
        print(json.dumps({"status": "already_complete", "result": str(result_path)}))
        return 0
    partial_exists = any(path.exists() for path in (records_path, attempts_path, progress_path))
    if partial_exists and not args.resume:
        raise RuntimeError("partial append-only state exists; pass --resume")
    completed = (
        load_resume_state(
            records_path, attempts_path, progress_path, binding, selected, args.arm
        )
        if args.resume
        else []
    )
    done = {str(row["group_id"]) for row in completed}

    if args.synthetic:
        client: Any = SyntheticToolClient(model_name, attempts_path)
    elif local:
        assert host is not None
        verify_model_digest(host, None, model_name, model_digest, exact=True)
        client = NativeOllamaToolClient(
            host,
            None,
            model_name,
            attempts_path,
            seed=int(matrix["model"]["seed"]),
            think=None,
        )
    else:
        assert host is not None
        verify_model_digest(host, key, model_name, model_digest)
        client = NativeOllamaToolClient(
            host,
            key,
            model_name,
            attempts_path,
            seed=int(matrix["model"]["seed"]),
        )

    write_json_atomic(progress_path, {"phase": "running", "binding": binding, "completed_rows": len(completed), "target_rows": len(selected)}, secret=key)
    try:
        for row in selected:
            if row["group_id"] in done: continue
            consensus_score = None
            if args.arm == "trigger_consensus_executable":
                consensus_score = (
                    1.0
                    if args.synthetic
                    else frozen_routing_score(row, consensus_router or {})
                )
            case = prepare_case(
                row,
                corpora,
                consensus_routing_score=consensus_score,
                require_consensus_metadata=(
                    args.arm == "trigger_consensus_executable"
                ),
            ); outcome = run_case(args.arm, case, client, model_name); record = evaluation_record(row, case, outcome, corpora, args.arm)
            append_jsonl(records_path, record, secret=key); completed.append(record); done.add(str(row["group_id"]))
            write_json_atomic(progress_path, {"phase": "running", "binding": binding, "completed_rows": len(completed), "target_rows": len(selected), "last_group_id_sha256": hashlib.sha256(str(row["group_id"]).encode()).hexdigest()}, secret=key)
            print(json.dumps({"arm": args.arm, "completed": len(completed), "target": len(selected), "terminal_status": outcome.terminal_status, "protocol_valid": record["protocol_valid"]}), flush=True)
    finally: client.close()
    order = {str(x["group_id"]): i for i, x in enumerate(selected)}; completed.sort(key=lambda x: order[str(x["group_id"])])
    if len(completed) != len(selected) or ids_hash(completed) != ids_hash(selected): raise RuntimeError("completed alignment changed")
    result = {"status": "completed", "binding": binding, "metrics": summarize_records(completed)}; write_json_atomic(result_path, result, secret=key)
    write_json_atomic(progress_path, {"phase": "complete", "binding": binding, "completed_rows": len(completed), "target_rows": len(selected), "output": str(result_path), "output_sha256": sha256_file(result_path)}, secret=key)
    print(json.dumps({"status": "completed", "arm": args.arm, "rows": len(completed)})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
