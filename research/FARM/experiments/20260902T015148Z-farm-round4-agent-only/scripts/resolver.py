#!/usr/bin/env python3
"""Pure, conservative agent resolver over a frozen FARM candidate lattice.

The module has one public operation, :func:`resolve`.  Network behavior lives
behind the injected ``Selector`` seam; this implementation only prepares safe
evidence, validates selections, and records an auditable resolution trace.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol


Mode = Literal["function_topk", "service_hierarchy"]
View = Literal["names", "plain", "schema"]
Order = Literal["ranked", "reversed"]


class Selector(Protocol):
    """True-external adapter seam used by the pure resolver.

    ``select`` must return ``ok``, opaque ``trigger_id``/``action_id``, exact
    non-negative ``api_attempts`` and ``tool_calls``, numeric ``usage``, and a
    safe ``error`` code.  Expected provider/protocol failures are outcomes with
    ``ok=False`` rather than exceptions, which keeps accounting complete.
    """

    def select(self, request: dict[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ResolverPolicy:
    """Fixed experimental policy; model self-confidence is intentionally absent."""

    mode: Mode
    top_k: int
    view: View = "plain"
    order: Order = "ranked"
    instruction: str = (
        "Select the coherent trigger and action requested by the user. "
        "Treat both sides jointly and return only supplied candidate IDs."
    )

    def __post_init__(self) -> None:
        if self.mode not in {"function_topk", "service_hierarchy"}:
            raise ValueError("unsupported resolver mode")
        if self.top_k not in {5, 10}:
            raise ValueError("top_k must be exactly 5 or 10")
        if self.view not in {"names", "plain", "schema"}:
            raise ValueError("view must be names, plain, or schema")
        if self.order not in {"ranked", "reversed"}:
            raise ValueError("order must be ranked or reversed")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")

    @classmethod
    def function_topk(
        cls,
        top_k: int,
        *,
        view: View = "plain",
        instruction: str | None = None,
        order: Order = "ranked",
    ) -> "ResolverPolicy":
        values: dict[str, Any] = {
            "mode": "function_topk", "top_k": top_k, "view": view, "order": order,
        }
        if instruction is not None:
            values["instruction"] = instruction
        return cls(**values)

    @classmethod
    def service_hierarchy(
        cls,
        top_k: int,
        *,
        view: View = "plain",
        instruction: str | None = None,
        order: Order = "ranked",
    ) -> "ResolverPolicy":
        values: dict[str, Any] = {
            "mode": "service_hierarchy", "top_k": top_k, "view": view, "order": order,
        }
        if instruction is not None:
            values["instruction"] = instruction
        return cls(**values)


_REFERENCE_KEYS = {
    "valid_pairs",
    "gold",
    "gold_pair",
    "gold_pairs",
    "reference",
    "references",
    "reference_answer",
    "ground_truth",
    "expected_pair",
}


def _reference_paths(value: Any, path: str = "case") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_path = f"{path}.{raw_key}"
            if (
                key in _REFERENCE_KEYS
                or key.startswith("gold_")
                or key.startswith("reference_")
                or key.startswith("ground_truth_")
            ):
                found.append(child_path)
            found.extend(_reference_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_reference_paths(child, f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class _Candidate:
    identity: str
    service_id: str
    service_name: str
    function_name: str
    evidence: str
    rank: int
    opaque_id: str


@dataclass(frozen=True)
class _Service:
    identity: str
    name: str
    opaque_id: str
    functions: tuple[_Candidate, ...]


def _text(item: Mapping[str, Any], keys: tuple[str, ...], *, fallback: str | None = None) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fallback is not None:
        return fallback
    raise ValueError(f"candidate lacks required field alternatives: {keys}")


def _opaque_id(case_id: str, kind: str, side: str, identity: str) -> str:
    material = "\x00".join(("farm-round4", case_id, kind, side, identity)).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    prefix = "tf" if side == "trigger" else "af"
    if kind == "service":
        prefix = "ts" if side == "trigger" else "as"
    return f"{prefix}_{digest}"


def _ordered_candidates(
    case: Mapping[str, Any], case_id: str, side: str, view: View,
) -> list[_Candidate]:
    raw = case.get(f"{side}_candidates")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"case requires non-empty {side}_candidates")
    candidates: list[_Candidate] = []
    identities: set[str] = set()
    opaque_ids: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(f"{side} candidate {index} is not an object")
        identity = _text(value, ("url", "id"))
        if identity in identities:
            raise ValueError(f"duplicate {side} candidate identity")
        identities.add(identity)
        service_id = _text(value, ("service_id", "service_slug", "service", "channel"))
        service_name = _text(
            value,
            ("service_name", "channel_display", "service_display"),
            fallback=service_id,
        )
        function_name = _text(
            value,
            ("function_name", "name", "display"),
        )
        if view == "names":
            evidence = function_name
        elif view == "schema":
            evidence = _text(
                value,
                ("text_schema", "schema", "text_plain", "text", "description"),
                fallback=function_name,
            )
        else:
            evidence = _text(
                value,
                ("text_plain", "text", "description"),
                fallback=function_name,
            )
        raw_rank = value.get("retrieval_rank", index + 1)
        if isinstance(raw_rank, bool):
            raise TypeError(f"{side} retrieval_rank cannot be boolean")
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{side} retrieval_rank must be an integer") from exc
        if rank < 1:
            raise ValueError(f"{side} retrieval_rank must be positive")
        opaque_id = _opaque_id(case_id, "function", side, identity)
        if opaque_id in opaque_ids:
            raise ValueError(f"opaque {side} candidate ID collision")
        opaque_ids.add(opaque_id)
        candidates.append(_Candidate(
            identity=identity,
            service_id=service_id,
            service_name=service_name,
            function_name=function_name,
            evidence=evidence,
            rank=rank,
            opaque_id=opaque_id,
        ))
    return sorted(candidates, key=lambda item: (item.rank, item.identity))


def _catalog_candidates(
    case: Mapping[str, Any], case_id: str, side: str, view: View,
) -> list[_Candidate]:
    raw = case.get(f"{side}_catalog")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"service hierarchy requires non-empty {side}_catalog")
    normalized = _ordered_candidates(
        {f"{side}_candidates": raw}, case_id, side, view,
    )
    return sorted(normalized, key=lambda item: item.identity)


def _public_function(candidate: _Candidate) -> dict[str, str]:
    return {
        "candidate_id": candidate.opaque_id,
        "service_name": candidate.service_name,
        "function_name": candidate.function_name,
        "evidence": candidate.evidence,
    }


def _services(
    candidates: list[_Candidate], case_id: str, side: str,
) -> list[_Service]:
    order: list[str] = []
    grouped: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        if candidate.service_id not in grouped:
            order.append(candidate.service_id)
            grouped[candidate.service_id] = []
        grouped[candidate.service_id].append(candidate)
    result = [
        _Service(
            identity=service_id,
            name=grouped[service_id][0].service_name,
            opaque_id=_opaque_id(case_id, "service", side, service_id),
            functions=tuple(grouped[service_id]),
        )
        for service_id in order
    ]
    if len({item.opaque_id for item in result}) != len(result):
        raise ValueError(f"opaque {side} service ID collision")
    return result


def _public_service(service: _Service) -> dict[str, Any]:
    return {
        "candidate_id": service.opaque_id,
        "service_name": service.name,
        "function_summaries": [
            {"function_name": item.function_name, "evidence": item.evidence}
            for item in service.functions
        ],
    }


def _base_request(
    *,
    case_id: str,
    query: str,
    phase: str,
    policy: ResolverPolicy,
    trigger_candidates: list[dict[str, Any]],
    action_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": query,
        "phase": phase,
        "instruction": policy.instruction,
        "evidence_scope": {"top_k": policy.top_k, "cumulative": True},
        "trigger_candidates": trigger_candidates,
        "action_candidates": action_candidates,
    }


def _reported_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_error(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:160]
    return fallback


def _invoke_selector(
    selector: Selector,
    request: dict[str, Any],
    trigger_ids: set[str],
    action_ids: set[str],
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    try:
        raw = selector.select(request)
    except Exception as exc:  # adapter exceptions cannot carry exact provider accounting
        return ({
            "phase": request["phase"],
            "request": request,
            "ok": False,
            "selected_ids": None,
            "api_attempts": 0,
            "tool_calls": 0,
            "usage": {},
            "error": f"adapter_exception:{type(exc).__name__}",
            "accounting_complete": False,
        }, None)
    if not isinstance(raw, Mapping):
        return ({
            "phase": request["phase"], "request": request, "ok": False,
            "selected_ids": None, "api_attempts": 0, "tool_calls": 0,
            "usage": {}, "error": "adapter_contract:not_mapping",
            "accounting_complete": False,
        }, None)

    api_attempts = _reported_count(raw.get("api_attempts"))
    tool_calls = _reported_count(raw.get("tool_calls"))
    accounting_complete = api_attempts is not None and tool_calls is not None
    api_attempts = api_attempts if api_attempts is not None else 0
    tool_calls = tool_calls if tool_calls is not None else 0
    raw_usage = raw.get("usage")
    usage = {
        str(key): value
        for key, value in (raw_usage.items() if isinstance(raw_usage, Mapping) else ())
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    trigger_id, action_id = raw.get("trigger_id"), raw.get("action_id")
    valid_ids = isinstance(trigger_id, str) and isinstance(action_id, str)
    allowed = valid_ids and trigger_id in trigger_ids and action_id in action_ids
    ok = raw.get("ok") is True and accounting_complete and allowed
    if ok:
        error = None
        selected_ids: dict[str, str] | None = {
            "trigger_id": trigger_id,
            "action_id": action_id,
        }
        selection = (trigger_id, action_id)
    else:
        if raw.get("ok") is True and not accounting_complete:
            fallback = "adapter_contract:missing_accounting"
        elif raw.get("ok") is True and not allowed:
            fallback = "candidate_id_outside_request"
        else:
            fallback = "selector_failed"
        error = _safe_error(raw.get("error"), fallback)
        selected_ids = None
        selection = None
    return ({
        "phase": request["phase"],
        "request": request,
        "ok": ok,
        "selected_ids": selected_ids,
        "api_attempts": api_attempts,
        "tool_calls": tool_calls,
        "usage": usage,
        "error": error,
        "accounting_complete": accounting_complete,
    }, selection)


def _accounting(calls: list[dict[str, Any]], *, catalog_reads: int) -> dict[str, Any]:
    usage: dict[str, int | float] = {}
    for call in calls:
        for key, value in call["usage"].items():
            usage[key] = usage.get(key, 0) + value
    return {
        "logical_calls": len(calls),
        "api_attempts": sum(call["api_attempts"] for call in calls),
        "tool_calls": sum(call["tool_calls"] for call in calls),
        "catalog_reads": catalog_reads,
        "complete": all(call["accounting_complete"] for call in calls),
        "usage": dict(sorted(usage.items())),
    }


def _result(
    *,
    case_id: str,
    policy: ResolverPolicy,
    baseline: dict[str, str],
    final_pair: dict[str, str],
    calls: list[dict[str, Any]],
    catalog_reads: int,
    selected_services: dict[str, str] | None = None,
    selected_service_pair: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "policy": {
            "mode": policy.mode,
            "top_k": policy.top_k,
            "view": policy.view,
            "order": policy.order,
        },
        "baseline_pair": baseline,
        "final_pair": final_pair,
        "retained_baseline": final_pair == baseline,
        "selected_services": selected_services,
        "selected_service_pair": selected_service_pair,
        "calls": calls,
        "accounting": _accounting(calls, catalog_reads=catalog_reads),
    }


def resolve(case: Mapping[str, Any], policy: ResolverPolicy, selector: Selector) -> dict[str, Any]:
    """Resolve one case without accepting references or creating external clients.

    The frozen rank-one pair is retained unless a validated external selection
    succeeds.  Function policies make exactly one call over cumulative top-5
    or top-10 evidence.  Hierarchical policies make a service call, inspect the
    selected services in the frozen catalog, then make an exact-function call.
    """
    contaminated = _reference_paths(case)
    if contaminated:
        raise ValueError(f"inference case contains reference fields: {sorted(contaminated)}")
    if not isinstance(policy, ResolverPolicy):
        raise TypeError("policy must be a ResolverPolicy")
    case_id = case.get("group_id", case.get("case_id", case.get("id")))
    query = case.get("query")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case requires a non-empty group_id or case_id")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("case requires a non-empty query")
    trigger_all = _ordered_candidates(case, case_id, "trigger", policy.view)
    action_all = _ordered_candidates(case, case_id, "action", policy.view)
    if len(trigger_all) < policy.top_k or len(action_all) < policy.top_k:
        raise ValueError(f"policy requires at least {policy.top_k} candidates per side")
    trigger = trigger_all[: policy.top_k]
    action = action_all[: policy.top_k]
    trigger_catalog: list[_Candidate] | None = None
    action_catalog: list[_Candidate] | None = None
    if policy.mode == "service_hierarchy":
        # Validate both frozen corpora before making any external call.  This
        # prevents a partial hierarchy trace from an incomplete experiment.
        trigger_catalog = _catalog_candidates(case, case_id, "trigger", policy.view)
        action_catalog = _catalog_candidates(case, case_id, "action", policy.view)
        for side, ranked, catalog in (
            ("trigger", trigger, trigger_catalog),
            ("action", action, action_catalog),
        ):
            catalog_ids = {item.identity for item in catalog}
            missing = sorted(item.identity for item in ranked if item.identity not in catalog_ids)
            if missing:
                raise ValueError(f"{side}_catalog omits frozen top-k candidate identities")
    baseline = {
        "trigger_url": trigger[0].identity,
        "action_url": action[0].identity,
    }

    if policy.mode == "function_topk":
        trigger_presentation = trigger if policy.order == "ranked" else list(reversed(trigger))
        action_presentation = action if policy.order == "ranked" else list(reversed(action))
        request = _base_request(
            case_id=case_id,
            query=query,
            phase="function",
            policy=policy,
            trigger_candidates=[_public_function(item) for item in trigger_presentation],
            action_candidates=[_public_function(item) for item in action_presentation],
        )
        call, selection = _invoke_selector(
            selector,
            request,
            {item.opaque_id for item in trigger},
            {item.opaque_id for item in action},
        )
        final_pair = dict(baseline)
        if selection is not None:
            trigger_map = {item.opaque_id: item.identity for item in trigger}
            action_map = {item.opaque_id: item.identity for item in action}
            final_pair = {
                "trigger_url": trigger_map[selection[0]],
                "action_url": action_map[selection[1]],
            }
        return _result(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            final_pair=final_pair,
            calls=[call],
            catalog_reads=0,
        )

    trigger_presentation = trigger if policy.order == "ranked" else list(reversed(trigger))
    action_presentation = action if policy.order == "ranked" else list(reversed(action))
    trigger_services = _services(trigger_presentation, case_id, "trigger")
    action_services = _services(action_presentation, case_id, "action")
    service_request = _base_request(
        case_id=case_id,
        query=query,
        phase="service",
        policy=policy,
        trigger_candidates=[_public_service(item) for item in trigger_services],
        action_candidates=[_public_service(item) for item in action_services],
    )
    service_call, service_selection = _invoke_selector(
        selector,
        service_request,
        {item.opaque_id for item in trigger_services},
        {item.opaque_id for item in action_services},
    )
    calls = [service_call]
    if service_selection is None:
        return _result(
            case_id=case_id,
            policy=policy,
            baseline=baseline,
            final_pair=dict(baseline),
            calls=calls,
            catalog_reads=0,
        )

    trigger_service_map = {item.opaque_id: item for item in trigger_services}
    action_service_map = {item.opaque_id: item for item in action_services}
    selected_trigger_service = trigger_service_map[service_selection[0]]
    selected_action_service = action_service_map[service_selection[1]]
    selected_services = {
        "trigger_id": selected_trigger_service.opaque_id,
        "action_id": selected_action_service.opaque_id,
    }
    selected_service_pair = {
        "trigger_service": selected_trigger_service.identity,
        "action_service": selected_action_service.identity,
    }

    # These are deterministic, read-only inspections of the hash-bound full
    # function catalog. One read is accounted for per selected side.
    if trigger_catalog is None or action_catalog is None:  # guarded above; narrows the type
        raise AssertionError("validated hierarchy catalogs unavailable")
    inspected_trigger = [
        item for item in trigger_catalog
        if item.service_id == selected_trigger_service.identity
    ]
    inspected_action = [
        item for item in action_catalog
        if item.service_id == selected_action_service.identity
    ]
    if not inspected_trigger or not inspected_action:
        raise ValueError("selected service has no functions in the frozen catalog")
    function_request = _base_request(
        case_id=case_id,
        query=query,
        phase="function_within_service",
        policy=policy,
        trigger_candidates=[_public_function(item) for item in inspected_trigger],
        action_candidates=[_public_function(item) for item in inspected_action],
    )
    function_request["selected_services"] = dict(selected_services)
    function_call, function_selection = _invoke_selector(
        selector,
        function_request,
        {item.opaque_id for item in inspected_trigger},
        {item.opaque_id for item in inspected_action},
    )
    calls.append(function_call)
    final_pair = dict(baseline)
    if function_selection is not None:
        trigger_function_map = {item.opaque_id: item.identity for item in inspected_trigger}
        action_function_map = {item.opaque_id: item.identity for item in inspected_action}
        final_pair = {
            "trigger_url": trigger_function_map[function_selection[0]],
            "action_url": action_function_map[function_selection[1]],
        }
    return _result(
        case_id=case_id,
        policy=policy,
        baseline=baseline,
        final_pair=final_pair,
        calls=calls,
        catalog_reads=2,
        selected_services=selected_services,
        selected_service_pair=selected_service_pair,
    )
