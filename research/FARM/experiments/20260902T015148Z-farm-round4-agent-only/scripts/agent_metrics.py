#!/usr/bin/env python3
"""Deterministic, gold-only evaluation for FARM agent selection traces.

The evaluator deliberately does not resolve model output or inspect prompts.  It
accepts already-resolved predictions and compares them with *sets* of valid gold
pairs.  This separation makes harness failures, protocol failures, and semantic
selection errors measurable instead of silently repairing one into another.

Two record layouts are accepted:

* multi-arm records with ``arms = {name: trace}``; and
* flat records with ``baseline_pair`` and ``final_pair``.  Flat records are
  normalized to the arms ``baseline`` and ``agent``.

Every record must contain ``valid_pairs``, trigger/action candidates (or rank
metadata for coverage), and service identity for every gold service pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OUTCOMES = (
    "function_trigger",
    "function_action",
    "function_joint",
    "service_trigger",
    "service_action",
    "service_joint",
)
RANK_BUCKETS = ("rank1", "rank2_5", "rank6_10", "outside10")
K_VALUES = (1, 5, 10)


def _first(mapping: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _scalar_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = _first(
            value,
            ("url", "id", "identifier", "function_url", "candidate_id", "value", "name"),
        )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _service_identifier(mapping: Mapping[str, Any], side: str) -> str | None:
    value = _first(
        mapping,
        (
            f"{side}_service",
            f"{side}_service_id",
            f"{side}_channel",
            f"{side}_channel_id",
        ),
    )
    return _scalar_identifier(value)


def _side_identifier(mapping: Mapping[str, Any], side: str) -> str | None:
    value = _first(
        mapping,
        (
            f"{side}_url",
            f"{side}_function_url",
            f"{side}_id",
            f"selected_{side}",
            side,
        ),
    )
    return _scalar_identifier(value)


def _candidate_rows(record: Mapping[str, Any], side: str) -> list[Any]:
    direct = record.get(f"{side}_candidates")
    if direct is None and isinstance(record.get("candidates"), Mapping):
        direct = record["candidates"].get(side)
    if direct is None and isinstance(record.get("rankings"), Mapping):
        direct = record["rankings"].get(side)
    if direct is None:
        return []
    if not isinstance(direct, Sequence) or isinstance(direct, (str, bytes)):
        raise ValueError(f"{side}_candidates must be a sequence")
    return list(direct)


def _candidate_identity(candidate: Any) -> str | None:
    return _scalar_identifier(candidate)


def _candidate_service(candidate: Any) -> str | None:
    if not isinstance(candidate, Mapping):
        return None
    return _scalar_identifier(
        _first(candidate, ("service", "service_id", "channel", "channel_id"))
    )


def _ranked_candidate_rows(record: Mapping[str, Any], side: str) -> list[Any]:
    ranked: list[tuple[int, str, int, Any]] = []
    for index, candidate in enumerate(_candidate_rows(record, side)):
        identifier = _candidate_identity(candidate)
        if identifier is None:
            continue
        raw_rank = candidate.get("retrieval_rank") if isinstance(candidate, Mapping) else None
        try:
            rank = int(raw_rank) if raw_rank is not None else index + 1
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid {side} retrieval_rank: {raw_rank!r}") from error
        if rank < 1:
            raise ValueError(f"{side} retrieval_rank must be positive")
        ranked.append((rank, identifier, index, candidate))
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    result: list[Any] = []
    seen: set[str] = set()
    for _rank, identifier, _index, candidate in ranked:
        if identifier not in seen:
            result.append(candidate)
            seen.add(identifier)
    return result


def _candidate_order(record: Mapping[str, Any], side: str) -> list[str]:
    return [
        identifier
        for candidate in _ranked_candidate_rows(record, side)
        if (identifier := _candidate_identity(candidate)) is not None
    ]


def _url_service_map(record: Mapping[str, Any], side: str) -> dict[str, str]:
    memberships: dict[str, set[str]] = defaultdict(set)
    for candidate in _candidate_rows(record, side):
        identifier = _candidate_identity(candidate)
        service = _candidate_service(candidate)
        if identifier and service:
            memberships[identifier].add(service)

    pairs = _raw_valid_pairs(record)
    for pair in pairs:
        if not isinstance(pair, Mapping):
            continue
        identifier = _side_identifier(pair, side)
        service = _service_identifier(pair, side)
        if identifier and service:
            memberships[identifier].add(service)

    explicit = record.get(f"{side}_url_to_service")
    if isinstance(explicit, Mapping):
        for identifier, service in explicit.items():
            parsed = _scalar_identifier(service)
            if parsed:
                memberships[str(identifier)].add(parsed)

    # Ambiguous URL-to-service mappings are not guessed.
    return {
        identifier: next(iter(services))
        for identifier, services in memberships.items()
        if len(services) == 1
    }


def _raw_valid_pairs(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("valid_pairs")
    if value is None:
        value = record.get("gold_pairs")
    if value is None and isinstance(record.get("gold"), Mapping):
        value = _first(record["gold"], ("valid_pairs", "pairs"))
    if value is None:
        raise ValueError("record is missing valid_pairs")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("valid_pairs must be a sequence")
    if not value:
        raise ValueError("valid_pairs cannot be empty")
    return list(value)


def _explicit_service_pairs(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("gold_service_pairs")
    if value is None:
        value = record.get("valid_service_pairs")
    if value is None and isinstance(record.get("gold"), Mapping):
        value = record["gold"].get("service_pairs")
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("gold_service_pairs must be a sequence")
    return list(value)


def _parse_pair(value: Any) -> tuple[str | None, str | None, str | None, str | None]:
    if isinstance(value, Mapping):
        return (
            _side_identifier(value, "trigger"),
            _side_identifier(value, "action"),
            _service_identifier(value, "trigger"),
            _service_identifier(value, "action"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) == 2:
            return (_scalar_identifier(items[0]), _scalar_identifier(items[1]), None, None)
        if len(items) >= 4:
            return tuple(_scalar_identifier(item) for item in items[:4])  # type: ignore[return-value]
    raise ValueError("a valid pair must be a mapping or a 2/4-element sequence")


def _parse_service_pair(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, Mapping):
        return (_service_identifier(value, "trigger"), _service_identifier(value, "action"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return (_scalar_identifier(value[0]), _scalar_identifier(value[1]))
    raise ValueError("a service pair must be a mapping or a 2-element sequence")


def _gold(record: Mapping[str, Any]) -> dict[str, Any]:
    trigger_map = _url_service_map(record, "trigger")
    action_map = _url_service_map(record, "action")
    resolved: set[tuple[str, str, str | None, str | None]] = set()
    url_pairs: set[tuple[str, str]] = set()
    service_pairs: set[tuple[str, str]] = set()

    for raw in _raw_valid_pairs(record):
        trigger, action, trigger_service, action_service = _parse_pair(raw)
        if trigger is None or action is None:
            raise ValueError("every valid pair needs trigger_url and action_url")
        trigger_service = trigger_service or trigger_map.get(trigger)
        action_service = action_service or action_map.get(action)
        url_pairs.add((trigger, action))
        resolved.add((trigger, action, trigger_service, action_service))
        if trigger_service is not None and action_service is not None:
            service_pairs.add((trigger_service, action_service))

    for raw in _explicit_service_pairs(record):
        trigger_service, action_service = _parse_service_pair(raw)
        if trigger_service is None or action_service is None:
            raise ValueError("every gold service pair needs trigger and action services")
        service_pairs.add((trigger_service, action_service))

    if not service_pairs:
        raise ValueError(
            "service metrics require service identity in valid_pairs, candidates, "
            "URL maps, or gold_service_pairs"
        )
    return {
        "resolved_pairs": resolved,
        "url_pairs": url_pairs,
        "trigger_urls": {pair[0] for pair in url_pairs},
        "action_urls": {pair[1] for pair in url_pairs},
        "service_pairs": service_pairs,
        "trigger_services": {pair[0] for pair in service_pairs},
        "action_services": {pair[1] for pair in service_pairs},
    }


def _nested_prediction(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    for key in ("prediction", "final_pair", "selected_pair", "pair", "resolved_pair"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return _nested_prediction(nested)
    selection = value.get("selection")
    if isinstance(selection, Mapping):
        arguments = selection.get("arguments")
        if isinstance(arguments, Mapping):
            return _nested_prediction(arguments)
        return _nested_prediction(selection)
    return value


def _prediction(record: Mapping[str, Any], value: Any) -> dict[str, str | None]:
    pair = _nested_prediction(value)
    trigger = _side_identifier(pair, "trigger")
    action = _side_identifier(pair, "action")
    trigger_service = _service_identifier(pair, "trigger")
    action_service = _service_identifier(pair, "action")
    if trigger_service is None and trigger is not None:
        trigger_service = _url_service_map(record, "trigger").get(trigger)
    if action_service is None and action is not None:
        action_service = _url_service_map(record, "action").get(action)
    return {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": trigger_service,
        "action_service": action_service,
    }


def score_prediction(record: Mapping[str, Any], prediction: Any) -> dict[str, bool]:
    """Score one resolved prediction against all valid function/service pairs."""
    gold = _gold(record)
    selected = _prediction(record, prediction)
    trigger = selected["trigger_url"]
    action = selected["action_url"]
    trigger_service = selected["trigger_service"]
    action_service = selected["action_service"]
    return {
        "function_trigger": trigger in gold["trigger_urls"] if trigger is not None else False,
        "function_action": action in gold["action_urls"] if action is not None else False,
        "function_joint": (trigger, action) in gold["url_pairs"],
        "service_trigger": (
            trigger_service in gold["trigger_services"] if trigger_service is not None else False
        ),
        "service_action": (
            action_service in gold["action_services"] if action_service is not None else False
        ),
        "service_joint": (trigger_service, action_service) in gold["service_pairs"],
    }


def _selected_service_pair(arm: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if arm is None:
        return (None, None)
    value = arm.get("selected_service_pair")
    if isinstance(value, Mapping):
        return (_service_identifier(value, "trigger"), _service_identifier(value, "action"))
    # Compatibility for early resolver traces where these IDs were already
    # real channel identities, not opaque candidate IDs.
    value = arm.get("selected_services")
    if isinstance(value, Mapping):
        return (
            _scalar_identifier(_first(value, ("trigger_service", "trigger_id"))),
            _scalar_identifier(_first(value, ("action_service", "action_id"))),
        )
    return (None, None)


def _service_stage_applicable(arm: Mapping[str, Any] | None) -> bool:
    if arm is None:
        return False
    policy = arm.get("policy")
    return (
        isinstance(policy, Mapping) and policy.get("mode") == "service_hierarchy"
    ) or isinstance(arm.get("selected_service_pair"), Mapping) or isinstance(
        arm.get("selected_services"), Mapping
    )


def _service_call_protocol_valid(arm: Mapping[str, Any] | None) -> bool:
    if arm is None:
        return False
    calls = arm.get("calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return bool(arm.get("service_protocol_valid", False))
    service_calls = [
        call
        for call in calls
        if isinstance(call, Mapping) and call.get("phase") == "service"
    ]
    return bool(service_calls) and all(
        bool(call.get("ok")) and bool(call.get("accounting_complete", True))
        for call in service_calls
    )


def _service_stage_score(record: Mapping[str, Any], arm: Mapping[str, Any] | None) -> dict[str, bool]:
    trigger, action = _selected_service_pair(arm)
    gold = _gold(record)
    return {
        "service_trigger": trigger in gold["trigger_services"] if trigger is not None else False,
        "service_action": action in gold["action_services"] if action is not None else False,
        "service_joint": (trigger, action) in gold["service_pairs"],
    }


def _service_stage_summary(
    records: Sequence[Mapping[str, Any]], arm_name: str, baseline_arm: str
) -> dict[str, Any]:
    scores: list[dict[str, bool]] = []
    conditioned: list[dict[str, bool]] = []
    valid_calls = 0
    selections = 0
    for record in records:
        active = _arm(record, arm_name, baseline_arm)
        if not _service_stage_applicable(active):
            continue
        score = _service_stage_score(record, active)
        scores.append(score)
        selected = _selected_service_pair(active)
        complete = selected[0] is not None and selected[1] is not None
        if complete:
            selections += 1
        valid = complete and _service_call_protocol_valid(active)
        if valid:
            valid_calls += 1
            conditioned.append(score)

    def summarize(values: Sequence[Mapping[str, bool]], *, excluded: int | None = None) -> dict[str, Any]:
        denominator = len(values)
        result: dict[str, Any] = {"denominator_cases": denominator}
        if excluded is not None:
            result["excluded_cases"] = excluded
        for outcome in ("service_trigger", "service_action", "service_joint"):
            hits = sum(bool(value[outcome]) for value in values)
            result[outcome] = {"hits": hits, "rate": hits / denominator if denominator else None}
        return result

    return {
        "applicable_cases": len(scores),
        "selected_service_pair_cases": selections,
        "service_call_protocol_valid_cases": valid_calls,
        "intent_to_treat": summarize(scores),
        "protocol_conditioned": summarize(conditioned, excluded=len(scores) - len(conditioned)),
    }


def exact_mcnemar(before: Sequence[int | bool], after: Sequence[int | bool]) -> dict[str, Any]:
    """Return the exact two-sided McNemar test for paired binary outcomes."""
    if len(before) != len(after):
        raise ValueError("paired vectors must have equal length")
    recoveries = sum(not bool(left) and bool(right) for left, right in zip(before, after))
    regressions = sum(bool(left) and not bool(right) for left, right in zip(before, after))
    discordant = recoveries + regressions
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(recoveries, regressions)
        probability = Fraction(sum(math.comb(discordant, k) for k in range(lower + 1)), 1 << discordant)
        p_value = min(1.0, float(2 * probability))
    return {
        "denominator_pairs": len(before),
        "recoveries": recoveries,
        "regressions": regressions,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def paired_bootstrap_delta(
    before: Sequence[int | bool],
    after: Sequence[int | bool],
    *,
    seed: int,
    iterations: int = 5000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paired percentile-bootstrap interval for an accuracy difference."""
    if len(before) != len(after):
        raise ValueError("paired vectors must have equal length")
    if not before:
        raise ValueError("paired bootstrap needs at least one pair")
    if iterations < 2:
        raise ValueError("bootstrap iterations must be at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    differences = [int(bool(right)) - int(bool(left)) for left, right in zip(before, after)]
    count = len(differences)
    rng = random.Random(seed)
    draws = [sum(rng.choices(differences, k=count)) / count for _ in range(iterations)]
    draws.sort()
    alpha = 1.0 - confidence_level
    lower_index = min(iterations - 1, max(0, math.floor((alpha / 2.0) * iterations)))
    upper_index = min(
        iterations - 1,
        max(0, math.ceil((1.0 - alpha / 2.0) * iterations) - 1),
    )
    return {
        "denominator_pairs": count,
        "observed_delta": sum(differences) / count,
        "confidence_level": confidence_level,
        "method": "paired_percentile_bootstrap",
        "iterations": iterations,
        "seed": seed,
        "ci_lower": draws[lower_index],
        "ci_upper": draws[upper_index],
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed like the supplied family."""
    for label, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid p-value for {label}: {value}")
    ordered = sorted(((label, float(value)) for label, value in p_values.items()), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - index) * value))
        adjusted[label] = running
    return {label: adjusted[label] for label in p_values}


def _case_id(record: Mapping[str, Any], index: int) -> str:
    return str(_first(record, ("case_id", "group_id", "id"), f"row-{index}"))


def _flat_arm(record: Mapping[str, Any], *, baseline: bool) -> dict[str, Any]:
    if baseline:
        pair = _first(record, ("baseline_pair", "retrieval_pair", "top1_pair"), {})
        return {
            "prediction": pair,
            "protocol_valid": True,
            "routed": False,
            "accounting": {
                "logical_calls": 0,
                "api_attempts": 0,
                "tool_calls": 0,
                "catalog_reads": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
        }
    result = dict(record)
    result["prediction"] = _first(record, ("final_pair", "prediction", "selected_pair"), {})
    return result


def _infer_arm_names(records: Sequence[Mapping[str, Any]], baseline_arm: str) -> list[str]:
    names: set[str] = set()
    flat = False
    for record in records:
        arms = record.get("arms")
        if isinstance(arms, Mapping):
            names.update(str(name) for name in arms)
        else:
            flat = True
    if flat:
        names.update((baseline_arm, "agent"))
    if baseline_arm not in names:
        raise ValueError(f"baseline arm {baseline_arm!r} is absent")
    return [baseline_arm, *sorted(name for name in names if name != baseline_arm)]


def _arm(record: Mapping[str, Any], name: str, baseline_arm: str) -> dict[str, Any] | None:
    arms = record.get("arms")
    if isinstance(arms, Mapping):
        value = arms.get(name)
        return dict(value) if isinstance(value, Mapping) else None
    if name == baseline_arm:
        return _flat_arm(record, baseline=True)
    if name == "agent":
        return _flat_arm(record, baseline=False)
    return None


def _numeric(mapping: Mapping[str, Any], names: Sequence[str]) -> float | None:
    value = _first(mapping, names)
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric accounting value for {names[0]}: {value!r}") from error


def _iteration_usage(iteration: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = iteration.get("selection")
    if isinstance(selection, Mapping):
        accounting = selection.get("accounting")
        if isinstance(accounting, Mapping):
            return accounting
        usage = selection.get("usage")
        if isinstance(usage, Mapping):
            return usage
        return selection
    accounting = iteration.get("accounting")
    if isinstance(accounting, Mapping):
        return accounting
    usage = iteration.get("usage")
    if isinstance(usage, Mapping):
        return usage
    return iteration


def _case_accounting(arm: Mapping[str, Any] | None) -> dict[str, float | int]:
    if arm is None:
        return {
            "logical_calls": 0,
            "api_attempts": 0,
            "tool_calls": 0,
            "catalog_reads": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "latency_observed": False,
        }
    accounting = arm.get("accounting")
    usage = arm.get("usage")
    if isinstance(accounting, Mapping):
        primary = accounting
    elif isinstance(usage, Mapping):
        primary = usage
    else:
        primary = arm
    iterations = arm.get("iterations")
    if iterations is None:
        iterations = arm.get("calls")
    iteration_rows = (
        [item for item in iterations if isinstance(item, Mapping)]
        if isinstance(iterations, Sequence) and not isinstance(iterations, (str, bytes))
        else []
    )

    sources: list[Mapping[str, Any]] = [primary]
    primary_usage = primary.get("usage")
    if isinstance(primary_usage, Mapping):
        sources.append(primary_usage)
    if primary is not arm:
        sources.append(arm)
        arm_usage = arm.get("usage")
        if isinstance(arm_usage, Mapping):
            sources.append(arm_usage)

    def value(names: Sequence[str], iteration_names: Sequence[str] | None = None) -> float | None:
        for source in sources:
            direct = _numeric(source, names)
            if direct is not None:
                return direct
        targets = iteration_names or names
        values: list[float | None] = []
        for item in iteration_rows:
            direct = _numeric(item, targets)
            values.append(direct if direct is not None else _numeric(_iteration_usage(item), targets))
        observed = [item for item in values if item is not None]
        return sum(observed) if observed else None

    logical_calls = value(("logical_calls", "calls"))
    if logical_calls is None:
        logical_calls = float(len(iteration_rows))
    attempts = value(("api_attempts", "attempts")) or 0.0
    tools = value(("tool_calls", "tool_call_count")) or 0.0
    catalog_reads = value(("catalog_reads", "catalog_read_count")) or 0.0
    prompt = value(("prompt_tokens", "input_tokens", "prompt_eval_count")) or 0.0
    completion = value(("completion_tokens", "output_tokens", "eval_count")) or 0.0
    total = value(("total_tokens",))
    if total is None:
        total = prompt + completion
    latency_seconds = value(("latency_seconds", "wall_seconds", "elapsed_seconds"))
    latency_ms = None
    duration_ns = None
    if latency_seconds is None:
        latency_ms = value(("latency_ms", "elapsed_ms"))
    if latency_seconds is None and latency_ms is None:
        duration_ns = value(("provider_duration_ns", "total_duration_ns", "total_duration"))
    latency_observed = any(item is not None for item in (latency_seconds, latency_ms, duration_ns))
    latency = latency_seconds
    if latency is None:
        latency = latency_ms / 1000.0 if latency_ms is not None else None
    if latency is None:
        latency = duration_ns / 1_000_000_000.0 if duration_ns is not None else 0.0

    integer_fields = {
        "logical_calls": logical_calls,
        "api_attempts": attempts,
        "tool_calls": tools,
        "catalog_reads": catalog_reads,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    normalized: dict[str, float | int] = {}
    for name, raw in integer_fields.items():
        if raw < 0 or not float(raw).is_integer():
            raise ValueError(f"{name} must be a non-negative integer")
        normalized[name] = int(raw)
    if latency < 0 or not math.isfinite(latency):
        raise ValueError("latency_seconds must be finite and non-negative")
    normalized["latency_seconds"] = float(latency)
    normalized["latency_observed"] = latency_observed
    return normalized


def _fallback_used(arm: Mapping[str, Any] | None) -> bool:
    if arm is None:
        return False
    explicit_names = ("fallback_used", "fallback", "protocol_fallback")
    if any(name in arm for name in explicit_names):
        return bool(_first(arm, explicit_names, False))
    calls = arm.get("calls")
    if bool(arm.get("retained_baseline")) and isinstance(calls, Sequence) and not isinstance(
        calls, (str, bytes)
    ):
        return any(isinstance(call, Mapping) and not bool(call.get("ok")) for call in calls)
    return False


def _model_protocol_valid(arm: Mapping[str, Any] | None) -> bool:
    if arm is None:
        return False
    if "protocol_valid" in arm:
        return bool(arm["protocol_valid"])
    status = arm.get("protocol_status")
    if status is not None:
        return str(status).lower() in {"valid", "ok", "success", "passed"}
    calls = arm.get("calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        call_rows = [call for call in calls if isinstance(call, Mapping)]
        if not call_rows:
            return False
        accounting = arm.get("accounting")
        aggregate_complete = (
            bool(accounting.get("complete", True)) if isinstance(accounting, Mapping) else True
        )
        return aggregate_complete and all(
            bool(call.get("ok")) and bool(call.get("accounting_complete", True))
            for call in call_rows
        )
    return not _fallback_used(arm)


def _selection_within_candidates(
    record: Mapping[str, Any],
    arm: Mapping[str, Any] | None,
    selected: Mapping[str, str | None],
) -> bool:
    complete = selected["trigger_url"] is not None and selected["action_url"] is not None
    if not complete:
        return False
    if arm is not None and "selection_within_candidates" in arm:
        return bool(arm["selection_within_candidates"])

    policy = arm.get("policy") if arm is not None else None
    policy = policy if isinstance(policy, Mapping) else {}
    mode = policy.get("mode")
    top_k = policy.get("top_k")
    if mode == "function_topk" and isinstance(top_k, int) and not isinstance(top_k, bool):
        trigger_order = _candidate_order(record, "trigger")[:top_k]
        action_order = _candidate_order(record, "action")[:top_k]
    elif mode == "service_hierarchy":
        trigger_catalog = record.get("trigger_catalog")
        action_catalog = record.get("action_catalog")
        if isinstance(trigger_catalog, Sequence) and isinstance(action_catalog, Sequence):
            trigger_order = [
                identifier
                for item in trigger_catalog
                if (identifier := _candidate_identity(item)) is not None
            ]
            action_order = [
                identifier
                for item in action_catalog
                if (identifier := _candidate_identity(item)) is not None
            ]
        else:
            # The pure resolver validates opaque IDs against each call's exact
            # universe.  If catalogs are intentionally not persisted, a valid
            # canonical call trace is stronger evidence than the retrieval top-k.
            calls = arm.get("calls") if arm is not None else None
            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
                call_rows = [call for call in calls if isinstance(call, Mapping)]
                return bool(call_rows) and all(bool(call.get("ok")) for call in call_rows)
            trigger_order = _candidate_order(record, "trigger")
            action_order = _candidate_order(record, "action")
    else:
        trigger_order = _candidate_order(record, "trigger")
        action_order = _candidate_order(record, "action")

    trigger_valid = not trigger_order or selected["trigger_url"] in set(trigger_order)
    action_valid = not action_order or selected["action_url"] in set(action_order)
    return trigger_valid and action_valid


def _protocol_state(
    record: Mapping[str, Any], arm: Mapping[str, Any] | None, selected: Mapping[str, str | None]
) -> dict[str, bool]:
    complete = selected["trigger_url"] is not None and selected["action_url"] is not None
    model_valid = _model_protocol_valid(arm)
    within = _selection_within_candidates(record, arm, selected)
    fallback = _fallback_used(arm)
    return {
        "model_protocol_valid": model_valid,
        "complete_prediction": complete,
        "selection_within_candidates": within,
        "fallback_used": fallback,
        "conditioned": model_valid and complete and within and not fallback,
    }


def _rank_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("rank_metadata")
    return value if isinstance(value, Mapping) else record


def _rank_value(record: Mapping[str, Any], names: Sequence[str]) -> int | None:
    value = _first(_rank_metadata(record), names)
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid rank value: {value!r}") from error
    return integer if integer > 0 else None


def _coverage_state(record: Mapping[str, Any], k: int) -> dict[str, dict[str, bool]]:
    gold = _gold(record)
    trigger_order = _candidate_order(record, "trigger")
    action_order = _candidate_order(record, "action")

    if trigger_order:
        trigger_top = set(trigger_order[:k])
        function_trigger = bool(trigger_top & gold["trigger_urls"])
    else:
        rank = _rank_value(record, ("trigger_rank", "gold_trigger_rank"))
        function_trigger = rank is not None and rank <= k
        trigger_top = set()
    if action_order:
        action_top = set(action_order[:k])
        function_action = bool(action_top & gold["action_urls"])
    else:
        rank = _rank_value(record, ("action_rank", "gold_action_rank"))
        function_action = rank is not None and rank <= k
        action_top = set()

    if trigger_order and action_order:
        function_joint = any(
            trigger in trigger_top and action in action_top for trigger, action in gold["url_pairs"]
        )
    else:
        rank = _rank_value(record, ("joint_rank", "gold_joint_rank", "valid_pair_rank"))
        function_joint = rank is not None and rank <= k

    trigger_services = {
        service
        for candidate in _ranked_candidate_rows(record, "trigger")[:k]
        if (service := _candidate_service(candidate)) is not None
    }
    action_services = {
        service
        for candidate in _ranked_candidate_rows(record, "action")[:k]
        if (service := _candidate_service(candidate)) is not None
    }
    if trigger_services:
        service_trigger = bool(trigger_services & gold["trigger_services"])
    else:
        rank = _rank_value(record, ("trigger_service_rank", "gold_trigger_service_rank"))
        service_trigger = rank is not None and rank <= k
    if action_services:
        service_action = bool(action_services & gold["action_services"])
    else:
        rank = _rank_value(record, ("action_service_rank", "gold_action_service_rank"))
        service_action = rank is not None and rank <= k
    if trigger_services and action_services:
        service_joint = any(
            trigger in trigger_services and action in action_services
            for trigger, action in gold["service_pairs"]
        )
    else:
        rank = _rank_value(record, ("service_joint_rank", "gold_service_joint_rank"))
        service_joint = rank is not None and rank <= k

    return {
        "function": {
            "trigger": function_trigger,
            "action": function_action,
            "joint": function_joint,
        },
        "service": {
            "trigger": service_trigger,
            "action": service_action,
            "joint": service_joint,
        },
    }


def _joint_rank(record: Mapping[str, Any]) -> int | None:
    trigger_order = _candidate_order(record, "trigger")
    action_order = _candidate_order(record, "action")
    if trigger_order and action_order:
        trigger_ranks = {identifier: index for index, identifier in enumerate(trigger_order, start=1)}
        action_ranks = {identifier: index for index, identifier in enumerate(action_order, start=1)}
        return min(
            (
                max(trigger_ranks[trigger], action_ranks[action])
                for trigger, action in _gold(record)["url_pairs"]
                if trigger in trigger_ranks and action in action_ranks
            ),
            default=None,
        )
    return _rank_value(record, ("joint_rank", "gold_joint_rank", "valid_pair_rank"))


def _rank_bucket(record: Mapping[str, Any]) -> str:
    rank = _joint_rank(record)
    if rank == 1:
        return "rank1"
    if rank is not None and rank <= 5:
        return "rank2_5"
    if rank is not None and rank <= 10:
        return "rank6_10"
    if rank is None:
        supplied = _first(record, ("rank_bucket",), None)
        aliases = {
            "rank1": "rank1",
            "1": "rank1",
            "rank2_5": "rank2_5",
            "2-5": "rank2_5",
            "rank6_10": "rank6_10",
            "6-10": "rank6_10",
            "outside10": "outside10",
            ">10": "outside10",
        }
        if supplied is not None and str(supplied) in aliases:
            return aliases[str(supplied)]
    return "outside10"


def _candidate_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(records)
    buckets = {bucket: 0 for bucket in RANK_BUCKETS}
    ceilings: dict[str, Any] = {}
    for record in records:
        buckets[_rank_bucket(record)] += 1
    for k in K_VALUES:
        states = [_coverage_state(record, k) for record in records]
        by_level: dict[str, Any] = {}
        for level in ("function", "service"):
            trigger_hits = sum(state[level]["trigger"] for state in states)
            action_hits = sum(state[level]["action"] for state in states)
            joint_hits = sum(state[level]["joint"] for state in states)
            by_level[level] = {
                "denominator_cases": denominator,
                "trigger_hits": trigger_hits,
                "trigger_rate": trigger_hits / denominator,
                "action_hits": action_hits,
                "action_rate": action_hits / denominator,
                "joint_hits": joint_hits,
                "joint_rate": joint_hits / denominator,
            }
        ceilings[f"k{k}"] = by_level
    return {"denominator_cases": denominator, "rank_buckets": buckets, "ceilings": ceilings}


def _score_summary(scores: Sequence[Mapping[str, bool]], *, excluded_cases: int | None = None) -> dict[str, Any]:
    denominator = len(scores)
    summary: dict[str, Any] = {"denominator_cases": denominator}
    if excluded_cases is not None:
        summary["excluded_cases"] = excluded_cases
    for outcome in OUTCOMES:
        hits = sum(bool(row[outcome]) for row in scores)
        summary[outcome] = {"hits": hits, "rate": hits / denominator if denominator else None}
    return summary


def _transition(before: Sequence[Mapping[str, bool]], after: Sequence[Mapping[str, bool]]) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("transition states must be paired")
    result: dict[str, Any] = {}
    denominator = len(before)
    for outcome in OUTCOMES:
        left = [bool(row[outcome]) for row in before]
        right = [bool(row[outcome]) for row in after]
        before_hits = sum(left)
        after_hits = sum(right)
        recoveries = sum(not x and y for x, y in zip(left, right))
        regressions = sum(x and not y for x, y in zip(left, right))
        unchanged_correct = sum(x and y for x, y in zip(left, right))
        unchanged_wrong = sum(not x and not y for x, y in zip(left, right))
        result[outcome] = {
            "denominator_cases": denominator,
            "before_hits": before_hits,
            "after_hits": after_hits,
            "recoveries": recoveries,
            "regressions": regressions,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "delta": (after_hits - before_hits) / denominator if denominator else None,
        }
    return result


def _iteration_predictions(arm: Mapping[str, Any] | None) -> list[Any]:
    if arm is None:
        return []
    values = arm.get("iterations")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [value for value in values if isinstance(value, Mapping)]


def _transitions(
    records: Sequence[Mapping[str, Any]], arm_name: str, baseline_arm: str
) -> dict[str, Any]:
    baseline_scores: list[dict[str, bool]] = []
    final_scores: list[dict[str, bool]] = []
    per_stage: dict[int, list[tuple[dict[str, bool], dict[str, bool]]]] = defaultdict(list)
    consecutive: dict[int, list[tuple[dict[str, bool], dict[str, bool]]]] = defaultdict(list)
    for record in records:
        baseline = _arm(record, baseline_arm, baseline_arm)
        active = _arm(record, arm_name, baseline_arm)
        baseline_prediction = _prediction(record, baseline or {})
        final_prediction = _prediction(record, active or {})
        base_score = score_prediction(record, baseline_prediction)
        final_score = score_prediction(record, final_prediction)
        baseline_scores.append(base_score)
        final_scores.append(final_score)
        previous = base_score
        for index, iteration in enumerate(_iteration_predictions(active), start=1):
            current = score_prediction(record, iteration)
            per_stage[index].append((base_score, current))
            consecutive[index].append((previous, current))
            previous = current

    result: dict[str, Any] = {"baseline_to_final": _transition(baseline_scores, final_scores)}
    for index in sorted(per_stage):
        pairs = per_stage[index]
        result[f"baseline_to_iteration_{index}"] = _transition(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        )
    for index in sorted(consecutive):
        if index == 1:
            continue
        pairs = consecutive[index]
        result[f"iteration_{index - 1}_to_iteration_{index}"] = _transition(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        )
    return result


def _rate(
    numerator_name: str,
    numerator_value: int | float | None,
    denominator_name: str,
    denominator_value: int | float,
) -> dict[str, Any]:
    return {
        "numerator": numerator_name,
        "numerator_value": numerator_value,
        "denominator": denominator_name,
        "denominator_value": denominator_value,
        "value": (
            numerator_value / denominator_value
            if numerator_value is not None and denominator_value
            else None
        ),
    }


def _accounting(
    accounting_rows: Sequence[Mapping[str, int | float]],
    protocol_rows: Sequence[Mapping[str, bool]],
    routed: Sequence[bool],
) -> dict[str, Any]:
    latency_rows = [row for row in accounting_rows if bool(row["latency_observed"])]
    latency_total = (
        sum(float(row["latency_seconds"]) for row in latency_rows) if latency_rows else None
    )
    totals = {
        "logical_calls": sum(int(row["logical_calls"]) for row in accounting_rows),
        "api_attempts": sum(int(row["api_attempts"]) for row in accounting_rows),
        "tool_calls": sum(int(row["tool_calls"]) for row in accounting_rows),
        "catalog_reads": sum(int(row["catalog_reads"]) for row in accounting_rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in accounting_rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in accounting_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in accounting_rows),
        "latency_seconds": latency_total,
    }
    denominators = {
        "intent_to_treat_cases": len(accounting_rows),
        "routed_cases": sum(routed),
        "called_cases": sum(int(row["logical_calls"]) > 0 for row in accounting_rows),
        "model_protocol_valid_cases": sum(row["model_protocol_valid"] for row in protocol_rows),
        "protocol_conditioned_cases": sum(row["conditioned"] for row in protocol_rows),
        "logical_calls": totals["logical_calls"],
        "api_attempts": totals["api_attempts"],
        "tool_calls": totals["tool_calls"],
        "catalog_reads": totals["catalog_reads"],
        "latency_observed_cases": len(latency_rows),
        "latency_observed_called_cases": sum(
            int(row["logical_calls"]) > 0 for row in latency_rows
        ),
        "latency_observed_logical_calls": sum(
            int(row["logical_calls"]) for row in latency_rows
        ),
    }
    rates = {
        "logical_calls_per_itt_case": _rate(
            "logical_calls",
            totals["logical_calls"],
            "intent_to_treat_cases",
            denominators["intent_to_treat_cases"],
        ),
        "logical_calls_per_routed_case": _rate(
            "logical_calls", totals["logical_calls"], "routed_cases", denominators["routed_cases"]
        ),
        "api_attempts_per_itt_case": _rate(
            "api_attempts",
            totals["api_attempts"],
            "intent_to_treat_cases",
            denominators["intent_to_treat_cases"],
        ),
        "api_attempts_per_logical_call": _rate(
            "api_attempts", totals["api_attempts"], "logical_calls", totals["logical_calls"]
        ),
        "tool_calls_per_logical_call": _rate(
            "tool_calls", totals["tool_calls"], "logical_calls", totals["logical_calls"]
        ),
        "catalog_reads_per_itt_case": _rate(
            "catalog_reads",
            totals["catalog_reads"],
            "intent_to_treat_cases",
            denominators["intent_to_treat_cases"],
        ),
        "catalog_reads_per_logical_call": _rate(
            "catalog_reads",
            totals["catalog_reads"],
            "logical_calls",
            totals["logical_calls"],
        ),
        "tokens_per_itt_case": _rate(
            "total_tokens",
            totals["total_tokens"],
            "intent_to_treat_cases",
            denominators["intent_to_treat_cases"],
        ),
        "tokens_per_logical_call": _rate(
            "total_tokens", totals["total_tokens"], "logical_calls", totals["logical_calls"]
        ),
        "latency_seconds_per_itt_case": _rate(
            "latency_seconds",
            totals["latency_seconds"],
            "latency_observed_cases",
            denominators["latency_observed_cases"],
        ),
        "latency_seconds_per_called_case": _rate(
            "latency_seconds",
            totals["latency_seconds"],
            "latency_observed_called_cases",
            denominators["latency_observed_called_cases"],
        ),
        "latency_seconds_per_logical_call": _rate(
            "latency_seconds",
            totals["latency_seconds"],
            "latency_observed_logical_calls",
            denominators["latency_observed_logical_calls"],
        ),
    }
    return {"denominators": denominators, "totals": totals, "rates": rates}


def _membership_rows(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, str | None]],
    dimension: str,
) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for index, (record, selected) in enumerate(zip(records, predictions)):
        gold = _gold(record)
        case_id = _case_id(record, index)
        if dimension == "trigger":
            for service in gold["trigger_services"]:
                function_hit = any(
                    trigger == selected["trigger_url"] and trigger_service == service
                    for trigger, _action, trigger_service, _action_service in gold["resolved_pairs"]
                )
                groups[service].append(
                    {
                        "case_id": case_id,
                        "service_hit": selected["trigger_service"] == service,
                        "function_hit": function_hit,
                    }
                )
        elif dimension == "action":
            for service in gold["action_services"]:
                function_hit = any(
                    action == selected["action_url"] and action_service == service
                    for _trigger, action, _trigger_service, action_service in gold["resolved_pairs"]
                )
                groups[service].append(
                    {
                        "case_id": case_id,
                        "service_hit": selected["action_service"] == service,
                        "function_hit": function_hit,
                    }
                )
        else:
            for service_pair in gold["service_pairs"]:
                function_hit = any(
                    trigger == selected["trigger_url"]
                    and action == selected["action_url"]
                    and trigger_service == service_pair[0]
                    and action_service == service_pair[1]
                    for trigger, action, trigger_service, action_service in gold["resolved_pairs"]
                )
                groups[service_pair].append(
                    {
                        "case_id": case_id,
                        "service_hit": (
                            selected["trigger_service"], selected["action_service"]
                        )
                        == service_pair,
                        "function_hit": function_hit,
                    }
                )
    return groups


def _membership_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support = len(rows)
    service_hits = sum(bool(row["service_hit"]) for row in rows)
    function_hits = sum(bool(row["function_hit"]) for row in rows)
    return {
        "support_memberships": support,
        "unique_cases": len({row["case_id"] for row in rows}),
        "service_exact_hits": service_hits,
        "service_exact_rate": service_hits / support if support else None,
        "function_exact_hits": function_hits,
        "function_exact_rate": function_hits / support if support else None,
    }


def _service_dimension(
    groups: Mapping[Any, Sequence[Mapping[str, Any]]], *, minimum_support: int, dimension: str
) -> dict[str, Any]:
    reported: list[dict[str, Any]] = []
    tail: list[Mapping[str, Any]] = []
    withheld = 0
    for group in sorted(groups, key=lambda value: str(value)):
        rows = list(groups[group])
        if len(rows) < minimum_support:
            withheld += 1
            tail.extend(rows)
            continue
        summary = _membership_summary(rows)
        entry = {
            "support_cases": summary["support_memberships"],
            "service_exact_hits": summary["service_exact_hits"],
            "service_exact_rate": summary["service_exact_rate"],
            "function_exact_hits": summary["function_exact_hits"],
            "function_exact_rate": summary["function_exact_rate"],
        }
        if dimension == "pair":
            entry = {
                "trigger_service": group[0],
                "action_service": group[1],
                **entry,
            }
        else:
            entry = {"service": group, **entry}
        reported.append(entry)
    return {
        "reported": reported,
        "withheld_group_count": withheld,
        "pooled_long_tail": _membership_summary(tail),
    }


def _per_service(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, str | None]],
    minimum_support: int,
) -> dict[str, Any]:
    return {
        "minimum_support": minimum_support,
        "support_unit": "case-service membership; multi-gold cases may belong to multiple groups",
        "trigger_services": _service_dimension(
            _membership_rows(records, predictions, "trigger"),
            minimum_support=minimum_support,
            dimension="trigger",
        ),
        "action_services": _service_dimension(
            _membership_rows(records, predictions, "action"),
            minimum_support=minimum_support,
            dimension="action",
        ),
        "service_pairs": _service_dimension(
            _membership_rows(records, predictions, "pair"),
            minimum_support=minimum_support,
            dimension="pair",
        ),
    }


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str = "baseline",
    arm_names: Sequence[str] | None = None,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 42,
    min_service_support: int = 10,
) -> dict[str, Any]:
    """Evaluate resolved FARM agent records without performing resolution.

    Intent-to-treat summaries retain every input case.  Protocol-conditioned
    summaries include only cases whose model protocol was valid, whose resolved
    prediction was complete and inside both supplied candidate sets, and which
    did not use a fallback.
    """
    rows = list(records)
    if not rows:
        raise ValueError("at least one evaluation record is required")
    if min_service_support < 1:
        raise ValueError("min_service_support must be positive")
    case_ids = [_case_id(record, index) for index, record in enumerate(rows)]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case identifiers must be unique")

    # Validate all gold sets before producing any partial report.
    for record in rows:
        _gold(record)

    inferred = _infer_arm_names(rows, baseline_arm)
    names = list(arm_names) if arm_names is not None else inferred
    if baseline_arm not in names:
        names.insert(0, baseline_arm)
    if len(names) != len(set(names)):
        raise ValueError("arm names must be unique")

    report: dict[str, Any] = {
        "schema_version": "farm_agent_metrics_v1",
        "denominator_cases": len(rows),
        "baseline_arm": baseline_arm,
        "candidate_coverage": _candidate_coverage(rows),
        "arms": {},
        "comparisons": {},
        "multiplicity": {
            "method": "Holm step-down",
            "family": "all reported non-baseline arms x six exact outcomes",
        },
    }
    score_vectors: dict[str, dict[str, list[int]]] = {}

    for name in names:
        predictions: list[dict[str, str | None]] = []
        scores: list[dict[str, bool]] = []
        protocol_rows: list[dict[str, bool]] = []
        accounting_rows: list[dict[str, float | int]] = []
        routed_rows: list[bool] = []
        by_bucket: dict[str, list[dict[str, bool]]] = defaultdict(list)
        for record in rows:
            active = _arm(record, name, baseline_arm)
            selected = _prediction(record, active or {})
            state = _protocol_state(record, active, selected)
            scored = score_prediction(record, selected)
            accounting = _case_accounting(active)
            routed = (
                bool(active.get("routed"))
                if active and "routed" in active
                else int(accounting["logical_calls"]) > 0
            )
            predictions.append(selected)
            scores.append(scored)
            protocol_rows.append(state)
            accounting_rows.append(accounting)
            routed_rows.append(routed)
            by_bucket[_rank_bucket(record)].append(scored)

        conditioned = [score for score, state in zip(scores, protocol_rows) if state["conditioned"]]
        protocol = {
            "model_protocol_valid_cases": sum(state["model_protocol_valid"] for state in protocol_rows),
            "model_protocol_invalid_cases": sum(not state["model_protocol_valid"] for state in protocol_rows),
            "complete_prediction_cases": sum(state["complete_prediction"] for state in protocol_rows),
            "selection_within_candidates_cases": sum(
                state["selection_within_candidates"] for state in protocol_rows
            ),
            "selection_outside_candidates_cases": sum(
                not state["selection_within_candidates"] for state in protocol_rows
            ),
            "fallback_cases": sum(state["fallback_used"] for state in protocol_rows),
            "conditioned_cases": sum(state["conditioned"] for state in protocol_rows),
        }
        report["arms"][name] = {
            "intent_to_treat": _score_summary(scores),
            "protocol_conditioned": _score_summary(
                conditioned, excluded_cases=len(rows) - len(conditioned)
            ),
            "protocol": protocol,
            "service_stage": _service_stage_summary(rows, name, baseline_arm),
            "by_rank_bucket": {
                bucket: _score_summary(by_bucket.get(bucket, [])) for bucket in RANK_BUCKETS
            },
            "transitions": _transitions(rows, name, baseline_arm),
            "accounting": _accounting(accounting_rows, protocol_rows, routed_rows),
            "per_service": _per_service(rows, predictions, min_service_support),
        }
        score_vectors[name] = {
            outcome: [int(score[outcome]) for score in scores] for outcome in OUTCOMES
        }

    p_values: dict[str, float] = {}
    for name in names:
        if name == baseline_arm:
            continue
        comparison_name = f"{name}_vs_{baseline_arm}"
        outcomes: dict[str, Any] = {}
        for outcome in OUTCOMES:
            before = score_vectors[baseline_arm][outcome]
            after = score_vectors[name][outcome]
            mcnemar = exact_mcnemar(before, after)
            label = f"{comparison_name}:{outcome}"
            bootstrap = paired_bootstrap_delta(
                before,
                after,
                seed=_stable_seed(bootstrap_seed, label),
                iterations=bootstrap_iterations,
            )
            outcomes[outcome] = {
                "denominator_cases": len(rows),
                "baseline_rate": sum(before) / len(before),
                "arm_rate": sum(after) / len(after),
                "delta": (sum(after) - sum(before)) / len(before),
                "mcnemar": mcnemar,
                "paired_bootstrap": bootstrap,
            }
            p_values[label] = float(mcnemar["exact_two_sided_p"])
        report["comparisons"][comparison_name] = {"outcomes": outcomes}

    adjusted = holm_adjust(p_values)
    family_size = len(p_values)
    for comparison_name, comparison in report["comparisons"].items():
        for outcome, result in comparison["outcomes"].items():
            label = f"{comparison_name}:{outcome}"
            result["raw_p"] = p_values[label]
            result["holm_adjusted_p"] = adjusted[label]
            result["holm_family_size"] = family_size
    report["multiplicity"]["family_size"] = family_size
    return report


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("records", "rows", "rows_detail"):
            rows = value.get(key)
            if isinstance(rows, list):
                return rows
    raise ValueError("input JSON must be a record list or contain records/rows/rows_detail")


def _write_json_atomic(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="resolved JSON or JSONL traces")
    parser.add_argument("--output", type=Path, required=True, help="new metrics JSON")
    parser.add_argument("--baseline-arm", default="baseline")
    parser.add_argument("--arm", dest="arms", action="append", help="arm to report; repeat as needed")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--min-service-support", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.input)
    arms = None
    if args.arms:
        arms = [args.baseline_arm, *[arm for arm in args.arms if arm != args.baseline_arm]]
    report = evaluate_records(
        records,
        baseline_arm=args.baseline_arm,
        arm_names=arms,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        min_service_support=args.min_service_support,
    )
    _write_json_atomic(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "complete",
                "cases": report["denominator_cases"],
                "arms": list(report["arms"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
