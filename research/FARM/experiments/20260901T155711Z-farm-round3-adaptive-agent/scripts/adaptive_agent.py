#!/usr/bin/env python3
"""Stateless, bounded top-k agent policy and auditable call accounting."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Protocol


class PairSelectionClient(Protocol):
    """Injected external-model port; tests replace only this boundary."""

    def select_pair(self, request: dict) -> dict: ...


@dataclass(frozen=True)
class AdaptivePolicy:
    view: str = "schema"
    second_view: str | None = None
    first_k: int = 5
    max_k: int = 10
    confidence_threshold: float = 0.55
    verify_overrides: bool = False

    def __post_init__(self) -> None:
        if self.view not in {"names", "plain", "schema"}:
            raise ValueError("agent view must be names, plain, or schema")
        if self.second_view is not None and self.second_view not in {"names", "plain", "schema"}:
            raise ValueError("second agent view must be names, plain, or schema")
        if not 1 <= self.first_k < self.max_k:
            raise ValueError("policy requires 1 <= first_k < max_k")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0,1]")


def _public_candidate(item: dict, view: str) -> dict:
    base = {
        "url": item["url"],
        "service": item["channel"],
        "retrieval_rank": int(item["retrieval_rank"]),
    }
    if view in {"plain", "schema"}:
        base["function_name"] = item["function_name"]
        base["evidence"] = item[f"text_{view}"]
    return base


def _request(
    row: dict,
    policy: AdaptivePolicy,
    *,
    iteration: int,
    start: int,
    stop: int,
    prior: dict | None = None,
    view: str | None = None,
    baseline: dict | None = None,
) -> dict:
    selected_view = view or policy.view
    ranks = list(range(start + 1, stop + 1))
    request = {
        "case_id": row["group_id"],
        "query": row["query"],
        "iteration": iteration,
        "view": selected_view,
        "candidate_ranks": ranks,
        "trigger_candidates": [
            _public_candidate(item, selected_view)
            for item in row["trigger_candidates"][start:stop]
        ],
        "action_candidates": [
            _public_candidate(item, selected_view)
            for item in row["action_candidates"][start:stop]
        ],
        "instruction": (
            "Select trigger and action jointly. Use supplied exact URLs only. "
            "Set request_more only when the displayed batch is insufficient."
        ),
    }
    if prior:
        request["prior_proposal"] = prior
        request["instruction"] += " You may retain either URL from the prior proposal."
    if baseline:
        request["baseline_proposal"] = baseline
        request["instruction"] += " You may revert either URL to the non-agentic baseline."
    return request


def _normalize_selection(value: dict) -> dict:
    trigger = value.get("trigger_url")
    action = value.get("action_url")
    confidence = value.get("confidence", 0.0)
    if not isinstance(trigger, str) or not isinstance(action, str):
        raise ValueError("model selection lacks exact trigger/action URLs")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("model selection confidence is not numeric") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("model selection confidence is outside [0,1]")
    return {
        "trigger_url": trigger,
        "action_url": action,
        "confidence": confidence,
        "request_more": bool(value.get("request_more", False)),
        "usage": value.get("usage") or {},
        "api_attempts": int(value.get("api_attempts") or 1),
        "protocol_fallback": value.get("protocol_fallback"),
    }


def _proposal_with_evidence(row: dict, pair: dict, view: str) -> dict:
    proposal = dict(pair)
    for side in ("trigger", "action"):
        url = pair[f"{side}_url"]
        candidate = next(
            (item for item in row[f"{side}_candidates"] if item["url"] == url),
            None,
        )
        if candidate is None:
            raise ValueError(f"{side} proposal is outside the frozen candidate lattice")
        proposal[f"{side}_candidate"] = _public_candidate(candidate, view)
    return proposal


def _selection_allowed(selection: dict, request: dict) -> bool:
    trigger_ids = {item["url"] for item in request["trigger_candidates"]}
    action_ids = {item["url"] for item in request["action_candidates"]}
    prior = request.get("prior_proposal") or {}
    baseline = request.get("baseline_proposal") or {}
    if isinstance(prior.get("trigger_url"), str):
        trigger_ids.add(prior["trigger_url"])
    if isinstance(prior.get("action_url"), str):
        action_ids.add(prior["action_url"])
    if isinstance(baseline.get("trigger_url"), str):
        trigger_ids.add(baseline["trigger_url"])
    if isinstance(baseline.get("action_url"), str):
        action_ids.add(baseline["action_url"])
    return selection["trigger_url"] in trigger_ids and selection["action_url"] in action_ids


def run_adaptive_case(row: dict, policy: AdaptivePolicy, client: PairSelectionClient) -> dict:
    """Run at most two fresh-context calls; this function never receives gold."""
    forbidden = {"valid_pairs", "gold_trigger_urls", "gold_action_urls"} & set(row)
    if forbidden:
        raise ValueError(f"inference row contains reference fields: {sorted(forbidden)}")
    if len(row["trigger_candidates"]) < policy.max_k or len(row["action_candidates"]) < policy.max_k:
        raise ValueError("adaptive policy requires max_k frozen candidates per side")

    first_request = _request(
        row, policy, iteration=1, start=0, stop=policy.first_k,
    )
    first = _normalize_selection(client.select_pair(first_request))
    if not _selection_allowed(first, first_request):
        raise ValueError("iteration-one selection is outside its frozen batch")
    iterations = [{"request": first_request, "selection": first}]
    final = first

    baseline = {
        "trigger_url": row["trigger_candidates"][0]["url"],
        "action_url": row["action_candidates"][0]["url"],
    }
    proposed = {"trigger_url": first["trigger_url"], "action_url": first["action_url"]}
    expand = (
        first["request_more"]
        or first["confidence"] < policy.confidence_threshold
        or (policy.verify_overrides and proposed != baseline)
    )
    if expand:
        second_view = policy.second_view or policy.view
        prior = _proposal_with_evidence(
            row,
            {"trigger_url": first["trigger_url"], "action_url": first["action_url"]},
            second_view,
        )
        hydrated_baseline = _proposal_with_evidence(row, baseline, second_view)
        second_request = _request(
            row, policy, iteration=2, start=policy.first_k, stop=policy.max_k, prior=prior,
            view=second_view, baseline=hydrated_baseline,
        )
        second = _normalize_selection(client.select_pair(second_request))
        if not _selection_allowed(second, second_request):
            raise ValueError("iteration-two selection is outside new batch plus prior proposal")
        iterations.append({"request": second_request, "selection": second})
        final = second

    return {
        "case_id": row["group_id"],
        "final_pair": {"trigger_url": final["trigger_url"], "action_url": final["action_url"]},
        "final_confidence": final["confidence"],
        "calls": len(iterations),
        "api_attempts": sum(item["selection"]["api_attempts"] for item in iterations),
        "iterations": iterations,
        "stateless_between_iterations": True,
    }


def run_one_shot_case(row: dict, *, view: str, client: PairSelectionClient, k: int = 5) -> dict:
    """Run exactly one fresh-context joint selection call over the first k candidates."""
    forbidden = {"valid_pairs", "gold_trigger_urls", "gold_action_urls"} & set(row)
    if forbidden:
        raise ValueError(f"inference row contains reference fields: {sorted(forbidden)}")
    policy = AdaptivePolicy(view=view, first_k=k, max_k=max(k + 1, 10), confidence_threshold=0.0)
    request = _request(row, policy, iteration=1, start=0, stop=k)
    selection = _normalize_selection(client.select_pair(request))
    if not _selection_allowed(selection, request):
        raise ValueError("one-shot selection is outside its frozen batch")
    return {
        "case_id": row["group_id"],
        "final_pair": {
            "trigger_url": selection["trigger_url"],
            "action_url": selection["action_url"],
        },
        "final_confidence": selection["confidence"],
        "calls": 1,
        "api_attempts": selection["api_attempts"],
        "iterations": [{"request": request, "selection": selection}],
        "stateless_between_iterations": True,
    }


def score_agent_trace(trace: dict, valid_pairs: list[dict]) -> dict:
    """Attach references only after inference has terminated."""
    valid = {(row["trigger_url"], row["action_url"]) for row in valid_pairs}
    pair = trace["final_pair"]
    trigger, action = pair["trigger_url"], pair["action_url"]
    iteration_exact = [
        (
            iteration["selection"]["trigger_url"],
            iteration["selection"]["action_url"],
        ) in valid
        for iteration in trace.get("iterations", [])
    ]
    return {
        "trigger_exact": any(trigger == gold_trigger for gold_trigger, _ in valid),
        "action_exact": any(action == gold_action for _, gold_action in valid),
        "exact_pair": (trigger, action) in valid,
        "iteration_exact_pair": iteration_exact,
    }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _group_summary(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        label: {
            "cases": len(items),
            "calls": sum(int(item.get("calls", 0)) for item in items),
            "accuracy": sum(bool(item.get("exact_pair")) for item in items) / len(items),
        }
        for label, items in sorted(grouped.items())
    }


def summarize_agent_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty agent result")
    calls = [int(row.get("calls", 0)) for row in rows]
    routed = [row for row in rows if int(row.get("calls", 0)) > 0]
    iteration_flags = [row.get("iteration_exact_pair") or [] for row in rows]
    iteration_summary = {}
    for index in range(2):
        observed = [flags[index] for flags in iteration_flags if len(flags) > index]
        iteration_summary[f"iteration_{index + 1}"] = {
            "cases": len(observed),
            "exact": sum(bool(value) for value in observed),
            "accuracy": sum(bool(value) for value in observed) / len(observed) if observed else None,
        }
    transitions = [flags[:2] for flags in iteration_flags if len(flags) >= 2]
    fallback_codes = [
        iteration["selection"].get("protocol_fallback")
        for row in rows
        for iteration in row.get("iterations", [])
        if iteration["selection"].get("protocol_fallback")
    ]
    iteration_summary["second_call_transition"] = {
        "cases": len(transitions),
        "recoveries": sum(not first and second for first, second in transitions),
        "regressions": sum(first and not second for first, second in transitions),
        "unchanged_correct": sum(first and second for first, second in transitions),
        "unchanged_wrong": sum(not first and not second for first, second in transitions),
        "net_recoveries": sum(int(second) - int(first) for first, second in transitions),
    }
    return {
        "cases": len(rows),
        "routed_cases": len(routed),
        "total_calls": sum(calls),
        "calls_per_case": sum(calls) / len(rows),
        "calls_per_routed_case": sum(calls) / len(routed) if routed else 0.0,
        "median_calls": float(median(calls)),
        "p95_calls": _percentile(calls, 0.95),
        "max_calls": max(calls),
        "zero_call_fraction": sum(value == 0 for value in calls) / len(rows),
        "protocol_fallbacks": len(fallback_codes),
        "protocol_fallbacks_by_code": {
            code: fallback_codes.count(code) for code in sorted(set(fallback_codes))
        },
        "micro_exact_pair": sum(bool(row.get("exact_pair")) for row in rows) / len(rows),
        "by_trigger_service": _group_summary(rows, "trigger_service"),
        "by_action_service": _group_summary(rows, "action_service"),
        "by_service_pair": _group_summary(
            [row | {"service_pair": f"{row['trigger_service']} || {row['action_service']}"} for row in rows],
            "service_pair",
        ),
        "iterations": iteration_summary,
    }
