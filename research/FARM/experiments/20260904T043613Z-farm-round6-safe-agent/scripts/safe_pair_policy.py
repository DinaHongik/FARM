#!/usr/bin/env python3
"""Pure routed pair-card policy with a true binary verifier.

The proposer may abstain. A non-baseline proposal is compared only with the
retrieval baseline; no arbitrary third card can become the final answer.
Failures, abstention, disagreement, or a baseline vote all retain top1.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence


View = Literal["plain", "schema"]
Mode = Literal["binary_single", "binary_dual"]


class Chooser(Protocol):
    def select(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SafePairPolicy:
    card_count: Literal[5, 10]
    proposer_view: View
    verifier_view: View = "schema"
    verification: Mode = "binary_dual"
    candidate_depth: int = 10

    def __post_init__(self) -> None:
        if self.card_count not in {5, 10}:
            raise ValueError("card_count must be 5 or 10")
        if self.proposer_view not in {"plain", "schema"} or self.verifier_view not in {"plain", "schema"}:
            raise ValueError("unsupported evidence view")
        if self.verification not in {"binary_single", "binary_dual"}:
            raise ValueError("unsupported verification mode")
        if self.candidate_depth != 10:
            raise ValueError("candidate depth is frozen at ten")


REFERENCE_KEYS = {
    "valid_pairs", "gold", "gold_pair", "gold_pairs", "gold_trigger_urls",
    "gold_action_urls", "gold_channel_pairs", "gold_service_pairs",
    "ground_truth", "reference", "references", "reference_answer",
    "expected_pair", "observed_pairs", "pair_rank",
}


def reference_paths(value: Any, path: str = "case") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            child_path = f"{path}.{raw_key}"
            if key in REFERENCE_KEYS or key.startswith(("gold_", "reference_", "ground_truth_")):
                found.append(child_path)
            found.extend(reference_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(reference_paths(child, f"{path}[{index}]"))
    return found


def _text(item: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"candidate lacks any of {keys}")


def _candidate(item: Mapping[str, Any], side: str, view: View) -> dict[str, Any]:
    identity = _text(item, "url", "id")
    service = _text(item, "service_name", "channel_display", "service_display", "channel", "service")
    function = _text(item, "function_name", "name", "display")
    evidence = _text(item, f"text_{view}", view, "text", "description")
    return {
        "identity": identity,
        "side": side,
        "rank": int(item.get("retrieval_rank", 0)),
        "public": {"service_name": service, "function_name": function, "evidence": evidence},
    }


def _parse_pair(value: Mapping[str, Any] | str) -> tuple[str, str]:
    if isinstance(value, str):
        pieces = value.split(" || ")
        if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
            raise ValueError("invalid pair-ranking string")
        return pieces[0].strip(), pieces[1].strip()
    if not isinstance(value, Mapping) or set(value) != {"trigger_url", "action_url"}:
        raise ValueError("pair ranking requires trigger_url/action_url only")
    return _text(value, "trigger_url"), _text(value, "action_url")


def _build_cards(
    case: Mapping[str, Any], ranking: Sequence[Mapping[str, Any] | str], policy: SafePairPolicy
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_id = _text(case, "group_id", "case_id", "id")
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for side in ("trigger", "action"):
        raw = case.get(f"{side}_candidates")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < policy.candidate_depth:
            raise ValueError(f"{side} requires ten candidates")
        normalized = [_candidate(item, side, policy.proposer_view) for item in raw[: policy.candidate_depth]]
        if len({item["identity"] for item in normalized}) != len(normalized):
            raise ValueError(f"duplicate {side} candidate")
        normalized.sort(key=lambda item: (item["rank"], item["identity"]))
        candidates[side] = {item["identity"]: item for item in normalized}
    baseline_pair = (
        min(candidates["trigger"].values(), key=lambda item: (item["rank"], item["identity"]))["identity"],
        min(candidates["action"].values(), key=lambda item: (item["rank"], item["identity"]))["identity"],
    )
    selected: list[tuple[str, str]] = []
    for pair in (baseline_pair, *(_parse_pair(item) for item in ranking)):
        if pair[0] not in candidates["trigger"] or pair[1] not in candidates["action"]:
            raise ValueError("ranking escaped frozen top-ten lattice")
        if pair not in selected:
            selected.append(pair)
        if len(selected) == policy.card_count:
            break
    if len(selected) != policy.card_count:
        raise ValueError("ranking cannot fill the declared card deck")
    presentation = sorted(
        selected,
        key=lambda pair: hashlib.sha256(
            f"farm-r6-order\0{case_id}\0{pair[0]}\0{pair[1]}".encode()
        ).hexdigest(),
    )
    aliases = {pair: f"C{index + 1:02d}" for index, pair in enumerate(presentation)}
    cards: list[dict[str, Any]] = []
    for pair in presentation:
        cards.append({
            "card_id": aliases[pair],
            "pair": {"trigger_url": pair[0], "action_url": pair[1]},
            "trigger": candidates["trigger"][pair[0]],
            "action": candidates["action"][pair[1]],
        })
    baseline = next(card for card in cards if (card["pair"]["trigger_url"], card["pair"]["action_url"]) == baseline_pair)
    return cards, baseline


def _public_card(card: Mapping[str, Any], view: View, case: Mapping[str, Any]) -> dict[str, Any]:
    result = {"card_id": card["card_id"]}
    for side in ("trigger", "action"):
        identity = card["pair"][f"{side}_url"]
        raw = next(item for item in case[f"{side}_candidates"] if item["url"] == identity)
        result[side] = {
            "service_name": _text(raw, "service_name", "channel_display", "service_display", "channel", "service"),
            "function_name": _text(raw, "function_name", "name", "display"),
            "evidence": _text(raw, f"text_{view}", view, "text", "description"),
        }
    return result


def _request(
    case: Mapping[str, Any], cards: Sequence[Mapping[str, Any]], *, phase: str,
    view: View, allow_abstain: bool, instruction: str,
) -> dict[str, Any]:
    return {
        "case_id": _text(case, "group_id", "case_id", "id"),
        "query": _text(case, "query"),
        "phase": phase,
        "instruction": instruction,
        "evidence_view": view,
        "allow_abstain": allow_abstain,
        "cards": [_public_card(card, view, case) for card in cards],
    }


def _invoke(chooser: Chooser, request: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    allowed = {card["card_id"] for card in request["cards"]}
    if request["allow_abstain"]:
        allowed.add("ABSTAIN")
    try:
        raw = chooser.select(request)
    except Exception as exc:
        raw = {"ok": False, "choice_id": None, "api_attempts": 0, "tool_calls": 0,
               "usage": {}, "error": f"adapter_exception:{type(exc).__name__}"}
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    attempts, tools = raw.get("api_attempts"), raw.get("tool_calls")
    counts_valid = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (attempts, tools))
    choice = raw.get("choice_id")
    ok = raw.get("ok") is True and counts_valid and isinstance(choice, str) and choice in allowed
    call = {
        "phase": request["phase"], "request": request, "ok": ok,
        "choice_id": choice if ok else None,
        "api_attempts": attempts if isinstance(attempts, int) else 0,
        "tool_calls": tools if isinstance(tools, int) else 0,
        "usage": dict(raw.get("usage") or {}) if isinstance(raw.get("usage"), Mapping) else {},
        "error": None if ok else str(raw.get("error") or "adapter_contract_failure")[:160],
        "accounting_complete": counts_valid,
    }
    return call, call["choice_id"]


def zero_accounting() -> dict[str, Any]:
    return {
        "logical_calls": 0, "api_attempts": 0, "tool_calls": 0, "catalog_reads": 0,
        "complete": True,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_seconds": 0.0},
        "evidence_presentations": {
            "proposer_pair_cards": 0, "proposer_endpoints": 0,
            "verifier_pair_cards": 0, "verifier_endpoints": 0,
            "verifier_unique_schema_catalog_rows": 0,
        },
    }


def _accounting(calls: Sequence[Mapping[str, Any]], card_count: int) -> dict[str, Any]:
    result = zero_accounting()
    result["logical_calls"] = len(calls)
    result["api_attempts"] = sum(int(call.get("api_attempts") or 0) for call in calls)
    result["tool_calls"] = sum(int(call.get("tool_calls") or 0) for call in calls)
    result["complete"] = all(bool(call.get("accounting_complete")) for call in calls)
    usage: dict[str, int | float] = result["usage"]
    for call in calls:
        for key, value in (call.get("usage") or {}).items():
            if key in usage and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                usage[key] += value
    verifier_calls = sum(call.get("phase") == "pair_verification" for call in calls)
    result["catalog_reads"] = 4 if verifier_calls else 0
    result["evidence_presentations"] = {
        "proposer_pair_cards": card_count,
        "proposer_endpoints": 2 * card_count,
        "verifier_pair_cards": 2 * verifier_calls,
        "verifier_endpoints": 4 * verifier_calls,
        "verifier_unique_schema_catalog_rows": 4 if verifier_calls else 0,
    }
    return result


def unrouted_trace(case: Mapping[str, Any], *, routing_score: float) -> dict[str, Any]:
    baseline = {
        "trigger_url": case["trigger_candidates"][0]["url"],
        "action_url": case["action_candidates"][0]["url"],
    }
    return {
        "schema_version": "farm_round6_safe_pair_trace_v1", "case_id": case["group_id"],
        "routed": False, "routing_score": routing_score,
        "policy": {"card_count": 0, "candidate_depth": 10, "safe_fallback": "retrieval_top1"},
        "baseline_pair": baseline, "proposer_pair": None, "alternative_pair": None,
        "final_pair": baseline, "retained_baseline": True, "card_ids": [],
        "candidate_pairs": [], "verifier_decisions": [], "stable_decision": "NOT_ROUTED",
        "fallback_reason": None, "calls": [], "accounting": zero_accounting(),
    }


def resolve(
    case: Mapping[str, Any], ranking: Sequence[Mapping[str, Any] | str],
    policy: SafePairPolicy, chooser: Chooser, *, routing_score: float,
) -> dict[str, Any]:
    leaked = reference_paths(case)
    if leaked:
        raise ValueError(f"reference fields crossed inference seam: {sorted(leaked)}")
    cards, baseline = _build_cards(case, ranking, policy)
    proposer_request = _request(
        case, cards, phase="pair_proposal", view=policy.proposer_view, allow_abstain=True,
        instruction=("Choose the visible complete pair that best and most exactly implements "
                     "the request. Use ABSTAIN only if every supplied pair is incompatible; "
                     "do not abstain merely because two plausible pairs are close."),
    )
    proposer_call, choice = _invoke(chooser, proposer_request)
    calls = [proposer_call]
    proposal = next((card for card in cards if card["card_id"] == choice), None)
    final, stable, fallback = baseline, "KEEP_TOP1", None
    decisions: list[str] = []
    if choice is None:
        stable, fallback = "PROPOSER_FAILURE", "proposer_protocol_failure"
    elif choice == "ABSTAIN":
        stable, fallback = "PROPOSER_ABSTAIN", "proposer_abstained"
    elif proposal is baseline:
        stable = "KEEP_TOP1"
    else:
        pairwise = [baseline, proposal]
        presentations = [pairwise]
        if policy.verification == "binary_dual":
            presentations.append(list(reversed(pairwise)))
        votes: list[str | None] = []
        for presented in presentations:
            request = _request(
                case, presented, phase="pair_verification", view=policy.verifier_view,
                allow_abstain=True,
                instruction=("Compare only these two complete pairs. Select the challenger only "
                             "when its trigger and action are both clearly better supported; otherwise "
                             "select the other card or ABSTAIN."),
            )
            call, vote = _invoke(chooser, request)
            calls.append(call)
            votes.append(vote)
            if vote == baseline["card_id"]:
                decisions.append("KEEP_TOP1")
            elif vote == proposal["card_id"]:
                decisions.append("ACCEPT_PROPOSAL")
            elif vote == "ABSTAIN":
                decisions.append("ABSTAIN")
            else:
                decisions.append("INVALID")
        if votes and all(vote == proposal["card_id"] for vote in votes):
            final, stable = proposal, "ACCEPT_PROPOSAL"
        elif any(vote is None for vote in votes):
            stable, fallback = "VERIFIER_FAILURE", "verifier_protocol_failure"
        elif any(vote == "ABSTAIN" for vote in votes):
            stable, fallback = "VERIFIER_ABSTAIN", "verifier_abstained"
        elif len(set(votes)) > 1:
            stable, fallback = "VERIFIER_DISAGREE", "verifier_disagreement"
        else:
            stable = "KEEP_TOP1"
    return {
        "schema_version": "farm_round6_safe_pair_trace_v1", "case_id": case["group_id"],
        "routed": True, "routing_score": routing_score,
        "policy": {
            "card_count": policy.card_count, "candidate_depth": policy.candidate_depth,
            "proposer_view": policy.proposer_view, "verifier_view": policy.verifier_view,
            "verification": policy.verification, "safe_fallback": "retrieval_top1",
        },
        "baseline_pair": baseline["pair"],
        "proposer_pair": proposal["pair"] if proposal is not None else None,
        "alternative_pair": None, "final_pair": final["pair"],
        "retained_baseline": final is baseline,
        "card_ids": [card["card_id"] for card in cards],
        "candidate_pairs": [card["pair"] for card in cards],
        "verifier_decisions": decisions, "stable_decision": stable,
        "fallback_reason": fallback, "calls": calls,
        "accounting": _accounting(calls, policy.card_count),
    }


__all__ = ["SafePairPolicy", "reference_paths", "resolve", "unrouted_trace", "zero_accounting"]
