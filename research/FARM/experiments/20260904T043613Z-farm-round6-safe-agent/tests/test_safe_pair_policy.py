from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from safe_pair_policy import SafePairPolicy, resolve, unrouted_trace  # noqa: E402


def candidate(side: str, index: int) -> dict:
    return {
        "url": f"{side}://{index}", "channel": f"service-{index % 3}",
        "function_name": f"{side}-function-{index}", "retrieval_rank": index + 1,
        "retrieval_score": 10.0 - index, "text_plain": f"plain {side} {index}",
        "text_schema": f"schema {side} {index}",
    }


def case() -> dict:
    return {
        "group_id": "case-1", "query": "when x then y",
        "trigger_candidates": [candidate("trigger", i) for i in range(10)],
        "action_candidates": [candidate("action", i) for i in range(10)],
    }


def ranking() -> list[dict]:
    return [
        {"trigger_url": f"trigger://{i}", "action_url": f"action://{i}"}
        for i in range(10)
    ]


class ScriptedChooser:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        decision = self.decisions.pop(0)
        if decision == "ABSTAIN":
            choice = "ABSTAIN"
        elif decision == "FIRST":
            choice = request["cards"][0]["card_id"]
        elif decision == "SECOND":
            choice = request["cards"][1]["card_id"]
        elif decision == "BASELINE":
            baseline = next(
                card for card in request["cards"]
                if card["trigger"]["function_name"] == "trigger-function-0"
                and card["action"]["function_name"] == "action-function-0"
            )
            choice = baseline["card_id"]
        elif decision == "NONBASELINE":
            choice = next(
                card["card_id"] for card in request["cards"]
                if not (
                    card["trigger"]["function_name"] == "trigger-function-0"
                    and card["action"]["function_name"] == "action-function-0"
                )
            )
        else:
            return {"ok": False, "choice_id": None, "api_attempts": 1, "tool_calls": 0, "usage": {}, "error": "forced"}
        return {
            "ok": True, "choice_id": choice, "api_attempts": 1, "tool_calls": 1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "latency_seconds": .1},
            "error": None,
        }


def test_unrouted_case_has_zero_calls_and_cost():
    trace = unrouted_trace(case(), routing_score=.1)
    assert trace["routed"] is False
    assert trace["calls"] == []
    assert trace["accounting"]["logical_calls"] == 0
    assert trace["final_pair"] == trace["baseline_pair"]


def test_proposer_abstention_is_safe_and_accounted():
    chooser = ScriptedChooser(["ABSTAIN"])
    trace = resolve(
        case(), ranking(), SafePairPolicy(5, "plain", verification="binary_dual"),
        chooser, routing_score=.8,
    )
    assert trace["stable_decision"] == "PROPOSER_ABSTAIN"
    assert trace["retained_baseline"] is True
    assert trace["accounting"]["logical_calls"] == 1


def test_binary_dual_verifier_sees_only_two_cards_in_swapped_order():
    chooser = ScriptedChooser(["NONBASELINE", "NONBASELINE", "NONBASELINE"])
    trace = resolve(
        case(), ranking(), SafePairPolicy(5, "plain", verification="binary_dual"),
        chooser, routing_score=.8,
    )
    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert len(chooser.requests) == 3
    first, second = chooser.requests[1], chooser.requests[2]
    assert len(first["cards"]) == len(second["cards"]) == 2
    assert [card["card_id"] for card in first["cards"]] == list(
        reversed([card["card_id"] for card in second["cards"]])
    )
    assert trace["alternative_pair"] is None


def test_disagreement_cannot_override_top1():
    chooser = ScriptedChooser(["NONBASELINE", "NONBASELINE", "BASELINE"])
    trace = resolve(
        case(), ranking(), SafePairPolicy(5, "plain", verification="binary_dual"),
        chooser, routing_score=.8,
    )
    assert trace["stable_decision"] == "VERIFIER_DISAGREE"
    assert trace["final_pair"] == trace["baseline_pair"]


def test_reference_field_is_rejected_before_call():
    contaminated = case() | {"valid_pairs": []}
    chooser = ScriptedChooser(["ABSTAIN"])
    try:
        resolve(contaminated, ranking(), SafePairPolicy(5, "plain"), chooser, routing_score=.8)
    except ValueError as error:
        assert "reference fields" in str(error)
    else:
        raise AssertionError("contaminated case was accepted")
    assert chooser.requests == []
