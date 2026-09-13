#!/usr/bin/env python3
"""Behavioral tests for the round-five complete-pair policy seam."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))


def _candidate(side: str, rank: int, service: str | None = None) -> dict:
    service_id = service or f"{side}-service-{rank}"
    return {
        "url": f"https://ifttt.com/{service_id}/{side}s/function-{rank}",
        "channel": service_id,
        "service_name": service_id.replace("-", " ").title(),
        "function_name": f"{side.title()} function {rank}",
        "text_plain": f"Plain evidence for {side} function {rank}",
        "text_schema": f"Schema evidence for {side} function {rank}: field_{rank}",
        "retrieval_rank": rank,
        "retrieval_score": 1.0 - rank / 100.0,
    }


def _case() -> dict:
    return {
        "group_id": "case-001",
        "query": "When the trigger happens, perform the action",
        "trigger_candidates": [_candidate("trigger", rank) for rank in range(1, 11)],
        "action_candidates": [_candidate("action", rank) for rank in range(1, 11)],
    }


def _ranking(*, include_baseline: bool = False) -> list[dict[str, str]]:
    result = []
    for trigger_rank in range(1, 11):
        for action_rank in range(1, 11):
            if not include_baseline and trigger_rank == action_rank == 1:
                continue
            result.append(
                {
                    "trigger_url": _candidate("trigger", trigger_rank)["url"],
                    "action_url": _candidate("action", action_rank)["url"],
                }
            )
    return result


class ScriptedChooser:
    """Fake only the true external chooser seam."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests: list[dict] = []

    def select(self, request: dict) -> dict:
        self.requests.append(copy.deepcopy(request))
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        return script(request) if callable(script) else script


def _success(choice, *, attempts: int = 1, tools: int = 1, tokens: int = 7):
    def select(request: dict) -> dict:
        value = choice(request) if callable(choice) else choice
        return {
            "ok": True,
            "choice_id": value,
            "api_attempts": attempts,
            "tool_calls": tools,
            "usage": {"total_tokens": tokens, "latency_seconds": 0.25},
            "error": None,
        }

    return select


def _failure(error: str = "scripted_failure") -> dict:
    return {
        "ok": False,
        "choice_id": None,
        "api_attempts": 2,
        "tool_calls": 0,
        "usage": {"total_tokens": 3},
        "error": error,
    }


def _is_baseline_card(card: dict) -> bool:
    return (
        card["trigger"]["function_name"] == "Trigger function 1"
        and card["action"]["function_name"] == "Action function 1"
    )


def _baseline_id(request: dict) -> str:
    return next(card["card_id"] for card in request["cards"] if _is_baseline_card(card))


def _nonbaseline_id(request: dict) -> str:
    return next(card["card_id"] for card in request["cards"] if not _is_baseline_card(card))


def _role_id(request: dict, proposal_id: str, role: str) -> str:
    if role == "baseline":
        return _baseline_id(request)
    if role == "proposal":
        return proposal_id
    if role == "alternative":
        return next(
            card["card_id"]
            for card in request["cards"]
            if card["card_id"] not in {_baseline_id(request), proposal_id}
        )
    raise AssertionError(f"unknown role: {role}")


class PairCardPolicyTests(unittest.TestCase):
    def test_rejects_nested_gold_before_either_external_seam(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        proposer = ScriptedChooser([])
        verifier = ScriptedChooser([])
        contaminated = _case() | {
            "metadata": {"audit": {"gold_pair": {"trigger_url": "secret"}}}
        }

        with self.assertRaisesRegex(ValueError, "reference fields"):
            resolve_pair_cards(
                contaminated,
                _ranking(),
                PairCardPolicy(5),
                proposer,
                verifier,
            )

        self.assertEqual(proposer.requests, [])
        self.assertEqual(verifier.requests, [])

    def test_builds_exactly_five_complete_plain_cards_and_forces_baseline(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        proposer = ScriptedChooser([_success(_baseline_id)])
        verifier = ScriptedChooser([])
        result = resolve_pair_cards(
            _case(), _ranking(include_baseline=False), PairCardPolicy(5), proposer, verifier
        )
        request = proposer.requests[0]

        self.assertEqual(request["phase"], "pair_proposal")
        self.assertEqual(request["evidence_view"], "plain")
        self.assertFalse(request["allow_abstain"])
        self.assertEqual(len(request["cards"]), 5)
        self.assertEqual(len({card["card_id"] for card in request["cards"]}), 5)
        for card in request["cards"]:
            self.assertEqual(set(card), {"card_id", "trigger", "action"})
            self.assertEqual(
                set(card["trigger"]), {"service_name", "function_name", "evidence"}
            )
            self.assertIn("Plain evidence", card["trigger"]["evidence"])
            self.assertNotIn("Schema evidence", card["trigger"]["evidence"])
            self.assertNotIn("retrieval_rank", repr(card))
            self.assertNotIn("ifttt.com", repr(card))
        self.assertEqual(
            result["baseline_pair"],
            {
                "trigger_url": _candidate("trigger", 1)["url"],
                "action_url": _candidate("action", 1)["url"],
            },
        )
        self.assertEqual(result["final_pair"], result["baseline_pair"])
        self.assertEqual(len(result["candidate_pairs"]), 5)
        self.assertIn(result["baseline_pair"], result["candidate_pairs"])
        self.assertEqual(len(verifier.requests), 0)

    def test_proposer_presentation_blinds_baseline_position_by_case(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        positions = []
        for index in range(16):
            sample = _case()
            sample["group_id"] = f"case-position-{index}"
            proposer = ScriptedChooser([_success(_baseline_id)])
            resolve_pair_cards(
                sample, _ranking(), PairCardPolicy(5), proposer, ScriptedChooser([])
            )
            positions.append(
                next(
                    offset
                    for offset, card in enumerate(proposer.requests[0]["cards"])
                    if _is_baseline_card(card)
                )
            )

        self.assertGreater(len(set(positions)), 1)

    def test_pair_ranking_integrity_failures_happen_before_calls(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        bad_inputs = []
        outside = _ranking()
        outside[-1] = dict(outside[-1]) | {"action_url": "https://outside.invalid/action"}
        bad_inputs.append(outside)
        duplicate = _ranking()
        duplicate[-1] = dict(duplicate[0])
        bad_inputs.append(duplicate)
        contaminated_shape = _ranking()
        contaminated_shape[0] = dict(contaminated_shape[0]) | {"pair_rank": 1}
        bad_inputs.append(contaminated_shape)

        for pair_ranking in bad_inputs:
            with self.subTest(kind=repr(pair_ranking[-1])):
                proposer = ScriptedChooser([])
                verifier = ScriptedChooser([])
                with self.assertRaises((ValueError, TypeError)):
                    resolve_pair_cards(
                        _case(), pair_ranking, PairCardPolicy(5), proposer, verifier
                    )
                self.assertEqual(proposer.requests, [])
                self.assertEqual(verifier.requests, [])

    def test_opaque_card_ids_do_not_depend_on_retrieval_rank_values(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        original_proposer = ScriptedChooser(
            [_success(_baseline_id)]
        )
        shifted_proposer = ScriptedChooser(
            [_success(_baseline_id)]
        )
        original = resolve_pair_cards(
            _case(), _ranking(), PairCardPolicy(5), original_proposer, ScriptedChooser([])
        )
        shifted_case = _case()
        for side in ("trigger", "action"):
            for item in shifted_case[f"{side}_candidates"]:
                item["retrieval_rank"] += 100
        shifted = resolve_pair_cards(
            shifted_case,
            _ranking(),
            PairCardPolicy(5),
            shifted_proposer,
            ScriptedChooser([]),
        )

        self.assertEqual(original["card_ids"], shifted["card_ids"])

    def test_proposer_failure_and_exception_retain_top1_without_verification(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        for script, complete in ((_failure(), True), (RuntimeError("secret"), False)):
            with self.subTest(script=type(script).__name__):
                proposer = ScriptedChooser([script])
                verifier = ScriptedChooser([])
                result = resolve_pair_cards(
                    _case(), _ranking(), PairCardPolicy(5), proposer, verifier
                )
                self.assertEqual(result["final_pair"], result["baseline_pair"])
                self.assertEqual(result["fallback_reason"], "proposer_protocol_failure")
                self.assertEqual(result["accounting"]["logical_calls"], 1)
                self.assertEqual(result["accounting"]["complete"], complete)
                self.assertEqual(verifier.requests, [])

    def test_proposer_baseline_stops_after_one_call_even_if_model_requests_more(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        def choose_baseline(request: dict) -> dict:
            return {
                "ok": True,
                "choice_id": _baseline_id(request),
                "api_attempts": 1,
                "tool_calls": 1,
                "usage": {},
                "error": None,
                "confidence": 0.0,
                "request_more": True,
            }

        verifier = ScriptedChooser([])
        result = resolve_pair_cards(
            _case(),
            _ranking(),
            PairCardPolicy(10),
            ScriptedChooser([choose_baseline]),
            verifier,
        )

        self.assertTrue(result["retained_baseline"])
        self.assertEqual(result["stable_decision"], "NOT_RUN")
        self.assertEqual(result["accounting"]["logical_calls"], 1)
        self.assertEqual(verifier.requests, [])

    def test_unverified_pilot_accepts_nonbaseline_after_exactly_one_call(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        proposer = ScriptedChooser([_success(_nonbaseline_id)])
        result = resolve_pair_cards(
            _case(), _ranking(), PairCardPolicy(5, verify=False), proposer, None
        )

        self.assertNotEqual(result["final_pair"], result["baseline_pair"])
        self.assertEqual(result["final_pair"], result["proposer_pair"])
        self.assertEqual(result["stable_decision"], "NOT_RUN")
        self.assertEqual(result["accounting"]["logical_calls"], 1)

    def test_stable_accept_uses_two_schema_calls_in_opposite_orders(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        selected: dict[str, str] = {}

        def choose_proposal(request: dict) -> str:
            selected["proposal"] = _nonbaseline_id(request)
            return selected["proposal"]

        proposer = ScriptedChooser([_success(choose_proposal, attempts=2, tokens=10)])
        verifier = ScriptedChooser(
            [
                _success(lambda request: selected["proposal"], tokens=4),
                _success(lambda request: selected["proposal"], tokens=5),
            ]
        )
        result = resolve_pair_cards(
            _case(), _ranking(), PairCardPolicy(5), proposer, verifier
        )

        self.assertEqual(result["stable_decision"], "ACCEPT_PROPOSAL")
        self.assertEqual(result["final_pair"], result["proposer_pair"])
        self.assertEqual(result["verifier_decisions"], ["ACCEPT_PROPOSAL"] * 2)
        self.assertEqual(len(verifier.requests), 2)
        first_ids = [card["card_id"] for card in verifier.requests[0]["cards"]]
        second_ids = [card["card_id"] for card in verifier.requests[1]["cards"]]
        self.assertEqual(second_ids, [first_ids[2], first_ids[0], first_ids[1]])
        for request in verifier.requests:
            self.assertEqual(request["phase"], "pair_verification")
            self.assertEqual(request["evidence_view"], "schema")
            self.assertTrue(request["allow_abstain"])
            self.assertTrue(
                all(
                    "Schema evidence" in card[side]["evidence"]
                    for card in request["cards"]
                    for side in ("trigger", "action")
                )
            )
            self.assertNotIn("Plain evidence", repr(request["cards"]))
            self.assertNotIn("ifttt.com", repr(request["cards"]))
        self.assertEqual(
            result["accounting"],
            {
                "logical_calls": 3,
                "api_attempts": 4,
                "tool_calls": 3,
                "catalog_reads": 4,
                "evidence_presentations": {
                    "proposer_pair_cards": 5,
                    "proposer_endpoints": 10,
                    "verifier_pair_cards": 6,
                    "verifier_endpoints": 12,
                    "verifier_unique_schema_catalog_rows": 4,
                },
                "complete": True,
                "usage": {"latency_seconds": 0.75, "total_tokens": 19},
            },
        )

    def test_stable_alternative_and_stable_keep_map_private_roles(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        scenarios = (
            ("CHOOSE_ALT", "alternative", "alternative_pair"),
            ("KEEP_TOP1", "baseline", "baseline_pair"),
        )
        for expected_decision, role, expected_pair_key in scenarios:
            with self.subTest(expected_decision=expected_decision):
                selected: dict[str, str] = {}

                def propose(request: dict) -> str:
                    selected["proposal"] = _nonbaseline_id(request)
                    return selected["proposal"]

                result = resolve_pair_cards(
                    _case(),
                    _ranking(),
                    PairCardPolicy(5),
                    ScriptedChooser([_success(propose)]),
                    ScriptedChooser([
                        _success(lambda request, role=role: _role_id(request, selected["proposal"], role)),
                        _success(lambda request, role=role: _role_id(request, selected["proposal"], role)),
                    ]),
                )
                self.assertEqual(result["stable_decision"], expected_decision)
                self.assertEqual(result["final_pair"], result[expected_pair_key])
                self.assertIsNone(result["fallback_reason"])

    def test_abstain_disagreement_and_one_invalid_verifier_call_are_safe(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        scenarios = (
            (
                [
                    _success("ABSTAIN"),
                    _success("ABSTAIN"),
                ],
                "ABSTAIN",
                "verifier_abstained",
            ),
            (
                [
                    "baseline",
                    "proposal",
                ],
                "DISAGREE",
                "verifier_disagreement",
            ),
            (
                [
                    _failure("bad_tool_call"),
                    "proposal",
                ],
                "DISAGREE",
                "verifier_protocol_failure",
            ),
        )
        for scripts, decision, reason in scenarios:
            with self.subTest(reason=reason):
                selected: dict[str, str] = {}

                def propose(request: dict) -> str:
                    selected["proposal"] = _nonbaseline_id(request)
                    return selected["proposal"]

                verifier_scripts = [
                    _success(
                        lambda request, role=script: _role_id(
                            request, selected["proposal"], role
                        )
                    )
                    if isinstance(script, str) and script != "ABSTAIN"
                    else (_success("ABSTAIN") if script == "ABSTAIN" else script)
                    for script in scripts
                ]
                verifier = ScriptedChooser(verifier_scripts)
                result = resolve_pair_cards(
                    _case(),
                    _ranking(),
                    PairCardPolicy(5),
                    ScriptedChooser([_success(propose)]),
                    verifier,
                )
                self.assertEqual(len(verifier.requests), 2)
                self.assertEqual(result["final_pair"], result["baseline_pair"])
                self.assertEqual(result["stable_decision"], decision)
                self.assertEqual(result["fallback_reason"], reason)
                self.assertEqual(result["accounting"]["logical_calls"], 3)

    def test_out_of_enum_choice_and_missing_accounting_never_override(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        bad = (
            {
                "ok": True,
                "choice_id": "pc_not_supplied",
                "api_attempts": 1,
                "tool_calls": 1,
                "usage": {},
                "error": None,
            },
            {
                "ok": True,
                "choice_id": "will-not-be-read",
                "api_attempts": None,
                "tool_calls": 1,
                "usage": {},
                "error": None,
            },
        )
        for response in bad:
            with self.subTest(response=response):
                verifier = ScriptedChooser([])
                result = resolve_pair_cards(
                    _case(),
                    _ranking(),
                    PairCardPolicy(5),
                    ScriptedChooser([response]),
                    verifier,
                )
                self.assertEqual(result["final_pair"], result["baseline_pair"])
                self.assertEqual(result["fallback_reason"], "proposer_protocol_failure")
                self.assertEqual(verifier.requests, [])

    def test_verified_policy_requires_verifier_before_any_external_call(self) -> None:
        from pair_card_policy import PairCardPolicy, resolve_pair_cards

        proposer = ScriptedChooser([])
        with self.assertRaisesRegex(ValueError, "requires a verifier"):
            resolve_pair_cards(_case(), _ranking(), PairCardPolicy(5), proposer, None)
        self.assertEqual(proposer.requests, [])


if __name__ == "__main__":
    unittest.main()
