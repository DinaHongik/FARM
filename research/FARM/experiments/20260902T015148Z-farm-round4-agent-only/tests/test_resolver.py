#!/usr/bin/env python3
"""Behavioral tests for the round-four pure agent resolver seam."""
from __future__ import annotations

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


def _hierarchy_case() -> dict:
    case = _case()
    trigger_services = ["weather", "weather", "location", "weather", "calendar"]
    action_services = ["dropbox", "social", "dropbox", "dropbox", "email"]
    for rank, service in enumerate(trigger_services, start=1):
        case["trigger_candidates"][rank - 1] = _candidate("trigger", rank, service)
    for rank, service in enumerate(action_services, start=1):
        case["action_candidates"][rank - 1] = _candidate("action", rank, service)
    case["trigger_catalog"] = [dict(item) for item in case["trigger_candidates"]] + [
        _candidate("trigger", 11, "weather")
    ]
    case["action_catalog"] = [dict(item) for item in case["action_candidates"]] + [
        _candidate("action", 11, "dropbox")
    ]
    return case


class ScriptedSelector:
    """Deterministic adapter at the true external-model seam."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests: list[dict] = []

    def select(self, request: dict) -> dict:
        self.requests.append(request)
        script = self.scripts.pop(0)
        return script(request) if callable(script) else script


class ResolverContractTests(unittest.TestCase):
    def test_rejects_gold_before_crossing_selector_seam(self) -> None:
        from resolver import ResolverPolicy, resolve

        selector = ScriptedSelector([])
        contaminated = _case() | {
            "metadata": {"gold_pair": {"trigger_url": "secret", "action_url": "secret"}}
        }

        with self.assertRaisesRegex(ValueError, "reference fields"):
            resolve(contaminated, ResolverPolicy.function_topk(5), selector)

        self.assertEqual(selector.requests, [])

    def test_function_topk_is_one_cumulative_call_with_exact_accounting(self) -> None:
        from resolver import ResolverPolicy, resolve

        def choose_last(request: dict) -> dict:
            return {
                "ok": True,
                "trigger_id": request["trigger_candidates"][-1]["candidate_id"],
                "action_id": request["action_candidates"][-1]["candidate_id"],
                "api_attempts": 2,
                "tool_calls": 1,
                "usage": {"prompt_tokens": 30, "completion_tokens": 4},
                "error": None,
            }

        for top_k in (5, 10):
            with self.subTest(top_k=top_k):
                selector = ScriptedSelector([choose_last])
                result = resolve(_case(), ResolverPolicy.function_topk(top_k), selector)
                request = selector.requests[0]

                self.assertEqual((len(selector.requests), request["phase"]), (1, "function"))
                self.assertEqual(request["evidence_scope"], {"top_k": top_k, "cumulative": True})
                self.assertEqual(
                    (len(request["trigger_candidates"]), len(request["action_candidates"])),
                    (top_k, top_k),
                )
                for public in request["trigger_candidates"] + request["action_candidates"]:
                    self.assertEqual(
                        set(public),
                        {"candidate_id", "service_name", "function_name", "evidence"},
                    )
                    self.assertNotIn("ifttt.com", repr(public))
                self.assertEqual(result["final_pair"], {
                    "trigger_url": _candidate("trigger", top_k)["url"],
                    "action_url": _candidate("action", top_k)["url"],
                })
                self.assertIsNone(result["selected_service_pair"])
                self.assertEqual(result["accounting"], {
                    "logical_calls": 1,
                    "api_attempts": 2,
                    "tool_calls": 1,
                    "catalog_reads": 0,
                    "complete": True,
                    "usage": {"completion_tokens": 4, "prompt_tokens": 30},
                })

    def test_opaque_ids_are_rank_independent_and_reversal_only_changes_presentation(self) -> None:
        from resolver import ResolverPolicy, resolve

        failure = {
            "ok": False,
            "trigger_id": None,
            "action_id": None,
            "api_attempts": 1,
            "tool_calls": 0,
            "usage": {},
            "error": "scripted_failure",
        }
        ranked_selector = ScriptedSelector([failure])
        reversed_selector = ScriptedSelector([failure])
        shifted_selector = ScriptedSelector([failure])
        resolve(_case(), ResolverPolicy.function_topk(5), ranked_selector)
        reversed_result = resolve(
            _case(),
            ResolverPolicy.function_topk(5, order="reversed"),
            reversed_selector,
        )
        shifted = _case()
        for side in ("trigger", "action"):
            for item in shifted[f"{side}_candidates"]:
                item["retrieval_rank"] += 100
        resolve(shifted, ResolverPolicy.function_topk(5), shifted_selector)

        ranked = ranked_selector.requests[0]["trigger_candidates"]
        reversed_candidates = reversed_selector.requests[0]["trigger_candidates"]
        shifted_candidates = shifted_selector.requests[0]["trigger_candidates"]
        ranked_by_name = {item["function_name"]: item["candidate_id"] for item in ranked}
        shifted_by_name = {item["function_name"]: item["candidate_id"] for item in shifted_candidates}

        self.assertEqual(ranked_by_name, shifted_by_name)
        self.assertEqual(
            [item["candidate_id"] for item in reversed_candidates],
            list(reversed([item["candidate_id"] for item in ranked])),
        )
        self.assertEqual(reversed_result["final_pair"], {
            "trigger_url": _candidate("trigger", 1)["url"],
            "action_url": _candidate("action", 1)["url"],
        })

    def test_identity_is_never_used_as_public_function_evidence(self) -> None:
        from resolver import ResolverPolicy, resolve

        unsafe = _case()
        del unsafe["trigger_candidates"][0]["function_name"]
        selector = ScriptedSelector([])

        with self.assertRaisesRegex(ValueError, "required field"):
            resolve(unsafe, ResolverPolicy.function_topk(5), selector)

        self.assertEqual(selector.requests, [])

    def test_service_hierarchy_deduplicates_then_inspects_and_selects_exact_functions(self) -> None:
        from resolver import ResolverPolicy, resolve

        def select_services(request: dict) -> dict:
            trigger = next(
                item for item in request["trigger_candidates"] if item["service_name"] == "weather"
            )
            action = next(
                item for item in request["action_candidates"] if item["service_name"] == "dropbox"
            )
            return {
                "ok": True,
                "trigger_id": trigger["candidate_id"],
                "action_id": action["candidate_id"],
                "api_attempts": 1,
                "tool_calls": 1,
                "usage": {"prompt_tokens": 20},
                "error": None,
            }

        def select_functions(request: dict) -> dict:
            trigger = next(
                item for item in request["trigger_candidates"]
                if item["function_name"] == "Trigger function 11"
            )
            action = next(
                item for item in request["action_candidates"]
                if item["function_name"] == "Action function 11"
            )
            return {
                "ok": True,
                "trigger_id": trigger["candidate_id"],
                "action_id": action["candidate_id"],
                "api_attempts": 2,
                "tool_calls": 1,
                "usage": {"prompt_tokens": 40, "completion_tokens": 5},
                "error": None,
            }

        selector = ScriptedSelector([select_services, select_functions])
        result = resolve(_hierarchy_case(), ResolverPolicy.service_hierarchy(5), selector)
        service_request, function_request = selector.requests

        self.assertEqual([item["service_name"] for item in service_request["trigger_candidates"]], [
            "weather", "location", "calendar",
        ])
        self.assertEqual([item["service_name"] for item in service_request["action_candidates"]], [
            "dropbox", "social", "email",
        ])
        self.assertEqual(
            [item["function_name"] for item in function_request["trigger_candidates"]],
            [
                "Trigger function 1",
                "Trigger function 11",
                "Trigger function 2",
                "Trigger function 4",
            ],
        )
        self.assertEqual(
            [item["function_name"] for item in function_request["action_candidates"]],
            [
                "Action function 1",
                "Action function 11",
                "Action function 3",
                "Action function 4",
            ],
        )
        self.assertEqual(function_request["evidence_scope"], {"top_k": 5, "cumulative": True})
        self.assertEqual(function_request["selected_services"], result["selected_services"])
        self.assertEqual(result["selected_service_pair"], {
            "trigger_service": "weather",
            "action_service": "dropbox",
        })
        self.assertEqual(result["final_pair"], {
            "trigger_url": _candidate("trigger", 11, "weather")["url"],
            "action_url": _candidate("action", 11, "dropbox")["url"],
        })
        self.assertEqual(result["accounting"], {
            "logical_calls": 2,
            "api_attempts": 3,
            "tool_calls": 2,
            "catalog_reads": 2,
            "complete": True,
            "usage": {"completion_tokens": 5, "prompt_tokens": 60},
        })

    def test_hierarchy_retains_baseline_on_function_adapter_failure_and_propagates_instruction(self) -> None:
        from resolver import ResolverPolicy, resolve

        def select_services(request: dict) -> dict:
            return {
                "ok": True,
                "trigger_id": next(
                    item["candidate_id"] for item in request["trigger_candidates"]
                    if item["service_name"] == "location"
                ),
                "action_id": next(
                    item["candidate_id"] for item in request["action_candidates"]
                    if item["service_name"] == "social"
                ),
                "api_attempts": 1,
                "tool_calls": 1,
                "usage": {"prompt_tokens": 10},
                "error": None,
                "confidence": 0.01,
            }

        failure = {
            "ok": False,
            "trigger_id": None,
            "action_id": None,
            "api_attempts": 2,
            "tool_calls": 0,
            "usage": {"prompt_tokens": 20},
            "error": "tool_call_missing",
            "confidence": 0.99,
        }
        instruction = "CUSTOM EXPERIMENTAL INSTRUCTION — preserve this verbatim."
        selector = ScriptedSelector([select_services, failure])
        result = resolve(
            _hierarchy_case(),
            ResolverPolicy.service_hierarchy(5, instruction=instruction),
            selector,
        )

        self.assertEqual([request["instruction"] for request in selector.requests], [
            instruction, instruction,
        ])
        self.assertEqual(result["final_pair"], result["baseline_pair"])
        self.assertTrue(result["retained_baseline"])
        self.assertEqual(result["accounting"], {
            "logical_calls": 2,
            "api_attempts": 3,
            "tool_calls": 1,
            "catalog_reads": 2,
            "complete": True,
            "usage": {"prompt_tokens": 30},
        })
        self.assertEqual(result["calls"][1]["error"], "tool_call_missing")

    def test_hierarchy_catalog_gates_run_before_selector(self) -> None:
        from resolver import ResolverPolicy, resolve

        missing = _hierarchy_case()
        del missing["action_catalog"]
        duplicate = _hierarchy_case()
        duplicate["trigger_catalog"].append(dict(duplicate["trigger_catalog"][0]))
        contaminated = _hierarchy_case()
        contaminated["action_catalog"][0]["gold_label"] = True

        for label, case in (
            ("missing", missing),
            ("duplicate", duplicate),
            ("contaminated", contaminated),
        ):
            with self.subTest(label=label):
                selector = ScriptedSelector([])
                with self.assertRaises((ValueError, TypeError)):
                    resolve(case, ResolverPolicy.service_hierarchy(5), selector)
                self.assertEqual(selector.requests, [])

    def test_function_policy_does_not_branch_on_model_confidence_or_request_more(self) -> None:
        from resolver import ResolverPolicy, resolve

        def uncertain_selection(request: dict) -> dict:
            return {
                "ok": True,
                "trigger_id": request["trigger_candidates"][1]["candidate_id"],
                "action_id": request["action_candidates"][2]["candidate_id"],
                "api_attempts": 1,
                "tool_calls": 1,
                "usage": {},
                "error": None,
                "confidence": 0.0,
                "request_more": True,
            }

        selector = ScriptedSelector([uncertain_selection])
        result = resolve(_case(), ResolverPolicy.function_topk(5), selector)

        self.assertEqual(len(selector.requests), 1)
        self.assertEqual(result["accounting"]["logical_calls"], 1)
        self.assertEqual(result["final_pair"], {
            "trigger_url": _candidate("trigger", 2)["url"],
            "action_url": _candidate("action", 3)["url"],
        })


if __name__ == "__main__":
    unittest.main()
