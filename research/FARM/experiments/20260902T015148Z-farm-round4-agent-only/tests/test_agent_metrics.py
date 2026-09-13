"""Behavioral tests for the reviewer-grade agent evaluator.

The public seam is ``evaluate_records`` (plus the small statistical helpers).
All expected values below are worked examples rather than values recomputed with
the implementation under test.
"""

from __future__ import annotations

import json
import inspect
import sys
import unittest
from pathlib import Path

try:  # The production evaluator is stdlib-only; tests also run without pytest.
    import pytest
except ModuleNotFoundError:  # pragma: no cover - exercised by the stdlib runner
    pytest = None


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from agent_metrics import (  # noqa: E402
    evaluate_records,
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap_delta,
    score_prediction,
)


def _candidate(identifier: str, service: str) -> dict:
    return {"url": identifier, "service": service}


def _pair(trigger: str, action: str, trigger_service: str, action_service: str) -> dict:
    return {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": trigger_service,
        "action_service": action_service,
    }


def _arm(
    trigger: str | None,
    action: str | None,
    trigger_service: str | None,
    action_service: str | None,
    *,
    protocol_valid: bool = True,
    calls: int = 0,
    attempts: int = 0,
    tools: int = 0,
    catalog_reads: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_seconds: float = 0.0,
    iteration: bool = False,
) -> dict:
    prediction = {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": trigger_service,
        "action_service": action_service,
    }
    return {
        "prediction": prediction,
        "protocol_valid": protocol_valid,
        "iterations": [{"prediction": prediction}] if iteration else [],
        "accounting": {
            "logical_calls": calls,
            "api_attempts": attempts,
            "tool_calls": tools,
            "catalog_reads": catalog_reads,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency_seconds,
        },
    }


def _records() -> list[dict]:
    # The first case deliberately has two valid gold pairs.  The correct agent
    # answer is the *second* pair, which catches accidental canonical/lexical
    # gold selection.
    return [
        {
            "case_id": "multi-gold",
            "valid_pairs": [
                _pair("t1", "a1", "s1", "x1"),
                _pair("t2", "a2", "s2", "x2"),
            ],
            "trigger_candidates": [
                _candidate("t9", "s9"),
                _candidate("t2", "s2"),
                _candidate("t1", "s1"),
            ],
            "action_candidates": [
                _candidate("a9", "x9"),
                _candidate("a2", "x2"),
                _candidate("a1", "x1"),
            ],
            "arms": {
                "retrieval": _arm("t9", "a9", "s9", "x9"),
                "agent": _arm(
                    "t2",
                    "a2",
                    "s2",
                    "x2",
                    calls=1,
                    attempts=2,
                    tools=1,
                    prompt_tokens=100,
                    completion_tokens=20,
                    latency_seconds=2.0,
                    iteration=True,
                ),
            },
        },
        {
            "case_id": "rank-one-regression",
            "valid_pairs": [_pair("t3", "a3", "s1", "x3")],
            "trigger_candidates": [_candidate("t3", "s1"), _candidate("t9", "s9")],
            "action_candidates": [_candidate("a3", "x3"), _candidate("a9", "x9")],
            "arms": {
                "retrieval": _arm("t3", "a3", "s1", "x3"),
                "agent": _arm(
                    "t9",
                    "a9",
                    "s9",
                    "x9",
                    calls=1,
                    attempts=1,
                    tools=1,
                    prompt_tokens=60,
                    completion_tokens=10,
                    latency_seconds=1.0,
                    iteration=True,
                ),
            },
        },
        {
            "case_id": "rank-six-protocol-failure",
            "valid_pairs": [_pair("t4", "a4", "s4", "x4")],
            "trigger_candidates": [
                *[_candidate(f"tz{i}", f"sz{i}") for i in range(1, 6)],
                _candidate("t4", "s4"),
            ],
            "action_candidates": [_candidate("a4", "x4")],
            "arms": {
                "retrieval": _arm("tz1", "a4", "sz1", "x4"),
                "agent": _arm(
                    "tz1",
                    "a4",
                    "sz1",
                    "x4",
                    protocol_valid=False,
                    calls=1,
                    attempts=2,
                    prompt_tokens=30,
                    latency_seconds=4.0,
                ),
            },
        },
        {
            "case_id": "outside-ten-invalid-selection",
            "valid_pairs": [_pair("t5", "a5", "s5", "x5")],
            "trigger_candidates": [_candidate(f"tu{i}", f"su{i}") for i in range(1, 11)],
            "action_candidates": [_candidate(f"au{i}", f"xu{i}") for i in range(1, 11)],
            "arms": {
                "retrieval": _arm("tu1", "au1", "su1", "xu1"),
                # It is correct for intent-to-treat scoring but is not eligible
                # for protocol-conditioned scoring because it escaped the
                # supplied candidate set.
                "agent": _arm(
                    "t5",
                    "a5",
                    "s5",
                    "x5",
                    calls=2,
                    attempts=2,
                    tools=2,
                    prompt_tokens=80,
                    completion_tokens=30,
                    latency_seconds=3.0,
                    iteration=True,
                ),
            },
        },
    ]


if pytest is not None:
    records = pytest.fixture(_records)
else:
    records = _records


def test_multi_gold_exactness_scores_sets_and_preserves_pairing(records: list[dict]) -> None:
    second_gold = score_prediction(
        records[0],
        {"trigger_url": "t2", "action_url": "a2", "trigger_service": "s2", "action_service": "x2"},
    )
    crossed = score_prediction(
        records[0],
        {"trigger_url": "t2", "action_url": "a1", "trigger_service": "s2", "action_service": "x1"},
    )

    assert second_gold == {
        "function_trigger": True,
        "function_action": True,
        "function_joint": True,
        "service_trigger": True,
        "service_action": True,
        "service_joint": True,
    }
    assert crossed == {
        "function_trigger": True,
        "function_action": True,
        "function_joint": False,
        "service_trigger": True,
        "service_action": True,
        "service_joint": False,
    }


def test_candidate_ceilings_and_rank_buckets_are_joint_pair_aware(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=100,
        min_service_support=2,
    )

    assert report["candidate_coverage"]["rank_buckets"] == {
        "rank1": 1,
        "rank2_5": 1,
        "rank6_10": 1,
        "outside10": 1,
    }
    assert report["candidate_coverage"]["ceilings"]["k1"]["function"] == {
        "denominator_cases": 4,
        "trigger_hits": 1,
        "trigger_rate": 0.25,
        "action_hits": 2,
        "action_rate": 0.5,
        "joint_hits": 1,
        "joint_rate": 0.25,
    }
    assert report["candidate_coverage"]["ceilings"]["k5"]["function"]["joint_hits"] == 2
    assert report["candidate_coverage"]["ceilings"]["k10"]["function"]["joint_hits"] == 3


def test_intent_to_treat_protocol_conditioning_and_transitions(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=100,
        min_service_support=2,
    )
    agent = report["arms"]["agent"]

    assert agent["intent_to_treat"]["denominator_cases"] == 4
    assert agent["intent_to_treat"]["function_joint"] == {"hits": 2, "rate": 0.5}
    assert agent["protocol_conditioned"]["denominator_cases"] == 2
    assert agent["protocol_conditioned"]["excluded_cases"] == 2
    assert agent["protocol_conditioned"]["function_joint"] == {"hits": 1, "rate": 0.5}
    assert agent["protocol"]["model_protocol_valid_cases"] == 3
    assert agent["protocol"]["selection_within_candidates_cases"] == 3
    assert agent["protocol"]["conditioned_cases"] == 2

    transition = agent["transitions"]["baseline_to_final"]["function_joint"]
    assert transition == {
        "denominator_cases": 4,
        "before_hits": 1,
        "after_hits": 2,
        "recoveries": 2,
        "regressions": 1,
        "unchanged_correct": 0,
        "unchanged_wrong": 1,
        "delta": 0.25,
    }


def test_accounting_names_every_numerator_and_denominator(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=100,
        min_service_support=2,
    )
    accounting = report["arms"]["agent"]["accounting"]

    assert accounting["denominators"] == {
        "intent_to_treat_cases": 4,
        "routed_cases": 4,
        "called_cases": 4,
        "model_protocol_valid_cases": 3,
        "protocol_conditioned_cases": 2,
        "logical_calls": 5,
        "api_attempts": 7,
        "tool_calls": 4,
        "catalog_reads": 0,
        "latency_observed_cases": 4,
        "latency_observed_called_cases": 4,
        "latency_observed_logical_calls": 5,
    }
    assert accounting["totals"] == {
        "logical_calls": 5,
        "api_attempts": 7,
        "tool_calls": 4,
        "catalog_reads": 0,
        "prompt_tokens": 270,
        "completion_tokens": 60,
        "total_tokens": 330,
        "latency_seconds": 10.0,
    }
    assert accounting["rates"]["logical_calls_per_itt_case"] == {
        "numerator": "logical_calls",
        "numerator_value": 5,
        "denominator": "intent_to_treat_cases",
        "denominator_value": 4,
        "value": 1.25,
    }
    assert accounting["rates"]["api_attempts_per_logical_call"]["value"] == 1.4
    assert accounting["rates"]["catalog_reads_per_itt_case"]["value"] == 0.0
    assert accounting["rates"]["catalog_reads_per_logical_call"]["value"] == 0.0
    assert accounting["rates"]["tokens_per_logical_call"]["value"] == 66.0
    assert accounting["rates"]["latency_seconds_per_logical_call"]["value"] == 2.0


def test_service_tables_withhold_sparse_groups_and_pool_long_tail(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=100,
        min_service_support=2,
    )
    services = report["arms"]["agent"]["per_service"]

    assert services["minimum_support"] == 2
    assert services["trigger_services"]["reported"] == [
        {
            "service": "s1",
            "support_cases": 2,
            "service_exact_hits": 0,
            "service_exact_rate": 0.0,
            "function_exact_hits": 0,
            "function_exact_rate": 0.0,
        }
    ]
    assert services["trigger_services"]["withheld_group_count"] == 3
    assert services["trigger_services"]["pooled_long_tail"]["support_memberships"] == 3
    assert services["trigger_services"]["pooled_long_tail"]["unique_cases"] == 3
    assert services["action_services"]["reported"] == []
    assert services["action_services"]["withheld_group_count"] == 5
    assert services["service_pairs"]["withheld_group_count"] == 5


def test_exact_inference_and_holm_adjustment_are_deterministic() -> None:
    mcnemar = exact_mcnemar([0, 0, 0, 0], [1, 1, 1, 1])
    first = paired_bootstrap_delta([0, 0, 0, 0], [1, 1, 1, 1], seed=7, iterations=100)
    second = paired_bootstrap_delta([0, 0, 0, 0], [1, 1, 1, 1], seed=7, iterations=100)

    assert mcnemar == {
        "denominator_pairs": 4,
        "recoveries": 4,
        "regressions": 0,
        "discordant_pairs": 4,
        "exact_two_sided_p": 0.125,
    }
    assert first == second == {
        "denominator_pairs": 4,
        "observed_delta": 1.0,
        "confidence_level": 0.95,
        "method": "paired_percentile_bootstrap",
        "iterations": 100,
        "seed": 7,
        "ci_lower": 1.0,
        "ci_upper": 1.0,
    }
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {
        "a": 0.03,
        "b": 0.06,
        "c": 0.06,
    }


def test_comparisons_include_six_exact_outcomes_and_global_holm(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=100,
        min_service_support=2,
    )
    comparison = report["comparisons"]["agent_vs_retrieval"]

    assert set(comparison["outcomes"]) == {
        "function_trigger",
        "function_action",
        "function_joint",
        "service_trigger",
        "service_action",
        "service_joint",
    }
    joint = comparison["outcomes"]["function_joint"]
    assert joint["baseline_rate"] == 0.25
    assert joint["arm_rate"] == 0.5
    assert joint["mcnemar"]["recoveries"] == 2
    assert joint["mcnemar"]["regressions"] == 1
    assert "holm_family_size" in joint
    assert joint["holm_family_size"] == 6
    assert joint["holm_adjusted_p"] >= joint["mcnemar"]["exact_two_sided_p"]


def test_flat_runner_records_and_nested_adapter_usage_are_supported() -> None:
    rows = [
        {
            "group_id": "flat-case",
            "valid_pairs": [_pair("t1", "a1", "s1", "x1")],
            "trigger_candidates": [_candidate("t9", "s9"), _candidate("t1", "s1")],
            "action_candidates": [_candidate("a9", "x9"), _candidate("a1", "x1")],
            "baseline_pair": {
                "trigger_url": "t9",
                "action_url": "a9",
                "trigger_service": "s9",
                "action_service": "x9",
            },
            "final_pair": {
                "trigger_url": "t1",
                "action_url": "a1",
                "trigger_service": "s1",
                "action_service": "x1",
            },
            "calls": 2,
            "api_attempts": 3,
            "tool_calls": 2,
            "catalog_reads": 4,
            "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            "latency_seconds": 1.5,
            "protocol_valid": True,
            "fallback": False,
        }
    ]

    report = evaluate_records(rows, bootstrap_iterations=20, min_service_support=1)

    assert set(report["arms"]) == {"baseline", "agent"}
    assert report["arms"]["agent"]["intent_to_treat"]["function_joint"] == {
        "hits": 1,
        "rate": 1.0,
    }
    assert report["arms"]["agent"]["accounting"]["totals"] == {
        "logical_calls": 2,
        "api_attempts": 3,
        "tool_calls": 2,
        "catalog_reads": 4,
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
        "latency_seconds": 1.5,
    }
    assert report["comparisons"]["agent_vs_baseline"]["outcomes"]["function_joint"][
        "mcnemar"
    ]["recoveries"] == 1
    assert report["arms"]["agent"]["accounting"]["rates"]["catalog_reads_per_logical_call"][
        "value"
    ] == 2.0


def test_canonical_resolver_trace_derives_services_and_complete_protocol() -> None:
    rows = [
        {
            "case_id": "resolver-case",
            "valid_pairs": [{"trigger_url": "t1", "action_url": "a1"}],
            "trigger_candidates": [
                {"url": "t9", "channel": "s9", "retrieval_rank": 1},
                {"url": "t1", "channel": "s1", "retrieval_rank": 2},
            ],
            "action_candidates": [
                {"url": "a9", "channel": "x9", "retrieval_rank": 1},
                {"url": "a1", "channel": "x1", "retrieval_rank": 2},
            ],
            "baseline_pair": {"trigger_url": "t9", "action_url": "a9"},
            "final_pair": {"trigger_url": "t1", "action_url": "a1"},
            "retained_baseline": False,
            "calls": [
                {
                    "phase": "function",
                    "ok": True,
                    "api_attempts": 2,
                    "tool_calls": 1,
                    "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
                    "error": None,
                    "accounting_complete": True,
                }
            ],
            "accounting": {
                "logical_calls": 1,
                "api_attempts": 2,
                "tool_calls": 1,
                "catalog_reads": 0,
                "complete": True,
                "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            },
        }
    ]

    report = evaluate_records(rows, bootstrap_iterations=20, min_service_support=1)
    arm = report["arms"]["agent"]

    assert arm["intent_to_treat"]["service_joint"] == {"hits": 1, "rate": 1.0}
    assert arm["protocol"]["model_protocol_valid_cases"] == 1
    assert arm["protocol_conditioned"]["denominator_cases"] == 1
    assert arm["accounting"]["totals"]["prompt_tokens"] == 30
    assert arm["accounting"]["totals"]["completion_tokens"] == 5
    assert arm["accounting"]["denominators"]["latency_observed_cases"] == 0
    assert arm["accounting"]["totals"]["latency_seconds"] is None
    assert report["candidate_coverage"]["ceilings"]["k5"]["service"]["joint_hits"] == 1


def test_failed_resolver_call_is_not_protocol_conditioned() -> None:
    rows = [
        {
            "case_id": "failed-call",
            "valid_pairs": [_pair("t1", "a1", "s1", "x1")],
            "trigger_candidates": [_candidate("t1", "s1")],
            "action_candidates": [_candidate("a1", "x1")],
            "baseline_pair": {"trigger_url": "t1", "action_url": "a1"},
            "final_pair": {"trigger_url": "t1", "action_url": "a1"},
            "retained_baseline": True,
            "calls": [
                {
                    "phase": "function",
                    "ok": False,
                    "api_attempts": 2,
                    "tool_calls": 0,
                    "error": "invalid_tool_payload",
                    "accounting_complete": True,
                }
            ],
            "accounting": {
                "logical_calls": 1,
                "api_attempts": 2,
                "tool_calls": 0,
                "complete": True,
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            },
        }
    ]

    report = evaluate_records(rows, bootstrap_iterations=20, min_service_support=1)
    arm = report["arms"]["agent"]

    assert arm["intent_to_treat"]["function_joint"] == {"hits": 1, "rate": 1.0}
    assert arm["protocol"]["model_protocol_valid_cases"] == 0
    assert arm["protocol"]["fallback_cases"] == 1
    assert arm["protocol_conditioned"]["denominator_cases"] == 0
    assert arm["protocol_conditioned"]["excluded_cases"] == 1


def test_retrieval_rank_not_list_position_defines_coverage_and_topk_validity() -> None:
    rows = [
        {
            "case_id": "shuffled-ranking",
            "valid_pairs": [_pair("t2", "a2", "s2", "x2")],
            "trigger_candidates": [
                {"url": "t2", "service": "s2", "retrieval_rank": 2},
                {"url": "t1", "service": "s1", "retrieval_rank": 1},
            ],
            "action_candidates": [
                {"url": "a2", "service": "x2", "retrieval_rank": 2},
                {"url": "a1", "service": "x1", "retrieval_rank": 1},
            ],
            "arms": {
                "retrieval": _arm("t1", "a1", "s1", "x1"),
                "agent": _arm("t2", "a2", "s2", "x2", calls=1, attempts=1, tools=1)
                | {"policy": {"mode": "function_topk", "top_k": 1}},
            },
        }
    ]

    report = evaluate_records(
        rows,
        baseline_arm="retrieval",
        bootstrap_iterations=20,
        min_service_support=1,
    )

    assert report["candidate_coverage"]["rank_buckets"] == {
        "rank1": 0,
        "rank2_5": 1,
        "rank6_10": 0,
        "outside10": 0,
    }
    assert report["candidate_coverage"]["ceilings"]["k1"]["function"]["joint_hits"] == 0
    assert report["arms"]["agent"]["intent_to_treat"]["function_joint"]["hits"] == 1
    assert report["arms"]["agent"]["protocol_conditioned"]["denominator_cases"] == 0


def test_hierarchy_service_stage_is_scored_separately_from_failed_function_stage() -> None:
    rows = [
        {
            "case_id": "hierarchy-stage",
            "valid_pairs": [
                _pair("t1", "a1", "s1", "x1"),
                _pair("t2", "a2", "s2", "x2"),
            ],
            "trigger_candidates": [_candidate("t9", "s9"), _candidate("t2", "s2")],
            "action_candidates": [_candidate("a9", "x9"), _candidate("a2", "x2")],
            "baseline_pair": {"trigger_url": "t9", "action_url": "a9"},
            "final_pair": {"trigger_url": "t9", "action_url": "a9"},
            "retained_baseline": True,
            "policy": {"mode": "service_hierarchy", "top_k": 10},
            # The second (non-lexical) gold service pair is a successful first
            # stage even though the later exact-function call fails.
            "selected_service_pair": {"trigger_service": "s2", "action_service": "x2"},
            "calls": [
                {
                    "phase": "service",
                    "ok": True,
                    "api_attempts": 1,
                    "tool_calls": 1,
                    "usage": {},
                    "accounting_complete": True,
                },
                {
                    "phase": "function_within_service",
                    "ok": False,
                    "api_attempts": 2,
                    "tool_calls": 0,
                    "usage": {},
                    "error": "invalid_tool_payload",
                    "accounting_complete": True,
                },
            ],
            "accounting": {
                "logical_calls": 2,
                "api_attempts": 3,
                "tool_calls": 1,
                "catalog_reads": 2,
                "complete": True,
                "usage": {"latency_seconds": 2.5},
            },
        }
    ]

    report = evaluate_records(rows, bootstrap_iterations=20, min_service_support=1)
    arm = report["arms"]["agent"]

    assert arm["intent_to_treat"]["function_joint"] == {"hits": 0, "rate": 0.0}
    assert arm["service_stage"]["intent_to_treat"]["service_joint"] == {
        "hits": 1,
        "rate": 1.0,
    }
    assert arm["service_stage"]["protocol_conditioned"]["denominator_cases"] == 1
    assert arm["service_stage"]["service_call_protocol_valid_cases"] == 1
    assert arm["accounting"]["totals"]["catalog_reads"] == 2
    assert arm["accounting"]["denominators"]["latency_observed_cases"] == 1


def test_null_selected_service_pair_does_not_create_a_fake_service_stage() -> None:
    row = _records()[0]
    row["arms"]["agent"]["selected_service_pair"] = None

    report = evaluate_records(
        [row],
        baseline_arm="retrieval",
        bootstrap_iterations=20,
        min_service_support=1,
    )

    stage = report["arms"]["agent"]["service_stage"]
    assert stage["applicable_cases"] == 0
    assert stage["intent_to_treat"]["denominator_cases"] == 0


def test_report_is_json_serializable(records: list[dict]) -> None:
    report = evaluate_records(
        records,
        baseline_arm="retrieval",
        bootstrap_iterations=20,
        min_service_support=2,
    )
    assert json.loads(json.dumps(report, sort_keys=True))["schema_version"] == "farm_agent_metrics_v1"


def load_tests(_loader: unittest.TestLoader, _tests: unittest.TestSuite, _pattern: str | None) -> unittest.TestSuite:
    """Expose the same behavioral functions to Python's stdlib test runner."""
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not inspect.isfunction(function):
            continue
        if inspect.signature(function).parameters:
            suite.addTest(unittest.FunctionTestCase(lambda fn=function: fn(_records()), description=name))
        else:
            suite.addTest(unittest.FunctionTestCase(function, description=name))
    return suite
