from __future__ import annotations

import copy
import contextlib
import itertools
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from constraint_agent import ABSTAIN, KEEP, Policy, dual_unanimity_accepts, resolve  # noqa: E402


@contextlib.contextmanager
def raises(exception_type, *, match: str):
    try:
        yield
    except exception_type as error:
        assert re.search(match, str(error)), str(error)
    else:
        raise AssertionError(f"{exception_type.__name__} was not raised")


def field(
    name: str, type_: str, *, required: bool = False, source: str = "unknown"
) -> dict:
    return {"name": name, "type": type_, "required": required, "source": source}


def trigger_candidate(index: int) -> dict:
    output_name = "payload" if index != 2 else "unrelated"
    return {
        "url": f"trigger://{index}",
        "side": "trigger",
        "channel": f"trigger-service-{index % 3}",
        "function_name": f"trigger-fn-{index}",
        "retrieval_rank": index + 1,
        "text_plain": f"plain trigger evidence {index}",
        "text_schema": f"schema trigger evidence {index}",
        "schema": {
            "role": "trigger",
            "outputs": [field(output_name, "string")],
        },
    }


def action_candidate(index: int) -> dict:
    input_type = "integer" if index == 2 else "string"
    return {
        "url": f"action://{index}",
        "side": "action",
        "channel": f"action-service-{index % 3}",
        "function_name": f"action-fn-{index}",
        "retrieval_rank": index + 1,
        "text_plain": f"plain action evidence {index}",
        "text_schema": f"schema action evidence {index}",
        "schema": {
            "role": "action",
            "inputs": [field("payload", input_type, required=True, source="trigger")],
        },
    }


def case(*, count: int = 10) -> dict:
    return {
        "group_id": "case-factorized-1",
        "query": "When an item arrives, send its payload to the destination",
        "trigger_candidates": [trigger_candidate(index) for index in range(count)],
        "action_candidates": [action_candidate(index) for index in range(count)],
    }


def native_trigger_candidate(index: int) -> dict:
    """Representative Dataset-v2 trigger corpus shape (no nested `schema`)."""

    return {
        "api_info_coverage": {"present": 2, "total": 2},
        "categories": ["Smart home & IoT"],
        "channel": "2smart_cloud",
        "channel_display": "2Smart Cloud",
        "description": "This trigger fires when the selected sensor is enabled.",
        "function_name": f"Turned on {index}",
        "function_slug": f"switched_on_{index}",
        "ingredients": [
            {
                "example": "2020-07-08T04:17:06.000+05:00",
                "filter_code": f"_2smartCloud.switchedOn{index}.CreatedAt",
                "label": "created_at",
                "slug": "created_at",
                "type": "Date with time (ISO8601)",
            }
        ],
        "input_fields": [
            {
                "bindable": False,
                "can_have_default": False,
                "filter_code_method": "",
                "label": "Sensor",
                "required": True,
                "slug": "sensor",
            }
        ],
        "kind": "trigger",
        "retrieval_rank": index + 1,
        "schema_status": "present",
        "text_plain": f"channel: 2smart cloud | function: Turned on {index}",
        "text_schema": (
            f"channel: 2smart cloud | function: Turned on {index}\n"
            "Trigger fields: Sensor [required; no-setter-metadata].\n"
            "Provides: created_at (Date with time (ISO8601))."
        ),
        "url": f"https://ifttt.com/2smart_cloud/triggers/switched_on_{index}",
    }


def native_action_candidate(index: int, *, bindable: bool = True) -> dict:
    """Representative Dataset-v2 action corpus shape (no nested `schema`)."""

    return {
        "api_info_coverage": {"present": 7, "total": 7},
        "categories": ["Smart home & IoT"],
        "channel": "abode",
        "channel_display": "abode",
        "description": "This action will change your system mode.",
        "function_name": f"Change mode {index}",
        "function_slug": f"change_mode_{index}",
        "ingredients": [],
        "input_fields": [
            {
                "bindable": bindable,
                "can_have_default": bindable,
                "filter_code_method": (
                    f"Abode.changeMode{index}.setMode(string: mode)" if bindable else ""
                ),
                "label": "What mode?",
                "required": True,
                "slug": "mode",
            }
        ],
        "kind": "action",
        "retrieval_rank": index + 1,
        "schema_status": "present",
        "text_plain": f"channel: abode | function: Change mode {index}",
        "text_schema": (
            f"channel: abode | function: Change mode {index}\n"
            f"Action fields: What mode? [required; {'ingredient-bindable' if bindable else 'no-setter-metadata'}]."
        ),
        "url": f"https://ifttt.com/abode/actions/change_mode_{index}",
    }


def native_case() -> dict:
    return {
        "group_id": "native-v2-case",
        "query": "When the sensor changes, change the house mode",
        "trigger_candidates": [native_trigger_candidate(index) for index in range(10)],
        "action_candidates": [
            native_action_candidate(index, bindable=index != 0) for index in range(10)
        ],
    }


def good_response(**choice) -> dict:
    return {
        "ok": True,
        **choice,
        "api_attempts": 1,
        "tool_calls": 1,
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
            "latency_seconds": 0.1,
        },
        "error": None,
    }


def candidate_alias(request: dict, side: str, function_name: str) -> str:
    return next(
        candidate["candidate_id"]
        for candidate in request["candidates"][side]
        if candidate["function_name"] == function_name
    )


def pair_alias(
    request: dict, *, trigger: str | None = None, action: str | None = None
) -> str:
    for pair in request["pairs"]:
        if trigger is not None and pair["trigger"]["function_name"] != trigger:
            continue
        if action is not None and pair["action"]["function_name"] != action:
            continue
        return pair["pair_id"]
    raise AssertionError("requested pair was not presented")


class FunctionChooser:
    def __init__(self, function):
        self.function = function
        self.requests: list[dict] = []

    def select(self, request):
        request = copy.deepcopy(dict(request))
        self.requests.append(request)
        return self.function(request, len(self.requests))


def approve_trigger_one(request: dict, _: int) -> dict:
    if request["phase"] == "factorized_selection":
        return good_response(
            decision="CHANGE_TRIGGER",
            trigger_choice=candidate_alias(request, "trigger", "trigger-fn-1"),
            action_choice=KEEP,
        )
    return good_response(choice_id=pair_alias(request, trigger="trigger-fn-1"))


def test_requests_use_independent_top10_endpoint_lists_and_hide_private_rank_and_identity():
    chooser = FunctionChooser(
        lambda request, _: good_response(
            decision="KEEP", trigger_choice=KEEP, action_choice=KEEP
        )
    )
    trace = resolve(case(), chooser)

    request = chooser.requests[0]
    assert set(request["candidates"]) == {"trigger", "action"}
    assert len(request["candidates"]["trigger"]) == 10
    assert len(request["candidates"]["action"]) == 10
    assert "pairs" not in request and "cards" not in request
    serialized = json.dumps(request)
    assert "trigger://" not in serialized
    assert "action://" not in serialized
    assert "retrieval_rank" not in serialized
    assert trace["stable_decision"] == "KEEP_TOP1"
    assert trace["accounting"]["tool_counts"] == {
        "search_catalog": 2,
        "read_endpoint_schema": 0,
        "validate_pair": 0,
    }


def test_reference_fields_are_rejected_before_any_chooser_call():
    contaminated = case() | {
        "gold_pair": {"trigger_url": "trigger://1", "action_url": "action://1"}
    }
    chooser = FunctionChooser(
        lambda request, _: (_ for _ in ()).throw(AssertionError(request))
    )

    with raises(ValueError, match="reference fields crossed inference seam"):
        resolve(contaminated, chooser)

    assert chooser.requests == []


def test_nested_reference_field_inside_candidate_is_also_rejected():
    contaminated = case()
    contaminated["trigger_candidates"][4]["metadata"] = {"is_correct": True}
    chooser = FunctionChooser(
        lambda request, _: (_ for _ in ()).throw(AssertionError(request))
    )

    with raises(ValueError, match="is_correct"):
        resolve(contaminated, chooser)
    assert chooser.requests == []


def test_choice_cannot_escape_independently_retrieved_top10():
    def malicious(request: dict, _: int) -> dict:
        return good_response(
            decision="CHANGE_TRIGGER", trigger_choice="T11", action_choice=KEEP
        )

    chooser = FunctionChooser(malicious)
    trace = resolve(case(count=11), chooser)

    assert trace["retained_baseline"] is True
    assert trace["stable_decision"] == "SELECTION_FAILURE"
    assert trace["fallback_reason"] == "trigger_choice_outside_top10_or_not_changed"
    assert len(trace["candidate_universe"]["trigger"]) == 10
    assert "trigger://10" not in trace["candidate_universe"]["trigger"]
    assert len(chooser.requests) == 1


def test_one_side_trigger_edit_keeps_action_and_can_be_accepted():
    chooser = FunctionChooser(approve_trigger_one)
    trace = resolve(case(), chooser)

    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert trace["final_pair"] == {
        "trigger_url": "trigger://1",
        "action_url": "action://0",
    }
    assert [request["phase"] for request in chooser.requests] == [
        "factorized_selection",
        "pair_verification",
        "pair_verification",
    ]
    first, second = chooser.requests[1:]
    assert [pair["pair_id"] for pair in first["pairs"]] == list(
        reversed([pair["pair_id"] for pair in second["pairs"]])
    )
    assert trace["compiled"] is False


def test_one_side_action_edit_keeps_trigger_and_can_be_accepted():
    def choose_action(request: dict, _: int) -> dict:
        if request["phase"] == "factorized_selection":
            return good_response(
                decision="CHANGE_ACTION",
                trigger_choice=KEEP,
                action_choice=candidate_alias(request, "action", "action-fn-1"),
            )
        return good_response(choice_id=pair_alias(request, action="action-fn-1"))

    trace = resolve(case(), FunctionChooser(choose_action))
    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert trace["final_pair"] == {
        "trigger_url": "trigger://0",
        "action_url": "action://1",
    }


def test_no_verifier_ablation_accepts_valid_proposal_after_one_model_call():
    chooser = FunctionChooser(approve_trigger_one)
    trace = resolve(
        case(),
        chooser,
        Policy(evidence_view="plain", verification="none", max_repairs=0),
    )

    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert trace["final_pair"]["trigger_url"] == "trigger://1"
    assert [request["phase"] for request in chooser.requests] == [
        "factorized_selection"
    ]
    assert trace["policy"]["verification"] == "none"
    assert trace["policy"]["max_repairs"] == 0
    assert trace["policy"]["evidence_view"] == "plain"
    assert trace["accounting"]["tool_counts"] == {
        "search_catalog": 2,
        "read_endpoint_schema": 2,
        "validate_pair": 1,
    }


def test_native_dataset_v2_rows_hydrate_nonempty_schema_tools_without_false_binding():
    def choose_both(request: dict, _: int) -> dict:
        return good_response(
            decision="CHANGE_BOTH",
            trigger_choice=candidate_alias(request, "trigger", "Turned on 1"),
            action_choice=candidate_alias(request, "action", "Change mode 1"),
        )

    trace = resolve(
        native_case(),
        FunctionChooser(choose_both),
        Policy(verification="none", max_repairs=0),
    )

    reads = [
        event["result"]
        for event in trace["tool_trace"]
        if event["tool"] == "read_endpoint_schema"
    ]
    trigger_schema = next(schema for schema in reads if schema["role"] == "trigger")
    action_schema = next(schema for schema in reads if schema["role"] == "action")
    assert trigger_schema["outputs"] == [
        {
            "name": "created_at",
            "type": "date with time (iso8601)",
            "required": False,
            "source": "unknown",
            "bindable": None,
            "can_have_default": None,
        }
    ]
    assert action_schema["inputs"] == [
        {
            "name": "mode",
            "type": "unknown",
            "required": True,
            "source": "unknown",
            "bindable": True,
            "can_have_default": True,
        }
    ]
    assert trace["validations"][0]["status"] == "unknown"
    assert [issue["code"] for issue in trace["validations"][0]["issues"]] == [
        "UNVERIFIED_REQUIRED_INPUT"
    ]
    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"


def test_native_bindable_false_is_not_mislabeled_as_user_or_trigger_source():
    chooser = FunctionChooser(
        lambda request, _: good_response(
            decision="CHANGE_TRIGGER",
            trigger_choice=candidate_alias(request, "trigger", "Turned on 1"),
            action_choice=KEEP,
        )
    )
    trace = resolve(native_case(), chooser, Policy(verification="none", max_repairs=0))
    action_read = next(
        event["result"]
        for event in trace["tool_trace"]
        if event["tool"] == "read_endpoint_schema"
        and event["result"]["role"] == "action"
    )
    field_schema = action_read["inputs"][0]
    assert field_schema["bindable"] is False
    assert field_schema["required"] is True
    assert field_schema["source"] == "unknown"
    assert trace["validations"][0]["status"] == "unknown"


def test_native_schema_sequence_accepts_label_when_slug_is_absent():
    value = native_case()
    for candidate in value["action_candidates"]:
        candidate["input_fields"][0].pop("slug")

    chooser = FunctionChooser(
        lambda request, _: good_response(
            decision="CHANGE_TRIGGER",
            trigger_choice=candidate_alias(request, "trigger", "Turned on 1"),
            action_choice=KEEP,
        )
    )
    trace = resolve(value, chooser, Policy(verification="none", max_repairs=0))
    action_read = next(
        event["result"]
        for event in trace["tool_trace"]
        if event["tool"] == "read_endpoint_schema"
        and event["result"]["role"] == "action"
    )
    assert action_read["inputs"][0]["name"] == "What mode?"


def test_single_verifier_ablation_accepts_only_one_challenger_vote():
    chooser = FunctionChooser(approve_trigger_one)
    trace = resolve(case(), chooser, Policy(verification="single"))

    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert trace["final_pair"]["trigger_url"] == "trigger://1"
    assert [request["phase"] for request in chooser.requests] == [
        "factorized_selection",
        "pair_verification",
    ]
    assert trace["policy"]["verification"] == "single"


def test_single_verifier_vote_for_baseline_vetoes_proposal():
    def veto(request: dict, _: int) -> dict:
        if request["phase"] == "factorized_selection":
            return good_response(
                decision="CHANGE_TRIGGER",
                trigger_choice=candidate_alias(request, "trigger", "trigger-fn-1"),
                action_choice=KEEP,
            )
        return good_response(choice_id=pair_alias(request, trigger="trigger-fn-0"))

    chooser = FunctionChooser(veto)
    trace = resolve(case(), chooser, Policy(verification="single"))
    assert trace["stable_decision"] == "KEEP_TOP1"
    assert trace["retained_baseline"] is True
    assert [request["phase"] for request in chooser.requests].count(
        "pair_verification"
    ) == 1


def test_zero_repair_ablation_rejects_incompatible_proposal_without_repair_call():
    def invalid_action(request: dict, _: int) -> dict:
        return good_response(
            decision="CHANGE_ACTION",
            trigger_choice=KEEP,
            action_choice=candidate_alias(request, "action", "action-fn-2"),
        )

    chooser = FunctionChooser(invalid_action)
    trace = resolve(case(), chooser, Policy(verification="none", max_repairs=0))

    assert trace["stable_decision"] == "VALIDATION_REJECT"
    assert trace["fallback_reason"] == "schema_incompatible_repair_disabled"
    assert trace["repair_attempted"] is False
    assert trace["retained_baseline"] is True
    assert [request["phase"] for request in chooser.requests] == [
        "factorized_selection"
    ]
    assert trace["policy"]["max_repairs"] == 0


def test_validator_error_drives_exactly_one_repair_then_verification():
    def choose(request: dict, _: int) -> dict:
        if request["phase"] == "factorized_selection":
            return good_response(
                decision="CHANGE_ACTION",
                trigger_choice=KEEP,
                action_choice=candidate_alias(request, "action", "action-fn-2"),
            )
        if request["phase"] == "validator_repair":
            assert request["validator_feedback"]["status"] == "incompatible"
            assert {
                issue["code"] for issue in request["validator_feedback"]["issues"]
            } == {"TRIGGER_ACTION_TYPE_MISMATCH"}
            return good_response(
                decision="CHANGE_ACTION",
                trigger_choice=KEEP,
                action_choice=candidate_alias(request, "action", "action-fn-1"),
            )
        return good_response(choice_id=pair_alias(request, action="action-fn-1"))

    chooser = FunctionChooser(choose)
    trace = resolve(case(), chooser)

    assert trace["repair_attempted"] is True
    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert trace["final_pair"]["action_url"] == "action://1"
    assert [request["phase"] for request in chooser.requests].count(
        "validator_repair"
    ) == 1
    assert [validation["status"] for validation in trace["validations"]] == [
        "incompatible",
        "compatible",
        "compatible",
    ]
    assert trace["accounting"]["tool_counts"] == {
        "search_catalog": 2,
        "read_endpoint_schema": 6,
        "validate_pair": 3,
    }
    assert trace["accounting"]["catalog_reads"] == sum(
        event["tool"] == "read_endpoint_schema" for event in trace["tool_trace"]
    )


def test_second_bad_repair_fails_closed_without_a_third_selection():
    def choose(request: dict, _: int) -> dict:
        if request["phase"] == "factorized_selection":
            return good_response(
                decision="CHANGE_ACTION",
                trigger_choice=KEEP,
                action_choice=candidate_alias(request, "action", "action-fn-2"),
            )
        assert request["phase"] == "validator_repair"
        return good_response(
            decision="CHANGE_BOTH",
            trigger_choice=candidate_alias(request, "trigger", "trigger-fn-2"),
            action_choice=candidate_alias(request, "action", "action-fn-1"),
        )

    chooser = FunctionChooser(choose)
    trace = resolve(case(), chooser)
    assert trace["stable_decision"] == "VALIDATION_REJECT"
    assert trace["retained_baseline"] is True
    assert [request["phase"] for request in chooser.requests] == [
        "factorized_selection",
        "validator_repair",
    ]


def test_early_exit_has_same_acceptance_semantics_as_exhaustive_dual_unanimity():
    choices = ["BASELINE", "CHALLENGER", ABSTAIN, None]
    for first, second in itertools.product(choices, repeat=2):
        exhaustive = dual_unanimity_accepts((first, second), "CHALLENGER")
        if first != "CHALLENGER":
            early = False
        else:
            early = dual_unanimity_accepts((first, second), "CHALLENGER")
        assert early == exhaustive

    def reject_first(request: dict, _: int) -> dict:
        if request["phase"] == "factorized_selection":
            return good_response(
                decision="CHANGE_TRIGGER",
                trigger_choice=candidate_alias(request, "trigger", "trigger-fn-1"),
                action_choice=KEEP,
            )
        return good_response(choice_id=pair_alias(request, trigger="trigger-fn-0"))

    chooser = FunctionChooser(reject_first)
    trace = resolve(case(), chooser)
    assert trace["retained_baseline"] is True
    assert trace["stable_decision"] == "KEEP_TOP1"
    assert [request["phase"] for request in chooser.requests].count(
        "pair_verification"
    ) == 1
    assert trace["accounting"]["logical_model_calls"] == 2


def test_each_chooser_invocation_receives_a_fresh_stateless_request():
    class MutatingChooser:
        def __init__(self):
            self.requests = []

        def select(self, request):
            assert "poison" not in request
            assert "history" not in request
            assert "messages" not in request
            snapshot = copy.deepcopy(request)
            self.requests.append(snapshot)
            request["poison"] = "must not leak into the next call"
            if snapshot["phase"] == "factorized_selection":
                return good_response(
                    decision="CHANGE_TRIGGER",
                    trigger_choice=candidate_alias(snapshot, "trigger", "trigger-fn-1"),
                    action_choice=KEEP,
                )
            return good_response(choice_id=pair_alias(snapshot, trigger="trigger-fn-1"))

    chooser = MutatingChooser()
    trace = resolve(case(), chooser)
    assert trace["stable_decision"] == "ACCEPT_PROPOSAL"
    assert len({request["request_id"] for request in chooser.requests}) == 3
    assert all("poison" not in call["request"] for call in trace["calls"])


def test_identical_inputs_and_stateless_decisions_produce_identical_trace():
    first = resolve(case(), FunctionChooser(approve_trigger_one))
    second = resolve(case(), FunctionChooser(approve_trigger_one))
    assert first == second


def test_real_accounting_is_derived_from_recorded_calls_and_tool_events():
    trace = resolve(case(), FunctionChooser(approve_trigger_one))
    accounting = trace["accounting"]

    assert accounting["logical_model_calls"] == len(trace["calls"])
    assert accounting["api_attempts"] == sum(
        call["api_attempts"] for call in trace["calls"]
    )
    assert accounting["model_tool_calls"] == sum(
        call["model_tool_calls"] for call in trace["calls"]
    )
    assert accounting["deterministic_tool_calls"] == len(trace["tool_trace"])
    assert accounting["catalog_reads"] == 4
    assert [event["sequence"] for event in trace["tool_trace"]] == list(
        range(1, len(trace["tool_trace"]) + 1)
    )


def test_strict_contract_rejects_extra_response_fields():
    def extra_field(request: dict, _: int) -> dict:
        response = good_response(
            decision="KEEP", trigger_choice=KEEP, action_choice=KEEP
        )
        response["rationale"] = "not allowed"
        return response

    trace = resolve(case(), FunctionChooser(extra_field))
    assert trace["stable_decision"] == "SELECTION_FAILURE"
    assert trace["retained_baseline"] is True


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
