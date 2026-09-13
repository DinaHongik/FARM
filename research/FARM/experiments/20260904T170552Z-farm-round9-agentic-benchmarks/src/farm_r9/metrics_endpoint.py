"""Exact service/function/field-name outcomes for endpoint candidates."""
from __future__ import annotations

from typing import Any, Mapping

from farm_r9.adapters.common import normalize_label
from farm_r9.contracts import EndpointPrediction


def _selected(case: Mapping[str, Any], prediction: EndpointPrediction | None) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if prediction is None:
        return None, None
    triggers = {row["alias"]: row for row in case["public_evidence"]["trigger_candidates"]}
    actions = {row["alias"]: row for row in case["public_evidence"]["action_candidates"]}
    return triggers.get(prediction.trigger_alias), actions.get(prediction.action_alias)


def _service_identity(candidate: Mapping[str, Any]) -> str:
    """Return the canonical service identity, not a potentially different display name."""
    return str(candidate.get("service_id") or candidate["service"])


def score_endpoint(case: Mapping[str, Any], prediction: EndpointPrediction | None) -> dict[str, Any]:
    trigger, action = _selected(case, prediction)
    gold = case["private"]["gold"]
    benchmark = case["benchmark"]
    if benchmark == "farm_v2_test":
        t_id = case["private"]["trigger_alias_map"].get(prediction.trigger_alias) if prediction else None
        a_id = case["private"]["action_alias_map"].get(prediction.action_alias) if prediction else None
        t_function = t_id in set(gold["trigger_ids"])
        a_function = a_id in set(gold["action_ids"])
        t_service = bool(trigger) and any(normalize_label(_service_identity(trigger)) == normalize_label(pair["trigger_channel"]) for pair in gold["channel_pairs"])
        a_service = bool(action) and any(normalize_label(_service_identity(action)) == normalize_label(pair["action_channel"]) for pair in gold["channel_pairs"])
        fields_trigger = fields_action = None
    else:
        t_service = bool(trigger) and normalize_label(_service_identity(trigger)) == gold["trigger_channel_norm"]
        a_service = bool(action) and normalize_label(_service_identity(action)) == gold["action_channel_norm"]
        t_function = bool(trigger) and (
            normalize_label(trigger.get("function") or "") == gold["trigger_function_norm"]
            or normalize_label(trigger.get("function_event") or "") == gold["trigger_function_norm"]
        )
        a_function = bool(action) and (
            normalize_label(action.get("function") or "") == gold["action_function_norm"]
            or normalize_label(action.get("function_event") or "") == gold["action_function_norm"]
        )
        # Field-name exactness is an end-to-end program metric: matching the
        # empty/common schema of a wrong endpoint must never receive credit.
        # When field-name gold exists, a protocol/parse failure is an incorrect
        # end-to-end prediction rather than a missing observation.  Keeping it
        # in the intention-to-evaluate denominator prevents malformed model
        # output from making field accuracy look artificially better.
        fields_trigger = (
            bool(
                prediction is not None
                and t_service
                and t_function
                and [normalize_label(value) for value in prediction.trigger_field_names]
                == gold.get("trigger_fields_norm")
            )
            if "trigger_fields_norm" in gold
            else None
        )
        fields_action = (
            bool(
                prediction is not None
                and a_service
                and a_function
                and [normalize_label(value) for value in prediction.action_field_names]
                == gold.get("action_fields_norm")
            )
            if "action_fields_norm" in gold
            else None
        )
    return {
        "service_trigger": bool(t_service), "service_action": bool(a_service),
        "service_joint": bool(t_service and a_service),
        "function_trigger": bool(t_service and t_function), "function_action": bool(a_service and a_function),
        "function_joint": bool(t_service and a_service and t_function and a_function),
        "field_trigger_exact": fields_trigger, "field_action_exact": fields_action,
        "field_joint_exact": fields_trigger and fields_action if fields_trigger is not None and fields_action is not None else None,
    }
