"""Frozen, reviewer-auditable Yao interactive semantic-parsing experiment.

The endpoint gold, pseudo-ask labels, and gold-derived consistency constraints
are never placed in a model message.  The interactive arm can observe only the
official simulator answer for a component that it explicitly asks about.
"""
from __future__ import annotations

import fcntl
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from farm_r9.artifact_io import canonical_json, read_json, read_jsonl, sha256_text
from farm_r9.ollama_client import (
    OllamaChatClient,
    parse_json_object,
)
from farm_r9.privacy import DataClassification, DataSource


COMPONENTS = (
    "trigger_channel",
    "trigger_function",
    "action_channel",
    "action_function",
)
ARMS = frozenset({"same_model_one_shot", "bounded_clarification_agent"})
MAX_QUESTIONS = 4
_Z_95 = 1.959963984540054
_SYSTEM_BASE = """You map a natural-language IFTTT request to exactly four canonical endpoint labels.
The trigger is the event and the action is the consequence. Choose labels only from the exhaustive
canonical vocabularies supplied in the user message. Copy spelling and capitalization exactly.
Return exactly one JSON object, with no markdown, commentary, or additional keys. Never invent a label."""
_COMMIT_SHAPE = (
    '{"action":"commit","prediction":{"trigger_channel":"...",'
    '"trigger_function":"...","action_channel":"...","action_function":"..."}}'
)
_QUESTIONS = {
    "trigger_channel": "Which service should detect the event?",
    "trigger_function": "What exact event should start the applet?",
    "action_channel": "Which service should perform the consequence?",
    "action_function": "What exact consequence should the applet perform?",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EndpointLabels(_StrictModel):
    trigger_channel: str = Field(min_length=1)
    trigger_function: str = Field(min_length=1)
    action_channel: str = Field(min_length=1)
    action_function: str = Field(min_length=1)


class CommitAction(_StrictModel):
    action: Literal["commit"]
    prediction: EndpointLabels


class AskAction(_StrictModel):
    action: Literal["ask"]
    component: Literal[
        "trigger_channel", "trigger_function", "action_channel", "action_function"
    ]


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate the four exhaustive official output vocabularies."""
    if set(catalog) != set(COMPONENTS):
        raise ValueError("Yao catalog must contain exactly the four endpoint components")
    result: dict[str, tuple[str, ...]] = {}
    for component in COMPONENTS:
        labels = catalog[component]
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(f"invalid Yao catalog for {component}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate Yao catalog label for {component}")
        result[component] = tuple(labels)
    return result


def load_official_catalog(converted_path: Path) -> dict[str, tuple[str, ...]]:
    """Load only public catalog metadata from the pinned inert conversion."""
    converted = read_json(converted_path)
    if not isinstance(converted, dict):
        raise ValueError("Yao converted artifact must be a JSON object")
    source = converted.get("source")
    if not isinstance(source, dict):
        raise ValueError("Yao converted artifact lacks source provenance")
    if source.get("repository_commit") != "cd190229b0b6f237fd3534a297d138c2d834168d":
        raise ValueError("Yao repository commit does not match the frozen protocol")
    if source.get("pickle_sha256") != "3d47f7047808085c75826d90997f05498fae4747fd3d78e6130e93742806a535":
        raise ValueError("Yao source artifact does not match the frozen protocol")
    catalog = converted.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("Yao converted artifact lacks a catalog")
    return validate_catalog(catalog)


def validate_final_cases(cases: Sequence[Mapping[str, Any]], *, expected_n: int = 150) -> None:
    """Fail closed before any final-case model call is made."""
    if len(cases) != expected_n:
        raise ValueError(f"frozen Yao run requires exactly {expected_n} cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Yao final cases require unique nonempty case IDs")
        seen.add(case_id)
        if case.get("benchmark") != "interactive_ifttt":
            raise ValueError("non-Yao case in Yao final sample")
        request = case.get("input")
        simulator = case.get("simulator")
        gold = case.get("private_gold")
        if not isinstance(request, dict) or not isinstance(request.get("query"), str):
            raise ValueError(f"invalid request in {case_id}")
        if not isinstance(simulator, dict) or simulator.get("max_asks_per_component") != 1:
            raise ValueError(f"invalid frozen simulator in {case_id}")
        if not isinstance(gold, dict) or any(not isinstance(gold.get(key), str) for key in COMPONENTS):
            raise ValueError(f"invalid withheld endpoint gold in {case_id}")
        options, frozen = simulator.get("answer_options"), simulator.get("frozen_answers")
        if not isinstance(options, dict) or not isinstance(frozen, dict):
            raise ValueError(f"invalid simulator pools in {case_id}")
        for component in COMPONENTS:
            pool, selected = options.get(component), frozen.get(component)
            if not isinstance(pool, list) or not isinstance(selected, dict):
                raise ValueError(f"invalid {component} simulator pool in {case_id}")
            index, answer = selected.get("answer_index"), selected.get("answer")
            if not isinstance(index, int) or index < 0 or index >= len(pool) or pool[index] != answer:
                raise ValueError(f"frozen {component} answer is not bound to its official pool")


def _catalog_message(query: str, catalog: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    return {
        "request": query,
        "canonical_vocabularies": {component: list(catalog[component]) for component in COMPONENTS},
    }


def build_initial_messages(
    *, query: str, catalog: Mapping[str, Sequence[str]], arm: str
) -> list[dict[str, Any]]:
    """Build equivalent evidence envelopes for both same-model arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown Yao arm: {arm}")
    if arm == "same_model_one_shot":
        instruction = f"You must commit now using this exact shape: {_COMMIT_SHAPE}"
    else:
        instruction = (
            "You may either commit using " + _COMMIT_SHAPE + " or ask for one unresolved component using "
            '{"action":"ask","component":"trigger_channel|trigger_function|action_channel|action_function"}. '
            "Ask each component at most once. The environment, not you, supplies the official answer."
        )
    return [
        {"role": "system", "content": _SYSTEM_BASE + "\n" + instruction},
        {"role": "user", "content": canonical_json(_catalog_message(query, catalog))},
    ]


def _parse_action(content: str, *, interactive: bool) -> tuple[CommitAction | AskAction, bool]:
    payload, strict_json = parse_json_object(content)
    if not strict_json:
        raise ValueError("non_strict_json_envelope")
    try:
        if payload.get("action") == "commit":
            return CommitAction.model_validate(payload, strict=True), strict_json
        if interactive and payload.get("action") == "ask":
            return AskAction.model_validate(payload, strict=True), strict_json
    except ValidationError as error:
        raise ValueError("strict action schema validation failed") from error
    raise ValueError("model action is not allowed by the frozen arm")


def _validate_prediction(
    prediction: EndpointLabels, catalog: Mapping[str, Sequence[str]]
) -> None:
    for component in COMPONENTS:
        if getattr(prediction, component) not in catalog[component]:
            raise ValueError(f"prediction is outside canonical {component} vocabulary")


def score_prediction(
    prediction: EndpointLabels | None, case: Mapping[str, Any]
) -> dict[str, bool]:
    gold = case["private_gold"]
    scores = {
        component: bool(prediction is not None and getattr(prediction, component) == gold[component])
        for component in COMPONENTS
    }
    scores["joint"] = all(scores.values())
    return scores


def _model_call(
    *, client: OllamaChatClient, semantic_id: str, messages: Sequence[Mapping[str, Any]]
) -> Any:
    return client.chat(
        semantic_id=semantic_id,
        benchmark_label="interactive_ifttt",
        data_classification=DataClassification.PUBLIC,
        data_source=DataSource.INTERACTIVE_IFTTT,
        messages=messages,
        temperature=0.0,
        seed=42,
        think="low" if bool(getattr(client, "cloud", False)) else None,
    )


def execute_case(
    *,
    case: Mapping[str, Any],
    catalog: Mapping[str, Sequence[str]],
    arm: str,
    client: OllamaChatClient,
) -> dict[str, Any]:
    """Execute one immutable arm trajectory; validation failure is terminal."""
    case_id = str(case["case_id"])
    messages = build_initial_messages(query=str(case["input"]["query"]), catalog=catalog, arm=arm)
    asked: list[str] = []
    events: list[dict[str, Any]] = []
    usage = Counter()
    prediction: EndpointLabels | None = None
    terminal_status = "protocol_failure"
    failure_code: str | None = None
    strict_outputs = 0
    max_calls = 1 if arm == "same_model_one_shot" else MAX_QUESTIONS + 1

    try:
        for turn in range(max_calls):
            usage["semantic_calls"] += 1
            result = _model_call(
                client=client,
                semantic_id=f"yao-v1:{arm}:{client.model}:{case_id}:turn-{turn}",
                messages=messages,
            )
            usage.update({
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "physical_attempts": result.physical_attempts,
                "cache_hits": int(result.cache_hit),
            })
            usage["provider_latency_micros"] += round(result.provider_latency_ms * 1000)
            usage["queue_wait_micros"] += round(result.queue_wait_ms * 1000)
            action, strict_json = _parse_action(result.content, interactive=arm != "same_model_one_shot")
            strict_outputs += int(strict_json)
            if isinstance(action, CommitAction):
                _validate_prediction(action.prediction, catalog)
                prediction = action.prediction
                events.append({
                    "turn": turn,
                    "action": "commit",
                    "prediction": action.prediction.model_dump(),
                    "strict_json": strict_json,
                    "request_sha256": result.request_sha256,
                    "response_sha256": result.response_sha256,
                    "cache_hit": result.cache_hit,
                })
                terminal_status = "committed"
                break
            component = action.component
            if component in asked:
                raise ValueError("repeated_component_question")
            if len(asked) >= MAX_QUESTIONS:
                raise ValueError("question_budget_exhausted")
            frozen = case["simulator"]["frozen_answers"][component]
            official_answer = frozen["answer"]
            # This equality is rechecked at execution time, not merely during
            # preflight, so no caller can substitute an answer after validation.
            options = case["simulator"]["answer_options"][component]
            if options[frozen["answer_index"]] != official_answer:
                raise ValueError("frozen_answer_binding_changed")
            asked.append(component)
            events.append({
                "turn": turn,
                "action": "ask",
                "component": component,
                "question": _QUESTIONS[component],
                "official_answer": official_answer,
                "answer_index": frozen["answer_index"],
                "strict_json": strict_json,
                "request_sha256": result.request_sha256,
                "response_sha256": result.response_sha256,
                "cache_hit": result.cache_hit,
            })
            messages.append({"role": "assistant", "content": result.content})
            messages.append({
                "role": "user",
                "content": canonical_json({
                    "official_simulator_observation": {
                        "component": component,
                        "question": _QUESTIONS[component],
                        "answer": official_answer,
                    },
                    "questions_remaining": MAX_QUESTIONS - len(asked),
                    "instruction": "Ask another unresolved component or commit exactly four canonical labels.",
                }),
            })
        else:
            failure_code = "no_commit_within_budget"
    except (ValueError, ValidationError) as error:
        # An invalid model action is an incorrect final outcome, never a rank-1
        # fallback. Transport/provider and unexpected harness failures propagate
        # without a terminal record, so the semantic cache can safely resume.
        failure_code = str(error) or type(error).__name__

    scores = score_prediction(prediction, case)
    pseudo = case["private_gold"].get("pseudo_ask_labels")
    pseudo_components = {
        component for component, flag in zip(COMPONENTS, pseudo)
        if flag == 1
    } if isinstance(pseudo, list) and len(pseudo) == 4 else set()
    asked_set = set(asked)
    return {
        "schema_version": "round9-yao-case-result-v1",
        "case_id": case_id,
        "stratum": case["stratum"],
        "arm": arm,
        "terminal": True,
        "terminal_status": terminal_status,
        "failure_code": failure_code,
        "prediction": prediction.model_dump() if prediction else None,
        "scores": scores,
        "questions": asked,
        "question_count": len(asked),
        "ask_policy": {
            "true_positive": len(asked_set & pseudo_components),
            "false_positive": len(asked_set - pseudo_components),
            "false_negative": len(pseudo_components - asked_set),
            "true_negative": len(set(COMPONENTS) - asked_set - pseudo_components),
        },
        "events": events,
        "usage": {
            "semantic_calls": usage["semantic_calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "physical_attempts": usage["physical_attempts"],
            "provider_latency_ms": usage["provider_latency_micros"] / 1000,
            "queue_wait_ms": usage["queue_wait_micros"] / 1000,
            "strict_json_outputs": strict_outputs,
            "cache_hits": usage["cache_hits"],
        },
    }


def wilson_interval(successes: int, n: int) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    p = successes / n
    z2 = _Z_95 * _Z_95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = _Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [max(0.0, center - radius) * 100, min(1.0, center + radius) * 100]


def exact_mcnemar(current_only: int, reference_only: int) -> float:
    """Two-sided exact paired-binomial p-value for discordant outcomes."""
    if current_only < 0 or reference_only < 0:
        raise ValueError("McNemar counts must be nonnegative")
    discordant = current_only + reference_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(current_only, reference_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * tail)


def _latest_terminal(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not record.get("terminal"):
            raise ValueError("Yao result ledger contains a malformed/nonterminal record")
        if case_id in result:
            raise ValueError("Yao result ledger contains a duplicate terminal case")
        result[case_id] = record
    return result


def _validated_report_scores(record: Mapping[str, Any]) -> dict[str, bool]:
    """Reject ledger score coercion before computing reviewer-facing counts."""
    metric_names = (*COMPONENTS, "joint")
    scores = record.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != set(metric_names):
        raise ValueError("Yao result ledger has an invalid score schema")
    if any(type(scores[metric]) is not bool for metric in metric_names):
        raise ValueError("Yao result ledger scores must be JSON booleans")
    validated = {metric: scores[metric] for metric in metric_names}
    if validated["joint"] != all(validated[component] for component in COMPONENTS):
        raise ValueError("Yao result ledger has an inconsistent joint score")
    return validated


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    intended_n: int,
    paired_reference: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    latest = _latest_terminal(records)
    if len(latest) != intended_n:
        raise ValueError("aggregate requires one terminal record for every intended case")
    metric_names = (*COMPONENTS, "joint")
    scores_by_id = {
        case_id: _validated_report_scores(record)
        for case_id, record in latest.items()
    }
    numerators = {
        metric: sum(scores[metric] for scores in scores_by_id.values())
        for metric in metric_names
    }
    status_counts = Counter(str(record["terminal_status"]) for record in latest.values())
    failure_counts = Counter(
        str(record["failure_code"]) for record in latest.values() if record.get("failure_code")
    )
    integer_usage_keys = (
        "semantic_calls", "prompt_tokens", "completion_tokens", "physical_attempts",
        "strict_json_outputs", "cache_hits",
    )
    totals: dict[str, int | float] = {
        key: sum(int(record["usage"].get(key, 0)) for record in latest.values())
        for key in integer_usage_keys
    }
    totals.update({
        key: sum(float(record["usage"].get(key, 0)) for record in latest.values())
        for key in ("provider_latency_ms", "queue_wait_ms")
    })
    question_total = sum(int(record["question_count"]) for record in latest.values())
    ask_policy = {
        key: sum(int(record["ask_policy"].get(key, 0)) for record in latest.values())
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    asked_by_component = Counter(
        component for record in latest.values() for component in record.get("questions", [])
    )
    ask_precision_denominator = ask_policy["true_positive"] + ask_policy["false_positive"]
    ask_recall_denominator = ask_policy["true_positive"] + ask_policy["false_negative"]
    aggregate: dict[str, Any] = {
        "schema_version": "round9-yao-aggregate-v1",
        "n": intended_n,
        "raw_numerators": numerators,
        "percentages": {metric: numerators[metric] * 100 / intended_n for metric in metric_names},
        "confidence_intervals": {
            metric: {"method": "wilson_95", "percent": wilson_interval(numerators[metric], intended_n)}
            for metric in metric_names
        },
        "terminal_status_counts": dict(status_counts),
        "failure_counts": dict(failure_counts),
        "questions": {
            "total": question_total,
            "mean_per_case": question_total / intended_n,
            "zero_question_cases": sum(int(record["question_count"] == 0) for record in latest.values()),
            "count_distribution": dict(sorted(Counter(
                str(record["question_count"]) for record in latest.values()
            ).items())),
            "by_component": {component: asked_by_component[component] for component in COMPONENTS},
            "ask_policy_confusion": ask_policy,
            "ask_policy_precision": (
                ask_policy["true_positive"] / ask_precision_denominator
                if ask_precision_denominator else None
            ),
            "ask_policy_recall": (
                ask_policy["true_positive"] / ask_recall_denominator
                if ask_recall_denominator else None
            ),
        },
        "usage": {
            "totals": totals,
            "means_per_case": {key: value / intended_n for key, value in totals.items()},
        },
        "by_stratum": {},
    }
    strata = sorted({str(record["stratum"]) for record in latest.values()})
    for stratum in strata:
        subset = [record for record in latest.values() if record["stratum"] == stratum]
        stratum_numerators = {
            metric: sum(_validated_report_scores(record)[metric] for record in subset)
            for metric in metric_names
        }
        aggregate["by_stratum"][stratum] = {
            "n": len(subset),
            "raw_numerators": stratum_numerators,
            "percentages": {
                metric: stratum_numerators[metric] * 100 / len(subset)
                for metric in metric_names
            },
            "confidence_intervals": {
                metric: {
                    "method": "wilson_95",
                    "percent": wilson_interval(stratum_numerators[metric], len(subset)),
                }
                for metric in metric_names
            },
        }
    if paired_reference is not None:
        reference = _latest_terminal(paired_reference)
        if set(reference) != set(latest):
            raise ValueError("paired arm case IDs differ")
        reference_scores_by_id = {
            case_id: _validated_report_scores(record)
            for case_id, record in reference.items()
        }
        paired: dict[str, Any] = {}
        for metric in metric_names:
            cells = {"both_correct": 0, "current_only": 0, "reference_only": 0, "both_wrong": 0}
            for case_id in latest:
                current_ok = scores_by_id[case_id][metric]
                reference_ok = reference_scores_by_id[case_id][metric]
                key = (
                    "both_correct" if current_ok and reference_ok else
                    "current_only" if current_ok else
                    "reference_only" if reference_ok else "both_wrong"
                )
                cells[key] += 1
            paired[metric] = {
                **cells,
                "n": intended_n,
                "delta_percentage_points": (
                    cells["current_only"] - cells["reference_only"]
                ) * 100 / intended_n,
                "mcnemar_exact_two_sided_p": exact_mcnemar(
                    cells["current_only"], cells["reference_only"]
                ),
            }
        aggregate["paired_outcomes"] = paired
    return aggregate


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (canonical_json(dict(value)) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(value)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, (canonical_json(dict(record)) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_experiment(
    *,
    cases: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Sequence[str]],
    arm: str,
    client: OllamaChatClient,
    output_directory: Path,
    immutable_manifest: Mapping[str, Any],
    paired_reference_path: Path | None = None,
) -> dict[str, Any]:
    """Run/resume one arm under a single-writer lock and write exact aggregates."""
    if arm not in ARMS:
        raise ValueError(f"unknown Yao arm: {arm}")
    validate_final_cases(cases)
    validate_catalog({key: list(value) for key, value in catalog.items()})
    for case in cases:
        gold = case["private_gold"]
        for component in COMPONENTS:
            if gold[component] not in catalog[component]:
                raise ValueError(f"withheld {component} gold is absent from the official catalog")
    _mkdir_private(output_directory)
    manifest_path = output_directory / "manifest.json"
    records_path = output_directory / "records.jsonl"
    expected_manifest = dict(immutable_manifest)
    if manifest_path.exists():
        if read_json(manifest_path) != expected_manifest:
            raise RuntimeError("existing Yao ledger manifest differs")
    else:
        _write_exclusive(manifest_path, expected_manifest)
    os.chmod(manifest_path, 0o600)

    if paired_reference_path:
        paired_manifest_path = paired_reference_path.parent / "manifest.json"
        if not paired_manifest_path.exists():
            raise ValueError("paired reference has no immutable manifest")
        paired_manifest = read_json(paired_manifest_path)
        if paired_manifest.get("sample_sha256") != expected_manifest.get("sample_sha256"):
            raise ValueError("paired reference uses a different final sample")
        if paired_manifest.get("model_metadata") != expected_manifest.get("model_metadata"):
            raise ValueError("paired comparison requires the exact same model/provider")
        if paired_manifest.get("catalog_sha256") != expected_manifest.get("catalog_sha256"):
            raise ValueError("paired comparison requires the exact same endpoint vocabulary")
        current_protocol = expected_manifest.get("protocol_metadata", {})
        reference_protocol = paired_manifest.get("protocol_metadata", {})
        for parameter in ("temperature", "seed"):
            if reference_protocol.get(parameter) != current_protocol.get(parameter):
                raise ValueError(f"paired comparison changed model parameter: {parameter}")
        if paired_manifest.get("arm") == arm:
            raise ValueError("paired reference must be the other frozen arm")

    lock_descriptor = os.open(output_directory / ".run.lock", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.fchmod(lock_descriptor, 0o600)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process already owns this Yao arm ledger") from error
        existing = read_jsonl(records_path) if records_path.exists() else []
        completed = _latest_terminal(existing)
        for case in cases:
            case_id = str(case["case_id"])
            if case_id in completed:
                continue
            record = execute_case(case=case, catalog=catalog, arm=arm, client=client)
            _append_record(records_path, record)
            completed[case_id] = record
        records = read_jsonl(records_path)
        paired = None
        if paired_reference_path:
            paired = read_jsonl(paired_reference_path)
        aggregate = aggregate_records(records, intended_n=len(cases), paired_reference=paired)
        aggregate.update({
            "benchmark_label": "interactive_ifttt",
            "arm": arm,
            "hashes": {
                "sample_sha256": str(expected_manifest["sample_sha256"]),
                "catalog_sha256": str(expected_manifest["catalog_sha256"]),
                "prompt_sha256": str(expected_manifest["prompt_sha256"]),
            },
            "model_metadata": dict(expected_manifest["model_metadata"]),
            "protocol_metadata": dict(expected_manifest["protocol_metadata"]),
        })
        pricing = expected_manifest.get("pricing", {})
        prompt_price = pricing.get("prompt_cost_per_million_usd") if isinstance(pricing, Mapping) else None
        completion_price = pricing.get("completion_cost_per_million_usd") if isinstance(pricing, Mapping) else None
        totals = aggregate["usage"]["totals"]
        aggregate["cost"] = {
            "currency": "USD",
            "prompt_cost_per_million": prompt_price,
            "completion_cost_per_million": completion_price,
            "estimated_total": (
                totals["prompt_tokens"] * float(prompt_price) / 1_000_000
                + totals["completion_tokens"] * float(completion_price) / 1_000_000
            ) if prompt_price is not None and completion_price is not None else None,
        }
        _write_private_atomic(output_directory / "aggregate.json", aggregate)
        return aggregate
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def prompt_sha256(arm: str) -> str:
    """Hash the frozen prompt contract without any case or gold material."""
    messages = build_initial_messages(
        query="<PUBLIC_REQUEST>",
        catalog={component: ("<CANONICAL_LABEL>",) for component in COMPONENTS},
        arm=arm,
    )
    return sha256_text(canonical_json(messages))
