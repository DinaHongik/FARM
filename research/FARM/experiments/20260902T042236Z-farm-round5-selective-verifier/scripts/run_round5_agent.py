#!/usr/bin/env python3
"""Run one resumable round-five selective pair-card arm.

Only the already-consumed, SHA-256 ordered ``dev[0:600]`` region is eligible.
References are attached after ``resolve_pair_cards`` returns; neither the pair
policy nor the Ollama boundary receives gold labels or gold-derived ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ollama_pair_adapter import OllamaPairChooser, REFERENCE_KEYS


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
CONSUMED_SLICE_STOP = 600
ARM_SETTINGS = {
    "paircard_plain_m5": {"card_count": 5, "ranking": "plain"},
    "paircard_fused_m10": {"card_count": 10, "ranking": "rrf_plain_schema"},
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any, *, secret: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if secret and secret in payload:
        raise RuntimeError("secret persistence gate failed")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Mapping[str, Any], *, secret: str) -> None:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if secret and secret in payload:
        raise RuntimeError("secret persistence gate failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            group_id = row.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"JSONL line {line_number} lacks group_id")
            if group_id in seen:
                raise ValueError(f"duplicate completed case: {group_id}")
            seen.add(group_id)
            rows.append(row)
    return rows


def _hash_order(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row["group_id"]).encode()).hexdigest(),
            str(row["group_id"]),
        ),
    )


def selected_ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps([row["group_id"] for row in rows]).encode()
    ).hexdigest()


def select_consumed_rows(
    rows: Sequence[Mapping[str, Any]], *, start: int = 0, stop: int = CONSUMED_SLICE_STOP
) -> list[Mapping[str, Any]]:
    """Select only a subrange of the already-consumed hash-ordered dev slice."""
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= CONSUMED_SLICE_STOP
    ):
        raise ValueError("agent exploration must stay within hash-ordered dev[0:600]")
    if len(rows) < CONSUMED_SLICE_STOP:
        raise ValueError("candidate artifact cannot establish consumed dev[0:600]")
    return list(_hash_order(rows)[start:stop])


def _load_environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if " #" in value and not value.startswith(("'", '"')):
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key.strip()] = value
    return result


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def verify_model_digest(
    base_url: str,
    api_key: str,
    model: str,
    expected_digest: str,
    *,
    client: Any | None = None,
) -> None:
    """Fail closed if the configured cloud model identity has changed."""
    if client is None:
        from ollama import Client

        client = Client(
            host=base_url,
            headers={"Authorization": "Bearer " + api_key},
            timeout=240,
        )
    models = _plain(client.list()).get("models") or []
    record = next(
        (
            item
            for item in models
            if isinstance(item, Mapping)
            and (item.get("model") or item.get("name")) == model
        ),
        None,
    )
    digest = record.get("digest") if isinstance(record, Mapping) else None
    if not isinstance(digest, str) or not (
        digest == expected_digest or digest.startswith(expected_digest)
    ):
        raise RuntimeError("Ollama model/digest is unavailable or changed")


def _corpus_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = row.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("corpus row lacks url")
        if url in result:
            raise ValueError(f"duplicate corpus URL: {url}")
        result[url] = dict(row)
    return result


def _hydrate_candidates(
    rows: Sequence[Mapping[str, Any]], corpus: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        url = row.get("url")
        if not isinstance(url, str) or url not in corpus:
            raise ValueError(f"candidate absent from frozen corpus: {url!r}")
        candidate = dict(corpus[url])
        candidate.update(row)
        if candidate.get("channel") != corpus[url].get("channel"):
            raise ValueError(f"candidate service mismatch: {url}")
        candidate["retrieval_rank"] = int(row.get("retrieval_rank", index))
        output.append(candidate)
    return output


def inference_case(
    row: Mapping[str, Any], corpus_maps: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    case = {
        "group_id": row["group_id"],
        "query": row["query"],
        "trigger_candidates": _hydrate_candidates(
            row["trigger_candidates"][:10], corpus_maps["trigger"]
        ),
        "action_candidates": _hydrate_candidates(
            row["action_candidates"][:10], corpus_maps["action"]
        ),
    }
    leaked = REFERENCE_KEYS & set(case)
    if leaked:
        raise AssertionError(f"reference fields crossed inference seam: {sorted(leaked)}")
    if any(len(case[f"{side}_candidates"]) < 10 for side in ("trigger", "action")):
        raise ValueError("round-five policy requires ten candidates per side")
    return case


def _pair_string(value: Any) -> str:
    if isinstance(value, str):
        pieces = value.split(" || ")
        if len(pieces) != 2 or not all(pieces):
            raise ValueError(f"invalid pair ranking entry: {value!r}")
        return f"{pieces[0]} || {pieces[1]}"
    if isinstance(value, Mapping) and set(value) == {"trigger_url", "action_url"}:
        trigger = value.get("trigger_url")
        action = value.get("action_url")
        if isinstance(trigger, str) and trigger and isinstance(action, str) and action:
            return f"{trigger} || {action}"
    raise ValueError("pair ranking entries must contain only trigger_url/action_url")


def _pair_object(value: str) -> dict[str, str]:
    trigger, action = value.split(" || ", 1)
    return {"trigger_url": trigger, "action_url": action}


def load_pair_rankings(path: Path) -> dict[str, list[str]]:
    """Discard gold-derived pair_rank/scores and retain ranking identities only."""
    value = read_json(path)
    if isinstance(value, Mapping):
        if value.get("dataset_id") not in {None, DATASET_ID}:
            raise ValueError("pair result dataset identity mismatch")
        if value.get("split") not in {None, "dev"}:
            raise ValueError("pair result must be from dev")
        rows = value.get("rows_detail")
    else:
        rows = value
    if not isinstance(rows, list) or not rows:
        raise ValueError("pair result has no rows_detail")
    result: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pair result row is not an object")
        group_id = row.get("group_id")
        ranking = row.get("pair_ranking")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("pair result row lacks group_id")
        if group_id in result:
            raise ValueError(f"duplicate pair result group_id: {group_id}")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError(f"pair result row lacks ranking: {group_id}")
        normalized = [_pair_string(item) for item in ranking]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"pair ranking contains duplicates: {group_id}")
        result[group_id] = normalized
    return result


def rrf_pair_ranking(
    plain: Sequence[str], schema: Sequence[str], *, constant: int = 60
) -> list[dict[str, str]]:
    """Fuse two rankings without using scores, labels, or gold-derived ranks."""
    if isinstance(constant, bool) or not isinstance(constant, int) or constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranking in (plain, schema):
        seen: set[str] = set()
        for rank, raw in enumerate(ranking, start=1):
            pair = _pair_string(raw)
            if pair in seen:
                raise ValueError("RRF input ranking contains duplicates")
            seen.add(pair)
            scores[pair] = scores.get(pair, 0.0) + 1.0 / (constant + rank)
    return [
        _pair_object(pair)
        for pair in sorted(scores, key=lambda item: (-scores[item], item))
    ]


def arm_pair_ranking(
    arm: str,
    group_id: str,
    plain: Mapping[str, Sequence[str]],
    schema: Mapping[str, Sequence[str]] | None,
) -> list[dict[str, str]]:
    if arm not in ARM_SETTINGS:
        raise ValueError(f"unsupported arm: {arm}")
    if group_id not in plain:
        raise ValueError(f"plain pair ranking missing group: {group_id}")
    if ARM_SETTINGS[arm]["ranking"] == "plain":
        return [_pair_object(item) for item in plain[group_id]]
    if schema is None or group_id not in schema:
        raise ValueError(f"schema pair ranking missing group: {group_id}")
    return rrf_pair_ranking(plain[group_id], schema[group_id])


def validate_ranking_against_case(
    ranking: Sequence[Mapping[str, str]], case: Mapping[str, Any]
) -> None:
    allowed = {
        side: {candidate["url"] for candidate in case[f"{side}_candidates"]}
        for side in ("trigger", "action")
    }
    if not ranking:
        raise ValueError("pair ranking is empty")
    for pair in ranking:
        if pair.get("trigger_url") not in allowed["trigger"]:
            raise ValueError("pair ranking trigger escaped frozen top ten")
        if pair.get("action_url") not in allowed["action"]:
            raise ValueError("pair ranking action escaped frozen top ten")


def _pair_with_services(
    pair: Mapping[str, Any], corpus_maps: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, str]:
    trigger = pair.get("trigger_url")
    action = pair.get("action_url")
    if not isinstance(trigger, str) or trigger not in corpus_maps["trigger"]:
        raise ValueError(f"unknown trigger URL: {trigger!r}")
    if not isinstance(action, str) or action not in corpus_maps["action"]:
        raise ValueError(f"unknown action URL: {action!r}")
    return {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": str(corpus_maps["trigger"][trigger]["channel"]),
        "action_service": str(corpus_maps["action"][action]["channel"]),
    }


def _score_pair(prediction: Mapping[str, str], valid_pairs: Sequence[Mapping[str, str]]) -> dict[str, bool]:
    trigger = prediction["trigger_url"]
    action = prediction["action_url"]
    function_pairs = {(row["trigger_url"], row["action_url"]) for row in valid_pairs}
    service_pairs = {
        (row["trigger_service"], row["action_service"]) for row in valid_pairs
    }
    return {
        "function_trigger": trigger in {pair[0] for pair in function_pairs},
        "function_action": action in {pair[1] for pair in function_pairs},
        "function_joint": (trigger, action) in function_pairs,
        "service_trigger": prediction["trigger_service"] in {pair[0] for pair in service_pairs},
        "service_action": prediction["action_service"] in {pair[1] for pair in service_pairs},
        "service_joint": (
            prediction["trigger_service"], prediction["action_service"]
        ) in service_pairs,
    }


def _rank_bucket(row: Mapping[str, Any]) -> str:
    valid = {
        (pair["trigger_url"], pair["action_url"])
        for pair in row["valid_pairs"]
    }
    trigger_rank = {
        candidate["url"]: index
        for index, candidate in enumerate(row["trigger_candidates"][:10], start=1)
    }
    action_rank = {
        candidate["url"]: index
        for index, candidate in enumerate(row["action_candidates"][:10], start=1)
    }
    rank = min(
        (
            max(trigger_rank[trigger], action_rank[action])
            for trigger, action in valid
            if trigger in trigger_rank and action in action_rank
        ),
        default=None,
    )
    if rank == 1:
        return "rank1"
    if rank is not None and rank <= 5:
        return "rank2_5"
    if rank is not None and rank <= 10:
        return "rank6_10"
    return "outside10"


def evaluation_record(
    row: Mapping[str, Any],
    trace: Mapping[str, Any],
    corpus_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    arm: str,
    pair_ranking: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    valid_pairs = [_pair_with_services(pair, corpus_maps) for pair in row["valid_pairs"]]
    baseline = _pair_with_services(trace["baseline_pair"], corpus_maps)
    final_pair = _pair_with_services(trace["final_pair"], corpus_maps)
    proposer_raw = trace.get("proposer_pair")
    proposer_pair = (
        _pair_with_services(proposer_raw, corpus_maps)
        if isinstance(proposer_raw, Mapping)
        else None
    )
    alternative_raw = trace.get("alternative_pair")
    alternative_pair = (
        _pair_with_services(alternative_raw, corpus_maps)
        if isinstance(alternative_raw, Mapping)
        else None
    )
    deck: list[dict[str, str]] = []
    deck_seen: set[tuple[str, str]] = set()
    card_count = int(trace.get("policy", {}).get("card_count") or 0)
    for raw_pair in (trace["baseline_pair"], *(pair_ranking or ())):
        key = (raw_pair["trigger_url"], raw_pair["action_url"])
        if key in deck_seen:
            continue
        deck_seen.add(key)
        deck.append(_pair_with_services(raw_pair, corpus_maps))
        if card_count and len(deck) == card_count:
            break
    if pair_ranking is not None and len(deck) != card_count:
        raise ValueError("evaluation could not reconstruct the complete pair-card deck")
    function_gold = {
        (pair["trigger_url"], pair["action_url"]) for pair in valid_pairs
    }
    service_gold = {
        (pair["trigger_service"], pair["action_service"]) for pair in valid_pairs
    }
    oracle_function_positions = [
        index
        for index, pair in enumerate(deck, start=1)
        if (pair["trigger_url"], pair["action_url"]) in function_gold
    ]
    oracle_service_positions = [
        index
        for index, pair in enumerate(deck, start=1)
        if (pair["trigger_service"], pair["action_service"]) in service_gold
    ]
    decision_pairs = {
        "KEEP_TOP1": baseline,
        "ACCEPT_PROPOSAL": proposer_pair,
        "CHOOSE_ALT": alternative_pair,
    }
    verifier_choices = []
    for decision in trace.get("verifier_decisions") or []:
        pair = decision_pairs.get(str(decision))
        verifier_choices.append(
            {
                "decision": decision,
                "pair": pair,
                "score": _score_pair(pair, valid_pairs) if pair is not None else None,
            }
        )
    calls = list(trace.get("calls") or [])
    accounting = dict(trace.get("accounting") or {})
    protocol_valid = bool(accounting.get("complete")) and bool(calls) and all(
        isinstance(call, Mapping)
        and bool(call.get("accounting_complete"))
        and call.get("ok") is True
        for call in calls
    )
    return {
        "schema_version": "farm_round5_paircard_record_v1",
        "group_id": row["group_id"],
        "query": row["query"],
        "arm_id": arm,
        "valid_pairs": valid_pairs,
        "gold_service_pairs": [
            {
                "trigger_service": pair["trigger_service"],
                "action_service": pair["action_service"],
            }
            for pair in valid_pairs
        ],
        "trigger_candidates": [
            dict(candidate)
            | {"service": str(corpus_maps["trigger"][candidate["url"]]["channel"])}
            for candidate in row["trigger_candidates"]
        ],
        "action_candidates": [
            dict(candidate)
            | {"service": str(corpus_maps["action"][candidate["url"]]["channel"])}
            for candidate in row["action_candidates"]
        ],
        "baseline_pair": baseline,
        "final_pair": final_pair,
        "baseline_score": _score_pair(baseline, valid_pairs),
        "final_score": _score_pair(final_pair, valid_pairs),
        "proposer_score": _score_pair(proposer_pair, valid_pairs) if proposer_pair else None,
        "pair_card_deck": deck,
        "pair_card_oracle": {
            "function_joint": bool(oracle_function_positions),
            "service_joint": bool(oracle_service_positions),
            "first_function_joint_position": min(oracle_function_positions, default=None),
            "first_service_joint_position": min(oracle_service_positions, default=None),
        },
        "rank_bucket": _rank_bucket(row),
        "policy": dict(trace.get("policy") or {}),
        "proposer_pair": proposer_pair,
        "alternative_pair": alternative_pair,
        "card_ids": list(trace.get("card_ids") or []),
        "calls": calls,
        "verifier_decisions": list(trace.get("verifier_decisions") or []),
        "verifier_choices": verifier_choices,
        "stable_decision": trace.get("stable_decision"),
        "fallback_reason": trace.get("fallback_reason"),
        "accounting": accounting,
        "protocol_valid": protocol_valid,
        "fallback_used": not protocol_valid or trace.get("fallback_reason") is not None,
        "retained_baseline": bool(trace.get("retained_baseline")),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize no records")
    outcomes = (
        "function_trigger",
        "function_action",
        "function_joint",
        "service_trigger",
        "service_action",
        "service_joint",
    )
    metrics: dict[str, Any] = {"rows": len(records)}
    for outcome in outcomes:
        baseline_hits = sum(bool(row["baseline_score"][outcome]) for row in records)
        final_hits = sum(bool(row["final_score"][outcome]) for row in records)
        metrics[outcome] = {
            "baseline_hits": baseline_hits,
            "baseline_rate": baseline_hits / len(records),
            "agent_hits": final_hits,
            "agent_rate": final_hits / len(records),
            "absolute_delta": (final_hits - baseline_hits) / len(records),
        }
    recovered = sum(
        not row["baseline_score"]["function_joint"] and row["final_score"]["function_joint"]
        for row in records
    )
    regressed = sum(
        row["baseline_score"]["function_joint"] and not row["final_score"]["function_joint"]
        for row in records
    )
    changed = sum(row["baseline_pair"] != row["final_pair"] for row in records)
    proposer_rows = [row for row in records if row.get("proposer_pair") is not None]
    proposer_changed = [
        row for row in proposer_rows if row["proposer_pair"] != row["baseline_pair"]
    ]
    proposer_hits = sum(
        bool((row.get("proposer_score") or {}).get("function_joint")) for row in proposer_rows
    )
    proposer_recovered = sum(
        not row["baseline_score"]["function_joint"]
        and bool((row.get("proposer_score") or {}).get("function_joint"))
        for row in proposer_rows
    )
    proposer_regressed = sum(
        row["baseline_score"]["function_joint"]
        and not bool((row.get("proposer_score") or {}).get("function_joint"))
        for row in proposer_rows
    )
    metrics["proposer"] = {
        "decisions": len(proposer_rows),
        "changed_from_top1": len(proposer_changed),
        "function_joint_hits": proposer_hits,
        "function_joint_accuracy": proposer_hits / len(proposer_rows) if proposer_rows else None,
        "function_joint_accuracy_all_cases": proposer_hits / len(records),
        "recoveries": proposer_recovered,
        "regressions": proposer_regressed,
        "net_vs_top1": proposer_recovered - proposer_regressed,
    }
    verification_rows = [row for row in records if row.get("verifier_decisions")]
    verifier_choices = [
        choice
        for row in verification_rows
        for choice in row.get("verifier_choices", [])
    ]
    scored_verifier_choices = [
        choice for choice in verifier_choices if isinstance(choice.get("score"), Mapping)
    ]
    verifier_choice_hits = sum(
        bool(choice["score"]["function_joint"]) for choice in scored_verifier_choices
    )
    agreement_rows = [
        row
        for row in verification_rows
        if len(row.get("verifier_decisions", [])) == 2
        and row["verifier_decisions"][0] == row["verifier_decisions"][1]
        and row["verifier_decisions"][0] != "INVALID"
    ]
    non_abstain_agreement_rows = [
        row
        for row in agreement_rows
        if row["verifier_decisions"][0]
        in {"KEEP_TOP1", "ACCEPT_PROPOSAL", "CHOOSE_ALT"}
    ]
    agreed_choice_hits = sum(
        bool(row["verifier_choices"][0]["score"]["function_joint"])
        for row in non_abstain_agreement_rows
    )
    verifier_recovered_from_proposer = sum(
        not bool((row.get("proposer_score") or {}).get("function_joint"))
        and row["final_score"]["function_joint"]
        for row in verification_rows
    )
    verifier_regressed_from_proposer = sum(
        bool((row.get("proposer_score") or {}).get("function_joint"))
        and not row["final_score"]["function_joint"]
        for row in verification_rows
    )
    metrics["verifier"] = {
        "cases_run": len(verification_rows),
        "logical_decisions": len(verifier_choices),
        "scored_non_abstain_decisions": len(scored_verifier_choices),
        "decision_function_joint_hits": verifier_choice_hits,
        "decision_conditional_accuracy": (
            verifier_choice_hits / len(scored_verifier_choices)
            if scored_verifier_choices
            else None
        ),
        "agreement_cases": len(agreement_rows),
        "agreement_rate": (
            len(agreement_rows) / len(verification_rows) if verification_rows else None
        ),
        "non_abstain_agreement_cases": len(non_abstain_agreement_rows),
        "agreed_choice_function_joint_hits": agreed_choice_hits,
        "agreed_choice_conditional_accuracy": (
            agreed_choice_hits / len(non_abstain_agreement_rows)
            if non_abstain_agreement_rows
            else None
        ),
        "second_call_changed_decision": sum(
            len(row.get("verifier_decisions", [])) == 2
            and row["verifier_decisions"][0] != row["verifier_decisions"][1]
            for row in verification_rows
        ),
        "final_recoveries_vs_proposer": verifier_recovered_from_proposer,
        "final_regressions_vs_proposer": verifier_regressed_from_proposer,
        "net_final_vs_proposer": verifier_recovered_from_proposer
        - verifier_regressed_from_proposer,
    }
    oracle_function = sum(
        bool(row.get("pair_card_oracle", {}).get("function_joint")) for row in records
    )
    oracle_service = sum(
        bool(row.get("pair_card_oracle", {}).get("service_joint")) for row in records
    )
    covered_rows = [
        row for row in records if row.get("pair_card_oracle", {}).get("function_joint")
    ]
    metrics["pair_card_oracle"] = {
        "function_joint_covered": oracle_function,
        "function_joint_coverage": oracle_function / len(records),
        "service_joint_covered": oracle_service,
        "service_joint_coverage": oracle_service / len(records),
        "mean_card_count": sum(len(row.get("pair_card_deck", [])) for row in records)
        / len(records),
        "proposer_accuracy_when_function_gold_in_deck": (
            sum(bool((row.get("proposer_score") or {}).get("function_joint")) for row in covered_rows)
            / len(covered_rows)
            if covered_rows
            else None
        ),
        "final_accuracy_when_function_gold_in_deck": (
            sum(bool(row["final_score"]["function_joint"]) for row in covered_rows)
            / len(covered_rows)
            if covered_rows
            else None
        ),
    }
    accounting_keys = ("logical_calls", "api_attempts", "tool_calls", "catalog_reads")
    evidence_keys = (
        "proposer_pair_cards",
        "proposer_endpoints",
        "verifier_pair_cards",
        "verifier_endpoints",
        "verifier_unique_schema_catalog_rows",
    )
    metrics["comparison"] = {
        "changed": changed,
        "recovered": recovered,
        "regressed": regressed,
        "net_function_joint": recovered - regressed,
        "mcnemar_discordant": {"baseline_wrong_agent_right": recovered, "baseline_right_agent_wrong": regressed},
    }
    metrics["protocol"] = {
        "valid_rows": sum(bool(row["protocol_valid"]) for row in records),
        "fallback_rows": sum(bool(row["fallback_used"]) for row in records),
        **{
            key: sum(int(row.get("accounting", {}).get(key) or 0) for row in records)
            for key in accounting_keys
        },
        "prompt_tokens": sum(
            int(row.get("accounting", {}).get("usage", {}).get("prompt_tokens") or 0)
            for row in records
        ),
        "completion_tokens": sum(
            int(row.get("accounting", {}).get("usage", {}).get("completion_tokens") or 0)
            for row in records
        ),
        "total_tokens": sum(
            int(row.get("accounting", {}).get("usage", {}).get("total_tokens") or 0)
            for row in records
        ),
        "latency_seconds": sum(
            float(row.get("accounting", {}).get("usage", {}).get("latency_seconds") or 0.0)
            for row in records
        ),
        "evidence_presentations": {
            key: sum(
                int(
                    row.get("accounting", {})
                    .get("evidence_presentations", {})
                    .get(key)
                    or 0
                )
                for row in records
            )
            for key in evidence_keys
        },
    }
    metrics["per_case_averages"] = {
        "logical_calls": metrics["protocol"]["logical_calls"] / len(records),
        "api_attempts": metrics["protocol"]["api_attempts"] / len(records),
        "tool_calls": metrics["protocol"]["tool_calls"] / len(records),
        "catalog_reads": metrics["protocol"]["catalog_reads"] / len(records),
        "prompt_tokens": metrics["protocol"]["prompt_tokens"] / len(records),
        "completion_tokens": metrics["protocol"]["completion_tokens"] / len(records),
        "total_tokens": metrics["protocol"]["total_tokens"] / len(records),
        "latency_seconds": metrics["protocol"]["latency_seconds"] / len(records),
        "evidence_presentations": {
            key: metrics["protocol"]["evidence_presentations"][key] / len(records)
            for key in evidence_keys
        },
    }
    metrics["stable_decisions"] = dict(
        sorted(Counter(str(row.get("stable_decision")) for row in records).items())
    )
    metrics["by_rank_bucket"] = {}
    for bucket in ("rank1", "rank2_5", "rank6_10", "outside10"):
        subset = [row for row in records if row["rank_bucket"] == bucket]
        metrics["by_rank_bucket"][bucket] = {
            "rows": len(subset),
            "baseline_joint_rate": (
                sum(bool(row["baseline_score"]["function_joint"]) for row in subset) / len(subset)
                if subset
                else None
            ),
            "agent_joint_rate": (
                sum(bool(row["final_score"]["function_joint"]) for row in subset) / len(subset)
                if subset
                else None
            ),
            "logical_calls": sum(
                int(row.get("accounting", {}).get("logical_calls") or 0) for row in subset
            ),
        }
    return metrics


def _load_inputs(
    candidates_path: Path,
    manifest_path: Path,
    data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("split") != "dev"
        or manifest.get("output_sha256") != sha256_file(candidates_path)
    ):
        raise ValueError("candidate artifact identity mismatch")
    rows = read_json(candidates_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate artifact is empty")
    data_manifest = read_json(data_root / "manifest.json")
    if data_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("Dataset-v2 identity mismatch")
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for side, relative in (("trigger", "corpus/triggers.json"), ("action", "corpus/actions.json")):
        path = data_root / relative
        expected = data_manifest["artifacts"][relative]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"frozen {side} corpus hash mismatch")
        corpus = read_json(path)
        if not isinstance(corpus, list) or not corpus:
            raise ValueError(f"frozen {side} corpus is empty")
        maps[side] = _corpus_map(corpus)
    return rows, maps, dict(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--pair-plain-results", type=Path)
    parser.add_argument("--pair-schema-results", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SETTINGS), required=True)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--expected-digest", default="221b330d11a8")
    parser.add_argument("--key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--host-env", default="OLLAMA_CLOUD_HOST")
    parser.add_argument("--slice-start", type=int)
    parser.add_argument("--slice-stop", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    if args.smoke < 0 or args.smoke > 10:
        raise ValueError("smoke must be between zero and ten")
    run_root = args.run_root.resolve()
    matrix_path = run_root / "EXPERIMENT_MATRIX.json"
    matrix = read_json(matrix_path)
    if matrix.get("dataset_id") != DATASET_ID:
        raise ValueError("experiment matrix dataset identity mismatch")
    sources = matrix.get("sources")
    if not isinstance(sources, Mapping):
        sources = {}

    def source_path(argument: Path | None, key: str) -> Path:
        value: Any = argument if argument is not None else sources.get(key)
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"missing --{key.replace('_', '-')} and matrix source {key}")
        return Path(value).resolve()

    candidates_path = source_path(args.candidates, "candidate_dev")
    manifest_path = source_path(args.candidate_manifest, "candidate_manifest")
    data_root = source_path(args.data_root, "data_root")
    plain_path = source_path(args.pair_plain_results, "pair_plain")
    raw_schema: Any = args.pair_schema_results or sources.get("pair_schema")
    schema_path = Path(raw_schema).resolve() if raw_schema else None
    if args.arm == "paircard_fused_m10" and schema_path is None:
        raise ValueError("fused arm requires --pair-schema-results")
    pinned_sources = (
        (candidates_path, "candidate_dev_sha256"),
        (manifest_path, "candidate_manifest_sha256"),
        (plain_path, "pair_plain_sha256"),
        (schema_path, "pair_schema_sha256"),
    )
    for path, hash_key in pinned_sources:
        expected = sources.get(hash_key)
        if path is not None and isinstance(expected, str) and sha256_file(path) != expected:
            raise ValueError(f"matrix-pinned source hash changed: {hash_key}")

    split = matrix.get("splits", {}).get("agent_exploratory", {})
    if split.get("ordering") != "sha256(group_id) ascending":
        raise ValueError("matrix agent split must use SHA-256 group ordering")
    slice_start = args.slice_start if args.slice_start is not None else split.get("slice_start")
    slice_stop = args.slice_stop if args.slice_stop is not None else split.get("slice_stop")
    if not isinstance(slice_start, int) or isinstance(slice_start, bool):
        raise ValueError("agent slice_start is unconfigured")
    if not isinstance(slice_stop, int) or isinstance(slice_stop, bool):
        raise ValueError("agent slice_stop is unconfigured")

    all_rows, corpus_maps, candidate_manifest = _load_inputs(
        candidates_path, manifest_path, data_root
    )
    consumed = select_consumed_rows(all_rows, start=0, stop=CONSUMED_SLICE_STOP)
    selected = select_consumed_rows(
        all_rows, start=slice_start, stop=slice_stop
    )
    expected_selected_hash = split.get("selected_group_ids_sha256")
    if (
        slice_start == split.get("slice_start")
        and slice_stop == split.get("slice_stop")
        and isinstance(expected_selected_hash, str)
        and selected_ids_hash(selected) != expected_selected_hash
    ):
        raise ValueError("matrix agent split identity hash changed")
    arm_spec = next(
        (
            item
            for item in matrix.get("agent_arms", [])
            if isinstance(item, Mapping) and item.get("id") == args.arm
        ),
        None,
    )
    if not isinstance(arm_spec, Mapping):
        raise ValueError("arm is absent from experiment matrix")
    if int(arm_spec.get("card_count", -1)) != int(ARM_SETTINGS[args.arm]["card_count"]):
        raise ValueError("matrix arm card_count conflicts with frozen runtime")
    if int(arm_spec.get("scope", -1)) != len(selected):
        raise ValueError("matrix arm scope conflicts with selected exploratory slice")
    output_root = run_root
    mode = "consumed_dev_exploration"
    if args.smoke:
        selected = selected[: args.smoke]
        output_root = run_root / "smoke"
        mode = "smoke_consumed_dev"

    plain_rankings = load_pair_rankings(plain_path)
    schema_rankings = load_pair_rankings(schema_path) if schema_path else None
    environment = _load_environment(args.env_file.resolve())
    key = os.environ.get(args.key_env) or environment.get(args.key_env)
    host = os.environ.get(args.host_env) or environment.get(args.host_env)
    if not key or not host:
        raise RuntimeError("Ollama credential/host slot is unconfigured")
    verify_model_digest(host, key, args.model, args.expected_digest)

    from pair_card_policy import PairCardPolicy, resolve_pair_cards

    setting = ARM_SETTINGS[args.arm]
    policy = PairCardPolicy(
        card_count=int(setting["card_count"]),
        verify=True,
        candidate_depth=10,
    )
    records_path = output_root / "records" / f"{args.arm}.jsonl"
    attempts_path = output_root / "attempts" / f"{args.arm}.jsonl"
    progress_path = output_root / "manifests" / f"{args.arm}.progress.json"
    result_path = output_root / "results" / f"{args.arm}.json"
    if result_path.exists():
        print(json.dumps({"status": "already_complete", "output": str(result_path)}))
        return
    if records_path.exists() and not args.resume:
        raise RuntimeError("work file exists; use --resume")

    policy_path = run_root / "scripts" / "pair_card_policy.py"
    adapter_path = run_root / "scripts" / "ollama_pair_adapter.py"
    runtime_path = run_root / "scripts" / "run_round5_agent.py"
    binding = {
        "run_id": run_root.name,
        "mode": mode,
        "arm": args.arm,
        "dataset_id": DATASET_ID,
        "split": "dev",
        "eligible_slice": {
            "ordering": "sha256(group_id) ascending",
            "slice_start": 0,
            "slice_stop": CONSUMED_SLICE_STOP,
            "selected_group_ids_sha256": selected_ids_hash(consumed),
            "status": "previously_consumed_by_round3_and_round4",
        },
        "executed_slice": {
            "slice_start": slice_start,
            "slice_stop": slice_stop,
            "selected_group_ids_sha256": selected_ids_hash(selected),
            "target_rows": len(selected),
        },
        "candidate_sha256": candidate_manifest["output_sha256"],
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "matrix_sha256": sha256_file(matrix_path),
        "pair_plain_result_sha256": sha256_file(plain_path),
        "pair_schema_result_sha256": sha256_file(schema_path) if schema_path else None,
        "trigger_corpus_sha256": sha256_file(data_root / "corpus/triggers.json"),
        "action_corpus_sha256": sha256_file(data_root / "corpus/actions.json"),
        "policy_sha256": sha256_file(policy_path),
        "adapter_sha256": sha256_file(adapter_path),
        "runtime_sha256": sha256_file(runtime_path),
        "model": args.model,
        "model_digest": args.expected_digest,
        "host_env": args.host_env,
        "key_env": args.key_env,
        "protocol": {
            "transport": "ollama_openai_compatible_chat_completions",
            "temperature": 0,
            "seed": 42,
            "reasoning_effort": "none",
            "max_tokens": 768,
            "max_attempts_per_logical_call": 2,
            "attempt_write_ahead_log": True,
            "gold_in_inference": False,
        },
        "policy": {
            "card_count": int(setting["card_count"]),
            "ranking": setting["ranking"],
            "verify": True,
            "candidate_depth": 10,
        },
    }

    completed = load_jsonl(records_path) if args.resume else []
    if args.resume and progress_path.exists():
        prior = read_json(progress_path).get("binding")
        if prior != binding:
            raise RuntimeError("resume binding changed")
    done = {row["group_id"] for row in completed}
    selected_ids = {row["group_id"] for row in selected}
    if not done <= selected_ids:
        raise RuntimeError("work file contains cases outside selected consumed-dev slice")

    write_json_atomic(
        progress_path,
        {
            "phase": "running",
            "binding": binding,
            "completed_rows": len(completed),
            "target_rows": len(selected),
        },
        secret=key,
    )
    chooser = OllamaPairChooser(
        base_url=host,
        api_key=key,
        model=args.model,
        journal_path=attempts_path,
        timeout=240,
        max_tokens=768,
        reasoning_effort="none",
        retry_delay=0.25,
        seed=42,
    )
    try:
        for row in selected:
            if row["group_id"] in done:
                continue
            case = inference_case(row, corpus_maps)
            ranking = arm_pair_ranking(
                args.arm, row["group_id"], plain_rankings, schema_rankings
            )
            validate_ranking_against_case(ranking, case)
            trace = resolve_pair_cards(case, ranking, policy, chooser, chooser)
            record = evaluation_record(
                row, trace, corpus_maps, arm=args.arm, pair_ranking=ranking
            )
            append_jsonl(records_path, record, secret=key)
            completed.append(record)
            done.add(row["group_id"])
            write_json_atomic(
                progress_path,
                {
                    "phase": "running",
                    "binding": binding,
                    "completed_rows": len(completed),
                    "target_rows": len(selected),
                    "last_group_id": row["group_id"],
                },
                secret=key,
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "completed": len(completed),
                        "target": len(selected),
                        "stable_decision": record["stable_decision"],
                        "protocol_valid": record["protocol_valid"],
                    }
                ),
                flush=True,
            )
    finally:
        chooser.close()

    order = {row["group_id"]: index for index, row in enumerate(selected)}
    completed.sort(key=lambda row: order[row["group_id"]])
    if len(completed) != len(selected):
        raise RuntimeError("controller terminated before all selected cases were persisted")
    result = {
        "status": "completed",
        "binding": binding,
        "metrics": summarize_records(completed),
    }
    write_json_atomic(result_path, result, secret=key)
    write_json_atomic(
        progress_path,
        {
            "phase": "complete",
            "binding": binding,
            "completed_rows": len(completed),
            "target_rows": len(selected),
            "output": str(result_path),
            "output_sha256": sha256_file(result_path),
        },
        secret=key,
    )
    print(json.dumps({"status": "completed", "arm": args.arm, "rows": len(completed)}))


if __name__ == "__main__":
    main()
