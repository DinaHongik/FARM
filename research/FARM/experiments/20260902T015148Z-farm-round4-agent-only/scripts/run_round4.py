#!/usr/bin/env python3
"""Run one frozen, resumable FARM round-four agent arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_metrics import evaluate_records
from ollama_adapter import OllamaSelector
from resolver import ResolverPolicy, resolve


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
REFERENCE_KEYS = {
    "valid_pairs",
    "gold_trigger_urls",
    "gold_action_urls",
    "gold_channel_pairs",
    "observed_pairs",
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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
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
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if secret and secret in serialized:
        raise RuntimeError("secret persistence gate failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        identifier = value.get("group_id", value.get("case_id"))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"JSONL line {line_number} lacks a case identifier")
        if identifier in identifiers:
            raise ValueError(f"duplicate completed case: {identifier}")
        identifiers.add(identifier)
        records.append(value)
    return records


def selected_ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    identifiers = [row["group_id"] for row in rows]
    return hashlib.sha256(json.dumps(identifiers).encode()).hexdigest()


def _hash_order(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (hashlib.sha256(str(row["group_id"]).encode()).hexdigest(), row["group_id"]),
    )


def select_frozen_rows(
    rows: Sequence[Mapping[str, Any]], split: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if split.get("ordering") != "sha256(group_id) ascending":
        raise ValueError("unsupported confirmation ordering")
    start = split.get("slice_start")
    stop = split.get("slice_stop")
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= len(rows)
    ):
        raise ValueError("invalid confirmation slice")
    selected = list(_hash_order(rows)[start:stop])
    if selected_ids_hash(selected) != split.get("selected_group_ids_sha256"):
        raise ValueError("selected group identity hash changed")
    return selected


def _corpus_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("url")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("corpus row lacks URL")
        if identifier in result:
            raise ValueError(f"duplicate corpus URL: {identifier}")
        result[identifier] = dict(row)
    return result


def _hydrate_candidates(
    candidates: Sequence[Mapping[str, Any]], corpus: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for candidate in candidates:
        identifier = candidate.get("url")
        if not isinstance(identifier, str) or identifier not in corpus:
            raise ValueError(f"candidate is absent from frozen corpus: {identifier!r}")
        source = dict(corpus[identifier])
        if candidate.get("channel") != source.get("channel"):
            raise ValueError(f"candidate service mismatch: {identifier}")
        source.update(candidate)
        hydrated.append(source)
    return hydrated


def inference_case(
    row: Mapping[str, Any],
    corpora: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    hierarchy: bool,
) -> dict[str, Any]:
    maps = {side: _corpus_map(corpora[side]) for side in ("trigger", "action")}
    case = {
        "group_id": row["group_id"],
        "query": row["query"],
        "trigger_candidates": _hydrate_candidates(row["trigger_candidates"], maps["trigger"]),
        "action_candidates": _hydrate_candidates(row["action_candidates"], maps["action"]),
    }
    if hierarchy:
        case["trigger_catalog"] = [dict(item) for item in corpora["trigger"]]
        case["action_catalog"] = [dict(item) for item in corpora["action"]]
    contaminated = REFERENCE_KEYS & set(case)
    if contaminated:
        raise AssertionError(f"reference fields crossed inference seam: {sorted(contaminated)}")
    return case


def _pair_with_services(
    pair: Mapping[str, Any], corpus_maps: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, str]:
    trigger = pair.get("trigger_url")
    action = pair.get("action_url")
    if not isinstance(trigger, str) or trigger not in corpus_maps["trigger"]:
        raise ValueError(f"unknown trigger URL in pair: {trigger!r}")
    if not isinstance(action, str) or action not in corpus_maps["action"]:
        raise ValueError(f"unknown action URL in pair: {action!r}")
    return {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": str(corpus_maps["trigger"][trigger]["channel"]),
        "action_service": str(corpus_maps["action"][action]["channel"]),
    }


def evaluation_record(
    row: Mapping[str, Any],
    trace: Mapping[str, Any],
    corpus_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    arm_id: str,
) -> dict[str, Any]:
    valid_pairs = [_pair_with_services(pair, corpus_maps) for pair in row["valid_pairs"]]
    baseline = _pair_with_services(trace["baseline_pair"], corpus_maps)
    final_pair = _pair_with_services(trace["final_pair"], corpus_maps)
    calls = list(trace.get("calls") or [])
    accounting = dict(trace.get("accounting") or {})
    protocol_valid = bool(accounting.get("complete")) and bool(calls) and all(
        isinstance(call, Mapping) and call.get("ok") is True for call in calls
    )
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    for side in ("trigger", "action"):
        candidate_rows[f"{side}_candidates"] = [
            dict(candidate) | {
                "service": str(corpus_maps[side][candidate["url"]]["channel"]),
            }
            for candidate in row[f"{side}_candidates"]
        ]
    return {
        "schema_version": "farm_round4_record_v1",
        "group_id": row["group_id"],
        "query": row["query"],
        "arm_id": arm_id,
        "valid_pairs": valid_pairs,
        "gold_service_pairs": [
            {
                "trigger_service": pair["trigger_service"],
                "action_service": pair["action_service"],
            }
            for pair in valid_pairs
        ],
        **candidate_rows,
        "baseline_pair": baseline,
        "final_pair": final_pair,
        "policy": dict(trace.get("policy") or {}),
        "selected_service_pair": trace.get("selected_service_pair"),
        "calls": calls,
        "accounting": accounting,
        "protocol_valid": protocol_valid,
        "fallback_used": not protocol_valid,
        "retained_baseline": bool(trace.get("retained_baseline")),
    }


def _load_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
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
        values[key.strip()] = value
    return values


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def verify_model(base_url: str, api_key: str, model: str, expected_digest: str) -> None:
    from ollama import Client

    client = Client(
        host=base_url,
        headers={"Authorization": "Bearer " + api_key},
        timeout=240,
    )
    models = _plain(client.list()).get("models") or []
    record = next(
        (item for item in models if (item.get("model") or item.get("name")) == model),
        None,
    )
    digest = record.get("digest") if isinstance(record, Mapping) else None
    if not isinstance(digest, str) or not (
        digest == expected_digest or digest.startswith(expected_digest) or expected_digest.startswith(digest)
    ):
        raise RuntimeError("Ollama model/digest is unavailable or changed")


def _load_inputs(
    candidates_path: Path,
    candidate_manifest_path: Path,
    data_root: Path,
    matrix: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidate_manifest = read_json(candidate_manifest_path)
    if (
        candidate_manifest.get("dataset_id") != DATASET_ID
        or candidate_manifest.get("split") != "dev"
        or candidate_manifest.get("output_sha256") != sha256_file(candidates_path)
    ):
        raise ValueError("candidate artifact identity mismatch")
    if matrix.get("dataset_id") != DATASET_ID:
        raise ValueError("experiment matrix dataset identity mismatch")
    data_manifest = read_json(data_root / "manifest.json")
    if data_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("Dataset-v2 identity mismatch")
    corpora: dict[str, list[dict[str, Any]]] = {}
    for side, relative in (("trigger", "corpus/triggers.json"), ("action", "corpus/actions.json")):
        path = data_root / relative
        expected = data_manifest["artifacts"][relative]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"frozen {side} corpus hash mismatch")
        value = read_json(path)
        if not isinstance(value, list) or not value:
            raise ValueError(f"frozen {side} corpus is empty")
        corpora[side] = value
    rows = read_json(candidates_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate artifact is empty")
    return rows, corpora, candidate_manifest


def _arm_policy(arm: Mapping[str, Any], prompt: str) -> ResolverPolicy:
    arm_id = arm["id"]
    if arm_id == "gpt_service_to_function":
        return ResolverPolicy.service_hierarchy(10, view="plain", instruction=prompt)
    if arm_id == "gpt_top10_reversed_audit":
        return ResolverPolicy.function_topk(10, view="plain", instruction=prompt, order="reversed")
    if arm_id == "gpt_top10_plain":
        return ResolverPolicy.function_topk(10, view="plain", instruction=prompt)
    if arm_id in {"gpt_top5_plain", "gemma_top5_plain_replication"}:
        return ResolverPolicy.function_topk(5, view="plain", instruction=prompt)
    raise ValueError(f"unsupported executable arm: {arm_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="run N exploratory rows, not confirmation")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    matrix_path = run_root / "EXPERIMENT_MATRIX.json"
    matrix = read_json(matrix_path)
    arm = next((item for item in matrix["arms"] if item["id"] == args.arm), None)
    if arm is None or arm.get("model") is None:
        raise ValueError("arm is absent or non-executable")
    if args.smoke < 0 or args.smoke > 10:
        raise ValueError("smoke size must be between zero and ten")

    all_rows, corpora, candidate_manifest = _load_inputs(
        args.candidates.resolve(),
        args.candidate_manifest.resolve(),
        args.data_root.resolve(),
        matrix,
    )
    if args.smoke:
        # Smoke cases come only from the already-exploratory first slice.
        selected = list(_hash_order(all_rows)[: args.smoke])
        output_root = run_root / "smoke"
        mode = "smoke_exploratory"
    else:
        selected = select_frozen_rows(all_rows, matrix["split"])
        selected = selected[: int(arm["scope"])]
        output_root = run_root
        mode = "confirmation"

    model_spec = matrix["models"][arm["model"]]
    environment = _load_environment(args.env_file.resolve())
    key = os.environ.get(model_spec["key_env"]) or environment.get(model_spec["key_env"])
    host = os.environ.get(model_spec["host_env"]) or environment.get(model_spec["host_env"])
    if not key or not host:
        raise RuntimeError("Ollama credential/host slot is unconfigured")
    verify_model(host, key, model_spec["model"], model_spec["digest"])

    prompt_path = run_root / "configs" / "resolver_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    policy = _arm_policy(arm, prompt)
    records_path = output_root / "records" / f"{args.arm}.jsonl"
    attempts_path = output_root / "attempts" / f"{args.arm}.jsonl"
    progress_path = output_root / "manifests" / f"{args.arm}.progress.json"
    result_path = output_root / "results" / f"{args.arm}.json"
    if result_path.exists():
        print(json.dumps({"status": "already_complete", "arm": args.arm, "output": str(result_path)}))
        return
    if records_path.exists() and not args.resume:
        raise RuntimeError("work file exists; use --resume")

    binding = {
        "run_id": matrix["run_id"],
        "mode": mode,
        "arm": args.arm,
        "dataset_id": DATASET_ID,
        "candidate_sha256": candidate_manifest["output_sha256"],
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest.resolve()),
        "trigger_corpus_sha256": sha256_file(args.data_root.resolve() / "corpus/triggers.json"),
        "action_corpus_sha256": sha256_file(args.data_root.resolve() / "corpus/actions.json"),
        "matrix_sha256": sha256_file(matrix_path),
        "prompt_sha256": sha256_file(prompt_path),
        "resolver_sha256": sha256_file(run_root / "scripts/resolver.py"),
        "adapter_sha256": sha256_file(run_root / "scripts/ollama_adapter.py"),
        "evaluator_sha256": sha256_file(run_root / "scripts/agent_metrics.py"),
        "model": model_spec["model"],
        "model_digest": model_spec["digest"],
        "selected_group_ids_sha256": selected_ids_hash(selected),
        "target_rows": len(selected),
        "protocol": matrix["shared_protocol"],
        "effective_reasoning_effort": model_spec.get(
            "reasoning_effort", matrix["shared_protocol"]["reasoning_effort"]
        ),
        "policy": {
            "mode": policy.mode,
            "top_k": policy.top_k,
            "view": policy.view,
            "order": policy.order,
        },
    }

    completed = load_jsonl(records_path) if args.resume else []
    if args.resume and progress_path.exists():
        prior_binding = read_json(progress_path).get("binding")
        if prior_binding != binding:
            raise RuntimeError("resume binding changed")
    done = {record["group_id"] for record in completed}
    if not done <= {row["group_id"] for row in selected}:
        raise RuntimeError("work file contains cases outside frozen selection")
    maps = {side: _corpus_map(corpora[side]) for side in ("trigger", "action")}
    hierarchy = policy.mode == "service_hierarchy"

    write_json_atomic(progress_path, {
        "phase": "running",
        "binding": binding,
        "completed_rows": len(completed),
        "target_rows": len(selected),
    })
    selector = OllamaSelector(
        base_url=host,
        api_key=key,
        model=model_spec["model"],
        journal_path=attempts_path,
        timeout=240,
        max_tokens=int(matrix["shared_protocol"]["max_tokens"]),
        reasoning_effort=str(model_spec.get(
            "reasoning_effort", matrix["shared_protocol"]["reasoning_effort"]
        )),
        retry_delay=float(matrix["shared_protocol"]["retry_delay_seconds"]),
    )
    try:
        for row in selected:
            if row["group_id"] in done:
                continue
            case = inference_case(row, corpora, hierarchy=hierarchy)
            trace = resolve(case, policy, selector)
            record = evaluation_record(row, trace, maps, arm_id=args.arm)
            append_jsonl(records_path, record, secret=key)
            completed.append(record)
            done.add(row["group_id"])
            write_json_atomic(progress_path, {
                "phase": "running",
                "binding": binding,
                "completed_rows": len(completed),
                "target_rows": len(selected),
                "last_group_id": row["group_id"],
            })
            print(json.dumps({
                "arm": args.arm,
                "completed": len(completed),
                "target": len(selected),
                "protocol_valid": record["protocol_valid"],
            }), flush=True)
    finally:
        selector.close()

    order = {row["group_id"]: index for index, row in enumerate(selected)}
    completed.sort(key=lambda record: order[record["group_id"]])
    if len(completed) != len(selected):
        raise RuntimeError("controller terminated before all selected cases were persisted")
    metrics = evaluate_records(
        completed,
        baseline_arm="baseline",
        arm_names=["baseline", "agent"],
        bootstrap_iterations=5000,
        bootstrap_seed=42,
        min_service_support=10,
    )
    result = {"status": "completed", "binding": binding, "metrics": metrics}
    serialized = json.dumps(result, sort_keys=True)
    if key in serialized:
        raise RuntimeError("final result secret-exclusion gate failed")
    write_json_atomic(result_path, result)
    write_json_atomic(progress_path, {
        "phase": "complete",
        "binding": binding,
        "completed_rows": len(completed),
        "target_rows": len(selected),
        "output": str(result_path),
        "output_sha256": sha256_file(result_path),
    })
    print(json.dumps({"status": "completed", "arm": args.arm, "rows": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
