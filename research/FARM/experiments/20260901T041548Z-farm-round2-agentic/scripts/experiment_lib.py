#!/usr/bin/env python3
"""Shared, isolated utilities for the FARM reviewer experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import fcntl
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
ALLOWED_TRAIN_SPLIT = "encoder_train"
ALLOWED_EVAL_SPLIT = "dev"
EXPERIMENT_IDS = ("e3_seed1337", "e3_seed2025", "cache0", "random4")
CONFIG_NAMES = (
    "e3_seed1337.yaml", "e3_seed2025.yaml", "cache0.yaml", "random4.yaml",
)
CORPUS_FILE = {"trigger": "triggers.json", "action": "actions.json"}
LEGAL_PHASES = {
    "not_started": {"not_started", "preflight", "failed"},
    "preflight": {"preflight", "trigger_training", "failed"},
    "trigger_training": {"trigger_training", "action_training", "failed"},
    "action_training": {"action_training", "dev_evaluation", "failed"},
    "dev_evaluation": {"dev_evaluation", "complete", "failed"},
    "complete": {"complete"},
    "failed": {"failed", "preflight", "trigger_training", "action_training", "dev_evaluation"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_from_path(path: Path) -> dict[str, Any]:
    config = read_json(path)
    required = {
        "run_id", "experiment_id", "gpu", "session", "level", "view",
        "base_model", "base_model_revision", "seed", "epochs", "batch_size", "learning_rate",
        "warmup_ratio", "weight_decay", "max_seq_length", "scale",
        "hard_negatives", "negative_source",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing fields: {sorted(missing)}")
    if config["experiment_id"] not in EXPERIMENT_IDS:
        raise ValueError("unknown experiment_id")
    if config["negative_source"] not in {"none", "mined4", "random4"}:
        raise ValueError("unknown negative_source")
    if bool(config["hard_negatives"]) != (config["negative_source"] != "none"):
        raise ValueError("hard_negatives and negative_source disagree")
    return config


def validate_dataset(data_root: Path, full_hash_check: bool = True) -> dict[str, Any]:
    manifest_path = data_root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(
            f"Dataset v2 ID mismatch: {manifest.get('dataset_id')} != {DATASET_ID}"
        )
    required = {
        "splits/encoder_train.json", "splits/reranker_train.json",
        "splits/dev.json", "splits/test.json", "corpus/triggers.json",
        "corpus/actions.json", "pairs/trigger_encoder_train_plain.json",
        "pairs/action_encoder_train_plain.json",
        "pairs/trigger_encoder_train_schema.json",
        "pairs/action_encoder_train_schema.json",
    }
    missing = required - set(manifest.get("artifacts", {}))
    if missing:
        raise RuntimeError(f"Dataset v2 manifest lacks artifacts: {sorted(missing)}")
    if full_hash_check:
        failures = []
        for rel, metadata in manifest["artifacts"].items():
            path = data_root / rel
            if rel == "splits/test.json":
                # Its manifest entry locks the identity; experiment code must not
                # open locked-test bytes during repeated validation.
                if not path.is_file():
                    failures.append(rel)
                continue
            if not path.is_file() or sha256_file(path) != metadata["sha256"]:
                failures.append(rel)
        if failures:
            raise RuntimeError(f"Dataset v2 artifact hash mismatch: {failures}")
    return manifest


def assert_split_allowed(split: str, purpose: str) -> None:
    expected = ALLOWED_TRAIN_SPLIT if purpose == "train" else ALLOWED_EVAL_SPLIT
    if split != expected:
        raise ValueError(f"{purpose} split must be {expected!r}; got {split!r}")
    if split == "test":
        raise AssertionError("locked test split is prohibited")


def assert_one_visible_gpu(expected_physical: int | None = None) -> str:
    entries = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if len(entries) != 1 or not entries[0].isdigit():
        raise RuntimeError(f"exactly one numeric CUDA_VISIBLE_DEVICES entry required, got {entries}")
    if expected_physical is not None and int(entries[0]) != expected_physical:
        raise RuntimeError(f"expected physical GPU {expected_physical}, got {entries[0]}")
    return entries[0]


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=project_root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()
    status = run("status", "--porcelain=v1")
    return {
        "revision": run("rev-parse", "HEAD") or "not-a-git-repository",
        "dirty": bool(status),
        "dirty_path_count": len(status.splitlines()) if status else 0,
    }


def _choose_display(rows: list[dict[str, Any]]) -> str:
    names = [str(row.get("channel_display") or "").strip() for row in rows]
    names = [name for name in names if name]
    if not names:
        slug = str(rows[0]["channel"])
        return slug.replace("_", " ").strip()
    counts = Counter(names)
    return sorted(counts, key=lambda name: (-counts[name], name.casefold(), name))[0]


def _corpus_maps(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_url: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind in ("trigger", "action"):
        rows = read_json(data_root / "corpus" / CORPUS_FILE[kind])
        by_kind[kind] = rows
        for row in rows:
            if row["url"] in by_url:
                raise AssertionError(f"duplicate corpus URL: {row['url']}")
            by_url[row["url"]] = row
    return by_url, by_kind


def project_service_group(group: dict[str, Any], by_url: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    projected = {
        (
            str(by_url[pair["trigger_url"]]["channel"]),
            str(by_url[pair["action_url"]]["channel"]),
        )
        for pair in group["valid_pairs"]
    }
    if not projected:
        raise AssertionError(f"group has no valid service pairs: {group['group_id']}")
    return [
        {"trigger_service": trigger, "action_service": action}
        for trigger, action in sorted(projected)
    ]


def build_service_view(data_root: Path, out_root: Path) -> dict[str, Any]:
    """Build E0 without inventing service pairs or touching Dataset v2."""
    manifest = validate_dataset(data_root)
    by_url, by_kind = _corpus_maps(data_root)
    destination = out_root / "e0_service"
    destination.mkdir(parents=True, exist_ok=True)

    service_corpora: dict[str, list[dict[str, Any]]] = {}
    for kind in ("trigger", "action"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_kind[kind]:
            grouped[str(row["channel"])].append(row)
        service_rows = []
        for channel in sorted(grouped):
            source_rows = grouped[channel]
            display = _choose_display(source_rows)
            if not display or "\n" in display:
                raise AssertionError(f"invalid service-only display name: {display!r}")
            service_rows.append({
                "service_id": channel,
                "text": display,
                "source_function_urls": sorted(row["url"] for row in source_rows),
                "display_name_variants": sorted({
                    str(row.get("channel_display") or "").strip()
                    for row in source_rows if str(row.get("channel_display") or "").strip()
                }),
            })
        service_corpora[kind] = service_rows
        write_json(destination / f"corpus_{kind}.json", service_rows)

    train_groups = read_json(data_root / "splits" / "encoder_train.json")
    dev_groups = read_json(data_root / "splits" / "dev.json")
    train_rows: dict[str, list[dict[str, Any]]] = {"trigger": [], "action": []}
    service_text = {
        kind: {row["service_id"]: row["text"] for row in service_corpora[kind]}
        for kind in ("trigger", "action")
    }
    seen: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        "trigger": {}, "action": {}
    }
    for group in train_groups:
        valid_pairs = project_service_group(group, by_url)
        for pair_index, pair in enumerate(valid_pairs):
            for kind in ("trigger", "action"):
                service = pair[f"{kind}_service"]
                key = (group["group_id"], service)
                row = seen[kind].get(key)
                provenance = {
                    "group_id": group["group_id"],
                    "service_pair": pair,
                    "valid_pair_index": pair_index,
                    "source_applet_urls": group["source_applet_urls"],
                }
                if row is None:
                    row = {
                        "anchor": group["query"],
                        "group_id": group["group_id"],
                        "label_id": service,
                        "positive": service_text[kind][service],
                        "provenance": [],
                    }
                    seen[kind][key] = row
                    train_rows[kind].append(row)
                row["provenance"].append(provenance)
    for kind in ("trigger", "action"):
        train_rows[kind].sort(key=lambda row: (row["group_id"], row["label_id"]))
        write_json(destination / f"train_{kind}.json", train_rows[kind])

    projected_dev = []
    for group in dev_groups:
        valid_pairs = project_service_group(group, by_url)
        projected_dev.append({
            "group_id": group["group_id"],
            "query": group["query"],
            "valid_pairs": valid_pairs,
            "source_function_pairs": group["valid_pairs"],
            "source_applet_urls": group["source_applet_urls"],
        })
    write_json(destination / "dev.json", projected_dev)

    artifacts = {}
    for path in sorted(destination.glob("*.json")):
        artifacts[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": manifest["dataset_id"],
        "source_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "policy": "project each observed valid function pair, then deduplicate; never Cartesian product",
        "document_policy": "canonical service display name only",
        "counts": {
            "trigger_services": len(service_corpora["trigger"]),
            "action_services": len(service_corpora["action"]),
            "trigger_train_rows": len(train_rows["trigger"]),
            "action_train_rows": len(train_rows["action"]),
            "dev_groups": len(projected_dev),
            "dev_service_pairs": sum(len(row["valid_pairs"]) for row in projected_dev),
        },
        "artifacts": artifacts,
    }
    result["derived_view_hash"] = sha256_json({"artifacts": artifacts, "counts": result["counts"]})
    write_json(destination / "manifest.json", result)
    return result


def function_pair_path(data_root: Path, kind: str, view: str) -> Path:
    if view not in {"plain", "schema"}:
        raise ValueError(f"invalid function view: {view}")
    return data_root / "pairs" / f"{kind}_encoder_train_{view}.json"


def load_training_rows(
    config: dict[str, Any], data_root: Path, run_root: Path, kind: str
) -> list[dict[str, Any]]:
    assert_split_allowed("encoder_train", "train")
    if config["level"] == "service":
        validate_service_derived(run_root, data_root)
        rows = read_json(run_root / "derived_data" / "e0_service" / f"train_{kind}.json")
    elif config["hard_negatives"]:
        root = negative_root(config, run_root)
        validate_negative_derived(config, run_root, data_root)
        rows = read_json(root / f"{kind}.json")
    else:
        rows = read_json(function_pair_path(data_root, kind, config["view"]))
    if not rows:
        raise RuntimeError(f"empty {kind} training rows")
    return rows


def training_document_map(
    config: dict[str, Any], data_root: Path, run_root: Path, kind: str
) -> dict[str, str]:
    if config["level"] == "service":
        corpus = read_json(run_root / "derived_data" / "e0_service" / f"corpus_{kind}.json")
        return {row["service_id"]: row["text"] for row in corpus}
    corpus = read_json(data_root / "corpus" / CORPUS_FILE[kind])
    return {row["url"]: row[f"text_{config['view']}"] for row in corpus}


def validate_training_rows(
    config: dict[str, Any], data_root: Path, run_root: Path, kind: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    documents = training_document_map(config, data_root, run_root, kind)
    labels = []
    negative_count = 0
    for row in rows:
        label = row.get("label_id") if config["level"] == "service" else row.get("label_url")
        if label not in documents or row["positive"] != documents[label]:
            raise AssertionError(f"positive/document mismatch for {label}")
        labels.append(label)
        negatives = row.get("negatives") or ([] if "negative" not in row else [row["negative"]])
        if config["hard_negatives"]:
            if len(negatives) != config["hard_negatives"]:
                raise AssertionError(f"expected {config['hard_negatives']} negatives")
            valid_urls = set(row.get("valid_label_urls") or [])
            negative_urls = list(row.get("negative_urls") or [])
            if len(negative_urls) != config["hard_negatives"] or len(set(negative_urls)) != len(negative_urls):
                raise AssertionError("negative URLs are missing or duplicated")
            if valid_urls & set(negative_urls):
                raise AssertionError(f"hard negative overlaps valid labels: {valid_urls & set(negative_urls)}")
            for negative_url, negative_text in zip(negative_urls, negatives):
                if negative_url not in documents or documents[negative_url] != negative_text:
                    raise AssertionError(f"negative URL/text mismatch: {negative_url}")
            if row["positive"] in negatives:
                raise AssertionError("positive text used as hard negative")
            negative_count += len(negatives)
    return {
        "rows": len(rows), "unique_labels": len(set(labels)),
        "negative_count": negative_count,
    }


def validate_service_derived(run_root: Path, data_root: Path) -> dict[str, Any]:
    root = run_root / "derived_data" / "e0_service"
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "passed" or manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("invalid E0 derived manifest identity/status")
    if manifest.get("source_manifest_sha256") != sha256_file(data_root / "manifest.json"):
        raise RuntimeError("E0 source Dataset v2 manifest changed")
    for name, metadata in manifest.get("artifacts", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"E0 derived artifact hash mismatch: {name}")
    return manifest


def negative_root(config: dict[str, Any], run_root: Path) -> Path:
    source = config.get("negative_source")
    if source == "mined4":
        return run_root / "derived_data" / "e3_hard_negatives"
    if source == "random4":
        return run_root / "derived_data" / "random4"
    raise ValueError(f"experiment has no explicit-negative artifact: {source!r}")


def validate_negative_derived(
    config: dict[str, Any], run_root: Path, data_root: Path,
) -> dict[str, Any]:
    root = negative_root(config, run_root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "passed" or manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("invalid negative-derived manifest identity/status")
    if manifest.get("source_manifest_sha256") != sha256_file(data_root / "manifest.json"):
        raise RuntimeError("negative-derived Dataset v2 manifest changed")
    if manifest.get("negative_source") not in {None, config["negative_source"]}:
        raise RuntimeError("negative-derived strategy mismatch")
    for kind in ("trigger", "action"):
        side = manifest.get("sides", {}).get(kind, {})
        output = root / f"{kind}.json"
        source = function_pair_path(data_root, kind, "schema")
        corpus = data_root / "corpus" / CORPUS_FILE[kind]
        checks = (
            (output, side.get("output_sha256"), "output"),
            (source, side.get("source_sha256"), "source"),
            (corpus, side.get("corpus_sha256"), "corpus"),
        )
        for path, expected, label in checks:
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"negative-derived {kind} {label} hash mismatch")
        if side.get("rows") != read_json(data_root / "manifest.json")["artifacts"][
            f"pairs/{kind}_encoder_train_schema.json"
        ]["rows"]:
            raise RuntimeError(f"negative-derived {kind} row count changed")
    return manifest


def validate_hardneg_derived(run_root: Path, data_root: Path) -> dict[str, Any]:
    """Compatibility helper for the copied seed-42 mined-negative artifact."""
    config = {"negative_source": "mined4"}
    return validate_negative_derived(config, run_root, data_root)


def verify_training_done(
    done_path: Path,
    final_root: Path,
    config: dict[str, Any],
    kind: str,
    config_sha256: str,
) -> dict[str, Any]:
    record = read_json(done_path)
    expected = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "experiment_id": config["experiment_id"],
        "kind": kind,
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"done sentinel mismatch for {kind}: {key}")
    if Path(record.get("checkpoint", "")).resolve() != final_root.resolve() or not final_root.is_dir():
        raise RuntimeError(f"done sentinel checkpoint missing/mismatched for {kind}")
    for relative, expected_hash in record.get("checkpoint_file_hashes", {}).items():
        path = final_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch for {kind}: {relative}")
    if not record.get("checkpoint_file_hashes"):
        raise RuntimeError(f"done sentinel lacks checkpoint hashes for {kind}")
    return record


def verify_evaluation_done(done_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    record = read_json(done_path)
    if record.get("status") != "passed" or record.get("dataset_id") != DATASET_ID:
        raise RuntimeError("evaluation done sentinel identity/status mismatch")
    if record.get("experiment_id") != config["experiment_id"] or record.get("split") != "dev":
        raise RuntimeError("evaluation done sentinel experiment/split mismatch")
    result_path = Path(record.get("result_path", ""))
    if not result_path.is_file() or sha256_file(result_path) != record.get("result_sha256"):
        raise RuntimeError("evaluation result missing or hash mismatch")
    return record


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def assert_legal_transition(previous: str, following: str) -> None:
    if previous not in LEGAL_PHASES or following not in LEGAL_PHASES[previous]:
        raise RuntimeError(f"illegal pipeline transition: {previous} -> {following}")


def assert_new_or_resumable_output(path: Path, resume: bool) -> None:
    if path.exists() and not resume:
        raise RuntimeError(f"refusing existing output without --resume: {path}")


def pipeline_output_evidence(run_root: Path, experiment_id: str, scope: str = "full") -> list[Path]:
    """Return artifacts proving that a real pipeline already started.

    Dry-run artifacts live under ``dry_run/`` and are deliberately excluded.
    A non-resume launcher must never overwrite any path returned here.
    """
    if scope not in {"full", "smoke"}:
        raise ValueError(f"unknown scope: {scope}")
    scope_root = run_root if scope == "full" else run_root / "smoke"
    evidence: list[Path] = []
    checkpoint_root = scope_root / "checkpoints" / experiment_id
    manifest_root = scope_root / "manifests" / experiment_id
    pid_path = scope_root / "pids" / f"{experiment_id}.pid"
    result_path = scope_root / "results" / f"{experiment_id}_dev.json"
    if checkpoint_root.exists():
        evidence.extend(path for path in checkpoint_root.rglob("*") if path.is_file())
    if manifest_root.exists():
        evidence.extend(
            path for path in manifest_root.rglob("*")
            if path.is_file()
            and path.name not in {"dry_run.json", "launch_command.txt", "resume_command.txt"}
        )
    if pid_path.is_file():
        evidence.append(pid_path)
    if result_path.is_file():
        evidence.append(result_path)
    return sorted(set(path.resolve() for path in evidence))


def assert_fresh_pipeline_output(run_root: Path, experiment_id: str, scope: str = "full") -> None:
    evidence = pipeline_output_evidence(run_root, experiment_id, scope)
    if evidence:
        sample = ", ".join(str(path) for path in evidence[:3])
        suffix = " ..." if len(evidence) > 3 else ""
        raise RuntimeError(
            f"refusing non-resume launch for {experiment_id}; existing pipeline artifacts: "
            f"{sample}{suffix}. Use resume_all.sh/--resume."
        )


def redact_text(text: str, secrets: Iterable[str]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    result = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+", r"\1[REDACTED]", result)
    result = re.sub(r"(?i)(api[_-]?key|password|passwd)(\s*[=:]\s*)[^\s,\"']+", r"\1\2[REDACTED]", result)
    return result


def source_hashes(run_root: Path) -> dict[str, str]:
    paths = sorted((run_root / "scripts").glob("*")) + sorted((run_root / "configs").glob("*"))
    return {
        str(path.relative_to(run_root)): sha256_file(path)
        for path in paths if path.is_file() and "__pycache__" not in path.parts
    }


def update_global_status(run_root: Path, experiment_id: str, fields: dict[str, Any]) -> None:
    """Serialize concurrent pipeline updates through a separate stable lock file."""
    status_path = run_root / "STATUS.json"
    lock_path = run_root / ".status.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = read_json(status_path)
        experiment = status.setdefault("experiments", {}).setdefault(experiment_id, {})
        if "phase" in fields:
            assert_legal_transition(experiment.get("phase", "not_started"), fields["phase"])
        experiment.update(fields)
        phases = [item.get("phase", "not_started") for item in status.get("experiments", {}).values()]
        agent_phases = [item.get("phase", "not_started") for item in status.get("agents", {}).values()]
        if "failed" in phases or "failed" in agent_phases:
            status["state"] = "failed"
        elif phases and all(phase == "complete" for phase in phases) and (
            not agent_phases or all(phase == "complete" for phase in agent_phases)
        ):
            status["state"] = "complete"
        elif any(phase != "not_started" for phase in phases + agent_phases):
            status["state"] = "running_detached"
        else:
            status["state"] = "prepared"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def update_agent_status(run_root: Path, agent_id: str, fields: dict[str, Any]) -> None:
    """Serialize Ollama progress into the canonical status document."""
    status_path = run_root / "STATUS.json"
    lock_path = run_root / ".status.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = read_json(status_path)
        status.setdefault("agents", {}).setdefault(agent_id, {}).update(fields)
        phases = [item.get("phase", "not_started") for item in status.get("experiments", {}).values()]
        agent_phases = [item.get("phase", "not_started") for item in status.get("agents", {}).values()]
        if "failed" in phases or "failed" in agent_phases:
            status["state"] = "failed"
        elif phases and all(phase == "complete" for phase in phases) and agent_phases and all(
            phase == "complete" for phase in agent_phases
        ):
            status["state"] = "complete"
        elif any(phase != "not_started" for phase in phases + agent_phases):
            status["state"] = "running_detached"
        else:
            status["state"] = "prepared"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def prompt_text(model: Any, role: str, text: str) -> str:
    if role not in {"query", "document"}:
        raise ValueError(role)
    prompt = getattr(model, "prompts", {}).get(role)
    if not prompt:
        raise RuntimeError(f"selected model lacks required {role!r} prompt")
    return f"{prompt}{text}"


def saved_model_kwargs(model_path: Path) -> dict[str, Any]:
    """Work around Transformers 4.57 tokenizer heuristic for locally saved Gemma."""
    config_path = model_path / "config.json"
    if config_path.is_file() and read_json(config_path).get("model_type") == "gemma3_text":
        return {"processor_kwargs": {"fix_mistral_regex": False}}
    return {}
