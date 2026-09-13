#!/usr/bin/env python3
"""Atomic sequential supervisor for one complete FARM condition."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    DATASET_ID,
    assert_fresh_pipeline_output,
    assert_one_visible_gpu,
    config_from_path,
    git_state,
    read_json,
    sha256_file,
    source_hashes,
    update_global_status,
    utc_now,
    validate_dataset,
    validate_negative_derived,
    validate_service_derived,
    verify_training_done,
    verify_evaluation_done,
    write_json,
)

PHASES = ("trigger_training", "action_training", "dev_evaluation", "complete")


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "sentence-transformers", "transformers", "datasets", "ragas", "langgraph", "langchain-ollama", "ollama"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def cuda_runtime_version() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def gpu_metadata(physical_gpu: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(physical_gpu),
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    return dict(zip(("index", "name", "uuid", "driver_version", "memory_total_mib"), fields))


def run_child(command: list[str], dry_run: bool) -> None:
    print("COMMAND", json.dumps(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    config = config_from_path(config_path)
    dataset_manifest = validate_dataset(data_root)
    if dataset_manifest["dataset_id"] != DATASET_ID:
        raise AssertionError("dataset identity changed after validation")

    negative_manifest = None
    if config["hard_negatives"]:
        negative_manifest = validate_negative_derived(config, run_root, data_root)

    if args.dry_run:
        physical_gpu = str(config["gpu"])
        gpu = {"index": physical_gpu, "name": "dry-run"}
    else:
        physical_gpu = assert_one_visible_gpu(config["gpu"])
        gpu = gpu_metadata(config["gpu"])

    if args.dry_run:
        scope_root = run_root / "dry_run"
    else:
        scope_root = run_root if args.scope == "full" else run_root / "smoke"
        if not args.resume:
            assert_fresh_pipeline_output(run_root, config["experiment_id"], args.scope)
    manifest_path = scope_root / "manifests" / config["experiment_id"] / "pipeline.json"
    pid_path = scope_root / "pids" / f"{config['experiment_id']}.pid"
    lock_path = scope_root / "pids" / f"{config['experiment_id']}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"pipeline already holds lock: {lock_path}") from exc
    write_json(pid_path, {"pid": os.getpid(), "started_at": utc_now(), "scope": args.scope})

    env_file = run_root / "ENVIRONMENT.json"
    local_provenance = read_json(env_file) if env_file.is_file() else {}
    derived_hash = None
    if config["hard_negatives"]:
        derived_hash = negative_manifest["derived_view_hash"]
    else:
        pair_rel = f"pairs/trigger_encoder_train_{config['view']}.json"
        derived_hash = dataset_manifest["artifacts"][pair_rel]["sha256"]
    pipeline_manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "run_id": config["run_id"],
        "experiment_id": config["experiment_id"],
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "derived_view_hash": derived_hash,
        "source_code_hashes": source_hashes(run_root),
        "config_sha256": sha256_file(config_path),
        "git": local_provenance.get("local_git", git_state(project_root)),
        "server_git": git_state(project_root),
        "python": platform.python_version(),
        "packages": package_versions(),
        "cuda_runtime": cuda_runtime_version(),
        "cuda_visible_devices": physical_gpu,
        "gpu": gpu,
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "seed": config["seed"],
        "hyperparameters": {
            key: config.get(key)
            for key in (
                "epochs", "batch_size", "learning_rate", "warmup_ratio",
                "weight_decay", "max_seq_length", "scale", "hard_negatives",
                "cached_loss", "mini_batch_size", "negative_source",
            )
        },
        "prompt_policy": "EmbeddingGemma built-in query/document prompts",
        "precision": "fp32",
        "start_time": utc_now(),
        "tmux_session": config["session"] if args.scope == "full" else f"{config['session']}-smoke",
        "pid": os.getpid(),
        "scope": args.scope,
        "current_phase": "preflight",
        "trigger_checkpoint": str(scope_root / "checkpoints" / config["experiment_id"] / "trigger" / "final"),
        "action_checkpoint": str(scope_root / "checkpoints" / config["experiment_id"] / "action" / "final"),
        "resume_command": (
            f"CUDA_VISIBLE_DEVICES={config['gpu']} .venv/bin/python {run_root / 'scripts/launch_pipeline.py'} "
            f"--config {config_path} --project-root {project_root} --run-root {run_root} "
            f"--data-root {data_root} --scope {args.scope} --resume"
        ),
        "evaluation_command": (
            f"CUDA_VISIBLE_DEVICES={config['gpu']} .venv/bin/python {run_root / 'scripts/evaluate.py'} "
            f"--config {config_path} --run-root {run_root} --data-root {data_root} --scope {args.scope}"
        ),
        "log_paths": {
            "pipeline": str(run_root / "logs" / config["experiment_id"] / "pipeline.log"),
            "trigger_progress": str(scope_root / "manifests" / config["experiment_id"] / "train_trigger.progress.json"),
            "action_progress": str(scope_root / "manifests" / config["experiment_id"] / "train_action.progress.json"),
        },
        "authorization_note": "current user explicitly authorized four GPUs 0-3; launcher rechecks occupancy and never kills processes",
    }
    write_json(manifest_path, pipeline_manifest)
    if args.scope == "full" and not args.dry_run:
        status_fields = {"pid": os.getpid(), "session": config["session"], "started_at": utc_now()}
        if not args.resume:
            status_fields["phase"] = "preflight"
        update_global_status(run_root, config["experiment_id"], status_fields)

    common = [
        sys.executable,
        str(SCRIPT_DIR / "train_one.py"),
        "--config", str(config_path),
        "--project-root", str(project_root),
        "--run-root", str(run_root),
        "--data-root", str(data_root),
        "--scope", args.scope,
    ]
    if args.resume:
        common.append("--resume")
    if args.scope == "smoke":
        common.extend(["--max-steps", "5", "--max-rows", "256"])

    try:
        for kind, phase in (("trigger", "trigger_training"), ("action", "action_training")):
            done_path = scope_root / "manifests" / config["experiment_id"] / f"train_{kind}.done.json"
            if done_path.is_file():
                final_root = scope_root / "checkpoints" / config["experiment_id"] / kind / "final"
                verify_training_done(done_path, final_root, config, kind, sha256_file(config_path))
                print(f"SKIP hash-verified completed phase: {done_path}", flush=True)
                continue
            pipeline_manifest["current_phase"] = phase
            pipeline_manifest["updated_at"] = utc_now()
            write_json(manifest_path, pipeline_manifest)
            if args.scope == "full" and not args.dry_run:
                update_global_status(run_root, config["experiment_id"], {"phase": phase})
            run_child(common + ["--kind", kind], args.dry_run)

        eval_done = scope_root / "manifests" / config["experiment_id"] / "evaluate.done.json"
        if eval_done.is_file():
            verify_evaluation_done(eval_done, config)
            print(f"SKIP hash-verified completed phase: {eval_done}", flush=True)
        else:
            pipeline_manifest["current_phase"] = "dev_evaluation"
            pipeline_manifest["updated_at"] = utc_now()
            write_json(manifest_path, pipeline_manifest)
            if args.scope == "full" and not args.dry_run:
                update_global_status(run_root, config["experiment_id"], {"phase": "dev_evaluation"})
            eval_command = [
                sys.executable,
                str(SCRIPT_DIR / "evaluate.py"),
                "--config", str(config_path),
                "--run-root", str(run_root),
                "--data-root", str(data_root),
                "--scope", args.scope,
            ]
            if args.scope == "smoke":
                eval_command.extend(["--max-rows", "64"])
            run_child(eval_command, args.dry_run)

        pipeline_manifest.update({
            "status": "dry_run_passed" if args.dry_run else "completed",
            "current_phase": "complete",
            "completed_at": utc_now(),
        })
        write_json(manifest_path, pipeline_manifest)
        if args.scope == "full" and not args.dry_run:
            update_global_status(run_root, config["experiment_id"], {"phase": "complete", "completed_at": utc_now()})
    except BaseException as exc:
        pipeline_manifest.update({
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        write_json(manifest_path, pipeline_manifest)
        if args.scope == "full" and not args.dry_run:
            update_global_status(
                run_root,
                config["experiment_id"],
                {"phase": "failed", "error_type": type(exc).__name__, "error": str(exc)},
            )
        traceback.print_exc()
        raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()
