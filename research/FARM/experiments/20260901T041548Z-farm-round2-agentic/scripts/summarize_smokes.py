#!/usr/bin/env python3
"""Summarize real five-step smoke evidence and derive a coarse full-run budget."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    CONFIG_NAMES,
    config_from_path,
    load_training_rows,
    read_json,
    sha256_file,
    utc_now,
    validate_dataset,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    validate_dataset(data_root)
    output = {
        "status": "passed",
        "created_at": utc_now(),
        "method": (
            "Measured wall time from real five-step trigger/action smoke runs. Full-run seconds are a coarse "
            "linear upper-planning estimate and are not reported as experimental results."
        ),
        "experiments": {},
    }
    for name in CONFIG_NAMES:
        config_path = run_root / "configs" / name
        config = config_from_path(config_path)
        exp = config["experiment_id"]
        exp_result = {
            "gpu": config["gpu"],
            "config_sha256": sha256_file(config_path),
            "sides": {},
        }
        for kind in ("trigger", "action"):
            path = run_root / "smoke" / "manifests" / exp / f"train_{kind}.done.json"
            record = read_json(path)
            if record.get("status") != "passed" or record.get("global_steps") != 5:
                raise RuntimeError(f"invalid smoke record: {path}")
            smoke_rows = int(record["rows"]["rows"])
            full_rows = len(load_training_rows(config, data_root, run_root, kind))
            full_steps = math.ceil(full_rows / int(config["batch_size"])) * int(config["epochs"])
            seconds_per_step = float(record["training_seconds"]) / int(record["global_steps"])
            exp_result["sides"][kind] = {
                "smoke_manifest": str(path),
                "smoke_manifest_sha256": sha256_file(path),
                "smoke_steps": int(record["global_steps"]),
                "smoke_rows": smoke_rows,
                "smoke_seconds": float(record["training_seconds"]),
                "smoke_seconds_per_step": seconds_per_step,
                "finite_training_loss": float(record["training_loss"]),
                "reload_max_abs_diff": float(record["reload_max_abs_diff"]),
                "full_rows": full_rows,
                "planned_epochs": int(config["epochs"]),
                "planned_full_steps": full_steps,
                "coarse_linear_seconds": seconds_per_step * full_steps,
            }
        evaluation_done = run_root / "smoke" / "manifests" / exp / "evaluate.done.json"
        evaluation = read_json(evaluation_done)
        if evaluation.get("status") != "passed" or evaluation.get("split") != "dev":
            raise RuntimeError(f"invalid smoke evaluation: {evaluation_done}")
        exp_result["evaluation_done_sha256"] = sha256_file(evaluation_done)
        exp_result["evaluation_result_sha256"] = evaluation["result_sha256"]
        output["experiments"][exp] = exp_result
    destination = run_root / "manifests" / "BUDGET_BENCHMARK.json"
    write_json(destination, output)
    print(destination)


if __name__ == "__main__":
    main()
