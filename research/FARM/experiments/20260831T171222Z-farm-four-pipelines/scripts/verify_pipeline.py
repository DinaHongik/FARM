#!/usr/bin/env python3
"""Hash-verify a completed full pipeline without allocating a GPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    config_from_path,
    sha256_file,
    validate_dataset,
    validate_hardneg_derived,
    validate_service_derived,
    verify_evaluation_done,
    verify_training_done,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    config = config_from_path(config_path)
    dataset = validate_dataset(data_root)
    if config["experiment_id"] == "e0":
        validate_service_derived(run_root, data_root)
    if config["experiment_id"] == "e3":
        validate_hardneg_derived(run_root, data_root)
    config_hash = sha256_file(config_path)
    verified = {}
    for kind in ("trigger", "action"):
        verified[kind] = verify_training_done(
            run_root / "manifests" / config["experiment_id"] / f"train_{kind}.done.json",
            run_root / "checkpoints" / config["experiment_id"] / kind / "final",
            config,
            kind,
            config_hash,
        )["checkpoint_file_hashes"]
    evaluation = verify_evaluation_done(
        run_root / "manifests" / config["experiment_id"] / "evaluate.done.json",
        config,
    )
    print(json.dumps({
        "status": "verified",
        "experiment_id": config["experiment_id"],
        "dataset_id": dataset["dataset_id"],
        "checkpoint_file_counts": {key: len(value) for key, value in verified.items()},
        "evaluation_result_sha256": evaluation["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
