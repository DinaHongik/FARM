#!/usr/bin/env python3
"""Atomically preflight that none of the four full pipelines has prior output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import CONFIG_NAMES, assert_fresh_pipeline_output, config_from_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    checked = []
    for name in CONFIG_NAMES:
        config = config_from_path(run_root / "configs" / name)
        assert_fresh_pipeline_output(run_root, config["experiment_id"], "full")
        checked.append(config["experiment_id"])
    print(json.dumps({"status": "fresh", "experiments": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
