#!/usr/bin/env python3
"""Fail closed unless both bounded Ollama smoke artifacts prove real calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import DATASET_ID, read_json, sha256_file, utc_now, write_json  # noqa: E402

LLM_ARMS = ("one_shot_plain", "one_shot_schema", "reflect_agent", "role_agent", "tool_agent")
EXPECTED = (
    ("gpt_oss_120b", "gpt-oss:120b", "d98fe6ba01e6"),
    ("gemma4_31b", "gemma4:31b", "221b330d11a8"),
)


def selected_ids_sha(candidates: list[dict], count: int) -> str:
    payload = json.dumps(
        [row["group_id"] for row in candidates[:count]], separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_one(path: Path, model: str, digest: str, candidate_sha: str, ids_sha: str) -> dict:
    value = read_json(path)
    checks = {
        "status": value.get("status") == "completed",
        "dataset": value.get("dataset_id") == DATASET_ID,
        "rows": value.get("rows") == 2 and value.get("max_rows") == 2,
        "model": value.get("model") == model,
        "digest": value.get("model_digest") == digest
        and value.get("model_record", {}).get("digest") == digest,
        "candidate_sha": value.get("candidate_sha256") == candidate_sha,
        "selected_ids_sha": value.get("selected_group_ids_sha256") == ids_sha,
        "generation_cap": value.get("max_generation_tokens_per_call") == 512,
    }
    rows = value.get("rows_detail") or []
    for arm in LLM_ARMS:
        metric = value.get("metrics", {}).get(arm, {})
        outcomes = [row.get("arms", {}).get(arm, {}) for row in rows]
        checks[f"{arm}_two_rows"] = len(outcomes) == 2
        checks[f"{arm}_no_failures"] = (
            metric.get("failures") == 0
            and metric.get("unknown_failed_call_cost_rows") == 0
            and all("error_type" not in outcome for outcome in outcomes)
        )
        checks[f"{arm}_real_calls"] = (
            int(metric.get("total_calls", 0)) > 0
            and all(int(outcome.get("calls", 0)) > 0 for outcome in outcomes)
            and all(len(outcome.get("usage") or []) > 0 for outcome in outcomes)
        )
        # Invalid selection is a measurable model outcome; however, at least
        # one of two rows must demonstrate that each protocol actually works.
        checks[f"{arm}_protocol_exercised"] = any(
            bool(outcome.get("protocol_valid")) for outcome in outcomes
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"agent smoke validation failed for {model}: {failed}")
    return {
        "path": str(path), "sha256": sha256_file(path), "model": model,
        "digest": digest, "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--write-validation", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    candidate_path = root / "derived_data/agent/dev300_candidates.json"
    manifest = read_json(root / "derived_data/agent/dev300_candidates.manifest.json")
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != "dev":
        raise RuntimeError("agent candidate manifest is not Dataset v2 dev")
    candidate_sha = sha256_file(candidate_path)
    if manifest.get("output_sha256") != candidate_sha:
        raise RuntimeError("agent candidate bytes changed")
    candidates = read_json(candidate_path)
    ids_sha = selected_ids_sha(candidates, 2)
    artifacts = []
    for slug, model, digest in EXPECTED:
        path = root / "smoke/results" / f"agent_{slug}_dev2.json"
        if not path.is_file():
            raise RuntimeError(f"missing agent smoke output: {path}")
        artifacts.append(validate_one(path, model, digest, candidate_sha, ids_sha))
    result = {
        "status": "passed", "validated_at": utc_now(), "dataset_id": DATASET_ID,
        "candidate_sha256": candidate_sha, "selected_group_ids_sha256": ids_sha,
        "artifacts": artifacts,
    }
    if args.write_validation:
        write_json(root / "smoke/manifests/agent_smoke_validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
