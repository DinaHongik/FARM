#!/usr/bin/env python3
"""Evaluate the Run C checkpoints that already exist - no retraining.

Run C trained both encoders successfully (41.3 + 37.8 min) and then crashed in the
evaluation call because the monkeypatched evaluate_retrieval_comprehensive was a
lambda whose parameter names did not match the keyword arguments run_ablation uses.
The weights are intact, so this scores them directly and writes a file in the same
shape as results/RUNB_ablation.json so jobs/lift.py can compare them.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sentence_transformers import SentenceTransformer  # noqa: E402
from train.train_lora_ablation import evaluate_retrieval_comprehensive  # noqa: E402

TRIG = ROOT / "models" / "runC_ablation_full_finetune_trigger" / "final"
ACT = ROOT / "models" / "runC_ablation_full_finetune_action" / "final"
TEST = ROOT / "data" / "test_dedup" / "gold.json"
OUT = ROOT / "results" / "RUNC_ablation.json"


def main():
    for p in (TRIG, ACT, TEST):
        if not p.exists():
            sys.exit(f"missing: {p}")

    print(f"trigger  {TRIG}")
    print(f"action   {ACT}")
    print(f"test     {TEST}   (deduped, leakage-free)\n", flush=True)

    tm = SentenceTransformer(str(TRIG), device="cuda")
    am = SentenceTransformer(str(ACT), device="cuda")

    t0 = time.time()
    tmet, amet, jmet = evaluate_retrieval_comprehensive(
        tm, am, TEST, ROOT / "data" / "triggers_rag.json",
        ROOT / "data" / "actions_rag.json")
    print(f"evaluated in {time.time() - t0:.0f}s\n")

    payload = {
        "study_info": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Run C - full_finetune trained on MINED hard negatives (triplets). "
                    "Evaluated from existing checkpoints after the first attempt "
                    "crashed in the eval call. Training itself was unaffected.",
            "train_data": "data/triplets_{trigger,action}.json",
            "eval_data": "data/test_dedup/gold.json",
            "epochs": 3, "batch_size": 16,
        },
        "results": [{
            "method": "full_finetune",
            "config": {"negatives": "mined_hard", "epochs": 3, "batch_size": 16},
            "trigger_metrics": tmet.to_dict(),
            "action_metrics": amet.to_dict(),
            "joint_metrics": jmet.to_dict(),
        }],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    t = tmet.to_dict()["service_level"]
    a = amet.to_dict()["service_level"]
    j = jmet.to_dict()["service_level"]
    print(f"{'':<8}{'trigger':>9}{'action':>9}{'product':>9}{'joint':>8}{'lift':>9}")
    for k in ("R@1", "R@5"):
        print(f"{k:<8}{t[k]:>9.3f}{a[k]:>9.3f}{t[k]*a[k]:>9.3f}{j[k]:>8.3f}{j[k]-t[k]*a[k]:>+9.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
