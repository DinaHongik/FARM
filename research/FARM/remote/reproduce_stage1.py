#!/usr/bin/env python3
"""Stage 1 reproduction: load EXISTING checkpoints, rerun the authentic protocol.

No training happens here. This answers one question: do the checkpoints on disk
still produce the numbers the paper reports?

It calls train_lora_ablation.evaluate_retrieval_comprehensive - the same function
that produced the published table - rather than rag/evaluate_rag.py, which has a
self-retrieval flaw and produced none of the reported numbers.

Usage:
    python reproduce_stage1.py                    # gold only
    python reproduce_stage1.py gold noisy oneshot
    python reproduce_stage1.py --testdir data/test_dedup gold noisy oneshot
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sentence_transformers import SentenceTransformer  # noqa: E402
from train.train_lora_ablation import evaluate_retrieval_comprehensive  # noqa: E402

MODELS = ROOT / "models"
DATA = ROOT / "data"

METHODS = [
    ("production",    "trigger-encoder",                 "action-encoder"),
    ("full_finetune", "ablation_full_finetune_trigger",  "ablation_full_finetune_action"),
    ("layer_freeze",  "ablation_layer_freeze_trigger",   "ablation_layer_freeze_action"),
    ("lora_r8",       "ablation_lora_r8_trigger",        "ablation_lora_r8_action"),
    ("lora_r16",      "ablation_lora_r16_trigger",       "ablation_lora_r16_action"),
    ("lora_r32",      "ablation_lora_r32_trigger",       "ablation_lora_r32_action"),
]

_TMPDIRS = []


def resolve(name):
    for cand in (MODELS / name / "final", MODELS / name):
        if (cand / "config.json").exists() or (cand / "modules.json").exists():
            return cand
    return None


def loadable(path):
    """Return a directory SentenceTransformer can load without network access.

    The LoRA checkpoints ship BOTH merged full weights and the adapter files.
    PEFT sees adapter_config.json, reads base_model_name_or_path, and tries to
    fetch google/embeddinggemma-300m - which is gated, so it 401s. The merged
    weights are self-sufficient, so hand it a view of the directory with the
    adapter files omitted.
    """
    if not (path / "adapter_config.json").exists():
        return path
    if not (path / "model.safetensors").exists():
        return path  # adapter-only: genuinely needs the base model
    tmp = Path(tempfile.mkdtemp(prefix="farm_ckpt_"))
    _TMPDIRS.append(tmp)
    for f in path.iterdir():
        if f.name.startswith("adapter"):
            continue
        (tmp / f.name).symlink_to(f)
    return tmp


def fmt(v):
    return "  -  " if v is None else f"{v:.4f}"


def compare(reproduced, split="gold"):
    """Print reproduced vs the published ablation table."""
    orig_path = ROOT / "results" / "ablation_comprehensive.json"
    if not orig_path.exists():
        print("\n(no results/ablation_comprehensive.json - skipping comparison)")
        return
    orig = {r["method"]: r for r in json.load(open(orig_path, encoding="utf-8"))["results"]}

    print("\n" + "=" * 78)
    print(f"REPRODUCED vs PUBLISHED   (split={split}, n=100)")
    print("=" * 78)
    print(f"{'config':<15}{'metric':<20}{'published':>11}{'reproduced':>12}{'delta':>9}")
    print("-" * 78)

    for method, _, _ in METHODS:
        if method not in orig or method not in reproduced:
            continue
        if split not in reproduced[method]:
            continue
        o, r = orig[method], reproduced[method][split]
        pairs = [
            ("trigger R@1",  o["trigger_metrics"]["service_level"]["R@1"], r["trigger"]["service_level"]["R@1"]),
            ("trigger R@5",  o["trigger_metrics"]["service_level"]["R@5"], r["trigger"]["service_level"]["R@5"]),
            ("action  R@1",  o["action_metrics"]["service_level"]["R@1"],  r["action"]["service_level"]["R@1"]),
            ("action  R@5",  o["action_metrics"]["service_level"]["R@5"],  r["action"]["service_level"]["R@5"]),
            ("joint   R@1",  o["joint_metrics"]["service_level"]["R@1"],   r["joint"]["service_level"]["R@1"]),
            ("joint   R@5",  o["joint_metrics"]["service_level"]["R@5"],   r["joint"]["service_level"]["R@5"]),
        ]
        first = True
        for label, ov, rv in pairs:
            d = rv - ov
            flag = ""
            if abs(d) >= 0.05:
                flag = "  <-- BIG"
            elif abs(d) >= 0.02:
                flag = "  <--"
            print(f"{method if first else '':<15}{label:<20}{ov:>11.4f}{rv:>12.4f}{d:>+9.4f}{flag}")
            first = False
        print("-" * 78)
    print("Deltas under 0.02 are ordinary library-version drift. Anything larger means")
    print("the checkpoint on disk is not the one that produced the published number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["gold"])
    ap.add_argument("--testdir", default="data/test",
                    help="data/test (original) or data/test_dedup (leakage-free)")
    ap.add_argument("--out", default="results/REPRODUCED_stage1.json")
    args = ap.parse_args()

    splits = args.splits or ["gold"]
    test_dir = ROOT / args.testdir
    out_path = ROOT / args.out

    out = {"_meta": {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "testdir": args.testdir,
        "note": "eval-only from existing checkpoints; no training",
    }}

    for method, tname, aname in METHODS:
        tp, ap_ = resolve(tname), resolve(aname)
        if not tp or not ap_:
            print(f"SKIP {method}: checkpoint missing", flush=True)
            continue
        print(f"\n=== {method} ===", flush=True)
        t0 = time.time()
        try:
            tm = SentenceTransformer(str(loadable(tp)), device="cuda")
            am = SentenceTransformer(str(loadable(ap_)), device="cuda")
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue

        out[method] = {}
        for split in splits:
            tf = test_dir / f"{split}.json"
            if not tf.exists():
                print(f"  [{split}] missing {tf}", flush=True)
                continue
            tmet, amet, jmet = evaluate_retrieval_comprehensive(
                tm, am, tf, DATA / "triggers_rag.json", DATA / "actions_rag.json")
            out[method][split] = {
                "trigger": tmet.to_dict(), "action": amet.to_dict(), "joint": jmet.to_dict()}
            sl = jmet.to_dict()["service_level"]
            print(f"  [{split}] joint R@1={sl['R@1']:.2f} R@5={sl['R@5']:.2f}", flush=True)

        del tm, am
        import torch
        torch.cuda.empty_cache()
        print(f"  ({time.time() - t0:.0f}s)", flush=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)

    print(f"\nwrote {out_path.relative_to(ROOT)}")
    if "gold" in splits and args.testdir == "data/test":
        compare(out, "gold")


if __name__ == "__main__":
    main()
