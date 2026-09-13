#!/usr/bin/env python3
"""RUN B - retrain the Stage 1 ablation on the LEAKAGE-FREE split.

WHY A DRIVER INSTEAD OF CALLING train_lora_ablation.py DIRECTLY
    That script hardcodes three things that would each silently invalidate or
    destroy Run B. This wrapper redirects all three at import time, so the
    original source is left untouched:

    1. train data   train_lora_ablation.py:618,646 -> config.applets_path
                    = data/train_applets.json  (37% overlap with the test splits)
                    REDIRECTED -> data/train_applets_dedup.json

    2. eval data    train_lora_ablation.py:674 hardcodes
                    data/test/gold.json  (the contaminated split)
                    REDIRECTED -> data/test_dedup/gold.json

    3. output dirs  train_lora_ablation.py:624,652 write to
                    models/ablation_<method>_{trigger,action}
                    which are the EXISTING published checkpoints. Overwriting
                    them destroys the only evidence of the current paper state
                    (and they are already known to disagree with the published
                    table by up to -0.17).
                    REDIRECTED -> models/runB_ablation_<method>_{trigger,action}

    Hyperparameters are deliberately NOT changed. Run B must differ from the
    published run in exactly one variable - the leakage - or the delta cannot be
    attributed. Tuning belongs in a later run, against honest baselines.

Usage:
    python runB.py                       # full_finetune + lora 8/16/32 + layer_freeze
    python runB.py --ranks 8 16          # subset
    python runB.py --dry-run             # show the redirections and exit
"""
import argparse
import sys
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))

DEDUP_TRAIN = "./data/train_applets_dedup.json"
DEDUP_TEST = ROOT / "data" / "test_dedup" / "gold.json"


def preflight():
    """Refuse to start if the deduped data is missing or still leaks."""
    train_path = ROOT / DEDUP_TRAIN.lstrip("./")
    problems = []
    if not train_path.exists():
        problems.append(f"missing {train_path}  (run the 'splitfix' job first)")
    if not DEDUP_TEST.exists():
        problems.append(f"missing {DEDUP_TEST}  (run the 'splitfix' job first)")
    if problems:
        for p in problems:
            print(f"PREFLIGHT FAILED: {p}")
        sys.exit(1)

    import json

    def key(r):
        return (r.get("query", "").strip(),
                r.get("trigger", {}).get("service_name", ""),
                r.get("action", {}).get("service_name", ""))

    train_keys = {key(r) for r in json.load(open(train_path, encoding="utf-8"))}
    leaked = 0
    for split in ("gold", "noisy", "oneshot"):
        p = DEDUP_TEST.parent / f"{split}.json"
        if p.exists():
            leaked += sum(1 for r in json.load(open(p, encoding="utf-8")) if key(r) in train_keys)
    if leaked:
        print(f"PREFLIGHT FAILED: {leaked} deduped test records still occur in the deduped "
              f"train set. Do not train on this.")
        sys.exit(1)
    print(f"preflight OK - 0 test records occur in {DEDUP_TRAIN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--freeze-layers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", default="results/RUNB_ablation.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    preflight()

    # ---- redirection 1: training data ----
    from train.config import config
    config.applets_data = DEDUP_TRAIN
    print(f"train data  -> {config.applets_path}")

    import train.train_lora_ablation as tla

    # ---- redirection 2: evaluation split ----
    _orig_eval = tla.evaluate_retrieval_comprehensive

    def eval_on_dedup(trigger_model, action_model, test_path, triggers_path, actions_path, max_k=10):
        return _orig_eval(trigger_model, action_model, DEDUP_TEST, triggers_path, actions_path, max_k)

    tla.evaluate_retrieval_comprehensive = eval_on_dedup
    print(f"eval data   -> {DEDUP_TEST}")

    # ---- redirection 3: checkpoint output dirs (protects the published weights) ----
    _orig_train = tla.train_single_encoder

    def train_to_runB(model, train_dataset, val_dataset, output_dir, **kw):
        p = Path(output_dir)
        safe = p.parent / p.name.replace("ablation_", "runB_ablation_", 1)
        if safe == p:
            raise RuntimeError(f"refusing to write to {p} - redirection failed")
        print(f"  checkpoint -> {safe}")
        return _orig_train(model=model, train_dataset=train_dataset,
                           val_dataset=val_dataset, output_dir=safe, **kw)

    tla.train_single_encoder = train_to_runB
    print("checkpoints -> models/runB_ablation_*  (existing weights untouched)")

    if args.dry_run:
        print("\n--dry-run: redirections verified, exiting without training.")
        return

    print("\nhyperparameters unchanged from the published run "
          f"(epochs={args.epochs}, batch={args.batch_size}, freeze={args.freeze_layers}, "
          f"ranks={args.ranks}) - leakage is the only changed variable.\n")

    tla.run_full_ablation_study(
        lora_ranks=args.ranks,
        freeze_layers=args.freeze_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=ROOT / args.output,
    )
    print(f"\nRUN B complete -> {args.output}")


if __name__ == "__main__":
    main()
