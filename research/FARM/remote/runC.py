#!/usr/bin/env python3
"""RUN C - retrain full_finetune with MINED hard negatives.

The controlled comparison against Run B. Exactly one variable changes:

    Run B   (anchor, positive)            negatives = random, in-batch
    Run C   (anchor, positive, negative)  negatives = mined, hard

Everything else is identical - same deduped split, same hyperparameters
(epochs=3, batch=16, lr=2e-5, temperature=0.05), same eval, same seed.
MultipleNegativesRankingLoss accepts triplets natively and uses the explicit
negative IN ADDITION to the in-batch ones, so this is a strict superset of the
Run B signal.

The number to compare is LIFT OVER INDEPENDENCE:
    lift = joint_R@1 - (trigger_R@1 x action_R@1)
Run B full_finetune: +0.015. If mined negatives teach cross-schema compatibility,
lift should rise. If it does not, that is a legitimate negative result.

Same three redirections as runB.py so nothing published is overwritten:
  train data -> data/train_applets_dedup.json (via the triplets)
  eval data  -> data/test_dedup/gold.json
  checkpoints-> models/runC_ablation_*
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))

DEDUP_TEST = ROOT / "data" / "test_dedup" / "gold.json"


def load_triplets(side, val_split=0.1, seed=42):
    """Build a (anchor, positive, negative) Dataset, split the same way Run B splits."""
    import random
    from datasets import Dataset

    p = ROOT / "data" / f"triplets_{side}.json"
    if not p.exists():
        sys.exit(f"missing {p} - run the 'mine' job first")
    rows = json.load(open(p, encoding="utf-8"))
    if not rows:
        sys.exit(f"{p} is empty")

    random.seed(seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    cut = int(len(idx) * (1 - val_split))
    tr = [rows[i] for i in idx[:cut]]
    va = [rows[i] for i in idx[cut:]]

    def ds(rs):
        return Dataset.from_dict({
            "anchor": [r["anchor"] for r in rs],
            "positive": [r["positive"] for r in rs],
            "negative": [r["negative"] for r in rs],
        })
    print(f"  [{side}] triplets train={len(tr)} val={len(va)}", flush=True)
    return ds(tr), ds(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", default="results/RUNC_ablation.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for side in ("trigger", "action"):
        p = ROOT / "data" / f"triplets_{side}.json"
        if not p.exists():
            sys.exit(f"PREFLIGHT FAILED: missing {p} - run the 'mine' job first")
    if not DEDUP_TEST.exists():
        sys.exit(f"PREFLIGHT FAILED: missing {DEDUP_TEST} - run 'splitfix' first")
    print("preflight OK")

    import train.train_lora_ablation as tla

    # --- memory: triplets need ~1.5x Run B ---
    # MultipleNegativesRankingLoss runs one forward pass PER COLUMN and keeps them
    # all for backprop. Run B has 2 columns (anchor, positive) and peaks ~14.5 GB;
    # Run C has 3 and OOMs a 32 GB V100 outright.
    #
    # We must NOT shrink the batch to fix this: in-batch negatives ARE the batch, so
    # batch_size is part of the loss definition. Changing it would mean Run C differs
    # from Run B in two variables and the comparison would be meaningless.
    # Gradient checkpointing recomputes activations instead of storing them - same
    # math, same batch, ~30% slower.
    _OrigArgs = tla.SentenceTransformerTrainingArguments

    def _args_with_checkpointing(*a, **kw):
        kw.setdefault("gradient_checkpointing", True)
        kw.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
        return _OrigArgs(*a, **kw)

    tla.SentenceTransformerTrainingArguments = _args_with_checkpointing
    print("memory      -> gradient checkpointing ON (batch stays 16, as Run B)")

    # --- redirection 1: triplet datasets instead of (anchor, positive) pairs ---
    tla.build_trigger_pairs = lambda *a, **k: load_triplets("trigger")
    tla.build_action_pairs = lambda *a, **k: load_triplets("action")
    print("train data  -> data/triplets_{trigger,action}.json  (mined hard negatives)")

    # --- redirection 2: evaluate on the deduped split ---
    _orig_eval = tla.evaluate_retrieval_comprehensive

    # Must be a def with the EXACT parameter names: run_ablation calls this with
    # keyword arguments (trigger_model=, action_model=, triggers_path=, actions_path=).
    # A lambda with abbreviated names raises TypeError *after* training completes -
    # which is exactly what happened on the first Run C attempt, wasting the eval
    # (though not the 79 minutes of training).
    def eval_on_dedup(trigger_model, action_model, test_path, triggers_path,
                      actions_path, max_k=10):
        return _orig_eval(trigger_model, action_model, DEDUP_TEST,
                          triggers_path, actions_path, max_k)

    tla.evaluate_retrieval_comprehensive = eval_on_dedup
    print(f"eval data   -> {DEDUP_TEST}")

    # --- redirection 3: never touch published or Run B checkpoints ---
    _orig_train = tla.train_single_encoder

    def train_to_runC(model, train_dataset, val_dataset, output_dir, **kw):
        p = Path(output_dir)
        safe = p.parent / p.name.replace("ablation_", "runC_ablation_", 1)
        if safe == p:
            raise RuntimeError(f"refusing to write to {p} - redirection failed")
        print(f"  checkpoint -> {safe}")
        return _orig_train(model=model, train_dataset=train_dataset,
                           val_dataset=val_dataset, output_dir=safe, **kw)

    tla.train_single_encoder = train_to_runC
    print("checkpoints -> models/runC_ablation_*  (Run B and published untouched)")

    if args.dry_run:
        print("\n--dry-run: redirections verified, exiting without training.")
        return

    print(f"\nhyperparameters identical to Run B (epochs={args.epochs}, batch={args.batch_size}) "
          "- mined negatives are the ONLY changed variable.\n")

    tla.run_full_ablation_study(
        lora_ranks=[],                 # full_finetune only - one controlled comparison
        freeze_layers=12,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=ROOT / args.output,
    )
    print(f"\nRUN C complete -> {args.output}")


if __name__ == "__main__":
    main()
