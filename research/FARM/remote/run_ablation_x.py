#!/usr/bin/env python3
"""Generalised Stage-1 ablation driver.

runB.py and runC.py each hardcode one experiment. This driver takes the same
three safety redirections and exposes the variables we actually want to sweep,
so a new experiment is a new command line rather than a new file.

THE THREE REDIRECTIONS (identical to runB.py - do not remove)
    1. train data   -> data/train_applets_dedup.json   (the leaky file is 37% test)
    2. eval data    -> data/test_dedup/gold.json       (the published split is contaminated)
    3. checkpoints  -> models/<tag>_ablation_*         (published weights are the only
                                                        evidence of the current paper state)

WHAT THIS ADDS OVER runB.py
    --seed N            Overrides BOTH seeds that train_lora_ablation.py hardcodes to 42:
                        the TrainingArguments seed (line 547: init + batch order) and the
                        train/val split seed (lines 620, 648). The TEST split is a fixed
                        file and is unaffected, so runs at different seeds stay comparable.

    --freeze-layers N   Already existed in runB.py but was never swept. This is the
                        interesting axis: at N=12 only 18% of parameters train, and 80%
                        of what is frozen is embed_tokens (201M of 252M), NOT layers.
                        N=0 therefore means "freeze the embedding table only" and still
                        leaves 65% of the model frozen while roughly doubling trainable
                        capacity vs N=12.

WHY THAT MATTERS
    On the leaky split layer_freeze beat full_finetune (0.580 vs 0.560). On the clean
    split it lost (0.440 vs 0.570). Two explanations are confounded in the current
    config - regularisation vs capacity - and one seed cannot separate them from noise.
    Sweeping seed answers "is the inversion real"; sweeping freeze_layers answers
    "can freezing be made to win". Reviewer 1(iv) asks exactly this.

Usage
    python run_ablation_x.py --tag runB2  --seed 1337 --ranks 8 16 32
    python run_ablation_x.py --tag freeze0 --freeze-layers 0 --ranks
"""
import argparse
import json
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
    ap.add_argument("--tag", required=True,
                    help="checkpoint/result namespace, e.g. runB2. Must not be 'ablation' "
                         "(that is the published namespace).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 32],
                    help="LoRA ranks; pass --ranks with no values to skip LoRA entirely")
    ap.add_argument("--freeze-layers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", default=None,
                    help="default: results/<TAG>_ablation.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.tag.strip().lower() in ("", "ablation"):
        sys.exit("--tag 'ablation' would overwrite the published checkpoints. Refusing.")
    out_rel = args.output or f"results/{args.tag.upper()}_ablation.json"

    preflight()

    # ---- redirection 1: training data ----
    from train.config import config
    config.applets_data = DEDUP_TRAIN
    print(f"train data  -> {config.applets_path}")

    import train.train_lora_ablation as tla

    # ---- redirection 2: evaluation split ----
    # Must be a def with the EXACT parameter names run_ablation uses as keywords
    # (trigger_model=, action_model=, triggers_path=, actions_path=). A lambda with
    # abbreviated names raises TypeError *after* training finishes - that cost Run C
    # its first eval.
    _orig_eval = tla.evaluate_retrieval_comprehensive

    def eval_on_dedup(trigger_model, action_model, test_path, triggers_path,
                      actions_path, max_k=10):
        return _orig_eval(trigger_model, action_model, DEDUP_TEST,
                          triggers_path, actions_path, max_k)

    tla.evaluate_retrieval_comprehensive = eval_on_dedup
    print(f"eval data   -> {DEDUP_TEST}")

    # ---- redirection 3: checkpoint output dirs ----
    _orig_train = tla.train_single_encoder
    prefix = f"{args.tag}_ablation_"

    def train_to_tag(model, train_dataset, val_dataset, output_dir, **kw):
        p = Path(output_dir)
        safe = p.parent / p.name.replace("ablation_", prefix, 1)
        if safe == p:
            raise RuntimeError(f"refusing to write to {p} - redirection failed")
        print(f"  checkpoint -> {safe}")
        return _orig_train(model=model, train_dataset=train_dataset,
                           val_dataset=val_dataset, output_dir=safe, **kw)

    tla.train_single_encoder = train_to_tag
    print(f"checkpoints -> models/{prefix}*  (published + Run B untouched)")

    # ---- seed override ----
    # Two independent seeds are hardcoded to 42 upstream and both must move together,
    # otherwise "a different seed" only changes half the run.
    _OrigArgs = tla.SentenceTransformerTrainingArguments

    def _args_with_seed(*a, **kw):
        kw["seed"] = args.seed
        return _OrigArgs(*a, **kw)

    tla.SentenceTransformerTrainingArguments = _args_with_seed

    _orig_tp, _orig_ap = tla.build_trigger_pairs, tla.build_action_pairs
    tla.build_trigger_pairs = lambda p, val_split=0.1, seed=42: _orig_tp(
        p, val_split=val_split, seed=args.seed)
    tla.build_action_pairs = lambda p, val_split=0.1, seed=42: _orig_ap(
        p, val_split=val_split, seed=args.seed)
    print(f"seed        -> {args.seed}  (TrainingArguments + train/val split; "
          f"test split is a fixed file)")

    methods = ["full_finetune"] + [f"lora_r{r}" for r in args.ranks] + \
              [f"layer_freeze(freeze={args.freeze_layers})"]
    print(f"methods     -> {', '.join(methods)}")
    print(f"output      -> {out_rel}")

    if args.dry_run:
        print("\n--dry-run: redirections verified, exiting without training.")
        return

    print()
    tla.run_full_ablation_study(
        lora_ranks=list(args.ranks),
        freeze_layers=args.freeze_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=ROOT / out_rel,
    )
    print(f"\n{args.tag} complete -> {out_rel}")


if __name__ == "__main__":
    main()
