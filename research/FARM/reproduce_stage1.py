#!/usr/bin/env python3
"""Eval-only Stage 1 reproduction: load EXISTING checkpoints, rerun the authentic
protocol from train_lora_ablation.evaluate_retrieval_comprehensive. No training."""
import sys, json, os, time
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sentence_transformers import SentenceTransformer
from train.train_lora_ablation import evaluate_retrieval_comprehensive

MODELS = ROOT / "models"
DATA = ROOT / "data"
TRIG_CORPUS, ACT_CORPUS = DATA / "triggers_rag.json", DATA / "actions_rag.json"

METHODS = {
    "production":     ("trigger-encoder", "action-encoder"),
    "full_finetune":  ("ablation_full_finetune_trigger", "ablation_full_finetune_action"),
    "layer_freeze":   ("ablation_layer_freeze_trigger", "ablation_layer_freeze_action"),
    "lora_r8":        ("ablation_lora_r8_trigger", "ablation_lora_r8_action"),
    "lora_r16":       ("ablation_lora_r16_trigger", "ablation_lora_r16_action"),
    "lora_r32":       ("ablation_lora_r32_trigger", "ablation_lora_r32_action"),
}

def resolve(name):
    for cand in (MODELS / name / "final", MODELS / name):
        if (cand / "config.json").exists() or (cand / "modules.json").exists():
            return cand
    return None

splits = sys.argv[1:] or ["gold"]
out = {"_meta": {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "eval-only from existing checkpoints"}}

for method, (tname, aname) in METHODS.items():
    tp, ap = resolve(tname), resolve(aname)
    if not tp or not ap:
        print(f"SKIP {method}: missing ({tname}={tp}, {aname}={ap})", flush=True)
        continue
    print(f"\n=== {method} ===\n  trigger: {tp}\n  action:  {ap}", flush=True)
    t0 = time.time()
    tm = SentenceTransformer(str(tp), device="cuda")
    am = SentenceTransformer(str(ap), device="cuda")
    out[method] = {}
    for split in splits:
        tmet, amet, jmet = evaluate_retrieval_comprehensive(
            tm, am, DATA / "test" / f"{split}.json", TRIG_CORPUS, ACT_CORPUS)
        out[method][split] = {"trigger": tmet.to_dict(), "action": amet.to_dict(), "joint": jmet.to_dict()}
        print(f"  [{split}] done", flush=True)
    del tm, am
    import torch; torch.cuda.empty_cache()
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    with open(ROOT / "results" / "REPRODUCED_stage1.json", "w") as f:
        json.dump(out, f, indent=2)

print("\nWROTE results/REPRODUCED_stage1.json")
