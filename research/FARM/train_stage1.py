"""Train one stage-1 bi-encoder on the rebuilt pairs.

Two deliberate changes from the previous recipe, everything else held fixed so the
numbers stay comparable:
  1. positives come from data/pairs/, so they are byte-identical to indexed documents
  2. BatchSamplers.NO_DUPLICATES, so a row's own gold is never an in-batch negative
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["trigger", "action"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sampler", default="NO_DUPLICATES", choices=["NO_DUPLICATES", "BATCH_SAMPLER"])
ap.add_argument("--negatives", type=int, default=0,
                help="explicit mined hard negatives per row; 0 keeps the (anchor, positive) pair format")
ap.add_argument("--hardness", default="none",
                choices=["none", "in_batch_negatives", "hard_negatives", "all_negatives"])
ap.add_argument("--hardness-strength", type=float, default=5.0,
                help="alpha. The EmbeddingGemma paper uses 5 for hard_negatives, Lan et al. 9 for in_batch")
ap.add_argument("--cached", action="store_true",
                help="CachedMultipleNegativesRankingLoss (GradCache): decouples contrastive batch from memory")
ap.add_argument("--bs", type=int, default=0, help="override config batch size")
ap.add_argument("--mini-bs", type=int, default=16, help="GradCache mini batch")
ap.add_argument("--scale", type=float, default=0.0,
                help="MNRL scale = 1/temperature. Config default 20 (temp 0.05). "
                     "Docs: 'values between 10 and 100 are common'; higher scale puts "
                     "more emphasis on the positive. Never swept before.")
ap.add_argument("--directions", default="",
                help="comma-separated: query_to_doc,doc_to_query,query_to_query,doc_to_doc. "
                     "Default is query_to_doc only. doc_to_doc contrasts documents against "
                     "each other excluding same-query ones, which is exactly the "
                     "sibling-channel problem ('New episode' across 91 podcast channels).")
ap.add_argument("--gist", default="", help="guide model for (Cached)GISTEmbedLoss")
ap.add_argument("--augmented", action="store_true",
                help="use the paraphrase-augmented pairs (+76.8% anchors)")
ap.add_argument("--freeze", type=int, default=0,
                help="freeze the bottom N transformer layers (0 = train everything)")
ap.add_argument("--lr", type=float, default=0.0, help="override config learning rate")
ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
ap.add_argument("--out", default="runs/rebuilt")
A = ap.parse_args()

cfg = json.loads(Path("train_config.json").read_text())
torch.manual_seed(A.seed)

if A.negatives:
    # n-tuple format: (anchor, positive, negative_1..negative_n). Rows with fewer
    # mined negatives are padded by resampling their own; rows with none are dropped,
    # because a row of all-padding teaches nothing and silently reweights the head.
    src = json.loads(Path(f"data/pairs/{A.kind}_mined{'_aug' if A.augmented else ''}.json").read_text())
    rows, dropped = [], 0
    for r in src:
        negs = r.get("negatives") or []
        if not negs:
            dropped += 1
            continue
        padded = [negs[i % len(negs)] for i in range(A.negatives)]
        rows.append({**r, "neg": padded})
    cols = {"anchor": [r["anchor"] for r in rows], "positive": [r["positive"] for r in rows]}
    for i in range(A.negatives):
        cols[f"negative_{i+1}"] = [r["neg"][i] for r in rows]
    ds = Dataset.from_dict(cols)
    print(f"n-tuple data: {len(rows)} rows x {A.negatives} negatives, {dropped} rows dropped "
          f"for having none mined", flush=True)
else:
    rows = json.loads(Path(f"data/pairs/{A.kind}_stage1_train.json").read_text())
    ds = Dataset.from_dict({"anchor": [r["anchor"] for r in rows],
                            "positive": [r["positive"] for r in rows]})
tag = f"{A.kind}-s{A.seed}" + ("" if A.sampler == "NO_DUPLICATES" else "-dupsampler")
if A.negatives: tag += f"-n{A.negatives}"
if A.hardness != "none": tag += f"-{A.hardness}{A.hardness_strength:g}"
if A.cached: tag += f"-cached{A.bs or cfg['batch_size']}"
if A.augmented: tag += "-aug"
if A.scale: tag += f"-sc{A.scale:g}"
if A.directions: tag += "-dir" + "".join(w[0]+w.split("_")[-1][0] for w in A.directions.split(","))
if A.gist: tag += "-gist"
if A.freeze: tag += f"-frz{A.freeze}"
if A.lr: tag += f"-lr{A.lr:g}"
if A.epochs: tag += f"-ep{A.epochs}"
out = Path(A.out) / tag

print(f"kind={A.kind} seed={A.seed} sampler={A.sampler} rows={len(rows)} "
      f"labels={len({r['label_url'] for r in rows})} base={cfg['base_model']} "
      f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

model = SentenceTransformer(cfg["base_model"], device="cuda:0",
                            model_kwargs={"torch_dtype": torch.float32})
model.max_seq_length = cfg["max_seq_length"]
if A.freeze:
    # Freeze embeddings + the bottom N encoder layers. The repo's earlier
    # layer-freezing claim was withdrawn (n=100, one seed, 94% contaminated eval),
    # so this is a fresh test on the clean pipeline, not a re-run of that result.
    auto = model._first_module().auto_model
    frozen = 0
    for name, param in auto.named_parameters():
        keep = False
        if "embed_tokens" in name or "embeddings" in name:
            keep = True
        else:
            import re as _re
            m = _re.search(r"layers?\.(\d+)\.", name)
            if m and int(m.group(1)) < A.freeze:
                keep = True
        if keep:
            param.requires_grad = False
            frozen += param.numel()
    total = sum(p.numel() for p in auto.parameters())
    print(f"froze bottom {A.freeze} layers + embeddings: "
          f"{frozen/1e6:.1f}M of {total/1e6:.1f}M params ({frozen/total:.1%}) not trained", flush=True)
kw = dict(scale=A.scale or cfg["scale"])
if A.hardness != "none":
    kw.update(hardness_mode=A.hardness, hardness_strength=A.hardness_strength)
if A.directions:
    kw["directions"] = tuple(d.strip() for d in A.directions.split(",") if d.strip())
if A.gist:
    guide = SentenceTransformer(A.gist, device="cuda:0",
                                model_kwargs={"torch_dtype": torch.float32})
    kw.pop("hardness_mode", None); kw.pop("hardness_strength", None)
    kw.pop("directions", None)
    # GIST parameterises the softmax by temperature; MNRL by scale = 1/temperature.
    # Convert rather than take GIST's own default of 0.01, so this arm differs from the
    # scale-20 baseline only in the loss and not also in the temperature.
    kw["temperature"] = 1.0 / kw.pop("scale")
    loss = (losses.CachedGISTEmbedLoss(model, guide, mini_batch_size=A.mini_bs, **kw)
            if A.cached else losses.GISTEmbedLoss(model, guide, **kw))
elif A.cached:
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=A.mini_bs, **kw)
else:
    loss = losses.MultipleNegativesRankingLoss(model, **kw)
print(f"loss={type(loss).__name__} {kw}", flush=True)

args = SentenceTransformerTrainingArguments(
    output_dir=str(out),
    num_train_epochs=A.epochs or cfg["epochs"],
    per_device_train_batch_size=A.bs or cfg["batch_size"],
    learning_rate=A.lr or cfg["learning_rate"],
    warmup_ratio=cfg["warmup_ratio"],
    weight_decay=cfg["weight_decay"],
    seed=A.seed,
    fp16=False, bf16=False,          # EmbeddingGemma has no fp16 path; V100 emulates bf16
    batch_sampler=getattr(BatchSamplers, A.sampler),
    logging_steps=100, save_strategy="no", report_to=[],
)
t0 = time.time()
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
model.save(str(out / "final"))
print(f"DONE {tag} in {time.time()-t0:.0f}s -> {out}/final", flush=True)
