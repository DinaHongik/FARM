"""Reproduce the F2 memory/speed claim: MNRL bs=16 vs CachedMNRL bs>=128 on a V100."""
import json, sys, time
from pathlib import Path
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

ROOT = Path("/raid/session/aicontents/farm")
cfg = json.loads((ROOT / "train_config.json").read_text())
rows = json.loads((ROOT / "data/pairs/trigger_stage1_train.json").read_text())
ds = Dataset.from_dict({"anchor": [r["anchor"] for r in rows],
                        "positive": [r["positive"] for r in rows]})

mode, bs, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mini = int(sys.argv[4]) if len(sys.argv) > 4 else 16

torch.cuda.reset_peak_memory_stats()
model = SentenceTransformer(cfg["base_model"], device="cuda:0",
                            model_kwargs={"torch_dtype": torch.float32})
model.max_seq_length = cfg["max_seq_length"]
if mode == "mnrl":
    loss = losses.MultipleNegativesRankingLoss(model, scale=cfg["scale"])
else:
    loss = losses.CachedMultipleNegativesRankingLoss(model, scale=cfg["scale"], mini_batch_size=mini)

args = SentenceTransformerTrainingArguments(
    output_dir="/tmp/benchf2", max_steps=steps,
    per_device_train_batch_size=bs, learning_rate=cfg["learning_rate"],
    warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"], seed=42,
    fp16=False, bf16=False, batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=1000, save_strategy="no", report_to=[],
)
tr = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
t0 = time.time()
tr.train()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"RESULT mode={mode} bs={bs} mini={mini} steps={steps} "
      f"total={dt:.1f}s per_step={dt/steps:.2f}s peak_alloc={peak:.2f}GiB "
      f"epoch_est={(len(rows)/bs)*(dt/steps):.0f}s", flush=True)
