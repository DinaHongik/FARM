"""Retrain the Stage-1 encoders on the CLEAN query-grouped split.

Why: the shipped encoders were trained on data/train_applets.json, which shares
39.0% of its normalised queries with the stored eval set (88.3% for the dedup
variant). Any retrieval number measured against that encoder is optimistic, and
a reranker scored on its candidates inherits the contamination.

Constraints honoured:
  - dtype float32: V100 is sm_70, bfloat16 is emulated and slower
  - ONE GPU per model (DataParallel changes InfoNCE batch semantics)
  - GPUs 0-2 only
Nothing is overwritten: outputs go to runs/clean/, the shipped models/ are untouched.
"""
import sys, os, json, time
sys.path.insert(0, ".")
from train.config import TrainingConfig

WHICH = sys.argv[1]  # "trigger" or "action"

cfg = TrainingConfig()
cfg.applets_data = "./data/clean_split/stage1_train.json"
cfg.dtype = "float32"
cfg.trigger_output_dir = "./runs/clean/trigger-encoder"
cfg.action_output_dir  = "./runs/clean/action-encoder"

print("=" * 64, flush=True)
print(f"CLEAN RETRAIN :: {WHICH}", flush=True)
print(f"  data      : {cfg.applets_path}", flush=True)
print(f"  records   : {len(json.load(open(cfg.applets_path)))}", flush=True)
print(f"  dtype     : {cfg.dtype}   (bfloat16 is emulated on V100 sm_70)", flush=True)
print(f"  epochs    : {cfg.epochs}  batch: {cfg.batch_size}  lr: {cfg.learning_rate}", flush=True)
print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"  started   : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 64, flush=True)

t0 = time.time()
if WHICH == "trigger":
    from train.train_trigger import train_trigger_encoder
    train_trigger_encoder(cfg)
else:
    from train.train_action import train_action_encoder
    train_action_encoder(cfg)
print(f"\nDONE {WHICH} in {time.time()-t0:.0f}s at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
