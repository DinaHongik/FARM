"""Reranker v4: negatives mined the way the official recipe does it.

Why v1-v3 plateaued at ~11% of headroom. Their negatives were simply the
retriever's top-5 minus the gold. In this corpus candidates 2-5 are frequently
near-paraphrases of the gold ("Price rises above" vs "Today's price rises by
percentage"), so that scheme trains the model that near-correct answers are
wrong, and 12 epochs burns that noise in.

The official example (examples/cross_encoder/training/ms_marco/
training_ms_marco_lambda_hard_neg.py) does the opposite:

    mine_hard_negatives(..., num_negatives=9, range_min=3,
                        range_max=3 + 9*3, output_format="labeled-list")

range_min=3 deliberately SKIPS the three hardest candidates because they are
likely false negatives. range_max widens the band so the model also sees
clearly-wrong documents. The HF reranker guide adds: "if you only use hard
negatives, your model may unexpectedly perform worse for easier tasks ...
training using random negatives alongside hard negatives can mitigate this."
"""
import argparse, json, math, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder import losses as CL
from rag.indexer import extract_trigger_text, extract_action_text

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--num-neg", type=int, default=9)
ap.add_argument("--range-min", type=int, default=3)     # skip the hardest: likely false negatives
ap.add_argument("--range-max", type=int, default=30)
ap.add_argument("--max-score", type=float, default=0.8)
ap.add_argument("--sampling", default="top", choices=["top", "random"])
ap.add_argument("--loss", default="lambda", choices=["lambda", "bce"])
ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--tag", default="v4")
ap.add_argument("--data", default="rerank", choices=["rerank", "all"],
                help="'all' adds stage1_train. Disjoint from eval by query group (verified 0/0/0), "
                     "so no eval query is ever seen. v3 showed this is the dominant factor: "
                     "+0.082 vs +0.028 for a 12.6x larger base model.")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
def nz(x): return str(x or "").strip().lower()

trag = json.load(open("data/triggers_rag.json")); arag = json.load(open("data/actions_rag.json"))
SIDES = {
    "trigger": dict(docs=[extract_trigger_text(r) for r in trag],
                    names=[nz(r.get("service_name")) for r in trag],
                    enc="runs/clean/trigger-encoder/final"),
    "action":  dict(docs=[extract_action_text(r) for r in arag],
                    names=[nz(r.get("service_name")) for r in arag],
                    enc="runs/clean/action-encoder/final"),
}
tr = json.load(open("data/clean_split/rerank_train.json"))
if A.data == "all":
    tr = tr + json.load(open("data/clean_split/stage1_train.json"))
ev = json.load(open("data/clean_split/eval.json"))
print(f"[{A.tag}] seed={A.seed} epochs={A.epochs} num_neg={A.num_neg} "
      f"range=[{A.range_min},{A.range_max}] max_score={A.max_score} sampling={A.sampling} "
      f"loss={A.loss} base={A.base} data={A.data} train={len(tr)}", flush=True)

frames, cand_eval = [], {}
for side, S in SIDES.items():
    enc = SentenceTransformer(S["enc"], device=DEV, model_kwargs={"torch_dtype": torch.float32})
    # name -> its doc text, so a gold can be resolved to the positive document
    by_name = {}
    for n_, d_ in zip(S["names"], S["docs"]): by_name.setdefault(n_, d_)
    pairs = {"query": [], "answer": []}
    for r in tr:
        g = gold = nz(r[side].get("service_name"))
        if g in by_name:
            pairs["query"].append(r["query"]); pairs["answer"].append(by_name[g])
    print(f"  {side}: {len(pairs['query'])}/{len(tr)} training queries have a resolvable positive", flush=True)
    mined = mine_hard_negatives(
        dataset=Dataset.from_dict(pairs), model=enc, corpus=S["docs"],
        anchor_column_name="query", positive_column_name="answer",
        num_negatives=A.num_neg, range_min=A.range_min, range_max=A.range_max,
        max_score=A.max_score, sampling_strategy=A.sampling,
        output_format="labeled-list", batch_size=256, use_faiss=False, verbose=False)
    frames.append(mined)

    # evaluation candidates: unchanged from every earlier run, so numbers stay comparable
    D = enc.encode(S["docs"], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = enc.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    sims = Q @ D.T
    idx = np.argpartition(-sims, A.topk, axis=1)[:, :A.topk]
    cand_eval[side] = [sorted(row, key=lambda j: -sims[i, j]) for i, row in enumerate(idx)]
    del enc; torch.cuda.empty_cache()

from datasets import concatenate_datasets
ds = concatenate_datasets(frames).shuffle(seed=A.seed)
print(f"  mined dataset: {len(ds)} rows, columns={ds.column_names}", flush=True)
ex = ds[0]
print(f"  example list length: {len(ex[ds.column_names[1]])}  labels={ex[ds.column_names[2]]}", flush=True)

model = CrossEncoder(A.base, num_labels=1, device=DEV, max_length=320)
loss = CL.LambdaLoss(model, weighting_scheme=CL.NDCGLoss2PPScheme(), mini_batch_size=16) \
       if A.loss == "lambda" else CL.BinaryCrossEntropyLoss(model)
args = CrossEncoderTrainingArguments(
    output_dir=f"runs/rerank_v4/{A.tag}", num_train_epochs=A.epochs,
    per_device_train_batch_size=16, learning_rate=2e-5, warmup_ratio=0.1, seed=A.seed,
    fp16=False, bf16=False, logging_steps=200, save_strategy="no", report_to=[])
t0 = time.time()
CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
print(f"  trained in {time.time()-t0:.0f}s", flush=True)

def gold(side, r): return nz(r[side].get("service_name"))
def joint(scorer=None):
    picks = {}
    for side, S in SIDES.items():
        got = []
        for r, cs in zip(ev, cand_eval[side]):
            if scorer is None: got.append(S["names"][cs[0]])
            else:
                sc = scorer.predict([[r["query"], S["docs"][j]] for j in cs], show_progress_bar=False)
                got.append(S["names"][cs[int(np.argmax(sc))]])
        picks[side] = got
    hits = [1 if (picks["trigger"][i] == gold("trigger", r) and picks["action"][i] == gold("action", r)) else 0
            for i, r in enumerate(ev)]
    return sum(hits)/len(hits), hits

base_j1, bh = joint(None); rr_j1, rh = joint(model)
ceil_ = sum(1 for i, r in enumerate(ev)
            if gold("trigger", r) in {SIDES["trigger"]["names"][j] for j in cand_eval["trigger"][i]}
            and gold("action", r) in {SIDES["action"]["names"][j] for j in cand_eval["action"][i]}) / len(ev)
b = sum(1 for x, y in zip(bh, rh) if x == 1 and y == 0)
c = sum(1 for x, y in zip(bh, rh) if x == 0 and y == 1)
chi2 = ((abs(b-c)-1)**2/(b+c)) if (b+c) else 0.0
p = math.erfc(math.sqrt(chi2/2.0)) if chi2 else 1.0
print(f"\nRESULT [{A.tag}]")
print(f"  rank-1 floor      {base_j1:.3f}")
print(f"  reranked          {rr_j1:.3f}   ({rr_j1-base_j1:+.3f})")
print(f"  top-{A.topk} ceiling     {ceil_:.3f}")
print(f"  headroom captured {(rr_j1-base_j1)/max(1e-9, ceil_-base_j1):.1%}")
print(f"  McNemar b={b} c={c} chi2={chi2:.2f} p={p:.4g}  {'SIGNIFICANT' if p<0.05 else 'not significant'}")
json.dump({"tag": A.tag, "floor": base_j1, "rr": rr_j1, "ceiling": ceil_, "p": p,
           "num_neg": A.num_neg, "range": [A.range_min, A.range_max], "sampling": A.sampling,
           "epochs": A.epochs, "loss": A.loss, "base": A.base},
          open(f"logs/{A.tag}.json", "w"))
