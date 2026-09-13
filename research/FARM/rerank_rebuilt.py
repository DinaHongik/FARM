"""Cross-encoder reranking on the rebuilt corpus, graded on url identity.

The previous reranker programme ran on a label that credited any of 91 podcast
channels' "New episode", and capped candidates at k=5 (ceiling 0.716). Here the
label is the url and k is a swept parameter.

API per sentence-transformers 5.7: CrossEncoderTrainer + a listwise ranking loss.
CrossEncoder.fit() is deprecated. lambda_loss.py notes LambdaLoss "anecdotally
performs better than the other losses with the same input format", which matched
our own sweep (lambda/listnet/listmle within 0.001 of each other on the old label).
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                 CrossEncoderTrainingArguments)
from sentence_transformers.cross_encoder import losses as CL

ap = argparse.ArgumentParser()
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--loss", default="lambda", choices=["lambda", "listnet", "listmle", "ranknet"])
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--maxlen", type=int, default=384)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--graded", action="store_true",
                help="graded relevance: gold=3, same-channel-as-gold=1, else 0. With binary "
                     "single-positive labels the NDCG gain term has one non-zero level, which "
                     "is why lambda/listnet/listmle/ranknet landed within 0.001 of each other.")
ap.add_argument("--tag", default="")
A = ap.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
ENC = {"trigger": "runs/rebuilt/trigger-s42/final", "action": "runs/rebuilt/action-s42/final"}

tr = json.loads(Path("data/split/stage1_train.json").read_text()) \
   + json.loads(Path("data/split/rerank_train.json").read_text())
ev = json.loads(Path("data/split/eval.json").read_text())
tag = A.tag or (f"k{A.topk}_{A.base.split('/')[-1]}_e{A.epochs}_s{A.seed}"
                + ("_graded" if A.graded else ""))
print(f"{tag}: train={len(tr)} eval={len(ev)} k={A.topk} base={A.base} loss={A.loss} "
      f"bs={A.bs}x{A.accum} dev={DEV}", flush=True)

SIDES = {}
for kind in ("trigger", "action"):
    corpus = json.loads(Path(f"data/corpus/{kind}s.json").read_text())
    SIDES[kind] = {"urls": [c["url"] for c in corpus], "texts": [c["text"] for c in corpus],
                   "chans": [c["channel"] for c in corpus],
                   "u2i": {c["url"]: i for i, c in enumerate(corpus)}}


def retrieve(kind, recs):
    S = SIDES[kind]
    m = SentenceTransformer(ENC[kind], device=DEV, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(S["texts"], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in recs], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    assert np.isfinite(D).all() and np.isfinite(Q).all(), (
        "non-finite embeddings - check dtype. fp16 on EmbeddingGemma produces all-NaN, "
        "and a rank of (row > row[gold]).sum() is then 0 for every row: a silent R@1 of 1.000.")
    sims = Q @ D.T
    idx = np.argpartition(-sims, A.topk, axis=1)[:, :A.topk]
    return [sorted(row, key=lambda j: -sims[i, j]) for i, row in enumerate(idx)]


cand = {k: {"train": retrieve(k, tr), "eval": retrieve(k, ev)} for k in SIDES}

rows = {"query": [], "docs": [], "labels": []}
for kind, S in SIDES.items():
    for r, cs in zip(tr, cand[kind]["train"]):
        g = S["u2i"][r[f"{kind}_url"]]
        rows["query"].append(r["query"])
        rows["docs"].append([S["texts"][j] for j in cs])
        if A.graded:
            gch = S["chans"][g]
            rows["labels"].append([3.0 if j == g else (1.0 if S["chans"][j] == gch else 0.0)
                                   for j in cs])
        else:
            rows["labels"].append([1.0 if j == g else 0.0 for j in cs])
with_gold = sum(1 for l in rows["labels"] if max(l) > 0)
print(f"listwise rows={len(rows['query'])} containing the gold={with_gold} "
      f"({with_gold/len(rows['query']):.3f})", flush=True)

model = CrossEncoder(A.base, num_labels=1, device=DEV, max_length=A.maxlen)
LOSS = {"lambda": CL.LambdaLoss, "listnet": CL.ListNetLoss,
        "listmle": CL.ListMLELoss, "ranknet": CL.RankNetLoss}[A.loss]
args = CrossEncoderTrainingArguments(
    output_dir=f"runs/rerank_rebuilt/{tag}", num_train_epochs=A.epochs,
    per_device_train_batch_size=A.bs, gradient_accumulation_steps=A.accum,
    learning_rate=A.lr, warmup_ratio=0.1, seed=A.seed,
    fp16=False, bf16=False, logging_steps=200, save_strategy="no", report_to=[])
t0 = time.time()
CrossEncoderTrainer(model=model, args=args, train_dataset=Dataset.from_dict(rows),
                    loss=LOSS(model)).train()
print(f"trained in {time.time()-t0:.0f}s", flush=True)


def joint(use_model):
    """Joint accuracy: both sides' selected url equals the gold url."""
    picks = {}
    for kind, S in SIDES.items():
        sel = []
        for r, cs in zip(ev, cand[kind]["eval"]):
            if use_model is None:
                sel.append(cs[0])
            else:
                sc = use_model.predict([(r["query"], S["texts"][j]) for j in cs],
                                       batch_size=128, show_progress_bar=False)
                sel.append(cs[int(np.argmax(sc))])
        picks[kind] = sel
    ok = [1 if (picks["trigger"][i] == SIDES["trigger"]["u2i"][r["trigger_url"]] and
                picks["action"][i] == SIDES["action"]["u2i"][r["action_url"]]) else 0
          for i, r in enumerate(ev)]
    return np.array(ok)


floor = joint(None)
rr = joint(model)
ceil = np.array([1 if (SIDES["trigger"]["u2i"][r["trigger_url"]] in cand["trigger"]["eval"][i] and
                       SIDES["action"]["u2i"][r["action_url"]] in cand["action"]["eval"][i]) else 0
                 for i, r in enumerate(ev)])
b = int(((floor == 1) & (rr == 0)).sum())
c = int(((floor == 0) & (rr == 1)).sum())
chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
p = math.erfc(math.sqrt(chi / 2)) if chi else 1.0
res = {"tag": tag, "topk": A.topk, "base": A.base, "loss": A.loss, "epochs": A.epochs,
       "seed": A.seed, "floor": float(floor.mean()), "reranked": float(rr.mean()),
       "ceiling": float(ceil.mean()), "b": b, "c": c, "p": p, "n": len(ev),
       "per_record": [{"applet_url": r["applet_url"], "floor": int(floor[i]),
                       "reranked": int(rr[i]), "in_candidates": int(ceil[i])}
                      for i, r in enumerate(ev)]}
model.save_pretrained(f"runs/rerank_rebuilt/{tag}/final")
print(f"saved reranker -> runs/rerank_rebuilt/{tag}/final", flush=True)
shortlists = {}
for i, r in enumerate(ev):
    entry = {}
    for kind, S in SIDES.items():
        cs = cand[kind]["eval"][i]
        sc = model.predict([(r["query"], S["texts"][j]) for j in cs],
                           batch_size=128, show_progress_bar=False)
        entry[kind] = [S["urls"][j] for j in sorted(cs, key=lambda j: -sc[list(cs).index(j)])]
    shortlists[r["applet_url"]] = entry
Path("data/candidates").mkdir(parents=True, exist_ok=True)
Path(f"data/candidates/reranked_{tag}.json").write_text(json.dumps(shortlists))
Path("results/rerank_rebuilt").mkdir(parents=True, exist_ok=True)
Path(f"results/rerank_rebuilt/{tag}.json").write_text(json.dumps(res, indent=1))
print(f"\n  encoder rank-1 floor  {res['floor']:.3f}")
print(f"  reranked              {res['reranked']:.3f}   ({res['reranked']-res['floor']:+.3f})")
print(f"  top-{A.topk} ceiling        {res['ceiling']:.3f}")
hr = (res['reranked'] - res['floor']) / (res['ceiling'] - res['floor']) if res['ceiling'] > res['floor'] else 0
print(f"  headroom captured     {hr:.1%}")
print(f"  McNemar b={b} c={c} chi2={chi:.2f} p={p:.3e} "
      f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}", flush=True)
