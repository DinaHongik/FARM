"""Cross-encoder reranker, v2: modern API and a real ranking loss.

v1 used CrossEncoder.fit(), whose own docstring says it is "Deprecated ... from
before Sentence Transformers v4.0", with pointwise binary labels. Our task is
listwise: pick the right function out of 5 candidates. sentence-transformers
5.7.0 ships 11 ranking losses for exactly this, and lambda_loss.py:175 says it
"anecdotally performs better than the other losses with the same input format".
Input format, per lambda_loss.py:167 -- (query, [doc1..docN]) + [score1..scoreN].

v1 for reference: floor 0.423, best +0.041 at 12 epochs, McNemar p<0.05.
"""
import argparse, json, math, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder import losses as CL
from rag.indexer import extract_trigger_text, extract_action_text

ap = argparse.ArgumentParser()
ap.add_argument("--loss", default="lambda", choices=["lambda", "listnet", "listmle", "ranknet", "bce"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--data", default="rerank", choices=["rerank", "all"],
                help="'all' adds stage1_train: 8x more lists, but its candidates come from an "
                     "encoder that trained on them, so the negatives are easier than at test time")
ap.add_argument("--maxlen", type=int, default=320)
ap.add_argument("--bs", type=int, default=16, help="per-device batch in LISTS")
ap.add_argument("--accum", type=int, default=1,
                help="grad accumulation; bs*accum is the effective batch, keep it at 16")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
def nz(x): return str(x or "").strip().lower()

trag = json.load(open("data/triggers_rag.json"))
arag = json.load(open("data/actions_rag.json"))
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
print(f"loss={A.loss} seed={A.seed} epochs={A.epochs} base={A.base} data={A.data} "
      f"k={A.topk} maxlen={A.maxlen} | train={len(tr)} eval={len(ev)} dev={DEV}", flush=True)

def retrieve(side, records):
    S = SIDES[side]
    m = SentenceTransformer(S["enc"], device=DEV, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(S["docs"], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in records], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    sims = Q @ D.T
    idx = np.argpartition(-sims, A.topk, axis=1)[:, :A.topk]
    out = [sorted(row, key=lambda j: -sims[i, j]) for i, row in enumerate(idx)]
    del m; torch.cuda.empty_cache()
    return out

cand = {s: {"train": retrieve(s, tr), "eval": retrieve(s, ev)} for s in SIDES}
def gold(side, r): return nz(r[side].get("service_name"))

rows = {"query": [], "docs": [], "labels": []}
for side, S in SIDES.items():
    for r, cs in zip(tr, cand[side]["train"]):
        g = gold(side, r)
        rows["query"].append(r["query"])
        rows["docs"].append([S["docs"][j] for j in cs])
        rows["labels"].append([1.0 if S["names"][j] == g else 0.0 for j in cs])
with_gold = sum(1 for l in rows["labels"] if max(l) > 0)
print(f"listwise rows={len(rows['query'])}  lists containing the gold={with_gold} ({with_gold/len(rows['query']):.3f})", flush=True)

if A.loss == "bce":
    flat = {"query": [], "doc": [], "label": []}
    for q, dl, ll in zip(rows["query"], rows["docs"], rows["labels"]):
        for d, l in zip(dl, ll):
            flat["query"].append(q); flat["doc"].append(d); flat["label"].append(l)
    ds = Dataset.from_dict(flat)
else:
    ds = Dataset.from_dict(rows)

model = CrossEncoder(A.base, num_labels=1, device=DEV, max_length=A.maxlen)
LOSS = {"lambda": CL.LambdaLoss, "listnet": CL.ListNetLoss, "listmle": CL.ListMLELoss,
        "ranknet": CL.RankNetLoss, "bce": CL.BinaryCrossEntropyLoss}[A.loss]
loss = LOSS(model)

args = CrossEncoderTrainingArguments(
    output_dir=f"runs/rerank_v2/{A.loss}_s{A.seed}_e{A.epochs}_k{A.topk}_{A.base.split(chr(47))[-1]}",
    num_train_epochs=A.epochs, per_device_train_batch_size=A.bs,
    gradient_accumulation_steps=A.accum,
    learning_rate=A.lr, warmup_ratio=0.1, seed=A.seed,
    fp16=False, bf16=False,           # V100 sm_70: bf16 is emulated
    logging_steps=200, save_strategy="no", report_to=[],
)
t0 = time.time()
CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
print(f"trained in {time.time()-t0:.0f}s", flush=True)

def joint(scorer=None):
    picks = {}
    for side, S in SIDES.items():
        got = []
        for r, cs in zip(ev, cand[side]["eval"]):
            if scorer is None:
                got.append(S["names"][cs[0]])
            else:
                sc = scorer.predict([[r["query"], S["docs"][j]] for j in cs], show_progress_bar=False)
                got.append(S["names"][cs[int(np.argmax(sc))]])
        picks[side] = got
    hits = [1 if (picks["trigger"][i] == gold("trigger", r) and picks["action"][i] == gold("action", r)) else 0
            for i, r in enumerate(ev)]
    return sum(hits)/len(hits), hits

base_j1, base_hits = joint(None)
rr_j1, rr_hits = joint(model)
ceiling = sum(1 for i, r in enumerate(ev)
              if gold("trigger", r) in {SIDES["trigger"]["names"][j] for j in cand["trigger"]["eval"][i]}
              and gold("action", r) in {SIDES["action"]["names"][j] for j in cand["action"]["eval"][i]}) / len(ev)

b = sum(1 for x, y in zip(base_hits, rr_hits) if x == 1 and y == 0)
c = sum(1 for x, y in zip(base_hits, rr_hits) if x == 0 and y == 1)
chi2 = ((abs(b-c)-1)**2/(b+c)) if (b+c) else 0.0
p = math.erfc(math.sqrt(chi2/2.0)) if chi2 else 1.0

print(f"\nLOSS={A.loss}  seed={A.seed}  epochs={A.epochs}")
print(f"  rank-1 floor      {base_j1:.3f}")
print(f"  reranked          {rr_j1:.3f}   ({rr_j1-base_j1:+.3f})")
print(f"  top-5 ceiling     {ceiling:.3f}")
print(f"  headroom captured {(rr_j1-base_j1)/max(1e-9, ceiling-base_j1):.1%}")
print(f"  McNemar b={b} c={c} chi2={chi2:.2f} p={p:.4g}  {'SIGNIFICANT' if p<0.05 else 'not significant'}")
json.dump({"base": A.base, "topk": A.topk, "loss": A.loss, "seed": A.seed, "epochs": A.epochs, "floor": base_j1, "rr": rr_j1,
           "ceiling": ceiling, "b": b, "c": c, "p": p},
          open(f"logs/v2_{A.loss}_s{A.seed}_e{A.epochs}_{A.data}_k{A.topk}_{A.base.split(chr(47))[-1]}.json", "w"))
