"""Re-measure Stage-1 retrieval on the CLEAN eval set.

Runs BOTH encoders over the SAME clean eval queries:
  - models/            : shipped encoder, trained on the leaky split
  - runs/clean/        : retrained on data/clean_split/stage1_train.json
Holding the eval set fixed isolates the encoder, so the delta is what the
contamination was actually worth.

Identity is the function's service_name, matching how the index is keyed and how
every other number in this project is scored. Exact match, never service_matches()
(which accepts 44.9% of non-gold candidates from the same top-5).
"""
import json, sys, time, os
import numpy as np, torch
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
from rag.indexer import extract_trigger_text, extract_action_text

TOPK = 5
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
def nz(x): return str(x or "").strip().lower()

ev = json.load(open("data/clean_split/eval.json"))
trag = json.load(open("data/triggers_rag.json"))
arag = json.load(open("data/actions_rag.json"))
print(f"eval queries={len(ev)}  trigger docs={len(trag)}  action docs={len(arag)}  device={DEV}", flush=True)

tdocs = [extract_trigger_text(r) for r in trag]
adocs = [extract_action_text(r) for r in arag]
tnames = [nz(r.get("service_name")) for r in trag]
anames = [nz(r.get("service_name")) for r in arag]
queries = [r["query"] for r in ev]
gt = [nz(r["trigger"].get("service_name")) for r in ev]
ga = [nz(r["action"].get("service_name")) for r in ev]

def topk(model_path, docs, names, golds):
    m = SentenceTransformer(model_path, device=DEV, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode(queries, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = Q @ D.T
    idx = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]
    ranked = [sorted(row, key=lambda j: -sims[i, j]) for i, row in enumerate(idx)]
    del m; torch.cuda.empty_cache()
    r1 = [names[r[0]] == g for r, g in zip(ranked, golds)]
    r5 = [g in {names[j] for j in r} for r, g in zip(ranked, golds)]
    return r1, r5

rows = {}
for label, tp, ap in (("shipped (leaky-split encoder)", "models/trigger-encoder/final", "models/action-encoder/final"),
                      ("CLEAN retrain",                "runs/clean/trigger-encoder/final", "runs/clean/action-encoder/final")):
    if not os.path.isdir(tp):
        print(f"skip {label}: {tp} missing", flush=True); continue
    t0 = time.time()
    t1, t5 = topk(tp, tdocs, tnames, gt)
    a1, a5 = topk(ap, adocs, anames, ga)
    n = len(ev)
    rows[label] = dict(
        t1=sum(t1)/n, t5=sum(t5)/n, a1=sum(a1)/n, a5=sum(a5)/n,
        j1=sum(1 for x, y in zip(t1, a1) if x and y)/n,
        j5=sum(1 for x, y in zip(t5, a5) if x and y)/n,
        secs=time.time()-t0)
    print(f"  {label} done in {rows[label]['secs']:.0f}s", flush=True)

print(f"\n{'encoder':<32}{'trig@1':>8}{'trig@5':>8}{'act@1':>8}{'act@5':>8}{'JOINT@1':>9}{'JOINT@5':>9}")
print("-"*82)
for k, v in rows.items():
    print(f"{k:<32}{v['t1']:>8.3f}{v['t5']:>8.3f}{v['a1']:>8.3f}{v['a5']:>8.3f}{v['j1']:>9.3f}{v['j5']:>9.3f}")

if len(rows) == 2:
    a, b = list(rows.values())
    print(f"\ncontamination delta (shipped - clean), same eval set:")
    for k, lbl in (("j1", "JOINT@1"), ("j5", "JOINT@5")):
        print(f"  {lbl:<9} {a[k]:.3f} -> {b[k]:.3f}   ({b[k]-a[k]:+.3f})")
print(f"\nfor reference, the contaminated-eval numbers this project has been quoting:")
print(f"  joint rank-1 0.455   joint top-5 0.779   (n=299, old split)")
