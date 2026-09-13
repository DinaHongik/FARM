"""Cross-encoder reranker over the CLEAN split.

Targets the selection headroom: joint rank-1 = 0.423, joint top-5 = 0.716, so
0.293 sits between them. A hand-designed schema heuristic reaches 0.458-at-best
(on the old split) and a prompted LLM reaches no better than rank-1, so the
remaining option is a learned reranker.

Training data is `rerank_train`, which is held out from the encoder's own
training set. That matters: candidates generated over stage1_train would reflect
memorised retrieval, giving unrealistically easy negatives.
"""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader
from rag.indexer import extract_trigger_text, extract_action_text

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
def nz(x): return str(x or "").strip().lower()

trag = json.load(open("data/triggers_rag.json"))
arag = json.load(open("data/actions_rag.json"))
SIDES = {
    "trigger": dict(rag=trag, docs=[extract_trigger_text(r) for r in trag],
                    names=[nz(r.get("service_name")) for r in trag],
                    enc="runs/clean/trigger-encoder/final"),
    "action":  dict(rag=arag, docs=[extract_action_text(r) for r in arag],
                    names=[nz(r.get("service_name")) for r in arag],
                    enc="runs/clean/action-encoder/final"),
}

def retrieve(side, records):
    """Top-k candidate indices from the CLEAN encoder."""
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

tr = json.load(open("data/clean_split/rerank_train.json"))
ev = json.load(open("data/clean_split/eval.json"))
print(f"seed={A.seed}  rerank_train={len(tr)}  eval={len(ev)}  device={DEV}", flush=True)

cand = {s: {"train": retrieve(s, tr), "eval": retrieve(s, ev)} for s in SIDES}
print("candidates generated", flush=True)

def gold(side, r): return nz(r[side].get("service_name"))

# ---- training pairs: gold in the candidate list = positive, the rest = negatives ----
examples = []
for side, S in SIDES.items():
    for r, cs in zip(tr, cand[side]["train"]):
        g = gold(side, r)
        for j in cs:
            examples.append(InputExample(texts=[r["query"], S["docs"][j]],
                                         label=1.0 if S["names"][j] == g else 0.0))
pos = sum(1 for e in examples if e.label == 1.0)
print(f"training pairs={len(examples)}  positives={pos}  negatives={len(examples)-pos}", flush=True)

def side_at1(side, scorer=None):
    """Per-side R@1, plus accuracy restricted to lists that CONTAIN the gold.
    The second number is the diagnostic: if it is near 1/topk the model learned nothing."""
    S = SIDES[side]; hit = 0; hit_c = 0; n_c = 0
    for r, cs in zip(ev, cand[side]["eval"]):
        g = gold(side, r)
        names = [S["names"][j] for j in cs]
        if scorer is None:
            pick = names[0]
        else:
            sc = scorer.predict([[r["query"], S["docs"][j]] for j in cs], show_progress_bar=False)
            pick = names[int(np.argmax(sc))]
        if pick == g: hit += 1
        if g in names:
            n_c += 1
            if pick == g: hit_c += 1
    return hit/len(ev), (hit_c/n_c if n_c else None), n_c

def joint_at1(scorer=None):
    """JOINT@1 over the eval set, optionally reordering candidates with the reranker."""
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

base_j1, base_hits = joint_at1(None)
ceiling = sum(1 for i, r in enumerate(ev)
              if gold("trigger", r) in {SIDES["trigger"]["names"][j] for j in cand["trigger"]["eval"][i]}
              and gold("action", r) in {SIDES["action"]["names"][j] for j in cand["action"]["eval"][i]}) / len(ev)
print(f"\nrank-1 floor (no reranker) : {base_j1:.3f}")
print(f"top-{A.topk} ceiling           : {ceiling:.3f}")
print(f"headroom                   : {ceiling-base_j1:.3f}\n", flush=True)

ce = CrossEncoder(A.base, num_labels=1, device=DEV, max_length=256)
t0 = time.time()
ce.fit(train_dataloader=DataLoader(examples, shuffle=True, batch_size=32),
       epochs=A.epochs, warmup_steps=int(0.1*len(examples)/32), show_progress_bar=False)
print(f"trained in {time.time()-t0:.0f}s", flush=True)

rr_j1, rr_hits = joint_at1(ce)
print(f"\n{'':<28}{'JOINT@1':>9}")
print("-"*38)
print(f"{'rank-1 (no reranker)':<28}{base_j1:>9.3f}")
print(f"{'cross-encoder reranked':<28}{rr_j1:>9.3f}")
print(f"{'top-5 ceiling':<28}{ceiling:>9.3f}")
print(f"\ndelta vs floor: {rr_j1-base_j1:+.3f}   headroom captured: "
      f"{(rr_j1-base_j1)/max(1e-9, ceiling-base_j1):.1%}")
import math
b = sum(1 for x, y in zip(base_hits, rr_hits) if x == 1 and y == 0)   # floor right, reranker wrong
c = sum(1 for x, y in zip(base_hits, rr_hits) if x == 0 and y == 1)   # reranker right, floor wrong
chi2 = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0          # continuity-corrected
pval = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 else 1.0
print(f"\nPAIRED test (McNemar). The floor and the reranker are scored on identical samples")
print(f"with identical candidate lists - only the selector differs - so only discordant")
print(f"pairs carry information and an unpaired interval is the wrong test.")
print(f"  floor right / reranker wrong (b) : {b}")
print(f"  reranker right / floor wrong (c) : {c}")
print(f"  chi2={chi2:.2f}  p={pval:.4g}   {'SIGNIFICANT at .05' if pval < 0.05 else 'NOT significant'}")
print(f"  (unpaired Wilson half-width, too conservative here: +/-{1.96*(0.25/len(ev))**0.5:.3f})")
json.dump({"seed": A.seed, "epochs": A.epochs, "base": base_hits, "rr": rr_hits},
          open(f"logs/rr_hits_s{A.seed}_e{A.epochs}.json", "w"))

print(f"\nper-side diagnostic (oracle-in-list accuracy; random would be {1.0/A.topk:.3f}):")
print(f"{'side':<10}{'R@1 base':>10}{'R@1 rerank':>12}{'in-list base':>14}{'in-list rerank':>16}")
for side in SIDES:
    b1, bc, nc = side_at1(side, None)
    r1, rc, _  = side_at1(side, ce)
    print(f"{side:<10}{b1:>10.3f}{r1:>12.3f}{bc:>14.3f}{rc:>16.3f}   (lists containing gold: {nc})")
