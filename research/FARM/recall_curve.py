"""What does top_k actually buy? No training, pure retrieval.

top_k=5 has been assumed all session. A reranker can only ever reach the
gold-in-candidates ceiling, so if that ceiling rises steeply with k, widening
the candidate set is worth more than any improvement to the reranker itself.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
from rag.indexer import extract_trigger_text, extract_action_text

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
def nz(x): return str(x or "").strip().lower()
KS = [1, 3, 5, 10, 20, 50, 100]

ev = json.load(open("data/clean_split/eval.json"))
trag = json.load(open("data/triggers_rag.json")); arag = json.load(open("data/actions_rag.json"))
SIDES = {
    "trigger": (["%s" % extract_trigger_text(r) for r in trag], [nz(r.get("service_name")) for r in trag],
                "runs/clean/trigger-encoder/final"),
    "action":  (["%s" % extract_action_text(r) for r in arag], [nz(r.get("service_name")) for r in arag],
                "runs/clean/action-encoder/final"),
}
hit = {}
for side, (docs, names, enc) in SIDES.items():
    m = SentenceTransformer(enc, device=DEV, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    sims = Q @ D.T
    order = np.argsort(-sims, axis=1)[:, :max(KS)]
    golds = [nz(r[side].get("service_name")) for r in ev]
    hit[side] = {k: [g in {names[j] for j in order[i, :k]} for i, g in enumerate(golds)] for k in KS}
    del m; torch.cuda.empty_cache()

n = len(ev)
print(f"n={n}   (clean encoder, clean eval)\n")
print(f"{'k':>5}{'trigger R@k':>14}{'action R@k':>13}{'JOINT R@k':>12}{'vs k=5':>10}")
print("-"*54)
j5 = None
for k in KS:
    t = sum(hit['trigger'][k])/n; a = sum(hit['action'][k])/n
    j = sum(1 for i in range(n) if hit['trigger'][k][i] and hit['action'][k][i])/n
    if k == 5: j5 = j
    d = "" if j5 is None or k == 5 else f"{j-j5:+.3f}"
    print(f"{k:>5}{t:>14.3f}{a:>13.3f}{j:>12.3f}{d:>10}")
print(f"\nrank-1 joint (the floor a reranker must beat): {sum(1 for i in range(n) if hit['trigger'][1][i] and hit['action'][1][i])/n:.3f}")
print("A reranker capturing ~11% of headroom at each k would reach:")
for k in KS:
    j = sum(1 for i in range(n) if hit['trigger'][k][i] and hit['action'][k][i])/n
    f = sum(1 for i in range(n) if hit['trigger'][1][i] and hit['action'][1][i])/n
    print(f"   k={k:<4} ceiling {j:.3f}  ->  {f + 0.11*(j-f):.3f}")
