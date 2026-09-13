"""Cache top-N retrieval candidates so the LLM experiments never touch a GPU."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="eval")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--trigger-model", default="runs/rebuilt/trigger-s42/final")
ap.add_argument("--action-model", default="runs/rebuilt/action-s42/final")
ap.add_argument("--out", default="data/candidates")
A = ap.parse_args()

recs = json.loads(Path(f"data/split/{A.split}.json").read_text())
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
out = {r["applet_url"]: {"query": r["query"],
                         "gold": {"trigger": r["trigger_url"], "action": r["action_url"]}}
       for r in recs}

for kind, mp in (("trigger", A.trigger_model), ("action", A.action_model)):
    corpus = json.loads(Path(f"data/corpus/{kind}s.json").read_text())
    m = SentenceTransformer(mp, device=dev, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode([c["text"] for c in corpus], batch_size=32, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in recs], batch_size=32, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    assert np.isfinite(D).all() and np.isfinite(Q).all(), (
        "non-finite embeddings - check dtype. fp16 on EmbeddingGemma produces all-NaN, "
        "and a rank of (row > row[gold]).sum() is then 0 for every row: a silent R@1 of 1.000.")
    S = (Q @ D.T).astype(np.float32)
    idx = np.argpartition(-S, A.n, axis=1)[:, :A.n]
    hit = 0
    for i, r in enumerate(recs):
        order = sorted(idx[i], key=lambda j: -S[i, j])
        out[r["applet_url"]][kind] = [
            {"url": corpus[j]["url"], "score": float(S[i, j])} for j in order]
        if r[f"{kind}_url"] in {corpus[j]["url"] for j in order}:
            hit += 1
    print(f"{kind}: recall@{A.n} = {hit/len(recs):.3f}", flush=True)

Path(A.out).mkdir(parents=True, exist_ok=True)
p = Path(A.out) / f"{A.split}_top{A.n}.json"
p.write_text(json.dumps(out, ensure_ascii=False))
print(f"wrote {p} ({len(out)} records)")
