"""Evaluate stage-1 retrieval on the rebuilt corpus, graded on url identity.

A hit requires retrieved_url == gold_url. No service_name equality (which credits
any of 91 podcast channels' "New episode") and no substring containment (which
credits "Turn on" against "Turn on light").
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np, torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from farm.render import pretty_channel

ap = argparse.ArgumentParser()
ap.add_argument("--trigger-model", required=True)
ap.add_argument("--action-model", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default="results/rebuilt")
A = ap.parse_args()

KS = [1, 5, 10]
ev = json.loads(Path("data/split/eval.json").read_text())
s1 = json.loads(Path("data/split/stage1_train.json").read_text())
dev = "cuda:0" if torch.cuda.is_available() else "cpu"


def canon(s): return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


out = {"tag": A.tag, "n_eval": len(ev), "sides": {}}
ranks = {}
for kind, mp in (("trigger", A.trigger_model), ("action", A.action_model)):
    corpus = json.loads(Path(f"data/corpus/{kind}s.json").read_text())
    urls = [c["url"] for c in corpus]
    u2i = {u: i for i, u in enumerate(urls)}
    names = [str(c["service_name"]).strip().lower() for c in corpus]
    gold_u = [r[f"{kind}_url"] for r in ev]
    gi = [u2i[u] for u in gold_u]

    m = SentenceTransformer(mp, device=dev, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode([c["text"] for c in corpus], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    assert np.isfinite(D).all() and np.isfinite(Q).all(), (
        "non-finite embeddings - check dtype. fp16 on EmbeddingGemma produces all-NaN, "
        "and a rank of (row > row[gold]).sum() is then 0 for every row: a silent R@1 of 1.000.")
    S = (Q @ D.T).astype(np.float32)
    rk = np.array([int((S[i] > S[i, g]).sum()) for i, g in enumerate(gi)])
    ranks[kind] = rk

    # bridge row: the OLD name-level label, so new numbers map onto old ones
    by_name = {}
    for i, n in enumerate(names): by_name.setdefault(n, []).append(i)
    gname = [names[g] for g in gi]
    rk_name = np.array([int((S[i] > max(S[i, j] for j in by_name[n])).sum())
                        for i, n in enumerate(gname)])

    side = {f"R@{k}": float((rk < k).mean()) for k in KS}
    side["R@1_name_level_bridge"] = float((rk_name < 1).mean())
    out["sides"][kind] = side
    print(f"{kind:8} " + "  ".join(f"R@{k}={side[f'R@{k}']:.3f}" for k in KS)
          + f"   [name-level bridge R@1={side['R@1_name_level_bridge']:.3f}]", flush=True)

j = {f"R@{k}": float(((ranks['trigger'] < k) & (ranks['action'] < k)).mean()) for k in KS}
out["joint"] = j
print("joint    " + "  ".join(f"R@{k}={j[f'R@{k}']:.3f}" for k in KS), flush=True)

# strata
seen = {k: {r[f"{k}_url"] for r in s1} for k in ("trigger", "action")}
named = np.array([canon(r["trigger_channel"]) in canon(r["query"]) for r in ev])
unseen = np.array([r["trigger_url"] not in seen["trigger"] or r["action_url"] not in seen["action"]
                   for r in ev])
hit1 = (ranks["trigger"] < 1) & (ranks["action"] < 1)
out["strata"] = {
    "query_names_trigger_channel": {"n": int(named.sum()), "joint_R@1": float(hit1[named].mean()) if named.any() else None},
    "query_does_not": {"n": int((~named).sum()), "joint_R@1": float(hit1[~named].mean()) if (~named).any() else None},
    "gold_unseen_in_stage1": {"n": int(unseen.sum()), "joint_R@1": float(hit1[unseen].mean()) if unseen.any() else None},
    "gold_seen": {"n": int((~unseen).sum()), "joint_R@1": float(hit1[~unseen].mean()) if (~unseen).any() else None},
}
for k, v in out["strata"].items():
    print(f"  {k:32} n={v['n']:5}  joint R@1={v['joint_R@1']}", flush=True)

Path(A.out).mkdir(parents=True, exist_ok=True)
Path(f"{A.out}/{A.tag}.json").write_text(json.dumps(out, indent=1))
print(f"wrote {A.out}/{A.tag}.json")
