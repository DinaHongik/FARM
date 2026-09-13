"""Mine hard negatives for the rebuilt corpus.

Three things the old miner got wrong, all fixed here:
  1. it mined from train_applets_dedup.json, 89.5% of whose anchors are eval
     queries. This mines from stage1_train only.
  2. its gold-exclusion looked the gold up by RENDERED TEXT, which never matched
     (0/10786 rows), so the filters were dead code. This excludes by url.
  3. its max_sim=0.85 ceiling existed to suppress the same function rendered twice
     by two code paths. With one renderer that ceiling only discards legitimate
     sibling-channel negatives, so it goes to 0.98.

Same-name-different-channel candidates are KEPT. Under url identity those are
exactly the hard negatives the task needs: "turn off" is 27 distinct urls.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, torch
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--n-neg", type=int, default=4)
ap.add_argument("--max-sim", type=float, default=0.98)
ap.add_argument("--min-sim", type=float, default=0.35)
ap.add_argument("--trigger-model", default="runs/rebuilt/trigger-s42/final")
ap.add_argument("--action-model", default="runs/rebuilt/action-s42/final")
ap.add_argument("--augmented", action="store_true",
                help="mine per anchor from the paraphrase-augmented pairs")
A = ap.parse_args()

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
if A.augmented:
    # mine per ANCHOR, not per applet: the paraphrase rows are extra anchors that
    # point at the same gold, so each needs its own negatives mined against its own
    # wording. Rebuilt from the trigger-side pairs file, which carries both anchors.
    _t = json.loads(Path("data/pairs/trigger_stage1_train_aug.json").read_text())
    _a = {(x["applet_url"], x["anchor"]): x for x in
          json.loads(Path("data/pairs/action_stage1_train_aug.json").read_text())}
    split = []
    for x in _t:
        m = _a.get((x["applet_url"], x["anchor"]))
        if m is None:
            continue
        split.append({"query": x["anchor"], "nq": x["nq"], "applet_url": x["applet_url"],
                      "trigger_url": x["label_url"], "action_url": m["label_url"]})
    print(f"mining over {len(split)} anchors (augmented)", flush=True)
else:
    split = json.loads(Path("data/split/stage1_train.json").read_text())

# multi-gold: every url that is gold for ANY record sharing this record's nq group
by_nq = defaultdict(lambda: {"trigger": set(), "action": set()})
for r in split:
    by_nq[r["nq"]]["trigger"].add(r["trigger_url"])
    by_nq[r["nq"]]["action"].add(r["action_url"])

for kind, mp in (("trigger", A.trigger_model), ("action", A.action_model)):
    corpus = json.loads(Path(f"data/corpus/{kind}s.json").read_text())
    urls = [c["url"] for c in corpus]
    texts = [c["text"] for c in corpus]
    u2i = {u: i for i, u in enumerate(urls)}
    # byte-identical identities: same service_name AND description
    ident = defaultdict(set)
    for i, c in enumerate(corpus):
        ident[(str(c["service_name"]).strip().lower(), str(c["description"]).strip())].add(i)

    m = SentenceTransformer(mp, device=dev, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False)
    Q = m.encode([r["query"] for r in split], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    S = (Q @ D.T).astype(np.float32)

    rows, stats = [], defaultdict(int)
    for i, r in enumerate(split):
        g = r[f"{kind}_url"]
        gi = u2i[g]
        banned = {u2i[u] for u in by_nq[r["nq"]][kind] if u in u2i}
        banned |= ident[(str(corpus[gi]["service_name"]).strip().lower(),
                         str(corpus[gi]["description"]).strip())]
        order = np.argsort(-S[i])
        negs = []
        for j in order:
            if len(negs) >= A.n_neg:
                break
            if j in banned:
                stats["banned"] += 1
                continue
            s = float(S[i, j])
            if s > A.max_sim:
                stats["too_similar"] += 1
                continue
            if s < A.min_sim:
                break
            negs.append(int(j))
        if negs:
            same_chan = sum(1 for j in negs if corpus[j]["channel"] == corpus[gi]["channel"])
            same_name = sum(1 for j in negs
                            if str(corpus[j]["service_name"]).strip().lower()
                            == str(corpus[gi]["service_name"]).strip().lower())
            stats["same_channel"] += same_chan
            stats["same_name_other_channel"] += same_name
            stats["negs"] += len(negs)
        rows.append({"anchor": r["query"], "positive": texts[gi], "label_url": g,
                     "nq": r["nq"], "negatives": [texts[j] for j in negs],
                     "negative_urls": [urls[j] for j in negs]})

    out = Path(f"data/pairs/{kind}_mined{'_aug' if A.augmented else ''}.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n = stats["negs"] or 1
    print(f"{kind}: rows={len(rows)} negatives={stats['negs']} "
          f"(banned {stats['banned']}, over max_sim {stats['too_similar']}) | "
          f"same channel as gold {stats['same_channel']}/{n} = {stats['same_channel']/n:.1%}, "
          f"same name other channel {stats['same_name_other_channel']}/{n} = "
          f"{stats['same_name_other_channel']/n:.1%}", flush=True)
