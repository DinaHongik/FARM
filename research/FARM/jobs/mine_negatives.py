#!/usr/bin/env python3
"""Mine hard negatives for FARM's contrastive training.

WHY
    3-approach.tex:96 claims "in-batch negative pairs (including hard negatives)".
    train/dataset.py:47-67 emits only (anchor, positive) - negatives are whatever
    else lands in the batch, i.e. RANDOM. In IR, "hard negative" means a MINED
    difficult example. The claim is currently unsupported.

    Measured consequence: joint R@1 equals trigger R@1 x action R@1 (lift +0.015
    on Run B, -0.046 to +0.040 across every config and split). The encoders learn
    topical similarity and nothing about which functions are confusable.

MODES
    similarity   (default) Standard IR hard negatives: functions with HIGH cosine
                 similarity to the query that are NOT the gold function. Directly
                 attacks the measured bottleneck - joint = product of marginals, so
                 raising marginal R@1 raises joint.

    schema       Experimental: negatives that are topically close to the gold but
                 whose schema signature differs (different required-field profile).
                 NOTE: do NOT naively mine "low coverage" negatives - 76% of gold
                 pairs themselves have zero required-field coverage, so coverage and
                 correctness are anti-aligned in this dataset and that objective
                 would push the model away from the gold answer.

Output: data/triplets_{trigger,action}.json  -> [{anchor, positive, negative}, ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from rag.indexer import extract_trigger_text, extract_action_text  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def schema_sig(api, is_trigger):
    """Coarse signature of a function's schema: its required-field / ingredient slugs."""
    if is_trigger:
        d = api.get("Ingredients") or {}
    else:
        d = {k: v for k, v in (api.get("Action fields") or {}).items()
             if isinstance(v, dict) and str(v.get("Required", "")).lower() == "true"}
    out = set()
    for name, spec in d.items():
        out.add(name.strip().lower())
        if isinstance(spec, dict) and spec.get("Slug"):
            out.add(str(spec["Slug"]).strip().lower())
    return out


def mine(side, applets, corpus, model_path, mode, n_neg, batch, device,
         max_sim=0.85, min_sim=0.35):
    is_trigger = side == "trigger"
    extract = extract_trigger_text if is_trigger else extract_action_text

    texts, metas = [], []
    for f in corpus:
        t = extract(f)
        if t:
            texts.append(t)
            metas.append(f)
    print(f"  [{side}] corpus {len(texts)}", flush=True)

    model = SentenceTransformer(str(model_path), device=device)
    cemb = model.encode(texts, convert_to_tensor=True, batch_size=batch,
                        show_progress_bar=False, normalize_embeddings=True)

    # index the gold text of each corpus entry so we can exclude it
    text_to_idx = {t: i for i, t in enumerate(texts)}
    sigs = [schema_sig(m.get("api_info") or {}, is_trigger) for m in metas]

    rows = []
    for a in applets:
        q = (a.get("query") or "").strip()
        if len(q) < 10:
            continue
        gold = a.get(side)
        if not gold:
            continue
        pos = extract(gold)
        if not pos:
            continue
        rows.append((q, pos))
    print(f"  [{side}] training rows {len(rows)}", flush=True)

    qs = [r[0] for r in rows]
    qemb = model.encode(qs, convert_to_tensor=True, batch_size=batch,
                        show_progress_bar=False, normalize_embeddings=True)

    # Embed the positives too, so we can measure candidate-vs-POSITIVE similarity.
    # This is the guard the first version lacked and it is not optional: the IFTTT
    # catalog contains near-duplicate entries for the same function (differing only
    # by an inserted "[category]" tag). Taking top-K by query similarity made 47% of
    # negatives cosine>0.95 to their own positive - FALSE NEGATIVES. Training InfoNCE
    # to separate two identical functions destroyed the encoder: trigger R@1 fell
    # 0.750 -> 0.030. A similarity CEILING is the standard fix.
    pemb = model.encode([r[1] for r in rows], convert_to_tensor=True, batch_size=batch,
                        show_progress_bar=False, normalize_embeddings=True)

    svc = [str(m.get("service_name", "")).strip().lower() for m in metas]
    gold_svc = []
    for a in applets:
        g = a.get(side) or {}
        gold_svc.append(str(g.get("service_name", "")).strip().lower())

    triplets = []
    rejected = {"gold": 0, "same_text": 0, "too_similar": 0, "too_easy": 0, "same_service": 0}
    K = max(n_neg + 40, 60)          # look deeper: most of the top-K is now filtered out

    for i in range(0, len(rows), 512):
        chunk = qemb[i:i + 512]
        sims = chunk @ cemb.T                      # cosine, both normalized
        top = torch.topk(sims, k=min(K, sims.shape[1]), dim=1).indices.tolist()
        pchunk = pemb[i:i + 512]
        for j, cand in enumerate(top):
            q, pos = rows[i + j]
            gold_idx = text_to_idx.get(pos)
            gold_sig = sigs[gold_idx] if gold_idx is not None else set()
            gsvc = gold_svc[i + j] if i + j < len(gold_svc) else ""
            # candidate-vs-positive similarity for this row
            pn = (cemb[cand] @ pchunk[j]).tolist()
            picked = 0
            for ci, s_pos in zip(cand, pn):
                if ci == gold_idx:
                    rejected["gold"] += 1; continue
                if texts[ci] == pos:
                    rejected["same_text"] += 1; continue
                if s_pos > max_sim:
                    rejected["too_similar"] += 1; continue      # <- false negative
                if s_pos < min_sim:
                    rejected["too_easy"] += 1; continue         # not a HARD negative
                # NOTE: no same-service filter. gold_svc was indexed over `applets`
                # while this loop indexes `rows` (which skips short queries), so the
                # two were misaligned and the check was a near-noop. More importantly
                # it is UNDESIRABLE: same-service/different-function pairs
                # ("Every year on" vs "Every day at") are the single best hard negative
                # for FUNCTION-level retrieval. The similarity ceiling above already
                # removes the near-duplicate entries that caused the first failure.
                if mode == "schema" and gold_sig and sigs[ci] == gold_sig:
                    continue
                triplets.append({"anchor": q, "positive": pos, "negative": texts[ci]})
                picked += 1
                if picked >= n_neg:
                    break
        if (i // 512) % 5 == 0:
            print(f"  [{side}] {min(i + 512, len(rows))}/{len(rows)}  kept={len(triplets)}", flush=True)

    print(f"  [{side}] rejected: {rejected}", flush=True)
    del model, cemb, qemb, pemb
    torch.cuda.empty_cache()
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applets", default="data/train_applets_dedup.json")
    ap.add_argument("--mode", choices=["similarity", "schema"], default="similarity")
    ap.add_argument("--n-neg", type=int, default=1, help="negatives per positive")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-sim", type=float, default=0.85,
                    help="reject candidates MORE similar than this to the positive "
                         "(they are false negatives - the catalog has near-duplicates)")
    ap.add_argument("--min-sim", type=float, default=0.35,
                    help="reject candidates LESS similar than this (not hard negatives)")
    ap.add_argument("--trigger-model", default="models/runB_ablation_full_finetune_trigger/final")
    ap.add_argument("--action-model", default="models/runB_ablation_full_finetune_action/final")
    args = ap.parse_args()

    applets = load(ROOT / args.applets)
    print(f"applets: {len(applets)}  mode={args.mode}  n_neg={args.n_neg}\n", flush=True)
    triggers = load(ROOT / "data" / "triggers_rag.json")
    actions = load(ROOT / "data" / "actions_rag.json")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for side, corpus, mpath in (("trigger", triggers, args.trigger_model),
                                ("action", actions, args.action_model)):
        p = ROOT / mpath
        if not (p / "config.json").exists() and not (p / "modules.json").exists():
            sys.exit(f"missing mining model: {p}\n"
                     f"(Run B must have finished full_finetune first)")
        trip = mine(side, applets, corpus, p, args.mode, args.n_neg, args.batch, dev,
                    max_sim=args.max_sim, min_sim=args.min_sim)
        out = ROOT / "data" / f"triplets_{side}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(trip, f)
        print(f"  [{side}] wrote {len(trip)} triplets -> {out.relative_to(ROOT)}\n", flush=True)

    print("done. Run C can now train on triplets instead of (anchor, positive) pairs.")


if __name__ == "__main__":
    main()
