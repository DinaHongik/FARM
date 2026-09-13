#!/usr/bin/env python3
"""A/B test: does the planner's query extraction help or hurt Stage 1 retrieval?

THE SUSPECTED BUG
    agents/prompts/planner.py:205 extract_concept() scans the LLM's reasoning PROSE
    for the first substring hit of a keyword list (["detect","when","if","sensor",
    "event","receives","new"]) and returns a fixed 5-word lowercased window around it.

      - `if kw in word` is a SUBSTRING match: "if" fires inside "notify"/"specific",
        "new" inside "renew", "do" inside "window".
      - It returns 5 words from the middle of a sentence, e.g.
        "price rising significantly) 2. the"
      - Measured on results/e2e_eval.json: 136/300 trigger and 116/300 action queries
        are verbatim slices of planner_reasoning.

    planner_node.py:81  `plan.get("trigger_query") or query`  means extraction FAILING
    is the good case - it falls back to the real user query.

THIS EXPERIMENT
    Holds the encoders and corpus fixed and varies ONLY the query text:
      A) the fragment the planner actually produced (what shipped)
      B) the full user query (what the fallback would have used)
    Reports R@1/R@5/MRR@5 for both, service-level and schema-level.

    That isolates the bug's cost in retrieval terms, with no retraining and no LLM calls.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from rag.indexer import extract_trigger_text, extract_action_text  # noqa: E402
from train.train_lora_ablation import normalize_name, get_schema_key  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rank_of(query_emb, corpus_emb, names, schemas, gold_name, gold_schema, max_k=10):
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), corpus_emb, dim=1)
    order = torch.argsort(sims, descending=True)[:max_k].tolist()
    svc_rank = sch_rank = None
    for pos, idx in enumerate(order, start=1):
        if svc_rank is None and normalize_name(names[idx]) == normalize_name(gold_name):
            svc_rank = pos
        if sch_rank is None and schemas[idx] == gold_schema:
            sch_rank = pos
    return svc_rank, sch_rank


def metrics(ranks, n):
    out = {}
    for k in (1, 5):
        out[f"R@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    out["MRR@5"] = sum(1.0 / r for r in ranks if r is not None and r <= 5) / n if n else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/e2e_eval.json")
    ap.add_argument("--out", default="results/AB_planner_query.json")
    args = ap.parse_args()

    print("loading corpora + production encoders ...", flush=True)
    triggers = load(ROOT / "data" / "triggers_rag.json")
    actions = load(ROOT / "data" / "actions_rag.json")

    t_texts, t_names, t_schemas = [], [], []
    for t in triggers:
        txt = extract_trigger_text(t)
        if txt:
            t_texts.append(txt); t_names.append(t.get("service_name", ""))
            t_schemas.append(get_schema_key(t.get("api_info", {}), True))
    a_texts, a_names, a_schemas = [], [], []
    for a in actions:
        txt = extract_action_text(a)
        if txt:
            a_texts.append(txt); a_names.append(a.get("service_name", ""))
            a_schemas.append(get_schema_key(a.get("api_info", {}), False))

    tm = SentenceTransformer(str(ROOT / "models" / "trigger-encoder" / "final"), device="cuda")
    am = SentenceTransformer(str(ROOT / "models" / "action-encoder" / "final"), device="cuda")
    t_emb = tm.encode(t_texts, convert_to_tensor=True, show_progress_bar=False)
    a_emb = am.encode(a_texts, convert_to_tensor=True, show_progress_bar=False)
    print(f"  corpus: {len(t_texts)} triggers, {len(a_texts)} actions", flush=True)

    # gold labels keyed by query, from the test splits
    gold = {}
    for split in ("gold", "noisy", "oneshot"):
        p = ROOT / "data" / "test" / f"{split}.json"
        if p.exists():
            for r in load(p):
                gold[r["query"].strip()] = r

    ps = load(ROOT / args.eval)["per_sample"]
    print(f"  eval samples: {len(ps)}, gold lookup: {len(gold)}\n", flush=True)

    res = {"A_fragment": {"t_svc": [], "t_sch": [], "a_svc": [], "a_sch": []},
           "B_full_query": {"t_svc": [], "t_sch": [], "a_svc": [], "a_sch": []}}
    n = 0
    n_differ = 0

    for r in ps:
        q = (r.get("query") or "").strip()
        g = gold.get(q)
        if not g:
            continue
        rp = r.get("raw_prediction", {})
        tq = (rp.get("trigger_search_query") or "").strip() or q
        aq = (rp.get("action_search_query") or "").strip() or q
        if tq != q or aq != q:
            n_differ += 1
        n += 1

        gt_t = g["trigger"].get("service_name", "")
        gt_ts = get_schema_key(g["trigger"].get("api_info", {}), True)
        gt_a = g["action"].get("service_name", "")
        gt_as = get_schema_key(g["action"].get("api_info", {}), False)

        for label, tqq, aqq in (("A_fragment", tq, aq), ("B_full_query", q, q)):
            te = tm.encode(tqq, convert_to_tensor=True, show_progress_bar=False)
            ae = am.encode(aqq, convert_to_tensor=True, show_progress_bar=False)
            ts, tsch = rank_of(te, t_emb, t_names, t_schemas, gt_t, gt_ts)
            as_, asch = rank_of(ae, a_emb, a_names, a_schemas, gt_a, gt_as)
            res[label]["t_svc"].append(ts); res[label]["t_sch"].append(tsch)
            res[label]["a_svc"].append(as_); res[label]["a_sch"].append(asch)

        if n % 50 == 0:
            print(f"  {n} samples ...", flush=True)

    print(f"\nmatched {n} samples ({n_differ} where the planner query differs from the user query)\n")
    print("=" * 78)
    print(f"{'':<30}{'A: planner fragment':>22}{'B: full user query':>22}{'B-A':>8}")
    print("=" * 78)
    summary = {}
    for field, label in (("t_svc", "trigger service"), ("t_sch", "trigger schema"),
                         ("a_svc", "action service"), ("a_sch", "action schema")):
        ma = metrics(res["A_fragment"][field], n)
        mb = metrics(res["B_full_query"][field], n)
        summary[label] = {"A": ma, "B": mb}
        for k in ("R@1", "R@5", "MRR@5"):
            d = mb[k] - ma[k]
            flag = "  <-- BUG COSTS" if d > 0.02 else ("  <-- extraction helps" if d < -0.02 else "")
            print(f"{label + ' ' + k:<30}{ma[k]:>22.4f}{mb[k]:>22.4f}{d:>+8.4f}{flag}")
        print("-" * 78)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "n_planner_query_differs": n_differ, "summary": summary}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nPositive B-A means using the full user query beats the planner's extracted")
    print("fragment - i.e. extract_concept() is actively destroying retrieval quality.")


if __name__ == "__main__":
    main()
