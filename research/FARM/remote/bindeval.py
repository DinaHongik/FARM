#!/usr/bin/env python3
"""Field-level binding evaluation - the metric the paper's title claims and never reports.

WHY THIS EXISTS
    eval/evaluate_e2e.py:516-517 aliases binding_precision/recall to the slot metrics
    ("# Reuse slot metrics"), and NO test record carries a `bindings` key, so
    binding_accuracy / slot_recall / JGA all return 1.0 by empty-collection default.
    Those numbers are vacuous and must not be quoted.

    But the data needed for real, annotation-free binding metrics is already present:
      - api_info["Trigger fields"] / ["Ingredients"]  -> what the trigger EMITS
      - api_info["Action fields"]  (Required: true)   -> what the action DEMANDS
      - final_applet.bindings / action.field_values   -> what the system PRODUCED

    So three metrics need no gold annotation at all:

      1. REQUIRED-FIELD COVERAGE - of the action fields marked Required:true, how many
         did the system actually fill? An unfilled required field means the applet
         cannot execute. This is the executability floor.

      2. INGREDIENT GROUNDING - a value of the form {{X}} references an ingredient X.
         Is X actually an Ingredient of the SELECTED trigger? If not it is a
         hallucinated reference and the applet breaks at runtime. This is the
         precise, checkable version of what the RAGAS faithfulness proxy was groping at.

      3. BINDING SOURCE MIX - what fraction of filled fields are wired to a trigger
         ingredient versus filled with an invented static literal? This goes straight
         at the paper's central claim. "Field-Aware Resolution" means cross-schema
         ingredient->field wiring. If most fields are static literals, the system is
         doing form-filling, not resolution.

Usage:
    python bindeval.py                               # score results/e2e_eval.json
    python bindeval.py --eval results/OTHER.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
INGREDIENT_REF = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def required_action_fields(api_info):
    """Action fields marked Required: true."""
    out = []
    for name, spec in (api_info.get("Action fields") or {}).items():
        if not isinstance(spec, dict):
            continue
        req = str(spec.get("Required", "")).strip().lower()
        if req == "true":
            out.append(name)
    return out


def all_action_fields(api_info):
    return list((api_info.get("Action fields") or {}).keys())


def trigger_ingredient_slugs(api_info):
    """Slugs the selected trigger actually emits."""
    slugs = set()
    for name, spec in (api_info.get("Ingredients") or {}).items():
        slugs.add(name.strip().lower())
        if isinstance(spec, dict) and spec.get("Slug"):
            slugs.add(str(spec["Slug"]).strip().lower())
    return slugs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/e2e_eval.json")
    ap.add_argument("--out", default="results/BINDING_eval.json")
    args = ap.parse_args()

    data = load(ROOT / args.eval)
    ps = data.get("per_sample") or []
    print(f"scoring {len(ps)} samples from {args.eval}\n")

    n_scored = 0
    cov_num = cov_den = 0          # required-field coverage
    ref_total = ref_grounded = 0   # ingredient grounding
    filled_ingredient = filled_static = 0
    per_sample = []
    no_required = 0
    hallucinated_examples = []
    source_types = Counter()

    for r in ps:
        rp = r.get("raw_prediction") or {}
        fa = rp.get("final_applet") or {}
        act = fa.get("action") or {}
        trg = fa.get("trigger") or {}
        field_values = act.get("field_values") or {}
        bindings = fa.get("bindings") or []
        if not fa:
            continue
        n_scored += 1

        a_api = act.get("api_info") or {}
        t_api = trg.get("api_info") or {}
        req = required_action_fields(a_api)
        ing_slugs = trigger_ingredient_slugs(t_api)

        # 1) required-field coverage
        if req:
            filled = [f for f in req if str(field_values.get(f, "")).strip()]
            cov_num += len(filled)
            cov_den += len(req)
            s_cov = len(filled) / len(req)
        else:
            no_required += 1
            s_cov = None

        # 2) ingredient grounding + 3) source mix
        s_refs = s_grounded = 0
        for fname, val in field_values.items():
            sval = str(val).strip()
            if not sval:
                continue
            refs = INGREDIENT_REF.findall(sval)
            if refs:
                filled_ingredient += 1
                for ref in refs:
                    s_refs += 1
                    ref_total += 1
                    if ref.strip().lower() in ing_slugs:
                        s_grounded += 1
                        ref_grounded += 1
                    elif len(hallucinated_examples) < 12:
                        hallucinated_examples.append(
                            {"query": (r.get("query") or "")[:60], "field": fname,
                             "ref": ref, "trigger": trg.get("service_name", ""),
                             "available": sorted(ing_slugs)[:6]})
            else:
                filled_static += 1

        for b in bindings:
            if isinstance(b, dict):
                source_types[str(b.get("source_type", "unknown"))] += 1

        per_sample.append({
            "query": (r.get("query") or "")[:70],
            "required_coverage": s_cov,
            "ingredient_refs": s_refs,
            "grounded_refs": s_grounded,
        })

    def pct(a, b):
        return (a / b) if b else float("nan")

    print("=" * 72)
    print("FIELD-LEVEL BINDING EVALUATION")
    print("=" * 72)
    print(f"samples scored                       {n_scored}")
    print(f"samples whose action has no Required fields   {no_required}")
    print()
    print("1) REQUIRED-FIELD COVERAGE  (executability floor)")
    print(f"     required fields filled           {cov_num}/{cov_den}  = {pct(cov_num, cov_den):.3f}")
    per_s = [p['required_coverage'] for p in per_sample if p['required_coverage'] is not None]
    if per_s:
        full = sum(1 for v in per_s if v >= 1.0)
        print(f"     applets with ALL required filled {full}/{len(per_s)} = {full/len(per_s):.3f}")
    print()
    print("2) INGREDIENT GROUNDING  (are {{refs}} real ingredients of the chosen trigger?)")
    print(f"     ingredient references            {ref_total}")
    print(f"     grounded in the trigger schema   {ref_grounded}  = {pct(ref_grounded, ref_total):.3f}")
    print(f"     HALLUCINATED references          {ref_total - ref_grounded}  = {pct(ref_total - ref_grounded, ref_total):.3f}")
    print()
    print("3) BINDING SOURCE MIX  (is this resolution, or form-filling?)")
    tot = filled_ingredient + filled_static
    print(f"     fields wired to an ingredient    {filled_ingredient}/{tot} = {pct(filled_ingredient, tot):.3f}")
    print(f"     fields filled with a static value{filled_static:>4}/{tot} = {pct(filled_static, tot):.3f}")
    if source_types:
        print(f"     declared source_type counts      {dict(source_types)}")
    print("=" * 72)

    if hallucinated_examples:
        print("\nEXAMPLES OF HALLUCINATED INGREDIENT REFERENCES")
        for h in hallucinated_examples[:8]:
            print(f"  {h['query']!r}")
            print(f"     field={h['field']!r} -> {{{{{h['ref']}}}}}   trigger={h['trigger']!r}")
            print(f"     trigger actually emits: {h['available']}")

    out = {
        "n_scored": n_scored,
        "required_field_coverage": pct(cov_num, cov_den),
        "required_fields_filled": cov_num,
        "required_fields_total": cov_den,
        "ingredient_refs": ref_total,
        "ingredient_grounding": pct(ref_grounded, ref_total),
        "hallucinated_refs": ref_total - ref_grounded,
        "fields_from_ingredient": filled_ingredient,
        "fields_from_static": filled_static,
        "ingredient_share": pct(filled_ingredient, tot),
        "source_type_counts": dict(source_types),
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "per_sample": per_sample}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
