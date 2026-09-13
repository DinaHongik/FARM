#!/usr/bin/env python3
"""Binding architecture search - measure before building.

bindeval.py established WHAT is wrong: only 44.9% of filled action fields are wired
to a trigger ingredient; 55.1% are invented static literals. The paper's title
claims field-aware RESOLUTION, so that ratio is the contribution gap.

This script answers the three questions that decide what to build:

  E1 HEADROOM      Of the fields filled with a static literal, how many had a real
                   ingredient available that a matcher could have found? If the
                   ceiling is low, most static fills are legitimate user-config
                   (e.g. "Which light(s)?" is not derivable from a stock trigger)
                   and there is nothing to win. If it is high, there is a real gap.

  E2 MATCHER       Using the bindings where the LLM DID choose an ingredient as
                   ground truth, which automatic matcher recovers that choice?
                   Candidates: exact slug, substring, token-overlap, embedding
                   cosine, and type-filtered variants. This picks the algorithm.

  E3 TYPE SAFETY   Ingredients declare Type ("String", "Number", ...). Action fields
                   declare a signature in "Filter code method", e.g.
                   "Wiz.turnOn.setDimming(string: dimming)". Nothing checks them.
                   How often is an existing binding type-incompatible?

No LLM calls. Embeddings optional (--embed) and use the BASE sentence encoder,
matching what cross_scorer_node._get_semantic_model actually loads.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
sys.path.insert(0, str(ROOT))

INGREDIENT_REF = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
SIG_TYPE = re.compile(r"\(\s*([a-zA-Z_]+)\s*:", re.I)
TOKEN = re.compile(r"[a-z0-9]+")


def norm(s):
    return " ".join(TOKEN.findall(str(s).lower()))


def toks(s):
    return set(TOKEN.findall(str(s).lower()))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# schema helpers
# --------------------------------------------------------------------------
def ingredients_of(trigger_api):
    """[(display_name, slug, type, example)] for the trigger's Ingredients."""
    out = []
    for name, spec in (trigger_api.get("Ingredients") or {}).items():
        if not isinstance(spec, dict):
            spec = {}
        out.append((name, str(spec.get("Slug", name)), str(spec.get("Type", "")),
                    str(spec.get("Example", ""))))
    return out


def field_specs_of(action_api):
    """{field_name: spec} for the action's Action fields."""
    return {k: (v if isinstance(v, dict) else {})
            for k, v in (action_api.get("Action fields") or {}).items()}


def declared_type(spec):
    """Type declared in the field's Filter code method signature, if any."""
    m = SIG_TYPE.search(str(spec.get("Filter code method", "")))
    return m.group(1).lower() if m else None


# --------------------------------------------------------------------------
# matchers: each scores (field_name, field_spec) against one ingredient
# --------------------------------------------------------------------------
def m_exact(fname, fspec, ing):
    name, slug, _t, _e = ing
    targets = {norm(fname), norm(fspec.get("Slug", ""))}
    return 1.0 if (norm(name) in targets or norm(slug) in targets) else 0.0


def m_substring(fname, fspec, ing):
    name, slug, _t, _e = ing
    hay = f"{norm(fname)} {norm(fspec.get('Slug',''))}"
    for cand in (norm(name), norm(slug)):
        if cand and (cand in hay or hay in cand):
            return 1.0
    return 0.0


def m_token(fname, fspec, ing):
    """Jaccard over word tokens - cheap, order-free, no model needed."""
    name, slug, _t, _e = ing
    a = toks(fname) | toks(fspec.get("Slug", ""))
    b = toks(name) | toks(slug)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def type_compatible(fspec, ing):
    """False only when both sides declare a type and they conflict."""
    dt = declared_type(fspec)
    it = (ing[2] or "").strip().lower()
    if not dt or not it:
        return True
    numeric = {"number", "int", "integer", "float", "double"}
    if dt in numeric:
        return it in numeric
    return True  # 'string' accepts anything printable


class EmbedMatcher:
    def __init__(self):
        from rag.embeddings import EmbeddingModel
        self.model = EmbeddingModel().base_model
        self.cache = {}

    def vec(self, text):
        key = norm(text)
        if key not in self.cache:
            self.cache[key] = self.model.encode(key, convert_to_tensor=True,
                                                show_progress_bar=False)
        return self.cache[key]

    def __call__(self, fname, fspec, ing):
        import torch
        a = self.vec(f"{fname} {fspec.get('Slug','')}")
        b = self.vec(f"{ing[0]} {ing[1]}")
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0])


def best_ingredient(matcher, fname, fspec, ings, type_filter=False):
    """Rank ingredients for one field. Deterministic: sorted, stable tie-break."""
    scored = []
    for ing in sorted(ings, key=lambda x: (x[0], x[1])):
        if type_filter and not type_compatible(fspec, ing):
            continue
        scored.append((matcher(fname, fspec, ing), ing))
    if not scored:
        return 0.0, None
    scored.sort(key=lambda x: (-x[0], x[1][0]))
    return scored[0]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/e2e_eval.json")
    ap.add_argument("--embed", action="store_true", help="also evaluate the embedding matcher (GPU)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/BINDLAB.json")
    args = ap.parse_args()

    ps = load(ROOT / args.eval)["per_sample"]

    matchers = {"exact": m_exact, "substring": m_substring, "token": m_token}
    if args.embed:
        print("loading base sentence encoder ...", flush=True)
        matchers["embed"] = EmbedMatcher()

    # collected cases
    gt_cases = []      # (fname, fspec, ings, gold_ingredient_key)  - LLM wired an ingredient
    static_cases = []  # (fname, fspec, ings, static_value)         - LLM invented a literal
    type_stats = Counter()
    required_static = 0

    for r in ps:
        fa = (r.get("raw_prediction") or {}).get("final_applet") or {}
        act, trg = fa.get("action") or {}, fa.get("trigger") or {}
        fvals = act.get("field_values") or {}
        if not fvals:
            continue
        ings = ingredients_of(trg.get("api_info") or {})
        specs = field_specs_of(act.get("api_info") or {})
        if not ings:
            continue

        ing_keys = {norm(n) for n, s, _, _ in ings} | {norm(s) for n, s, _, _ in ings}

        for fname, val in fvals.items():
            sval = str(val).strip()
            if not sval:
                continue
            fspec = specs.get(fname, {})
            refs = INGREDIENT_REF.findall(sval)
            if refs:
                ref = refs[0].strip()
                if norm(ref) in ing_keys:            # grounded -> usable as ground truth
                    gt_cases.append((fname, fspec, ings, norm(ref)))
                    # E3 type safety
                    match = [i for i in ings if norm(i[0]) == norm(ref) or norm(i[1]) == norm(ref)]
                    if match:
                        dt, it = declared_type(fspec), (match[0][2] or "").lower()
                        if dt and it:
                            type_stats["checked"] += 1
                            type_stats["compatible" if type_compatible(fspec, match[0]) else "INCOMPATIBLE"] += 1
                        else:
                            type_stats["undeclared"] += 1
            else:
                static_cases.append((fname, fspec, ings, sval))
                if str(fspec.get("Required", "")).lower() == "true":
                    required_static += 1

    print("=" * 74)
    print("BINDING ARCHITECTURE SEARCH")
    print("=" * 74)
    print(f"fields wired to an ingredient (ground truth) : {len(gt_cases)}")
    print(f"fields filled with a static literal          : {len(static_cases)}"
          f"  ({required_static} of them REQUIRED)")

    # ---------------- E2: which matcher recovers the LLM's choice ----------
    print("\nE2  MATCHER RECOVERY  (rank-1 agreement with the LLM's ingredient choice)")
    print(f"    {'matcher':<22}{'top-1':>9}{'top-1 +type':>14}")
    print("    " + "-" * 45)
    e2 = {}
    for name, fn in matchers.items():
        hit = hit_t = 0
        for fname, fspec, ings, gold in gt_cases:
            _s, ing = best_ingredient(fn, fname, fspec, ings)
            if ing and (norm(ing[0]) == gold or norm(ing[1]) == gold):
                hit += 1
            _s, ing = best_ingredient(fn, fname, fspec, ings, type_filter=True)
            if ing and (norm(ing[0]) == gold or norm(ing[1]) == gold):
                hit_t += 1
        n = len(gt_cases) or 1
        e2[name] = {"top1": hit / n, "top1_type": hit_t / n}
        print(f"    {name:<22}{hit/n:>9.3f}{hit_t/n:>14.3f}")
    rand = 1.0 / (sum(len(c[2]) for c in gt_cases) / max(len(gt_cases), 1))
    print(f"    {'(random baseline)':<22}{rand:>9.3f}")

    # ---------------- E1: headroom on the static fills ---------------------
    print(f"\nE1  HEADROOM  (static fills that had a plausible ingredient, score >= {args.threshold})")
    print(f"    {'matcher':<22}{'recoverable':>13}{'of static':>11}")
    print("    " + "-" * 46)
    e1 = {}
    examples = defaultdict(list)
    for name, fn in matchers.items():
        rec = 0
        for fname, fspec, ings, sval in static_cases:
            s, ing = best_ingredient(fn, fname, fspec, ings, type_filter=True)
            if ing and s >= args.threshold:
                rec += 1
                if len(examples[name]) < 6:
                    examples[name].append((fname, sval[:28], ing[0], round(s, 3)))
        n = len(static_cases) or 1
        e1[name] = rec / n
        print(f"    {name:<22}{rec:>13}{rec/n:>11.3f}")

    best = max(e1, key=e1.get)
    print(f"\n    examples the '{best}' matcher would have wired instead of inventing:")
    for fname, sval, ing, s in examples[best]:
        print(f"      field={fname[:34]!r:36s} static={sval!r:30s} -> {{{{{ing}}}}} ({s})")

    # ---------------- E3: type safety --------------------------------------
    print("\nE3  TYPE SAFETY of existing ingredient bindings")
    ch = type_stats["checked"]
    print(f"    both sides declare a type : {ch}")
    print(f"    compatible                : {type_stats['compatible']}")
    print(f"    INCOMPATIBLE              : {type_stats['INCOMPATIBLE']}")
    print(f"    one/both undeclared       : {type_stats['undeclared']}")

    print("\n" + "=" * 74)
    print("READ THIS AS:")
    print("  E2 high  -> a cheap deterministic matcher reproduces the LLM's binding")
    print("              choices, so binding can be made schema-driven and auditable.")
    print("  E1 high  -> many invented literals had a real ingredient available;")
    print("              that is the measurable win. E1 low -> static fills are")
    print("              legitimate user-config and the 44.9% is near its ceiling.")

    out = {"n_ground_truth": len(gt_cases), "n_static": len(static_cases),
           "n_static_required": required_static, "E2_matcher_recovery": e2,
           "E1_headroom": e1, "E3_type": dict(type_stats)}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
