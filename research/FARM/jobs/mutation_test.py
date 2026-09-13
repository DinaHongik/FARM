"""Mutation testing for the FARM arm metrics.

A metric that survives a mutation which should destroy it is vacuous: it cannot
distinguish a correct system from a broken one, so any number it reports is
uninformative. For each metric we apply a mutation whose correct response is
known, and flag every metric that fails to move.

No LLM calls: bindings are synthesised, scoring is the real score() from the harness.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms_fixed as A

random.seed(0)
ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]

def perfect_bindings(t, a, q):
    """An oracle binder: fills every required field with a legitimate value."""
    ing = list(A.ings(t).keys()) if isinstance(A.ings(t), dict) else []
    qw = list(A.toks(q)) or ["x"]
    tb = [{"field": k, "source": "ingredient" if ing else "static",
           "value": (ing[0] if ing else qw[0])} for k in A.required(A.tfields(t))]
    ab = [{"field": k, "source": "ingredient" if ing else "static",
           "value": (ing[0] if ing else qw[0])} for k in A.required(A.afields(a))]
    return {"trigger_bindings": tb, "action_bindings": ab}

MUTATIONS = {
    "M0_baseline":        lambda b, t, a, q: b,
    "M1_drop_all":        lambda b, t, a, q: {"trigger_bindings": [], "action_bindings": []},
    "M2_empty_values":    lambda b, t, a, q: mutval(b, lambda v: ""),
    "M3_null_values":     lambda b, t, a, q: mutval(b, lambda v: None),
    "M4_false_values":    lambda b, t, a, q: mutval(b, lambda v: False),
    "M5_garbage_values":  lambda b, t, a, q: mutval(b, lambda v: "zzqqxx_not_a_real_value_9137"),
    "M6_scramble_fields": lambda b, t, a, q: mutfield(b, lambda k: k[::-1] + "_x"),
    "M7_flip_source":     lambda b, t, a, q: mutsrc(b),
    "M8_drop_trigger_side": lambda b, t, a, q: {"trigger_bindings": [], "action_bindings": b["action_bindings"]},
    "M9_drop_action_side":  lambda b, t, a, q: {"trigger_bindings": b["trigger_bindings"], "action_bindings": []},
}

def mutval(b, f):
    return {k: [{**x, "value": f(x.get("value"))} for x in v] for k, v in b.items()}
def mutfield(b, f):
    return {k: [{**x, "field": f(str(x.get("field", "")))} for x in v] for k, v in b.items()}
def mutsrc(b):
    return {k: [{**x, "source": ("static" if x.get("source") == "ingredient" else "ingredient")} for x in v]
            for k, v in b.items()}

# Expected response: which metrics MUST drop for each mutation to be meaningful.
EXPECT = {
    "M1_drop_all":        ["tcov", "acov", "both"],
    "M2_empty_values":    ["tcov", "acov", "both"],
    "M3_null_values":     ["tcov", "acov", "both", "bothv"],
    "M4_false_values":    ["tcov", "acov", "both", "bothv"],
    "M5_garbage_values":  ["grounded_rate", "stat_ok_rate", "bothv"],
    "M6_scramble_fields": ["tcov", "acov", "both"],
    "M7_flip_source":     ["grounded_rate", "stat_ok_rate", "bothv"],
    "M8_drop_trigger_side": ["tcov", "both"],
    "M9_drop_action_side":  ["acov", "both"],
}

def agg(rows):
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals)/len(vals) if vals else None
    refs = sum(r["refs"] for r in rows); gr = sum(r["grounded"] for r in rows)
    st = sum(r["stats"] for r in rows); so = sum(r["stat_ok"] for r in rows)
    return {"tcov": mean("tcov"), "acov": mean("acov"), "both": mean("both"),
            "bothv": mean("bothv"),
            "sel": mean("sel"), "correct": mean("correct"),
            "grounded_rate": (gr/refs if refs else None),
            "stat_ok_rate": (so/st if st else None)}

results = {}
for name, mut in MUTATIONS.items():
    rows = []
    for s in ps:
        rp = s.get("raw_prediction") or {}
        tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
        if not tcs or not acs: continue
        ref = s.get("reference") or {}; q = s.get("query", "")
        gt, ga = A.nz(ref.get("trigger")), A.nz(ref.get("action"))
        t, a = tcs[0], acs[0]
        b = mut(perfect_bindings(t, a, q), t, a, q)
        r = A.score(t, a, b, gt, ga, q)
        rows.append({k: (float(v) if isinstance(v, bool) else v) for k, v in r.items()})
    results[name] = agg(rows)

base = results["M0_baseline"]
print(f"{'mutation':<24}{'tcov':>8}{'acov':>8}{'both':>8}{'bothv':>8}{'grnd':>8}{'stat':>8}   verdict")
print("-"*88)
def fmt(v): return "  n/a  " if v is None else f"{v:>7.3f}"
print(f"{'M0_baseline':<24}{fmt(base['tcov'])}{fmt(base['acov'])}{fmt(base['both'])}"
      f"{fmt(base['bothv'])}{fmt(base['grounded_rate'])}{fmt(base['stat_ok_rate'])}   (reference)")

survivors = []
for name in MUTATIONS:
    if name == "M0_baseline": continue
    r = results[name]
    failed = []
    for m in EXPECT[name]:
        b0, b1 = base.get(m), r.get(m)
        if b0 is None or b0 == 0:      # nothing to lose -> mutation is uninformative here
            continue
        if b1 is not None and b1 >= b0 - 1e-9:
            failed.append(m)
    verdict = "VACUOUS: " + ",".join(failed) if failed else "ok"
    if failed: survivors.append((name, failed))
    print(f"{name:<24}{fmt(r['tcov'])}{fmt(r['acov'])}{fmt(r['both'])}"
          f"{fmt(r['bothv'])}{fmt(r['grounded_rate'])}{fmt(r['stat_ok_rate'])}   {verdict}")

print(f"\n{len(survivors)} mutation(s) SURVIVED - each is a metric that cannot detect a broken system:")
for n, f in survivors:
    print(f"  {n}: did not move {f}")
if not survivors:
    print("  none - every metric moved when it should have")
