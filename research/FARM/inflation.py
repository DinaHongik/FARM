"""Exact-match vs service_matches on the SAME stored predictions.

The arm harness scores selection by exact string equality. The manuscript's
pipeline scores it with service_matches(). Same predictions, two rulers.
"""
import json, sys
sys.path.insert(0, ".")
from eval.ragas_metrics import service_matches, normalize_service_name

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]
def nz(x): return str(x or "").strip().lower()

rows = []
for s in ps:
    pr = s.get("prediction") or {}; ref = s.get("reference") or {}
    pt, pa = pr.get("trigger"), pr.get("action")
    gt, ga = ref.get("trigger"), ref.get("action")
    if not gt or not ga: continue
    rows.append({
        "t_exact": nz(pt) == nz(gt),
        "a_exact": nz(pa) == nz(ga),
        "t_fuzzy": bool(pt) and service_matches(pt, gt),
        "a_fuzzy": bool(pa) and service_matches(pa, ga),
    })

n = len(rows)
def r(k): return sum(1 for x in rows if x[k]) / n
te, ae = r("t_exact"), r("a_exact")
tf, af = r("t_fuzzy"), r("a_fuzzy")
je = sum(1 for x in rows if x["t_exact"] and x["a_exact"]) / n
jf = sum(1 for x in rows if x["t_fuzzy"] and x["a_fuzzy"]) / n

print(f"n={n}\n")
print(f"{'metric':<22}{'exact':>9}{'service_matches':>18}{'inflation':>12}")
print("-"*61)
for lbl, e, f in (("trigger accuracy", te, tf), ("action accuracy", ae, af), ("JOINT accuracy", je, jf)):
    print(f"{lbl:<22}{e:>9.3f}{f:>18.3f}{f-e:>+12.3f}")

cred = sum(1 for x in rows if (x["t_fuzzy"] and not x["t_exact"]) or (x["a_fuzzy"] and not x["a_exact"]))
print(f"\nsamples credited by the fuzzy matcher that exact match rejects: {cred}/{n} = {cred/n:.3f}")
