import json, sys
sys.path.insert(0, "jobs")
import arms_fixed as A

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]
rows = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: continue
    ref = s.get("reference") or {}; pr = s.get("prediction") or {}; q = s.get("query", "")
    gt, ga = A.nz(ref.get("trigger")), A.nz(ref.get("action"))
    bsel = (A.nz(pr.get("trigger")) == gt and A.nz(pr.get("action")) == ga)
    bt, ba = A.base_pair(rp, tcs, acs)
    if bt is None: continue
    b = A.score(bt, ba, A.base_binds(rp), gt, ga, q)
    b["sel"] = bsel
    if b["t_req"]:
        b["tcov"] = b["both"] = b["correct"] = None
        b["tcovv"] = b["bothv"] = b["correctv"] = None
    rows.append(b)

def m(k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return (sum(v)/len(v), len(v)) if v else (None, 0)

print(f"BASE recomputed over {len(rows)} samples")
for k in ("sel", "both", "correct", "bothv", "correctv", "acov", "acovv"):
    val, n = m(k)
    txt = "n/a" if val is None else f"{val:.3f}"
    print(f"  {k:<10} {txt:>7}   (measurable n={n})")
