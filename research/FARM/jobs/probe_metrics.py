"""Quantify how much the confirmed metric defects move the reported numbers."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms_lib as A

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]
n = 0; no_areq = 0; no_treq = 0; neither = 0
cov_freepass = 0; cov_total = 0; dom = 0
smin = 9e9; smax = -9e9

for s in ps:
    rp = s.get("raw_prediction") or {}
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: continue
    n += 1
    t0, a0 = tcs[0], acs[0]
    if not A.required(A.afields(a0)): no_areq += 1
    if not A.required(A.tfields(t0)): no_treq += 1
    if not A.required(A.afields(a0)) and not A.required(A.tfields(t0)): neither += 1

    # A3 argmax internals over the real 25 pairs
    covs = []
    for i, t in enumerate(tcs):
        for k, a in enumerate(acs):
            c = A.coverage(t, a)
            r = 0.5*((t.get("score") or 0)+(a.get("score") or 0))
            covs.append((0.7*c + 0.3*r, c, r, i, k))
            cov_total += 1
            if not A.required(A.afields(a)): cov_freepass += 1
            smin = min(smin, r); smax = max(smax, r)
    best = max(covs, key=lambda x: x[0])
    # would the winner change if the retrieval term were deleted entirely?
    best_cov_only = max(covs, key=lambda x: x[1])
    if (best[3], best[4]) != (best_cov_only[3], best_cov_only[4]): dom += 1

print(f"n={n}")
print(f"\n--- L127: is BOTH a conjunction? ---")
print(f"action has NO required fields (acov=None) : {no_areq}  ({no_areq/n:.3f})")
print(f"trigger has NO required fields (tcov=None): {no_treq}  ({no_treq/n:.3f})")
print(f"neither side has required fields          : {neither}  ({neither/n:.3f})")
print(f"=> {no_areq/n:.1%} of BOTH scores are decided by the TRIGGER side alone")

print(f"\n--- L64/L167: A3 argmax internals (over {cov_total} pairs) ---")
print(f"pairs where coverage() returns free 1.0   : {cov_freepass}  ({cov_freepass/cov_total:.3f})")
print(f"retrieval term range 0.3*r                : [{0.3*smin:.4f}, {0.3*smax:.4f}]  span={0.3*(smax-smin):.4f}")
print(f"one coverage step (1 required field)      : 0.7000")
print(f"samples where retrieval term changed winner: {dom}  ({dom/n:.3f})")
