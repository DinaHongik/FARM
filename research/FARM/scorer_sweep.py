"""Offline sweep of the A3 pair scorer. No LLM calls: selection is deterministic.

The shipped scorer is 0.7*coverage + 0.3*mean(retrieval_score). Measured earlier:
one coverage step is 0.7000 while the ENTIRE retrieval term spans 0.1964, so
coverage lexicographically dominates and the choice is made blind to the query.
This sweeps the weighting after putting both terms on a comparable scale.
"""
import json, sys, itertools
sys.path.insert(0, "jobs")
import arms_fixed as A

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]

samples = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: continue
    ref = s.get("reference") or {}
    gt, ga = A.nz(ref.get("trigger")), A.nz(ref.get("action"))
    grid = []
    for i, t in enumerate(tcs):
        for k, a in enumerate(acs):
            grid.append((i, k, A.coverage(t, a),
                         (t.get("score") or 0.0), (a.get("score") or 0.0),
                         A.nz(t.get("service_name")) == gt and A.nz(a.get("service_name")) == ga))
    samples.append(grid)

n = len(samples)
floor = sum(1 for g in samples if any(hit for i,k,_,_,_,hit in g if i==0 and k==0))/n
ceil_ = sum(1 for g in samples if any(hit for *_, hit in g))/n
print(f"n={n}   rank-1 floor={floor:.3f}   top-5 ceiling={ceil_:.3f}\n")

def norm(vals):
    """min-max within THIS sample's 25 pairs, so the term actually has range."""
    lo, hi = min(vals), max(vals)
    return [(v-lo)/(hi-lo) if hi > lo else 0.5 for v in vals]

def evaluate(wcov, wret, normalise):
    hit = 0
    for g in samples:
        rets = [(t+a)/2 for _,_,_,t,a,_ in g]
        covs = [c for _,_,c,_,_,_ in g]
        if normalise:
            rets = norm(rets); covs = norm(covs)
        best = max(range(len(g)), key=lambda j: wcov*covs[j] + wret*rets[j])
        if g[best][5]: hit += 1
    return hit/n

print(f"{'scorer':<44}{'joint sel':>10}")
print("-"*54)
print(f"{'rank-1 (no scorer)':<44}{floor:>10.3f}")
print(f"{'SHIPPED 0.7*cov + 0.3*ret, raw scales':<44}{evaluate(0.7, 0.3, False):>10.3f}")
print(f"{'retrieval only, raw':<44}{evaluate(0.0, 1.0, False):>10.3f}")
print(f"{'coverage only':<44}{evaluate(1.0, 0.0, False):>10.3f}")
print()
for wc in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    wr = 1.0 - wc
    print(f"{'normalised  cov=' + f'{wc:.1f}' + '  ret=' + f'{wr:.1f}':<44}{evaluate(wc, wr, True):>10.3f}")
