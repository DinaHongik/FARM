"""Build ONE query-grouped three-way split, and derive everything from it.

Why this exists: data/split_data.py:69-79 shuffles applet INSTANCES and slices
90/10 with no grouping. The loader emits an applet once per participating
service, so cross-service applets appear multiple times and twins land on both
sides. Measured consequence: 39.0% of eval queries appear verbatim in
train_applets.json, 88.3% in train_applets_dedup.json.

Design, per the split audit:
  - group key   = normalised query text (3,063 query texts are reused by more
                  than one applet, so grouping by (query,trigger,action) still
                  lets a query straddle the boundary)
  - function id = url, never title (272 title strings map to >1 function)
  - three disjoint parts, whole groups, sorted keys then one seeded shuffle
"""
import json, random, re, sys, os

SEED = 42
SRC = "data/iftttt_dataset_full_trigger_action.json"
OUT = "data/clean_split"

def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", str(q or "").lower()).strip()

def load_records():
    """Match data/split_data.py:16-60 exactly: services -> applets -> components,
    with label 'If' = trigger and 'Then' = action. De-duplicate on applet identity
    so an applet emitted once per participating service counts once."""
    services = json.load(open(SRC))
    recs, seen = [], set()
    for service in services:
        if not isinstance(service, dict): continue
        for applet in service.get("applets", []) or []:
            if not isinstance(applet, dict): continue
            comps = applet.get("components")
            if not isinstance(comps, list): continue
            t = a = None
            for c in comps:
                if not isinstance(c, dict): continue
                if c.get("label") == "If": t = c
                elif c.get("label") == "Then": a = c
            if not (t and a and applet.get("description")): continue
            tu, au = t.get("url"), a.get("url")
            if not tu or not au: continue
            nq = norm_q(applet["description"])
            if not nq: continue
            key = (applet.get("applet_url") or "", nq, tu, au)
            if key in seen: continue
            seen.add(key)
            recs.append({"query": applet["description"], "nq": nq,
                         "trigger": t, "action": a,
                         "trigger_url": tu, "action_url": au,
                         "applet_url": applet.get("applet_url", "")})
    return recs

recs = load_records()
print(f"distinct applet records after de-duplication : {len(recs)}")

groups = {}
for r in recs: groups.setdefault(r["nq"], []).append(r)
keys = sorted(groups)
print(f"distinct normalised-query groups             : {len(keys)}")
multi = sum(1 for k in keys if len(groups[k]) > 1)
print(f"groups holding more than one applet          : {multi}")

random.seed(SEED)
random.shuffle(keys)
n = len(keys)
n_eval = int(n * 0.10); n_rr = int(n * 0.10)
part = {"eval": keys[:n_eval],
        "rerank_train": keys[n_eval:n_eval + n_rr],
        "stage1_train": keys[n_eval + n_rr:]}

os.makedirs(OUT, exist_ok=True)
summary = {}
for name, ks in part.items():
    if name == "stage1_train":
        rows = [r for k in ks for r in groups[k]]           # all applets, for encoder training
    else:
        rows = [sorted(groups[k], key=lambda r: r["applet_url"] or "")[0] for k in ks]  # one per group
    json.dump(rows, open(f"{OUT}/{name}.json", "w"))
    summary[name] = {"groups": len(ks), "records": len(rows)}
    print(f"  {name:<14} groups={len(ks):>6}  records={len(rows):>7}")

# ---- leakage verification, the whole point of the exercise ----
print("\nleakage check (must all be 0):")
sets = {name: {r["nq"] for r in json.load(open(f"{OUT}/{name}.json"))} for name in part}
ok = True
for a in part:
    for b in part:
        if a >= b: continue
        inter = sets[a] & sets[b]
        print(f"  {a} INTERSECT {b} : {len(inter)}")
        if inter: ok = False

# function-inventory coverage is EXPECTED, not leakage; report it so it is not mistaken for one
tr_fns = {r["trigger_url"] for r in json.load(open(f"{OUT}/stage1_train.json"))}
ev = json.load(open(f"{OUT}/eval.json"))
cov = sum(1 for r in ev if r["trigger_url"] in tr_fns) / max(1, len(ev))
print(f"\neval trigger functions also in stage1_train inventory: {cov:.3f}  (expected high, NOT leakage)")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
nonempty = all(summary[k]["records"] > 0 for k in summary)
if not nonempty:
    print("\nVERDICT: INVALID - a split is empty, the leakage check passes vacuously")
    sys.exit(1)
print("\nVERDICT:", "CLEAN" if ok else "STILL LEAKING")
