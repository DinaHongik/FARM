"""Audit 2: stale bindings, ingredient availability, BASE at full n, D5 choice_ok."""
import json, importlib.util, collections
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
NEW = load("nf", "jobs/arms_fixed.py")
ps = json.load(open("results/e2e_eval.json"))["per_sample"]

# --- ingredients availability on the committed trigger
ing_ok = 0; ing_empty = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    t = (rp.get("final_applet") or {}).get("trigger") or {}
    ii = NEW.ings(t)
    if isinstance(ii, dict) and ii: ing_ok += 1
    else: ing_empty += 1
print(f"committed trigger has non-empty api_info.Ingredients: {ing_ok}, empty/missing: {ing_empty}")

# --- retries / abandoned pairs: can binding_map hold stale entries?
rc = collections.Counter(); ap = collections.Counter()
stale_samples = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    rc[rp.get("retry_count")] += 1
    ap[len(rp.get("attempted_pairs") or [])] += 1
    a = (rp.get("final_applet") or {}).get("action") or {}
    allf = {str(k).strip().lower() for k in NEW.afields(a)}
    bad = [b.get("action_field") for b in rp.get("binding_map") or []
           if str(b.get("action_field","")).strip().lower() not in allf]
    if bad: stale_samples += 1
print("retry_count histogram:", dict(rc))
print("len(attempted_pairs) histogram:", dict(ap))
print(f"samples with >=1 binding_map field not in the committed action's fields: {stale_samples}/299")

# --- does binding_map cover exactly the current_requirements?
mism = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    cr = rp.get("current_requirements")
    if isinstance(cr, dict):
        keys = sorted(cr.keys())
    else:
        keys = None
    if mism == 0 and keys is not None:
        print("current_requirements keys (sample):", keys)
        print("  value shape:", json.dumps(cr, ensure_ascii=False)[:300])
        mism = 1

# --- BASE over the full stored set, through the real code path
rows = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    ref = s.get("reference") or {}; pr = s.get("prediction") or {}
    q = s.get("query",""); gt, ga = NEW.nz(ref.get("trigger")), NEW.nz(ref.get("action"))
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    bt, ba = NEW.base_pair(rp, tcs, acs)
    base = NEW.score(bt, ba, NEW.base_binds(rp), gt, ga, q)
    base["sel"] = (NEW.nz(pr.get("trigger"))==gt and NEW.nz(pr.get("action"))==ga)
    base["correct"] = 1.0 if (base["sel"] and base["both"]==1.0) else 0.0
    if base["t_req"]:
        base["tcov"]=base["both"]=base["correct"]=None
    rows.append(base)
def m(k):
    v=[r[k] for r in rows if r[k] is not None]
    return (sum(v)/len(v), len(v)) if v else (None,0)
refs=sum(r["refs"] for r in rows); gr=sum(r["grounded"] for r in rows)
st=sum(r["stats"] for r in rows); so=sum(r["stat_ok"] for r in rows)
print(f"\nBASE at n={len(rows)}:")
for k in ("sel","both","correct","tcov","acov"):
    val,nn = m(k) if k!="sel" else ((sum(1 for r in rows if r['sel'])/len(rows)), len(rows))
    print(f"  {k}: {val if val is None else round(val,4)}  (n={nn})")
print(f"  ing_grnd = {gr}/{refs} = {gr/refs if refs else None}")
print(f"  stat_grnd = {so}/{st} = {so/st if st else None}")
print(f"  a_req==0 samples: {sum(1 for r in rows if r['a_req']==0)}")

# --- D5: does choice_ok really mean 'both indices present and int-coercible'?
class FakeDict(dict): pass
for name, j in [("missing both keys", {"trigger_bindings":[], "action_bindings":[]}),
                ("only trigger_choice", {"trigger_choice":3}),
                ("explicit nulls", {"trigger_choice":None,"action_choice":None}),
                ("string ints", {"trigger_choice":"2","action_choice":"1"}),
                ("floats", {"trigger_choice":2.9,"action_choice":0.4})]:
    ti=ai=0; ok=False
    try:
        ti = max(0,min(4,int(j.get("trigger_choice",0))))
        ai = max(0,min(4,int(j.get("action_choice",0))))
        ok = True
    except (TypeError,ValueError,AttributeError):
        pass
    print(f"D5 {name:<22} -> ti={ti} ai={ai} choice_ok={ok} choice_raw={[j.get('trigger_choice'),j.get('action_choice')]}")
