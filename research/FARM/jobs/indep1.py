"""Independent structural audit of results/e2e_eval.json for the D2 claims."""
import json, collections
d = json.load(open("results/e2e_eval.json"))
ps = d["per_sample"]
print("top-level keys:", sorted(d.keys()))
print("n per_sample:", len(ps))
print("sample keys:", sorted(ps[0].keys()))
rp0 = ps[0]["raw_prediction"]
print("raw_prediction keys:", sorted(rp0.keys()))
fa0 = rp0.get("final_applet")
print("final_applet keys:", sorted(fa0.keys()) if isinstance(fa0, dict) else type(fa0))
print("final_applet.trigger keys:", sorted(fa0["trigger"].keys()))
print("final_applet.action  keys:", sorted(fa0["action"].keys()))
print("bm[0]:", json.dumps(rp0["binding_map"][0], ensure_ascii=False)[:400])

# (a) binding_map key vocabulary
keys = collections.Counter(); srcs = collections.Counter(); tot = 0
withrp = 0
trigger_side_hits = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    withrp += 1
    for b in rp.get("binding_map") or []:
        tot += 1
        keys.update(b.keys()); srcs[b.get("source_type")] += 1
        for k in b:
            if "trigger" in k.lower(): trigger_side_hits.append(k)
print("\nsamples with raw_prediction:", withrp)
print("binding_map entries total:", tot)
print("binding_map key freq:", dict(keys))
print("source_type freq:", dict(srcs))
print("keys mentioning 'trigger':", collections.Counter(trigger_side_hits))

# (c) does final_applet.trigger carry api_info / field_values?
has_api_t = has_api_a = has_fv_t = has_fv_a = 0
nonempty_fv_t = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    fa = rp.get("final_applet") or {}
    t, a = fa.get("trigger") or {}, fa.get("action") or {}
    has_api_t += isinstance(t.get("api_info"), dict)
    has_api_a += isinstance(a.get("api_info"), dict)
    has_fv_t += "field_values" in t
    has_fv_a += "field_values" in a
    if t.get("field_values"): nonempty_fv_t += 1
print(f"\nfinal_applet.trigger has api_info: {has_api_t}/{withrp}   action has api_info: {has_api_a}/{withrp}")
print(f"trigger has field_values key: {has_fv_t}   action has field_values key: {has_fv_a}   trigger field_values non-empty: {nonempty_fv_t}")

# (d) committed pair vs current_*_idx  vs prediction
agree_idx = agree_pred = 0; n = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    n += 1
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    fa = rp.get("final_applet") or {}
    ti, ai = rp.get("current_trigger_idx"), rp.get("current_action_idx")
    ok = (isinstance(ti,int) and isinstance(ai,int) and 0<=ti<len(tcs) and 0<=ai<len(acs)
          and tcs[ti].get("service_name")==fa.get("trigger",{}).get("service_name")
          and acs[ai].get("service_name")==fa.get("action",{}).get("service_name"))
    agree_idx += ok
    pr = s.get("prediction") or {}
    agree_pred += (str(pr.get("trigger","")).strip().lower()
                   == str(fa.get("trigger",{}).get("service_name","")).strip().lower())
print(f"\ncommitted pair agrees with current_*_idx: {agree_idx}/{n}")
print(f"final_applet.trigger.service_name == prediction.trigger: {agree_pred}/{n}")

# (e) do binding_map action_field names match the committed action's required Action fields?
def afields(c): return ((c.get("api_info") or {}).get("Action fields")) or {}
def req(f): return {k:v for k,v in f.items() if isinstance(v,dict) and str(v.get("Required","")).lower()=="true"}
matched = unmatched = 0
unmatched_ex = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    fa = rp.get("final_applet") or {}
    a = fa.get("action") or {}
    allf = {str(k).strip().lower() for k in afields(a)}
    for b in rp.get("binding_map") or []:
        f = str(b.get("action_field","")).strip().lower()
        if f in allf: matched += 1
        else:
            unmatched += 1
            if len(unmatched_ex) < 5: unmatched_ex.append((b.get("action_field"), sorted(allf)[:6]))
print(f"\nbinding_map action_field in committed action's Action fields: {matched} matched / {unmatched} unmatched")
for e in unmatched_ex: print("   unmatched:", e)
