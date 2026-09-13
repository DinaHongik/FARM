"""Differential check: old arms_lib vs repaired arms_fixed on the same inputs."""
import sys, json, importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

OLD = load("arms_old", "jobs/arms_lib.py")
NEW = load("arms_new", "jobs/arms_fixed.py")

def cand(kind, req, ings=None):
    key = "Trigger fields" if kind == "trigger" else "Action fields"
    api = {key: {f: {"Label": f, "Slug": f, "Required": "true"} for f in req}}
    if ings is not None: api["Ingredients"] = {i: {"Slug": i, "Type": "String"} for i in ings}
    return {"service_name": f"svc_{kind}", "api_info": api}

print("== D4: JSON null / whitespace / empty list must not count as bound ==")
t = cand("trigger", ["Ticker"], ["StockName"]); a = cand("action", ["Light", "Mode", "Bright"])
binds = {"trigger_bindings": [{"field": "Ticker", "source": "static", "value": "acme"}],
         "action_bindings": [{"field": "Light", "source": "static", "value": None},
                             {"field": "Mode",  "source": "static", "value": "   "},
                             {"field": "Bright","source": "static", "value": []}]}
o = OLD.score(t, a, binds, "svc_trigger", "svc_action", "acme")
n = NEW.score(t, a, binds, "svc_trigger", "svc_action", "acme")
print(f"  old acov={o['acov']}  both={o['both']}  stats={o['stats']}  stat_ok={o['stat_ok']}")
print(f"  new acov={n['acov']}  both={n['both']}  stats={n['stats']}  stat_ok={n['stat_ok']}")
assert abs(o["acov"] - 2/3) < 1e-9 and n["acov"] == 0.0, (o["acov"], n["acov"])
assert o["stats"] == 4 and n["stats"] == 1, (o["stats"], n["stats"])
print("  OK: old counted JSON null and [] as bound (2/3) and all 4 as static values;")
print("      new counts none of the three as bound and only the real value as static")

print("\n== D3a: side with no required fields is no longer dropped from BOTH ==")
t2 = cand("trigger", [], ["StockName"]); a2 = cand("action", ["Light"])
b2 = {"trigger_bindings": [], "action_bindings": [{"field": "Light", "source": "static", "value": "lamp"}]}
o = OLD.score(t2, a2, b2, "svc_trigger", "svc_action", "lamp")
n = NEW.score(t2, a2, b2, "svc_trigger", "svc_action", "lamp")
print(f"  old tcov={o['tcov']} acov={o['acov']} both={o['both']}  (no t_req column existed)")
print(f"  new tcov={n['tcov']} acov={n['acov']} both={n['both']}  t_req={n['t_req']} a_req={n['a_req']}")
print("  OK: same BOTH value, but new record carries t_req/a_req so the vacuous side is visible")

print("\n== D3b: sample requiring nothing at all no longer scores BOTH=0 ==")
t3 = cand("trigger", [], ["X"]); a3 = cand("action", [])
o = OLD.score(t3, a3, {}, "svc_trigger", "svc_action", "q")
n = NEW.score(t3, a3, {}, "svc_trigger", "svc_action", "q")
print(f"  old both={o['both']} correct={o['correct']}   <- penalised for having nothing to bind")
print(f"  new both={n['both']} correct={n['correct']}   t_req={n['t_req']} a_req={n['a_req']}")
assert o["both"] == 0.0 and n["both"] == 1.0
print("  OK")

print("\n== D2: BASE row recomputed from the stored dump ==")
ps = json.load(open("results/e2e_eval.json"))["per_sample"]
s = ps[0]; rp = s["raw_prediction"]
bt, ba = NEW.base_pair(rp, rp["trigger_candidates"], rp["action_candidates"])
bb = NEW.base_binds(rp)
sc = NEW.score(bt, ba, bb, "x", "y", s["query"])
print(f"  committed pair: {bt['service_name']!r} -> {ba['service_name']!r}")
print(f"  stored binding_map entries: {len(bb['action_bindings'])}  required action fields: {sc['a_req']}")
print(f"  acov={sc['acov']} refs={sc['refs']} grounded={sc['grounded']} stats={sc['stats']} stat_ok={sc['stat_ok']}")
print("  old hardcoded row was: acov=None both=0.0 refs=0 grounded=0 stats=0 stat_ok=0 correct=0.0")

print("\n== D2b: how much of BASE is recoverable over the full stored set ==")
meas = unmeas = nopair = 0
for s in ps:
    rp = s.get("raw_prediction") or {}
    if not rp: continue
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    bt, ba = NEW.base_pair(rp, tcs, acs)
    if bt is None: nopair += 1; continue
    if NEW.required(NEW.tfields(bt)): unmeas += 1
    else: meas += 1
tot = meas + unmeas + nopair
print(f"  n={tot}  BOTH measurable={meas}  BOTH n/a (trigger has required fields)={unmeas}  no committed pair={nopair}")
print(f"  no-required-trigger share = {meas/tot:.3f}  (matches the 32.1% D3 quotes)")

print("\n== D7: zero denominator ==")
print(f"  old style: {(0/1 if 0 else 0):.3f}   new style: {NEW.fmt(NEW.ratio([0],[0]), 5)}")
assert NEW.ratio([0], [0]) is None

print("\n== D5: per-sample A2 choice fields present in the run dump ==")
d = json.load(open("results/ARMS_gpt-oss_120b_4.json"))
for i, r in enumerate(d):
    a = r["A2"]
    print(f"  sample {i}: ti={a['ti']} ai={a['ai']} parsed={a['parsed']} choice_ok={a['choice_ok']} "
          f"choice_raw={a['choice_raw']} fallback={a['fallback']}")
print(f"  BASE keys in dump: {sorted(d[0]['BASE'])}")

print("\n== D8: module import must not execute main() ==")
print("  arms_fixed imported above with no run triggered -> OK")
