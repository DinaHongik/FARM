import json, importlib.util
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
NEW=load("nf","jobs/arms_fixed.py")
ps=json.load(open("results/e2e_eval.json"))["per_sample"]
def norm(x): return "".join(ch for ch in str(x).lower() if ch.isalnum())
print("=== acov<1.0 samples, per-criterion recoverability ===")
tot=0
for s in ps:
    rp=s.get("raw_prediction") or {}
    if not rp: continue
    a=(rp.get("final_applet") or {}).get("action") or {}
    areq=NEW.required(NEW.afields(a))
    if not areq: continue
    bb=NEW.base_binds(rp)["action_bindings"]
    ab={NEW.nz(b.get("field")):b for b in bb}
    miss=[k for k in areq if NEW.bval(ab.get(NEW.nz(k))) is None]
    if not miss: continue
    tot+=1
    alt={norm(b["field"]) for b in bb}
    slugs={norm(rf.get("name")):norm(rf.get("slug")) for rf in (rp.get("current_requirements") or {}).get("required_fields") or []}
    for k in miss:
        why=[]
        if norm(k) in alt: why.append("normalised-name")
        if slugs.get(norm(k)) and slugs[norm(k)] in alt: why.append(f"slug={slugs.get(norm(k))}")
        print(f"  missed {k!r:55} bound={sorted(alt)}  -> {why or 'GENUINELY UNBOUND'}")
print(f"  total samples with BASE acov<1.0: {tot}")

print("\n=== does a repaired n=300 run overwrite the completed old-code run? ===")
old=json.load(open("results/ARMS_gpt-oss_120b_299.json"))
print("  existing ARMS_gpt-oss_120b_299.json A2 keys:", sorted(old[0]["A2"]))
print("  existing BASE row:", {k:v for k,v in old[0]["BASE"].items()})
print("  output name for a fresh 300-sample run: results/ARMS_gpt-oss_120b_%d.json" % 299)
