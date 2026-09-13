import json, importlib.util, collections
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
NEW=load("nf","jobs/arms_fixed.py")
ps=json.load(open("results/e2e_eval.json"))["per_sample"]

print("=== 1. D5: stub-run dump, fallback vs actually-missing-choice ===")
d=json.load(open("results/ARMS_STUB_299.json"))
a2=[r["A2"] for r in d]
noparse=sum(1 for r in a2 if not r["parsed"])
nokeys=sum(1 for r in a2 if r["parsed"] and r["choice_raw"]==[None,None])
flagged=sum(1 for r in a2 if r["fallback"]==1.0)
print(f"  n={len(a2)}  not-parsed={noparse}  parsed-but-no-choice-keys={nokeys}  flagged as fallback={flagged}")
print(f"  TRUE silent rank-1 fallbacks = {noparse+nokeys} ({(noparse+nokeys)/len(a2):.3f}); harness printed {flagged/len(a2):.3f}")
print(f"  choice_ok True while choice_raw==[None,None]: {sum(1 for r in a2 if r['choice_ok'] and r['choice_raw']==[None,None])}")

print("\n=== 2. BASE subset bias: sel on the BOTH-measurable subset vs all ===")
tot=meas=0; sel_all=sel_meas=0
for s in ps:
    rp=s.get("raw_prediction") or {}
    if not rp: continue
    ref=s.get("reference") or {}; pr=s.get("prediction") or {}
    gt,ga=NEW.nz(ref.get("trigger")),NEW.nz(ref.get("action"))
    sel=(NEW.nz(pr.get("trigger"))==gt and NEW.nz(pr.get("action"))==ga)
    bt,ba=NEW.base_pair(rp,rp["trigger_candidates"],rp["action_candidates"])
    treq=NEW.required(NEW.tfields(bt))
    tot+=1; sel_all+=sel
    if not treq: meas+=1; sel_meas+=sel
print(f"  BASE sel over all      : {sel_all}/{tot} = {sel_all/tot:.3f}")
print(f"  BASE sel over the n={meas} BOTH/CORRECT subset: {sel_meas}/{meas} = {sel_meas/meas:.3f}")
print("  -> BASE CORRECT is printed in the same column as arms whose denominator is n=299")

print("\n=== 3. acov join: binding_map.action_field vs api_info Action-fields keys ===")
def norm(x): return "".join(ch for ch in str(x).lower() if ch.isalnum())
loss=0; recoverable=0; ex=[]
for s in ps:
    rp=s.get("raw_prediction") or {}
    if not rp: continue
    a=(rp.get("final_applet") or {}).get("action") or {}
    areq=NEW.required(NEW.afields(a))
    if not areq: continue
    ab={NEW.nz(b.get("field")):b for b in NEW.base_binds(rp)["action_bindings"]}
    miss=[k for k in areq if NEW.bval(ab.get(NEW.nz(k))) is None]
    if not miss: continue
    loss+=1
    # would a slug/normalised join have found it?
    alt={norm(b["field"]) for b in NEW.base_binds(rp)["action_bindings"]}
    slugs={}
    for rf in (rp.get("current_requirements") or {}).get("required_fields") or []:
        slugs[norm(rf.get("name"))]=norm(rf.get("slug"))
    rec=[k for k in miss if norm(k) in alt or slugs.get(norm(k)) in alt
         or any(norm(k).startswith(x) or x.startswith(norm(k)) for x in alt if x)]
    if rec:
        recoverable+=1
        if len(ex)<4: ex.append((sorted(areq)[:3], sorted(alt)[:3]))
print(f"  samples with acov<1.0: {loss}   of which a normalised/prefix join would have matched: {recoverable}")
for e in ex: print("   e.g. required=",e[0]," bound=",e[1])

print("\n=== 4. base_binds source relabelling ===")
c=collections.Counter()
for s in ps:
    rp=s.get("raw_prediction") or {}
    for b in rp.get("binding_map") or []:
        st=NEW.nz(b.get("source_type"))
        if st=="static" and b.get("static_value") is None and b.get("ingredient_name") is not None: c["static_uses_ingredient_name"]+=1
        if st=="ingredient" and b.get("ingredient_name") is None and b.get("static_value") is not None: c["ingredient_uses_static_value"]+=1
        if st not in ("static","ingredient"): c[f"other:{st}"]+=1
print("  ", dict(c) or "none")
dups=0
for s in ps:
    rp=s.get("raw_prediction") or {}
    f=[NEW.nz(b.get("action_field")) for b in rp.get("binding_map") or []]
    dups += len(f)-len(set(f))
print(f"  duplicate action_field entries collapsed by the dict keying: {dups}")

print("\n=== 5. the n=4 self-check dump the summary quotes ===")
d4=json.load(open("results/ARMS_gpt-oss_120b_4.json"))
rs=[r["BASE"] for r in d4]
ac=[r["acov"] for r in rs if r["acov"] is not None]
print("  BASE acov:", ac, "mean", sum(ac)/len(ac))
print("  refs/grounded:", sum(r["refs"] for r in rs), sum(r["grounded"] for r in rs))
print("  stats/stat_ok:", sum(r["stats"] for r in rs), sum(r["stat_ok"] for r in rs))
print("  tcov_measurable:", [r["tcov_measurable"] for r in rs])
