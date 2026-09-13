"""
FARM architecture arms. Retrieval is FIXED (stored top-5 candidates) so every
difference is attributable to the Stage-2 design, not to retrieval variance.

A0 rag1        : rank-1 pair, 0 LLM calls, no bindings          (the floor)
A1 rag1_bind   : rank-1 pair + 1 constrained call, BOTH sides   (proposed)
A2 joint       : 1 constrained call selects 5x5 AND binds       (single-call)
A3 xs_bind     : cross-schema rerank of 25 pairs + 1 call       (scorer consumed)
BASE           : stored 4-call pipeline, read from file
"""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = sys.argv[1] if len(sys.argv) > 1 else "11435"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 40
MODEL = "granite4:small-h"
URL = f"http://127.0.0.1:{PORT}/api/chat"

WORD = re.compile(r"[a-z0-9]+")
def toks(s): return set(WORD.findall(str(s or "").lower()))
def nz(x): return str(x or "").strip().lower()

SYN = {"time":"date","date":"time","title":"name","name":"title","content":"message",
       "message":"content","body":"content","url":"link","link":"url","image":"photo",
       "photo":"image","author":"user","user":"author","location":"address","address":"location"}

def field_ing_score(fname, fspec, ing_name, ing_spec):
    a = toks(fname) | toks((fspec or {}).get("Slug"))
    b = toks(ing_name) | toks((ing_spec or {}).get("Slug"))
    if not a or not b: return 0.0
    inter = a & b
    s = len(inter) / max(1, min(len(a), len(b)))
    if s == 0:
        for x in a:
            if SYN.get(x) in b: s = 0.5; break
    return s

def required(fields):
    if not isinstance(fields, dict): return {}
    return {k: v for k, v in fields.items()
            if isinstance(v, dict) and str(v.get("Required","")).lower() == "true"}

def tfields(c): return ((c.get("api_info") or {}).get("Trigger fields")) or {}
def afields(c): return ((c.get("api_info") or {}).get("Action fields")) or {}
def ingredients(c): return ((c.get("api_info") or {}).get("Ingredients")) or {}

def coverage(t, a):
    """Fraction of the action's REQUIRED fields matchable to a trigger ingredient."""
    req = required(afields(a)); ings = ingredients(t)
    if not req: return 1.0
    if not isinstance(ings, dict): return 0.0
    hit = 0
    for fn, fs in req.items():
        best = max((field_ing_score(fn, fs, inm, isp if isinstance(isp, dict) else {})
                    for inm, isp in ings.items()), default=0.0)
        if best > 0: hit += 1
    return hit / len(req)

BIND_SCHEMA = {
  "type":"object","additionalProperties":False,
  "properties":{
    "trigger_bindings":{"type":"array","items":{"type":"object","additionalProperties":False,
      "properties":{"field":{"type":"string"},
                    "source":{"type":"string","enum":["ingredient","static"]},
                    "value":{"type":"string"}},
      "required":["field","source","value"]}},
    "action_bindings":{"type":"array","items":{"type":"object","additionalProperties":False,
      "properties":{"field":{"type":"string"},
                    "source":{"type":"string","enum":["ingredient","static"]},
                    "value":{"type":"string"}},
      "required":["field","source","value"]}}},
  "required":["trigger_bindings","action_bindings"]}

SEL_SCHEMA = json.loads(json.dumps(BIND_SCHEMA))
SEL_SCHEMA["properties"]["trigger_choice"] = {"type":"integer"}
SEL_SCHEMA["properties"]["action_choice"]  = {"type":"integer"}
SEL_SCHEMA["required"] = ["trigger_choice","action_choice","trigger_bindings","action_bindings"]

def schema_block(c, kind):
    f = tfields(c) if kind=="trigger" else afields(c)
    lines=[f"{kind.upper()}: {c.get('service_name')}", f"  desc: {str(c.get('description'))[:150]}"]
    if isinstance(f,dict) and f:
        lines.append(f"  {kind} fields:")
        for k,v in f.items():
            vv = v if isinstance(v, dict) else {}
            r = "REQUIRED" if str(vv.get("Required","")).lower()=="true" else "optional"
            lines.append(f"    - {k} [{r}]")
    if kind=="trigger":
        ings=ingredients(c)
        if isinstance(ings,dict) and ings:
            lines.append("  ingredients this trigger emits:")
            for k,v in ings.items():
                vv = v if isinstance(v, dict) else {}
                lines.append(f"    - {{{{{k}}}}} ({vv.get('Type','?')}) e.g. {str(vv.get('Example',''))[:28]}")
    return "\n".join(lines)

def call(messages, schema, timeout=180):
    body={"model":MODEL,"messages":messages,"stream":False,"format":schema,
          "options":{"temperature":0,"num_ctx":8192,"num_predict":700}}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r: d=json.loads(r.read())
    except Exception as e:
        return None, time.time()-t0, f"REQFAIL {str(e)[:60]}"
    c=(d.get("message") or {}).get("content","")
    try: return json.loads(c), time.time()-t0, None
    except Exception: return None, time.time()-t0, "BADJSON"

RULES = ("Bind EVERY REQUIRED field on BOTH the trigger and the action.\n"
         "source='ingredient' -> value MUST be one of the trigger's ingredient names, no braces.\n"
         "source='static'     -> value MUST be a literal taken from the user's request.\n"
         "Never invent a value that is not in the request and not an ingredient.")

def arm_bind(q, t, a):
    msgs=[{"role":"user","content":
        f"User request: {q}\n\n{schema_block(t,'trigger')}\n\n{schema_block(a,'action')}\n\n{RULES}"}]
    return call(msgs, BIND_SCHEMA)

def arm_joint(q, tcs, acs):
    tb="\n".join(f"[{i}] {schema_block(c,'trigger')}" for i,c in enumerate(tcs))
    ab="\n".join(f"[{i}] {schema_block(c,'action')}" for i,c in enumerate(acs))
    msgs=[{"role":"user","content":
        f"User request: {q}\n\nTRIGGER OPTIONS:\n{tb}\n\nACTION OPTIONS:\n{ab}\n\n"
        f"Choose the best trigger index and action index, then bind their fields.\n{RULES}"}]
    return call(msgs, SEL_SCHEMA)

# ---------------- scoring ----------------
def score(t, a, binds, gold_t, gold_a, q=""):
    sel = (nz(t.get("service_name"))==gold_t) and (nz(a.get("service_name"))==gold_a)
    treq, areq = required(tfields(t)), required(afields(a))
    tb = {nz(b.get("field")): b for b in (binds or {}).get("trigger_bindings", []) if isinstance(b,dict)}
    ab = {nz(b.get("field")): b for b in (binds or {}).get("action_bindings", []) if isinstance(b,dict)}
    tcov = (sum(1 for k in treq if nz(k) in tb and str(tb[nz(k)].get("value","")).strip()) / len(treq)) if treq else None
    acov = (sum(1 for k in areq if nz(k) in ab and str(ab[nz(k)].get("value","")).strip()) / len(areq)) if areq else None
    parts=[p for p in (tcov,acov) if p is not None]
    both = 1.0 if parts and all(p==1.0 for p in parts) else 0.0
    ing_names={nz(k) for k in ingredients(t)} if isinstance(ingredients(t),dict) else set()
    refs=grounded=0; stats=stat_ok=0
    qt = toks(q)
    for b in list(tb.values())+list(ab.values()):
        src=b.get("source"); val=str(b.get("value","")).strip()
        if src=="ingredient":
            refs+=1
            if nz(val.strip("{} ")) in ing_names: grounded+=1
        elif src=="static":
            stats+=1
            vt=toks(val)
            if vt and vt <= qt: stat_ok+=1      # every token of the literal is in the query
    return dict(sel=sel, tcov=tcov, acov=acov, both=both, refs=refs, grounded=grounded,
                stats=stats, stat_ok=stat_ok,
                correct=(1.0 if (sel and both==1.0) else 0.0))

def main():
    d=json.load(open("results/e2e_eval.json")); ps=d["per_sample"][:N]
    print(f"model={MODEL} port={PORT} n={len(ps)}", flush=True)

    def one(s):
        rp=s.get("raw_prediction") or {}
        if not rp: return None
        ref=s.get("reference") or {}
        q=s["query"]; gt,ga=nz(ref.get("trigger")),nz(ref.get("action"))
        tcs=rp.get("trigger_candidates") or []; acs=rp.get("action_candidates") or []
        if not tcs or not acs: return None
        out={}
        t0,a0=tcs[0],acs[0]
        out["A0"]=dict(score(t0,a0,{},gt,ga,q), lat=0.0, calls=0)
        b,lat,err=arm_bind(q,t0,a0)
        out["A1"]=dict(score(t0,a0,b or {},gt,ga,q), lat=lat, calls=1, err=err)
        j,lat,err=arm_joint(q,tcs,acs)
        if j:
            ti=max(0,min(len(tcs)-1,int(j.get("trigger_choice",0))))
            ai=max(0,min(len(acs)-1,int(j.get("action_choice",0))))
        else: ti=ai=0
        out["A2"]=dict(score(tcs[ti],acs[ai],j or {},gt,ga,q), lat=lat, calls=1, err=err)
        best=max(((0.7*coverage(t,a)+0.3*(0.5*(t.get("score",0)+a.get("score",0))),i,k)
                  for i,t in enumerate(tcs) for k,a in enumerate(acs)), key=lambda x:x[0])
        tb_,ab_=tcs[best[1]],acs[best[2]]
        b3,lat,err=arm_bind(q,tb_,ab_)
        out["A3"]=dict(score(tb_,ab_,b3 or {},gt,ga,q), lat=lat, calls=1, err=err)
        pr=s.get("prediction") or {}
        bsel=(nz(pr.get("trigger"))==gt and nz(pr.get("action"))==ga)
        out["BASE"]=dict(sel=bsel, tcov=0.0 if required(tfields(t0)) else None, acov=None,
                         both=0.0, refs=0, grounded=0, stats=0, stat_ok=0, correct=0.0,
                         lat=s.get("elapsed_time",0.0), calls=4)
        return out

    t0=time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        res=[r for r in ex.map(one, ps) if r]
    wall=time.time()-t0
    print(f"wall={wall:.0f}s  ({wall/max(1,len(res)):.1f}s/sample)\n", flush=True)

    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/ARMS_{len(res)}.json","w") as f: json.dump(res,f)
    print(f"per-sample dumped -> results/ARMS_{len(res)}.json\n")
    print(f"{'arm':<6}{'calls':>6}{'joint_sel':>11}{'BOTH':>7}{'CORRECT':>9}{'trig_cov':>10}{'act_cov':>9}{'ing_grnd':>10}{'stat_grnd':>11}{'lat_s':>8}")
    print("-"*93)
    for arm,label in [("A0","A0"),("A1","A1"),("A2","A2"),("A3","A3"),("BASE","BASE")]:
        rs=[r[arm] for r in res]
        sel=sum(1 for r in rs if r["sel"])/len(rs)
        tc=[r["tcov"] for r in rs if r["tcov"] is not None]
        ac=[r["acov"] for r in rs if r["acov"] is not None]
        both=sum(r["both"] for r in rs)/len(rs)
        refs=sum(r["refs"] for r in rs); gr=sum(r["grounded"] for r in rs)
        lat=sum(r["lat"] for r in rs)/len(rs)
        corr=sum(r["correct"] for r in rs)/len(rs)
        st=sum(r.get("stats",0) for r in rs); sok=sum(r.get("stat_ok",0) for r in rs)
        print(f"{label:<6}{rs[0]['calls']:>6}{sel:>11.3f}{both:>7.3f}{corr:>9.3f}"
              f"{(sum(tc)/len(tc) if tc else float('nan')):>10.3f}"
              f"{(sum(ac)/len(ac) if ac else float('nan')):>9.3f}"
              f"{(gr/refs if refs else 0):>10.3f}{(sok/st if st else 0):>11.3f}{lat:>8.1f}")
    errs=[(a,r[a].get("err")) for r in res for a in ("A1","A2","A3") if r[a].get("err")]
    if errs:
        from collections import Counter
        print("\nerrors:", Counter(f"{a}:{e}" for a,e in errs).most_common(6))

main()
