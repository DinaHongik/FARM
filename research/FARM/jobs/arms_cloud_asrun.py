"""
FARM Stage-2 architecture arms, driven against the Ollama Cloud API.

Retrieval is FIXED (stored top-5) so every difference is attributable to the
Stage-2 design. The SAME lenient parser is applied to every arm and every model,
because cloud cannot enforce schemas and enforcement must not be a confound.

  A0 rag1      0 calls  rank-1 pair, no bindings                 (floor)
  A1 rag1_bind 1 call   rank-1 pair + bind BOTH sides            (proposed)
  A2 joint     1 call   select from 5x5 AND bind                 (single-call)
  A3 xs_bind   1 call   cross-schema rerank of 25 pairs + bind   (scorer consumed)
  BASE         4 calls  stored pipeline, read from file
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss:120b"
N       = int(sys.argv[2]) if len(sys.argv) > 2 else 40
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
KEY     = os.environ.get("OLLAMA_API_KEY", "")
CLOUD   = bool(KEY)
URL     = "https://ollama.com/api/chat" if CLOUD else "http://127.0.0.1:11435/api/chat"
FENCE   = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
WORD    = re.compile(r"[a-z0-9]+")
STATS   = {"parse_fail": 0, "req_fail": 0}

def toks(s): return set(WORD.findall(str(s or "").lower()))
def nz(x):   return str(x or "").strip().lower()

def lenient(txt):
    if not txt: return None
    m = FENCE.search(txt)
    if m: txt = m.group(1)
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i: return None
    body = txt[i:j+1]
    for cand in (body, body.replace(",}", "}").replace(",]", "]")):
        try: return json.loads(cand)
        except Exception: pass
    return None

SYN = {"time":"date","date":"time","title":"name","name":"title","content":"message",
       "message":"content","body":"content","url":"link","link":"url","image":"photo",
       "photo":"image","author":"user","user":"author","location":"address","address":"location"}

def required(f):
    if not isinstance(f, dict): return {}
    return {k:v for k,v in f.items() if isinstance(v,dict) and str(v.get("Required","")).lower()=="true"}
def tfields(c): return ((c.get("api_info") or {}).get("Trigger fields")) or {}
def afields(c): return ((c.get("api_info") or {}).get("Action fields")) or {}
def ings(c):    return ((c.get("api_info") or {}).get("Ingredients")) or {}

def fscore(fn, fs, inm, isp):
    a = toks(fn) | toks((fs or {}).get("Slug")); b = toks(inm) | toks((isp or {}).get("Slug"))
    if not a or not b: return 0.0
    s = len(a & b) / max(1, min(len(a), len(b)))
    if s == 0:
        for x in a:
            if SYN.get(x) in b: return 0.5
    return s

def coverage(t, a):
    req = required(afields(a)); ii = ings(t)
    if not req: return 1.0
    if not isinstance(ii, dict): return 0.0
    hit = sum(1 for fn, fs in req.items()
              if max((fscore(fn, fs, k, v if isinstance(v,dict) else {}) for k,v in ii.items()), default=0.0) > 0)
    return hit / len(req)

def block(c, kind):
    f = tfields(c) if kind == "trigger" else afields(c)
    L = [f"{kind.upper()}: {c.get('service_name')}", f"  desc: {str(c.get('description'))[:150]}"]
    if isinstance(f, dict) and f:
        L.append(f"  {kind} fields:")
        for k, v in f.items():
            vv = v if isinstance(v, dict) else {}
            L.append(f"    - {k} [{'REQUIRED' if str(vv.get('Required','')).lower()=='true' else 'optional'}]")
    if kind == "trigger" and isinstance(ings(c), dict):
        L.append("  ingredients this trigger emits:")
        for k, v in ings(c).items():
            vv = v if isinstance(v, dict) else {}
            L.append(f"    - {k} ({vv.get('Type','?')}) e.g. {str(vv.get('Example',''))[:28]}")
    return "\n".join(L)

EX_B = [{"field": "<field name>", "source": "ingredient|static", "value": "<value>"}]
RULES = ("Bind EVERY REQUIRED field on BOTH the trigger and the action.\n"
         "source='ingredient' -> value MUST be one of the trigger's ingredient names (no braces).\n"
         "source='static'     -> value MUST be a literal taken from the user's request.\n"
         "Never invent a value that is neither in the request nor an ingredient.")

def call(prompt, with_choice, timeout=150):
    shape = {"trigger_bindings": EX_B, "action_bindings": EX_B}
    if with_choice: shape = {"trigger_choice": 0, "action_choice": 0, **shape}
    msg = prompt + "\n\nReply with ONLY this JSON, no prose, no markdown fence:\n" + json.dumps(shape)
    body = {"model": MODEL, "messages": [{"role": "user", "content": msg}],
            "stream": False, "options": {"temperature": 0, "num_predict": 3000}}
    hdr = {"Content-Type": "application/json"}
    if CLOUD: hdr["Authorization"] = f"Bearer {KEY}"
    t0 = time.time(); d = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read()); break
        except Exception as e:
            if attempt == 2:
                STATS["req_fail"] += 1
                return None, time.time()-t0, f"REQFAIL {str(e)[:40]}"
            time.sleep(2)
    if d is None or "error" in (d or {}):
        STATS["req_fail"] += 1
        return None, time.time()-t0, f"ERR {str((d or {}).get('error'))[:40]}"
    p = lenient((d.get("message") or {}).get("content", ""))
    if p is None:
        STATS["parse_fail"] += 1
        return None, time.time()-t0, "BADJSON"
    return p, time.time()-t0, None

def score(t, a, binds, gt, ga, q=""):
    sel  = (nz(t.get("service_name")) == gt) and (nz(a.get("service_name")) == ga)
    treq, areq = required(tfields(t)), required(afields(a))
    tb = {nz(b.get("field")): b for b in (binds or {}).get("trigger_bindings", []) if isinstance(b, dict)}
    ab = {nz(b.get("field")): b for b in (binds or {}).get("action_bindings", [])  if isinstance(b, dict)}
    tcov = (sum(1 for k in treq if nz(k) in tb and str(tb[nz(k)].get("value","")).strip())/len(treq)) if treq else None
    acov = (sum(1 for k in areq if nz(k) in ab and str(ab[nz(k)].get("value","")).strip())/len(areq)) if areq else None
    parts = [p for p in (tcov, acov) if p is not None]
    both  = 1.0 if parts and all(p == 1.0 for p in parts) else 0.0
    names = {nz(k) for k in ings(t)} if isinstance(ings(t), dict) else set()
    qt = toks(q); refs = grounded = stats = stat_ok = 0
    for b in list(tb.values()) + list(ab.values()):
        val = str(b.get("value", "")).strip()
        if b.get("source") == "ingredient":
            refs += 1
            if nz(val.strip("{} ")) in names: grounded += 1
        elif b.get("source") == "static":
            stats += 1
            vt = toks(val)
            if vt and vt <= qt: stat_ok += 1
    return dict(sel=sel, tcov=tcov, acov=acov, both=both, refs=refs, grounded=grounded,
                stats=stats, stat_ok=stat_ok, correct=1.0 if (sel and both == 1.0) else 0.0)

def one(s):
    rp = s.get("raw_prediction") or {}
    if not rp: return None
    ref = s.get("reference") or {}; pr = s.get("prediction") or {}
    q = s.get("query", ""); gt, ga = nz(ref.get("trigger")), nz(ref.get("action"))
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: return None
    t0, a0 = tcs[0], acs[0]
    out = {"A0": dict(score(t0, a0, {}, gt, ga, q), lat=0.0, calls=0, err=None)}

    b, lat, err = call(f"User request: {q}\n\n{block(t0,'trigger')}\n\n{block(a0,'action')}\n\n{RULES}", False)
    out["A1"] = dict(score(t0, a0, b or {}, gt, ga, q), lat=lat, calls=1, err=err)

    tb = "\n".join(f"[{i}] {block(c,'trigger')}" for i, c in enumerate(tcs))
    ab = "\n".join(f"[{i}] {block(c,'action')}"  for i, c in enumerate(acs))
    j, lat, err = call(f"User request: {q}\n\nTRIGGER OPTIONS:\n{tb}\n\nACTION OPTIONS:\n{ab}\n\n"
                       f"Choose the best trigger index and action index, then bind their fields.\n{RULES}", True)
    ti = ai = 0
    if j:
        try:
            ti = max(0, min(len(tcs)-1, int(j.get("trigger_choice", 0))))
            ai = max(0, min(len(acs)-1, int(j.get("action_choice", 0))))
        except Exception: pass
    out["A2"] = dict(score(tcs[ti], acs[ai], j or {}, gt, ga, q), lat=lat, calls=1, err=err)

    best = max(((0.7*coverage(t, a) + 0.3*0.5*((t.get("score") or 0)+(a.get("score") or 0)), i, k)
                for i, t in enumerate(tcs) for k, a in enumerate(acs)), key=lambda x: x[0])
    tb_, ab_ = tcs[best[1]], acs[best[2]]
    b3, lat, err = call(f"User request: {q}\n\n{block(tb_,'trigger')}\n\n{block(ab_,'action')}\n\n{RULES}", False)
    out["A3"] = dict(score(tb_, ab_, b3 or {}, gt, ga, q), lat=lat, calls=1, err=err)

    out["BASE"] = dict(sel=(nz(pr.get("trigger")) == gt and nz(pr.get("action")) == ga),
                       tcov=0.0 if required(tfields(t0)) else None, acov=None, both=0.0,
                       refs=0, grounded=0, stats=0, stat_ok=0, correct=0.0,
                       lat=s.get("elapsed_time", 0.0), calls=4, err=None)
    return out

def main():
    ps = json.load(open("results/e2e_eval.json"))["per_sample"][:N]
    print(f"model={MODEL}  backend={'CLOUD' if CLOUD else 'LOCAL'}  n={len(ps)}  workers={WORKERS}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = [r for r in ex.map(one, ps) if r]
    wall = time.time() - t0
    print(f"wall={wall:.0f}s  ({wall/max(1,len(res)):.2f}s/sample)   "
          f"parse_fail={STATS['parse_fail']}  req_fail={STATS['req_fail']}\n", flush=True)
    tag = MODEL.replace(":", "_").replace("/", "_")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/ARMS_{tag}_{len(res)}.json", "w"))
    print(f"per-sample -> results/ARMS_{tag}_{len(res)}.json\n")
    hdr = f"{'arm':<6}{'calls':>6}{'sel':>8}{'BOTH':>7}{'CORRECT':>9}{'trig_cov':>10}{'act_cov':>9}{'ing_grnd':>10}{'stat_grnd':>11}{'lat_s':>8}"
    print(hdr); print("-"*len(hdr))
    for arm in ("A0", "A1", "A2", "A3", "BASE"):
        rs = [r[arm] for r in res]; n = len(rs)
        tc = [r["tcov"] for r in rs if r["tcov"] is not None]
        ac = [r["acov"] for r in rs if r["acov"] is not None]
        rf, gr = sum(r["refs"] for r in rs), sum(r["grounded"] for r in rs)
        st, so = sum(r["stats"] for r in rs), sum(r["stat_ok"] for r in rs)
        print(f"{arm:<6}{rs[0]['calls']:>6}{sum(1 for r in rs if r['sel'])/n:>8.3f}"
              f"{sum(r['both'] for r in rs)/n:>7.3f}{sum(r['correct'] for r in rs)/n:>9.3f}"
              f"{(sum(tc)/len(tc) if tc else float('nan')):>10.3f}"
              f"{(sum(ac)/len(ac) if ac else float('nan')):>9.3f}"
              f"{(gr/rf if rf else 0):>10.3f}{(so/st if st else 0):>11.3f}"
              f"{sum(r['lat'] for r in rs)/n:>8.1f}")

main()
