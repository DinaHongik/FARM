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

Repairs relative to jobs/arms_cloud.py (harness defects only; prompts, RULES,
choice template, lenient parser, temperature and num_predict are untouched so
completed runs stay comparable):

  D2 BASE is measured from results/e2e_eval.json instead of hardcoded to zero.
     The stored pipeline binds ACTION fields only (binding_map entries are keyed
     by 'action_field'; final_applet.trigger carries no field_values), so action
     coverage and all four grounding counters are recoverable and trigger
     coverage is NOT. Unrecoverable quantities are None and print as n/a.
  D3 BOTH is a real conjunction over both sides, and no longer scores 0 for a
     sample that requires nothing. The vacuous population is now a column
     (no_req) plus a two-sided-only figure (BOTH2), so single-sidedness is
     visible rather than folded into the headline number.
  D4 A field counts as bound only on a real value. JSON null, empty containers
     and whitespace no longer count via str() stringification.
  D5 A2 persists ti, ai, the raw choice, and whether the choice actually parsed,
     so the rank-1 fallback rate is recoverable from the per-sample dump.
  D7 A zero denominator prints n/a, never 0.000.
  D8 main() is guarded by __name__.
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def bval(b):
    """D4: the bound value as a real non-empty string, else None.

    Raw None is rejected BEFORE stringification, because str(None) is the
    non-empty string 'None' and a JSON null would otherwise count as bound.
    Empty containers are rejected for the same reason (str([]) == '[]').
    """
    if not isinstance(b, dict): return None
    v = b.get("value")
    if v is None: return None
    if isinstance(v, bool): return None
    if isinstance(v, (list, dict, tuple, set)) and not v: return None
    s = str(v).strip()
    return s or None

def bvalid(b, ing_names, qt):
    """D9: is this binding's value legitimate, not merely non-empty?

    source=ingredient -> must name a real ingredient the trigger emits.
    source=static     -> must be traceable to the user's request.
    Mutation testing showed the lenient check scores 'zzqqxx_not_a_real_value'
    identically to a correct bind, so BOTH measured string length, not binding.
    """
    val = bval(b)
    if val is None: return False
    src = nz(b.get("source"))
    if src == "ingredient":
        return nz(val.strip("{} ")) in ing_names
    if src == "static":
        vt = toks(val)
        return bool(vt) and vt <= qt
    return False

def side_cov_strict(req, binds, ing_names, qt):
    if not req: return None
    return sum(1 for k in req if bvalid(binds.get(nz(k)), ing_names, qt)) / len(req)

def side_cov(req, binds):
    """Coverage of one side. None means the side requires nothing (vacuous),
    which is distinct from 0.0 (required fields present, none of them bound)."""
    if not req: return None
    return sum(1 for k in req if bval(binds.get(nz(k))) is not None) / len(req)

def score(t, a, binds, gt, ga, q=""):
    sel  = (nz(t.get("service_name")) == gt) and (nz(a.get("service_name")) == ga)
    treq, areq = required(tfields(t)), required(afields(a))
    tb = {nz(b.get("field")): b for b in (binds or {}).get("trigger_bindings", []) if isinstance(b, dict)}
    ab = {nz(b.get("field")): b for b in (binds or {}).get("action_bindings", [])  if isinstance(b, dict)}
    tcov = side_cov(treq, tb)
    acov = side_cov(areq, ab)
    _names = {nz(k) for k in ings(t)} if isinstance(ings(t), dict) else set()
    _qt = toks(q)
    tcovv = side_cov_strict(treq, tb, _names, _qt)
    acovv = side_cov_strict(areq, ab, _names, _qt)
    bothv = 1.0 if (tcovv in (None, 1.0) and acovv in (None, 1.0)) else 0.0
    # D3: a real conjunction. A side that requires nothing is vacuously
    # satisfied rather than dropped, and a sample that requires nothing on
    # EITHER side no longer scores 0 for having nothing to bind. The vacuous
    # population is reported separately (no_req / BOTH2) so this stays visible.
    both = 1.0 if (tcov in (None, 1.0) and acov in (None, 1.0)) else 0.0
    names = {nz(k) for k in ings(t)} if isinstance(ings(t), dict) else set()
    qt = toks(q); refs = grounded = stats = stat_ok = 0
    for b in list(tb.values()) + list(ab.values()):
        val = bval(b)
        if val is None: continue
        if b.get("source") == "ingredient":
            refs += 1
            if nz(val.strip("{} ")) in names: grounded += 1
        elif b.get("source") == "static":
            stats += 1
            vt = toks(val)
            if vt and vt <= qt: stat_ok += 1
    return dict(sel=sel, tcov=tcov, acov=acov, both=both,
                tcovv=tcovv, acovv=acovv, bothv=bothv,
                correctv=1.0 if (sel and bothv == 1.0) else 0.0,
                t_req=len(treq), a_req=len(areq),
                refs=refs, grounded=grounded, stats=stats, stat_ok=stat_ok,
                correct=1.0 if (sel and both == 1.0) else 0.0)

def base_binds(rp):
    """D2: re-express the stored pipeline's binding_map in the shape score()
    consumes. Every stored entry is keyed by 'action_field' and the dump holds
    no trigger-side binding anywhere, so the trigger list is empty by
    construction and its coverage is unrecoverable, not zero."""
    bm = rp.get("binding_map")
    if not isinstance(bm, list):
        fa = rp.get("final_applet")
        bm = fa.get("bindings") if isinstance(fa, dict) else None
    out = []
    for b in bm or []:
        if not isinstance(b, dict): continue
        src = nz(b.get("source_type"))
        val = b.get("ingredient_name") if src == "ingredient" else b.get("static_value")
        if val is None:
            val = b.get("static_value") if src == "ingredient" else b.get("ingredient_name")
        out.append({"field": b.get("action_field"), "source": src, "value": val})
    return {"trigger_bindings": [], "action_bindings": out}

def base_pair(rp, tcs, acs):
    """The pair the stored pipeline actually committed to, or (None, None) when
    the dump does not identify one."""
    fa = rp.get("final_applet") if isinstance(rp.get("final_applet"), dict) else None
    if fa:
        bt, ba = fa.get("trigger"), fa.get("action")
        if isinstance(bt, dict) and isinstance(ba, dict): return bt, ba
    ti, ai = rp.get("current_trigger_idx"), rp.get("current_action_idx")
    if isinstance(ti, int) and isinstance(ai, int) and 0 <= ti < len(tcs) and 0 <= ai < len(acs):
        return tcs[ti], acs[ai]
    return None, None

UNMEASURED = dict(tcov=None, acov=None, both=None, correct=None, t_req=None, a_req=None,
                  refs=None, grounded=None, stats=None, stat_ok=None)

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
    # D5: persist what the choice actually was, so the rank-1 fallback rate is
    # recoverable from the dump. choice_ok is False whenever the arm silently
    # fell back to index 0 because the reply did not parse or carried no usable
    # integer choice; choice_ok True with ti==0 is a CHOSEN rank-1, not a fallback.
    ti = ai = 0
    parsed = j is not None
    choice_ok = False
    choice_raw = None
    if j:
        choice_raw = [j.get("trigger_choice"), j.get("action_choice")]
        try:
            ti = max(0, min(len(tcs)-1, int(j.get("trigger_choice", 0))))
            ai = max(0, min(len(acs)-1, int(j.get("action_choice", 0))))
            choice_ok = True
        except (TypeError, ValueError, AttributeError):
            # Non-integer / missing / non-dict choice. Kept lenient on purpose:
            # enforcement must not become a confound between models.
            pass
    out["A2"] = dict(score(tcs[ti], acs[ai], j or {}, gt, ga, q), lat=lat, calls=1, err=err,
                     ti=ti, ai=ai, parsed=parsed, choice_ok=choice_ok, choice_raw=choice_raw,
                     fallback=1.0 if not choice_ok else 0.0)

    best = max(((0.7*coverage(t, a) + 0.3*0.5*((t.get("score") or 0)+(a.get("score") or 0)), i, k)
                for i, t in enumerate(tcs) for k, a in enumerate(acs)), key=lambda x: x[0])
    tb_, ab_ = tcs[best[1]], acs[best[2]]
    b3, lat, err = call(f"User request: {q}\n\n{block(tb_,'trigger')}\n\n{block(ab_,'action')}\n\n{RULES}", False)
    out["A3"] = dict(score(tb_, ab_, b3 or {}, gt, ga, q), lat=lat, calls=1, err=err)

    # D2: BASE measured from the stored pipeline output, not hardcoded.
    # sel keeps the as-run definition (prediction vs reference) so the number
    # stays comparable with completed runs; everything else is recomputed from
    # the committed pair and its stored binding_map.
    bsel = (nz(pr.get("trigger")) == gt and nz(pr.get("action")) == ga)
    bt, ba = base_pair(rp, tcs, acs)
    if bt is None:
        base = dict(UNMEASURED, sel=bsel, tcov_measurable=False)
    else:
        base = score(bt, ba, base_binds(rp), gt, ga, q)
        base["sel"] = bsel
        base["correct"] = 1.0 if (bsel and base["both"] == 1.0) else 0.0
        # The stored pipeline has no trigger-binding stage at all, so when the
        # committed trigger DOES require fields their coverage is unrecoverable
        # (n/a), and BOTH/CORRECT are unrecoverable with it. When it requires
        # nothing, the trigger side is vacuous and BOTH is fully measurable.
        base["tcov_measurable"] = not base["t_req"]
        if base["t_req"]:
            base["tcov"] = base["both"] = base["correct"] = None
            # the validated variants are unrecoverable for the same reason;
            # leaving them at 0.0 would score an unmeasurable sample as a failure
            base["tcovv"] = base["bothv"] = base["correctv"] = None
    out["BASE"] = dict(base, lat=s.get("elapsed_time", 0.0), calls=4, err=None)
    return out

def mean(vals):
    """Mean over the measurable entries, plus how many there were."""
    v = [x for x in vals if x is not None]
    return (sum(v)/len(v), len(v)) if v else (None, 0)

def ratio(nums, dens):
    """D7: None (prints n/a) on a zero or absent denominator, never 0.000."""
    d = sum(x for x in dens if x is not None)
    if not d: return None
    return sum(x for x in nums if x is not None) / d

def fmt(x, w, p=3):
    if x is None or x != x: return f"{'n/a':>{w}}"
    return f"{x:>{w}.{p}f}"

def main():
    ps = json.load(open("results/e2e_eval.json"))["per_sample"][:N]
    print(f"model={MODEL}  backend={'CLOUD' if CLOUD else 'LOCAL'}  n={len(ps)}  workers={WORKERS}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, p): i for i, p in enumerate(ps)}
        got = {}
        for k, f in enumerate(as_completed(futs), 1):
            got[futs[f]] = f.result()
            if k % 10 == 0 or k == len(ps):
                el = time.time() - t0
                print(f"[progress] {k}/{len(ps)}  {el:.0f}s  {el/k:.2f}s/sample  "
                      f"eta={(len(ps)-k)*el/k:.0f}s  "
                      f"parse_fail={STATS['parse_fail']} req_fail={STATS['req_fail']}", flush=True)
        res = [got[i] for i in range(len(ps)) if got.get(i)]
    wall = time.time() - t0
    print(f"wall={wall:.0f}s  ({wall/max(1,len(res)):.2f}s/sample)   "
          f"parse_fail={STATS['parse_fail']}  req_fail={STATS['req_fail']}\n", flush=True)
    tag = MODEL.replace(":", "_").replace("/", "_")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/ARMS_{tag}_{len(res)}.json", "w"))
    print(f"per-sample -> results/ARMS_{tag}_{len(res)}.json\n")
    hdr = (f"{'arm':<6}{'calls':>6}{'sel':>8}{'BOTH':>7}{'CORRECT':>9}{'trig_cov':>10}{'act_cov':>9}"
           f"{'ing_grnd':>10}{'stat_grnd':>11}{'lat_s':>8}{'BOTH2':>8}{'no_req':>8}"
           f"{'BOTHv':>8}{'CORRv':>8}")
    print(hdr); print("-"*len(hdr))
    foot = []
    for arm in ("A0", "A1", "A2", "A3", "BASE"):
        rs = [r[arm] for r in res]; n = len(rs)
        selm, _   = mean([1.0 if r["sel"] else 0.0 for r in rs])
        bothm, nb = mean([r["both"] for r in rs])
        corm, nc  = mean([r["correct"] for r in rs])
        tcm, ntc  = mean([r["tcov"] for r in rs])
        acm, nac  = mean([r["acov"] for r in rs])
        ig = ratio([r["grounded"] for r in rs], [r["refs"] for r in rs])
        sg = ratio([r["stat_ok"] for r in rs], [r["stats"] for r in rs])
        latm, _   = mean([r["lat"] for r in rs])
        # BOTH2: BOTH restricted to samples where BOTH sides require something,
        # i.e. the subset on which BOTH is a genuinely two-sided conjunction.
        bvm, _    = mean([r.get("bothv") for r in rs])
        cvm, _    = mean([r.get("correctv") for r in rs])
        two = [r["both"] for r in rs if r["t_req"] and r["a_req"]]
        b2, nb2 = mean(two)
        # no_req: samples where at least one side requires nothing, so that side
        # is vacuously satisfied inside BOTH.
        vac = [None if (r["t_req"] is None or r["a_req"] is None)
               else (1.0 if (not r["t_req"] or not r["a_req"]) else 0.0) for r in rs]
        vacm, nvac = mean(vac)
        print(f"{arm:<6}{rs[0]['calls']:>6}{fmt(selm,8)}{fmt(bothm,7)}{fmt(corm,9)}"
              f"{fmt(tcm,10)}{fmt(acm,9)}{fmt(ig,10)}{fmt(sg,11)}{fmt(latm,8,1)}"
              f"{fmt(b2,8)}{fmt(vacm,8)}{fmt(bvm,8)}{fmt(cvm,8)}")
        vt = sum(1 for r in rs if r["t_req"] == 0)
        va = sum(1 for r in rs if r["a_req"] == 0)
        foot.append(f"  {arm:<5} n={n:<4} both_n={nb:<4} both2_n={nb2:<4} correct_n={nc:<4} tcov_n={ntc:<4} "
                    f"acov_n={nac:<4} no_req_n={nvac:<4} (no_req_trigger={vt}, no_req_action={va})")
    print("\ndenominators (measurable samples behind each mean; n/a means nothing measurable):")
    for line in foot: print(line)
    a2 = [r["A2"] for r in res]
    fb, nfb = mean([r["fallback"] for r in a2])
    pr_, _  = mean([1.0 if r["parsed"] else 0.0 for r in a2])
    print(f"\nA2 choice: parsed={fmt(pr_,6)}  rank-1 fallback rate={fmt(fb,6)}  (n={nfb}) "
          f"[per-sample ti/ai/parsed/choice_ok/choice_raw are in the dump]")
    print("BASE trig_cov/BOTH/CORRECT are n/a where the committed trigger has required fields: "
          "the stored pipeline binds action fields only, so trigger coverage is not recoverable.")

if __name__ == "__main__":
    main()
