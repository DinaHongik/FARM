"""Does A2 actually choose, or does it silently fall back to rank-1?

arms_cloud.py sets ti=ai=0 and only overwrites inside a try/except that
swallows every failure, so a missing or non-integer choice is indistinguishable
from a genuine vote for rank-1. This measures how often that fallback fires.
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms_lib as A
A.MODEL = os.environ.get("PROBE_MODEL", "gpt-oss:120b")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
assert A.KEY, "OLLAMA_API_KEY not set"
print(f"probe model={A.MODEL} cloud={A.CLOUD}", flush=True)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 10

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:N]

def probe(s):
    rp = s.get("raw_prediction") or {}
    if not rp: return None
    ref = s.get("reference") or {}
    q = s.get("query", "")
    gt, ga = A.nz(ref.get("trigger")), A.nz(ref.get("action"))
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: return None

    # gold position in the retrieved list == the headroom a selector could win
    gti = next((i for i, c in enumerate(tcs) if A.nz(c.get("service_name")) == gt), -1)
    gai = next((i for i, c in enumerate(acs) if A.nz(c.get("service_name")) == ga), -1)

    tb = "\n".join(f"[{i}] {A.block(c,'trigger')}" for i, c in enumerate(tcs))
    ab = "\n".join(f"[{i}] {A.block(c,'action')}"  for i, c in enumerate(acs))
    j, lat, err = A.call(f"User request: {q}\n\nTRIGGER OPTIONS:\n{tb}\n\nACTION OPTIONS:\n{ab}\n\n"
                         f"Choose the best trigger index and action index, then bind their fields.\n{A.RULES}", True)

    raw_t = j.get("trigger_choice", "<MISSING>") if j else "<NOJSON>"
    raw_a = j.get("action_choice",  "<MISSING>") if j else "<NOJSON>"
    ti = ai = 0; coerced = False
    if j:
        try:
            ti = max(0, min(len(tcs)-1, int(j.get("trigger_choice", 0))))
            ai = max(0, min(len(acs)-1, int(j.get("action_choice", 0))))
        except Exception:
            coerced = True
    return dict(raw_t=raw_t, raw_a=raw_a, ti=ti, ai=ai, coerced=coerced,
                nojson=(j is None), gti=gti, gai=gai, err=err)

t0 = time.time(); res = []
with ThreadPoolExecutor(max_workers=W) as ex:
    futs = [ex.submit(probe, p) for p in ps]
    for k, f in enumerate(as_completed(futs), 1):
        r = f.result()
        if r: res.append(r)
        if k % 20 == 0: print(f"[progress] {k}/{len(ps)} {time.time()-t0:.0f}s", flush=True)

n = len(res)
missing = sum(1 for r in res if r["raw_t"] in ("<MISSING>", "<NOJSON>"))
nonzero = sum(1 for r in res if r["ti"] != 0 or r["ai"] != 0)
print(f"\nn={n}  wall={time.time()-t0:.0f}s")
print(f"no JSON at all            : {sum(1 for r in res if r['nojson'])}")
print(f"choice key MISSING        : {sum(1 for r in res if r['raw_t']=='<MISSING>')}")
print(f"int() coercion FAILED     : {sum(1 for r in res if r['coerced'])}")
print(f"parsed a choice           : {n - missing}")
print(f"chose something != rank-1 : {nonzero}  ({nonzero/max(1,n):.3f})")
print(f"chose rank-1 for BOTH     : {n - nonzero}  ({(n-nonzero)/max(1,n):.3f})")

# ceilings
r1 = sum(1 for r in res if r["gti"] == 0 and r["gai"] == 0) / max(1, n)
top5 = sum(1 for r in res if r["gti"] >= 0 and r["gai"] >= 0) / max(1, n)
hit = sum(1 for r in res if r["ti"] == r["gti"] and r["ai"] == r["gai"]) / max(1, n)
print(f"\njoint gold @rank-1 (A0 floor) : {r1:.3f}")
print(f"joint gold in top-5 (ceiling) : {top5:.3f}")
print(f"A2 actually landed on gold    : {hit:.3f}")
print(f"headroom the selector ignores : {top5-r1:.3f}")
print("\n--- first 15 raw choices ---")
for r in res[:15]:
    print(f"  raw_t={str(r['raw_t'])[:22]:<24} raw_a={str(r['raw_a'])[:22]:<24} "
          f"-> ti={r['ti']} ai={r['ai']}  gold=({r['gti']},{r['gai']})")
