"""A/B the A2 choice template.

arms_cloud.py line 93 emits {"trigger_choice": 0, "action_choice": 0, ...} as the
required reply shape. Every other slot is a placeholder ("<field name>", "<value>"),
so the choice fields are the only ones handed a concrete valid answer -- and that
answer is exactly the rank-1 pair A2 is benchmarked against, at temperature 0.
Variant B replaces the literal with a placeholder. Nothing else differs.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms_lib as A
A.MODEL = os.environ.get("PROBE_MODEL", "gpt-oss:120b")
assert A.KEY, "OLLAMA_API_KEY not set"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
W = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ps = json.load(open("results/e2e_eval.json"))["per_sample"][:N]

def call_variant(prompt, anchored):
    shape = {"trigger_bindings": A.EX_B, "action_bindings": A.EX_B}
    choice = {"trigger_choice": 0, "action_choice": 0} if anchored else \
             {"trigger_choice": "<index of chosen trigger>", "action_choice": "<index of chosen action>"}
    shape = {**choice, **shape}
    msg = prompt + "\n\nReply with ONLY this JSON, no prose, no markdown fence:\n" + json.dumps(shape)
    body = {"model": A.MODEL, "messages": [{"role": "user", "content": msg}],
            "stream": False, "options": {"temperature": 0, "num_predict": 3000}}
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {A.KEY}"}
    import urllib.request
    try:
        req = urllib.request.Request(A.URL, data=json.dumps(body).encode(), headers=hdr)
        with urllib.request.urlopen(req, timeout=150) as r:
            d = json.loads(r.read())
        return A.lenient((d.get("message") or {}).get("content", ""))
    except Exception:
        return None

def one(s):
    rp = s.get("raw_prediction") or {}
    if not rp: return None
    ref = s.get("reference") or {}; q = s.get("query", "")
    gt, ga = A.nz(ref.get("trigger")), A.nz(ref.get("action"))
    tcs, acs = rp.get("trigger_candidates") or [], rp.get("action_candidates") or []
    if not tcs or not acs: return None
    gti = next((i for i, c in enumerate(tcs) if A.nz(c.get("service_name")) == gt), -1)
    gai = next((i for i, c in enumerate(acs) if A.nz(c.get("service_name")) == ga), -1)
    tb = "\n".join(f"[{i}] {A.block(c,'trigger')}" for i, c in enumerate(tcs))
    ab = "\n".join(f"[{i}] {A.block(c,'action')}"  for i, c in enumerate(acs))
    prompt = (f"User request: {q}\n\nTRIGGER OPTIONS:\n{tb}\n\nACTION OPTIONS:\n{ab}\n\n"
              f"Choose the best trigger index and action index, then bind their fields.\n{A.RULES}")
    out = {"gti": gti, "gai": gai}
    for name, anchored in (("A", True), ("B", False)):
        j = call_variant(prompt, anchored)
        ti = ai = 0; parsed = False
        if j:
            try:
                ti = max(0, min(len(tcs)-1, int(j.get("trigger_choice", 0))))
                ai = max(0, min(len(acs)-1, int(j.get("action_choice", 0))))
                parsed = True
            except Exception: pass
        out[name] = dict(ti=ti, ai=ai, parsed=parsed,
                         sel=(A.nz(tcs[ti].get("service_name")) == gt and A.nz(acs[ai].get("service_name")) == ga))
    return out

t0 = time.time(); res = []
with ThreadPoolExecutor(max_workers=W) as ex:
    futs = [ex.submit(one, p) for p in ps]
    for k, f in enumerate(as_completed(futs), 1):
        r = f.result()
        if r: res.append(r)
        if k % 50 == 0: print(f"[progress] {k}/{len(ps)} {time.time()-t0:.0f}s", flush=True)

n = len(res)
floor = sum(1 for r in res if r["gti"] == 0 and r["gai"] == 0)/n
ceil_ = sum(1 for r in res if r["gti"] >= 0 and r["gai"] >= 0)/n
print(f"\nmodel={A.MODEL}  n={n}  wall={time.time()-t0:.0f}s")
print(f"rank-1 floor (A0) {floor:.3f}   top-5 ceiling {ceil_:.3f}\n")
print(f"{'variant':<26}{'parsed':>8}{'!=rank1':>9}{'sel':>8}")
print("-"*51)
for name, lbl in (("A", "A anchored (as-run)"), ("B", "B placeholder")):
    p = sum(1 for r in res if r[name]["parsed"])/n
    nz_ = sum(1 for r in res if r[name]["ti"] or r[name]["ai"])/n
    sl = sum(1 for r in res if r[name]["sel"])/n
    print(f"{lbl:<26}{p:>8.3f}{nz_:>9.3f}{sl:>8.3f}")
