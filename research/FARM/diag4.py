"""Rebuild the corpus keyed on url (the true identity) and measure two things:

  A. flat retrieval over the TRUE label space, with the channel always in the
     document text -- how much of the loss was the corpus, not the model?
  B. a two-level decomposition: pick the channel, then the function inside it.
     There are ~740/582 channels but only ~2.7 functions per channel, so if the
     channel is resolvable the second choice is nearly free.

Uses the EXISTING encoders unchanged, so every number here is a lower bound:
they were trained on the old name-deduped text, not on this.
"""
import json, re, sys
from collections import Counter, defaultdict
import numpy as np, torch
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer

def nz(x): return str(x or "").strip().lower()
def chan_of(u):
    m = re.search(r"ifttt\.com/([^/]+)/", str(u or ""))
    return nz(m.group(1)) if m else ""
def pretty(slug): return slug.replace("_", " ").strip()

raw = json.load(open("data/iftttt_dataset_full_trigger_action.json"))
comp = {"trigger": {}, "action": {}}
chan_title = {}
for svc in raw:
    if not isinstance(svc, dict): continue
    cslug = chan_of(str(svc.get("service_url") or "") + "/")
    if cslug: chan_title.setdefault(cslug, nz(svc.get("service_name")))
    for ap in svc.get("applets") or []:
        if not isinstance(ap, dict): continue
        for c in ap.get("components") or []:
            if not isinstance(c, dict): continue
            lab = nz(c.get("label"))
            side = "trigger" if lab == "if" else ("action" if lab == "then" else None)
            if side and nz(c.get("url")):
                comp[side].setdefault(nz(c.get("url")), c)

def render(c, u):
    """Same spirit as the old indexer, but the channel comes from the url so it
    is ALWAYS present, and the function slug is included as an identity token."""
    ch = chan_of(u)
    fn = u.rstrip("/").rsplit("/", 1)[-1]
    disp = chan_title.get(ch, pretty(ch))
    ai = c.get("api_info") or {}
    fields = ai.get("Trigger fields") or ai.get("Action fields") or {}
    labels = [i.get("Label", n) for n, i in fields.items() if isinstance(i, dict)]
    ing = list((ai.get("Ingredients") or {}).keys())
    parts = [f"[{disp}] [{pretty(ch)}] {nz(c.get('service_name'))}.",
             str(c.get("description") or "")]
    if labels: parts.append("Fields: " + ", ".join(labels) + ".")
    if ing: parts.append("Provides: " + ", ".join(ing) + ".")
    parts.append(f"id: {ch}/{fn}")
    return " ".join(p for p in parts if p.strip())

ev = json.load(open("data/clean_split/eval.json"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ENC = {"trigger": "runs/clean/trigger-encoder/final", "action": "runs/clean/action-encoder/final"}
KS = [1, 5, 20, 50]
joint = {}

for side in ("trigger", "action"):
    U = comp[side]
    urls = list(U)
    docs = [render(U[u], u) for u in urls]
    chans = [chan_of(u) for u in urls]
    u2i = {u: i for i, u in enumerate(urls)}
    per_chan = Counter(chans)
    gold_u = [nz(r[side].get("url")) for r in ev]
    gi = [u2i.get(u) for u in gold_u]
    print(f"\n================ {side} ================", flush=True)
    print(f"functions={len(urls)}  channels={len(per_chan)}  "
          f"functions/channel mean={np.mean(list(per_chan.values())):.2f} "
          f"median={np.median(list(per_chan.values())):.0f} max={max(per_chan.values())}")
    gsz = np.array([per_chan[chan_of(u)] for u in gold_u])
    print(f"  eval golds: functions inside the gold channel mean={gsz.mean():.2f} "
          f"median={np.median(gsz):.0f} p90={np.percentile(gsz,90):.0f}")
    print(f"  ORACLE channel + random function = {np.mean(1.0/gsz):.3f}")

    m = SentenceTransformer(ENC[side], device=dev, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    S = (Q @ D.T).astype(np.float32)
    order = np.argsort(-S, axis=1)

    # A. flat retrieval, TRUE url label
    ranks = [int((S[i] > S[i, g]).sum()) for i, g in enumerate(gi)]
    print("  A. flat, url-identity label:  " +
          "  ".join(f"R@{k}={sum(1 for r in ranks if r<k)/len(ranks):.3f}" for k in KS), flush=True)

    # B1. channel retrieval: one document per channel
    cnames = sorted(per_chan)
    cdocs = []
    fn_by_chan = defaultdict(list)
    for u in urls: fn_by_chan[chan_of(u)].append(nz(U[u].get("service_name")))
    for c in cnames:
        cdocs.append(f"{chan_title.get(c, pretty(c))} ({pretty(c)}). "
                     + ", ".join(sorted(set(fn_by_chan[c]))[:25]))
    CD = m.encode(cdocs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    CS = (Q @ CD.T).astype(np.float32)
    c2i = {c: i for i, c in enumerate(cnames)}
    gc = [c2i.get(chan_of(u)) for u in gold_u]
    cranks = [int((CS[i] > CS[i, g]).sum()) if g is not None else None for i, g in enumerate(gc)]
    okc = [r for r in cranks if r is not None]
    print("  B1. channel retrieval:        " +
          "  ".join(f"R@{k}={sum(1 for r in okc if r<k)/len(okc):.3f}" for k in KS), flush=True)

    # B2. function choice restricted to the ORACLE channel
    hit = 0
    for i, u in enumerate(gold_u):
        c = chan_of(u)
        cand = [u2i[x] for x in urls if chan_of(x) == c] if per_chan[c] else []
        if not cand: continue
        if max(cand, key=lambda j: S[i, j]) == gi[i]: hit += 1
    print(f"  B2. function | ORACLE channel: acc={hit/len(ev):.3f}", flush=True)

    # B3. end-to-end two-level, no oracle: best channel from B1, then best function in it
    hit2 = 0
    for i in range(len(ev)):
        c = cnames[int(np.argmax(CS[i]))]
        cand = [u2i[x] for x in urls if chan_of(x) == c]
        if cand and max(cand, key=lambda j: S[i, j]) == gi[i]: hit2 += 1
    print(f"  B3. two-level end-to-end:     acc={hit2/len(ev):.3f}", flush=True)
    del m; torch.cuda.empty_cache()
    joint[side] = dict(flat=ranks, twolevel_oracle=None)

print("\n================ JOINT flat, url-identity ================")
for k in KS:
    t, a = joint["trigger"]["flat"], joint["action"]["flat"]
    print(f"  R@{k} = {sum(1 for x, y in zip(t, a) if x < k and y < k)/len(t):.3f}")
