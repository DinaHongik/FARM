"""Ablate the corrected document text. All four variants are indexed over the
TRUE url label space (1985/1501 functions) and scored with the EXISTING encoders,
so the only thing changing is what goes in the document string."""
import json, re, sys
from collections import defaultdict, Counter
import numpy as np, torch
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer

def nz(x): return str(x or "").strip().lower()
def chan_of(u):
    m = re.search(r"ifttt\.com/([^/]+)/", str(u or "")); return nz(m.group(1)) if m else ""
def pretty(s): return s.replace("_", " ").strip()

raw = json.load(open("data/iftttt_dataset_full_trigger_action.json"))
comp = {"trigger": {}, "action": {}}
ctitle = {}
for svc in raw:
    if not isinstance(svc, dict): continue
    cs = chan_of(str(svc.get("service_url") or "") + "/")
    if cs: ctitle.setdefault(cs, nz(svc.get("service_name")))
    for ap in svc.get("applets") or []:
        if not isinstance(ap, dict): continue
        for c in ap.get("components") or []:
            if not isinstance(c, dict): continue
            l = nz(c.get("label"))
            s = "trigger" if l == "if" else ("action" if l == "then" else None)
            if s and nz(c.get("url")): comp[s].setdefault(nz(c.get("url")), c)

def render(c, u, chan=True, idtok=True):
    ch = chan_of(u); fn = u.rstrip("/").rsplit("/", 1)[-1]
    ai = c.get("api_info") or {}
    fields = ai.get("Trigger fields") or ai.get("Action fields") or {}
    labels = [i.get("Label", n) for n, i in fields.items() if isinstance(i, dict)]
    ing = list((ai.get("Ingredients") or {}).keys())
    p = []
    if chan: p.append(f"[{ctitle.get(ch, pretty(ch))}] [{pretty(ch)}]")
    p.append(f"{nz(c.get('service_name'))}.")
    p.append(str(c.get("description") or ""))
    if labels: p.append("Fields: " + ", ".join(labels) + ".")
    if ing: p.append("Provides: " + ", ".join(ing) + ".")
    if idtok: p.append(f"id: {ch}/{fn}")
    return " ".join(x for x in p if str(x).strip())

ev = json.load(open("data/clean_split/eval.json"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ENC = {"trigger": "runs/clean/trigger-encoder/final", "action": "runs/clean/action-encoder/final"}
VAR = [("baseline (no channel, no id)", False, False),
       ("+ channel only", True, False),
       ("+ id token only", False, True),
       ("+ both", True, True)]
ranks = defaultdict(dict)
for side in ("trigger", "action"):
    U = comp[side]; urls = list(U); u2i = {u: i for i, u in enumerate(urls)}
    gi = [u2i[nz(r[side].get("url"))] for r in ev]
    m = SentenceTransformer(ENC[side], device=dev, model_kwargs={"torch_dtype": torch.float32})
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    print(f"\n--- {side} (url-identity label, {len(urls)} functions) ---", flush=True)
    for tag, ch, it in VAR:
        D = m.encode([render(U[u], u, ch, it) for u in urls], batch_size=64,
                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        S = (Q @ D.T).astype(np.float32)
        r = [int((S[i] > S[i, g]).sum()) for i, g in enumerate(gi)]
        ranks[tag][side] = r
        print(f"  {tag:<30} R@1={sum(1 for x in r if x<1)/len(r):.3f}  "
              f"R@5={sum(1 for x in r if x<5)/len(r):.3f}  "
              f"R@20={sum(1 for x in r if x<20)/len(r):.3f}", flush=True)
    del m; torch.cuda.empty_cache()

print("\n--- JOINT (both sides right) ---")
for tag, _, _ in VAR:
    t, a = ranks[tag]["trigger"], ranks[tag]["action"]
    print(f"  {tag:<30} " + "  ".join(
        f"R@{k}={sum(1 for x, y in zip(t, a) if x<k and y<k)/len(t):.3f}" for k in (1, 5, 20)))
print("\n  reference: OLD corpus, name-label joint R@1=0.423 R@5=0.716 (easier label)")
