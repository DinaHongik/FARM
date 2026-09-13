"""The encoders were trained on positives rendered from RAW applet components, which
carry no 'category' key, so the positive reads "[Channel] Name. ...". The index is
built from the *_rag.json corpora, which all carry 'category', so every indexed doc
reads "[Channel] [Category] Name. ...". Not one training positive matches an indexed
document. Question: what does that mismatch cost at retrieval time?"""
import json, re, sys
import numpy as np, torch
sys.path.insert(0, ".")
from rag.indexer import extract_trigger_text, extract_action_text
from sentence_transformers import SentenceTransformer

def nz(x): return str(x or "").strip().lower()
ev = json.load(open("data/clean_split/eval.json"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ENC = {"trigger": "runs/clean/trigger-encoder/final", "action": "runs/clean/action-encoder/final"}
EX = {"trigger": extract_trigger_text, "action": extract_action_text}
KS = [1, 5, 20]
res = {}
for side in ("trigger", "action"):
    corpus = json.load(open(f"data/{side}s_rag.json"))
    cats = {nz(r.get("category")) for r in corpus if r.get("category")}
    ex = EX[side]
    shipped = [ex(r) for r in corpus]
    # identical rendering, category key removed -> the format training actually saw
    stripped = [ex({k: v for k, v in r.items() if k != "category"}) for r in corpus]
    names = [nz(r.get("service_name")) for r in corpus]
    gold = [nz(r[side].get("service_name")) for r in ev]
    rows = {}
    for i, n in enumerate(names): rows.setdefault(n, []).append(i)
    grows = [rows.get(g, []) for g in gold]

    m = SentenceTransformer(ENC[side], device=dev, model_kwargs={"torch_dtype": torch.float32})
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    out = {}
    for tag, docs in (("shipped index  [Channel] [Category]", shipped),
                      ("training format [Channel] only", stripped)):
        D = m.encode(docs, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        S = (Q @ D.T).astype(np.float32)
        rk = []
        for i, gr in enumerate(grows):
            if not gr: rk.append(None); continue
            best = max(S[i, j] for j in gr)
            rk.append(int((S[i] > best).sum()))
        ok = [r for r in rk if r is not None]
        out[tag] = rk
        print(f"  {side:8} {tag:38} " +
              "  ".join(f"R@{k}={sum(1 for r in ok if r<k)/len(ok):.3f}" for k in KS), flush=True)
    res[side] = out
    del m; torch.cuda.empty_cache()

print("\nJOINT (both sides correct):")
for tag in list(res["trigger"]):
    t, a = res["trigger"][tag], res["action"][tag]
    pairs = [(x, y) for x, y in zip(t, a) if x is not None and y is not None]
    print(f"  {tag:38} " + "  ".join(
        f"R@{k}={sum(1 for x,y in pairs if x<k and y<k)/len(pairs):.3f}" for k in KS))
