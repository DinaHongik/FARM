"""Is the task label itself ambiguous, and does a lexical signal help?

Candidates are graded on service_name -- the bare FUNCTION name. The gold's true
identity is its url: ifttt.com/<channel>/<kind>/<function>. This measures the same
retrieval under BOTH label definitions, plus BM25 and dense+BM25 fusion.
"""
import json, re, sys, math
from collections import Counter, defaultdict
import numpy as np, torch
sys.path.insert(0, ".")
from rag.indexer import (extract_trigger_text, extract_action_text,
                         extract_channel_from_filter_code)
from sentence_transformers import SentenceTransformer

KS = [1, 5, 20, 50, 100]
TOKRE = re.compile(r"[a-z0-9]+")
def toks(s): return TOKRE.findall(str(s or "").lower())
def nz(x): return str(x or "").strip().lower()
def canon(s): return re.sub(r"[^a-z0-9]", "", str(s or "").lower())

def trig_chan(rec):
    for _, info in ((rec.get("api_info") or {}).get("Ingredients") or {}).items():
        if isinstance(info, dict) and "Filter code" in info:
            return canon(extract_channel_from_filter_code(info["Filter code"]))
    return ""

def act_chan(rec):
    for _, info in ((rec.get("api_info") or {}).get("Action fields") or {}).items():
        if isinstance(info, dict) and "Filter code method" in info:
            return canon(extract_channel_from_filter_code(info["Filter code method"]))
    return ""

def gold_chan(g):
    m = re.search(r"ifttt\.com/([^/]+)/", str(g.get("url") or ""))
    return canon(m.group(1)) if m else ""

class BM25:
    def __init__(self, corpus_toks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_toks)
        self.lens = np.array([len(d) for d in corpus_toks], dtype=np.float32)
        self.avgdl = float(self.lens.mean())
        self.post = defaultdict(list)
        for i, d in enumerate(corpus_toks):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.N - len(pl) + 0.5) / (len(pl) + 0.5))
                    for t, pl in self.post.items()}
    def scores(self, q):
        s = np.zeros(self.N, dtype=np.float32)
        dl = self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
        for t in set(q):
            pl = self.post.get(t)
            if not pl: continue
            idx = np.fromiter((i for i, _ in pl), np.int64, len(pl))
            f = np.fromiter((c for _, c in pl), np.float32, len(pl))
            s[idx] += self.idf[t] * (f * (self.k1 + 1)) / (f + dl[idx])
        return s

def rank_matrix(S):
    o = np.argsort(-S, axis=1)
    r = np.empty_like(o)
    r[np.arange(S.shape[0])[:, None], o] = np.arange(S.shape[1])[None, :]
    return o, r

def best_rank(S, gold_rows):
    """Rank of the BEST-scoring acceptable row (a gold may accept several rows)."""
    out = []
    for i, rows in enumerate(gold_rows):
        if not rows: out.append(None); continue
        row = S[i]; best = max(row[j] for j in rows)
        out.append(int((row > best).sum()))
    return out

def rec(ranks, k):
    ok = [r for r in ranks if r is not None]
    return sum(1 for r in ok if r < k) / len(ok) if ok else 0.0

ev = json.load(open("data/clean_split/eval.json"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
res = {}
for path, side, extract, chanfn, enc in [
        ("data/triggers_rag.json", "trigger", extract_trigger_text, trig_chan, "runs/clean/trigger-encoder/final"),
        ("data/actions_rag.json", "action", extract_action_text, act_chan, "runs/clean/action-encoder/final")]:
    C = json.load(open(path))
    docs = [extract(r) for r in C]
    names = [nz(r.get("service_name")) for r in C]
    chans = [chanfn(r) for r in C]
    print(f"\n================ {side} ================", flush=True)
    print(f"docs={len(C)}  docs with NO channel: {sum(1 for c in chans if not c)} "
          f"({sum(1 for c in chans if not c)/len(C):.1%})")

    by_name = defaultdict(list)
    by_pair = defaultdict(list)
    for i, (n, c) in enumerate(zip(names, chans)):
        by_name[n].append(i)
        by_pair[(c, n)].append(i)

    gname = [nz(r[side].get("service_name")) for r in ev]
    gchan = [gold_chan(r[side]) for r in ev]
    rows_name = [by_name.get(n, []) for n in gname]
    rows_pair = [by_pair.get((c, n), []) for c, n in zip(gchan, gname)]

    amb = sum(1 for r in rows_name if len(r) > 1)
    unres = sum(1 for r in rows_pair if not r)
    print(f"gold name matches >1 corpus row: {amb}/{len(ev)} ({amb/len(ev):.3f})")
    print(f"gold (channel,name) resolves to no corpus row: {unres}/{len(ev)} ({unres/len(ev):.3f})")
    qnamed = sum(1 for r, c in zip(ev, gchan) if c and c in canon(r["query"]))
    print(f"gold channel appears in query string: {qnamed}/{len(ev)} ({qnamed/len(ev):.3f})")

    m = SentenceTransformer(enc, device=dev, model_kwargs={"torch_dtype": torch.float32})
    D = m.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    Sd = (Q @ D.T).astype(np.float32)
    bm = BM25([toks(d) for d in docs])
    Sl = np.stack([bm.scores(toks(r["query"])) for r in ev]).astype(np.float32)
    _, rd = rank_matrix(Sd); _, rl = rank_matrix(Sl)
    Sh = 1.0 / (60 + rd) + 1.0 / (60 + rl)

    print(f"{'k':>4} | {'NAME-only label':^26} | {'(channel,name) label':^26}")
    print(f"{'':>4} | {'dense':>8}{'bm25':>9}{'hybrid':>9} | {'dense':>8}{'bm25':>9}{'hybrid':>9}")
    store = {}
    for lbl, rows in [("name", rows_name), ("pair", rows_pair)]:
        store[lbl] = {m_: best_rank(S, rows) for m_, S in [("dense", Sd), ("bm25", Sl), ("hybrid", Sh)]}
    for k in KS:
        a = [rec(store["name"][m_], k) for m_ in ("dense", "bm25", "hybrid")]
        b = [rec(store["pair"][m_], k) for m_ in ("dense", "bm25", "hybrid")]
        print(f"{k:>4} | {a[0]:>8.3f}{a[1]:>9.3f}{a[2]:>9.3f} | {b[0]:>8.3f}{b[1]:>9.3f}{b[2]:>9.3f}", flush=True)
    res[side] = store

print("\n================ JOINT (both sides) ================")
print(f"{'k':>4} | {'NAME-only label':^26} | {'(channel,name) label':^26}")
print(f"{'':>4} | {'dense':>8}{'bm25':>9}{'hybrid':>9} | {'dense':>8}{'bm25':>9}{'hybrid':>9}")
for k in KS:
    out = []
    for lbl in ("name", "pair"):
        for m_ in ("dense", "bm25", "hybrid"):
            t, a = res["trigger"][lbl][m_], res["action"][lbl][m_]
            ok = [(x, y) for x, y in zip(t, a) if x is not None and y is not None]
            out.append(sum(1 for x, y in ok if x < k and y < k) / len(ev))
    print(f"{k:>4} | {out[0]:>8.3f}{out[1]:>9.3f}{out[2]:>9.3f} | {out[3]:>8.3f}{out[4]:>9.3f}{out[5]:>9.3f}", flush=True)
