"""Where is the accuracy actually lost? Measure before training anything else.

The reranker can only pick from what stage 1 retrieves, so a reranker gain is
capped by recall@k. This script decomposes the loss into: (a) dense recall@k per
side, (b) whether the gold CHANNEL is literally named in the query -- IFTTT urls
are ifttt.com/<channel>/triggers/<function>, a hierarchy flat dense retrieval
ignores, (c) BM25 and hybrid recall, (d) the size of the within-channel choice.
"""
import json, re, sys, math
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, ".")
from rag.indexer import extract_trigger_text, extract_action_text

KS = [1, 5, 10, 20, 50, 100]
TOKRE = re.compile(r"[a-z0-9]+")


def toks(s):
    return TOKRE.findall(str(s or "").lower())


def channel_of(url):
    m = re.search(r"ifttt\.com/([^/]+)/", str(url or ""))
    return m.group(1).lower() if m else ""


class BM25:
    """Standard Robertson/Sparck-Jones BM25, k1=1.5 b=0.75. Implemented here
    rather than adding rank_bm25 as a dependency."""

    def __init__(self, corpus_toks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_toks)
        self.lens = np.array([len(d) for d in corpus_toks], dtype=np.float32)
        self.avgdl = float(self.lens.mean())
        self.postings = defaultdict(list)
        for i, d in enumerate(corpus_toks):
            for t, f in Counter(d).items():
                self.postings[t].append((i, f))
        self.idf = {}
        for t, pl in self.postings.items():
            n = len(pl)
            self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, q_toks):
        s = np.zeros(self.N, dtype=np.float32)
        denom_len = self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
        for t in set(q_toks):
            pl = self.postings.get(t)
            if not pl:
                continue
            idf = self.idf[t]
            idx = np.fromiter((i for i, _ in pl), dtype=np.int64, count=len(pl))
            f = np.fromiter((f for _, f in pl), dtype=np.float32, count=len(pl))
            s[idx] += idf * (f * (self.k1 + 1)) / (f + denom_len[idx])
        return s


def ranks_from_scores(S, gold_idx_per_row):
    """Rank of the gold doc for each query row (0-based). S: [nq, ndocs]."""
    out = []
    for i, g in enumerate(gold_idx_per_row):
        if g is None:
            out.append(None)
            continue
        row = S[i]
        out.append(int((row > row[g]).sum()))
    return out


def recall_at(ranks, k):
    ok = [r for r in ranks if r is not None]
    return sum(1 for r in ok if r < k) / len(ok) if ok else 0.0


def rrf_fuse(rank_a, rank_b, n, kconst=60):
    """Reciprocal rank fusion over two full score matrices' argsort ranks."""
    return 1.0 / (kconst + rank_a) + 1.0 / (kconst + rank_b)


def main():
    trag = json.load(open("data/triggers_rag.json"))
    arag = json.load(open("data/actions_rag.json"))
    ev = json.load(open("data/clean_split/eval.json"))
    print(f"eval={len(ev)}  trigger_docs={len(trag)}  action_docs={len(arag)}", flush=True)

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    sides = {
        "trigger": dict(rag=trag, docs=[extract_trigger_text(r) for r in trag],
                        enc="runs/clean/trigger-encoder/final", key="trigger"),
        "action": dict(rag=arag, docs=[extract_action_text(r) for r in arag],
                       enc="runs/clean/action-encoder/final", key="action"),
    }

    per_side_ranks = {}
    for name, S in sides.items():
        urls = [str(r.get("url") or "").strip().lower() for r in S["rag"]]
        url_to_i = {u: i for i, u in enumerate(urls)}
        chans = [channel_of(u) for u in urls]
        gold_urls = [str(r[S["key"]].get("url") or "").strip().lower() for r in ev]
        gold_i = [url_to_i.get(u) for u in gold_urls]
        missing = sum(1 for g in gold_i if g is None)
        print(f"\n===== {name} =====  gold url not in corpus: {missing}/{len(ev)}", flush=True)

        m = SentenceTransformer(S["enc"], device=dev, model_kwargs={"torch_dtype": torch.float32})
        D = m.encode(S["docs"], batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        Q = m.encode([r["query"] for r in ev], batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        del m
        torch.cuda.empty_cache()
        Sd = (Q @ D.T).astype(np.float32)

        bm = BM25([toks(d) for d in S["docs"]])
        Sl = np.stack([bm.scores(toks(r["query"])) for r in ev]).astype(np.float32)

        rd = ranks_from_scores(Sd, gold_i)
        rl = ranks_from_scores(Sl, gold_i)

        # full rank vectors for RRF
        ord_d = np.argsort(-Sd, axis=1)
        ord_l = np.argsort(-Sl, axis=1)
        rank_d = np.empty_like(ord_d)
        rank_l = np.empty_like(ord_l)
        rows = np.arange(Sd.shape[0])[:, None]
        rank_d[rows, ord_d] = np.arange(Sd.shape[1])[None, :]
        rank_l[rows, ord_l] = np.arange(Sl.shape[1])[None, :]
        Sh = rrf_fuse(rank_d, rank_l, Sd.shape[1])
        rh = ranks_from_scores(Sh, gold_i)

        print(f"{'k':>5} {'dense':>8} {'bm25':>8} {'hybrid':>8}")
        for k in KS:
            print(f"{k:>5} {recall_at(rd,k):>8.3f} {recall_at(rl,k):>8.3f} {recall_at(rh,k):>8.3f}", flush=True)

        # --- channel structure ---
        gold_chan = [chans[g] if g is not None else None for g in gold_i]
        named = 0
        for r, gc in zip(ev, gold_chan):
            if gc is None:
                continue
            qt = set(toks(r["query"]))
            ct = set(toks(gc))
            if ct and ct <= qt:
                named += 1
        n_ok = sum(1 for g in gold_chan if g is not None)
        print(f"gold channel literally named in query: {named}/{n_ok} = {named/n_ok:.3f}")

        # channel-level recall of dense retrieval (is the top-k in the right channel?)
        chan_arr = np.array(chans)
        for k in [1, 5, 20]:
            hit = 0
            for i, gc in enumerate(gold_chan):
                if gc is None:
                    continue
                if gc in set(chan_arr[ord_d[i, :k]]):
                    hit += 1
            print(f"  dense: gold CHANNEL present in top-{k}: {hit/n_ok:.3f}")

        # how big is the within-channel choice?
        by_chan = Counter(chans)
        sizes = [by_chan[gc] for gc in gold_chan if gc]
        sizes = np.array(sizes)
        print(f"  functions per gold channel: mean={sizes.mean():.1f} median={np.median(sizes):.0f} "
              f"p90={np.percentile(sizes,90):.0f} max={sizes.max()}")
        print(f"  oracle-channel random pick would score {np.mean(1.0/sizes):.3f}")

        per_side_ranks[name] = dict(dense=rd, bm25=rl, hybrid=rh)

    # --- joint ---
    print("\n===== JOINT (both sides correct) =====")
    print(f"{'k':>5} {'dense':>8} {'bm25':>8} {'hybrid':>8}")
    for k in KS:
        row = []
        for meth in ["dense", "bm25", "hybrid"]:
            t = per_side_ranks["trigger"][meth]
            a = per_side_ranks["action"][meth]
            ok = [(x, y) for x, y in zip(t, a) if x is not None and y is not None]
            row.append(sum(1 for x, y in ok if x < k and y < k) / len(ok))
        print(f"{k:>5} {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f}", flush=True)


main()
