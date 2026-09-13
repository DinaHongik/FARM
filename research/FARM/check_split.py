"""Preflight assertions for the FARM data pipeline.

Each assertion is paired with the value it returns on the OLD data. A checker that
only passes on the new data proves nothing; these are written so that every one of
them fails, with a specific number, on the files that produced the current paper.

Run:  python3 check_split.py           # new pipeline, exits non-zero on failure
      python3 check_split.py --old     # the same six checks against the old files
"""
from __future__ import annotations

import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from farm.render import parse_url

STOP = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "if", "when",
        "your", "my", "with", "from", "at", "is", "it", "this", "that", "be", "by"}
BATCH = 16


def nq(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s or "").lower())).strip()
def content(s): return frozenset(t for t in nq(s).split() if len(t) > 2 and t not in STOP)
def jload(p): return json.loads(Path(p).read_text())


class Report:
    def __init__(self): self.rows = []; self.failed = 0
    def add(self, tag, desc, value, ok, note=""):
        self.rows.append((tag, desc, value, ok, note))
        if not ok: self.failed += 1
    def show(self, title):
        print(f"\n{'='*94}\n{title}\n{'='*94}")
        for tag, desc, value, ok, note in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {tag}  {desc}")
            print(f"        {value}" + (f"   ({note})" if note else ""))
        print(f"\n{len(self.rows) - self.failed}/{len(self.rows)} passed")


def batch_dup_rate(labels, seed=42):
    idx = list(range(len(labels)))
    random.Random(seed).shuffle(idx)
    bad = tot = 0
    for s in range(0, len(idx) - BATCH + 1, BATCH):
        b = [labels[i] for i in idx[s:s + BATCH]]
        tot += 1
        if len(set(b)) < len(b):
            bad += 1
    return bad, tot


def check_new(r: Report):
    ev = jload("data/split/eval.json")
    s1 = jload("data/split/stage1_train.json")
    corp = {k: {x["url"] for x in jload(f"data/corpus/{k}s.json")} for k in ("trigger", "action")}
    texts = {k: {x["text"] for x in jload(f"data/corpus/{k}s.json")} for k in ("trigger", "action")}

    s1q, s1nq, s1au = ({x[k] for x in s1} for k in ("query", "nq", "applet_url"))
    hits = sum(1 for x in ev if x["query"] in s1q or x["nq"] in s1nq or x["applet_url"] in s1au)
    r.add("A1", "no eval record shares query / nq / applet_url with stage1",
          f"{hits}/{len(ev)}", hits == 0)

    s1sets = [content(x["nq"]) for x in s1]
    inv = defaultdict(list)
    for i, cs in enumerate(s1sets):
        for t in cs: inv[t].append(i)
    worst = 0.0; n_over = 0
    for x in ev:
        cs = content(x["nq"])
        if not cs: continue
        cand = Counter()
        for t in cs:
            for i in inv.get(t, ()): cand[i] += 1
        for i, _ in cand.most_common(30):
            o = s1sets[i]
            j = len(cs & o) / len(cs | o) if (cs | o) else 0.0
            worst = max(worst, j)
            if j >= 0.9: n_over += 1; break
    r.add("A2", "no eval group is a >=0.9 token-set paraphrase of a stage1 group",
          f"{n_over}/{len(ev)} over threshold, max jaccard {worst:.3f}", n_over == 0)

    miss = sum(1 for x in ev for k in ("trigger", "action") if x[f"{k}_url"] not in corp[k])
    r.add("A3", "every eval gold url has a document in the index",
          f"{miss}/{2*len(ev)} gold sides missing", miss == 0)

    bad = 0; tot = 0
    for k in ("trigger", "action"):
        for sp in ("stage1_train", "rerank_train"):
            rows = jload(f"data/pairs/{k}_{sp}.json")
            tot += len(rows)
            bad += sum(1 for x in rows if x["positive"] not in texts[k])
    r.add("A4", "every training positive is a string that exists in the index",
          f"{tot-bad}/{tot} positives indexed", bad == 0)

    tp = Path("data/pairs/trigger_mined.json")
    if tp.exists():
        anchors = {x["anchor"] for x in jload(tp)}
        evq = {x["query"] for x in ev}
        n = len(anchors & evq)
        r.add("A5", "no mined-triplet anchor is an eval query", f"{n} overlapping anchors", n == 0)
    else:
        r.add("A5", "no mined-triplet anchor is an eval query",
              "no mined triplets built yet", True, "vacuous until mining runs")

    cfg = Path("train_config.json")
    sampler = json.loads(cfg.read_text()).get("batch_sampler") if cfg.exists() else None
    rates = {}
    for k in ("trigger", "action"):
        rows = jload(f"data/pairs/{k}_stage1_train.json")
        bad_b, tot_b = batch_dup_rate([x["label_url"] for x in rows])
        rates[k] = f"{bad_b}/{tot_b} = {bad_b/tot_b:.1%}"
    ok = sampler == "NO_DUPLICATES"
    r.add("A6", f"trainer uses NO_DUPLICATES (default sampler would collide at bs={BATCH})",
          f"batch_sampler={sampler!r}; default-sampler duplicate-label batches "
          f"trigger {rates['trigger']}, action {rates['action']}", ok,
          "" if ok else "set batch_sampler in train_config.json")


def check_old(r: Report):
    train = jload("data/server_copy/train_applets.json")
    trq = {str(a.get("query", "")) for a in train}
    trnq = {nq(a.get("query")) for a in train}
    tot = hit = 0
    for f in ("gold", "noisy", "oneshot"):
        rows = jload(f"data/test/{f}.json")
        tot += len(rows)
        hit += sum(1 for x in rows if str(x.get("query", "")) in trq or nq(x.get("query")) in trnq)
    r.add("A1", "no eval record shares query / nq / applet_url with stage1",
          f"{hit}/{tot} = {hit/tot:.1%} of data/test/* is in train_applets.json", hit == 0)

    ev = jload("data/server_copy/clean_split/eval.json")
    s1 = jload("data/server_copy/clean_split/stage1_train.json")
    s1sets = [content(nq(x.get("query"))) for x in s1]
    inv = defaultdict(list)
    for i, cs in enumerate(s1sets):
        for t in cs: inv[t].append(i)
    n_over = 0; worst = 0.0
    for x in ev:
        cs = content(nq(x.get("query")))
        if not cs: continue
        cand = Counter()
        for t in cs:
            for i in inv.get(t, ()): cand[i] += 1
        for i, _ in cand.most_common(30):
            o = s1sets[i]
            j = len(cs & o) / len(cs | o) if (cs | o) else 0.0
            worst = max(worst, j)
            if j >= 0.9: n_over += 1; break
    r.add("A2", "no eval group is a >=0.9 token-set paraphrase of a stage1 group",
          f"{n_over}/{len(ev)} over threshold, max jaccard {worst:.3f}", n_over == 0)

    # reconstruct the old (service_name, category) first-wins dedup to find survivors
    raw = jload("data/iftttt_dataset_full_trigger_action.json")
    survivor = {"trigger": {}, "action": {}}
    for s in raw:
        if not isinstance(s, dict): continue
        aps = s.get("applets")
        if not isinstance(aps, list): continue
        for ap in aps:
            if not isinstance(ap, dict): continue
            cats = {"trigger": ap.get("trigger_categories") or [],
                    "action": ap.get("action_categories") or []}
            for c in ap.get("components") or []:
                if not isinstance(c, dict): continue
                try: _, kind, _ = parse_url(c.get("url"))
                except ValueError: continue
                if kind not in ("trigger", "action"): continue
                cat = (cats[kind] or [""])[0]
                survivor[kind].setdefault((str(c.get("service_name") or "").strip().lower(),
                                          str(cat).strip()), str(c.get("url")).strip())
    reachable = {k: set(v.values()) for k, v in survivor.items()}
    miss = sum(1 for x in ev for k in ("trigger", "action")
               if str(x[k].get("url") or "").strip() not in reachable[k])
    r.add("A3", "every eval gold url has a document in the index",
          f"{miss}/{2*len(ev)} gold sides have no document "
          f"(unreachable functions: trigger {1985-len(reachable['trigger'])}, "
          f"action {1520-len(reachable['action'])})", miss == 0)

    src = "".join(open("rag/indexer.py").readlines()[33:176])
    ns = {"Dict": dict, "Any": object, "List": list, "Path": Path, "re": re, "json": json}
    exec(compile(src, "rag/indexer.py", "exec"), ns)
    tot = bad = 0
    for kind, key in (("trigger", "trigger"), ("action", "action")):
        ex = ns[f"extract_{kind}_text"]
        indexed = {ex(x).strip() for x in jload(f"data/{kind}s_rag.json")}
        pos = [ex(a[key]).strip() for a in train if a.get(key)]
        tot += len(pos); bad += sum(1 for p in pos if p not in indexed)
    r.add("A4", "every training positive is a string that exists in the index",
          f"{tot-bad}/{tot} positives indexed", bad == 0)

    evq = {str(x.get("query", "")) for x in ev}
    n = 0; tt = 0
    for k in ("trigger", "action"):
        rows = jload(f"data/server_copy/triplets_{k}.json")
        tt += len(rows)
        n += len({x["anchor"] for x in rows} & evq)
    r.add("A5", "no mined-triplet anchor is an eval query",
          f"{n} distinct triplet anchors are clean_split eval queries (of {tt} rows)", n == 0)

    src2 = "".join(open("rag/indexer.py").readlines()[33:176])
    ns2 = {"Dict": dict, "Any": object, "List": list, "Path": Path, "re": re, "json": json}
    exec(compile(src2, "rag/indexer.py", "exec"), ns2)
    rates = {}
    for kind in ("trigger", "action"):
        ex = ns2[f"extract_{kind}_text"]
        labels = [ex(a[kind]) for a in train if a.get(kind)]
        b, t = batch_dup_rate(labels)
        rates[kind] = f"{b}/{t} = {b/t:.1%}"
    r.add("A6", f"trainer uses NO_DUPLICATES (default sampler would collide at bs={BATCH})",
          f"batch_sampler=None (train/train_trigger.py never sets one); duplicate-label "
          f"batches trigger {rates['trigger']}, action {rates['action']}", False)


if __name__ == "__main__":
    old = "--old" in sys.argv
    rep = Report()
    (check_old if old else check_new)(rep)
    rep.show("OLD PIPELINE (must fail)" if old else "REBUILT PIPELINE")
    sys.exit(1 if (rep.failed and not old) else 0)
