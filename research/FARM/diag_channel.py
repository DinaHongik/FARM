"""Two questions the reranker work never asked.

1. The indexed document only gets a channel string if the function happens to
   expose an ingredient with a Filter code. How many documents have no channel?
2. Candidates are matched to gold on service_name -- the bare FUNCTION name.
   The gold's real identity is its url (ifttt.com/<channel>/<kind>/<function>).
   How much accuracy is being credited to a channel-blind match?
"""
import json, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from rag.indexer import extract_trigger_text, extract_action_text, extract_channel_from_filter_code

TOKRE = re.compile(r"[a-z0-9]+")
def toks(s): return TOKRE.findall(str(s or "").lower())
def nz(x): return str(x or "").strip().lower()

def doc_channel(rec):
    ings = (rec.get("api_info") or {}).get("Ingredients") or {}
    if isinstance(ings, dict):
        for _, info in ings.items():
            if isinstance(info, dict) and "Filter code" in info:
                return nz(extract_channel_from_filter_code(info["Filter code"]))
    return ""

def gold_channel(g):
    m = re.search(r"ifttt\.com/([^/]+)/", str(g.get("url") or ""))
    return nz(m.group(1)) if m else ""

ev = json.load(open("data/clean_split/eval.json"))
for path, side, extract in [("data/triggers_rag.json", "trigger", extract_trigger_text),
                            ("data/actions_rag.json", "action", extract_action_text)]:
    corpus = json.load(open(path))
    print(f"\n================ {side} ================")

    # Q1: documents with no channel string
    chans = [doc_channel(r) for r in corpus]
    empty = sum(1 for c in chans if not c)
    print(f"docs={len(corpus)}  with NO channel in text: {empty} ({empty/len(corpus):.1%})")
    texts = [extract(r) for r in corpus]
    # sanity: is the channel token actually present in the rendered text?
    present = sum(1 for c, t in zip(chans, texts) if c and c in nz(t))
    print(f"  of those that have one, channel token present in rendered text: {present}/{len(corpus)-empty}")

    # Q2: channel-blind matching
    by_name = defaultdict(list)
    for i, r in enumerate(corpus):
        by_name[nz(r.get("service_name"))].append(i)

    gold = [r[side] for r in ev]
    hit_name = sum(1 for g in gold if nz(g.get("service_name")) in by_name)
    print(f"gold service_name present in corpus: {hit_name}/{len(ev)} ({hit_name/len(ev):.3f})")

    # how many corpus rows does a gold name match -> that many rows are all scored 'correct'
    multi = [len(by_name[nz(g.get("service_name"))]) for g in gold if nz(g.get("service_name")) in by_name]
    amb = sum(1 for m in multi if m > 1)
    print(f"  gold names matching >1 corpus row (channel-blind credit): {amb}/{len(multi)} ({amb/len(multi):.3f})")
    print(f"  mean rows per gold name: {sum(multi)/len(multi):.2f}  max: {max(multi)}")

    # of the ambiguous ones, does the CHANNEL actually disagree?
    wrong_chan = 0
    for g in gold:
        n = nz(g.get("service_name"))
        if n not in by_name or len(by_name[n]) < 2:
            continue
        gc = gold_channel(g)
        cand_chans = {chans[i] for i in by_name[n]}
        if gc and gc not in cand_chans:
            wrong_chan += 1
    print(f"  ambiguous AND no candidate row carries the gold channel: {wrong_chan}")

    # Q3: does the query literally name the gold channel?
    named = tot = 0
    for r in ev:
        gc = gold_channel(r[side])
        if not gc:
            continue
        tot += 1
        qt = set(toks(r["query"]))
        if set(toks(gc)) <= qt:
            named += 1
    print(f"gold channel literally named in the query: {named}/{tot} ({named/tot:.3f})")

    print("sample rendered doc:\n  " + texts[0][:260].replace("\n", "\n  "))
