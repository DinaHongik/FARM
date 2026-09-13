"""Build the train/eval split, grouped so it cannot leak.

The shipped data/test/*.json is not a held-out set: 500/500 gold rows are in the
full applet pool and only 76 are in the seed-42 holdout, so the repo's own
check_leakage.py reports 94%. The cause is that the scrape lists each applet under
every related service (one applet appears under 19 services, 20.7% of records are
repeats) and split_data.py shuffles instances rather than groups.

Order here: dedup by applet_url, then group by normalised query, then split GROUPS,
then drop eval groups that are token-set paraphrases of a training group.
"""
from __future__ import annotations

import json, random, re, sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from farm.render import parse_url

RAW = Path("data/iftttt_dataset_full_trigger_action.json")
OUT = Path("data/split")
SEED = 42
JACCARD_MAX = 0.9
STOP = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "if", "when",
        "your", "my", "with", "from", "at", "is", "it", "this", "that", "be", "by"}


def nq(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s or "").lower())).strip()


def content(s: str) -> frozenset:
    return frozenset(t for t in nq(s).split() if len(t) > 2 and t not in STOP)


def main() -> None:
    services = json.loads(RAW.read_text())
    pool: dict[str, dict] = {}
    multi_action = with_query = no_pair = no_desc = 0

    for s in services:
        if not isinstance(s, dict):
            continue
        aps = s.get("applets")
        if not isinstance(aps, list):
            continue
        for ap in aps:
            if not isinstance(ap, dict):
                continue
            comps = ap.get("components")
            if not isinstance(comps, list):
                continue
            url = str(ap.get("applet_url") or "").strip()
            if not url or url in pool:
                continue                      # dedup by applet_url, first wins
            desc = str(ap.get("description") or "").strip()
            trig = act = None
            n_and = n_with = 0
            for c in comps:
                if not isinstance(c, dict):
                    continue
                try:
                    _, kind, _ = parse_url(c.get("url"))
                except ValueError:
                    continue
                lab = str(c.get("label") or "")
                if kind == "trigger" and lab == "If" and trig is None:
                    trig = c
                elif kind == "action" and lab == "Then" and act is None:
                    act = c
                elif lab == "And":
                    n_and += 1
                elif lab == "With":
                    n_with += 1
            if not desc:
                no_desc += 1
                continue
            if not (trig and act):
                no_pair += 1
                continue
            if n_with:
                with_query += 1
            if n_and:
                multi_action += 1              # excluded from the headline task
                continue
            tch, _, _ = parse_url(trig["url"])
            ach, _, _ = parse_url(act["url"])
            pool[url] = {
                "applet_url": url, "query": desc, "nq": nq(desc),
                "trigger_url": str(trig["url"]).strip(),
                "action_url": str(act["url"]).strip(),
                "trigger_channel": tch, "action_channel": ach,
                "user_count": ap.get("user_count", "0"),
            }

    rows = list(pool.values())
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["nq"]].append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    n = len(keys)
    n_eval = round(n * 0.10)
    n_rr = round(n * 0.10)
    eval_k, rr_k, s1_k = keys[:n_eval], keys[n_eval:n_eval + n_rr], keys[n_eval + n_rr:]

    # paraphrase filter: drop eval groups whose content-token set is >= JACCARD_MAX
    # similar to any stage1 group. Inverted index so this is not 1159 x 9277 set ops.
    s1_sets = [(k, content(k)) for k in s1_k]
    inv: dict[str, list[int]] = defaultdict(list)
    for i, (_, cs) in enumerate(s1_sets):
        for t in cs:
            inv[t].append(i)
    dropped = []
    kept_eval = []
    for k in eval_k:
        cs = content(k)
        if not cs:
            kept_eval.append(k)
            continue
        cand = Counter()
        for t in cs:
            for i in inv.get(t, ()):
                cand[i] += 1
        hit = False
        for i, _ in cand.most_common(50):
            o = s1_sets[i][1]
            j = len(cs & o) / len(cs | o) if (cs | o) else 0.0
            if j >= JACCARD_MAX:
                hit = True
                break
        (dropped if hit else kept_eval).append(k)

    def flat(ks, one_per_group):
        out = []
        for k in ks:
            g = sorted(groups[k], key=lambda r: r["applet_url"])
            out.extend(g[:1] if one_per_group else g)
        return out

    ev = flat(kept_eval, True)
    rr = flat(rr_k, True)
    s1 = flat(s1_k, False)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("eval", ev), ("rerank_train", rr), ("stage1_train", s1)):
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))

    summary = {
        "seed": SEED, "jaccard_max": JACCARD_MAX,
        "raw_applets_deduped_by_applet_url": len(rows) + multi_action,
        "excluded_multi_action_applets": multi_action,
        "applets_containing_a_query_component_kept": with_query,
        "dropped_no_pair": no_pair, "dropped_no_description": no_desc,
        "pool_records": len(rows), "nq_groups": n,
        "eval_groups_before_filter": len(eval_k),
        "eval_groups_dropped_as_paraphrase": len(dropped),
        "records": {"eval": len(ev), "rerank_train": len(rr), "stage1_train": len(s1)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
