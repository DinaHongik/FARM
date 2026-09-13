"""Build the retrieval corpus keyed on the component url.

The previous corpus (data/{triggers,actions}_rag.json) was keyed on
(service_name, applet category) with the first-scraped occurrence winning. That
collapsed 263 trigger and 308 action functions out of existence: "New episode" is
91 distinct urls across 91 podcast channels, and only one of them was indexed.
Under url identity nothing is unreachable, and the name collisions become hard
negatives instead of grading ambiguity.

Writes data/corpus/{triggers,actions,queries}.json plus manifest.json.
"""
from __future__ import annotations

import json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from farm.render import parse_url, render, pretty_channel

RAW = Path("data/iftttt_dataset_full_trigger_action.json")
OUT = Path("data/corpus")


def main() -> None:
    services = json.loads(RAW.read_text())

    # channel slug -> display name and category, taken from the service record
    chan_display: dict[str, str] = {}
    chan_cat: dict[str, str] = {}
    for s in services:
        if not isinstance(s, dict):
            continue
        url = str(s.get("service_url") or "")
        slug = url.rstrip("/").rsplit("/", 1)[-1].lower() if "ifttt.com/" in url else ""
        if not slug:
            continue
        chan_display.setdefault(slug, str(s.get("service_name") or "").strip())
        cats = s.get("categories") or []
        if isinstance(cats, list) and cats:
            chan_cat.setdefault(slug, str(cats[0]).strip())

    # url -> merged record
    recs: dict[str, dict] = {}
    occ: dict[str, int] = Counter()
    descs: dict[str, Counter] = defaultdict(Counter)
    cat_votes: dict[str, Counter] = defaultdict(Counter)
    label_kind_mismatch = 0
    no_api = 0
    bad_url = 0

    for s in services:
        if not isinstance(s, dict):
            continue
        aps = s.get("applets")
        if not isinstance(aps, list):
            continue
        for ap in aps:
            if not isinstance(ap, dict):
                continue
            tcats = ap.get("trigger_categories") or []
            acats = ap.get("action_categories") or []
            comps = ap.get("components")
            if not isinstance(comps, list):
                continue
            for c in comps:
                if not isinstance(c, dict):
                    continue
                try:
                    channel, kind, fn = parse_url(c.get("url"))
                except ValueError:
                    bad_url += 1
                    continue
                url = str(c.get("url")).strip()
                label = str(c.get("label") or "")
                if (label == "If") != (kind == "trigger") and label in ("If", "Then"):
                    label_kind_mismatch += 1
                api = c.get("api_info")
                if not isinstance(api, dict):
                    api = {}
                    no_api += 1
                occ[url] += 1
                d = str(c.get("description") or "").strip()
                if d:
                    descs[url][d] += 1
                votes = tcats if kind == "trigger" else acats
                for v in (votes if isinstance(votes, list) else []):
                    cat_votes[url][str(v).strip()] += 1

                r = recs.get(url)
                if r is None:
                    recs[url] = {
                        "url": url, "kind": kind, "channel": channel,
                        "function": fn,
                        "service_name": str(c.get("service_name") or "").strip(),
                        "title": str(c.get("title") or "").strip(),
                        "api_info": json.loads(json.dumps(api)),
                    }
                else:
                    # union the field and ingredient dicts across occurrences
                    for key in ("Trigger fields", "Action fields", "Ingredients"):
                        new = api.get(key)
                        if not isinstance(new, dict):
                            continue
                        cur = r["api_info"].get(key)
                        if not isinstance(cur, dict):
                            r["api_info"][key] = json.loads(json.dumps(new))
                        else:
                            for k, v in new.items():
                                cur.setdefault(k, v)

    # resolve description and category deterministically
    for url, r in recs.items():
        if descs[url]:
            r["description"] = descs[url].most_common(1)[0][0]
        else:
            r["description"] = ""
        cat = chan_cat.get(r["channel"], "")
        if not cat and cat_votes[url]:
            cat = cat_votes[url].most_common(1)[0][0]
        r["category"] = cat
        r["channel_display"] = chan_display.get(r["channel"], pretty_channel(r["channel"]))
        r["n_occurrences"] = occ[url]
        r["text"] = render(r)

    OUT.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list] = defaultdict(list)
    for r in recs.values():
        by_kind[r["kind"]].append(r)
    for kind, rows in by_kind.items():
        rows.sort(key=lambda x: x["url"])
        (OUT / f"{kind}s.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    man = {
        "source": str(RAW),
        "counts": {k: len(v) for k, v in sorted(by_kind.items())},
        "total_component_occurrences": sum(occ.values()),
        "unparseable_urls": bad_url,
        "components_without_api_info": no_api,
        "label_vs_url_kind_mismatch": label_kind_mismatch,
        "channels": {k: len({r["channel"] for r in v}) for k, v in sorted(by_kind.items())},
        "channel_coverage": {k: sum(1 for r in v if r["channel"]) / len(v) for k, v in sorted(by_kind.items())},
        "category_coverage": {k: sum(1 for r in v if r["category"]) / len(v) for k, v in sorted(by_kind.items())},
        "urls_merged_from_multiple_occurrences": {
            k: sum(1 for r in v if r["n_occurrences"] > 1) for k, v in sorted(by_kind.items())},
        "name_collisions": {
            k: sum(c for c in Counter(r["service_name"].lower() for r in v).values() if c > 1)
            for k, v in sorted(by_kind.items())},
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    for kind in ("trigger", "action"):
        if by_kind.get(kind):
            print(f"\nsample {kind}:\n" + by_kind[kind][0]["text"])


if __name__ == "__main__":
    main()
