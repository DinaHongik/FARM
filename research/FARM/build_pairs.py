"""Build (anchor, positive) training pairs.

The positive is looked up in the corpus BY URL and rendered with the same function
the index uses, so it is byte-identical to the document the retriever must return.
The old builder rendered the raw applet component instead, which has no 'category'
key, and 0 of 14,180 positives matched any indexed document.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sys as _sys
AUGMENT = "--augment" in _sys.argv
SPLIT = Path("data/split")
CORPUS = Path("data/corpus")
OUT = Path("data/pairs")


def load_corpus(kind: str) -> dict[str, dict]:
    return {r["url"]: r for r in json.loads((CORPUS / f"{kind}s.json").read_text())}


def load_paraphrases() -> dict:
    """applet_url -> a SECOND natural-language phrasing of the same applet.

    Every raw applet carries `additional_description`, and in 78.3% of training
    applets it differs from `description`, which is the only anchor training has ever
    used. Both describe the same applet and both come from the same applet_url, so
    they land in the same split group and add no leakage. `related_services` is
    deliberately NOT used: it names the trigger and action outright, so it is the
    label, and it is not available at inference time.
    """
    raw = json.loads(Path("data/iftttt_dataset_full_trigger_action.json").read_text())
    out = {}
    for svc in raw:
        if not isinstance(svc, dict):
            continue
        for ap in svc.get("applets") or []:
            if not isinstance(ap, dict):
                continue
            u = str(ap.get("applet_url") or "").strip()
            d = str(ap.get("description") or "").strip()
            ad = str(ap.get("additional_description") or "").strip()
            if u and ad and ad != d and len(ad) >= 10:
                out.setdefault(u, ad)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    para = load_paraphrases() if AUGMENT else {}
    for kind in ("trigger", "action"):
        corpus = load_corpus(kind)
        indexed = {r["text"] for r in corpus.values()}
        for split in ("stage1_train", "rerank_train", "eval"):
            recs = json.loads((SPLIT / f"{split}.json").read_text())
            rows, missing = [], 0
            for r in recs:
                gold = corpus.get(r[f"{kind}_url"])
                if gold is None:
                    missing += 1
                    continue
                rows.append({"anchor": r["query"], "positive": gold["text"],
                             "label_url": gold["url"], "applet_url": r["applet_url"],
                             "nq": r["nq"]})
                # second phrasing of the SAME applet, training splits only
                if split == "stage1_train" and r["applet_url"] in para:
                    rows.append({"anchor": para[r["applet_url"]], "positive": gold["text"],
                                 "label_url": gold["url"], "applet_url": r["applet_url"],
                                 "nq": r["nq"], "paraphrase": True})
            # A4 as a hard precondition, not a report
            bad = [x for x in rows if x["positive"] not in indexed]
            assert not bad, f"{kind}/{split}: {len(bad)} positives are not indexed documents"
            assert missing == 0, f"{kind}/{split}: {missing} gold urls absent from the corpus"
            suffix = "_aug" if (AUGMENT and split == "stage1_train") else ""
            (OUT / f"{kind}_{split}{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            labels = {x["label_url"] for x in rows}
            npara = sum(1 for x in rows if x.get("paraphrase"))
            print(f"{kind:8} {split:13} rows={len(rows):5} (+{npara} paraphrase) "
                  f"distinct labels={len(labels):5} positives indexed={len(rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
