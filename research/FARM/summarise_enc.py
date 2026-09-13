"""Aggregate every finished encoder arm into one table.

Runs unattended so the sweep is readable the moment the last eval lands, without a live
session. Reads only results/rebuilt/*.json - eval_stage1.py's --out default - and
recomputes nothing.
"""
import json, math
from pathlib import Path

RES = Path("results/rebuilt")
BASE = "hardneg_s42"          # 4 mined hard negatives, hardness_strength 5, scale 20
LEVER = {
    "enc_n8":     "8 hard negatives (vs 4)",
    "enc_frz6":   "freeze embeddings + bottom 6 layers",
    "enc_docdoc": "directions = query_to_doc + doc_to_doc",
    "enc_sc50":   "scale 50  (temperature 0.02)",
    "enc_sc10":   "scale 10  (temperature 0.10)",
    "enc_gist":   "CachedGISTEmbedLoss, MiniLM-L6 guide",
    "enc_hs10":   "hardness_strength 10 (vs 5)",
    "enc_aug":    "+76.8% anchors (paraphrase augmentation)",
    BASE:         "BASELINE: 4 hard negatives, strength 5, scale 20",
    "rebuilt_dedup_s42": "reference: no hard negatives",
    "baseline_untrained": "reference: untrained EmbeddingGemma",
}
rows = []
for f in sorted(RES.glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError:
        continue
    if "joint" in d and "sides" in d:
        rows.append((f.stem, d))

by = dict(rows)
b = by[BASE]["joint"]["R@1"] if BASE in by else None
n = by[BASE]["n_eval"] if BASE in by else 1131
se = math.sqrt(b * (1 - b) / n) if b else None

out = ["# Encoder sweep", ""]
if se:
    out += [f"Baseline `{BASE}` = {b:.3f} at n={n}. Binomial SE on one arm = {se:.4f}, so two SE "
            f"is {2*se*100:.1f} pp.",
            "Seed noise measured separately on the no-hard-negative config was 0.484 / 0.487 / 0.489,",
            "a 0.5 pp spread. Treat anything under ~1 pp as seed noise and anything under ~3 pp as",
            "unresolved until paired McNemar is run against the baseline's per-query hits.", ""]
out += ["| arm | lever | trigger R@1 | action R@1 | joint R@1 | delta pp |",
        "|---|---|---|---|---|---|"]
for tag, d in sorted(rows, key=lambda r: -r[1]["joint"]["R@1"]):
    j = d["joint"]["R@1"]
    delta = "baseline" if tag == BASE else (f"{(j - b) * 100:+.1f}" if b is not None else "")
    out.append(f"| `{tag}` | {LEVER.get(tag, '')} | {d['sides']['trigger']['R@1']:.3f} | "
               f"{d['sides']['action']['R@1']:.3f} | **{j:.3f}** | {delta} |")

missing = [k for k in LEVER if k.startswith("enc_") and k not in by]
out += ["", f"still running: {', '.join(sorted(missing)) if missing else 'none - sweep complete'}", ""]

arms = [(t, d) for t, d in rows if t.startswith("enc_")]
if arms:
    top, td = max(arms, key=lambda r: r[1]["joint"]["R@1"])
    out += [f"## Strata, leading arm `{top}` vs baseline", "",
            "| stratum | n | leading arm | baseline |", "|---|---|---|---|"]
    for k, v in td.get("strata", {}).items():
        bv = by.get(BASE, {}).get("strata", {}).get(k, {}).get("joint_R@1")
        f = lambda x: "-" if x is None else f"{x:.3f}"
        out.append(f"| {k} | {v['n']} | {f(v['joint_R@1'])} | {f(bv)} |")
Path("ENCODER_SWEEP.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
