"""Compare every arm on the SAME samples, not on each arm's own denominator.

BASE is only measurable where its committed trigger requires no fields (the
stored pipeline has no trigger-binding stage), so a comparison of BASE's mean
against arms' means over all 299 is not like-for-like.
"""
import json, glob

def load(tag):
    import re as _re
    cands = glob.glob(f"results/ARMS_{tag}_*.json")
    # sort by the sample count in the filename, not lexicographically
    f = max(cands, key=lambda x: int(_re.search(r"_(\d+)\.json$", x).group(1)))
    return f, json.load(open(f))

for tag in ("gemma4_31b", "gpt-oss_120b"):
    f, res = load(tag)
    common = [r for r in res if r["BASE"].get("both") is not None]
    print(f"\n=== {tag}  ({f})  n_all={len(res)}  n_BASE_measurable={len(common)} ===")
    print(f"{'arm':<7}{'BOTH':>8}{'BOTHv':>8}{'CORRECT':>9}{'CORRv':>8}{'stat_grnd':>11}   (restricted to the shared subset)")
    print("-"*62)
    for arm in ("A0", "A1", "A2", "A3", "BASE"):
        def m(k):
            v = [r[arm][k] for r in common if r[arm].get(k) is not None]
            return sum(v)/len(v) if v else None
        so = sum(r[arm].get("stat_ok", 0) or 0 for r in common)
        st = sum(r[arm].get("stats", 0) or 0 for r in common)
        def f2(x): return "  n/a  " if x is None else f"{x:>7.3f}"
        sg = (so/st) if st else None
        print(f"{arm:<7}{f2(m('both'))}{f2(m('bothv'))}{f2(m('correct'))}{f2(m('correctv'))}{f2(sg):>11}")
