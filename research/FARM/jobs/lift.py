#!/usr/bin/env python3
"""The headline comparison: does hard-negative mining create cross-schema lift?

LIFT OVER INDEPENDENCE
    lift = joint_R@k  -  (trigger_R@k x action_R@k)

If a retrieval system picks trigger and action independently, joint accuracy is
exactly the product of the marginals and lift is 0. Any positive lift means the
two retrievals are positively dependent - the system has learned something about
which pairs go together.

Measured across every FARM config so far, lift is indistinguishable from zero
(-0.046 to +0.040), which is what you would predict from training that emits only
(anchor, positive) with random in-batch negatives.

This compares Run B (random negatives) against Run C (mined hard negatives).
Reads whatever result files exist; missing ones are skipped.
"""
import json
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")

FILES = [
    ("published (leaky split)", "results/ablation_comprehensive.json"),
    ("RUN B  random negatives", "results/RUNB_ablation.json"),
    ("RUN C  mined negatives", "results/RUNC_ablation.json"),
]


def rows_of(p):
    d = json.load(open(p, encoding="utf-8"))
    return d.get("results", d if isinstance(d, list) else [])


def main():
    print("=" * 84)
    print("LIFT OVER INDEPENDENCE      lift = joint - (trigger x action)")
    print("=" * 84)
    print(f"{'run':<26}{'method':<16}{'k':>3}{'trig':>8}{'act':>8}{'prod':>8}{'joint':>8}{'lift':>9}")
    print("-" * 84)

    seen = {}
    for label, rel in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"{label:<26}(not present yet: {rel})")
            continue
        for r in rows_of(p):
            m = r.get("method")
            if m != "full_finetune":      # the controlled comparison
                continue
            for k in ("R@1", "R@5"):
                t = r["trigger_metrics"]["service_level"][k]
                a = r["action_metrics"]["service_level"][k]
                j = r["joint_metrics"]["service_level"][k]
                lift = j - t * a
                seen.setdefault(label, {})[k] = lift
                print(f"{label:<26}{m:<16}{k[-1]:>3}{t:>8.3f}{a:>8.3f}{t*a:>8.3f}"
                      f"{j:>8.3f}{lift:>+9.3f}")
        print("-" * 84)

    b, c = seen.get("RUN B  random negatives"), seen.get("RUN C  mined negatives")
    if b and c:
        print("\nVERDICT (full_finetune, deduped split)")
        for k in ("R@1", "R@5"):
            if k in b and k in c:
                d = c[k] - b[k]
                verdict = ("mined negatives INCREASED dependence" if d > 0.02 else
                           "mined negatives DECREASED dependence" if d < -0.02 else
                           "no significant change (inside n=100 noise)")
                print(f"  {k}: lift {b[k]:+.3f} -> {c[k]:+.3f}   delta {d:+.3f}   {verdict}")
        print("\nAt n=100 a lift difference under ~0.05 is inside sampling noise.")
        print("A negative result here is publishable: it would show that topical hard")
        print("negatives do not teach cross-schema compatibility, motivating an")
        print("explicitly schema-conditioned objective.")
    else:
        print("\n(run 'trainC' to produce the Run C side of this comparison)")


if __name__ == "__main__":
    main()
