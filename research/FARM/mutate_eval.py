"""Mutation test the PRODUCTION evaluation path.

service_matches() decides whether a predicted function counts as correct. If it
accepts functions that are plainly wrong, every accuracy in the paper is inflated.
We measure its false-positive rate directly: feed it (wrong_function, gold) pairs
drawn from the real inventory and count how many it accepts.
"""
import json, random, sys
sys.path.insert(0, ".")
from eval.ragas_metrics import service_matches, word_overlap_score, normalize_service_name

random.seed(0)
ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]

# real inventory of function names, from the retrieved candidates
inv = set()
for s in ps:
    rp = s.get("raw_prediction") or {}
    for k in ("trigger_candidates", "action_candidates"):
        for c in rp.get(k) or []:
            n = c.get("service_name")
            if n: inv.add(n)
inv = sorted(inv)
print(f"inventory of distinct function names: {len(inv)}")

golds = [s["reference"]["trigger"] for s in ps if (s.get("reference") or {}).get("trigger")]
print(f"gold trigger names: {len(golds)}\n")

# TRUE POSITIVE control: gold vs itself must always match
tp = sum(1 for g in golds if service_matches(g, g))
print(f"control  gold vs itself accepted      : {tp}/{len(golds)} = {tp/len(golds):.3f}   (must be 1.000)")

# FALSE POSITIVE: a RANDOM DIFFERENT function vs gold must almost never match
trials = 2000
fp = 0; examples = []
for _ in range(trials):
    g = random.choice(golds)
    w = random.choice(inv)
    if normalize_service_name(w) == normalize_service_name(g): continue
    if service_matches(w, g):
        fp += 1
        if len(examples) < 12: examples.append((w, g))
print(f"MUTATION random wrong function accepted: {fp}/{trials} = {fp/trials:.3f}   (should be ~0)")

print("\nexamples of accepted-but-wrong pairs (predicted -> gold):")
for w, g in examples:
    wo = word_overlap_score(normalize_service_name(w), normalize_service_name(g))
    print(f"  {w[:44]:<46} -> {g[:44]:<46} overlap={wo:.2f}")

# which strategy is doing the damage
print("\nstrategy attribution over the accepted-but-wrong pairs:")
sub = ov = 0
for w, g in examples:
    pn, en = normalize_service_name(w), normalize_service_name(g)
    if pn in en or en in pn: sub += 1
    elif word_overlap_score(pn, en) >= 0.6: ov += 1
print(f"  substring match : {sub}/{len(examples)}")
print(f"  word overlap>=.6: {ov}/{len(examples)}")

# binding_accuracy vacuity
nb = sum(1 for s in ps if (s.get("reference") or {}).get("bindings"))
print(f"\neval records carrying gold 'bindings': {nb}/{len(ps)}")
print("  -> binding_accuracy returns the literal 1.0 for every record without them")
