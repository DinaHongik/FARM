"""The realistic false-positive test: confusion WITHIN the retrieved candidate set.

Random pairs from a 1258-function inventory are mostly obviously different. The
candidates the system actually chooses between are, by construction, similar.
If service_matches() accepts a non-gold candidate as a match for the gold, then
a wrong selection is scored as correct.
"""
import json, sys
sys.path.insert(0, ".")
from eval.ragas_metrics import service_matches, normalize_service_name

ps = json.load(open("results/e2e_eval.json"))["per_sample"][:300]
tot = acc = 0
samples_with_fp = 0
ex = []
for s in ps:
    rp = s.get("raw_prediction") or {}
    ref = s.get("reference") or {}
    for key, gk in (("trigger_candidates", "trigger"), ("action_candidates", "action")):
        gold = ref.get(gk)
        if not gold: continue
        gn = normalize_service_name(gold)
        hit = False
        for c in rp.get(key) or []:
            nm = c.get("service_name")
            if not nm or normalize_service_name(nm) == gn: continue
            tot += 1
            if service_matches(nm, gold):
                acc += 1; hit = True
                if len(ex) < 14: ex.append((nm, gold))
        if hit: samples_with_fp += 1

print(f"non-gold candidates tested against their gold : {tot}")
print(f"ACCEPTED as a match (false positives)         : {acc} = {acc/tot:.3f}")
print(f"samples where >=1 wrong candidate is accepted : {samples_with_fp}/{len(ps)} = {samples_with_fp/len(ps):.3f}")
print("\nwrong candidate accepted as the gold:")
for a, b in ex:
    print(f"  {a[:46]:<48} == {b[:46]}")
