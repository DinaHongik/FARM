"""What is the true label space? The raw dataset carries a url per component
(ifttt.com/<channel>/<kind>/<function>); the indexed corpora carry only
service_name. Measure the gap -- and count malformed records while we are here."""
import json, re
from collections import Counter

def nz(x): return str(x or "").strip().lower()
def chan_of(u):
    m = re.search(r"ifttt\.com/([^/]+)/", str(u or ""))
    return nz(m.group(1)) if m else ""

raw = json.load(open("data/iftttt_dataset_full_trigger_action.json"))
sides = {"trigger": {}, "action": {}}
n_comp = bad_ap = bad_c = no_url = other_label = 0
for svc in raw:
    if not isinstance(svc, dict):
        continue
    for ap in svc.get("applets") or []:
        if not isinstance(ap, dict):
            bad_ap += 1
            continue
        for c in ap.get("components") or []:
            if not isinstance(c, dict):
                bad_c += 1
                continue
            lab = nz(c.get("label"))
            side = "trigger" if lab == "if" else ("action" if lab == "then" else None)
            if side is None:
                other_label += 1
                continue
            u = nz(c.get("url"))
            if not u:
                no_url += 1
                continue
            n_comp += 1
            sides[side].setdefault(u, c)

print(f"raw services={len(raw)}  usable components={n_comp}")
print(f"  malformed applets={bad_ap}  malformed components={bad_c}  "
      f"non-If/Then labels={other_label}  components with no url={no_url}")

for side in ("trigger", "action"):
    U = sides[side]
    names = Counter(nz(c.get("service_name")) for c in U.values())
    chans = Counter(chan_of(u) for u in U)
    dup_urls = sum(v for v in names.values() if v > 1)
    print(f"\n--- {side} ---")
    print(f"  distinct URLs (true functions): {len(U)}")
    print(f"  distinct service_names:         {len(names)}")
    print(f"  distinct channels:              {len(chans)}")
    print(f"  urls whose name is shared: {dup_urls} ({dup_urls/len(U):.1%})")
    print(f"  worst name collisions: {names.most_common(4)}")
    rag = json.load(open(f"data/{side}s_rag.json"))
    print(f"  indexed corpus rows: {len(rag)}   true functions lost: {len(U)-len(rag)} "
          f"({(len(U)-len(rag))/len(U):.1%})")

ev = json.load(open("data/clean_split/eval.json"))
for side in ("trigger", "action"):
    U = sides[side]
    gu = [nz(r[side].get("url")) for r in ev]
    inraw = sum(1 for u in gu if u in U)
    print(f"\neval gold {side} url in raw url set: {inraw}/{len(ev)} ({inraw/len(ev):.3f})  "
          f"distinct gold urls: {len(set(gu))}")
