#!/usr/bin/env python3
"""Architecture search for the actual goal: query -> fully-populated applet.

The target is a COMPLETE applet: every required action field filled, and every
{{ref}} pointing at a real ingredient of the chosen trigger.

DESIGN - retrieval is held constant.
    Each sample reuses the trigger/action pair that the published run already
    selected (results/e2e_eval.json -> raw_prediction.final_applet). Only the
    BINDING strategy varies, so any difference is attributable to architecture
    rather than to retrieval luck.

VARIANTS
    v1_freeform   Replicates today's behaviour: hand the model the schemas and
                  ask for field_values. This is the baseline.
    v2_schema     Schema-constrained: enumerate the required fields explicitly,
                  list every ingredient with Type and Example, state the key
                  convention, and forbid inventing a literal when an ingredient
                  fits. Same single call - only the prompt changes.
    v3_twostage   Decide per-field FIRST (ingredient vs literal, with a reason),
                  then emit values. Costs 1 extra call per applet.

HEADLINE METRIC - executable_rate: fraction of applets where every required
field is filled AND every ingredient reference resolves. Partial credit hides
exactly the failures that matter for execution.

KEY CONVENTION - accepts a field keyed by display name OR by Slug. The probe
found nemotron keying by Slug, which is a defensible reading of a schema that
carries both; scoring only display names penalises a correct answer.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter

ROOT = "/raid/session/aicontents/farm"
CLOUD = os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com")
LOCAL = "http://127.0.0.1:11434"
KEY = os.environ.get("OLLAMA_API_KEY", "")

JSON_OBJ = re.compile(r"\{.*\}", re.S)
REF = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def chat(model, prompt, local=False, timeout=180):
    host = LOCAL if local else CLOUD
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0}}
    headers = {"Content-Type": "application/json"}
    if not local:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("message", {}).get("content", "") or ""


def parse_obj(text):
    m = JSON_OBJ.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # tolerate trailing prose after the object
        depth, end = 0, None
        for i, ch in enumerate(m.group(0)):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                end = i + 1
                break
        if end:
            try:
                return json.loads(m.group(0)[:end])
            except Exception:
                return None
        return None


# ---------------------------------------------------------------- schema
def ing_table(trigger_api):
    rows, keys = [], set()
    for name, spec in (trigger_api.get("Ingredients") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        slug = str(spec.get("Slug", name))
        rows.append({"name": name, "slug": slug,
                     "type": str(spec.get("Type", "")),
                     "example": str(spec.get("Example", ""))})
        keys |= {name.strip().lower(), slug.strip().lower()}
    return rows, keys


def field_table(action_api):
    rows = []
    for name, spec in (action_api.get("Action fields") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        rows.append({"name": name, "slug": str(spec.get("Slug", name)),
                     "required": str(spec.get("Required", "")).lower() == "true",
                     "helper": str(spec.get("Helper text", ""))})
    return rows


# ---------------------------------------------------------------- variants
def p_v1(ings, fields):
    return (
        "Bind a trigger's outputs to an action's input fields.\n\n"
        f"TRIGGER INGREDIENTS:\n{json.dumps(ings, indent=1)}\n\n"
        f"ACTION FIELDS:\n{json.dumps(fields, indent=1)}\n\n"
        'Return ONLY JSON: {"field_values": {"<field>": "<value>"}}\n'
        "Use {{Ingredient}} to reference a trigger ingredient."
    )


def p_v2(ings, fields):
    req = [f["name"] for f in fields if f["required"]]
    ilist = "\n".join(
        f"  - {i['name']}  (slug: {i['slug']}, type: {i['type'] or 'unknown'}, e.g. {i['example'] or 'n/a'})"
        for i in ings) or "  (none)"
    flist = "\n".join(
        f"  - {f['name']}  (slug: {f['slug']}){'  [REQUIRED]' if f['required'] else ''}"
        + (f"  -- {f['helper']}" if f["helper"] else "")
        for f in fields) or "  (none)"
    return (
        "You are completing a trigger-action applet so it can execute.\n\n"
        f"AVAILABLE TRIGGER INGREDIENTS (the only dynamic values that exist):\n{ilist}\n\n"
        f"ACTION FIELDS TO FILL:\n{flist}\n\n"
        "RULES:\n"
        "1. Every field marked [REQUIRED] must have a non-empty value.\n"
        "2. If an ingredient above carries the value the field needs, reference it as "
        "{{slug}} using the EXACT slug shown. Never invent an ingredient name.\n"
        "3. Only use a literal when NO ingredient fits (e.g. a device the user must pick).\n"
        "4. Key the output by the field NAME exactly as written above.\n\n"
        f"REQUIRED FIELDS: {json.dumps(req)}\n\n"
        'Return ONLY JSON: {"field_values": {"<field name>": "<value>"}}'
    )


def p_v3a(ings, fields):
    ilist = "\n".join(f"  - {i['name']} (slug: {i['slug']}, type: {i['type'] or 'unknown'})"
                      for i in ings) or "  (none)"
    flist = "\n".join(f"  - {f['name']}{'  [REQUIRED]' if f['required'] else ''}"
                      + (f"  -- {f['helper']}" if f["helper"] else "")
                      for f in fields) or "  (none)"
    return (
        "Decide, for each action field, whether a trigger ingredient can supply its "
        "value, or whether it needs a user-chosen literal.\n\n"
        f"INGREDIENTS:\n{ilist}\n\nFIELDS:\n{flist}\n\n"
        'Return ONLY JSON: {"decisions": [{"field": "<name>", "source": "ingredient"|"literal", '
        '"ingredient_slug": "<slug or null>", "why": "<short>"}]}'
    )


def p_v3b(ings, fields, decisions):
    return (
        "Produce the final field values from these decisions.\n\n"
        f"DECISIONS:\n{json.dumps(decisions, indent=1)}\n\n"
        f"INGREDIENTS:\n{json.dumps(ings, indent=1)}\n\n"
        "For source=ingredient emit {{slug}} with the exact slug. For source=literal emit a "
        "concrete sensible value. Every field in DECISIONS must appear.\n\n"
        'Return ONLY JSON: {"field_values": {"<field name>": "<value>"}}'
    )


# ---------------------------------------------------------------- scoring
def score(fv, ings, ing_keys, fields):
    """Accept a field keyed by display name OR slug."""
    if not isinstance(fv, dict):
        return None
    lookup = {}
    for k, v in fv.items():
        lookup[str(k).strip().lower()] = v
    req = [f for f in fields if f["required"]]
    filled = 0
    for f in req:
        v = lookup.get(f["name"].strip().lower())
        if v is None:
            v = lookup.get(f["slug"].strip().lower())
        if str(v or "").strip():
            filled += 1
    refs, grounded = 0, 0
    static = 0
    for v in fv.values():
        s = str(v).strip()
        if not s:
            continue
        rs = REF.findall(s)
        if rs:
            for r in rs:
                refs += 1
                if r.strip().lower() in ing_keys:
                    grounded += 1
        else:
            static += 1
    return {"req_total": len(req), "req_filled": filled, "refs": refs,
            "grounded": grounded, "static": static,
            "executable": (filled == len(req)) and (refs == grounded)}


def main():
    ap = argparse.ArgumentParser()
    # nargs="*" so a local-only control run can pass no cloud models at all
    ap.add_argument("--models", nargs="*", default=["gemma4:31b"])
    ap.add_argument("--variants", nargs="+", default=["v1_freeform", "v2_schema"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--local-model", default=None,
                    help="also run this model against the LOCAL ollama (e.g. granite4:small-h)")
    ap.add_argument("--out", default="results/APPLET_LAB.json")
    args = ap.parse_args()

    samples = json.load(open(f"{ROOT}/results/e2e_eval.json", encoding="utf-8"))["per_sample"]
    cases = []
    for r in samples:
        fa = (r.get("raw_prediction") or {}).get("final_applet") or {}
        trg, act = fa.get("trigger") or {}, fa.get("action") or {}
        if not trg or not act:
            continue
        ings, keys = ing_table(trg.get("api_info") or {})
        fields = field_table(act.get("api_info") or {})
        if not fields or not ings:
            continue
        cases.append({"query": r.get("query", ""), "ings": ings, "keys": keys,
                      "fields": fields})
        if len(cases) >= args.n:
            break
    print(f"{len(cases)} cases (retrieval held constant from the published run)\n", flush=True)

    runs = [(m, False) for m in args.models]
    if args.local_model:
        runs.append((args.local_model, True))

    results = {}
    for model, is_local in runs:
        tag = f"{model}{' (local)' if is_local else ''}"
        for variant in args.variants:
            agg = Counter()
            errs = 0
            t0 = time.time()
            for i, c in enumerate(cases):
                try:
                    if variant == "v1_freeform":
                        obj = parse_obj(chat(model, p_v1(c["ings"], c["fields"]), is_local))
                    elif variant == "v2_schema":
                        obj = parse_obj(chat(model, p_v2(c["ings"], c["fields"]), is_local))
                    elif variant == "v3_twostage":
                        d = parse_obj(chat(model, p_v3a(c["ings"], c["fields"]), is_local))
                        decs = (d or {}).get("decisions", [])
                        obj = parse_obj(chat(model, p_v3b(c["ings"], c["fields"], decs), is_local))
                    else:
                        raise ValueError(variant)
                    s = score((obj or {}).get("field_values"), c["ings"], c["keys"], c["fields"])
                    if s is None:
                        errs += 1
                        continue
                    for k, v in s.items():
                        agg[k] += int(v) if isinstance(v, bool) else v
                    agg["n"] += 1
                except Exception:
                    errs += 1
                if (i + 1) % 10 == 0:
                    print(f"  {tag}/{variant}: {i+1}/{len(cases)}", flush=True)

            n = agg["n"] or 1
            row = {
                "n": agg["n"], "errors": errs,
                "required_coverage": agg["req_filled"] / max(agg["req_total"], 1),
                "executable_rate": agg["executable"] / n,
                "ingredient_grounding": agg["grounded"] / max(agg["refs"], 1),
                "ingredient_share": agg["refs"] / max(agg["refs"] + agg["static"], 1),
                "sec_per_applet": round((time.time() - t0) / max(agg["n"], 1), 2),
            }
            results[f"{tag} | {variant}"] = row
            print(f"\n>>> {tag} | {variant}: exec={row['executable_rate']:.3f} "
                  f"cov={row['required_coverage']:.3f} grounded={row['ingredient_grounding']:.3f} "
                  f"ing_share={row['ingredient_share']:.3f} errs={errs}\n", flush=True)

    print("=" * 96)
    print(f"{'configuration':<44}{'exec':>8}{'req cov':>9}{'grounded':>10}{'ing share':>11}{'s/applet':>10}")
    print("=" * 96)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["executable_rate"]):
        print(f"{k:<44}{v['executable_rate']:>8.3f}{v['required_coverage']:>9.3f}"
              f"{v['ingredient_grounding']:>10.3f}{v['ingredient_share']:>11.3f}{v['sec_per_applet']:>10.2f}")
    print("\nexecutable_rate = every required field filled AND every {{ref}} resolves.")
    print("Published baseline for reference: req_cov 0.980, grounding 0.960, ing_share 0.449")

    p = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(results, open(p, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
