#!/usr/bin/env python3
"""Probe candidate LLMs on the three things FARM's agents actually require.

WHY: the paper reports one LLM (granite4:small-h) and reviewers asked how the
choice of LLM affects results. Ollama Cloud gives free access to several models,
and cloud inference needs no local GPU - so this runs alongside trainB.

Before spending a full 300-sample e2e run on any model, check it can do the three
things the pipeline depends on:

  T1 TOOL CALLING   agents/nodes/planner_node.py calls parse_tool_call() on the
                    response, and agents/prompts/planner.py asks for a
                    <tool_call>{...}</tool_call> block. Does the model emit native
                    tool_calls, or the tagged text form, or neither?

  T2 STRICT JSON    agents/nodes/output_parser.py:parse_json_response() must find a
                    JSON object in response.content. Models that stream reasoning
                    into a separate field, or fence their JSON, or prepend prose,
                    break this.

  T3 SCHEMA BINDING The real task: given a trigger's Ingredients and an action's
                    Action fields, emit field_values using {{Ingredient}} refs.
                    This is what produces the binding metrics.

Any model failing T2 cannot be swapped in without changing the parser, which is
itself a finding worth reporting.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com")
KEY = os.environ.get("OLLAMA_API_KEY", "")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_triggers",
        "description": "Search for trigger APIs matching a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search text"},
                "top_k": {"type": "integer", "description": "how many results"},
            },
            "required": ["query"],
        },
    },
}]

# A real case from data/test/gold.json (test_000).
TRIGGER_INGREDIENTS = {
    "StockName": {"Slug": "StockName", "Type": "String", "Example": "Google Inc."},
    "StockTicker": {"Slug": "StockTicker", "Type": "String", "Example": "GOOG"},
    "Price": {"Slug": "Price", "Type": "String", "Example": "1024.50"},
}
ACTION_FIELDS = {
    "Which light(s)?": {"Slug": "entity", "Required": "true"},
    "What light mode / color?": {"Slug": "mode", "Required": "true"},
    "What brightness?": {"Slug": "dimming", "Required": "true"},
}

JSON_OBJ = re.compile(r"\{.*\}", re.S)
TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def chat(model, messages, tools=None, timeout=120):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data, time.time() - t0


def t1_tool_calling(model):
    """Native tool_calls, tagged <tool_call>, or nothing?"""
    try:
        data, dt = chat(model, [
            {"role": "system", "content": "You are a planner. Use the provided tools."},
            {"role": "user", "content": "Find trigger APIs for 'stock price rises'. Use search_triggers."},
        ], tools=TOOLS)
    except Exception as e:
        return {"mode": f"ERROR:{type(e).__name__}", "ok": False, "sec": None,
                "detail": str(e)[:90]}
    msg = data.get("message", {})
    native = msg.get("tool_calls")
    if native:
        fn = native[0].get("function", {})
        return {"mode": "native", "ok": True, "sec": round(dt, 1),
                "detail": f"{fn.get('name')}({str(fn.get('arguments'))[:50]})"}
    tagged = TOOL_TAG.search(msg.get("content", "") or "")
    if tagged:
        return {"mode": "tagged", "ok": True, "sec": round(dt, 1),
                "detail": tagged.group(1)[:60]}
    return {"mode": "none", "ok": False, "sec": round(dt, 1),
            "detail": (msg.get("content", "") or "")[:70]}


def t2_strict_json(model):
    """Must yield a parseable JSON object from message.content."""
    try:
        data, dt = chat(model, [
            {"role": "system", "content": "Reply with ONLY a JSON object. No prose, no code fences."},
            {"role": "user", "content":
                'Return {"reasoning": "<one short sentence>", "ingredients": ["a","b"], "accept": true}'},
        ])
    except Exception as e:
        return {"ok": False, "sec": None, "detail": f"ERROR:{type(e).__name__}"}
    content = data.get("message", {}).get("content", "") or ""
    thinking = data.get("message", {}).get("thinking") or data.get("message", {}).get("reasoning")
    m = JSON_OBJ.search(content)
    if not m:
        return {"ok": False, "sec": round(dt, 1), "leaks_reasoning": bool(thinking),
                "detail": f"no JSON in content: {content[:60]!r}"}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"ok": False, "sec": round(dt, 1), "leaks_reasoning": bool(thinking),
                "detail": f"unparseable: {m.group(0)[:60]!r}"}
    keys_ok = {"reasoning", "ingredients", "accept"} <= set(obj)
    fenced = "```" in content
    return {"ok": keys_ok, "sec": round(dt, 1), "leaks_reasoning": bool(thinking),
            "fenced": fenced, "detail": f"keys={sorted(obj)[:4]}"}


def t3_schema_binding(model):
    """The real task: wire ingredients into required action fields."""
    prompt = (
        "You are binding a trigger's outputs to an action's input fields.\n\n"
        f"TRIGGER INGREDIENTS (available):\n{json.dumps(TRIGGER_INGREDIENTS, indent=1)}\n\n"
        f"ACTION FIELDS (must fill every Required one):\n{json.dumps(ACTION_FIELDS, indent=1)}\n\n"
        'Return ONLY JSON: {"field_values": {"<field name>": "<value>"}}\n'
        "Use {{IngredientSlug}} when a trigger ingredient supplies the value; "
        "use a literal only when no ingredient fits."
    )
    try:
        data, dt = chat(model, [{"role": "user", "content": prompt}])
    except Exception as e:
        return {"ok": False, "sec": None, "detail": f"ERROR:{type(e).__name__}"}
    content = data.get("message", {}).get("content", "") or ""
    m = JSON_OBJ.search(content)
    if not m:
        return {"ok": False, "sec": round(dt, 1), "detail": "no JSON"}
    try:
        fv = json.loads(m.group(0)).get("field_values", {})
    except Exception:
        return {"ok": False, "sec": round(dt, 1), "detail": "unparseable"}
    required = [f for f, s in ACTION_FIELDS.items() if s.get("Required") == "true"]
    filled = [f for f in required if str(fv.get(f, "")).strip()]
    refs = [r for v in fv.values() for r in re.findall(r"\{\{\s*([^}]+?)\s*\}\}", str(v))]
    slugs = {k.lower() for k in TRIGGER_INGREDIENTS} | {
        v["Slug"].lower() for v in TRIGGER_INGREDIENTS.values()}
    grounded = [r for r in refs if r.strip().lower() in slugs]
    return {"ok": len(filled) == len(required), "sec": round(dt, 1),
            "coverage": f"{len(filled)}/{len(required)}",
            "refs": len(refs), "grounded": len(grounded),
            "detail": json.dumps(fv)[:70]}


DEFAULT_MODELS = ["gpt-oss:20b", "nemotron-3-nano:30b", "gemma4:31b", "minimax-m3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--out", default="results/LLM_probe.json")
    args = ap.parse_args()
    if not KEY:
        sys.exit("OLLAMA_API_KEY not set")

    results = {}
    for model in args.models:
        print(f"\n{'='*70}\n{model}\n{'='*70}", flush=True)
        r = {}
        for name, fn in (("T1_tool_calling", t1_tool_calling),
                         ("T2_strict_json", t2_strict_json),
                         ("T3_schema_binding", t3_schema_binding)):
            r[name] = fn(model)
            status = "PASS" if r[name].get("ok") else "FAIL"
            print(f"  {name:<20} {status:<5} {r[name].get('sec')}s  {r[name].get('detail','')}", flush=True)
            extra = {k: v for k, v in r[name].items()
                     if k not in ("ok", "sec", "detail")}
            if extra:
                print(f"  {'':<20}       {extra}", flush=True)
        results[model] = r

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'model':<24}{'tools':>10}{'json':>8}{'binding':>10}{'grounded':>10}")
    print("-" * 70)
    for m, r in results.items():
        print(f"{m:<24}{r['T1_tool_calling']['mode']:>10}"
              f"{'ok' if r['T2_strict_json']['ok'] else 'FAIL':>8}"
              f"{r['T3_schema_binding'].get('coverage','-'):>10}"
              f"{str(r['T3_schema_binding'].get('grounded','-')):>10}")

    out = os.path.join("/raid/session/aicontents/farm", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
