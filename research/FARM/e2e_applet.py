"""End-to-end applet generation: query -> select functions -> bind fields -> validate -> repair.

Arms (each adds one stage, so the ablation is a strict nesting):
  R0    rank-1 retrieval, no LLM, no binding            (deterministic floor)
  S     LLM selects from the top-k candidate lists
  SB    LLM selects, then LLM binds the action's required fields
  SBR   SB plus up to --repair validation-repair rounds
  SBRA  SBR, but valid bindings ACCUMULATE and each round re-asks only for the
        fields still missing. Failure analysis of SBR: validity is 97-100% at
        <=3 required fields, 40.6% at 4 and 0.0% at 6+, and 35 of 37 failures are
        omitted fields, not wrong ones. Re-asking for the whole array lets the
        model drop fields again, so the round count never converges.

Ollama Cloud does not support structured outputs (docs.ollama.com/capabilities/
structured-outputs: "Ollama's Cloud currently does not support structured outputs"),
so the schema is grounded in the prompt as the docs advise and enforced here by
validating and repairing. That loop is measured, not assumed.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from farm.render import field_dict, ingredient_dict, required_fields, bindable, parse_url

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="SBR", choices=["R0", "S", "SB", "SBR", "SBRA"])
ap.add_argument("--model", default="gemma4:31b")
ap.add_argument("--key-env", default="OLLAMA_API_KEY")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--repair", type=int, default=2)
ap.add_argument("--timeout", type=int, default=180)
ap.add_argument("--candidates", default="data/candidates/eval_top50.json",
                help="retrieval order by default; pass a reranked_*.json to test the cascade")
ap.add_argument("--tag", default="")
A = ap.parse_args()

URL = "https://ollama.com/api/chat"
KEY = os.environ.get(A.key_env, "")
TOK = re.compile(r"[a-z0-9]+")


def toks(s): return set(TOK.findall(str(s or "").lower()))


def chat(messages, timeout):
    body = json.dumps({"model": A.model, "messages": messages, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def extract_json(text):
    """Cloud gives no schema enforcement, so recover the object from prose."""
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, *(m.group(0) for m in re.finditer(r"\{.*\}", t, re.S))):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


corpus = {k: {c["url"]: c for c in json.loads(Path(f"data/corpus/{k}s.json").read_text())}
          for k in ("trigger", "action")}
base = json.loads(Path("data/candidates/eval_top50.json").read_text())
if A.candidates != "data/candidates/eval_top50.json":
    # a reranked shortlist: {applet_url: {trigger: [url,...], action: [url,...]}}
    rr = json.loads(Path(A.candidates).read_text())
    for aurl, v in base.items():
        if aurl in rr:
            for kind in ("trigger", "action"):
                v[kind] = [{"url": u, "score": None} for u in rr[aurl][kind]]
    print(f"using reranked shortlists from {A.candidates}", flush=True)
cands = base
items = list(cands.items())[:A.n]


def brief(kind, url, i):
    c = corpus[kind][url]
    fs = field_dict(c.get("api_info") or {}, kind)
    line = (f"[{i}] channel={c['channel']} | function={c['service_name']} | "
            f"{str(c.get('description') or '')[:150]}")
    if fs:
        line += " | fields: " + ", ".join(str(v.get("Label") or n) for n, v in fs.items())
    return line


SELECT_SYS = ("You pick the two IFTTT functions that implement a user's automation. "
              "Answer with ONLY a JSON object, no prose: "
              '{"trigger": <index int>, "action": <index int>}')
BIND_SYS = ("You wire an IFTTT applet. For every REQUIRED action field, choose exactly one source.\n"
            "Reply with ONLY this JSON, no prose:\n"
            '{"bindings":[{"field":"<exact field name>","source":"ingredient|static|user_input",'
            '"value":"<ingredient name, or the literal static text, or the question to ask the user>"}]}\n'
            'Rules: "ingredient" means copy a value the trigger provides, and value MUST be one of the '
            'listed ingredient names. "static" means a constant you can justify from the user request, '
            'and its words must appear in the request. "user_input" means the value genuinely cannot be '
            'inferred (a device to pick, an account, a personal phone number) - then value is the question '
            "to ask. Prefer ingredient, then static. Use user_input only when neither is defensible.")


def validate(binds, trig, act, query=""):
    """Returns (errors, stats). This is the schema enforcement the cloud API cannot do.

    Two rules beyond well-formedness, both added after measurement:
      - a `static` value must be grounded in the user's request. Without this,
        gemma4 fabricated a value out of nothing on 7.1% of bindings and passed.
      - asking the user about a field the trigger could fill is a SOFT error: it is
        surfaced to the repair turn but does not by itself invalidate the applet,
        because a wrong auto-fill is worse than one extra question. 14.8% (gemma4)
        and 19.2% (gpt-oss) of bindings did this.
    """
    req = required_fields(act)
    ing = {n.lower() for n in ingredient_dict(act.get("api_info") or {})}
    ing |= {n.lower() for n in ingredient_dict(trig.get("api_info") or {})}
    trig_has_ing = bool(ingredient_dict(trig.get("api_info") or {}))
    qt = toks(query)
    errs, seen, stats = [], set(), Counter()
    soft = []
    for b in (binds if isinstance(binds, list) else []):
        if not isinstance(b, dict):
            errs.append("a binding is not an object")
            continue
        f, src, val = str(b.get("field", "")), str(b.get("source", "")), str(b.get("value", ""))
        match = next((n for n, v in req.items()
                      if f.strip().lower() in (n.lower(), str(v.get("Label", "")).lower())), None)
        if match is None:
            errs.append(f"field {f!r} is not a required field of this action")
            continue
        seen.add(match)
        if src == "ingredient":
            if val.strip().strip("{}").lower() not in ing:
                errs.append(f"{f!r}: {val!r} is not an ingredient the trigger provides")
            elif not bindable(req[match]):
                errs.append(f"{f!r} is a device/account selector and cannot take an ingredient")
            else:
                stats["ingredient"] += 1
        elif src == "static":
            vt = toks(val)
            if not val.strip():
                errs.append(f"{f!r}: static value is empty")
            elif qt and not (vt and vt <= qt):
                errs.append(f"{f!r}: static value {val!r} is not grounded in the user request; "
                            "use an ingredient, or ask the user")
            else:
                stats["static"] += 1
                stats["static_grounded"] += 1
        elif src == "user_input":
            if not val.strip():
                errs.append(f"{f!r}: user_input needs a question to ask")
            else:
                stats["user_input"] += 1
                if bindable(req[match]) and trig_has_ing:
                    stats["avoidable_question"] += 1
                    soft.append(f"{f!r} accepts a trigger ingredient and the trigger provides "
                                f"{sorted(ing)[:6]} - prefer one of those over asking the user")
        else:
            errs.append(f"{f!r}: source must be ingredient, static or user_input")
    for n in req:
        if n not in seen:
            errs.append(f"required field {n!r} has no binding")
    stats["required"] = len(req)
    stats["soft"] = len(soft)
    return errs, stats, soft


def split_valid(binds, trig, act):
    """Per-binding triage, so a good binding is not thrown away because a
    sibling in the same reply was bad."""
    req = required_fields(act)
    good = {}
    for b in (binds if isinstance(binds, list) else []):
        if not isinstance(b, dict):
            continue
        f = str(b.get("field", ""))
        match = next((n for n, v in req.items()
                      if f.strip().lower() in (n.lower(), str(v.get("Label", "")).lower())), None)
        if match is None:
            continue
        one, _, _ = validate([b], trig, act)
        # validate() also reports the OTHER required fields as unbound; ignore those here
        if not [e for e in one if "has no binding" not in e]:
            good[match] = {"field": match, "source": str(b.get("source", "")),
                           "value": str(b.get("value", ""))}
    return good


out = {"arm": A.arm, "model": A.model, "topk": A.topk, "n": len(items), "records": []}
agg = Counter()
t_start = time.time()

for n, (aurl, rec) in enumerate(items):
    q = rec["query"]
    tl = [c["url"] for c in rec["trigger"][:A.topk]]
    al = [c["url"] for c in rec["action"][:A.topk]]
    r = {"applet_url": aurl, "query": q, "gold": rec["gold"]}

    if A.arm == "R0":
        r["trigger"], r["action"] = tl[0], al[0]
    else:
        prompt = (f"User request: {q}\n\nTRIGGERS:\n" + "\n".join(brief("trigger", u, i) for i, u in enumerate(tl))
                  + "\n\nACTIONS:\n" + "\n".join(brief("action", u, i) for i, u in enumerate(al)))
        try:
            raw = chat([{"role": "system", "content": SELECT_SYS}, {"role": "user", "content": prompt}], A.timeout)
            j = extract_json(raw) or {}
            ti, ai = int(j.get("trigger", 0)), int(j.get("action", 0))
            r["trigger"] = tl[ti] if 0 <= ti < len(tl) else tl[0]
            r["action"] = al[ai] if 0 <= ai < len(al) else al[0]
            if not isinstance(j.get("trigger"), int):
                agg["select_parse_fail"] += 1
        except Exception as e:
            agg["select_error"] += 1
            r["trigger"], r["action"] = tl[0], al[0]
            r["select_error"] = type(e).__name__

    r["trigger_ok"] = r["trigger"] == rec["gold"]["trigger"]
    r["action_ok"] = r["action"] == rec["gold"]["action"]
    r["joint_ok"] = r["trigger_ok"] and r["action_ok"]
    agg["trigger_ok"] += r["trigger_ok"]; agg["action_ok"] += r["action_ok"]; agg["joint_ok"] += r["joint_ok"]

    if A.arm in ("SB", "SBR"):
        trig, act = corpus["trigger"][r["trigger"]], corpus["action"][r["action"]]
        req = required_fields(act)
        r["n_required"] = len(req)
        if req:
            ings = ingredient_dict(trig.get("api_info") or {})
            spec = (f"User request: {q}\n\n"
                    f"TRIGGER {trig['channel']} / {trig['service_name']}\n"
                    f"provides these ingredients: "
                    + (", ".join(ings) if ings else "(none)") + "\n\n"
                    f"ACTION {act['channel']} / {act['service_name']}\n"
                    "required fields:\n"
                    + "\n".join(f"  - {n} (label={v.get('Label', n)!r}, "
                                f"accepts_ingredient={bindable(v)})" for n, v in req.items()))
            msgs = [{"role": "system", "content": BIND_SYS}, {"role": "user", "content": spec}]
            rounds, errs, stats, binds, soft = 0, ["no attempt"], Counter(), None, []
            acc = {}
            accumulate = A.arm == "SBRA"
            max_rounds = 1 + (A.repair if A.arm in ("SBR", "SBRA") else 0)
            while rounds < max_rounds:
                rounds += 1
                try:
                    raw = chat(msgs, A.timeout)
                except Exception as e:
                    agg["bind_error"] += 1
                    errs = [f"request failed: {type(e).__name__}"]
                    break
                j = extract_json(raw)
                if j is None:
                    agg["bind_parse_fail"] += 1
                    errs = ["reply was not valid JSON"]
                else:
                    if accumulate:
                        acc.update(split_valid(j.get("bindings"), trig, act))
                        binds = list(acc.values())
                    else:
                        binds = j.get("bindings")
                    errs, stats, soft = validate(binds, trig, act, q)
                if rounds == 1:
                    r["valid_first_pass"] = not errs
                    agg["valid_first_pass"] += (not errs)
                if not errs:
                    # A valid binding is accepted as-is. An earlier version spent a
                    # round pushing the model to convert user_input into ingredient
                    # bindings, on the strength of an `avoidable_questions` metric that
                    # flagged any bindable field whenever the trigger offered ANY
                    # ingredient. Inspection showed every flagged case was a question
                    # the model was RIGHT to ask ("Spreadsheet name" against a trigger
                    # offering Latitude/Longitude; "Day of week" for a digest schedule).
                    # `bindable()` means IFTTT permits an ingredient in that slot, not
                    # that a suitable one exists. The nudge converted 0 of 2 and was
                    # pushing toward fabrication, so it is removed and the metric is
                    # kept only as an upper-bound diagnostic.
                    break
                if accumulate:
                    missing = [e.split("'")[1] for e in errs if "has no binding" in e]
                    other = [e for e in errs if "has no binding" not in e]
                    ask = ("Those bindings are accepted and kept: "
                           + (", ".join(sorted(acc)) if acc else "(none)") + ".\n")
                    if other:
                        ask += "Rejected:\n" + "\n".join(f"  - {e}" for e in other) + "\n"
                    if missing:
                        ask += ("Now bind ONLY these remaining required fields, nothing else: "
                                + ", ".join(repr(m) for m in missing) + "\n")
                    ask += "Return the same JSON shape containing only those fields."
                    msgs = [{"role": "system", "content": BIND_SYS},
                            {"role": "user", "content": spec},
                            {"role": "user", "content": ask}]
                else:
                    msgs += [{"role": "assistant", "content": raw if j is not None else "(unparseable)"},
                             {"role": "user", "content": "That was rejected by the schema validator:\n"
                              + "\n".join(f"  - {e}" for e in errs)
                              + "\nReturn the corrected JSON only."}]
            r["bind_rounds"], r["bind_errors"], r["bindings"] = rounds, errs, binds
            r["valid_final"] = not errs
            agg["valid_final"] += (not errs)
            agg["bind_rounds"] += rounds
            agg["applets_with_required"] += 1
            for k in ("ingredient", "static", "user_input", "required",
                      "static_grounded", "avoidable_question"):
                agg[f"f_{k}"] += stats[k]
            if not errs:
                agg["complete_applets"] += r["joint_ok"]
        else:
            r["valid_first_pass"] = r["valid_final"] = True
            r["bind_rounds"] = 0
            agg["no_required_field"] += 1
            agg["complete_applets"] += r["joint_ok"]

    out["records"].append(r)
    if (n + 1) % 25 == 0:
        print(f"  {n+1}/{len(items)}  joint={agg['joint_ok']/(n+1):.3f}  "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

N = len(items)
wr = agg["applets_with_required"] or 1
fr = agg["f_required"] or 1
out["metrics"] = {
    "trigger_acc": agg["trigger_ok"] / N, "action_acc": agg["action_ok"] / N,
    "joint_acc": agg["joint_ok"] / N,
    "select_parse_fail": agg["select_parse_fail"], "select_error": agg["select_error"],
    "applets_with_required_field": agg["applets_with_required"],
    "applets_with_no_required_field": agg["no_required_field"],
    "schema_valid_first_pass": agg["valid_first_pass"] / wr,
    "schema_valid_final": agg["valid_final"] / wr,
    "mean_repair_rounds": agg["bind_rounds"] / wr,
    "bind_parse_fail": agg["bind_parse_fail"], "bind_error": agg["bind_error"],
    "fields_required": agg["f_required"],
    "from_ingredient": agg["f_ingredient"] / fr, "from_static": agg["f_static"] / fr,
    "needs_human": agg["f_user_input"] / fr,
    "grounded_rate": (agg["f_ingredient"] + agg["f_static"]) / fr,
    # upper bound only: counts bindable fields, not semantically fillable ones
    "avoidable_questions_upper_bound": agg["f_avoidable_question"] / fr,
    "end_to_end_complete": agg["complete_applets"] / N,
    "wall_seconds": time.time() - t_start,
}
tag = A.tag or f"{A.arm}_{A.model.replace(':','-')}_k{A.topk}_n{N}"
Path("results/e2e").mkdir(parents=True, exist_ok=True)
Path(f"results/e2e/{tag}.json").write_text(json.dumps(out, indent=1))
print(f"\n=== {tag} ===")
for k, v in out["metrics"].items():
    print(f"  {k:28} {v:.4f}" if isinstance(v, float) else f"  {k:28} {v}")
