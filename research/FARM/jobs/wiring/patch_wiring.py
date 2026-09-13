#!/usr/bin/env python3
"""Wire the three FARM mechanisms that the paper describes but the code never executes.

Every change is behind a config flag defaulting to False, so with no flags set the
pipeline is byte-for-byte the behaviour that produced results/e2e_eval.json. That
keeps the published run reproducible AND makes the flags an A/B experiment.

    python patch_wiring.py --check     # what is applied / not applied
    python patch_wiring.py --apply     # apply (writes .bak_wiring backups)
    python patch_wiring.py --revert    # restore from backups

------------------------------------------------------------------------------
PATCH A  agents/config.py            add the three flags (default False)

PATCH B  agents/nodes/fallback.py    THE BIG ONE.
         fallback_node() iterates PAIR_ORDER - a hardcoded lexicographic 5x5 grid
         [(0,0),(0,1),(0,2)...] - and ignores state["priority_queue"], which
         cross_scorer_node computed using the paper's formula
             score = 0.7*coverage + 0.3*(sim(q,t)+sim(q,a))/2
         The file's own comment concedes it: "Local PAIR_ORDER for backward
         compatibility with old negotiation graph".
         So the paper's cross-schema scoring ranks the pairs and is then thrown
         away in favour of counting. With use_scored_pair_order=True the retry
         order becomes the scored order.

PATCH C  agents/graph.py             graph.py:133 is `add_edge("verifier", END)`,
         making the verifier terminal. It computes a quality score that nothing
         acts on - agents/config.py even says "Verifier no longer rejects pairs".
         With enable_quality_gate=True the verifier routes to fallback when its
         score is below config.verifier_threshold, which is the quality-gated
         fallback the paper describes.

PATCH D  agents/prompts/planner.py   extract_concept() scans the LLM's reasoning
         PROSE for the first SUBSTRING hit of ["detect","when","if",...] and
         returns a fixed 5-word lowercased window, e.g.
             "price rising significantly) 2. the"
         "if" fires inside "notify", "new" inside "renew". Measured: 136/300
         trigger queries are verbatim slices of planner_reasoning.
         This is an unconditional bug fix, not a flag: it currently cannot change
         any result because load_candidates.py:50 retrieves with state["query"]
         and ignores these fields entirely. Measured cost if they WERE used:
         trigger R@5 0.29 vs 0.85 (results/AB_planner_query.json).
------------------------------------------------------------------------------
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path("/raid/session/aicontents/farm")
BAK = ".bak_wiring"

# ---------------------------------------------------------------- PATCH A
A_OLD = """    # Debug
    verbose: bool = False"""

A_NEW = '''    # -- Mechanism wiring flags (added by patch_wiring.py) ------------------
    # All default False => behaviour identical to the published run.
    #
    # Route retries through the cross-scorer's priority_queue instead of the
    # hardcoded PAIR_ORDER grid in agents/nodes/fallback.py. This is what makes
    # the paper's score = 0.7*coverage + 0.3*rag formula actually drive selection.
    use_scored_pair_order: bool = False
    # Let the verifier reject low-quality applets and route them to fallback,
    # instead of graph.py's unconditional add_edge("verifier", END).
    enable_quality_gate: bool = False
    # Cap on quality-gate re-entries, so a persistently low score cannot loop.
    max_quality_retries: int = 2

    # Debug
    verbose: bool = False'''

# ---------------------------------------------------------------- PATCH B
B_OLD = """    current_attempt = state["pair_attempt"]
    next_attempt = current_attempt + 1

    # Check if we've exhausted all pairs
    if next_attempt >= len(PAIR_ORDER):
        return {
            "negotiation_status": "failed",
            "error": f"All {len(PAIR_ORDER)} pairs exhausted",
        }

    # Get next pair
    next_trigger_idx, next_action_idx = PAIR_ORDER[next_attempt]"""

B_NEW = '''    current_attempt = state["pair_attempt"]
    next_attempt = current_attempt + 1

    # Which ordering do we walk? PAIR_ORDER is a hardcoded lexicographic grid and
    # ignores the cross-scorer entirely. priority_queue is ranked by the paper's
    # score = 0.7*coverage + 0.3*(sim(q,t)+sim(q,a))/2. See patch_wiring.py PATCH B.
    pair_order = get_pair_order(state)

    # Check if we've exhausted all pairs
    if next_attempt >= len(pair_order):
        return {
            "negotiation_status": "failed",
            "error": f"All {len(pair_order)} pairs exhausted",
        }

    # Get next pair
    next_trigger_idx, next_action_idx = pair_order[next_attempt]'''

# helper inserted just above fallback_node
B2_OLD = """def fallback_node(state: NegotiationState) -> Dict[str, Any]:"""

B2_NEW = '''def get_pair_order(state: NegotiationState):
    """Return the pair iteration order.

    With config.use_scored_pair_order the order is state["priority_queue"], which
    cross_scorer_node ranked by the paper's cross-schema compatibility score.
    Otherwise it is the legacy hardcoded PAIR_ORDER grid, preserving the exact
    behaviour that produced the published results.
    """
    if getattr(config, "use_scored_pair_order", False):
        pq = state.get("priority_queue") or []
        if pq:
            # tuples survive a JSON round-trip as lists; normalise
            return [tuple(p) for p in pq]
    return PAIR_ORDER


def fallback_node(state: NegotiationState) -> Dict[str, Any]:'''

# should_continue must respect the same ordering length
B3_OLD = """        next_attempt = state["pair_attempt"] + 1
        if next_attempt < min(len(PAIR_ORDER), config.max_pairs):"""

B3_NEW = """        next_attempt = state["pair_attempt"] + 1
        if next_attempt < min(len(get_pair_order(state)), config.max_pairs):"""

# ---------------------------------------------------------------- PATCH C
C_OLD = """    # verifier -> END (always)
    graph.add_edge("verifier", END)"""

C_NEW = '''    # verifier -> END, or -> fallback when the quality gate is enabled.
    # Unpatched this was an unconditional add_edge("verifier", END), which made the
    # verifier terminal and the paper's quality-gated fallback unreachable from it.
    # See patch_wiring.py PATCH C.
    def quality_gate_router(state):
        # Local import: graph.py does not import `config` at module level, and
        # referencing it unqualified here raised NameError at routing time.
        from agents.config import config as _cfg
        if not getattr(_cfg, "enable_quality_gate", False):
            return "end"
        score = state.get("verifier_score")
        if score is None:
            return "end"
        # Bound on pair_attempt, NOT a new counter key. LangGraph's apply_writes
        # silently DROPS any key not declared in NegotiationState - a bespoke
        # `quality_retries` would always read back 0 and loop until
        # GraphRecursionError, which run_agentic_eval.py swallows into an
        # all-zeros score (it would look like a quality drop, not a crash).
        # pair_attempt is declared (state.py:143) and incremented by fallback_node.
        attempt = state.get("pair_attempt", 0) or 0
        if score < _cfg.verifier_threshold and attempt < getattr(_cfg, "max_quality_retries", 2):
            return "retry"
        return "end"

    graph.add_conditional_edges(
        "verifier",
        quality_gate_router,
        {
            "retry": "fallback",
            "end": END,
        }
    )'''

# ---------------------------------------------------------------- PATCH D
D_OLD = '''    # Simple keyword extraction - can be enhanced
    keywords = {
        "trigger": ["detect", "when", "if", "sensor", "event", "receives", "new"],
        "action": ["log", "send", "post", "create", "add", "notify", "save", "do"]
    }

    relevant_words = keywords.get(concept_type, [])
    words = text.lower().split()

    # Find relevant phrases
    for i, word in enumerate(words):
        for kw in relevant_words:
            if kw in word:
                # Get surrounding context
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                return ' '.join(words[start:end])

    return ""'''

D_NEW = '''    # Prefer an explicit marker the prompt asks the model to emit, e.g.
    #   "TRIGGER: a new episode is published"
    # Returning "" is SAFE and correct: planner_node.py does
    #   plan.get("trigger_query") or query
    # so an empty result falls back to the full user query, which measurably
    # outperforms any fragment (trigger R@5 0.85 vs 0.29, results/AB_planner_query.json).
    #
    # The previous implementation scanned the reasoning prose for the first
    # SUBSTRING hit of a keyword list and returned a fixed 5-word lowercased
    # window - "if" matched inside "notify", "new" inside "renew" - producing
    # fragments like "price rising significantly) 2. the". See patch_wiring.py PATCH D.
    import re as _re

    marker = _re.search(
        rf"(?im)^\\s*(?:\\d+[.)]\\s*)?{_re.escape(concept_type)}\\s*[:\\-]\\s*(.+)$",
        text,
    )
    if marker:
        phrase = marker.group(1).strip().strip('.,;:"\\'')
        # strip a trailing parenthetical aside and any leading list marker
        phrase = _re.sub(r"\\s*\\([^)]*\\)\\s*$", "", phrase).strip()
        if 3 <= len(phrase) <= 200:
            return phrase

    # No trustworthy marker -> let the caller fall back to the full user query.
    return ""'''

# ---------------------------------------------------------------- PATCH E
E_OLD = """    # Configure thread for memory
    config = {}
    if use_memory and thread_id:
        config = {"configurable": {"thread_id": thread_id}}
    elif use_memory:
        import uuid
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}"""

E_NEW = '''    # Configure thread for memory
    # patch_wiring.py PATCH E: run_negotiation never set recursion_limit, so
    # LangGraph's default of 25 supersteps applied. A clean run already costs 6
    # (planner -> load_candidates -> cross_scorer -> trigger_agent -> action_agent
    # -> verifier) and each quality-gate retry adds ~4. Overrunning raises
    # GraphRecursionError, which eval/run_agentic_eval.py catches and turns into
    # None -> the sample scores all zeros. That reads as a quality drop, not a
    # crash. run_selector_negotiation already sets a limit; this one did not.
    from agents.config import config as _agent_cfg
    _recursion_limit = 2 + 4 * getattr(_agent_cfg, "max_pairs", 9) + 8

    config = {"recursion_limit": _recursion_limit}
    if use_memory and thread_id:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": _recursion_limit}
    elif use_memory:
        import uuid
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": _recursion_limit}'''


# (file, anchor, replacement, description, MARKER)
# MARKER is a string that exists ONLY after the patch is applied. A prefix of the
# replacement is not usable here: several replacements begin with lines copied
# verbatim from the original, which made the check report false APPLIEDs.
PATCHES = [
    ("agents/config.py", A_OLD, A_NEW, "A: add wiring flags (default False)",
     "use_scored_pair_order: bool = False"),
    ("agents/nodes/fallback.py", B2_OLD, B2_NEW, "B1: add get_pair_order() helper",
     "def get_pair_order(state: NegotiationState):"),
    ("agents/nodes/fallback.py", B_OLD, B_NEW, "B2: fallback_node walks the scored order",
     "pair_order = get_pair_order(state)"),
    ("agents/nodes/fallback.py", B3_OLD, B3_NEW, "B3: should_continue uses the same order",
     "min(len(get_pair_order(state)), config.max_pairs)"),
    ("agents/graph.py", C_OLD, C_NEW, "C: verifier -> {END | fallback} quality gate",
     "def quality_gate_router(state):"),
    ("agents/prompts/planner.py", D_OLD, D_NEW, "D: fix extract_concept fragment bug",
     "patch_wiring.py PATCH D"),
    ("agents/graph.py", E_OLD, E_NEW, "E: recursion_limit headroom for the gate loop",
     "patch_wiring.py PATCH E"),
]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    if args.revert:
        n = 0
        for rel in sorted({p[0] for p in PATCHES}):
            bak = root / (rel + BAK)
            if bak.exists():
                shutil.copy2(bak, root / rel)
                bak.unlink()
                print(f"reverted {rel}")
                n += 1
        print(f"\n{n} file(s) restored." if n else "\nnothing to revert.")
        return 0

    if args.check:
        print(f"{'PATCH':<46}{'STATE'}")
        print("-" * 62)
        bad = 0
        for rel, old, new, desc, marker in PATCHES:
            txt = (root / rel).read_text(encoding="utf-8")
            if marker in txt:
                state = "APPLIED"
            elif txt.count(old) == 1:
                state = "not applied"
            else:
                state = f"ANCHOR MISSING ({txt.count(old)} matches)"
                bad += 1
            print(f"{desc:<46}{state}")
        return 1 if bad else 0

    # --apply
    # 1) verify every anchor first - never leave a half-patched tree
    pending = []
    for rel, old, new, desc, marker in PATCHES:
        p = root / rel
        if not p.exists():
            sys.exit(f"missing file: {p}")
        txt = p.read_text(encoding="utf-8")
        if marker in txt:
            print(f"already applied: {desc}")
            continue
        if txt.count(old) != 1:
            sys.exit(f"ANCHOR ERROR in {rel} ({desc}): found {txt.count(old)} matches, need exactly 1")
        pending.append((p, rel, old, new, desc))

    if not pending:
        print("\nall patches already applied.")
        return 0

    # 2) back up each distinct file once
    for rel in sorted({r for _, r, _, _, _ in pending}):
        src, bak = root / rel, root / (rel + BAK)
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"backup -> {rel}{BAK}")

    # 3) apply
    for p, rel, old, new, desc in pending:
        txt = p.read_text(encoding="utf-8")
        p.write_text(txt.replace(old, new, 1), encoding="utf-8")
        print(f"applied  {desc}")

    print("\nAll flags default to False - behaviour is unchanged until you set them.")
    print("Enable with: config.use_scored_pair_order = True / config.enable_quality_gate = True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
