"""Tests for the three wiring fixes.

Each mechanism gets two tests: flag OFF must reproduce the published behaviour
exactly (so the paper's numbers stay reproducible), flag ON must exercise the
mechanism the paper describes.

No LLM, no GPU - this is all routing logic.
"""
import pytest


# ===========================================================================
# PATCH A - flags exist and are off by default
# ===========================================================================

def test_flags_exist_and_default_false(cfg):
    """A fresh config must behave exactly like the published run."""
    from agents.config import AgentConfig
    fresh = AgentConfig()
    assert fresh.use_scored_pair_order is False, "must default off to keep the published run reproducible"
    assert fresh.enable_quality_gate is False, "must default off to keep the published run reproducible"
    assert fresh.max_quality_retries >= 1


# ===========================================================================
# PATCH B - the cross-scorer's ranking actually drives pair selection
# ===========================================================================

def test_pair_order_defaults_to_legacy_grid(cfg, base_state):
    """Flag off -> the hardcoded lexicographic PAIR_ORDER, unchanged."""
    from agents.nodes.fallback import get_pair_order, PAIR_ORDER
    cfg.use_scored_pair_order = False
    assert get_pair_order(base_state) is PAIR_ORDER
    assert get_pair_order(base_state)[:3] == [(0, 0), (0, 1), (0, 2)]


def test_pair_order_uses_priority_queue_when_enabled(cfg, base_state):
    """Flag on -> the cross-scorer's scored ranking."""
    from agents.nodes.fallback import get_pair_order
    cfg.use_scored_pair_order = True
    order = get_pair_order(base_state)
    assert order == [(3, 2), (0, 4), (1, 1), (2, 0), (4, 3)]
    assert order[0] != (0, 0), "scored order must not collapse to the lexicographic grid"


def test_pair_order_survives_json_roundtrip(cfg, base_state):
    """priority_queue comes back from JSON as lists, not tuples."""
    from agents.nodes.fallback import get_pair_order
    cfg.use_scored_pair_order = True
    base_state["priority_queue"] = [[3, 2], [0, 4]]
    assert get_pair_order(base_state) == [(3, 2), (0, 4)]


def test_pair_order_falls_back_when_queue_empty(cfg, base_state):
    """A missing/empty queue must not crash the pipeline - degrade to legacy."""
    from agents.nodes.fallback import get_pair_order, PAIR_ORDER
    cfg.use_scored_pair_order = True
    base_state["priority_queue"] = []
    assert get_pair_order(base_state) is PAIR_ORDER
    del base_state["priority_queue"]
    assert get_pair_order(base_state) is PAIR_ORDER


def test_fallback_node_advances_along_scored_order(cfg, base_state):
    """The observable behaviour: retry #1 goes to the 2nd-best SCORED pair."""
    from agents.nodes.fallback import fallback_node
    cfg.use_scored_pair_order = True
    out = fallback_node(base_state)
    assert (out["current_trigger_idx"], out["current_action_idx"]) == (0, 4)
    assert out["pair_attempt"] == 1
    assert out["negotiation_status"] == "negotiating"


def test_fallback_node_legacy_behaviour_unchanged(cfg, base_state):
    """Flag off -> retry #1 is (0,1), exactly as in the published run."""
    from agents.nodes.fallback import fallback_node
    cfg.use_scored_pair_order = False
    out = fallback_node(base_state)
    assert (out["current_trigger_idx"], out["current_action_idx"]) == (0, 1)


def test_fallback_node_resets_negotiation_fields(cfg, base_state):
    """Stale offers must not leak across pairs."""
    from agents.nodes.fallback import fallback_node
    cfg.use_scored_pair_order = True
    out = fallback_node(base_state)
    for k in ("current_offer", "trigger_reasoning", "current_requirements",
              "action_reasoning", "binding_map"):
        assert out[k] is None


def test_fallback_exhaustion_reports_scored_length(cfg, base_state):
    """Running off the end of the scored queue fails cleanly."""
    from agents.nodes.fallback import fallback_node
    cfg.use_scored_pair_order = True
    base_state["pair_attempt"] = 4          # next_attempt = 5 == len(priority_queue)
    out = fallback_node(base_state)
    assert out["negotiation_status"] == "failed"
    assert "5 pairs exhausted" in out["error"]


def test_should_continue_respects_scored_length(cfg, base_state):
    """should_continue must measure against the SAME order fallback_node walks."""
    from agents.nodes.fallback import should_continue
    cfg.use_scored_pair_order = True
    base_state["negotiation_status"] = "rejected"
    base_state["pair_attempt"] = 0
    assert should_continue(base_state) == "continue"
    base_state["pair_attempt"] = 4          # queue has 5 entries -> nothing left
    assert should_continue(base_state) == "end"


# ===========================================================================
# PATCH C - the verifier can gate quality instead of being terminal
# ===========================================================================

def _router(graph):
    """Pull the verifier's routing callable out of a built StateGraph.

    langgraph stores the branch condition wrapped in a RunnableCallable, which is
    not itself callable - the plain function lives on `.func`.
    """
    br = getattr(graph, "branches", {}).get("verifier")
    assert br, "verifier has no conditional edges - PATCH C not applied"
    branch = list(br.values())[0]
    path = getattr(branch, "path", None) or getattr(branch, "condition", None)
    assert path is not None, "could not locate the branch condition"
    fn = getattr(path, "func", None)
    if fn is not None:
        return fn
    if callable(path):
        return path
    return path.invoke


def test_verifier_is_no_longer_unconditionally_terminal(cfg):
    from langgraph.graph import END
    from agents.graph import create_negotiation_graph
    g = create_negotiation_graph(use_simple=True)
    assert ("verifier", END) not in getattr(g, "edges", set()), \
        "verifier still has an unconditional edge to END"
    assert "verifier" in getattr(g, "branches", {}), "verifier should route conditionally"


def test_quality_gate_off_always_ends(cfg):
    """Flag off -> verifier always ends, even on a terrible score."""
    from agents.graph import create_negotiation_graph
    cfg.enable_quality_gate = False
    route = _router(create_negotiation_graph(use_simple=True))
    assert route({"verifier_score": 0.01, "pair_attempt": 0}) == "end"


def test_quality_gate_on_routes_low_scores_to_fallback(cfg):
    from agents.graph import create_negotiation_graph
    cfg.enable_quality_gate = True
    cfg.verifier_threshold = 0.6
    cfg.max_quality_retries = 2
    route = _router(create_negotiation_graph(use_simple=True))
    assert route({"verifier_score": 0.30, "pair_attempt": 0}) == "retry"
    assert route({"verifier_score": 0.90, "pair_attempt": 0}) == "end"


def test_quality_gate_cannot_loop_forever(cfg):
    """A persistently bad applet must stop retrying."""
    from agents.graph import create_negotiation_graph
    cfg.enable_quality_gate = True
    cfg.verifier_threshold = 0.6
    cfg.max_quality_retries = 2
    route = _router(create_negotiation_graph(use_simple=True))
    assert route({"verifier_score": 0.1, "pair_attempt": 2}) == "end"
    assert route({"verifier_score": 0.1, "pair_attempt": 9}) == "end"


def test_quality_gate_counter_is_a_declared_state_key():
    """The loop guard must be a field LangGraph will actually persist.

    apply_writes silently DROPS keys not declared in NegotiationState, so a
    bespoke counter would read back 0 every pass and loop until
    GraphRecursionError - which run_agentic_eval.py turns into an all-zeros
    score, i.e. it looks like a quality regression rather than a crash.
    """
    from agents.state import NegotiationState
    keys = set(getattr(NegotiationState, "__annotations__", {}))
    assert "pair_attempt" in keys, "loop guard must be a declared state field"
    assert "quality_retries" not in keys, \
        "if a dedicated counter is introduced it MUST be declared in NegotiationState"


def test_quality_gate_handles_missing_score(cfg):
    """No score computed -> end, never crash."""
    from agents.graph import create_negotiation_graph
    cfg.enable_quality_gate = True
    route = _router(create_negotiation_graph(use_simple=True))
    assert route({}) == "end"
    assert route({"verifier_score": None}) == "end"


def test_graph_still_compiles_with_gate_enabled(cfg):
    """A conditional edge into an upstream node must not break compilation."""
    from agents.graph import create_negotiation_graph
    cfg.enable_quality_gate = True
    create_negotiation_graph(use_simple=True).compile()


def test_run_negotiation_sets_a_recursion_limit():
    """PATCH E. Without headroom the gate loop hits LangGraph's default of 25
    supersteps and raises GraphRecursionError, which the eval harness converts
    into an all-zeros sample rather than surfacing the failure."""
    import inspect
    from agents import graph as graph_mod
    src = inspect.getsource(graph_mod.run_negotiation)
    assert "recursion_limit" in src, "run_negotiation must set a recursion_limit"


def test_recursion_limit_exceeds_worst_case_loop(cfg):
    """The limit must cover max_pairs retries, each costing ~4 supersteps."""
    import inspect
    from agents import graph as graph_mod
    from agents.config import config as c
    src = inspect.getsource(graph_mod.run_negotiation)
    ns = {"getattr": getattr, "_agent_cfg": c}
    line = [l for l in src.splitlines() if "_recursion_limit =" in l][0].strip()
    exec(line, ns)
    assert ns["_recursion_limit"] >= 6 + 4 * c.max_pairs, \
        f"recursion limit {ns['_recursion_limit']} too small for {c.max_pairs} retries"


# ===========================================================================
# PATCH D - extract_concept no longer emits mid-sentence fragments
# ===========================================================================

# Verbatim from results/e2e_eval.json sample test_000, which produced the
# trigger query "price rising significantly) 2. the".
REAL_REASONING = (
    "1. The user wants an applet that changes a light color based on stock price "
    "movement (TRIGGER: stock price rising significantly)\n"
    "2. The action is changing the color of a smart light to green when triggered\n"
    "3. To build this, I need to find APIs for monitoring stock prices"
)


def test_regression_no_mid_sentence_fragment():
    """The exact input that shipped a garbage query must not do so again."""
    from agents.prompts.planner import extract_concept
    got = extract_concept(REAL_REASONING, "trigger")
    assert got != "price rising significantly) 2. the"
    assert not got.endswith(" the"), f"still a mid-sentence window: {got!r}"
    assert "2." not in got, f"leaked a list marker from the reasoning: {got!r}"


def test_extracts_from_explicit_marker():
    from agents.prompts.planner import extract_concept
    text = "THINKING:\nTRIGGER: a new episode is published\nACTION: post a Discord message"
    assert extract_concept(text, "trigger") == "a new episode is published"
    assert extract_concept(text, "action") == "post a Discord message"


def test_returns_empty_when_no_marker():
    """Empty is correct - planner_node does `plan.get(...) or query`, so the
    caller falls back to the full user query, which measurably retrieves better."""
    from agents.prompts.planner import extract_concept
    assert extract_concept("Some prose with no structure at all.", "trigger") == ""


def test_substring_keyword_false_positives_are_gone():
    """'if' inside 'notify', 'new' inside 'renew' used to fire the old matcher."""
    from agents.prompts.planner import extract_concept
    for text in ("I will notify the user about this specific thing",
                 "We should renew the subscription in the window"):
        got = extract_concept(text, "trigger")
        assert got == "", f"substring false positive still fires: {got!r}"


def test_does_not_return_absurdly_long_text():
    from agents.prompts.planner import extract_concept
    got = extract_concept("TRIGGER: " + "x" * 500, "trigger")
    assert got == ""


# ===========================================================================
# Integration - the default configuration is still the published pipeline
# ===========================================================================

def test_default_config_reproduces_published_routing(cfg, base_state):
    """With no flags set, every routing decision matches the pre-patch code."""
    from agents.config import AgentConfig
    from agents.nodes.fallback import fallback_node, get_pair_order, PAIR_ORDER
    fresh = AgentConfig()
    cfg.use_scored_pair_order = fresh.use_scored_pair_order
    cfg.enable_quality_gate = fresh.enable_quality_gate

    assert get_pair_order(base_state) is PAIR_ORDER
    seen = []
    st = dict(base_state)
    for _ in range(4):
        out = fallback_node(st)
        seen.append((out["current_trigger_idx"], out["current_action_idx"]))
        st = {**st, "pair_attempt": out["pair_attempt"]}
    assert seen == [(0, 1), (0, 2), (0, 3), (0, 4)], "legacy grid walk changed"
