"""
LangGraph StateGraph for FARM Multi-Agent Negotiation
=====================================================

Defines the WoT (Web of Things) multi-agent negotiation workflow
using LangGraph StateGraph with Granite 4 native features.

AGENTIC FLOW:
    START -> planner -> load_candidates -> cross_scorer -> trigger_agent
                                                                  |
                                                                  v
                                                           action_agent
                                                                  |
                                    +-----------------+-----------+----------+
                                    |                 |                      |
                                    v                 v                      v
                                [ACCEPT]          [REJECT]               [FAILED]
                                    |                 |                      |
                                    v                 v                      v
                                verifier          fallback                  END
                                    |                 |
                                    v                 |
                                   END <--------------+

NEW COMPONENTS:
- Planner Agent: Decomposes query, extracts search intents
- Cross-Scorer: Ranks all trigger-action pairs by compatibility
- Enhanced prompts: Chain-of-Thought + Granite 4 JSON Schema
- Best-first search: Uses priority queue from cross-scorer
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

# Add parent directory to path for module imports
AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import NegotiationState, create_initial_state
from agents.memory import get_checkpointer
from agents.nodes.load_candidates import load_candidates_node, load_candidates_mock
from agents.nodes.trigger_node import trigger_agent_node, trigger_agent_simple
from agents.nodes.action_node import action_agent_node, action_agent_simple
from agents.nodes.verifier_node import verifier_node, verifier_simple
from agents.nodes.fallback import fallback_node, should_continue
from agents.nodes.planner_node import planner_node, planner_simple
from agents.nodes.cross_scorer_node import cross_scorer_node, cross_scorer_simple
from agents.nodes.trigger_selector import trigger_selector_node, trigger_selector_simple
from agents.nodes.action_selector import action_selector_node, action_selector_simple


def create_negotiation_graph(use_mock: bool = False, use_simple: bool = False, use_planner: bool = True) -> StateGraph:
    """
    Create the negotiation StateGraph.

    Args:
        use_mock: If True, use mock candidates instead of RAG
        use_simple: If True, use simple (non-LLM) agent implementations
        use_planner: If True, include Planner Agent (default: True for full agentic)

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with state schema
    graph = StateGraph(NegotiationState)

    # Select node implementations based on mode
    planner_fn = planner_simple if use_simple else planner_node
    load_fn = load_candidates_mock if use_mock else load_candidates_node
    cross_scorer_fn = cross_scorer_simple if use_simple else cross_scorer_node
    trigger_fn = trigger_agent_simple if use_simple else trigger_agent_node
    action_fn = action_agent_simple if use_simple else action_agent_node
    verifier_fn = verifier_simple if use_simple else verifier_node

    # Add nodes
    if use_planner:
        graph.add_node("planner", planner_fn)
    graph.add_node("load_candidates", load_fn)
    graph.add_node("cross_scorer", cross_scorer_fn)
    graph.add_node("trigger_agent", trigger_fn)
    graph.add_node("action_agent", action_fn)
    graph.add_node("verifier", verifier_fn)
    graph.add_node("fallback", fallback_node)

    # Set entry point
    if use_planner:
        graph.set_entry_point("planner")
        # planner -> load_candidates
        graph.add_edge("planner", "load_candidates")
    else:
        graph.set_entry_point("load_candidates")

    # load_candidates -> cross_scorer (NEW: score all pairs first)
    graph.add_edge("load_candidates", "cross_scorer")

    # cross_scorer -> trigger_agent
    graph.add_edge("cross_scorer", "trigger_agent")

    # trigger_agent -> action_agent (always)
    graph.add_edge("trigger_agent", "action_agent")

    # action_agent -> conditional routing
    graph.add_conditional_edges(
        "action_agent",
        should_continue,
        {
            "verify": "verifier",
            "continue": "fallback",
            "end": END,
        }
    )

    # fallback -> conditional routing (either retry or end)
    def fallback_router(state):
        if state.get("negotiation_status") == "failed":
            return "end"
        return "retry"

    graph.add_conditional_edges(
        "fallback",
        fallback_router,
        {
            "retry": "trigger_agent",
            "end": END,
        }
    )

    # verifier -> END (always)
    graph.add_edge("verifier", END)

    return graph


def create_agentic_graph(use_mock: bool = False) -> StateGraph:
    """
    Create the FULL AGENTIC negotiation graph.

    This is the recommended entry point for the complete WoT
    multi-agent system with all Granite 4 features.

    Args:
        use_mock: If True, use mock candidates instead of RAG

    Returns:
        Compiled StateGraph with full agentic capabilities
    """
    return create_negotiation_graph(
        use_mock=use_mock,
        use_simple=False,  # Use LLM agents
        use_planner=True,  # Include planner
    )


def compile_graph(
    graph: StateGraph,
    use_memory: bool = True
) -> Any:
    """
    Compile the graph with optional checkpointer.

    Args:
        graph: StateGraph to compile
        use_memory: If True, add checkpointer for state persistence

    Returns:
        Compiled graph (CompiledGraph)
    """
    if use_memory:
        checkpointer = get_checkpointer()
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()


def run_negotiation(
    query: str,
    thread_id: Optional[str] = None,
    use_mock: bool = False,
    use_simple: bool = False,
    use_planner: bool = True,
    use_memory: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete negotiation workflow.

    This is the main entry point for the agentic system.

    Args:
        query: User query describing the desired applet
        thread_id: Optional thread ID for memory persistence
        use_mock: If True, use mock candidates
        use_simple: If True, use simple (non-LLM) agents
        use_planner: If True, include Planner Agent (default: True)
        use_memory: If True, persist state with checkpointer
        verbose: If True, print progress information

    Returns:
        Final state dictionary with results

    Example:
        result = run_negotiation("When darkness detected, log to spreadsheet")
        if result.get("final_applet"):
            print(result["final_applet"])
    """
    # Set verbose mode on global config for node access
    from agents.config import config as agent_config
    agent_config.verbose = verbose

    # Create and compile graph
    graph = create_negotiation_graph(
        use_mock=use_mock,
        use_simple=use_simple,
        use_planner=use_planner
    )
    compiled = compile_graph(graph, use_memory=use_memory)

    # Create initial state
    initial_state = create_initial_state(query)

    # Configure thread for memory
    config = {}
    if use_memory and thread_id:
        config = {"configurable": {"thread_id": thread_id}}
    elif use_memory:
        import uuid
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Run graph
    if verbose:
        print(f"{'='*60}")
        print(f"FARM WoT Multi-Agent Negotiation")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Mode: {'Agentic' if use_planner else 'Simple'}")
        print(f"{'='*60}\n")

    final_state = None

    if verbose:
        # Stream with progress output
        for event in compiled.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"\n{'─' * 60}")
                print(f"[{node_name.upper()}]")
                print(f"{'─' * 60}")

                # Planner output
                if node_output.get("planner_reasoning"):
                    reasoning = node_output['planner_reasoning']
                    print(f"\n  💭 REASONING:")
                    # Show full reasoning, nicely formatted
                    for line in reasoning.split('\n')[:10]:
                        if line.strip():
                            print(f"     {line.strip()}")
                    if len(reasoning.split('\n')) > 10:
                        print(f"     ...")

                # Cross-scorer output
                if node_output.get("pair_scores"):
                    scores = node_output["pair_scores"][:5]
                    print(f"\n  📊 TOP PAIRS BY COMPATIBILITY:")
                    for ps in scores:
                        t_idx, a_idx = ps["trigger_idx"], ps["action_idx"]
                        score = ps["score"]
                        coverage = ps.get("coverage", 0)
                        print(f"     (T{t_idx}, A{a_idx}): score={score:.3f}, coverage={coverage:.0%}")

                if node_output.get("priority_queue"):
                    queue = node_output["priority_queue"][:3]
                    print(f"  🎯 Priority Queue: {queue}")

                # Trigger output
                if node_output.get("current_offer"):
                    offer = node_output["current_offer"]
                    print(f"\n  📤 TRIGGER OFFER:")
                    print(f"     Service: {offer.get('service_name')}")
                    desc = offer.get('description', 'N/A') or 'N/A'
                    print(f"     Description: {desc[:60]}...")
                    ingredients = offer.get("ingredients", [])
                    if ingredients:
                        # Handle ingredients as list of dicts or strings
                        ing_names = []
                        for ing in ingredients:
                            if isinstance(ing, dict):
                                ing_names.append(ing.get("name", str(ing)))
                            else:
                                ing_names.append(str(ing))
                        print(f"     Ingredients: {', '.join(ing_names)}")

                if node_output.get("trigger_reasoning"):
                    print(f"\n  💭 TRIGGER REASONING:")
                    reasoning = node_output["trigger_reasoning"]
                    for line in reasoning.split('\n')[:5]:
                        if line.strip():
                            print(f"     {line.strip()}")

                # Action output
                if node_output.get("action_reasoning"):
                    print(f"\n  💭 ACTION REASONING:")
                    reasoning = node_output["action_reasoning"]
                    for line in reasoning.split('\n')[:5]:
                        if line.strip():
                            print(f"     {line.strip()}")

                if node_output.get("negotiation_status"):
                    status = node_output['negotiation_status']
                    emoji = {"accepted": "✅", "rejected": "❌", "negotiating": "🔄", "failed": "💔"}.get(status, "❓")
                    print(f"\n  {emoji} Status: {status.upper()}")

                if node_output.get("binding_map"):
                    bindings = node_output["binding_map"]
                    print(f"\n  🔗 BINDINGS ({len(bindings)} total):")
                    for b in bindings[:5]:
                        field = b.get("action_field", "?")
                        if b.get("source_type") == "ingredient":
                            ing = b.get("ingredient_name", "?")
                            print(f"     • {field} ← {{{{trigger.{ing}}}}}")
                        else:
                            val = b.get("static_value", "?")
                            print(f"     • {field} ← \"{val}\"")

                # Verifier output
                if node_output.get("verifier_score"):
                    score = node_output['verifier_score']
                    score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                    print(f"\n  📊 VERIFIER SCORE: [{score_bar}] {score:.2f}")

                if node_output.get("verifier_critique"):
                    critique = node_output["verifier_critique"]
                    print(f"  📝 Critique: {critique[:80]}...")

                if node_output.get("error"):
                    print(f"\n  ⚠️ Error: {node_output['error']}")

            final_state = node_output
    else:
        # Silent execution
        final_state = compiled.invoke(initial_state, config)

    return final_state


def display_final_applet(state: Dict[str, Any]) -> None:
    """
    Display the final applet in a nice format.

    Args:
        state: Final negotiation state containing the applet
    """
    applet = state.get("final_applet")
    if not applet:
        print("\n❌ No applet generated")
        return

    print("\n" + "=" * 70)
    print("📱 FINAL WoT APPLET")
    print("=" * 70)

    # Query
    print(f"\n📝 Query: {applet.get('query', 'N/A')}")

    # Trigger
    trigger = applet.get("trigger", {})
    print(f"\n{'─' * 70}")
    print("⚡ TRIGGER (When this happens...)")
    print(f"{'─' * 70}")
    print(f"  Service:     {trigger.get('service_name', 'N/A')}")
    print(f"  Category:    {trigger.get('category', 'N/A')}")
    print(f"  Description: {trigger.get('description', 'N/A')}")

    # Trigger ingredients
    ingredients = trigger.get("ingredients", [])
    if ingredients:
        print(f"\n  📤 Ingredients (data provided by trigger):")
        api_info = trigger.get("api_info", {})
        ing_details = api_info.get("Ingredients", {})
        for ing in ingredients:
            ing_info = ing_details.get(ing, {})
            ing_type = ing_info.get("type", "string")
            ing_example = ing_info.get("example", "")
            print(f"     • {ing}: {ing_type}")
            if ing_example:
                print(f"       Example: {ing_example}")

    # Action
    action = applet.get("action", {})
    print(f"\n{'─' * 70}")
    print("🎯 ACTION (Do this...)")
    print(f"{'─' * 70}")
    print(f"  Service:     {action.get('service_name', 'N/A')}")
    print(f"  Category:    {action.get('category', 'N/A')}")
    print(f"  Description: {action.get('description', 'N/A')}")

    # Action field values
    field_values = action.get("field_values", {})
    if field_values:
        print(f"\n  📥 Field Values (how action is configured):")
        for field, value in field_values.items():
            print(f"     • {field}: {value}")

    # Bindings
    bindings = applet.get("bindings", [])
    if bindings:
        print(f"\n{'─' * 70}")
        print("🔗 BINDINGS (How trigger connects to action)")
        print(f"{'─' * 70}")
        for i, binding in enumerate(bindings, 1):
            action_field = binding.get("action_field", "?")
            source_type = binding.get("source_type", "?")
            if source_type == "ingredient":
                ing_name = binding.get("ingredient_name", "?")
                print(f"  {i}. {action_field} ← {{{{trigger.{ing_name}}}}}")
            else:
                static_val = binding.get("static_value", "?")
                print(f"  {i}. {action_field} ← \"{static_val}\" (static)")

    # Metadata
    metadata = applet.get("metadata", {})
    print(f"\n{'─' * 70}")
    print("📊 VERIFICATION")
    print(f"{'─' * 70}")
    score = metadata.get("verifier_score", 0)
    is_exec = metadata.get("is_executable", False)
    rounds = metadata.get("negotiation_rounds", 0)
    critique = metadata.get("verifier_critique", "")

    # Score with visual indicator
    score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    status_emoji = "✅" if is_exec else "⚠️"

    print(f"  Score:    [{score_bar}] {score:.2f}")
    print(f"  Status:   {status_emoji} {'Executable' if is_exec else 'Needs Review'}")
    print(f"  Rounds:   {rounds} negotiation attempt(s)")
    if critique:
        print(f"  Critique: {critique[:100]}{'...' if len(critique) > 100 else ''}")

    print("\n" + "=" * 70)


def run_agentic_negotiation(
    query: str,
    verbose: bool = True,
    show_applet: bool = True,
) -> Dict[str, Any]:
    """
    Run the FULL AGENTIC negotiation with all Granite 4 features.

    This is the recommended entry point for production use.

    Features:
    - Planner Agent with Chain-of-Thought
    - Cross-Scorer for best-first search
    - Trigger Agent with OFFER protocol
    - Action Agent with JSON Schema validation
    - Verifier with Reflection

    Args:
        query: User query describing the desired WoT applet
        verbose: If True, print progress (default: True)
        show_applet: If True, display full applet at end (default: True)

    Returns:
        Final state with applet configuration

    Example:
        result = run_agentic_negotiation("When darkness detected, log to spreadsheet")
        print(f"Success: {result.get('negotiation_status') == 'accepted'}")
        print(f"Score: {result.get('verifier_score')}")
    """
    result = run_negotiation(
        query=query,
        use_mock=False,
        use_simple=False,
        use_planner=True,
        use_memory=True,
        verbose=verbose,
    )

    if show_applet:
        display_final_applet(result)

    return result


def run_negotiation_streaming(
    query: str,
    thread_id: Optional[str] = None,
    use_mock: bool = False,
):
    """
    Generator that yields state updates during negotiation.

    Useful for real-time UI updates or logging.

    Args:
        query: User query
        thread_id: Optional thread ID
        use_mock: If True, use mock candidates

    Yields:
        Tuple of (node_name, state_update) for each step
    """
    graph = create_negotiation_graph(use_mock=use_mock)
    compiled = compile_graph(graph, use_memory=True)

    initial_state = create_initial_state(query)

    import uuid
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    for event in compiled.stream(initial_state, config):
        for node_name, node_output in event.items():
            yield node_name, node_output


# Convenience function for quick testing
def quick_test(query: str = "When darkness detected, log to spreadsheet", show_applet: bool = True):
    """
    Quick test function with mock data.

    Args:
        query: Test query
        show_applet: If True, display full applet at end

    Returns:
        Final state
    """
    result = run_negotiation(
        query=query,
        use_mock=True,
        use_simple=True,
        use_memory=False,
        verbose=True,
    )

    if show_applet:
        display_final_applet(result)

    return result


# =============================================================================
# NEW DESIGN: Multi-Candidate Selection with Verifier-Driven Fallback
# =============================================================================

def create_selector_graph(use_mock: bool = False, use_simple: bool = False, use_reference: bool = False) -> StateGraph:
    """
    Create the NEW DESIGN graph with multi-candidate scoring.

    NEW FLOW:
        planner -> load_candidates -> trigger_selector -> action_selector -> verifier
                                                                               |
                                              +--------------------------------+
                                              |                                |
                                              v                                v
                                      [SCORE >= threshold]           [SCORE < threshold]
                                              |                                |
                                              v                                v
                                             END                       selector_fallback
                                                                              |
                                                                              v
                                                                       (try next best)

    Key differences from original:
    1. trigger_selector scores ALL candidates, picks best (not just top-1)
    2. action_selector scores ALL candidates, picks best (not just top-1)
    3. Verifier drives fallback: if score < config.verifier_threshold, try next pair

    Args:
        use_mock: If True, use mock candidates
        use_simple: If True, use simple (non-LLM) implementations
        use_reference: If True, use reference data loader (for evaluation without RAG)

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(NegotiationState)

    # Select implementations
    if use_reference:
        from agents.nodes.load_candidates import load_candidates_from_reference
        load_fn = load_candidates_from_reference
    elif use_mock:
        load_fn = load_candidates_mock
    else:
        load_fn = load_candidates_node
    trigger_selector_fn = trigger_selector_simple if use_simple else trigger_selector_node
    action_selector_fn = action_selector_simple if use_simple else action_selector_node
    verifier_fn = verifier_simple if use_simple else verifier_node

    # Add nodes
    graph.add_node("planner", planner_node if not use_simple else planner_simple)
    graph.add_node("load_candidates", load_fn)
    graph.add_node("trigger_selector", trigger_selector_fn)
    graph.add_node("action_selector", action_selector_fn)
    graph.add_node("verifier", verifier_fn)
    # NOTE: selector_fallback removed - we trust RAG selection, no fallback needed

    # Entry: planner
    graph.set_entry_point("planner")

    # Linear flow until action_selector
    graph.add_edge("planner", "load_candidates")
    graph.add_edge("load_candidates", "trigger_selector")
    graph.add_edge("trigger_selector", "action_selector")

    # MULTI-AGENT OFFER-VERIFY: Action Agent can REJECT and trigger retry
    def action_router(state):
        """Route based on Action Agent's verification result."""
        status = state.get("negotiation_status", "")
        pair_attempt = state.get("pair_attempt", 0)
        max_attempts = 3  # Try up to 3 pairs

        if status == "rejected" and pair_attempt < max_attempts:
            # Action Agent rejected - try next pair
            return "retry"
        else:
            # Accepted or max attempts reached
            return "proceed"

    graph.add_conditional_edges(
        "action_selector",
        action_router,
        {
            "proceed": "verifier",
            "retry": "trigger_selector",  # Try next pair
        }
    )

    # Verifier -> Conditional fallback when LLM override fails
    #
    # PRINCIPLED APPROACH:
    # - RAG has 90%+ R@5 accuracy for semantic matching (TRAINED model)
    # - LLM can override RAG when it has good reasoning (test_005)
    # - But LLM can also be wrong due to missing domain knowledge (test_008)
    # - Solution: Use verifier score to detect bad LLM overrides
    # - If verifier score < 0.5 AND llm_overrode_rag, try RAG's top choice

    graph.add_node("llm_fallback", llm_fallback_node)

    def verifier_router(state):
        """Route based on verifier score and whether LLM overrode RAG."""
        score = state.get("verifier_score", 1.0)
        llm_overrode = state.get("llm_overrode_rag", False)
        already_tried_fallback = state.get("tried_llm_fallback", False)

        # If verifier score is low AND LLM overrode RAG AND haven't tried fallback yet
        if score < 0.5 and llm_overrode and not already_tried_fallback:
            return "fallback"
        return "end"

    graph.add_conditional_edges(
        "verifier",
        verifier_router,
        {
            "fallback": "llm_fallback",
            "end": END,
        }
    )

    # LLM fallback -> back to verifier to check RAG's choice
    graph.add_edge("llm_fallback", "verifier")

    return graph


def llm_fallback_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Fallback node when LLM's action override fails verification.

    When verifier score is low AND llm_overrode_rag is True, this node:
    1. Switches to RAG's top action (index 0)
    2. Rebuilds bindings for RAG's choice
    3. Marks that we've tried this fallback to prevent loops

    This allows the system to recover when LLM made a bad override decision.
    """
    from agents.config import config
    from agents.nodes.action_selector import get_required_fields

    action_candidates = state.get("action_candidates", [])
    current_offer = state.get("current_offer", {})

    if not action_candidates:
        return {"tried_llm_fallback": True}

    # Switch to RAG's top action (index 0)
    rag_top_action = action_candidates[0]

    if config.verbose:
        llm_action_idx = state.get("current_action_idx", 0)
        llm_action_name = action_candidates[llm_action_idx].get("service_name", "?") if llm_action_idx < len(action_candidates) else "?"
        rag_action_name = rag_top_action.get("service_name", "?")
        print(f"\n    [LLM Fallback] Verifier rejected LLM pick '{llm_action_name}'")
        print(f"    [LLM Fallback] Trying RAG top: '{rag_action_name}'")

    # Rebuild bindings for RAG's choice
    ingredients = current_offer.get("ingredients", [])
    api_info = rag_top_action.get("api_info", {})
    required_fields = get_required_fields(api_info)

    bindings = []
    used_ingredients = set()

    for field in required_fields:
        field_lower = field.lower()
        matched = False

        for ing in ingredients:
            ing_name = ing.get("name", "")
            if ing_name in used_ingredients:
                continue
            ing_lower = ing_name.lower()

            if (ing_lower in field_lower or field_lower in ing_lower or
                any(w in field_lower for w in ing_lower.split("_")) or
                any(w in ing_lower for w in field_lower.split("_"))):
                bindings.append({
                    "field": field,
                    "source": "ingredient",
                    "ingredient": ing_name,
                })
                used_ingredients.add(ing_name)
                matched = True
                break

        if not matched:
            bindings.append({
                "field": field,
                "source": "static",
                "value": f"[User provides {field}]",
            })

    return {
        "current_action_idx": 0,  # RAG top
        "llm_overrode_rag": False,  # No longer an override
        "tried_llm_fallback": True,  # Prevent loops
        "binding_map": bindings,
        "current_requirements": {
            "service_name": rag_top_action.get("service_name", ""),
            "category": rag_top_action.get("category", ""),
            "required_fields": required_fields,
        },
    }


def selector_fallback_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Fallback node for selector-based graph.

    When verifier score is too low, this node:
    1. Adds current (trigger, action) pair to tried_pairs
    2. Increments pair_attempt
    3. Prepares state for re-selection

    The trigger_selector will find the best trigger with untried actions.
    The action_selector will find the best untried action for that trigger.
    """
    current_trigger = state.get("current_trigger_idx", 0)
    current_action = state.get("current_action_idx", 0)
    pair_attempt = state.get("pair_attempt", 0)

    tried_pairs = list(state.get("tried_pairs") or [])

    # Mark current pair as tried
    current_pair = (current_trigger, current_action)
    if current_pair not in tried_pairs:
        tried_pairs.append(current_pair)

    from agents.config import config
    if config.verbose:
        print(f"\n    [Selector Fallback] Trying next pair...")
        print(f"    Tried pairs: {tried_pairs}")

    return {
        "tried_pairs": tried_pairs,
        "pair_attempt": pair_attempt + 1,
        "negotiation_status": "negotiating",  # Reset status
    }


def run_selector_negotiation(
    query: str,
    verbose: bool = True,
    show_applet: bool = True,
) -> Dict[str, Any]:
    """
    Run the NEW DESIGN negotiation with multi-candidate scoring.

    This version:
    1. Scores ALL trigger candidates against query, picks best
    2. Scores ALL action candidates against query + trigger, picks best
    3. Uses verifier score to drive fallback (if < threshold, try next best)

    Expected improvement: ~90% accuracy (R@5) vs ~55% (R@1)

    Args:
        query: User query
        verbose: Print progress
        show_applet: Display final applet

    Returns:
        Final state with applet
    """
    from agents.config import config as agent_config
    agent_config.verbose = verbose

    graph = create_selector_graph(use_mock=False, use_simple=False)
    compiled = compile_graph(graph, use_memory=True)

    initial_state = create_initial_state(query)

    import uuid
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 50,  # Allow up to 50 node visits (3 attempts * ~5 nodes + overhead)
    }

    if verbose:
        print(f"{'='*60}")
        print(f"FARM WoT Multi-Agent (NEW SELECTOR DESIGN)")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

    final_state = None

    if verbose:
        # Accumulate full state from all nodes (not just last node's output)
        accumulated_state = dict(initial_state)

        for event in compiled.stream(initial_state, config):
            for node_name, node_output in event.items():
                # Accumulate state from each node
                accumulated_state.update(node_output)

                print(f"\n{'─' * 40}")
                print(f"[{node_name.upper()}]")

                if node_output.get("trigger_scores"):
                    print(f"  Trigger scores: {node_output['trigger_scores']}")

                if node_output.get("action_scores"):
                    print(f"  Action scores: {node_output['action_scores']}")

                if node_output.get("current_trigger_idx") is not None:
                    print(f"  Selected trigger: #{node_output['current_trigger_idx']}")

                if node_output.get("current_action_idx") is not None:
                    print(f"  Selected action: #{node_output['current_action_idx']}")

                if node_output.get("verifier_score"):
                    score = node_output['verifier_score']
                    status = "✓ Good" if score >= agent_config.verifier_threshold else "✗ Try again"
                    print(f"  Verifier score: {score:.2f} {status}")

                if node_output.get("pair_attempt"):
                    print(f"  Attempt: {node_output['pair_attempt']}")

                if node_output.get("error"):
                    print(f"  ⚠️ Error: {node_output['error']}")

        final_state = accumulated_state
    else:
        final_state = compiled.invoke(initial_state, config)

    if show_applet and final_state:
        display_final_applet(final_state)

    return final_state


def run_selector_with_reference(
    query: str,
    test_data: list,
    verbose: bool = False,
) -> dict:
    """
    Run selector negotiation using reference test data (no RAG needed).

    This is useful for evaluation when RAG is unavailable:
    1. Loads candidates from test data (correct + distractors)
    2. Tests selector logic without requiring RAG system
    3. Uses simple (non-LLM) implementations

    Args:
        query: User query
        test_data: List of test cases with trigger/action info
        verbose: Print progress

    Returns:
        Final state with applet
    """
    from agents.config import config as agent_config
    from agents.nodes.load_candidates import set_eval_reference_data

    agent_config.verbose = verbose

    # Set reference data for loader
    set_eval_reference_data(test_data)

    # Create graph with reference loader and simple implementations
    graph = create_selector_graph(use_mock=False, use_simple=True, use_reference=True)
    compiled = compile_graph(graph, use_memory=False)

    initial_state = create_initial_state(query)

    import uuid
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 50,
    }

    final_state = compiled.invoke(initial_state, config)

    return final_state
