"""
Agent Nodes for FARM WoT Multi-Agent System

LangGraph nodes that implement the agent logic.
Each node is a function that takes state and returns updated state.

Agentic Nodes:
    - intent_analyzer_node: Decomposes query and creates plan
    - load_candidates_node: Retrieves from fine-tuned RAG
    - cross_scorer_node: Scores all trigger-action pairs
    - trigger_agent_node: Selects trigger and makes OFFER
    - action_agent_node: Validates and generates bindings
    - verifier_node: Evaluates applet with reflection
    - fallback_node: Handles retries with priority queue
"""

from agents.nodes.load_candidates import load_candidates_node, load_candidates_mock
from agents.nodes.trigger_node import trigger_agent_node, trigger_agent_simple
from agents.nodes.action_node import action_agent_node, action_agent_simple
from agents.nodes.verifier_node import verifier_node, verifier_simple
from agents.nodes.fallback import fallback_node, should_continue
from agents.nodes.intent_analyzer_node import intent_analyzer_node, intent_analyzer_simple
from agents.nodes.cross_scorer_node import cross_scorer_node, cross_scorer_simple

__all__ = [
    # Core nodes
    "load_candidates_node",
    "load_candidates_mock",
    "trigger_agent_node",
    "trigger_agent_simple",
    "action_agent_node",
    "action_agent_simple",
    "verifier_node",
    "verifier_simple",
    "fallback_node",
    "should_continue",
    # Agentic nodes
    "intent_analyzer_node",
    "intent_analyzer_simple",
    "cross_scorer_node",
    "cross_scorer_simple",
]
