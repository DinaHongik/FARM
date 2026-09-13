"""
FARM Agentic AI System
======================

Contract-based multi-agent negotiation for executable trigger-action program synthesis.

This module implements a LangGraph-based agentic system where:
- Trigger Agent: Selects triggers and makes OFFERs (available ingredients)
- Action Agent: Selects actions and states REQUIREMENTS (required fields)
- Contract Verifier: Scores and critiques candidate applets

Architecture:
    User Query -> RAG Retrieval -> Agent Negotiation -> Executable Applet

Usage:
    from agents import run_negotiation

    result = run_negotiation("When darkness detected, log to spreadsheet")
    print(result.final_applet)
"""

from agents.state import NegotiationState
from agents.graph import create_negotiation_graph, run_negotiation
from agents.config import get_llm, AgentConfig

__version__ = "0.1.0"
__all__ = [
    "NegotiationState",
    "create_negotiation_graph",
    "run_negotiation",
    "get_llm",
    "AgentConfig",
]
