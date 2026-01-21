"""
FARM Agentic AI System
======================

Contract-based multi-agent resolution for executable trigger-action program synthesis.

This module implements a LangGraph-based agentic system where:
- Trigger Agent: Selects triggers and makes OFFERs (available ingredients)
- Action Agent: Selects actions and states REQUIREMENTS (required fields)
- Contract Verifier: Scores and critiques candidate applets

Architecture:
    User Query -> RAG Retrieval -> Agent Resolution -> Executable Applet

Usage:
    from agents import run_resolution

    result = run_resolution("When darkness detected, log to spreadsheet")
    print(result.final_applet)
"""

from agents.state import ResolutionState
from agents.graph import create_resolution_graph, run_resolution
from agents.config import get_llm, AgentConfig

__version__ = "0.1.0"
__all__ = [
    "ResolutionState",
    "create_resolution_graph",
    "run_resolution",
    "get_llm",
    "AgentConfig",
]
