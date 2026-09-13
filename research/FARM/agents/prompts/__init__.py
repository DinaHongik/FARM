"""
Prompt Templates for FARM Agents
================================

This module provides prompt templates for all agents in the negotiation system.
Prompts are designed for Granite 4 models with structured JSON output.
"""

from agents.prompts.trigger_agent import (
    TRIGGER_AGENT_SYSTEM_PROMPT,
    format_trigger_user_prompt,
)
from agents.prompts.action_agent import (
    ACTION_AGENT_SYSTEM_PROMPT,
    format_action_user_prompt,
)
from agents.prompts.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    format_verifier_user_prompt,
)

__all__ = [
    "TRIGGER_AGENT_SYSTEM_PROMPT",
    "format_trigger_user_prompt",
    "ACTION_AGENT_SYSTEM_PROMPT",
    "format_action_user_prompt",
    "VERIFIER_SYSTEM_PROMPT",
    "format_verifier_user_prompt",
]
