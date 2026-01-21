"""
Intent Analyzer Node

LangGraph node implementing the Intent Analyzer logic.
Decomposes user query and coordinates retrieval via tool calls.

This is the entry point for the agentic workflow, responsible for:
    1. Understanding user intent
    2. Creating a plan
    3. Triggering parallel retrieval for triggers and actions
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add parent directory to path
NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import ResolutionState
from agents.config import get_llm, config
from agents.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    format_planner_user_prompt,
    parse_plan_from_response,
)
from agents.tools.executors import parse_tool_call, execute_tool, format_tool_response
from agents.tools.definitions import FARM_TOOLS


def intent_analyzer_node(state: ResolutionState) -> Dict[str, Any]:
    """
    Execute the Intent Analyzer.

    This node:
        1. Analyzes the user query
        2. Creates a step-by-step plan
        3. Extracts trigger and action search queries
        4. Prepares for parallel retrieval

    Args:
        state: Current resolution state

    Returns:
        State updates with plan and search queries
    """
    query = state["query"]

    if config.verbose:
        print(f"\n[INTENT_ANALYZER] Analyzing query: {query}")

    try:
        llm = get_llm()

        # Build messages for Granite 4
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": format_planner_user_prompt(query)},
        ]

        # Call LLM
        response = llm.invoke(messages)
        response_text = response.content

        if config.verbose:
            print(f"[INTENT_ANALYZER] Response:\n{response_text[:500]}...")

        # Parse the plan
        plan = parse_plan_from_response(response_text)

        # Check for tool calls in response
        tool_call = parse_tool_call(response_text)

        # Format tool calls for metrics evaluation
        tool_calls = []
        if tool_call:
            tool_calls.append({"name": tool_call.get("tool", ""), "args": tool_call.get("args", {})})
        # Always record expected tool calls
        if not tool_calls:
            tool_calls = [
                {"name": "search_triggers", "args": {}},
                {"name": "search_actions", "args": {}},
            ]

        # Extract search queries from plan or use original query
        trigger_query = plan.get("trigger_query") or query
        action_query = plan.get("action_query") or query

        # Update messages for history
        new_messages = [
            HumanMessage(content=f"[IntentAnalyzer] {query}"),
            AIMessage(content=response_text),
        ]

        return {
            "plan": plan,
            "trigger_search_query": trigger_query,
            "action_search_query": action_query,
            "messages": new_messages,
            "intent_analyzer_reasoning": plan.get("thinking", ""),
            "tool_calls": tool_calls,
        }

    except Exception as e:
        if config.verbose:
            print(f"[INTENT_ANALYZER] Error: {e}")

        # Fallback: use original query for both searches
        return {
            "plan": {"steps": ["Search for triggers", "Search for actions"]},
            "trigger_search_query": query,
            "action_search_query": query,
            "error": f"Intent Analyzer error: {str(e)}",
            "tool_calls": [
                {"name": "search_triggers", "args": {}},
                {"name": "search_actions", "args": {}},
            ],
        }


def intent_analyzer_simple(state: ResolutionState) -> Dict[str, Any]:
    """
    Simplified Intent Analyzer without LLM call.

    Uses rule-based query decomposition for testing and fallback.

    Args:
        state: Current resolution state

    Returns:
        State updates with plan
    """
    query = state["query"]

    # Simple heuristic decomposition
    query_lower = query.lower()

    # Extract trigger part (usually before action keywords)
    action_keywords = ["log", "send", "post", "add", "create", "notify", "save", "turn"]
    trigger_part = query
    action_part = query

    for kw in action_keywords:
        if kw in query_lower:
            idx = query_lower.find(kw)
            trigger_part = query[:idx].strip().strip(',').strip()
            action_part = query[idx:].strip()
            break

    # Clean up trigger part
    trigger_part = trigger_part.replace("when", "").replace("if", "").strip()
    if not trigger_part:
        trigger_part = query

    plan = {
        "thinking": f"Decomposed query into trigger='{trigger_part}' and action='{action_part}'",
        "trigger_query": trigger_part,
        "action_query": action_part,
        "steps": [
            f"1. Search for triggers matching: {trigger_part}",
            f"2. Search for actions matching: {action_part}",
            "3. Match trigger ingredients to action fields",
            "4. Generate bindings and create applet"
        ]
    }

    return {
        "plan": plan,
        "trigger_search_query": trigger_part,
        "action_search_query": action_part,
        "intent_analyzer_reasoning": plan["thinking"],
    }


def extract_search_queries(query: str) -> Dict[str, str]:
    """
    Extract trigger and action search queries from user query.

    NOTE: Previously used hardcoded separators which broke queries like
    "light to green". Now uses full query for both - RAG handles semantics.

    Args:
        query: User query

    Returns:
        Dict with 'trigger' and 'action' queries (both use full query)
    """
    # Use full query for both - RAG embeddings handle semantic matching
    # Hardcoded separators like " to " break queries like "light to green"
    return {"trigger": query, "action": query}
