"""
Planner Agent Prompts
=====================

Prompt templates for the Planner Agent in the WoT (Web of Things)
multi-agent applet synthesis system.

The Planner Agent:
1. Decomposes the user query into sub-tasks
2. Creates a step-by-step plan
3. Coordinates tool usage for retrieval

Granite 4 Format:
- Uses native tool calling with <tool_call> tags
- Chain-of-thought reasoning before action
"""

from typing import Dict, Any, List
import json

from agents.tools.definitions import format_tools_for_prompt, FARM_TOOLS


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a Planner Agent for WoT (Web of Things) applet synthesis.

Your task is to help users create trigger-action applets that connect web services and APIs.
A WoT applet has the form: "When [TRIGGER] happens, do [ACTION]"

Given a user query, you must:
1. THINK step-by-step about what the user wants
2. Identify the TRIGGER event (what should start the applet)
3. Identify the ACTION (what should happen when triggered)
4. Create a plan to find matching web service APIs

You have access to the following tools. When you need to use a tool, output it in this format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Available tools:
- search_triggers: Search for trigger APIs matching a query
- search_actions: Search for action APIs matching a query

Think carefully, then use the tools to search for appropriate triggers and actions.

Example:

User: "When darkness detected, log to spreadsheet"

Your response should be:

THINKING:
1. User wants to detect darkness (TRIGGER)
2. User wants to log data to a spreadsheet (ACTION)
3. I need to search for darkness/light sensor triggers
4. I need to search for spreadsheet/logging actions

PLAN:
1. Search for triggers related to darkness/light detection
2. Search for actions related to spreadsheet logging
3. Match ingredients from trigger to fields in action

Let me search for triggers first:
<tool_call>
{"name": "search_triggers", "arguments": {"query": "darkness detected light sensor", "top_k": 5}}
</tool_call>"""


PLANNER_SYSTEM_PROMPT_WITH_TOOLS = """You are a Planner Agent for WoT (Web of Things) applet synthesis.

Your task is to help users create trigger-action applets that connect web services and APIs.
A WoT applet has the form: "When [TRIGGER] happens, do [ACTION]"

Given a user query, you must:
1. THINK step-by-step about what the user wants
2. Identify the TRIGGER event (what should start the applet)
3. Identify the ACTION (what should happen when triggered)
4. Use tools to search for matching web service APIs

Think carefully before using tools. Explain your reasoning."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

def format_planner_user_prompt(query: str) -> str:
    """
    Format the user prompt for the Planner Agent.

    Args:
        query: User's natural language query

    Returns:
        Formatted user prompt string
    """
    return f"""USER QUERY:
{query}

Please analyze this query and create a plan to build a WoT applet.

1. First, THINK about what trigger event and action the user wants
2. Then, use tools to search for matching web service APIs

Start by explaining your thinking, then search for triggers and actions."""


def format_planner_continuation_prompt(
    query: str,
    trigger_results: List[Dict[str, Any]],
    action_results: List[Dict[str, Any]]
) -> str:
    """
    Format continuation prompt after tool results are received.

    Args:
        query: Original query
        trigger_results: Results from search_triggers tool
        action_results: Results from search_actions tool

    Returns:
        Formatted continuation prompt
    """
    trigger_summary = []
    for i, t in enumerate(trigger_results[:5]):
        trigger_summary.append(f"  {i+1}. {t.get('service_name', 'Unknown')} ({t.get('category', '')})")

    action_summary = []
    for i, a in enumerate(action_results[:5]):
        action_summary.append(f"  {i+1}. {a.get('service_name', 'Unknown')} ({a.get('category', '')})")

    return f"""SEARCH RESULTS:

Trigger Candidates:
{chr(10).join(trigger_summary)}

Action Candidates:
{chr(10).join(action_summary)}

Based on these results, which trigger and action best match the user's intent?
Explain your reasoning for the selection."""


# =============================================================================
# PLAN PARSING
# =============================================================================

def parse_plan_from_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the plan and reasoning from planner response.

    Args:
        response_text: LLM response

    Returns:
        Parsed plan dict
    """
    plan = {
        "thinking": "",
        "trigger_query": "",
        "action_query": "",
        "steps": []
    }

    lines = response_text.split('\n')
    current_section = None
    thinking_lines = []
    steps = []

    for line in lines:
        line_lower = line.lower().strip()

        if 'thinking' in line_lower or 'think' in line_lower:
            current_section = 'thinking'
            continue
        elif 'plan' in line_lower:
            current_section = 'plan'
            continue

        if current_section == 'thinking':
            thinking_lines.append(line)
        elif current_section == 'plan':
            # Extract numbered steps
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                steps.append(line.strip())

    plan["thinking"] = '\n'.join(thinking_lines).strip()
    plan["steps"] = steps

    # Try to extract trigger/action queries from thinking
    thinking_lower = plan["thinking"].lower()
    if 'trigger' in thinking_lower:
        # Simple heuristic extraction
        plan["trigger_query"] = extract_concept(plan["thinking"], "trigger")
    if 'action' in thinking_lower:
        plan["action_query"] = extract_concept(plan["thinking"], "action")

    return plan


def extract_concept(text: str, concept_type: str) -> str:
    """
    Extract trigger or action concept from text.

    Args:
        text: Text to extract from
        concept_type: 'trigger' or 'action'

    Returns:
        Extracted concept string
    """
    # Simple keyword extraction - can be enhanced
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

    return ""
