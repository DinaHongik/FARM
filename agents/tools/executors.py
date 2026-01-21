"""
Tool Executors
==============

Execute tools by connecting to the existing fine-tuned RAG system.
These executors bridge Granite 4's tool calls to your RAG retriever.

IMPORTANT: This file does NOT modify the RAG system.
           It only USES the existing retriever interface.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None


class ToolExecutor:
    """
    Executes tools by interfacing with the existing RAG system.

    This class provides the bridge between Granite 4's tool calls
    and your fine-tuned retrieval system.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        """
        Initialize executor with optional state reference.

        Args:
            state: Reference to ResolutionState for accessing candidates
        """
        self.state = state or {}
        self._retriever = None
        self._triggers_loaded = False
        self._actions_loaded = False

    @property
    def retriever(self):
        """Lazy-load the retriever to avoid import issues."""
        if self._retriever is None:
            try:
                from rag.retriever import FARMRetriever
                self._retriever = FARMRetriever()
            except ImportError:
                # Fallback if retriever not available
                self._retriever = None
        return self._retriever

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dict

        Returns:
            ToolResult with success status and data
        """
        executor_map = {
            "search_triggers": self._search_triggers,
            "search_actions": self._search_actions,
            "get_trigger_schema": self._get_trigger_schema,
            "get_action_schema": self._get_action_schema,
            "check_compatibility": self._check_compatibility,
            "create_binding": self._create_binding,
            "finalize_applet": self._finalize_applet,
        }

        executor = executor_map.get(tool_name)
        if executor is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )

        try:
            result = executor(**arguments)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution error: {str(e)}"
            )

    def _search_triggers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for triggers using the fine-tuned RAG system.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of trigger candidates
        """
        if self.retriever is None:
            # Return from state if retriever not available
            return self.state.get("trigger_candidates", [])[:top_k]

        # Use the existing fine-tuned retriever
        results = self.retriever.search_triggers(query, top_k=top_k)

        # Format results
        candidates = []
        for r in results:
            candidates.append({
                "service_name": r.get("service_name", ""),
                "category": r.get("category", ""),
                "description": r.get("description", ""),
                "api_info": r.get("api_info", {}),
                "score": r.get("score", 0.0)
            })

        return candidates

    def _search_actions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for actions using the fine-tuned RAG system.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of action candidates
        """
        if self.retriever is None:
            # Return from state if retriever not available
            return self.state.get("action_candidates", [])[:top_k]

        # Use the existing fine-tuned retriever
        results = self.retriever.search_actions(query, top_k=top_k)

        # Format results
        candidates = []
        for r in results:
            candidates.append({
                "service_name": r.get("service_name", ""),
                "category": r.get("category", ""),
                "description": r.get("description", ""),
                "api_info": r.get("api_info", {}),
                "score": r.get("score", 0.0)
            })

        return candidates

    def _get_trigger_schema(
        self,
        trigger_index: int,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get full schema for a trigger candidate.

        Args:
            trigger_index: Index in candidate list
            service_name: Optional service name filter

        Returns:
            Trigger schema with ingredients
        """
        candidates = self.state.get("trigger_candidates", [])

        if trigger_index < 0 or trigger_index >= len(candidates):
            return {"error": f"Invalid trigger index: {trigger_index}"}

        trigger = candidates[trigger_index]
        api_info = trigger.get("api_info", {})

        # Extract ingredients
        ingredients_raw = api_info.get("Ingredients", {})
        ingredients = []
        for name, info in ingredients_raw.items():
            if isinstance(info, dict):
                ingredients.append({
                    "name": name,
                    "slug": info.get("Slug", name),
                    "type": info.get("Type", "String"),
                    "example": info.get("Example", "")
                })

        return {
            "service_name": trigger.get("service_name", ""),
            "category": trigger.get("category", ""),
            "description": trigger.get("description", ""),
            "ingredients": ingredients
        }

    def _get_action_schema(
        self,
        action_index: int,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get full schema for an action candidate.

        Args:
            action_index: Index in candidate list
            service_name: Optional service name filter

        Returns:
            Action schema with required fields
        """
        candidates = self.state.get("action_candidates", [])

        if action_index < 0 or action_index >= len(candidates):
            return {"error": f"Invalid action index: {action_index}"}

        action = candidates[action_index]
        api_info = action.get("api_info", {})

        # Extract fields
        fields_raw = api_info.get("Action fields", {})
        fields = []
        for name, info in fields_raw.items():
            if isinstance(info, dict):
                required = info.get("Required", "false").lower() == "true"
                fields.append({
                    "name": name,
                    "slug": info.get("Slug", name),
                    "label": info.get("Label", name),
                    "required": required,
                    "helper_text": info.get("Helper text", "")
                })

        return {
            "service_name": action.get("service_name", ""),
            "category": action.get("category", ""),
            "description": action.get("description", ""),
            "fields": fields
        }

    def _check_compatibility(
        self,
        trigger_index: int,
        action_index: int
    ) -> Dict[str, Any]:
        """
        Check compatibility between trigger and action.

        Args:
            trigger_index: Trigger candidate index
            action_index: Action candidate index

        Returns:
            Compatibility analysis
        """
        trigger_schema = self._get_trigger_schema(trigger_index)
        action_schema = self._get_action_schema(action_index)

        if "error" in trigger_schema or "error" in action_schema:
            return {
                "compatible": False,
                "score": 0.0,
                "error": trigger_schema.get("error") or action_schema.get("error")
            }

        # Get ingredients and required fields
        ingredients = trigger_schema.get("ingredients", [])
        fields = action_schema.get("fields", [])
        required_fields = [f for f in fields if f.get("required", False)]

        ingredient_names = {ing["name"].lower() for ing in ingredients}
        ingredient_names.update({ing.get("slug", "").lower() for ing in ingredients})

        # Check coverage
        covered_fields = []
        missing_fields = []

        for field in required_fields:
            field_name_lower = field["name"].lower()
            field_slug_lower = field.get("slug", "").lower()

            # Check if any ingredient matches
            matched = False
            for ing_name in ingredient_names:
                if (ing_name in field_name_lower or
                    field_name_lower in ing_name or
                    ing_name in field_slug_lower or
                    field_slug_lower in ing_name):
                    covered_fields.append(field["name"])
                    matched = True
                    break

            if not matched:
                missing_fields.append(field["name"])

        # Calculate score
        total_required = len(required_fields)
        if total_required == 0:
            score = 1.0
        else:
            score = len(covered_fields) / total_required

        return {
            "compatible": score >= 0.5 or len(missing_fields) == 0,
            "score": round(score, 3),
            "trigger_service": trigger_schema.get("service_name"),
            "action_service": action_schema.get("service_name"),
            "ingredients_count": len(ingredients),
            "required_fields_count": total_required,
            "covered_fields": covered_fields,
            "missing_fields": missing_fields,
            "note": "Missing fields can be filled with static values"
        }

    def _create_binding(
        self,
        action_field: str,
        source_type: str,
        ingredient_name: Optional[str] = None,
        static_value: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a field binding.

        Args:
            action_field: Field to bind
            source_type: 'ingredient' or 'static'
            ingredient_name: Source ingredient (if ingredient)
            static_value: Static value (if static)

        Returns:
            Binding dict
        """
        binding = {
            "action_field": action_field,
            "source_type": source_type,
        }

        if source_type == "ingredient":
            binding["ingredient_name"] = ingredient_name
            binding["static_value"] = None
        else:
            binding["ingredient_name"] = None
            binding["static_value"] = static_value

        return binding

    def _finalize_applet(
        self,
        trigger_index: int,
        action_index: int,
        bindings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Finalize the applet configuration.

        Args:
            trigger_index: Selected trigger
            action_index: Selected action
            bindings: List of field bindings

        Returns:
            Complete applet configuration
        """
        trigger_schema = self._get_trigger_schema(trigger_index)
        action_schema = self._get_action_schema(action_index)

        return {
            "trigger": {
                "service_name": trigger_schema.get("service_name"),
                "category": trigger_schema.get("category"),
                "ingredients": trigger_schema.get("ingredients", [])
            },
            "action": {
                "service_name": action_schema.get("service_name"),
                "category": action_schema.get("category"),
                "fields": action_schema.get("fields", [])
            },
            "bindings": bindings,
            "is_complete": True
        }

    def update_state(self, state: Dict[str, Any]):
        """Update the state reference."""
        self.state = state


def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """
    Convenience function to execute a tool.

    Args:
        tool_name: Tool to execute
        arguments: Tool arguments
        state: Optional state reference

    Returns:
        ToolResult
    """
    executor = ToolExecutor(state)
    return executor.execute(tool_name, arguments)


def parse_tool_call(response_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse Granite 4 tool call from response.

    Granite 4 returns tool calls in format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>

    Args:
        response_text: LLM response text

    Returns:
        Tuple of (tool_name, arguments) or None
    """
    # Match tool_call tags
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    match = re.search(pattern, response_text, re.DOTALL)

    if not match:
        return None

    try:
        tool_call = json.loads(match.group(1))
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        return (name, arguments)
    except json.JSONDecodeError:
        return None


def format_tool_response(tool_name: str, result: ToolResult) -> str:
    """
    Format tool result as Granite 4 tool response.

    Args:
        tool_name: Name of the tool that was called
        result: Tool execution result

    Returns:
        Formatted response string
    """
    if result.success:
        content = json.dumps(result.data, indent=2)
    else:
        content = json.dumps({"error": result.error})

    return f"<tool_response>\n{content}\n</tool_response>"
