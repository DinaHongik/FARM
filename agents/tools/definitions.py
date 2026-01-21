"""
Granite 4 Native Tool Definitions
=================================

Tool schemas following OpenAI function calling format,
which Granite 4 natively supports with <tool_call> tags.

These tools connect to the existing fine-tuned RAG system
without modifying any RAG or training code.
"""

from typing import Dict, Any, List, Optional


# =============================================================================
# TOOL DEFINITIONS (Granite 4 / OpenAI Format)
# =============================================================================

FARM_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_triggers",
            "description": "Search for trigger APIs from web services that match the user's intent. Returns top-K trigger candidates with their schemas and ingredients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the trigger event to search for (e.g., 'darkness detected', 'new email received', 'location changed')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of trigger candidates to retrieve",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_actions",
            "description": "Search for action APIs from web services that match the user's intent. Returns top-K action candidates with their schemas and required fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the action to search for (e.g., 'log to spreadsheet', 'send notification', 'post to social media')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of action candidates to retrieve",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trigger_schema",
            "description": "Get the full API schema for a specific trigger, including all available ingredients (data fields) it provides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The trigger service name to get schema for"
                    },
                    "trigger_index": {
                        "type": "integer",
                        "description": "Index of the trigger in the candidate list (0-4)"
                    }
                },
                "required": ["trigger_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_action_schema",
            "description": "Get the full API schema for a specific action, including all required fields it needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The action service name to get schema for"
                    },
                    "action_index": {
                        "type": "integer",
                        "description": "Index of the action in the candidate list (0-4)"
                    }
                },
                "required": ["action_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_compatibility",
            "description": "Check if a trigger's ingredients can satisfy an action's required fields. Returns compatibility score and missing fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_index": {
                        "type": "integer",
                        "description": "Index of the trigger candidate (0-4)"
                    },
                    "action_index": {
                        "type": "integer",
                        "description": "Index of the action candidate (0-4)"
                    }
                },
                "required": ["trigger_index", "action_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_binding",
            "description": "Create a binding between a trigger ingredient and an action field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_field": {
                        "type": "string",
                        "description": "The action field to bind"
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["ingredient", "static"],
                        "description": "Whether to bind from trigger ingredient or static value"
                    },
                    "ingredient_name": {
                        "type": "string",
                        "description": "Name of the trigger ingredient (if source_type is 'ingredient')"
                    },
                    "static_value": {
                        "type": "string",
                        "description": "Static value to use (if source_type is 'static')"
                    }
                },
                "required": ["action_field", "source_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_applet",
            "description": "Finalize the applet with selected trigger, action, and bindings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_index": {
                        "type": "integer",
                        "description": "Index of the selected trigger"
                    },
                    "action_index": {
                        "type": "integer",
                        "description": "Index of the selected action"
                    },
                    "bindings": {
                        "type": "array",
                        "description": "List of field bindings",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_field": {"type": "string"},
                                "source_type": {"type": "string"},
                                "ingredient_name": {"type": "string"},
                                "static_value": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["trigger_index", "action_index", "bindings"]
            }
        }
    }
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a tool definition by name.

    Args:
        name: Tool function name

    Returns:
        Tool definition dict or None if not found
    """
    for tool in FARM_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """Get list of all tool names."""
    return [tool["function"]["name"] for tool in FARM_TOOLS]


def format_tools_for_prompt() -> str:
    """
    Format tools as string for inclusion in prompts.
    Used when not using Granite 4's native tool handling.
    """
    lines = ["Available tools:"]
    for tool in FARM_TOOLS:
        func = tool["function"]
        lines.append(f"\n- {func['name']}: {func['description']}")
        if "parameters" in func:
            params = func["parameters"].get("properties", {})
            if params:
                lines.append("  Parameters:")
                for param_name, param_info in params.items():
                    required = param_name in func["parameters"].get("required", [])
                    req_str = " (required)" if required else ""
                    lines.append(f"    - {param_name}: {param_info.get('description', '')}{req_str}")
    return "\n".join(lines)


# =============================================================================
# JSON SCHEMAS FOR GRANITE 4 STRUCTURED OUTPUT
# =============================================================================

TRIGGER_OFFER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning about why this trigger matches the query"
        },
        "service_name": {
            "type": "string",
            "description": "Selected trigger service name"
        },
        "category": {
            "type": "string",
            "description": "Trigger category"
        },
        "ingredients": {
            "type": "array",
            "description": "Available ingredients (the OFFER)",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "slug": {"type": "string"},
                    "type": {"type": "string"},
                    "example": {"type": "string"}
                },
                "required": ["name", "slug"]
            }
        }
    },
    "required": ["reasoning", "service_name", "ingredients"]
}

ACTION_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning about compatibility"
        },
        "decision": {
            "type": "string",
            "enum": ["ACCEPT", "REJECT"],
            "description": "Whether to accept or reject this trigger-action pair"
        },
        "service_name": {
            "type": "string",
            "description": "Selected action service name"
        },
        "required_fields": {
            "type": "array",
            "description": "Fields required by this action",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "slug": {"type": "string"},
                    "required": {"type": "boolean"}
                }
            }
        },
        "bindings": {
            "type": "array",
            "description": "Field bindings (if ACCEPT)",
            "items": {
                "type": "object",
                "properties": {
                    "action_field": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["ingredient", "static"]},
                    "ingredient_name": {"type": "string"},
                    "static_value": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["action_field", "source_type"]
            }
        },
        "rejection_reason": {
            "type": "string",
            "description": "Reason for rejection (if REJECT)"
        }
    },
    "required": ["reasoning", "decision", "service_name"]
}

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {
            "type": "string",
            "description": "Step-by-step analysis of the applet"
        },
        "trigger_matches_intent": {
            "type": "boolean",
            "description": "Does the trigger match user's intent?"
        },
        "action_matches_intent": {
            "type": "boolean",
            "description": "Does the action match user's intent?"
        },
        "bindings_valid": {
            "type": "boolean",
            "description": "Are all bindings semantically valid?"
        },
        "score": {
            "type": "number",
            "description": "Overall quality score (0.0 to 1.0)"
        },
        "critique": {
            "type": "string",
            "description": "Critique and suggestions for improvement"
        },
        "is_executable": {
            "type": "boolean",
            "description": "Is this applet ready for execution?"
        }
    },
    "required": ["reflection", "score", "is_executable"]
}


def schema_to_prompt_string(schema: Dict[str, Any]) -> str:
    """
    Convert JSON schema to string for Granite 4 system prompt.

    Args:
        schema: JSON schema dict

    Returns:
        Formatted schema string for prompt
    """
    import json
    return f"<schema>\n{json.dumps(schema, indent=2)}\n</schema>"
