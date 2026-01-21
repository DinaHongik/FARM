"""
Output Parser for LLM Responses
===============================

Safe JSON parsing with retry logic for LLM outputs.
Handles common LLM output issues like markdown code blocks,
trailing text, and malformed JSON.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

# Add parent directory to path for module imports
NODES_DIR = Path(__file__).parent
AGENTS_DIR = NODES_DIR.parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.errors import ParsingError


T = TypeVar("T", bound=BaseModel)


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from LLM response, handling common formats.

    Handles:
    - Raw JSON
    - JSON in markdown code blocks
    - JSON with leading/trailing text

    Args:
        response: Raw LLM response string

    Returns:
        Cleaned JSON string

    Raises:
        ParsingError: If no valid JSON found
    """
    text = response.strip()

    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_matches = re.findall(code_block_pattern, text)
    if code_matches:
        # Use the first code block that looks like JSON
        for match in code_matches:
            cleaned = match.strip()
            if cleaned.startswith("{") or cleaned.startswith("["):
                return cleaned

    # Try to find raw JSON object or array
    # Look for outermost braces/brackets
    brace_start = text.find("{")
    bracket_start = text.find("[")

    if brace_start == -1 and bracket_start == -1:
        raise ParsingError(
            "No JSON object or array found in response",
            raw_output=response
        )

    # Determine which comes first
    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        # Array
        start = bracket_start
        open_char, close_char = "[", "]"
    else:
        # Object
        start = brace_start
        open_char, close_char = "{", "}"

    # Find matching closing brace/bracket
    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if depth != 0:
        raise ParsingError(
            f"Unbalanced {open_char}{close_char} in JSON",
            raw_output=response
        )

    return text[start:end]


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response as JSON dictionary.

    Args:
        response: Raw LLM response

    Returns:
        Parsed dictionary

    Raises:
        ParsingError: If parsing fails
    """
    try:
        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParsingError(
            f"Invalid JSON: {e}",
            raw_output=response,
            details={"json_error": str(e)}
        )


def parse_with_schema(
    response: str,
    schema: Type[T]
) -> T:
    """
    Parse LLM response and validate against Pydantic schema.

    Args:
        response: Raw LLM response
        schema: Pydantic model class for validation

    Returns:
        Validated Pydantic model instance

    Raises:
        ParsingError: If parsing or validation fails
    """
    try:
        data = parse_json_response(response)
        return schema.model_validate(data)
    except ValidationError as e:
        raise ParsingError(
            f"Schema validation failed: {e}",
            raw_output=response,
            details={"validation_errors": e.errors()}
        )


def safe_parse(
    response: str,
    default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse JSON with fallback to default on failure.

    Args:
        response: Raw LLM response
        default: Default value if parsing fails

    Returns:
        Parsed dictionary or default
    """
    try:
        return parse_json_response(response)
    except ParsingError:
        return default if default is not None else {}


def extract_field(
    response: str,
    field: str,
    default: Any = None
) -> Any:
    """
    Extract a specific field from JSON response.

    Args:
        response: Raw LLM response
        field: Field name to extract
        default: Default value if field not found

    Returns:
        Field value or default
    """
    try:
        data = parse_json_response(response)
        return data.get(field, default)
    except ParsingError:
        return default
