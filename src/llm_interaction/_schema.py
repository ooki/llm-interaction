"""Internal helpers for JSON Schema generation and docstring parsing."""

from __future__ import annotations

import re
from typing import Any, get_args, get_origin


# Type mapping: Python type → JSON Schema type string
_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        if args and len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": _python_type_to_json_schema(args[1]),
            }
        return {"type": "object"}

    if origin is tuple:
        if args:
            return {
                "type": "array",
                "items": [_python_type_to_json_schema(a) for a in args],
            }
        return {"type": "array"}

    # Fallback
    return {"type": "string"}


def _parse_google_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style Args section."""
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc_lines: list[str] = []

    for line in docstring.splitlines():
        stripped = line.strip()

        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue

        # Detect end of Args section (next section header or blank after content)
        if in_args and stripped and not stripped.startswith(" ") and ":" in stripped:
            # Check if this is a new section header (e.g. "Returns:", "Raises:")
            potential_header = stripped.split(":")[0].strip()
            if potential_header in (
                "Returns",
                "Raises",
                "Yields",
                "Note",
                "Notes",
                "Example",
                "Examples",
            ):
                # Save current param
                if current_param:
                    descriptions[current_param] = " ".join(current_desc_lines).strip()
                in_args = False
                continue

        if not in_args:
            continue

        # Empty line ends the section
        if not stripped:
            if current_param:
                descriptions[current_param] = " ".join(current_desc_lines).strip()
                current_param = None
                current_desc_lines = []
            continue

        # Try to match "param_name: description" or "param_name (type): description"
        param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
        if param_match:
            # Save previous param
            if current_param:
                descriptions[current_param] = " ".join(current_desc_lines).strip()

            current_param = param_match.group(1)
            desc = param_match.group(2).strip()
            current_desc_lines = [desc] if desc else []
        elif current_param:
            # Continuation line
            current_desc_lines.append(stripped)

    # Save last param
    if current_param:
        descriptions[current_param] = " ".join(current_desc_lines).strip()

    return descriptions
