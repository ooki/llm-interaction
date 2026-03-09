"""Parsing helpers for LLM output: JSON, YAML, scratchpad, and response extraction."""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair
import yaml


def _extract_fenced_block(content: str, language: str) -> str:
    """Extract the last fenced code block of the given language.

    Falls back to the entire content if no fenced block is found.
    """
    pattern = rf"```(?:{language})?\s*([\s\S]*?)\s*```"
    blocks = re.findall(pattern, content)
    if blocks:
        return blocks[-1]
    return content.strip()


def _split_scratchpad(content: str, language: str) -> tuple[str, str]:
    """Split content into scratchpad text and the last fenced block."""
    pattern = rf"```(?:{language})?\s*([\s\S]*?)\s*```"
    matches = list(re.finditer(pattern, content))
    if not matches:
        return "", content.strip()

    last_match = matches[-1]
    block = last_match.group(1)

    # Everything before the opening ``` of the last match
    fence_start = content.rfind("```", 0, last_match.start() + 3)
    if fence_start < 0:
        fence_start = last_match.start()
    scratchpad = content[:fence_start].strip()

    return scratchpad, block


def _parse_json(json_str: str) -> dict | list:
    """Parse JSON with json_repair fallback."""
    if not json_str or not json_str.strip():
        raise ValueError("LLM returned empty content. Expected JSON.")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            repaired = json_repair.repair_json(json_str)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON: {e}. Content: {json_str[:200]}..."
            ) from e


def _parse_yaml(yaml_str: str) -> dict | list:
    """Parse YAML string."""
    if not yaml_str or not yaml_str.strip():
        raise ValueError("LLM returned empty content. Expected YAML.")

    try:
        result = yaml.safe_load(yaml_str)
        if result is None:
            raise ValueError("YAML parsed to None.")
        return result
    except yaml.YAMLError as e:
        raise ValueError(
            f"Failed to parse YAML: {e}. Content: {yaml_str[:200]}..."
        ) from e


def _extract_text_from_output(output: list[Any]) -> str:
    """Extract text content from Responses API output list."""
    text_parts: list[str] = []
    for item in output:
        if hasattr(item, "type"):
            if item.type == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        text_parts.append(content_item.text)
            elif item.type == "text" and hasattr(item, "text"):
                text_parts.append(item.text)
        elif isinstance(item, dict):
            if item.get("type") == "message" and "content" in item:
                for content_item in item["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        text_parts.append(content_item["text"])
            elif item.get("type") == "text" and "text" in item:
                text_parts.append(item["text"])
    return "".join(text_parts)


def _extract_function_calls(output: list[Any]) -> list[dict[str, Any]]:
    """Extract function calls from Responses API output list."""
    calls: list[dict[str, Any]] = []
    for item in output:
        if hasattr(item, "type") and item.type == "function_call":
            calls.append(
                {
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": getattr(item, "arguments", "{}"),
                }
            )
        elif isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(
                {
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                }
            )
    return calls
