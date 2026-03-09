"""
llm-interaction — Async OpenAI Responses API client with typed tool calling,
lazy output parsing, and Azure/Databricks backends.

Usage::

    from llm_interaction import LLMInteraction, tool, ToolContext, LLMResponse, AgentResult
"""

from llm_interaction.client import LLMInteraction
from llm_interaction.response import AgentResult, LLMResponse
from llm_interaction.tool import ToolContext, ToolDef, tool

__all__ = [
    "LLMInteraction",
    "LLMResponse",
    "AgentResult",
    "ToolContext",
    "ToolDef",
    "tool",
]
