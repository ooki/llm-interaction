"""
llm-interaction — Async Azure OpenAI client with typed tool calling and lazy output parsing.

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
