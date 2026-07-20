"""
llm-interaction — Async multi-provider LLM client with typed tool calling,
lazy output parsing, and Azure/Databricks/OpenRouter/LiteLLM/DeepSeek/local backends.

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
