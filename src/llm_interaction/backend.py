"""Backend protocol and implementations for different LLM API protocols.

Three concrete backends:
- ResponsesBackend: OpenAI Responses API (azure, databricks, openrouter, local)
- ChatCompletionsBackend: OpenAI Chat Completions API (deepseek)
- LiteLLMBackend: LiteLLM bridge (universal provider support)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Callable

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized result — what every backend returns
# ---------------------------------------------------------------------------


class BackendResult:
    """Normalized result from a backend API call.

    Duck-typed to match the OpenAI Responses API response shape
    (``response.output`` and ``response.id``).
    """

    __slots__ = ("output", "id")

    def __init__(self, output: list[Any], response_id: str):
        self.output = output
        self.id = response_id


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Protocol for LLM API backends."""

    @abstractmethod
    async def call(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> BackendResult:
        """Make a single API call and return a normalized result."""
        ...


# ---------------------------------------------------------------------------
# Responses API backend (azure, databricks, openrouter, local)
# ---------------------------------------------------------------------------


class ResponsesBackend(Backend):
    """Backend for providers that support the OpenAI Responses API natively."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        token_refresher: Callable[[], None] | None = None,
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._token_refresher = token_refresher

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        self._client = value

    async def call(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> BackendResult:
        if self._token_refresher:
            self._token_refresher()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.responses.create(**kwargs)
                return BackendResult(output=response.output, response_id=response.id)
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.APIConnectionError,
            ) as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"API call failed after {self._max_retries} retries. Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Chat Completions backend (deepseek, any OpenAI-compat chat-only provider)
# ---------------------------------------------------------------------------


def _responses_tools_to_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tool schemas to Chat Completions format.

    Responses API: ``{"type": "function", "name": ..., "description": ..., "parameters": ...}``
    Chat Completions: ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``
    """
    chat_tools = []
    for t in tools:
        chat_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        })
    return chat_tools


def _input_items_to_messages(
    input_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages.

    Handles:
    - ``{"role": "system"|"user"|"assistant", "content": ...}`` → pass through
    - ``{"type": "function_call_output", "call_id": ..., "output": ...}``
      → ``{"role": "tool", "tool_call_id": ..., "content": ...}``
    """
    messages = []
    for item in input_items:
        if item.get("type") == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item["output"],
            })
        elif "role" in item:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def _chat_response_to_output(choice: Any) -> list[Any]:
    """Convert a Chat Completions choice to Responses API output format.

    Returns a list of SimpleNamespace items matching the shape that
    ``_extract_function_calls()`` and ``_extract_text_from_output()`` expect.
    """
    message = choice.message
    output: list[Any] = []

    # Tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            output.append(SimpleNamespace(
                type="function_call",
                call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            ))

    # Text content
    if message.content:
        output.append(SimpleNamespace(
            type="message",
            content=[SimpleNamespace(text=message.content)],
        ))

    return output


class ChatCompletionsBackend(Backend):
    """Backend for providers that only support the Chat Completions API.

    Manages full message history internally since there is no server-side
    state (no ``previous_response_id``). All messages are resent each turn.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        # Accumulated message history keyed by synthetic response ID
        self._history: dict[str, list[dict[str, Any]]] = {}

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        self._client = value

    async def call(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> BackendResult:
        # Build the full message list
        if previous_response_id and previous_response_id in self._history:
            messages = list(self._history[previous_response_id])
        else:
            messages = []

        messages.extend(_input_items_to_messages(input_items))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = _responses_tools_to_chat(tools)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.APIConnectionError,
            ) as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Chat API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
        else:
            raise RuntimeError(
                f"Chat API call failed after {self._max_retries} retries. "
                f"Last error: {last_exc}"
            )

        choice = response.choices[0]
        output = _chat_response_to_output(choice)

        # Store the assistant response in history for multi-turn
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if choice.message.content:
            assistant_msg["content"] = choice.message.content
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        messages.append(assistant_msg)
        self._history[response_id] = messages

        return BackendResult(output=output, response_id=response_id)


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------


class LiteLLMBackend(Backend):
    """Backend using LiteLLM's ``aresponses()`` bridge."""

    def __init__(
        self,
        litellm_module: Any,
        model: str,
        max_retries: int = 3,
        extra_kwargs: dict[str, str] | None = None,
    ):
        self._litellm = litellm_module
        self._model = model
        self._max_retries = max_retries
        self._extra_kwargs = extra_kwargs or {}

    async def call(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> BackendResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            **self._extra_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._litellm.aresponses(**kwargs)
                return BackendResult(output=response.output, response_id=response.id)
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "LiteLLM API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"LiteLLM API call failed after {self._max_retries} retries. "
            f"Last error: {last_exc}"
        )
