"""Backend protocol and implementations for different LLM API protocols.

Four concrete backends:
- ResponsesBackend: OpenAI Responses API (azure, databricks, openrouter, local)
- StatelessResponsesBackend: Responses API with client-side history (deepseek)
- ChatCompletionsBackend: OpenAI Chat Completions API (chat-only providers)
- LiteLLMBackend: LiteLLM bridge (universal provider support)
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized results
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


@dataclass
class _StreamTextDelta:
    """Internal: a chunk of streaming text from any backend."""
    text: str


@dataclass
class _StreamToolCallDone:
    """Internal: a complete tool call (arguments fully buffered by backend)."""
    call_id: str
    name: str
    arguments: str  # JSON string


@dataclass
class _StreamResponseDone:
    """Internal: the current response turn has completed."""
    response_id: str


_BackendStreamEvent = _StreamTextDelta | _StreamToolCallDone | _StreamResponseDone


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

    @abstractmethod
    def stream(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[_BackendStreamEvent]:
        """Stream a single API response, yielding normalized events."""
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
        self._last_request: dict[str, Any] | None = None
        self._last_response: Any = None

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
                self._last_request = kwargs
                self._last_response = response
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

    async def stream(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[_BackendStreamEvent]:
        if self._token_refresher:
            self._token_refresher()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        fc_buffers: dict[str, list[str]] = {}

        response = await self._client.responses.create(**kwargs)
        self._last_request = kwargs

        async for event in response:
            event_type: str = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                yield _StreamTextDelta(
                    text=getattr(event, "delta", "") or ""
                )

            elif event_type == "response.function_call_arguments.delta":
                item_id: str = getattr(event, "item_id", "")
                delta: str = getattr(event, "delta", "") or ""
                fc_buffers.setdefault(item_id, []).append(delta)

            elif event_type == "response.function_call_arguments.done":
                item_id = getattr(event, "item_id", "")
                buffered = fc_buffers.pop(item_id, None)
                full_args = "".join(buffered) if buffered else getattr(event, "arguments", "") or ""
                yield _StreamToolCallDone(
                    call_id=getattr(event, "call_id", "") or "",
                    name=getattr(event, "name", "") or "",
                    arguments=full_args,
                )

            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", "") if response else ""
                self._last_response = event
                yield _StreamResponseDone(response_id=response_id)


# ---------------------------------------------------------------------------
# Stateless Responses API backend (deepseek)
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """Recursively convert SDK models / namespaces to plain JSON-able data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, SimpleNamespace):
        value = vars(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


class StatelessResponsesBackend(ResponsesBackend):
    """Responses API backend for providers without server-side state.

    Such providers ignore ``previous_response_id``, so the full conversation
    (prior inputs plus the model's output items, including reasoning and
    function calls) is kept client-side and resent on every turn.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        token_refresher: Callable[[], None] | None = None,
    ):
        super().__init__(client, model, max_retries, token_refresher)
        self._history: dict[str, list[dict[str, Any]]] = {}

    def _full_input(
        self,
        input_items: list[dict[str, Any]],
        previous_response_id: str | None,
    ) -> list[dict[str, Any]]:
        prior = self._history.get(previous_response_id, []) if previous_response_id else []
        return [*prior, *input_items]

    def _record(
        self, response_id: str, full_input: list[dict[str, Any]], output: list[Any]
    ) -> None:
        self._history[response_id] = [*full_input, *(_to_plain(o) for o in output)]

    async def call(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> BackendResult:
        full_input = self._full_input(input_items, previous_response_id)
        result = await super().call(full_input, tools)
        self._record(result.id, full_input, result.output)
        return result

    async def stream(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[_BackendStreamEvent]:
        full_input = self._full_input(input_items, previous_response_id)
        async for event in super().stream(full_input, tools):
            if isinstance(event, _StreamResponseDone):
                response = getattr(self._last_response, "response", None)
                self._record(
                    event.response_id, full_input, getattr(response, "output", None) or []
                )
            yield event


# ---------------------------------------------------------------------------
# Chat Completions backend (any OpenAI-compat chat-only provider)
# ---------------------------------------------------------------------------


def _responses_tools_to_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tool schemas to Chat Completions format.

    Responses API:
      ``{"type": "function", "name": ..., "description": ..., "parameters": ...}``
    Chat Completions:
      ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``
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
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._last_request: dict[str, Any] | None = None
        self._last_response: Any = None

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        self._client = value

    async def stream(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[_BackendStreamEvent]:
        if previous_response_id and previous_response_id in self._history:
            messages = list(self._history[previous_response_id])
        else:
            messages = []

        messages.extend(_input_items_to_messages(input_items))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _responses_tools_to_chat(tools)

        tc_buffers: dict[int, dict[str, Any]] = {}

        response = await self._client.chat.completions.create(**kwargs)
        last_chunk = None

        async for chunk in response:
            last_chunk = chunk
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
                yield _StreamTextDelta(text=delta.content)

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tc_delta in tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_buffers:
                        tc_buffers[idx] = {
                            "id": "",
                            "name": "",
                            "arguments_parts": [],
                        }
                    buf = tc_buffers[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        buf["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        buf["arguments_parts"].append(
                            tc_delta.function.arguments
                        )

        # Flush tool calls
        for idx in sorted(tc_buffers):
            buf = tc_buffers[idx]
            yield _StreamToolCallDone(
                call_id=buf["id"],
                name=buf["name"],
                arguments="".join(buf["arguments_parts"]),
            )

        self._last_request = kwargs
        self._last_response = last_chunk

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if tc_buffers:
            assistant_msg["tool_calls"] = [
                {
                    "id": buf["id"],
                    "type": "function",
                    "function": {
                        "name": buf["name"],
                        "arguments": "".join(buf["arguments_parts"]),
                    },
                }
                for buf in tc_buffers.values()
            ]
        messages.append(assistant_msg)
        self._history[response_id] = messages

        yield _StreamResponseDone(response_id=response_id)

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

        self._last_request = kwargs
        self._last_response = response
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
        self._last_request: dict[str, Any] | None = None
        self._last_response: Any = None

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
                self._last_request = kwargs
                self._last_response = response
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

    async def stream(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[_BackendStreamEvent]:
        raise NotImplementedError(
            "LiteLLM streaming is not yet supported. "
            "Use a different backend for streaming."
        )
        yield  # pragma: no cover — makes this an async generator for the type signature
