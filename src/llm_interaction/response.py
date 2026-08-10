"""LLMResponse, AgentResult, and streaming event data types."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from llm_interaction.parsing import (
    _extract_fenced_block,
    _extract_text_from_output,
    _parse_json,
    _parse_yaml,
    _split_scratchpad,
)

if TYPE_CHECKING:
    from llm_interaction.client import LLMInteraction

logger = logging.getLogger(__name__)


class LLMResponse:
    """Response from an LLM query with lazy parsing methods.

    Attributes:
        text: Raw text output from the LLM.
        response_id: API response ID for stateful chaining.
    """

    def __init__(
        self,
        text: str,
        response_id: str,
        _client: LLMInteraction | None = None,
    ):
        self.text = text
        self.response_id = response_id
        self._client = _client

    def json(self) -> dict | list:
        """Parse the response as JSON, with json_repair fallback."""
        json_str = _extract_fenced_block(self.text, "json")
        return _parse_json(json_str)

    def yaml(self) -> dict | list:
        """Parse the response as YAML."""
        yaml_str = _extract_fenced_block(self.text, "yaml")
        return _parse_yaml(yaml_str)

    def scratchpad_json(self) -> tuple[str, dict | list]:
        """Split into scratchpad text + JSON data.

        Returns:
            (scratchpad, data) where scratchpad is free-form reasoning text
            before the last JSON code block.
        """
        scratchpad, block = _split_scratchpad(self.text, "json")
        return scratchpad, _parse_json(block)

    def scratchpad_yaml(self) -> tuple[str, dict | list]:
        """Split into scratchpad text + YAML data.

        Returns:
            (scratchpad, data) where scratchpad is free-form reasoning text
            before the last YAML code block.
        """
        scratchpad, block = _split_scratchpad(self.text, "yaml")
        return scratchpad, _parse_yaml(block)

    async def parse(self, model: type[BaseModel], max_retries: int = 2) -> BaseModel:
        """Parse and validate response as a Pydantic model.

        On validation failure, re-queries the LLM with the error message
        using ``previous_response_id`` for efficient context chaining.

        Args:
            model: Pydantic model class to validate against.
            max_retries: Number of retry attempts on validation failure.

        Returns:
            Validated Pydantic model instance.
        """
        text = self.text
        response_id = self.response_id

        for attempt in range(max_retries + 1):
            try:
                json_str = _extract_fenced_block(text, "json")
                data = _parse_json(json_str)
                return model.model_validate(data)
            except (ValidationError, ValueError) as e:
                if attempt >= max_retries:
                    raise
                if self._client is None:
                    raise RuntimeError(
                        "Cannot retry parse() — LLMResponse has no client reference."
                    ) from e

                error_msg = (
                    f"Your output did not match the required schema.\n"
                    f"Validation error: {e}\n"
                    f"Please output valid JSON matching this schema:\n"
                    f"{json.dumps(model.model_json_schema(), indent=2)}"
                )
                logger.warning(
                    "Parse validation failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                retry_response = await self._client._call_api(
                    input_items=[{"role": "user", "content": error_msg}],
                    previous_response_id=response_id,
                )
                text = _extract_text_from_output(retry_response.output)
                response_id = retry_response.id

        raise RuntimeError("Unreachable")  # pragma: no cover

    def __repr__(self) -> str:
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return f"LLMResponse(text={preview!r}, response_id={self.response_id!r})"


@dataclass
class AgentResult:
    """Result from an agent loop execution."""

    tool_call_count: int
    stop_reason: str  # "content" | "stop_tool" | "max_calls"
    final_content: str | None
    response_id: str


# ---------------------------------------------------------------------------
# Streaming types
# ---------------------------------------------------------------------------


@dataclass
class StreamTextDelta:
    """A chunk of streaming text from the model.

    During agent loop streaming, text deltas appear in natural
    order — reasoning (scratchpad) before tool calls is included.
    """

    text: str


@dataclass
class StreamToolCall:
    """A complete tool call requested by the model.

    Backend buffers argument deltas internally, so you receive the
    full, parseable call.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamToolResult:
    """Result of executing a tool call."""

    call_id: str
    name: str
    result: Any


StreamEvent = StreamTextDelta | StreamToolCall | StreamToolResult


class StreamAgentLoop:
    """Async iterable returned by ``agent_loop_streaming()``.

    Iterate to receive :class:`StreamEvent` items in real time.
    After iteration the ``.result`` property holds the final
    :class:`AgentResult`.  Call ``await .collect()`` to drain
    the stream silently and return the result directly.
    """

    def __init__(self) -> None:
        self._generator: AsyncIterator[StreamEvent] | None = None
        self._result: AgentResult | None = None
        self._drained: bool = False

    def _bind(self, generator: AsyncIterator[StreamEvent]) -> None:
        self._generator = generator

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        if self._generator is None:
            raise RuntimeError("StreamAgentLoop not bound to a generator.")
        return self._generator

    @property
    def result(self) -> AgentResult:
        """The final :class:`AgentResult` after the stream is exhausted."""
        if not self._drained or self._result is None:
            raise RuntimeError(
                "Stream not yet exhausted. Iterate over events or call .collect() first."
            )
        return self._result

    async def collect(self) -> AgentResult:
        """Drain all events silently and return the final :class:`AgentResult`."""
        if self._drained and self._result is not None:
            return self._result
        async for _ in self:
            pass
        return self.result
