"""Tests for backend streaming — ResponsesBackend, ChatCompletionsBackend, LiteLLMBackend."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_interaction.backend import (
    ChatCompletionsBackend,
    LiteLLMBackend,
    ResponsesBackend,
    _StreamResponseDone,
    _StreamTextDelta,
    _StreamToolCallDone,
)


async def _fake_responses_stream(events: list):
    """Return an async generator of mock events for Responses API."""
    for e in events:
        yield e


class TestResponsesBackendStream:
    """Unit tests for ResponsesBackend.stream()."""

    @pytest.mark.asyncio
    async def test_text_only_stream(self):
        """Yields text deltas then response done."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(type="response.output_text.delta", delta="Hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world!"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        assert len(events) == 3
        assert isinstance(events[0], _StreamTextDelta)
        assert events[0].text == "Hello "
        assert isinstance(events[1], _StreamTextDelta)
        assert events[1].text == "world!"
        assert isinstance(events[2], _StreamResponseDone)
        assert events[2].response_id == "resp_1"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Buffsers function call argument deltas into a single _StreamToolCallDone."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                delta='{"query":',
                item_id="fc_1",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                delta='"capital France"}',
                item_id="fc_1",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                call_id="call_1",
                name="search",
                item_id="fc_1",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        tc_events = [e for e in events if isinstance(e, _StreamToolCallDone)]
        assert len(tc_events) == 1
        assert tc_events[0].call_id == "call_1"
        assert tc_events[0].name == "search"
        assert tc_events[0].arguments == '{"query":"capital France"}'

    @pytest.mark.asyncio
    async def test_tool_call_done_without_deltas(self):
        """Falls back to event.arguments when no deltas were buffered."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.function_call_arguments.done",
                call_id="call_1",
                name="search",
                arguments='{"key": "val"}',
                item_id="fc_new",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        tc_events = [e for e in events if isinstance(e, _StreamToolCallDone)]
        assert len(tc_events) == 1
        assert tc_events[0].arguments == '{"key": "val"}'

    @pytest.mark.asyncio
    async def test_stream_with_tools(self):
        """Passes tools through to the API call."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        _ = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "name": "test"}],
        )]

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["tools"] == [{"type": "function", "name": "test"}]

    @pytest.mark.asyncio
    async def test_stream_with_previous_response_id(self):
        """Passes previous_response_id through to the API call."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_2"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        _ = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Continue"}],
            previous_response_id="resp_1",
        )]

        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["previous_response_id"] == "resp_1"

    @pytest.mark.asyncio
    async def test_stream_stores_last_response(self):
        """Sets _last_response on the completion event."""
        client = MagicMock()
        completed_event = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_zzz"),
        )
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            completed_event,
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        _ = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        assert backend._last_response is completed_event

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        """Handles an empty stream (only response completed)."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_empty"),
            ),
        ]))

        backend = ResponsesBackend(client=client, model="gpt-4o")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        assert len(events) == 1
        assert isinstance(events[0], _StreamResponseDone)
        assert events[0].response_id == "resp_empty"

    @pytest.mark.asyncio
    async def test_stream_token_refresher_called(self):
        """Calls token_refresher before the API call."""
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_fake_responses_stream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1"),
            ),
        ]))

        refresher = MagicMock()
        backend = ResponsesBackend(
            client=client, model="gpt-4o", token_refresher=refresher,
        )
        _ = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        refresher.assert_called_once()


class TestChatCompletionsBackendStream:
    """Unit tests for ChatCompletionsBackend.stream()."""

    def _make_chunk(self, *, content=None, tool_calls=None, finish_reason=None):
        """Build a mock ChatCompletionChunk."""
        delta = SimpleNamespace()
        delta.content = content
        if tool_calls:
            tc_list = []
            for tc_data in tool_calls:
                tc_func = SimpleNamespace(
                    name=tc_data.get("name"),
                    arguments=tc_data.get("arguments"),
                )
                tc_item = SimpleNamespace(
                    index=tc_data["index"],
                    id=tc_data.get("id"),
                    function=tc_func,
                )
                tc_list.append(tc_item)
            delta.tool_calls = tc_list

        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    @pytest.mark.asyncio
    async def test_text_only_stream(self):
        """Yields text deltas then response done."""
        chunks = [
            self._make_chunk(content="Hello "),
            self._make_chunk(content="world!"),
            self._make_chunk(finish_reason="stop"),
        ]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        assert len(events) == 3
        assert isinstance(events[0], _StreamTextDelta)
        assert events[0].text == "Hello "
        assert isinstance(events[1], _StreamTextDelta)
        assert events[1].text == "world!"
        assert isinstance(events[2], _StreamResponseDone)

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Buffsers tool call deltas across chunks and yields _StreamToolCallDone."""
        chunks = [
            self._make_chunk(tool_calls=[
                {"index": 0, "id": "call_x", "name": "search", "arguments": ""},
            ]),
            self._make_chunk(tool_calls=[
                {"index": 0, "arguments": '{"query": "test"}'},
            ]),
            self._make_chunk(finish_reason="tool_calls"),
        ]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        tc_events = [e for e in events if isinstance(e, _StreamToolCallDone)]
        assert len(tc_events) == 1
        assert tc_events[0].call_id == "call_x"
        assert tc_events[0].name == "search"
        assert tc_events[0].arguments == '{"query": "test"}'

    @pytest.mark.asyncio
    async def test_multi_tool_call_stream(self):
        """Handles multiple tool calls in a single response."""
        chunks = [
            self._make_chunk(tool_calls=[
                {"index": 0, "id": "call_a", "name": "search", "arguments": '{}'},
            ]),
            self._make_chunk(tool_calls=[
                {"index": 1, "id": "call_b", "name": "calculate", "arguments": '{"x": 1}'},
            ]),
            self._make_chunk(finish_reason="tool_calls"),
        ]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        tc_events = [e for e in events if isinstance(e, _StreamToolCallDone)]
        assert len(tc_events) == 2
        assert tc_events[0].name == "search"
        assert tc_events[1].name == "calculate"

    @pytest.mark.asyncio
    async def test_stream_with_tools(self):
        """Passes tools through in Chat Completions format."""
        chunks = [self._make_chunk(content="ok", finish_reason="stop")]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        _ = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "name": "test", "parameters": {}}],
        )]

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_stream_preserves_history(self):
        """Multi-turn streaming preserves message history."""
        chunks1 = [
            self._make_chunk(tool_calls=[
                {"index": 0, "id": "call_1", "name": "search", "arguments": '{"q": "x"}'},
            ]),
            self._make_chunk(finish_reason="tool_calls"),
        ]

        chunks2 = [
            self._make_chunk(content="found it", finish_reason="stop"),
        ]

        call_count = [0]

        async def _mock_create(**kwargs):
            call_count[0] += 1
            chunks = chunks1 if call_count[0] == 1 else chunks2
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=lambda **kw: _mock_create(**kw))

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")

        # Turn 1: tool call
        events1 = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Search x"}],
        )]
        tc_id = [e for e in events1 if isinstance(e, _StreamToolCallDone)][0].call_id
        resp_id_1 = [e for e in events1 if isinstance(e, _StreamResponseDone)][0].response_id

        # Turn 2: content with previous_response_id and function_call_output
        events2 = [e async for e in backend.stream(
            input_items=[
                {
                    "type": "function_call_output",
                    "call_id": tc_id,
                    "output": "result: found",
                }
            ],
            previous_response_id=resp_id_1,
        )]

        text_events = [e for e in events2 if isinstance(e, _StreamTextDelta)]
        assert len(text_events) == 1
        assert text_events[0].text == "found it"

        # Verify message count in history
        messages = backend._history[resp_id_1]
        # user message + assistant (tool_calls) = 2
        # Then: tool result + assistant (text) = 2 more
        # Total: 4 messages
        assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_stream_fresh_start(self):
        """New stream without previous_response_id starts fresh history."""
        chunks = [self._make_chunk(content="fresh", finish_reason="stop")]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        resp = [e for e in events if isinstance(e, _StreamResponseDone)][0]
        assert len(backend._history[resp.response_id]) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        """Handles a stream with no content (only finish_reason)."""
        chunks = [self._make_chunk(finish_reason="stop")]

        async def _mock_create(**kwargs):
            for c in chunks:
                yield c

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_create())

        backend = ChatCompletionsBackend(client=client, model="deepseek-flash")
        events = [e async for e in backend.stream(
            input_items=[{"role": "user", "content": "Hi"}],
        )]

        assert len(events) == 1
        assert isinstance(events[0], _StreamResponseDone)


class TestLiteLLMBackendStream:
    """Unit tests for LiteLLMBackend.stream()."""

    @pytest.mark.asyncio
    async def test_stream_raises_not_implemented(self):
        """LiteLLM streaming raises NotImplementedError."""
        backend = LiteLLMBackend(litellm_module=MagicMock(), model="test/model")

        with pytest.raises(NotImplementedError, match="LiteLLM streaming"):
            _ = [e async for e in backend.stream(
                input_items=[{"role": "user", "content": "Hi"}],
            )]
