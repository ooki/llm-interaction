"""Tests for streaming agent loop — agent_loop_streaming()."""

from unittest.mock import MagicMock

import pytest

from llm_interaction import (
    AgentResult,
    StreamAgentLoop,
    StreamTextDelta,
    StreamToolCall,
    StreamToolResult,
    ToolContext,
    tool,
)
from llm_interaction.backend import (
    _BackendStreamEvent,
    _StreamResponseDone,
    _StreamTextDelta,
    _StreamToolCallDone,
)


def _make_backend_stream(events: list[_BackendStreamEvent]):
    """Create an async generator that yields backend events."""
    async def _gen():
        for e in events:
            yield e
    return _gen()


class TestStreamAgentLoop:
    def test_result_raises_before_iteration(self):
        """result raises RuntimeError before iteration."""
        obj = StreamAgentLoop()
        with pytest.raises(RuntimeError, match="not yet exhausted"):
            _ = obj.result

    @pytest.mark.asyncio
    async def test_collect_drains_and_returns_result(self):
        """collect() drains the stream and returns AgentResult."""
        obj = StreamAgentLoop()
        obj._result = AgentResult(
            tool_call_count=0, stop_reason="content",
            final_content="hi", response_id="r1",
        )
        obj._drained = True
        result = await obj.collect()
        assert result.stop_reason == "content"
        assert result.final_content == "hi"


class TestStreamingContentStop:
    @pytest.mark.asyncio
    async def test_content_stop(self, llm):
        """Streaming agent loop stops when LLM returns text-only."""

        @tool
        def noop(x: int) -> str:
            """Do nothing."""
            return "ok"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="Hello ")
            yield _StreamTextDelta(text="world!")
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[noop],
        )
        async for event in stream:
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Hello "
        assert isinstance(events[1], StreamTextDelta)
        assert events[1].text == "world!"

        result = stream.result
        assert result.stop_reason == "content"
        assert result.final_content == "Hello world!"
        assert result.tool_call_count == 0

    @pytest.mark.asyncio
    async def test_content_stop_collect(self, llm):
        """collect() works for streaming content stop."""

        @tool
        def noop(x: int) -> str:
            """Do nothing."""
            return "ok"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="done")
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        result = await llm.agent_loop_streaming(
            system="S", user="U", tools=[noop],
        ).collect()

        assert result.stop_reason == "content"
        assert result.final_content == "done"
        assert result.tool_call_count == 0


class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call_stop(self, llm):
        """Single tool call with stop=True completes the stream."""

        @tool(stop=True)
        def finish(answer: str) -> str:
            """Submit final answer."""
            return "submitted"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="reasoning...")
            yield _StreamToolCallDone(
                call_id="call_1", name="finish",
                arguments='{"answer": "42"}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[finish],
        )
        async for event in stream:
            events.append(event)

        assert len(events) == 3
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "reasoning..."
        assert isinstance(events[1], StreamToolCall)
        assert events[1].name == "finish"
        assert events[1].arguments == {"answer": "42"}
        assert isinstance(events[2], StreamToolResult)
        assert events[2].name == "finish"
        assert events[2].result == "submitted"

        result = stream.result
        assert result.stop_reason == "stop_tool"
        assert result.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calls(self, llm):
        """Two turns: tool call then content."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for {query}: Paris is capital"

        @tool(stop=True)
        def finish(answer: str) -> str:
            """Submit final answer."""
            return answer

        call_count = [0]

        async def _fake_stream(input_items, tools=None, prev=None):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _StreamTextDelta(text="Let me search.")
                yield _StreamToolCallDone(
                    call_id="call_1", name="search",
                    arguments='{"query": "capital France"}',
                )
                yield _StreamResponseDone(response_id="resp_1")
            else:
                yield _StreamTextDelta(text="The capital is Paris")
                yield _StreamToolCallDone(
                    call_id="call_2", name="finish",
                    arguments='{"answer": "The capital is Paris"}',
                )
                yield _StreamResponseDone(response_id="resp_2")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[search, finish],
        )
        async for event in stream:
            events.append(event)

        assert len(events) == 6

        # Turn 1
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Let me search."
        assert isinstance(events[1], StreamToolCall)
        assert events[1].name == "search"
        assert isinstance(events[2], StreamToolResult)
        assert events[2].name == "search"
        assert "Paris" in str(events[2].result)

        # Turn 2
        assert isinstance(events[3], StreamTextDelta)
        assert events[3].text == "The capital is Paris"
        assert isinstance(events[4], StreamToolCall)
        assert events[4].name == "finish"
        assert isinstance(events[5], StreamToolResult)
        assert events[5].name == "finish"

        result = stream.result
        assert result.stop_reason == "stop_tool"
        assert result.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_callback(self, llm):
        """on_tool_call callback fires during streaming."""

        calls = []

        @tool(stop=True)
        def done(x: int) -> str:
            """Finish."""
            return "ok"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamToolCallDone(
                call_id="call_1", name="done",
                arguments='{"x": 42}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[done],
            on_tool_call=lambda i, name, args, res: calls.append(
                (i, name, args, res)
            ),
        )
        async for _ in stream:
            pass

        assert len(calls) == 1
        assert calls[0][1] == "done"
        assert calls[0][2] == {"x": 42}

    @pytest.mark.asyncio
    async def test_context_injection(self, llm):
        """ToolContext is loaded correctly."""

        class MyService:
            def query(self, q: str) -> str:
                return f"result:{q}"

        @tool(stop=True)
        def lookup(ctx: ToolContext[MyService], key: str) -> str:
            """Look up a key.

            Args:
                key: The key to look up
            """
            return ctx.query(key)

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamToolCallDone(
                call_id="call_1", name="lookup",
                arguments='{"key": "foo"}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[lookup],
            context=[MyService()],
        )
        async for event in stream:
            events.append(event)

        tool_result = events[1]
        assert isinstance(tool_result, StreamToolResult)
        assert tool_result.result == "result:foo"

    @pytest.mark.asyncio
    async def test_tool_call_with_json_repair(self, llm):
        """Malformed JSON arguments are repaired."""

        @tool(stop=True)
        def process(data: dict) -> str:
            """Process data."""
            return f"processed {data}"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamToolCallDone(
                call_id="call_1", name="process",
                arguments='{broken json here: true "key": "value"}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[process],
        )
        async for event in stream:
            events.append(event)

        tool_call = events[0]
        assert isinstance(tool_call, StreamToolCall)
        assert "key" in tool_call.arguments

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, llm):
        """Tool exceptions are caught and returned as error dict."""

        @tool(stop=True)
        def crash() -> str:
            """This raises."""
            raise ValueError("Boom!")

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamToolCallDone(
                call_id="call_1", name="crash",
                arguments='{}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        events = []
        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[crash],
        )
        async for event in stream:
            events.append(event)

        tool_result = events[1]
        assert isinstance(tool_result, StreamToolResult)
        assert tool_result.result == {"error": "Boom!"}


class TestStreamingTemplate:
    @pytest.mark.asyncio
    async def test_agent_loop_streaming_template(self, llm):
        """agent_loop_streaming_template() renders and streams."""

        @tool(stop=True)
        def finish(answer: str) -> str:
            """Submit final answer."""
            return answer

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="Answer: 42")
            yield _StreamToolCallDone(
                call_id="call_1", name="finish",
                arguments='{"answer": "42"}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming_template(
            prompt_name="test",
            variables={"topic": "AI", "question": "What?"},
            tools=[finish],
        )

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 3
        assert isinstance(events[0], StreamTextDelta)


class TestStreamingArgsFlow:
    @pytest.mark.asyncio
    async def test_text_only_no_tool_calls(self, llm):
        """Stream with no tools at all — just content."""
        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="Simple answer.")
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[],
        )
        async for event in stream:
            assert isinstance(event, StreamTextDelta)

        assert stream.result.final_content == "Simple answer."

    @pytest.mark.asyncio
    async def test_generator_result_set_after_iteration(self, llm):
        """The generator properly sets StreamAgentLoop._result and exits."""
        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamTextDelta(text="ok")
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[],
        )

        async for _ in stream:
            pass

        assert stream._drained is True
        assert stream._result is not None
        assert stream._result.final_content == "ok"


class TestStreamingMaxCalls:
    @pytest.mark.asyncio
    async def test_max_tool_calls_reached(self, llm):
        """Streaming stops with stop_reason='max_calls' when limit is exceeded."""

        @tool
        def work(x: int) -> str:
            """Do some work."""
            return f"done {x}"

        turn = [0]

        async def _fake_stream(_input, tools=None, prev=None):
            turn[0] += 1
            yield _StreamToolCallDone(
                call_id=f"call_{turn[0]}", name="work",
                arguments=f'{{"x": {turn[0]}}}',
            )
            yield _StreamResponseDone(response_id=f"resp_{turn[0]}")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[work], max_tool_calls=3,
        )
        async for _ in stream:
            pass

        result = stream.result
        assert result.stop_reason == "max_calls"
        assert result.tool_call_count == 3

    @pytest.mark.asyncio
    async def test_max_calls_stops_mid_turn(self, llm):
        """max_calls check happens before executing remaining tools in the same turn."""

        @tool
        def alpha(x: int) -> str:
            return f"alpha_{x}"

        @tool
        def beta(y: int) -> str:
            return f"beta_{y}"

        async def _fake_stream(_input, tools=None, prev=None):
            yield _StreamToolCallDone(
                call_id="call_1", name="alpha", arguments='{"x": 1}',
            )
            yield _StreamToolCallDone(
                call_id="call_2", name="beta", arguments='{"y": 2}',
            )
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="U", tools=[alpha, beta], max_tool_calls=1,
        )
        async for _ in stream:
            pass

        result = stream.result
        assert result.stop_reason == "max_calls"
        assert result.tool_call_count == 1


class TestStreamingPreviousResponseId:
    @pytest.mark.asyncio
    async def test_previous_response_id_omits_system_prompt(self, llm):
        """When continuing, only the user message is sent (no system prompt)."""

        @tool
        def noop(x: int) -> str:
            """No-op."""
            return "ok"

        sent_inputs = []

        async def _fake_stream(input_items, tools=None, prev=None):
            sent_inputs.append((prev, input_items))
            yield _StreamTextDelta(text="continued")
            yield _StreamResponseDone(response_id="resp_2")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="S", user="Continue...", tools=[noop],
            previous_response_id="resp_1",
        )
        async for _ in stream:
            pass

        assert len(sent_inputs) == 1
        prev_id, items = sent_inputs[0]
        assert prev_id == "resp_1"
        assert items == [{"role": "user", "content": "Continue..."}]

    @pytest.mark.asyncio
    async def test_fresh_start_includes_system_prompt(self, llm):
        """Without previous_response_id, the system prompt is included."""

        @tool
        def noop(x: int) -> str:
            """No-op."""
            return "ok"

        sent_inputs = []

        async def _fake_stream(input_items, tools=None, prev=None):
            sent_inputs.append(input_items)
            yield _StreamTextDelta(text="fresh")
            yield _StreamResponseDone(response_id="resp_1")

        llm._backend_impl.stream = MagicMock(side_effect=_fake_stream)

        stream = llm.agent_loop_streaming(
            system="My System", user="Hello", tools=[noop],
        )
        async for _ in stream:
            pass

        items = sent_inputs[0]
        assert items == [
            {"role": "system", "content": "My System"},
            {"role": "user", "content": "Hello"},
        ]
