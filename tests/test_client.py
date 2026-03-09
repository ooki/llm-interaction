"""Tests for LLMInteraction: init, render, query, agent_loop."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction import AgentResult, LLMInteraction, LLMResponse, ToolContext, tool


class TestLLMInteractionInit:
    def test_init(self, llm):
        assert llm.model == "gpt-4o"

    def test_init_missing_model_raises(self, tmp_path):
        with patch("llm_interaction.client.load_dotenv"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="Model required"):
                    LLMInteraction(prompt_dir=tmp_path)

    def test_init_missing_key_raises(self, tmp_path):
        with patch("llm_interaction.client.load_dotenv"):
            with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    LLMInteraction(prompt_dir=tmp_path)

    def test_render(self, llm):
        result = llm.render("test_system.jinja", {"topic": "AI"})
        assert result == "System: AI"

    def test_render_prompt(self, llm):
        result = llm._render_prompt("test", {"topic": "AI", "question": "What?"})
        assert result["system"] == "System: AI"
        assert result["user"] == "User: What?"


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_simple(self, llm):
        """Test query with a simple text response (no tool calls)."""
        mock_response = MagicMock()
        mock_response.output = [
            MagicMock(type="message", content=[MagicMock(text="Hello!")])
        ]
        mock_response.id = "resp_123"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(return_value=mock_response)

        result = await llm.query(system="Be helpful", user="Hi")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"
        assert result.response_id == "resp_123"

    @pytest.mark.asyncio
    async def test_query_with_tool_call(self, llm):
        """Test query where LLM calls a tool then responds."""

        @tool
        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # First response: tool call
        tool_call_response = MagicMock()
        tool_call_response.output = [
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="add",
                arguments='{"x": 2, "y": 3}',
            )
        ]
        tool_call_response.id = "resp_1"

        # Second response: content
        content_response = MagicMock()
        content_response.output = [
            MagicMock(type="message", content=[MagicMock(text="The sum is 5")])
        ]
        content_response.id = "resp_2"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(
            side_effect=[tool_call_response, content_response]
        )

        result = await llm.query(
            system="You are a calculator",
            user="Add 2 and 3",
            tools=[add],
        )

        assert result.text == "The sum is 5"
        assert llm._client.responses.create.call_count == 2


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_agent_loop_content_stop(self, llm):
        """Agent loop stops when LLM returns content."""

        @tool
        def noop(x: int) -> str:
            """Do nothing."""
            return "ok"

        mock_response = MagicMock()
        mock_response.output = [
            MagicMock(type="message", content=[MagicMock(text="All done")])
        ]
        mock_response.id = "resp_1"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(return_value=mock_response)

        result = await llm.agent_loop(
            system="System",
            user="User",
            tools=[noop],
        )

        assert isinstance(result, AgentResult)
        assert result.stop_reason == "content"
        assert result.final_content == "All done"
        assert result.tool_call_count == 0

    @pytest.mark.asyncio
    async def test_agent_loop_stop_tool(self, llm):
        """Agent loop stops when a stop tool is called."""

        @tool(stop=True)
        def finish(answer: str) -> str:
            """Submit the final answer."""
            return "submitted"

        tool_call_response = MagicMock()
        tool_call_response.output = [
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="finish",
                arguments='{"answer": "42"}',
            )
        ]
        tool_call_response.id = "resp_1"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(return_value=tool_call_response)

        result = await llm.agent_loop(
            system="System",
            user="User",
            tools=[finish],
        )

        assert result.stop_reason == "stop_tool"
        assert result.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_agent_loop_with_context(self, llm):
        """Agent loop injects ToolContext correctly."""

        class MyDB:
            def get(self, key: str) -> str:
                return f"value_{key}"

        @tool
        def lookup(ctx: ToolContext[MyDB], key: str) -> str:
            """Look up a value.

            Args:
                key: The key to look up
            """
            return ctx.get(key)

        # Tool call followed by content
        tool_call_response = MagicMock()
        tool_call_response.output = [
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="lookup",
                arguments='{"key": "foo"}',
            )
        ]
        tool_call_response.id = "resp_1"

        content_response = MagicMock()
        content_response.output = [
            MagicMock(type="message", content=[MagicMock(text="Got value_foo")])
        ]
        content_response.id = "resp_2"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(
            side_effect=[tool_call_response, content_response]
        )

        db = MyDB()
        result = await llm.agent_loop(
            system="System",
            user="Look up foo",
            tools=[lookup],
            context=[db],
        )

        assert result.stop_reason == "content"
        assert result.final_content == "Got value_foo"
        assert result.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_agent_loop_on_tool_call_callback(self, llm):
        """on_tool_call callback is invoked."""
        calls_log = []

        @tool(stop=True)
        def done(x: int) -> str:
            """Finish."""
            return "ok"

        tool_call_response = MagicMock()
        tool_call_response.output = [
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="done",
                arguments='{"x": 1}',
            )
        ]
        tool_call_response.id = "resp_1"

        llm._client = MagicMock()
        llm._client.responses = MagicMock()
        llm._client.responses.create = AsyncMock(return_value=tool_call_response)

        await llm.agent_loop(
            system="System",
            user="User",
            tools=[done],
            on_tool_call=lambda i, name, args, res: calls_log.append(
                (i, name, args, res)
            ),
        )

        assert len(calls_log) == 1
        assert calls_log[0][1] == "done"
        assert calls_log[0][2] == {"x": 1}
