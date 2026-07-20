"""Tests for ChatCompletionsBackend: format conversion, multi-turn history, and retry."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction.backend import (
    ChatCompletionsBackend,
    _chat_response_to_output,
    _input_items_to_messages,
    _responses_tools_to_chat,
)


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


class TestResponsesToChatToolConversion:
    def test_basic_conversion(self):
        """Responses API tool schema is wrapped in Chat Completions format."""
        tools = [
            {
                "type": "function",
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
            }
        ]
        result = _responses_tools_to_chat(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "add"
        assert result[0]["function"]["description"] == "Add two numbers."
        assert result[0]["function"]["parameters"]["properties"]["x"] == {"type": "integer"}

    def test_multiple_tools(self):
        """Multiple tools are all converted."""
        tools = [
            {"type": "function", "name": "a", "description": "A", "parameters": {}},
            {"type": "function", "name": "b", "description": "B", "parameters": {}},
        ]
        result = _responses_tools_to_chat(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_missing_description(self):
        """Missing description defaults to empty string."""
        tools = [{"type": "function", "name": "x", "parameters": {}}]
        result = _responses_tools_to_chat(tools)
        assert result[0]["function"]["description"] == ""


# ---------------------------------------------------------------------------
# Input items conversion
# ---------------------------------------------------------------------------


class TestInputItemsConversion:
    def test_system_and_user(self):
        """System and user messages pass through."""
        items = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        msgs = _input_items_to_messages(items)
        assert msgs == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]

    def test_function_call_output(self):
        """function_call_output is converted to tool role."""
        items = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "42",
            }
        ]
        msgs = _input_items_to_messages(items)
        assert msgs == [
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]

    def test_mixed_items(self):
        """Mix of roles and function_call_output."""
        items = [
            {"role": "user", "content": "compute"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "done",
            },
        ]
        msgs = _input_items_to_messages(items)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "tool"


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


class TestChatResponseToOutput:
    def test_text_response(self):
        """Text-only response produces a message item."""
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Hello!", tool_calls=None)
        )
        output = _chat_response_to_output(choice)
        assert len(output) == 1
        assert output[0].type == "message"
        assert output[0].content[0].text == "Hello!"

    def test_tool_call_response(self):
        """Tool call response produces function_call items."""
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="add", arguments='{"x": 1, "y": 2}'),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tc])
        )
        output = _chat_response_to_output(choice)
        assert len(output) == 1
        assert output[0].type == "function_call"
        assert output[0].call_id == "call_1"
        assert output[0].name == "add"
        assert output[0].arguments == '{"x": 1, "y": 2}'

    def test_tool_call_with_content(self):
        """Tool call + content produces both items."""
        tc = SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(name="search", arguments='{"q": "test"}'),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Searching...", tool_calls=[tc])
        )
        output = _chat_response_to_output(choice)
        assert len(output) == 2
        assert output[0].type == "function_call"
        assert output[1].type == "message"


# ---------------------------------------------------------------------------
# ChatCompletionsBackend
# ---------------------------------------------------------------------------


class TestChatCompletionsBackend:
    def _make_text_response(self, text: str, response_id: str = "resp-1"):
        """Create a mock Chat Completions response with text content."""
        response = MagicMock()
        response.id = response_id
        response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None)
            )
        ]
        return response

    def _make_tool_response(self, tool_calls: list, response_id: str = "resp-1"):
        """Create a mock Chat Completions response with tool calls."""
        response = MagicMock()
        response.id = response_id
        tcs = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"], arguments=tc["arguments"]
                ),
            )
            for tc in tool_calls
        ]
        response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=tcs)
            )
        ]
        return response

    @pytest.mark.asyncio
    async def test_simple_text_call(self):
        """Basic text response returns BackendResult with correct output."""
        client = MagicMock()
        mock_resp = self._make_text_response("Hello!")
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)

        backend = ChatCompletionsBackend(client=client, model="deepseek-chat")
        result = await backend.call(
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )

        assert result.output[0].type == "message"
        assert result.output[0].content[0].text == "Hello!"
        assert result.id.startswith("chatcmpl-")

        # Verify the Chat Completions API was called correctly
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
        # messages list is mutated after the call (assistant msg appended),
        # so we check it contains the original messages plus the assistant
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hi"}
        assert messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_tool_call(self):
        """Tool call response is normalized to Responses API format."""
        client = MagicMock()
        mock_resp = self._make_tool_response([
            {"id": "call_1", "name": "add", "arguments": '{"x": 1, "y": 2}'}
        ])
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)

        tools = [
            {
                "type": "function",
                "name": "add",
                "description": "Add.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        backend = ChatCompletionsBackend(client=client, model="deepseek-chat")
        result = await backend.call(
            [{"role": "user", "content": "add 1+2"}],
            tools=tools,
        )

        assert result.output[0].type == "function_call"
        assert result.output[0].name == "add"
        assert result.output[0].call_id == "call_1"

        # Verify tools were converted to Chat Completions format
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"][0]["function"]["name"] == "add"

    @pytest.mark.asyncio
    async def test_multi_turn_history(self):
        """Message history accumulates across turns using previous_response_id."""
        client = MagicMock()
        # First call: tool call
        tool_resp = self._make_tool_response([
            {"id": "call_1", "name": "add", "arguments": '{"x": 1, "y": 2}'}
        ])
        # Second call: text
        text_resp = self._make_text_response("The sum is 3")
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[tool_resp, text_resp]
        )

        tools = [{"type": "function", "name": "add", "description": "Add.", "parameters": {}}]
        backend = ChatCompletionsBackend(client=client, model="deepseek-chat")

        # First call — initial messages
        result1 = await backend.call(
            [{"role": "user", "content": "add 1+2"}],
            tools=tools,
        )
        first_id = result1.id

        # Second call — tool result, using previous_response_id
        result2 = await backend.call(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "3",
                }
            ],
            tools=tools,
            previous_response_id=first_id,
        )

        # The second call should have the full history
        second_call_kwargs = client.chat.completions.create.call_args[1]
        messages = second_call_kwargs["messages"]

        # Should contain: user, assistant (tool_call), tool result
        assert messages[0] == {"role": "user", "content": "add 1+2"}
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "3"}

    @pytest.mark.asyncio
    async def test_no_previous_id_starts_fresh(self):
        """Without previous_response_id, history starts fresh."""
        client = MagicMock()
        mock_resp = self._make_text_response("Hi")
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)

        backend = ChatCompletionsBackend(client=client, model="m")

        # First call
        await backend.call([{"role": "user", "content": "first"}])
        # Second call without previous_response_id — should NOT include history
        await backend.call([{"role": "user", "content": "second"}])

        second_call_kwargs = client.chat.completions.create.call_args[1]
        messages = second_call_kwargs["messages"]
        # First message should be the new user message (not history from first call)
        assert messages[0] == {"role": "user", "content": "second"}
        # The assistant response is appended after the call
        assert messages[1]["role"] == "assistant"
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_retry_on_error(self):
        """Retries on API errors."""
        client = MagicMock()
        import openai
        mock_resp = self._make_text_response("ok")
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                mock_resp,
            ]
        )

        backend = ChatCompletionsBackend(client=client, model="m", max_retries=2)
        with patch("llm_interaction.backend.asyncio.sleep", new_callable=AsyncMock):
            result = await backend.call([{"role": "user", "content": "hi"}])

        assert result.output[0].content[0].text == "ok"
        assert client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """All retries exhausted raises RuntimeError."""
        client = MagicMock()
        import openai
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="persistent",
                request=MagicMock(),
                body=None,
            )
        )

        backend = ChatCompletionsBackend(client=client, model="m", max_retries=2)
        with patch("llm_interaction.backend.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Chat API call failed"):
                await backend.call([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_client_property(self):
        """Client property getter/setter works."""
        client1 = MagicMock()
        client2 = MagicMock()
        backend = ChatCompletionsBackend(client=client1, model="m")
        assert backend.client is client1
        backend.client = client2
        assert backend.client is client2
