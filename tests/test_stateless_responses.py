"""Tests for StatelessResponsesBackend: client-side history over the Responses API."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from llm_interaction.backend import (
    StatelessResponsesBackend,
    _StreamResponseDone,
    _StreamToolCallDone,
    _to_plain,
)


def _reasoning(text: str = "thinking"):
    return SimpleNamespace(
        type="reasoning",
        id="rs_1",
        content=[SimpleNamespace(type="reasoning_text", text=text)],
        summary=[],
        encrypted_content=None,
    )


def _function_call(call_id: str = "call_1", name: str = "search", args: str = '{"q": "x"}'):
    return SimpleNamespace(
        type="function_call", id="fc_1", call_id=call_id, name=name, arguments=args
    )


def _message(text: str):
    return SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _response(resp_id: str, output: list):
    return SimpleNamespace(id=resp_id, output=output)


class _RecordingCreate:
    """AsyncMock replacement that deep-copies kwargs (inputs are mutated lists)."""

    def __init__(self, returns: list):
        self._returns = list(returns)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self._returns.pop(0)


class TestToPlain:
    def test_namespace_recursive_and_drops_none(self):
        assert _to_plain(_reasoning()) == {
            "type": "reasoning",
            "id": "rs_1",
            "content": [{"type": "reasoning_text", "text": "thinking"}],
            "summary": [],
        }

    def test_pydantic_model_dump(self):
        item = MagicMock()
        item.model_dump.return_value = {"type": "message"}
        assert _to_plain(item) == {"type": "message"}
        item.model_dump.assert_called_once_with(exclude_none=True)


class TestStatelessCall:
    @pytest.mark.asyncio
    async def test_resends_history_and_never_sends_previous_response_id(self):
        client = MagicMock()
        create = _RecordingCreate([
            _response("r1", [_reasoning(), _function_call()]),
            _response("r2", [_message("done")]),
        ])
        client.responses.create = create
        backend = StatelessResponsesBackend(client=client, model="deepseek-flash")

        first_input = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        r1 = await backend.call(first_input, tools=[{"type": "function", "name": "search"}])
        tool_out = {"type": "function_call_output", "call_id": "call_1", "output": "res"}
        r2 = await backend.call([tool_out], previous_response_id=r1.id)

        assert r2.id == "r2"
        assert all("previous_response_id" not in c for c in create.calls)
        assert create.calls[0]["input"] == first_input
        assert create.calls[1]["input"] == [
            *first_input,
            {
                "type": "reasoning",
                "id": "rs_1",
                "content": [{"type": "reasoning_text", "text": "thinking"}],
                "summary": [],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "search",
                "arguments": '{"q": "x"}',
            },
            tool_out,
        ]

    @pytest.mark.asyncio
    async def test_unknown_previous_response_id_starts_fresh(self):
        client = MagicMock()
        create = _RecordingCreate([_response("r1", [_message("hi")])])
        client.responses.create = create
        backend = StatelessResponsesBackend(client=client, model="m")

        items = [{"role": "user", "content": "hi"}]
        await backend.call(items, previous_response_id="does-not-exist")

        assert create.calls[0]["input"] == items
        assert "previous_response_id" not in create.calls[0]

    @pytest.mark.asyncio
    async def test_branching_from_older_response(self):
        """Continuing from an earlier id uses that branch's history, not the latest."""
        client = MagicMock()
        create = _RecordingCreate([
            _response("r1", [_message("a")]),
            _response("r2", [_message("b")]),
            _response("r3", [_message("c")]),
        ])
        client.responses.create = create
        backend = StatelessResponsesBackend(client=client, model="m")

        await backend.call([{"role": "user", "content": "1"}])
        await backend.call([{"role": "user", "content": "2"}], previous_response_id="r1")
        await backend.call([{"role": "user", "content": "3"}], previous_response_id="r1")

        assert create.calls[2]["input"] == create.calls[1]["input"][:-1] + [
            {"role": "user", "content": "3"}
        ]

    @pytest.mark.asyncio
    async def test_retries_inherited(self):
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=[
            openai.APIConnectionError(request=MagicMock()),
            _response("r1", [_message("ok")]),
        ])
        backend = StatelessResponsesBackend(client=client, model="m", max_retries=2)

        with patch("llm_interaction.backend.asyncio.sleep", new=AsyncMock()):
            result = await backend.call([{"role": "user", "content": "hi"}])

        assert result.id == "r1"
        assert client.responses.create.call_count == 2
        assert "r1" in backend._history


async def _fake_stream(events: list):
    for e in events:
        yield e


class TestStatelessStream:
    @pytest.mark.asyncio
    async def test_stream_records_history_for_next_turn(self):
        client = MagicMock()
        calls: list[dict] = []
        streams = [
            _fake_stream([
                SimpleNamespace(
                    type="response.function_call_arguments.done",
                    call_id="call_1",
                    name="search",
                    item_id="fc_1",
                    arguments='{"q": "x"}',
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=_response("s1", [_reasoning(), _function_call()]),
                ),
            ]),
            _fake_stream([
                SimpleNamespace(type="response.output_text.delta", delta="done"),
                SimpleNamespace(
                    type="response.completed",
                    response=_response("s2", [_message("done")]),
                ),
            ]),
        ]

        async def create(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return streams.pop(0)

        client.responses.create = create
        backend = StatelessResponsesBackend(client=client, model="m")

        first_input = [{"role": "user", "content": "hi"}]
        events = [e async for e in backend.stream(first_input)]
        assert isinstance(events[0], _StreamToolCallDone)
        assert isinstance(events[-1], _StreamResponseDone)
        assert events[-1].response_id == "s1"

        tool_out = {"type": "function_call_output", "call_id": "call_1", "output": "res"}
        _ = [e async for e in backend.stream([tool_out], previous_response_id="s1")]

        assert all("previous_response_id" not in c for c in calls)
        assert calls[1]["input"][0] == first_input[0]
        assert [i.get("type") for i in calls[1]["input"][1:]] == [
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        assert "s2" in backend._history


class TestDeepseekAgentLoop:
    @pytest.mark.asyncio
    async def test_agent_loop_resends_full_history(self, tmp_path):
        from llm_interaction import LLMInteraction, tool

        @tool
        def search(q: str) -> str:
            """Search.

            Args:
                q: query
            """
            return f"result for {q}"

        env = {"LLM_INTERACTION_API_KEY": "k", "LLM_INTERACTION_MODEL": "deepseek-flash"}
        with patch.dict("os.environ", env, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path, backend="deepseek")

        create = _RecordingCreate([
            _response("r1", [_function_call(args='{"q": "x"}')]),
            _response("r2", [_message("final answer")]),
        ])
        llm._client = MagicMock()
        llm._client.responses.create = create

        result = await llm.agent_loop(system="sys", user="hi", tools=[search])

        assert result.stop_reason == "content"
        assert result.final_content == "final answer"
        assert result.tool_call_count == 1
        assert all("previous_response_id" not in c for c in create.calls)
        second = create.calls[1]["input"]
        assert second[0] == {"role": "system", "content": "sys"}
        assert second[1] == {"role": "user", "content": "hi"}
        assert second[2]["type"] == "function_call"
        assert second[3] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result for x",
        }
