"""Integration tests with real LLM calls via the DeepSeek backend.

DeepSeek's Responses API is stateless (``previous_response_id`` is ignored), so
``StatelessResponsesBackend`` keeps history client-side and resends it every
turn -- including the model's own ``reasoning`` / ``function_call`` output
items. These tests check that DeepSeek accepts that resent history.

Skipped unless LLM_INTERACTION_BACKEND=deepseek and LLM_INTERACTION_API_KEY are set.

Run with:
    pytest tests/test_integration_deepseek.py -v

Example env setup:
    export LLM_INTERACTION_BACKEND=deepseek
    export LLM_INTERACTION_API_KEY=sk-...
    export LLM_INTERACTION_MODEL=deepseek-flash
"""

import os

from dotenv import load_dotenv

load_dotenv()

import pytest
from pydantic import BaseModel

from llm_interaction import (
    LLMInteraction,
    StreamTextDelta,
    StreamToolCall,
    StreamToolResult,
    tool,
)
from llm_interaction.backend import StatelessResponsesBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LLM_INTERACTION_BACKEND") != "deepseek"
        or not os.getenv("LLM_INTERACTION_API_KEY"),
        reason="LLM_INTERACTION_BACKEND=deepseek and LLM_INTERACTION_API_KEY not set",
    ),
]


@pytest.fixture
def llm(tmp_path):
    (tmp_path / "greet_system.jinja").write_text("You are a helpful assistant.")
    (tmp_path / "greet_user.jinja").write_text("Say hello to {{ name }} in one sentence.")
    return LLMInteraction(prompt_dir=tmp_path, model=os.getenv("LLM_INTERACTION_MODEL") or "deepseek-flash")


@tool
def add(x: int, y: int) -> int:
    """Add two numbers.

    Args:
        x: First number
        y: Second number
    """
    return x + y


@tool(stop=True)
def submit(answer: int) -> str:
    """Submit the final answer.

    Args:
        answer: The final number
    """
    return "ok"


def _input_types(llm: LLMInteraction) -> list[str]:
    request = llm.last_call["request"]
    return [item.get("type", item.get("role")) for item in request["input"]]


class TestQueryIntegration:
    def test_uses_stateless_responses_backend(self, llm):
        assert isinstance(llm._backend_impl, StatelessResponsesBackend)

    async def test_simple_text(self, llm):
        result = await llm.query(
            system="You are a helpful assistant. Be very brief.",
            user="What is 2 + 2? Reply with just the number.",
        )
        assert "4" in result.text
        assert "previous_response_id" not in llm.last_call["request"]

    async def test_json_output(self, llm):
        result = await llm.query(
            system="You are a helpful assistant. Always respond with valid JSON in a ```json block.",
            user='Return a JSON object with a "color" key set to "blue".',
        )
        assert result.json()["color"] == "blue"

    async def test_query_template(self, llm):
        result = await llm.query_template(prompt_name="greet", variables={"name": "World"})
        assert "hello" in result.text.lower() or "world" in result.text.lower()


class TestParseIntegration:
    async def test_pydantic_parse(self, llm):
        class CityInfo(BaseModel):
            city: str
            country: str
            population_millions: float

        result = await llm.query(
            system="You are a helpful assistant. Respond with valid JSON in a ```json block.",
            user="Return JSON about Oslo with keys: city, country, population_millions.",
        )
        info = await result.parse(CityInfo)
        assert info.city.lower() == "oslo"
        assert info.population_millions > 0


class TestToolCallingIntegration:
    async def test_tool_result_round_trip(self, llm):
        """Second request must resend the model's function_call plus our output."""
        result = await llm.query(
            system="You are a calculator. Always use the add tool; never compute in your head.",
            user="What is 1234 + 4321?",
            tools=[add],
        )
        assert "5555" in result.text.replace(",", "")
        types = _input_types(llm)
        assert "function_call" in types
        assert "function_call_output" in types

    async def test_agent_loop_with_error_recovery_and_stop_tool(self, llm):
        vault = {"alpha": 17, "beta": 25}

        @tool
        def read_vault(key: str) -> int:
            """Read a secret number from the vault.

            Args:
                key: Vault key
            """
            if key not in vault:
                raise ValueError(f"No key '{key}'. Valid keys: {sorted(vault)}")
            return vault[key]

        calls: list[tuple[str, dict, object]] = []
        result = await llm.agent_loop(
            system="Use tools for everything; never compute in your head. Finish by calling submit.",
            user=(
                "Read vault key 'gamma' (if it doesn't exist, use 'beta' instead), "
                "read 'alpha', add the two numbers, and submit the sum."
            ),
            tools=[read_vault, add, submit],
            on_tool_call=lambda i, name, args, out: calls.append((name, args, out)),
        )

        assert result.stop_reason == "stop_tool"
        assert calls[-1][0] == "submit"
        assert calls[-1][1]["answer"] == 42
        assert any(isinstance(out, dict) and "error" in out for _, _, out in calls)
        assert "previous_response_id" not in llm.last_call["request"]

    async def test_multi_turn_continuation(self, llm):
        """previous_response_id is resolved client-side into the full history."""
        first = await llm.agent_loop(
            system="You are terse.",
            user="Remember this codeword: PELICAN. Reply only with 'noted'.",
            tools=[add],
        )
        assert first.stop_reason == "content"

        second = await llm.agent_loop(
            system="You are terse.",
            user="What was the codeword? Reply with just the word.",
            tools=[add],
            previous_response_id=first.response_id,
        )
        assert "pelican" in (second.final_content or "").lower()
        assert _input_types(llm).count("user") >= 2


class TestStreamingIntegration:
    async def test_streaming_agent_loop_with_tool(self, llm):
        stream = llm.agent_loop_streaming(
            system=(
                "You are a calculator. Always use the add tool; never compute in your head. "
                "After the tool result, state the answer in one sentence."
            ),
            user="What is 1234 + 4321?",
            tools=[add],
        )
        text: list[str] = []
        tool_calls: list[StreamToolCall] = []
        tool_results: list[StreamToolResult] = []
        async for event in stream:
            if isinstance(event, StreamTextDelta):
                text.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(event)
            elif isinstance(event, StreamToolResult):
                tool_results.append(event)

        assert tool_calls and tool_calls[0].name == "add"
        assert tool_results and tool_results[0].result == 5555
        assert stream.result.stop_reason == "content"
        assert "5555" in "".join(text).replace(",", "")
