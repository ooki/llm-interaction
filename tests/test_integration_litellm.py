"""Integration tests with real LLM calls via LiteLLM backend.

Skipped unless LLM_INTERACTION_MODEL is set with a litellm-prefixed model
(and the provider's auth env vars are configured).

Run with:
    pytest tests/test_integration_litellm.py -v

Example env setup for Databricks Claude:
    export LLM_INTERACTION_MODEL=databricks/databricks-claude-opus-4-8
    export LLM_INTERACTION_API_KEY=dapi-...
    export LLM_INTERACTION_ENDPOINT=https://adb-xxx.azuredatabricks.net
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pytest
from pydantic import BaseModel

from llm_interaction import LLMInteraction, ToolContext, tool

_MODEL = os.getenv("LLM_INTERACTION_MODEL", "")

skip_no_litellm = pytest.mark.skipif(
    "/" not in _MODEL or not _MODEL.startswith(("databricks/", "anthropic/", "azure/", "openrouter/", "vertex_ai/", "bedrock/")),
    reason="LLM_INTERACTION_MODEL not set to a litellm-prefixed model (e.g. databricks/..., anthropic/...)",
)


@pytest.fixture
def llm(tmp_path):
    (tmp_path / "greet_system.jinja").write_text("You are a helpful assistant.")
    (tmp_path / "greet_user.jinja").write_text("Say hello to {{ name }} in one sentence.")
    return LLMInteraction(prompt_dir=tmp_path, backend="litellm")


@skip_no_litellm
class TestQueryIntegration:
    @pytest.mark.asyncio
    async def test_simple_text(self, llm):
        result = await llm.query(
            system="You are a helpful assistant. Be very brief.",
            user="What is 2 + 2? Reply with just the number.",
        )
        assert "4" in result.text

    @pytest.mark.asyncio
    async def test_json_output(self, llm):
        result = await llm.query(
            system="You are a helpful assistant. Always respond with valid JSON.",
            user='Return a JSON object with a "color" key set to "blue".',
        )
        data = result.json()
        assert data["color"] == "blue"

    @pytest.mark.asyncio
    async def test_yaml_output(self, llm):
        result = await llm.query(
            system="You are a helpful assistant. Always respond with valid YAML in a ```yaml block.",
            user='Return YAML with a key "animal" set to "cat".',
        )
        data = result.yaml()
        assert data["animal"] == "cat"


@skip_no_litellm
class TestToolCallingIntegration:
    @pytest.mark.asyncio
    async def test_tool_call(self, llm):
        @tool
        def add(x: int, y: int) -> int:
            """Add two numbers.

            Args:
                x: First number
                y: Second number
            """
            return x + y

        result = await llm.query(
            system="You are a calculator. Use the add tool to compute the answer.",
            user="What is 7 + 13?",
            tools=[add],
        )
        assert "20" in result.text

    @pytest.mark.asyncio
    async def test_agent_loop_with_stop_tool(self, llm):
        @tool
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers.

            Args:
                x: First number
                y: Second number
            """
            return x * y

        @tool(stop=True)
        def submit(answer: str) -> str:
            """Submit the final answer.

            Args:
                answer: The answer to submit
            """
            return answer

        result = await llm.agent_loop(
            system="You are a calculator. Use multiply to compute, then submit the answer.",
            user="What is 6 * 7?",
            tools=[multiply, submit],
        )
        assert result.stop_reason == "stop_tool"
        assert result.tool_call_count >= 1


@skip_no_litellm
class TestTemplateIntegration:
    @pytest.mark.asyncio
    async def test_query_template(self, llm):
        result = await llm.query_template(
            prompt_name="greet",
            variables={"name": "World"},
        )
        assert len(result.text) > 0
        assert "hello" in result.text.lower() or "world" in result.text.lower()
