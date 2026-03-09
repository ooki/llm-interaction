"""Integration tests with real LLM calls via Databricks backend.

Skipped unless LLM_INTERACTION_DATABRICKS_HOST and LLM_INTERACTION_MODEL env vars are set
(and databricks-sdk is installed with valid CLI auth).

Run with:
    pytest tests/test_integration_databricks.py -v

Authenticate first if running off-site:
    databricks auth login --host https://your-workspace.azuredatabricks.net
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pytest
from pydantic import BaseModel

from llm_interaction import LLMInteraction, ToolContext, tool

_REQUIRED_VARS = [
    "LLM_INTERACTION_DATABRICKS_HOST",
    "LLM_INTERACTION_MODEL",
]

skip_no_databricks = pytest.mark.skipif(
    not all(os.getenv(v) for v in _REQUIRED_VARS),
    reason="Databricks env vars not set (LLM_INTERACTION_DATABRICKS_HOST, LLM_INTERACTION_MODEL)",
)


@pytest.fixture
def llm(tmp_path):
    (tmp_path / "greet_system.jinja").write_text("You are a helpful assistant.")
    (tmp_path / "greet_user.jinja").write_text("Say hello to {{ name }} in one sentence.")
    return LLMInteraction(prompt_dir=tmp_path, backend="databricks")


@skip_no_databricks
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

    @pytest.mark.asyncio
    async def test_scratchpad_json(self, llm):
        result = await llm.query(
            system=(
                "You are a helpful assistant. First explain your reasoning in plain text, "
                "then output your answer in a ```json block."
            ),
            user='What is the capital of Norway? Return {"capital": "..."}',
        )
        scratchpad, data = result.scratchpad_json()
        assert len(scratchpad) > 0
        assert data["capital"].lower() == "oslo"


@skip_no_databricks
class TestParseIntegration:
    @pytest.mark.asyncio
    async def test_pydantic_parse(self, llm):
        class CityInfo(BaseModel):
            city: str
            country: str
            population_millions: float

        result = await llm.query(
            system="You are a helpful assistant. Respond with valid JSON.",
            user='Return JSON about Oslo with keys: city, country, population_millions.',
        )
        info = await result.parse(CityInfo)
        assert info.city.lower() == "oslo"
        assert info.country.lower() == "norway"
        assert info.population_millions > 0


@skip_no_databricks
class TestTemplateIntegration:
    @pytest.mark.asyncio
    async def test_query_template(self, llm):
        result = await llm.query_template(
            prompt_name="greet",
            variables={"name": "World"},
        )
        assert len(result.text) > 0
        assert "hello" in result.text.lower() or "world" in result.text.lower()


@skip_no_databricks
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


@skip_no_databricks
class TestContextInjectionIntegration:
    @pytest.mark.asyncio
    async def test_context_injection(self, llm):
        class WeatherAPI:
            def get(self, city: str) -> dict:
                return {"city": city, "temp_c": 22, "condition": "sunny"}

        @tool
        def get_weather(ctx: ToolContext[WeatherAPI], city: str) -> dict:
            """Get current weather for a city.

            Args:
                city: City name
            """
            return ctx.get(city)

        @tool(stop=True)
        def answer(response: str) -> str:
            """Give the final answer to the user.

            Args:
                response: The response to give
            """
            return response

        result = await llm.agent_loop(
            system="You are a weather assistant. Look up the weather, then answer the user.",
            user="What's the weather in Oslo?",
            tools=[get_weather, answer],
            context=[WeatherAPI()],
        )
        assert result.stop_reason == "stop_tool"
        assert result.tool_call_count >= 1
