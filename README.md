# llm-interaction

Async Azure OpenAI client with typed tool calling and lazy output parsing.

## Install

```bash
pip install llm-interaction
```

## Quick Start

```python
from pathlib import Path
from llm_interaction import LLMInteraction, tool, ToolContext

llm = LLMInteraction(prompt_dir=Path("prompts"))

result = await llm.query(system="Be helpful", user="Hello")
print(result.text)
```

## Output Parsing

Every `query()` returns an `LLMResponse`. Parsing is lazy — call the method you need:

```python
# Raw text
result.text

# JSON (extracts last ```json block, falls back to json_repair)
data = result.json()

# YAML
data = result.yaml()

# Scratchpad + JSON: splits reasoning text from structured data
scratchpad, data = result.scratchpad_json()
# scratchpad = "Let me analyze this step by step..."
# data = {"topics": ["ai", "ml"]}

# Scratchpad + YAML
scratchpad, data = result.scratchpad_yaml()

# Pydantic model (validates + auto-retries on failure)
from pydantic import BaseModel

class Analysis(BaseModel):
    topics: list[str]
    confidence: float

analysis = await result.parse(Analysis)
# On validation error, re-queries the LLM with the error message
# using previous_response_id for efficient context chaining
```

## Tool Calling

```python
@tool
def search(query: str, max_results: int = 10) -> list[dict]:
    """Search for documents.

    Args:
        query: The search query string
        max_results: Maximum number of results to return
    """
    return db.search(query, limit=max_results)

@tool(stop=True)
def submit(answer: str) -> str:
    """Submit the final answer."""
    return "done"

result = await llm.agent_loop(
    system="You are a research agent.",
    user="Find info about quantum computing.",
    tools=[search, submit],
)
```

## Jinja Templates

Templates use the naming convention `{name}_system.jinja` and `{name}_user.jinja`:

```python
# prompts/research_system.jinja
You are a {{ role }} assistant.

# prompts/research_user.jinja
Find information about {{ topic }}.
```

```python
result = await llm.query_template(
    prompt_name="research",
    variables={"role": "research", "topic": "quantum computing"},
)
```

## Context Injection

```python
class WeatherAPI:
    def get(self, city: str) -> dict:
        return {"city": city, "temp_c": 22, "condition": "sunny"}

@tool
def get_weather(ctx: ToolContext[WeatherAPI], city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: City name to look up
    """
    return ctx.get(city)

weather_api = WeatherAPI()

result = await llm.agent_loop(
    system="You are a helpful assistant with weather access.",
    user="What's the weather in Oslo?",
    tools=[get_weather],
    context=[weather_api],  # matched by type to ToolContext[WeatherAPI]
)
```

A single tool can use multiple contexts, each matched by type:

```python
class WeatherAPI:
    def get(self, city: str) -> dict:
        return {"city": city, "temp_c": 22, "condition": "sunny"}

class UserPreferences:
    def __init__(self, unit: str = "celsius"):
        self.unit = unit

@tool
def get_weather(
    weather: ToolContext[WeatherAPI],
    prefs: ToolContext[UserPreferences],
    city: str,
) -> str:
    """Get weather for a city in the user's preferred unit.

    Args:
        city: City name to look up
    """
    data = weather.get(city)
    if prefs.unit == "fahrenheit":
        data["temp_f"] = data["temp_c"] * 9 / 5 + 32
    return data

result = await llm.agent_loop(
    system="You are a weather assistant.",
    user="What's the weather in Oslo?",
    tools=[get_weather],
    context=[WeatherAPI(), UserPreferences(unit="fahrenheit")],
)
```

## Environment Variables

```
LLM_INTERACTION_API_KEY=your-api-key
LLM_INTERACTION_ENDPOINT=https://your-resource.openai.azure.com
LLM_INTERACTION_MODEL=gpt-4o
```

## License

MIT
