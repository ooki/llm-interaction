# llm-interaction

Multi-provider LLM client with typed tool calling, lazy output parsing, and Azure/Databricks/OpenRouter/LiteLLM/DeepSeek/local backends. Supports both async and sync (notebook-friendly) usage.

## Install

```bash
pip install llm-interaction

# For Databricks backend:
pip install llm-interaction[databricks]

# For LiteLLM backend (Anthropic, Databricks Claude, Vertex AI, etc.):
pip install llm-interaction[litellm]
```

## Quick Start

### Loading Environment Variables

The library reads configuration from environment variables. Use `python-dotenv` to load them from a `.env` file:

```python
from dotenv import load_dotenv

# Call this once at the start of your script/notebook
load_dotenv()
```

Example `.env` file:

```
LLM_INTERACTION_API_KEY=your-api-key
LLM_INTERACTION_ENDPOINT=https://your-resource.openai.azure.com
LLM_INTERACTION_MODEL=gpt-4o
LLM_INTERACTION_BACKEND=azure   # optional, defaults to "azure"
```

If your environment variables are already set (e.g., in production), you can skip `load_dotenv()`.

### Sync (Notebook-Friendly)

```python
from pathlib import Path
from llm_interaction import LLMInteraction

# Azure OpenAI (default)
llm = LLMInteraction(prompt_dir=Path("prompts"))

# Databricks
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="databricks")

# OpenRouter
llm = LLMInteraction(
    prompt_dir=Path("prompts"),
    backend="openrouter",
    api_key="your-openrouter-key",
    model="openai/gpt-4",
)

# LiteLLM (universal provider support)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="litellm")

# DeepSeek
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="deepseek")

# Local (llama-server)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="local")

# Use sync methods in notebooks (no await needed)
result = llm.sync_query(system="Be helpful", user="Hello")
print(result.text)

# Or with templates
result = llm.sync_query_template(
    prompt_name="greeting",
    variables={"name": "Alice"},
)
print(result.text)

# Or run an agentic loop
from llm_interaction import tool

@tool(stop=True)
def submit_answer(answer: str) -> str:
    """Submit the final answer."""
    return "done"

result = llm.sync_agent_loop(
    system="You are a helpful assistant.",
    user="What is 2+2?",
    tools=[submit_answer],
)
print(f"Tool calls: {result.tool_call_count}, Reason: {result.stop_reason}")
```

### Async

```python
from pathlib import Path
from llm_interaction import LLMInteraction

llm = LLMInteraction(prompt_dir=Path("prompts"))

# Use async methods in async contexts
result = await llm.query(system="Be helpful", user="Hello")
print(result.text)

# Or with templates
result = await llm.query_template(
    prompt_name="greeting",
    variables={"name": "Alice"},
)
print(result.text)
```

## Output Parsing

Both `query()` and `sync_query()` return an `LLMResponse`. Parsing is lazy — call the method you need:

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

Set `LLM_INTERACTION_BACKEND` to switch providers without changing code (defaults to `"azure"`).

**Azure OpenAI** (default backend):

```
LLM_INTERACTION_BACKEND=azure
LLM_INTERACTION_API_KEY=your-api-key
LLM_INTERACTION_ENDPOINT=https://your-resource.openai.azure.com
LLM_INTERACTION_MODEL=gpt-4o
```

**Databricks** (`backend="databricks"`):

```
LLM_INTERACTION_BACKEND=databricks
LLM_INTERACTION_DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
LLM_INTERACTION_MODEL=your-serving-endpoint
```

Databricks auth is handled automatically by `WorkspaceClient`:
- **On-site** (notebook): no setup needed
- **Off-site** (local dev): run `databricks auth login --host <your-host>` first

**OpenRouter** (`backend="openrouter"`):

```
LLM_INTERACTION_BACKEND=openrouter
LLM_INTERACTION_API_KEY=your-openrouter-key
LLM_INTERACTION_MODEL=openai/gpt-4
```

**LiteLLM** (`backend="litellm"`):

```
LLM_INTERACTION_BACKEND=litellm
LLM_INTERACTION_API_KEY=your-provider-key
LLM_INTERACTION_ENDPOINT=https://provider-endpoint   # optional, depends on provider
LLM_INTERACTION_MODEL=databricks/my-claude           # must use litellm's provider/model format
```

Env vars are automatically forwarded to provider-specific vars (e.g. `DATABRICKS_API_KEY`, `ANTHROPIC_API_KEY`).

**DeepSeek** (`backend="deepseek"`):

```
LLM_INTERACTION_BACKEND=deepseek
LLM_INTERACTION_API_KEY=your-deepseek-key
LLM_INTERACTION_MODEL=deepseek-chat
```

Uses the Chat Completions API (full message history is resent each turn).

**Local** (`backend="local"`):

```
LLM_INTERACTION_BACKEND=local
LLM_INTERACTION_ENDPOINT=http://localhost:8080/v1   # optional, this is the default
LLM_INTERACTION_MODEL=my-model
```

No API key required. Uses the Responses API (`/v1/responses`) which llama-server supports natively.

## License

MIT
