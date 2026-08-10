# AGENTS.md

## Overview

`llm-interaction` is a Python library for calling LLMs via Azure, Databricks, OpenRouter, LiteLLM, DeepSeek, or local (llama-server) backends. Install with `pip install llm-interaction`. For Databricks: `pip install llm-interaction[databricks]`. For LiteLLM (Anthropic, Databricks Claude, Vertex AI, etc.): `pip install llm-interaction[litellm]`. Import as `from llm_interaction import ...`.

## Public API

```python
from llm_interaction import LLMInteraction, tool, ToolContext, ToolDef, LLMResponse, AgentResult
from llm_interaction import StreamAgentLoop, StreamEvent, StreamTextDelta, StreamToolCall, StreamToolResult
```

### LLMInteraction

```python
# Azure OpenAI (default)
llm = LLMInteraction(prompt_dir=Path("prompts"))

# Databricks (on-site or off-site with CLI auth)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="databricks")

# LiteLLM (universal provider support — Claude on Databricks, Anthropic, etc.)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="litellm")

# DeepSeek (Chat Completions API)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="deepseek")

# Local (llama-server or any OpenAI-compat server)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="local", endpoint="http://localhost:8080/v1")

# Pre-built client
llm = LLMInteraction(prompt_dir=Path("prompts"), client=my_client, model="m")
```

Constructor reads env vars: `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_ENDPOINT`, `LLM_INTERACTION_MODEL`, `LLM_INTERACTION_BACKEND`. All can be overridden via kwargs. The `LLM_INTERACTION_BACKEND` env var defaults to `"azure"` and lets you switch providers without code changes.

**Databricks env vars:** `LLM_INTERACTION_DATABRICKS_HOST`, `LLM_INTERACTION_MODEL`. Auth is handled via `WorkspaceClient` (PAT, CLI OAuth, or notebook auth).

**LiteLLM env vars:** `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_ENDPOINT`, `LLM_INTERACTION_MODEL`. The model must use litellm's `provider/model` format (e.g. `databricks/my-claude`, `anthropic/claude-sonnet-4-20250514`). The `LLM_INTERACTION_*` env vars are automatically forwarded to the provider-specific env vars that LiteLLM expects (e.g. `DATABRICKS_API_KEY`, `ANTHROPIC_API_KEY`).

**DeepSeek env vars:** `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_MODEL`. Uses the Chat Completions API (resends full message history each turn).

**Local env vars:** `LLM_INTERACTION_ENDPOINT` (defaults to `http://localhost:8080/v1`), `LLM_INTERACTION_MODEL`. No API key required. Uses the Responses API (`/v1/responses`) which llama-server supports natively.

**Methods:**
- `await llm.query(system: str, user: str, tools?, context?, max_tool_calls?) -> LLMResponse`
- `await llm.query_template(prompt_name: str, variables: dict, tools?, context?, max_tool_calls?) -> LLMResponse`
- `await llm.agent_loop(system: str, user: str, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> AgentResult`
- `await llm.agent_loop_template(prompt_name: str, variables: dict, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> AgentResult`
- `llm.sync_query(system: str, user: str, tools?, context?, max_tool_calls?) -> LLMResponse`
- `llm.sync_query_template(prompt_name: str, variables: dict, tools?, context?, max_tool_calls?) -> LLMResponse`
- `llm.sync_agent_loop(system: str, user: str, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> AgentResult`
- `llm.sync_agent_loop_template(prompt_name: str, variables: dict, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> AgentResult`
- `llm.agent_loop_streaming(system: str, user: str, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> StreamAgentLoop`
- `llm.agent_loop_streaming_template(prompt_name: str, variables: dict, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> StreamAgentLoop`
- `llm.render(template_name: str, variables: dict) -> str`

### LLMResponse

Returned by `query()` and `query_template()`. Lazy parsing:
- `.text` — raw string
- `.json()` — parse JSON (with `json_repair` fallback)
- `.yaml()` — parse YAML
- `.scratchpad_json()` — returns `(scratchpad: str, data: dict)`
- `.scratchpad_yaml()` — returns `(scratchpad: str, data: dict)`
- `await .parse(PydanticModel, max_retries=2)` — validate + auto-retry

### AgentResult

Returned by `agent_loop()`. Fields:
- `tool_call_count: int`
- `stop_reason: str` — `"content"`, `"stop_tool"`, or `"max_calls"`
- `final_content: str | None`
- `response_id: str`

### StreamAgentLoop

Returned by `agent_loop_streaming()` and `agent_loop_streaming_template()`.
Async-iterable that yields `StreamEvent` items in real time. After iteration,
`.result` provides the final `AgentResult`, or call `await .collect()` to drain
silently and return the result.

**StreamEvent types:**

| Event | Fields | Description |
|-------|--------|-------------|
| `StreamTextDelta` | `text: str` | Text chunk from the model |
| `StreamToolCall` | `call_id: str` `name: str` `arguments: dict` | Complete tool call (backend buffers argument deltas) |
| `StreamToolResult` | `call_id: str` `name: str` `result: Any` | Result of executing a tool |

**Usage:**

```python
stream = llm.agent_loop_streaming(
    system="You are a helpful assistant.",
    user="What's the capital of France?",
    tools=[search, finish],
)

async for event in stream:
    match event:
        case StreamTextDelta(text):
            print(text, end="", flush=True)
        case StreamToolCall(name=name, arguments=args):
            print(f"\n→ {name}({args})")
        case StreamToolResult(name=name, result=result):
            print(f"← {result}")

result = stream.result  # AgentResult

# Or drain silently — just get AgentResult
result = await llm.agent_loop_streaming(...).collect()
```

**Backend support:**
- Azure, Databricks, OpenRouter, DeepSeek, Local: supported
- LiteLLM: raises `NotImplementedError`

### @tool decorator

```python
@tool
def my_tool(query: str, limit: int = 10) -> list[dict]:
    """Description from first line of docstring.

    Args:
        query: Per-parameter description from Google-style docstring
        limit: Also extracted automatically
    """
    ...
```

- Generates JSON Schema from type hints and docstrings
- `@tool(stop=True)` marks a tool as an agent loop terminator
- `@tool(description="override")` overrides the docstring description

### ToolContext[T]

Type-based dependency injection for tools:

```python
@tool
def my_tool(ctx: ToolContext[MyService], query: str) -> str:
    return ctx.do_something(query)

# Pass context when calling:
await llm.query(..., tools=[my_tool], context=[MyService()])
```

- `ToolContext[T]` params are excluded from the LLM schema
- Matched by type from the `context` list
- Multiple `ToolContext` params of different types are supported

## Templates

Jinja2 templates in `prompt_dir`. Naming: `{name}_system.jinja` and `{name}_user.jinja`.

## Module Structure

```
src/llm_interaction/
├── __init__.py    # public re-exports
├── backend.py     # Backend ABC, ResponsesBackend, ChatCompletionsBackend, LiteLLMBackend
├── tool.py        # @tool, ToolDef, ToolContext
├── _schema.py     # type-to-JSON-Schema, docstring parsing
├── parsing.py     # JSON/YAML/scratchpad extraction
├── response.py    # LLMResponse, AgentResult, StreamAgentLoop, StreamEvent, etc.
└── client.py      # LLMInteraction, context matching, provider init, streaming agent loop
```
