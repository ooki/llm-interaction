# AGENTS.md

## Overview

`llm-interaction` is a Python library for calling the OpenAI Responses API via Azure or Databricks backends. Install with `pip install llm-interaction`. For Databricks: `pip install llm-interaction[databricks]`. Import as `from llm_interaction import ...`.

## Public API

```python
from llm_interaction import LLMInteraction, tool, ToolContext, ToolDef, LLMResponse, AgentResult
```

### LLMInteraction

```python
# Azure OpenAI (default)
llm = LLMInteraction(prompt_dir=Path("prompts"))

# Databricks (on-site or off-site with CLI auth)
llm = LLMInteraction(prompt_dir=Path("prompts"), backend="databricks")

# Pre-built client
llm = LLMInteraction(prompt_dir=Path("prompts"), client=my_client, model="m")
```

Constructor reads env vars: `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_ENDPOINT`, `LLM_INTERACTION_MODEL`. All three can be overridden via kwargs.

**Databricks env vars:** `LLM_INTERACTION_DATABRICKS_HOST`, `LLM_INTERACTION_MODEL`. Auth is handled via `WorkspaceClient` (PAT, CLI OAuth, or notebook auth).

**Methods:**
- `await llm.query(system: str, user: str, tools?, context?, max_tool_calls?) -> LLMResponse`
- `await llm.query_template(prompt_name: str, variables: dict, tools?, context?, max_tool_calls?) -> LLMResponse`
- `await llm.agent_loop(system: str, user: str, tools: list[ToolDef], context?, max_tool_calls?, on_tool_call?, previous_response_id?) -> AgentResult`
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
├── tool.py        # @tool, ToolDef, ToolContext
├── _schema.py     # type-to-JSON-Schema, docstring parsing
├── parsing.py     # JSON/YAML/scratchpad extraction
├── response.py    # LLMResponse, AgentResult
└── client.py      # LLMInteraction, context matching, Azure/Databricks backends
```
