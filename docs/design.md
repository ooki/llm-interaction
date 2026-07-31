# Design

`llm-interaction` is an async multi-provider LLM client with typed tool calling,
lazy output parsing, and six backends: Azure, Databricks, OpenRouter, LiteLLM,
DeepSeek, and local (llama-server).

## Architecture

| Module | Responsibility |
|--------|---------------|
| `tool.py` | `@tool` decorator, `ToolDef`, `ToolContext` |
| `_schema.py` | Python type → JSON Schema conversion, Google-style docstring parsing |
| `parsing.py` | JSON/YAML parsing, fenced block extraction, scratchpad splitting, response output extraction |
| `response.py` | `LLMResponse` (lazy parsing), `AgentResult` |
| `client.py` | `LLMInteraction` class, context matching, provider initialization |
| `backend.py` | `Backend` ABC, `BackendResult`, `ResponsesBackend`, `ChatCompletionsBackend`, `LiteLLMBackend` |
| `__init__.py` | Public API re-exports |

### Backend Abstraction

All API calls flow through a `Backend` ABC with a single async method:

```python
class Backend(ABC):
    async def call(input_items, tools, previous_response_id) -> BackendResult:
        ...
```

`BackendResult` normalizes provider-specific responses into a common shape
(`output` list and response `id`), duck-typed to match the OpenAI Responses API.

Three concrete implementations:

| Backend | API Protocol | Providers |
|---------|-------------|-----------|
| `ResponsesBackend` | `client.responses.create()` | Azure, Databricks, OpenRouter, local |
| `ChatCompletionsBackend` | `client.chat.completions.create()` | DeepSeek |
| `LiteLLMBackend` | `litellm.aresponses()` | Universal (Anthropic, Databricks Claude, Vertex AI, etc.) |

`ChatCompletionsBackend` manages full message history internally (no server-side
state). It converts input items to chat messages, tools to chat format, and
chat choices back to Responses-format output.

All three backends share the same retry logic (exponential backoff with jitter).

### Request Flow

```
LLMInteraction.query() / .agent_loop()
  → _call_api(input_items, tools, previous_response_id)
    → backend.call(input_items, tools, previous_response_id)
      → BackendResult(output, response_id)
  → LLMResponse / AgentResult
```

### Client Initialization

The constructor selects a backend via the `backend` kwarg or
`LLM_INTERACTION_BACKEND` env var (defaults to `"azure"`). Each `_init_*`
method builds an `AsyncOpenAI` client and a backend instance:

- `_init_azure()` — API key + endpoint
- `_init_databricks()` — `WorkspaceClient` (PAT, CLI OAuth, or notebook auth)
- `_init_openrouter()` — API key, OpenRouter base URL
- `_init_litellm()` — API key + optional endpoint, env-var forwarding
- `_init_deepseek()` — API key, DeepSeek base URL
- `_init_local()` — no auth, uses endpoint (defaults to `http://localhost:8080/v1`)
- Pre-built client — wraps any existing `AsyncOpenAI` in a `ResponsesBackend`

## Tool Definition

Tools are defined with a `@tool` decorator that generates JSON Schema from:
- **Function name** → tool name
- **Docstring first line** → tool description
- **Type hints** → parameter types in JSON Schema
- **Google-style `Args:` section** → per-parameter descriptions
- **Default values** → required vs optional

A `description` kwarg on the decorator overrides the docstring. A `stop=True`
argument marks a tool as a loop terminator for `agent_loop()`.

## Dependency Injection

`ToolContext[T]` is a generic marker type for injecting runtime context into
tools. Parameters typed as `ToolContext[T]` are:
- Excluded from the JSON schema sent to the LLM
- Automatically matched by type from the `context` argument passed to
  `query()` / `query_template()` / `agent_loop()`

A single tool can have multiple `ToolContext` parameters of different types.
Matching is strict: each type may appear at most once in the context list.

## Output Parsing

`LLMResponse` provides lazy parsing — the raw text is always available, and
parsing happens on demand:
- `.text` — raw string
- `.json()` — extract last fenced JSON block, parse with `json_repair` fallback
- `.yaml()` — extract last fenced YAML block
- `.scratchpad_json()` / `.scratchpad_yaml()` — split reasoning text from
  structured data, returns `(scratchpad: str, data: dict)`
- `.parse(PydanticModel, max_retries=2)` — validate against a Pydantic model
  with auto-retry on failure, using `previous_response_id` for efficient
  context chaining

## Agent Loop

`agent_loop()` runs a multi-turn tool-calling loop that continues until:
1. The LLM responds with content (no tool calls)
2. A tool marked `stop=True` is called
3. `max_tool_calls` is reached

Returns an `AgentResult` with `tool_call_count`, `stop_reason` (`"content"`,
`"stop_tool"`, or `"max_calls"`), `final_content`, and `response_id`.

The `on_tool_call` callback receives `(tool_name, arguments)` after each tool
execution. `previous_response_id` enables continuing a prior conversation.

## Sync Wrappers

For notebook/repl usage, synchronous wrappers call `asyncio.run()` internally:
- `sync_query()` — synchronous `query()`
- `sync_query_template()` — synchronous `query_template()`
- `sync_agent_loop()` — synchronous `agent_loop()`

## Templating

Jinja2 is a first-class feature. `prompt_dir` is required in the constructor.
Templates follow the naming convention `{name}_system.jinja` and
`{name}_user.jinja`, rendered via `query_template()` or `sync_query_template()`.

## Configuration

All backends read from `LLM_INTERACTION_*` env vars, overridable via constructor
kwargs. Set `LLM_INTERACTION_BACKEND` to switch providers without code changes.

| Backend | Env Vars |
|---------|----------|
| `azure` | `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_ENDPOINT`, `LLM_INTERACTION_MODEL` |
| `databricks` | `LLM_INTERACTION_DATABRICKS_HOST`, `LLM_INTERACTION_MODEL` |
| `openrouter` | `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_MODEL` |
| `litellm` | `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_ENDPOINT` (optional), `LLM_INTERACTION_MODEL` |
| `deepseek` | `LLM_INTERACTION_API_KEY`, `LLM_INTERACTION_MODEL` |
| `local` | `LLM_INTERACTION_ENDPOINT` (defaults to `http://localhost:8080/v1`), `LLM_INTERACTION_MODEL` |

**Databricks auth** is handled by `WorkspaceClient` from `databricks-sdk`:
- On-site (notebook): automatic
- Off-site (local dev): run `databricks auth login --host <host>` first
- Token refresh is handled transparently per API call

**LiteLLM env forwarding**: `LLM_INTERACTION_API_KEY` is automatically forwarded
to provider-specific env vars (e.g. `DATABRICKS_API_KEY`, `ANTHROPIC_API_KEY`)
based on the model prefix. The model must use litellm's `provider/model` format
(e.g. `databricks/my-claude`).

## Retry & Error Handling

All backends retry transient errors (rate limits, connection errors, API errors)
with exponential backoff and jitter. The `max_retries` constructor arg (default 3)
controls the number of attempts. After exhausting retries, a `RuntimeError` is
raised with the last exception message.

## Debugging

Each backend stores the request kwargs and raw API response object from its most
recent successful call. The `LLMInteraction.last_call` property returns these as
a `{"request": ..., "response": ...}` dict, or `None` if no call has been made.

```python
details = llm.last_call
print(details["request"])          # full kwargs sent to provider
print(details["response"].usage)   # token counts
print(details["response"].status)  # finish reason
```

The `response` is the raw SDK object, giving full access to provider-specific
metadata.
