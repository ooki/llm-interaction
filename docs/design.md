# Design

`llm-interaction` is an async Azure OpenAI Responses API client with typed tool calling and lazy output parsing.

## Architecture

The library is split into focused modules:

| Module | Responsibility |
|--------|---------------|
| `tool.py` | `@tool` decorator, `ToolDef`, `ToolContext` |
| `_schema.py` | Python type → JSON Schema conversion, Google-style docstring parsing |
| `parsing.py` | JSON/YAML parsing, fenced block extraction, scratchpad splitting, response output extraction |
| `response.py` | `LLMResponse` (lazy parsing), `AgentResult` |
| `client.py` | `LLMInteraction` class, context matching, API retry logic |
| `__init__.py` | Public API re-exports |

Provider-specific logic is isolated in `client.py` behind clear method boundaries:
- `_call_api()` — makes the API call
- `_extract_text_from_output()` / `_extract_function_calls()` — parse provider response format

Everything else (tools, parsing, ToolContext, LLMResponse) is provider-agnostic, making it straightforward to add non-Azure or non-OpenAI backends later.

## Tool Definition

Tools are defined with a `@tool` decorator that generates JSON Schema from:
- **Function name** → tool name
- **Docstring first line** → tool description
- **Type hints** → parameter types in JSON Schema
- **Google-style `Args:` section** → per-parameter descriptions
- **Default values** → required vs optional

A `stop=True` argument marks a tool as a loop terminator for `agent_loop()`.

## Dependency Injection

`ToolContext[T]` is a generic marker type for injecting runtime context into tools. Parameters typed as `ToolContext[T]` are:
- Excluded from the JSON schema sent to the LLM
- Automatically matched by type from the `context` argument passed to `query()` / `agent_loop()`

A single tool can have multiple `ToolContext` parameters of different types. The matching is strict: each type may appear at most once in the context list.

## Output Parsing

`LLMResponse` provides lazy parsing — the raw text is always available, and parsing happens on demand:
- `.text` — raw string
- `.json()` — extract last fenced JSON block, parse with `json_repair` fallback
- `.yaml()` — extract last fenced YAML block
- `.scratchpad_json()` / `.scratchpad_yaml()` — split reasoning text from structured data
- `.parse(PydanticModel)` — validate against a Pydantic model with auto-retry on failure

The `.parse()` method uses `previous_response_id` to chain retry requests efficiently, sending the validation error back to the LLM as context.

## Agent Loop

`agent_loop()` runs a multi-turn tool-calling loop that continues until:
1. The LLM responds with content (no tool calls)
2. A tool marked `stop=True` is called
3. `max_tool_calls` is reached

It returns an `AgentResult` with the stop reason, tool call count, and response ID for continuation.

## Configuration

Environment variables with `LLM_INTERACTION_*` prefix:
- `LLM_INTERACTION_API_KEY`
- `LLM_INTERACTION_ENDPOINT`
- `LLM_INTERACTION_MODEL`

All three can be overridden via constructor arguments. `python-dotenv` is used to load `.env` files automatically.

## Templating

Jinja2 is a first-class feature. `prompt_dir` is required in the constructor. Templates follow the naming convention `{name}_system.jinja` and `{name}_user.jinja`, rendered via `query_template()`.

## Future Considerations

- **Backend abstraction**: The provider-specific methods in `client.py` can be extracted into backend classes (e.g., `AzureBackend`, `OpenAIBackend`, `ClaudeBackend`) without changing the rest of the library.
- **Claude support** would require: different tool schema format, message-based history instead of `previous_response_id`, and different response extraction logic. The parsing, tool, and response layers would remain unchanged.
