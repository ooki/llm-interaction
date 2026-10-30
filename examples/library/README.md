# Library: llm-interaction in four small steps

A tiny library checkout system that adds one feature of
[`llm-interaction`](../../README.md) at a time. All steps share the catalog in
`catalog.py`.

| Step | File | New concept |
|------|------|-------------|
| 1 | `01_simple_prompt.py` | `LLMInteraction` and `sync_query(system, user)` |
| 2 | `02_template_prompt.py` | Jinja templates (`prompts/librarian_*.jinja`), `sync_query_template`, `.json()` |
| 3 | `03_tools.py` | `@tool`, `@tool(stop=True)`, `sync_agent_loop`, `on_tool_call` |
| 4 | `04_tools_with_context.py` | `ToolContext[T]` and `context=[...]` instead of globals |

Steps 3 and 4 solve the same task, so their diff shows only the move to `ToolContext`.

## Run

From the repo root, with the `LLM_INTERACTION_*` env vars set (a `.env` in the repo root works):

```bash
uv run python examples/library/01_simple_prompt.py
uv run python examples/library/02_template_prompt.py
uv run python examples/library/03_tools.py
uv run python examples/library/04_tools_with_context.py
```

The examples use the `sync_*` methods to keep them short. Every one has an
async version (`query`, `query_template`, `agent_loop`) that you `await`.
