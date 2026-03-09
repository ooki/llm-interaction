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

# Simple query
result = await llm.query(system="Be helpful", user="Hello")
print(result.text)

# Parse as JSON
data = result.json()

# Parse as Pydantic model (with auto-retry on validation failure)
from pydantic import BaseModel

class Topics(BaseModel):
    topics: list[str]

topics = await result.parse(Topics)
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

## Context Injection

```python
@tool
def sample(ctx: ToolContext[DocumentSampler], n: int) -> list[dict]:
    """Sample documents from the database."""
    return ctx.sample(n)

result = await llm.agent_loop(
    system="...", user="...",
    tools=[sample],
    context=[my_sampler],  # matched by type
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
