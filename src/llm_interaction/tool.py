"""@tool decorator, ToolDef, and ToolContext for type-safe LLM tool calling."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, get_args, get_origin, get_type_hints

from llm_interaction._schema import _parse_google_docstring_args, _python_type_to_json_schema

T = TypeVar("T")


class ToolContext(Generic[T]):
    """Marker type for tool context injection.

    Parameters typed as ``ToolContext[X]`` are excluded from the LLM JSON
    schema and automatically injected at runtime from the ``context`` list
    passed to ``query()`` / ``agent_loop()``.

    Usage::

        @tool
        def sample(ctx: ToolContext[DocumentSampler], n: int) -> list[dict]:
            return ctx.sample(n)
    """

    def __init__(self, value: T):
        self._value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)


def _is_tool_context(annotation: Any) -> bool:
    """Check if an annotation is ToolContext[X]."""
    origin = get_origin(annotation)
    if origin is ToolContext:
        return True
    if annotation is ToolContext:
        return True
    return False


def _get_tool_context_type(annotation: Any) -> type | None:
    """Extract T from ToolContext[T]."""
    args = get_args(annotation)
    if args:
        return args[0]
    return None


@dataclass
class ToolDef:
    """Internal representation of a @tool-decorated function."""

    name: str
    description: str
    function: Callable
    is_stop: bool
    schema: dict[str, Any]
    context_params: dict[str, type]  # param_name -> required context type


def tool(
    fn: Callable | None = None,
    *,
    description: str | None = None,
    stop: bool = False,
) -> ToolDef | Callable[[Callable], ToolDef]:
    """Decorator to register a function as an LLM tool.

    Extracts name, description, and parameter schema from the function.

    Usage::

        @tool
        def my_tool(query: str) -> str:
            \"\"\"Search for something.\"\"\"
            ...

        @tool(stop=True)
        def submit(result: dict) -> str:
            \"\"\"Submit final answer.\"\"\"
            ...
    """

    def _wrap(f: Callable) -> ToolDef:
        hints = get_type_hints(f)
        sig = inspect.signature(f)
        doc = inspect.getdoc(f) or ""
        param_descriptions = _parse_google_docstring_args(doc)

        # Split first line as tool description
        tool_description = description or doc.split("\n")[0].strip() or f.__name__

        # Build schema, separating ToolContext params from LLM params
        properties: dict[str, Any] = {}
        required: list[str] = []
        context_params: dict[str, type] = {}

        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, str)

            if _is_tool_context(annotation):
                ctx_type = _get_tool_context_type(annotation)
                context_params[param_name] = ctx_type
                continue

            # Skip return annotation
            if param_name == "return":
                continue

            prop = _python_type_to_json_schema(annotation)
            if param_name in param_descriptions:
                prop["description"] = param_descriptions[param_name]

            properties[param_name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "name": f.__name__,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        return ToolDef(
            name=f.__name__,
            description=tool_description,
            function=f,
            is_stop=stop,
            schema=schema,
            context_params=context_params,
        )

    if fn is not None:
        # @tool without arguments
        return _wrap(fn)
    # @tool(...) with arguments
    return _wrap
