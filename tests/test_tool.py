"""Tests for @tool decorator, ToolDef, and ToolContext."""

from llm_interaction import ToolContext, ToolDef, tool
from llm_interaction._schema import _parse_google_docstring_args, _python_type_to_json_schema


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_basic_tool(self):
        @tool
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello {name}"

        assert isinstance(greet, ToolDef)
        assert greet.name == "greet"
        assert greet.description == "Say hello to someone."
        assert greet.is_stop is False
        assert greet.schema["name"] == "greet"
        props = greet.schema["parameters"]["properties"]
        assert "name" in props
        assert props["name"]["type"] == "string"
        assert greet.schema["parameters"]["required"] == ["name"]

    def test_tool_with_stop(self):
        @tool(stop=True)
        def submit(result: dict) -> str:
            """Submit final answer."""
            return "done"

        assert submit.is_stop is True

    def test_tool_with_description_override(self):
        @tool(description="Custom description")
        def my_func(x: int) -> int:
            """This docstring is ignored."""
            return x

        assert my_func.description == "Custom description"

    def test_tool_multiple_params(self):
        @tool
        def search(query: str, max_results: int, include_metadata: bool = False) -> list:
            """Search for items."""
            return []

        props = search.schema["parameters"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"
        assert props["include_metadata"]["type"] == "boolean"
        assert search.schema["parameters"]["required"] == ["query", "max_results"]

    def test_tool_complex_types(self):
        @tool
        def process(items: list[dict], config: dict[str, int]) -> list[str]:
            """Process items."""
            return []

        props = process.schema["parameters"]["properties"]
        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "object"
        assert props["config"]["type"] == "object"

    def test_tool_with_tool_context(self):
        class MyService:
            pass

        @tool
        def use_service(ctx: ToolContext[MyService], query: str) -> str:
            """Use the service."""
            return query

        assert "ctx" not in use_service.schema["parameters"]["properties"]
        assert "query" in use_service.schema["parameters"]["properties"]
        assert use_service.context_params == {"ctx": MyService}

    def test_tool_with_multiple_contexts(self):
        class ServiceA:
            pass

        class ServiceB:
            pass

        @tool
        def multi_ctx(
            a: ToolContext[ServiceA], b: ToolContext[ServiceB], value: int
        ) -> str:
            """Uses two services."""
            return str(value)

        assert "a" not in multi_ctx.schema["parameters"]["properties"]
        assert "b" not in multi_ctx.schema["parameters"]["properties"]
        assert "value" in multi_ctx.schema["parameters"]["properties"]
        assert multi_ctx.context_params == {"a": ServiceA, "b": ServiceB}

    def test_tool_callable(self):
        @tool
        def adder(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        assert adder.function(x=2, y=3) == 5


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


class TestDocstringParsing:
    def test_google_style(self):
        doc = """Search for documents.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
        """
        result = _parse_google_docstring_args(doc)
        assert result["query"] == "The search query string"
        assert result["max_results"] == "Maximum number of results to return"

    def test_multiline_description(self):
        doc = """Do something.

        Args:
            param: This is a long
                description that spans
                multiple lines
        """
        result = _parse_google_docstring_args(doc)
        assert "long" in result["param"]
        assert "multiple lines" in result["param"]

    def test_empty_docstring(self):
        assert _parse_google_docstring_args(None) == {}
        assert _parse_google_docstring_args("") == {}

    def test_no_args_section(self):
        doc = """Just a description with no arguments."""
        assert _parse_google_docstring_args(doc) == {}

    def test_args_with_returns_section(self):
        doc = """Do something.

        Args:
            x: The x value

        Returns:
            Something useful
        """
        result = _parse_google_docstring_args(doc)
        assert result == {"x": "The x value"}

    def test_param_descriptions_in_schema(self):
        @tool
        def search(query: str, limit: int = 10) -> list:
            """Search for items.

            Args:
                query: The search query string
                limit: Max results to return
            """
            return []

        props = search.schema["parameters"]["properties"]
        assert props["query"]["description"] == "The search query string"
        assert props["limit"]["description"] == "Max results to return"


# ---------------------------------------------------------------------------
# Type to JSON schema
# ---------------------------------------------------------------------------


class TestTypeToJsonSchema:
    def test_primitives(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}
        assert _python_type_to_json_schema(float) == {"type": "number"}
        assert _python_type_to_json_schema(str) == {"type": "string"}
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert _python_type_to_json_schema(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_dict(self):
        result = _python_type_to_json_schema(dict[str, int])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "integer"}

    def test_bare_list(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}
