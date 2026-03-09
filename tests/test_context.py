"""Tests for context matching: _build_context_map and _inject_context."""

import pytest

from llm_interaction import ToolContext, ToolDef
from llm_interaction.client import _build_context_map, _inject_context


class TestContextMatching:
    def test_build_context_map_single(self):
        obj = "hello"
        ctx_map = _build_context_map(obj)
        assert ctx_map == {str: "hello"}

    def test_build_context_map_list(self):
        ctx_map = _build_context_map([42, "text"])
        assert ctx_map[int] == 42
        assert ctx_map[str] == "text"

    def test_build_context_map_none(self):
        assert _build_context_map(None) == {}

    def test_build_context_map_duplicate_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _build_context_map(["a", "b"])

    def test_inject_context(self):
        class MyDB:
            pass

        db_instance = MyDB()

        td = ToolDef(
            name="test",
            description="test",
            function=lambda ctx, x: None,
            is_stop=False,
            schema={},
            context_params={"ctx": MyDB},
        )

        kwargs = _inject_context(td, {"x": 1}, {MyDB: db_instance})
        assert isinstance(kwargs["ctx"], ToolContext)
        assert kwargs["x"] == 1

    def test_inject_context_missing_raises(self):
        class MissingType:
            pass

        td = ToolDef(
            name="test",
            description="test",
            function=lambda ctx: None,
            is_stop=False,
            schema={},
            context_params={"ctx": MissingType},
        )

        with pytest.raises(RuntimeError, match="no matching context"):
            _inject_context(td, {}, {})

    def test_inject_context_multiple(self):
        class ServiceA:
            pass

        class ServiceB:
            pass

        a = ServiceA()
        b = ServiceB()

        td = ToolDef(
            name="test",
            description="test",
            function=lambda ca, cb, x: None,
            is_stop=False,
            schema={},
            context_params={"ca": ServiceA, "cb": ServiceB},
        )

        kwargs = _inject_context(td, {"x": 1}, {ServiceA: a, ServiceB: b})
        assert isinstance(kwargs["ca"], ToolContext)
        assert isinstance(kwargs["cb"], ToolContext)
        assert kwargs["x"] == 1
