"""Smoke tests for examples/library: templates render and tool schemas hide context."""

import importlib.util
import sys
from pathlib import Path

import pytest

from llm_interaction import LLMInteraction

LIBRARY_DIR = Path(__file__).parent.parent / "examples" / "library"


@pytest.fixture(scope="module")
def load():
    sys.path.insert(0, str(LIBRARY_DIR))

    def _load(name):
        spec = importlib.util.spec_from_file_location(name, LIBRARY_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield _load
    sys.path.remove(str(LIBRARY_DIR))


def test_examples_import(load):
    for name in ["01_simple_prompt", "02_template_prompt", "03_tools", "04_tools_with_context"]:
        assert callable(load(name).main)


def test_librarian_template_renders(load):
    books = load("catalog").BOOKS
    llm = LLMInteraction(prompt_dir=LIBRARY_DIR / "prompts", model="m", client=object())
    system = llm.render("librarian_system.jinja", {"books": books})
    user = llm.render("librarian_user.jinja", {"patron": "Ada", "request": "Dragons please."})
    assert '[b1] "The Hobbit" by J.R.R. Tolkien (fantasy)' in system
    assert user.strip() == "My name is Ada. Dragons please."


def test_tools_schema(load):
    tools = load("03_tools")
    assert tools.finish.is_stop
    assert tools.check_out.schema["parameters"]["required"] == ["book_id"]


def test_context_params_hidden_from_schema(load):
    m = load("04_tools_with_context")
    assert set(m.check_out.schema["parameters"]["properties"]) == {"book_id"}
    assert m.check_out.context_params == {"library": m.Library, "patron": m.Patron}


def test_library_check_out_rules(load):
    m = load("04_tools_with_context")
    library = m.Library(books=load("catalog").BOOKS, loans={"b4": "Grace"})
    assert "already checked out" in library.check_out("b4", "Ada", 1)
    assert library.check_out("b2", "Ada", 1) == "Checked out b2 to Ada."
    assert "limit" in library.check_out("b1", "Ada", 1)
    assert library.loans == {"b4": "Grace", "b2": "Ada"}
