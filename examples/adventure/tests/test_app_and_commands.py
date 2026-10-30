from fastapi.testclient import TestClient
from markupsafe import escape

from adventure.app import create_app
from adventure.tutorial import PAGES, SnippetError, extract, render_index, render_page


def test_commands_never_call_the_llm(make_game):
    game, client = make_game()  # no scripted responses: any LLM call would fail
    app = TestClient(create_app(game))
    for cmd in ["/help", "/look", "/inv", "/me", "/map", "/journal", "/facts", "/roll 2d6+1",
                "/worlds", "/debug", "/undo", "/nonsense"]:
        reply = app.post("/api/command", json={"text": cmd}).json()
        assert reply["kind"] == "command", cmd
    assert client.requests == []


def test_look_and_state(make_game):
    game, _ = make_game()
    app = TestClient(create_app(game))
    reply = app.post("/api/command", json={"text": "/look"}).json()
    assert "The Gilded Gull" in reply["text"]
    assert "Pewter Mug" in reply["text"]
    assert "Old Wenna (neutral)" in reply["text"]
    state = app.get("/api/state").json()["state"]
    assert state["place"] == "The Gilded Gull"
    assert state["counts"]["carried"] == 1


def test_no_world_hint(make_game):
    game, _ = make_game(with_world=False)
    reply = TestClient(create_app(game)).post("/api/command", json={"text": "dance"}).json()
    assert "/new" in reply["text"]


def test_index_and_static_served(make_game):
    game, _ = make_game()
    app = TestClient(create_app(game))
    assert "game.js" in app.get("/").text
    assert app.get("/static/game.js").status_code == 200


# --8<-- [start:test_tutorial]
def test_every_tutorial_page_renders_with_real_snippets(make_game):
    """A refactor that breaks a snippet reference fails this test."""
    app = TestClient(create_app(make_game()[0]))
    assert app.get("/tutorial").status_code == 200
    for slug, title in PAGES:
        response = app.get(f"/tutorial/{slug}")
        assert response.status_code == 200, slug
        assert str(escape(title)) in response.text
    render_index(lambda s: s)
    for slug, _ in PAGES:
        render_page(slug, lambda s: f"{s}.html")
# --8<-- [end:test_tutorial]


def test_missing_snippet_raises():
    import pytest

    with pytest.raises(SnippetError):
        extract("tools.py", "does-not-exist")
    with pytest.raises(SnippetError):
        extract("../../../../etc/passwd")
    code, first, last = extract("dice.py", "roll")
    assert code.startswith("def roll(") and first < last
    assert "--8<--" not in code
