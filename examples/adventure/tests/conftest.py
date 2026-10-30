"""Shared fixtures: a tiny hand-built world and a scripted fake LLM client."""

from __future__ import annotations

import itertools
import json
import random
from types import SimpleNamespace
from typing import Any

import pytest
from llm_interaction import LLMInteraction

from adventure.app import PROMPT_DIR
from adventure.game import Game
from adventure.world import Being, Place, Player, Thing, World, WorldStore

_ids = itertools.count(1)


# --8<-- [start:fake_client]
def call(tool_name: str, /, **arguments: Any) -> SimpleNamespace:
    """One scripted tool call, shaped like a Responses API output item."""
    return SimpleNamespace(type="function_call", call_id=f"call_{next(_ids)}",
                           name=tool_name, arguments=json.dumps(arguments))


def say(text: str) -> SimpleNamespace:
    """A plain text answer from the model."""
    return SimpleNamespace(type="message", content=[SimpleNamespace(text=text)])


class ScriptedClient:
    """Stands in for AsyncOpenAI. Each create() returns the next scripted response."""

    def __init__(self, *responses: list[SimpleNamespace]):
        self.responses = self  # so that client.responses.create(...) works
        self._queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if not self._queue:
            raise AssertionError("The fake LLM was called more times than scripted")
        return SimpleNamespace(id=f"resp_{next(_ids)}", output=self._queue.pop(0))


def scripted_llm(*responses: list[SimpleNamespace]) -> tuple[LLMInteraction, ScriptedClient]:
    client = ScriptedClient(*responses)
    return LLMInteraction(prompt_dir=PROMPT_DIR, client=client, model="fake"), client
# --8<-- [end:fake_client]


def make_world() -> World:
    tavern = Place(id="tavern", name="The Gilded Gull", visited=True,
                   description="A smoky harbour tavern.", tags=["tavern", "harbour"],
                   exits={"out the door": "docks"})
    docks = Place(id="docks", name="The Docks", description="Wet planks and gulls.",
                  tags=["harbour"], exits={"into the tavern": "tavern"})
    world = World(name="Test World", premise="A test.", player=Player(location="tavern"),
                  places={"tavern": tavern, "docks": docks})
    world.things["mug"] = Thing(id="mug", name="Pewter Mug", description="Dented.",
                                location="tavern")
    world.things["satchel"] = Thing(id="satchel", name="Leather Satchel",
                                    description="Worn.", location="player")
    world.beings["barkeep"] = Being(id="barkeep", name="Old Wenna", description="Stout.",
                                    personality="Gruff but fair.", disposition="neutral",
                                    location="tavern")
    return world


@pytest.fixture
def world() -> World:
    return make_world()


@pytest.fixture
def store(tmp_path) -> WorldStore:
    return WorldStore(tmp_path / "saves")


@pytest.fixture
def make_game(store):
    """Build a Game with a loaded test world and the given scripted LLM responses."""

    def _make(*responses: list[SimpleNamespace], with_world: bool = True):
        llm, client = scripted_llm(*responses)
        game = Game(llm, store, rng=random.Random(42))
        if with_world:
            game.world = make_world()
            game.save()
        return game, client

    return _make
