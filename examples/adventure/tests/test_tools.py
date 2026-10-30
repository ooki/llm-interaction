"""Call tool functions directly -- no LLM, no agent loop."""

import random

import pytest
from llm_interaction import ToolContext

from adventure import tools
from adventure.tools import TurnLog
from adventure.world import PLAYER, World


def run(tool_def, world: World, log: TurnLog, **kwargs):
    """Invoke a ToolDef the way agent_loop does: inject ToolContext params by type."""
    injected = {}
    for param, ctx_type in tool_def.context_params.items():
        injected[param] = ToolContext(world if ctx_type is World else log)
    return tool_def.function(**injected, **kwargs)


@pytest.fixture
def log() -> TurnLog:
    return TurnLog(rng=random.Random(7))


def test_create_place_with_exits(world, log):
    out = run(tools.create_place, world, log, name="Smugglers' Cellar",
              description="Damp and full of crates.", tags=["Cellar", " hidden "],
              exit_from_here="down the trapdoor", exit_back="up the ladder")
    pid = out["id"]
    assert pid == "smugglers-cellar"
    assert world.places[pid].tags == ["cellar", "hidden"]
    assert world.places["tavern"].exits["down the trapdoor"] == pid
    assert world.places[pid].exits["up the ladder"] == "tavern"
    assert log.changes == ["New place: Smugglers' Cellar"]


def test_create_thing_in_inventory_and_container(world, log):
    out = run(tools.create_thing, world, log, name="Cursed Coin", description="Cold.",
              location="player", properties={"cursed": "yes"})
    assert world.things[out["id"]].location == PLAYER
    inner = run(tools.create_thing, world, log, name="Map Scrap", description="Torn.",
                location="satchel", properties={})
    assert world.things[inner["id"]].location == "satchel"


def test_unknown_location_is_a_helpful_error(world, log):
    with pytest.raises(ValueError, match="search_world"):
        run(tools.create_thing, world, log, name="X", description="Y", location="atlantis",
            properties={})


def test_create_being_and_update_memory(world, log):
    out = run(tools.create_being, world, log, name="Grik the Merchant",
              description="A goblin with a cart.", personality="Haggles", disposition="greedy",
              location="here", properties={"gold": "40"})
    gid = out["id"]
    run(tools.update_being, world, log, being_id=gid, disposition="furious",
        remember="The player sold me a fake gem.")
    grik = world.beings[gid]
    assert grik.location == "tavern"
    assert grik.disposition == "furious"
    assert grik.memory == ["The player sold me a fake gem."]


def test_move_prevents_containment_loops(world, log):
    with pytest.raises(ValueError, match="inside itself"):
        run(tools.move, world, log, entity_id="satchel", destination="satchel")
    run(tools.move, world, log, entity_id="mug", destination="satchel")
    with pytest.raises(ValueError, match="inside itself"):
        run(tools.move, world, log, entity_id="satchel", destination="mug")


def test_move_player_requires_existing_place(world, log):
    with pytest.raises(ValueError, match="create_place"):
        run(tools.move_player, world, log, place_id="the-moon")
    run(tools.move_player, world, log, place_id="docks")
    assert world.player.location == "docks"
    assert world.places["docks"].visited


def test_destroy_drops_contents(world, log):
    run(tools.move, world, log, entity_id="mug", destination="satchel")
    run(tools.destroy, world, log, entity_id="satchel", how="eaten by a goat")
    assert "satchel" not in world.things
    assert world.things["mug"].location == PLAYER
    assert tools.destroy.description.startswith("Permanently remove")


def test_player_status_is_open_ended(world, log):
    run(tools.set_player_status, world, log, key="form", value="a sentient fog")
    assert world.player.status == {"form": "a sentient fog"}
    run(tools.set_player_status, world, log, key="form", value="")
    assert world.player.status == {}


def test_roll_uses_engine_rng_and_logs(world, log):
    out = run(tools.roll, world, log, dice_notation="1d20+2", reason="leap the chasm, DC 12")
    assert 3 <= out["total"] <= 22
    assert log.rolls == [out]
    assert log.changes[0].startswith("Rolled 1d20")


def test_narrate_is_a_stop_tool_and_finishes_the_log(world, log):
    assert tools.narrate.is_stop
    run(tools.narrate, world, log, text="You win.", summary="Won.")
    assert (log.narration, log.summary) == ("You win.", "Won.")


def test_context_params_are_hidden_from_the_schema():
    props = tools.create_thing.schema["parameters"]["properties"]
    assert "world" not in props and "log" not in props
    assert set(tools.create_thing.context_params) == {"world", "log"}
