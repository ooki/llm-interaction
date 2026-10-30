"""Full turns through the real agent loop, driven by a scripted fake LLM."""

from conftest import call, say

from adventure.commands import render_inventory, render_look
from adventure.turn import FALLBACK_NARRATION


# --8<-- [start:test_invent_and_save]
async def test_invented_npc_and_item_are_saved(make_game, store):
    game, client = make_game(
        # Round 1: the model looks before inventing
        [call("search_world", query="goblin merchant")],
        # Round 2: invents an NPC and an item, rolls for haggling
        [call("create_being", name="Grik the Merchant", description="A goblin with a cart.",
              personality="Haggles about everything", disposition="greedy",
              location="here", properties={}),
         call("create_thing", name="Cursed Coin", description="Ice cold to the touch.",
              location="player", properties={"cursed": "yes"}),
         call("roll", dice_notation="1d20", reason="haggle, DC 12")],
        # Round 3: finishes the turn
        [call("narrate", text="A goblin sells you a coin.", summary="Bought a cursed coin.")],
    )

    reply = await game.handle("I look for a goblin merchant and buy something weird")

    assert reply["kind"] == "turn"
    assert reply["text"] == "A goblin sells you a coin."
    assert "New being: Grik the Merchant" in reply["changes"]

    saved = store.load("Test World")  # read back from disk
    assert saved.beings["grik-the-merchant"].location == "tavern"
    assert saved.things["cursed-coin"].properties == {"cursed": "yes"}
    assert saved.journal[-1].summary == "Bought a cursed coin."

    # The slash commands read the same saved data -- no LLM needed.
    assert "Grik the Merchant" in render_look(saved)
    assert "Cursed Coin" in render_inventory(saved)
    assert len(client.requests) == 3
# --8<-- [end:test_invent_and_save]


async def test_tool_errors_are_sent_back_to_the_model(make_game):
    game, client = make_game(
        [call("move_player", place_id="the-moon")],
        [call("create_place", name="The Moon", description="Grey dust.", tags=["moon"],
              exit_from_here="", exit_back=""),
         call("move_player", place_id="the-moon")],
        [call("narrate", text="You arrive on the moon.", summary="Went to the moon.")],
    )
    await game.handle("I jump to the moon")

    second_request = client.requests[1]["input"]
    error_output = next(i for i in second_request if i.get("type") == "function_call_output")
    assert "create_place" in error_output["output"]  # the model saw the hint
    assert game.world.player.location == "the-moon"


async def test_plain_text_answer_is_used_as_narration(make_game):
    game, _ = make_game([say("Nothing happens. The tavern hums along.")])
    reply = await game.handle("I wait")
    assert reply["text"] == "Nothing happens. The tavern hums along."
    assert reply["debug"]["stop_reason"] == "content"


async def test_running_out_of_tool_calls_still_saves(make_game, monkeypatch):
    from adventure import turn

    monkeypatch.setattr(turn, "MAX_TOOL_CALLS", 1)
    game, _ = make_game([call("set_player_status", key="form", value="a teapot")])
    reply = await game.handle("I become a teapot")
    assert reply["text"] == FALLBACK_NARRATION
    assert reply["debug"]["stop_reason"] == "max_calls"
    assert game.world.player.status["form"] == "a teapot"


async def test_failed_turn_rolls_back(make_game):
    game, _ = make_game(
        [call("set_player_status", key="form", value="a dragon")],
        # no more scripted responses -> the fake client raises
    )
    reply = await game.handle("I become a dragon")
    assert reply["kind"] == "error"
    assert game.world.player.status == {}


async def test_undo_restores_previous_world(make_game):
    game, _ = make_game(
        [call("destroy", entity_id="mug", how="smashed"),
         call("narrate", text="Smash.", summary="Smashed the mug.")],
    )
    await game.handle("I smash the mug")
    assert "mug" not in game.world.things
    reply = await game.handle("/undo")
    assert "Pewter Mug" in reply["text"]
    assert "mug" in game.world.things


async def test_each_turn_starts_fresh_with_world_state_in_the_prompt(make_game):
    game, client = make_game(
        [call("narrate", text="ok", summary="first")],
        [call("narrate", text="ok", summary="second")],
    )
    await game.handle("first thing")
    await game.handle("second thing")
    first, second = client.requests[0], client.requests[1]
    assert "previous_response_id" not in second
    user_prompt = second["input"][1]["content"]
    assert 'player did "first thing" -> first' in user_prompt
    assert "Old Wenna" in user_prompt
    assert first["input"][0]["role"] == "system"
