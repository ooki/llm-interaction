import json

from conftest import say

from adventure import chronicler
from adventure.world import JournalEntry

SEED = {
    "world_name": "The Salt Road",
    "premise": "A caravan crosses a desert of salt.",
    "player_status": {"role": "caravan guard"},
    "start": {"name": "Caravan Camp", "description": "Wagons circled at dusk.", "tags": ["camp"]},
    "nearby": [{"name": "Salt Flats", "description": "Blinding white.", "tags": [],
                "exit_label": "east", "back_label": "back to camp"}],
    "things": [{"name": "Waterskin", "description": "Half full.", "carried": True},
               {"name": "Broken Wheel", "description": "Snapped spokes."}],
    "beings": [{"name": "Master Odo", "description": "Caravan master.",
                "personality": "Nervous", "disposition": "friendly"}],
    "facts": ["Nobody crosses the flats at night."],
    "opening": "The wagons creak to a halt.",
}


def fenced(data) -> str:
    return f"```json\n{json.dumps(data)}\n```"


async def test_new_world_with_random_premise(make_game, store):
    game, client = make_game(
        [say(fenced({"premise": "A caravan crosses a desert of salt."}))],  # .json()
        [say(fenced(SEED))],                                                 # .parse()
        with_world=False,
    )
    reply = await game.handle("/new")

    world = game.world
    assert world.name == "The Salt Road"
    assert reply["text"].startswith("The wagons creak to a halt.")
    assert world.things["waterskin"].location == "player"
    assert world.things["broken-wheel"].location == "caravan-camp"
    assert world.places["caravan-camp"].exits["east"] == "salt-flats"
    assert world.places["salt-flats"].exits["back to camp"] == "caravan-camp"
    assert world.beings["master-odo"].location == "caravan-camp"
    assert store.load("The Salt Road") == world
    assert "Draw inspiration" in client.requests[0]["input"][1]["content"]


async def test_genesis_parse_retries_on_invalid_output(make_game):
    bad = {k: v for k, v in SEED.items() if k != "start"}  # missing required field
    game, client = make_game([say(fenced(bad))], [say(fenced(SEED))], with_world=False)
    await game.handle("/new salt desert caravan")
    assert game.world.name == "The Salt Road"
    retry_prompt = client.requests[1]["input"][0]["content"]
    assert "did not match the required schema" in retry_prompt


async def test_chronicler_uses_scratchpad_json(make_game):
    game, _ = make_game(
        [say("Thinking: the player owes the witch a favour.\n\n"
             + fenced({"story_so_far": "You arrived and made a deal.",
                       "new_facts": ["The player owes Mother Grell a favour."]}))],
    )
    world = game.world
    world.journal += [JournalEntry(turn=i, action="a", summary="s") for i in range(1, 11)]
    world.turn = 10
    assert chronicler.is_due(world)

    result = await chronicler.chronicle(game.llm, world)

    assert "owes the witch" in result["scratchpad"]
    assert world.story_so_far == "You arrived and made a deal."
    assert world.facts[-1].text == "The player owes Mother Grell a favour."
    assert not chronicler.is_due(world)
