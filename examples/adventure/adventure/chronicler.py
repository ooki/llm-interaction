"""The chronicler: every few turns, condense the journal into a story summary + facts.

The turn prompt only shows the last few journal entries. Without this, older
events would fade from the model's view. The chronicler reasons in a free-form
scratchpad first and then answers with JSON -- ``scratchpad_json()`` splits the two.
"""

from __future__ import annotations

from typing import Any

from llm_interaction import LLMInteraction

from adventure.world import Fact, World

CHRONICLE_EVERY = 10


def is_due(world: World) -> bool:
    return world.turn - world.chronicled_turn >= CHRONICLE_EVERY


# --8<-- [start:chronicle]
async def chronicle(llm: LLMInteraction, world: World) -> dict[str, Any]:
    entries = [j for j in world.journal if j.turn > world.chronicled_turn]
    response = await llm.query_template("chronicle", {
        "world": world,
        "entries": entries,
        "known_facts": [f.text for f in world.facts],
    })
    scratchpad, data = response.scratchpad_json()

    world.story_so_far = str(data.get("story_so_far", world.story_so_far)).strip()
    new_facts = [str(f).strip() for f in data.get("new_facts", []) if str(f).strip()]
    world.facts.extend(Fact(text=f, tags=["chronicle"], turn=world.turn) for f in new_facts)
    world.chronicled_turn = world.turn
    return {"scratchpad": scratchpad, "story_so_far": world.story_so_far, "new_facts": new_facts}
# --8<-- [end:chronicle]
