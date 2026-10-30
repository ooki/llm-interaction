"""Build the per-turn prompt variables from the saved world.

The model gets no chat history. Every turn starts fresh, and *this* function
decides what it sees: where the player is, what's around, what they carry,
the last few journal entries and any facts that look relevant. If it isn't in
here, the model doesn't know it -- which is exactly why inventing must be saved.
"""

from __future__ import annotations

from typing import Any

from adventure.world import PLAYER, Thing, World

RECENT_JOURNAL = 8
MAX_FACTS = 10
MAX_KNOWN_PLACES = 40


def _thing_view(world: World, thing: Thing) -> dict[str, Any]:
    return {
        "id": thing.id,
        "name": thing.name,
        "description": thing.description,
        "properties": thing.properties,
        "contains": [f"{t.name} ({t.id})" for t in world.contents(thing.id)],
    }


# --8<-- [start:relevant_facts]
def relevant_facts(world: World, action: str) -> list[str]:
    """Facts matching the action or the current place, plus the newest few."""
    here = world.here
    query = f"{action} {here.name} {' '.join(here.tags)}"
    matched = [hit["text"] for hit in world.search(query, limit=50) if hit["kind"] == "fact"]
    newest = [f.text for f in world.facts[-3:]]
    return list(dict.fromkeys(matched + newest))[:MAX_FACTS]
# --8<-- [end:relevant_facts]


# --8<-- [start:build_turn_context]
def build_turn_context(world: World, action: str) -> dict[str, Any]:
    here = world.here
    return {
        "world": world,
        "status": world.player.status,
        "here": here,
        "exits": [
            {"label": label, "id": pid, "name": world.places[pid].name}
            for label, pid in here.exits.items()
            if pid in world.places
        ],
        "things_here": [_thing_view(world, t) for t in world.contents(here.id)],
        "beings_here": world.beings_at(here.id),
        "followers": world.beings_at(PLAYER),
        "inventory": [_thing_view(world, t) for t in world.contents(PLAYER)],
        "known_places": [
            {"id": p.id, "name": p.name}
            for p in list(world.places.values())[:MAX_KNOWN_PLACES]
            if p.id != here.id
        ],
        "recent": world.journal[-RECENT_JOURNAL:],
        "facts": relevant_facts(world, action),
        "action": action,
    }
# --8<-- [end:build_turn_context]
