"""The LLM's hands: every way the model can look at or change the world.

Rules of thumb used throughout this file:

* Tools are the *only* way the world changes. If the model narrates something
  without calling a tool, it didn't happen.
* Tools validate their input and ``raise ValueError`` with a helpful message.
  ``agent_loop`` catches the exception and sends ``{"error": "..."}`` back to the
  model, which then fixes its call. Errors are feedback, not crashes.
* Each mutating tool appends a short, human-readable line to ``TurnLog.changes``
  so the player can see exactly what the LLM did to the world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from llm_interaction import ToolContext, tool

from adventure import dice
from adventure.world import PLAYER, Being, Fact, Place, Thing, World


# --8<-- [start:turnlog]
@dataclass
class TurnLog:
    """Per-turn scratch space, injected next to the World via ToolContext."""

    changes: list[str] = field(default_factory=list)
    rolls: list[dict[str, Any]] = field(default_factory=list)
    narration: str = ""
    summary: str = ""
    rng: random.Random = field(default_factory=random.Random)

    def finish(self, narration: str, summary: str) -> None:
        # Tools receive a ToolContext *proxy*: reading attributes and calling
        # methods reaches the real TurnLog, but `log.narration = ...` would only
        # set an attribute on the proxy. So assignments go through a method.
        self.narration = narration
        self.summary = summary
# --8<-- [end:turnlog]


def _clean(props: dict[str, str]) -> dict[str, str]:
    return {str(k).strip(): str(v).strip() for k, v in props.items() if str(k).strip()}


def _require(text: str, what: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError(f"{what} must not be empty.")
    return text


# ---------------------------------------------------------------------------
# Looking
# ---------------------------------------------------------------------------


# --8<-- [start:search_world]
@tool
def search_world(world: ToolContext[World], query: str) -> list[dict[str, str]]:
    """Search everything that already exists (places, things, beings, facts) by keywords.

    Always search before inventing something, so you reuse what exists instead
    of creating duplicates.

    Args:
        query: A few keywords, e.g. "blacksmith hammer" or "crypt".
    """
    return world.search(query) or [{"result": "nothing found -- you may invent it"}]
# --8<-- [end:search_world]


@tool
def inspect(world: ToolContext[World], entity_id: str) -> dict[str, Any]:
    """Get full details of a place, thing or being, including what it contains.

    Args:
        entity_id: The id of the entity (from the context or search_world).
    """
    entity = world.entity(entity_id)
    data = entity.model_dump()
    data["kind"] = world.kind(entity_id)
    data["contains"] = [t.name + f" ({t.id})" for t in world.contents(entity_id)]
    if isinstance(entity, Place):
        data["beings_here"] = [b.name + f" ({b.id})" for b in world.beings_at(entity_id)]
    return data


# ---------------------------------------------------------------------------
# Inventing
# ---------------------------------------------------------------------------


# --8<-- [start:create_place]
@tool
def create_place(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    name: str,
    description: str,
    tags: list[str],
    exit_from_here: str = "",
    exit_back: str = "",
) -> dict[str, str]:
    """Invent a new place and save it to the world.

    Args:
        name: Short name, e.g. "The Drowned Chapel".
        description: Two or three sentences of what it looks, sounds and smells like.
        tags: Keywords for later searches, e.g. ["underground", "holy", "flooded"].
        exit_from_here: If set, adds an exit with this label from the player's current place
            to the new place (e.g. "down the well").
        exit_back: If set, adds the return exit from the new place (e.g. "up the rope").
    """
    place = Place(
        id=world.unique_id(_require(name, "name")),
        name=name.strip(),
        description=_require(description, "description"),
        tags=[t.strip().lower() for t in tags if t.strip()],
    )
    world.places[place.id] = place
    if exit_from_here:
        world.here.exits[exit_from_here.strip()] = place.id
    if exit_back:
        place.exits[exit_back.strip()] = world.player.location
    log.changes.append(f"New place: {place.name}")
    return {"id": place.id, "created": place.name}
# --8<-- [end:create_place]


# --8<-- [start:create_thing]
@tool
def create_thing(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    name: str,
    description: str,
    location: str,
    properties: dict[str, str],
) -> dict[str, str]:
    """Invent a new item or object and save it to the world.

    Args:
        name: Short name, e.g. "Cracked Lantern".
        description: One or two sentences.
        location: Where it is: "here" (the current place), "player" (the player's
            inventory), or the id of a place, being or container thing.
        properties: Open-ended key/value facts, e.g. {"lit": "no", "weight": "light"}.
            Use {} if there is nothing special.
    """
    where = world.resolve_location(location)
    thing = Thing(
        id=world.unique_id(_require(name, "name")),
        name=name.strip(),
        description=_require(description, "description"),
        location=where,
        properties=_clean(properties),
    )
    world.things[thing.id] = thing
    log.changes.append(f"New thing: {thing.name} ({world.location_name(where)})")
    return {"id": thing.id, "created": thing.name, "location": where}
# --8<-- [end:create_thing]


# --8<-- [start:create_being]
@tool
def create_being(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    name: str,
    description: str,
    personality: str,
    disposition: str,
    location: str,
    properties: dict[str, str],
) -> dict[str, str]:
    """Invent a new character or creature (an NPC) and save it to the world.

    Create a being BEFORE it speaks or acts, so it persists.

    Args:
        name: Name or title, e.g. "Mother Grell, the toll-witch".
        description: What they look like.
        personality: How they talk and what they want.
        disposition: How they feel about the player right now, e.g. "wary", "friendly".
        location: "here", "player" (they follow the player), or a place id.
        properties: Open-ended key/value facts, e.g. {"hp": "12", "secret": "is a dragon"}.
            Use {} if there is nothing special.
    """
    where = world.resolve_location(location)
    if where != PLAYER and where not in world.places:
        raise ValueError("A being's location must be 'here', 'player' or a place id.")
    being = Being(
        id=world.unique_id(_require(name, "name")),
        name=name.strip(),
        description=_require(description, "description"),
        personality=personality.strip(),
        disposition=disposition.strip() or "neutral",
        location=where,
        properties=_clean(properties),
    )
    world.beings[being.id] = being
    log.changes.append(f"New being: {being.name}")
    return {"id": being.id, "created": being.name}
# --8<-- [end:create_being]


@tool
def add_fact(
    world: ToolContext[World], log: ToolContext[TurnLog], text: str, tags: list[str]
) -> str:
    """Record a piece of lore, history, rumour or rule of this world so it stays true later.

    Args:
        text: One sentence, e.g. "The bridge trolls of Vell only accept payment in riddles."
        tags: Keywords for later searches.
    """
    world.facts.append(
        Fact(text=_require(text, "text"), tags=[t.lower() for t in tags], turn=world.turn)
    )
    log.changes.append(f"New fact: {text.strip()}")
    return "fact saved"


# ---------------------------------------------------------------------------
# Changing
# ---------------------------------------------------------------------------


@tool
def connect_places(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    from_id: str,
    to_id: str,
    exit_label: str,
    back_label: str = "",
) -> str:
    """Add an exit between two existing places (and optionally the way back).

    Args:
        from_id: Place id the exit starts from ("here" for the current place).
        to_id: Place id the exit leads to.
        exit_label: Label of the exit, e.g. "north", "through the mirror".
        back_label: If set, also adds the reverse exit with this label.
    """
    src, dst = world.resolve_location(from_id), world.resolve_location(to_id)
    if src not in world.places or dst not in world.places:
        raise ValueError("Both ends of a connection must be places.")
    world.places[src].exits[_require(exit_label, "exit_label")] = dst
    if back_label:
        world.places[dst].exits[back_label.strip()] = src
    log.changes.append(f"New path: {world.places[src].name} -> {world.places[dst].name}")
    return "connected"


@tool
def move_player(world: ToolContext[World], log: ToolContext[TurnLog], place_id: str) -> str:
    """Move the player to another existing place (walking, falling, teleporting, ...).

    Args:
        place_id: Id of the destination place. Create it first with create_place if needed.
    """
    if place_id not in world.places:
        raise ValueError(f"'{place_id}' is not a place. Create it first with create_place.")
    world.player.location = place_id
    world.places[place_id].visited = True
    log.changes.append(f"You are now in: {world.places[place_id].name}")
    return f"player is now in {place_id}"


# --8<-- [start:move]
@tool
def move(
    world: ToolContext[World], log: ToolContext[TurnLog], entity_id: str, destination: str
) -> str:
    """Move a thing or being somewhere else: pick up, drop, give, put inside, follow, ...

    Args:
        entity_id: Id of the thing or being to move.
        destination: "here", "player", or the id of a place, being or container thing.
    """
    entity = world.entity(entity_id)
    if isinstance(entity, Place):
        raise ValueError("Places can't be moved. Use connect_places to change the map.")
    where = world.resolve_location(destination)
    if where == entity_id or world.is_inside(entity_id, where):
        raise ValueError("Can't put something inside itself.")
    if isinstance(entity, Being) and where != PLAYER and where not in world.places:
        raise ValueError("Beings can only be moved to a place or to 'player' (following).")
    entity.location = where
    log.changes.append(f"{entity.name} -> {world.location_name(where)}")
    return f"{entity_id} is now in {where}"
# --8<-- [end:move]


@tool
def set_property(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    entity_id: str,
    key: str,
    value: str,
) -> str:
    """Set or remove an open-ended property on a place, thing or being.

    Args:
        entity_id: Id of the entity.
        key: Property name, e.g. "on_fire", "locked", "hp", "mood".
        value: New value. Use an empty string to remove the property.
    """
    entity = world.entity(entity_id)
    key = _require(key, "key")
    if value.strip():
        entity.properties[key] = value.strip()
        log.changes.append(f"{entity.name}: {key} = {value.strip()}")
    else:
        entity.properties.pop(key, None)
        log.changes.append(f"{entity.name}: {key} removed")
    return "ok"


@tool
def rewrite_description(
    world: ToolContext[World], log: ToolContext[TurnLog], entity_id: str, description: str
) -> str:
    """Replace the description of something that has visibly changed (burned, repaired, ...).

    Args:
        entity_id: Id of the place, thing or being.
        description: The full new description.
    """
    entity = world.entity(entity_id)
    entity.description = _require(description, "description")
    log.changes.append(f"{entity.name} changed")
    return "ok"


@tool
def update_being(
    world: ToolContext[World],
    log: ToolContext[TurnLog],
    being_id: str,
    disposition: str = "",
    remember: str = "",
) -> str:
    """Change how a being feels about the player and/or give it a memory of what happened.

    Args:
        being_id: Id of the being.
        disposition: New attitude toward the player (leave empty to keep it).
        remember: A short memory to keep, e.g. "The player stole my lute." (leave empty for none)
    """
    if being_id not in world.beings:
        raise ValueError(f"'{being_id}' is not a being.")
    being = world.beings[being_id]
    if disposition.strip():
        being.disposition = disposition.strip()
        log.changes.append(f"{being.name} now feels: {being.disposition}")
    if remember.strip():
        being.memory = [*being.memory, remember.strip()][-10:]
        log.changes.append(f"{being.name} will remember that")
    return "ok"


# --8<-- [start:destroy]
@tool(description="Permanently remove a thing or being from the world (eaten, burned, slain...).")
def destroy(
    world: ToolContext[World], log: ToolContext[TurnLog], entity_id: str, how: str
) -> str:
    """Destroy something. Anything it was carrying or containing falls to where it was.

    (The short ``description=`` above is what the LLM sees; this docstring is for humans
    reading the code -- and the ``Args`` below still become parameter descriptions.)

    Args:
        entity_id: Id of the thing or being.
        how: A few words on how it was destroyed.
    """
    entity = world.entity(entity_id)
    if isinstance(entity, Place):
        raise ValueError("Places can't be destroyed; rewrite_description it into ruins instead.")
    for item in world.contents(entity_id):
        item.location = entity.location
    table: dict[str, Any] = world.things if isinstance(entity, Thing) else world.beings
    del table[entity_id]
    log.changes.append(f"Destroyed: {entity.name} ({how.strip()})")
    return "destroyed"
# --8<-- [end:destroy]


@tool
def set_player_status(
    world: ToolContext[World], log: ToolContext[TurnLog], key: str, value: str
) -> str:
    """Set or remove anything about the player: wounds, curses, forms, titles, moods...

    Anything goes, e.g. {"form": "a sentient fog"}, {"wounds": "sprained ankle"},
    {"title": "Duke of Rats"}, {"gold": "14"}.

    Args:
        key: What aspect of the player.
        value: New value. Use an empty string to remove it.
    """
    key = _require(key, "key")
    if value.strip():
        world.player.status[key] = value.strip()
        log.changes.append(f"You: {key} = {value.strip()}")
    else:
        world.player.status.pop(key, None)
        log.changes.append(f"You: {key} removed")
    return "ok"


# ---------------------------------------------------------------------------
# Chance and ending the turn
# ---------------------------------------------------------------------------


# --8<-- [start:roll_tool]
@tool
def roll(log: ToolContext[TurnLog], dice_notation: str, reason: str) -> dict[str, Any]:
    """Roll dice for anything uncertain: attacks, sneaking, persuading, luck. The engine rolls.

    You choose the dice and what counts as success (e.g. "1d20, need 12+ to climb").
    Respect the result -- a bad roll means things go badly.

    Args:
        dice_notation: Dice like "1d20", "1d20+2", "2d6", "4d6kh3", "2d20kl1".
        reason: What the roll is for and the target, e.g. "climb the wet wall, DC 13".
    """
    result = dice.roll(dice_notation, rng=log.rng)
    entry = {**result.to_dict(), "reason": reason.strip()}
    log.rolls.append(entry)
    log.changes.append(f"Rolled {result.describe()} -- {reason.strip()}")
    return entry
# --8<-- [end:roll_tool]


# --8<-- [start:narrate]
@tool(stop=True)
def narrate(log: ToolContext[TurnLog], text: str, summary: str) -> str:
    """Finish the turn: tell the player what happened. Call this LAST, after all changes are saved.

    Args:
        text: What the player experiences, in second person, 1-3 short paragraphs.
        summary: One sentence for the journal, e.g. "Bribed the toll-witch with a live eel."
    """
    log.finish(text.strip(), summary.strip())
    return "ok"
# --8<-- [end:narrate]


# --8<-- [start:tool_list]
WORLD_TOOLS = [
    search_world, inspect,
    create_place, create_thing, create_being, add_fact,
    connect_places, move_player, move, set_property, rewrite_description,
    update_being, destroy, set_player_status,
    roll, narrate,
]
# --8<-- [end:tool_list]
