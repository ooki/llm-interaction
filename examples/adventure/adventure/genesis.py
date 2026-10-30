"""World creation: turn a premise into a validated starting world."""

from __future__ import annotations

import random

from llm_interaction import LLMInteraction
from pydantic import BaseModel, Field

from adventure.world import PLAYER, Being, Fact, Place, Player, Thing, World

INSPIRATION = [
    "tavern", "ruin", "caravan", "swamp", "monastery", "shipwreck", "border fort",
    "goblin market", "haunted mill", "dwarven mine", "wizard's tower", "festival",
    "plague village", "dragon's lair", "crossroads", "frozen pass", "sunken city",
    "thieves' guild", "elven grove", "desert oasis", "giant's bones", "bell tower",
]


# --8<-- [start:seed_models]
class PlaceSeed(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class NearbyPlaceSeed(PlaceSeed):
    exit_label: str = Field(description="Exit label from the start place, e.g. 'north'")
    back_label: str = Field(description="Exit label back to the start place, e.g. 'south'")


class ThingSeed(BaseModel):
    name: str
    description: str
    properties: dict[str, str] = Field(default_factory=dict)
    carried: bool = Field(default=False, description="True if the player starts with it")


class BeingSeed(BaseModel):
    name: str
    description: str
    personality: str
    disposition: str = "neutral"


class WorldSeed(BaseModel):
    world_name: str = Field(description="Short evocative name for this world/campaign")
    premise: str
    player_status: dict[str, str] = Field(
        default_factory=dict, description="Who the player is, e.g. {'role': 'disgraced knight'}"
    )
    start: PlaceSeed
    nearby: list[NearbyPlaceSeed] = Field(default_factory=list, max_length=3)
    things: list[ThingSeed] = Field(default_factory=list, max_length=6)
    beings: list[BeingSeed] = Field(default_factory=list, max_length=3)
    facts: list[str] = Field(default_factory=list, max_length=5)
    opening: str = Field(description="Opening narration in second person, 1-2 paragraphs")
# --8<-- [end:seed_models]


# --8<-- [start:pick_premise]
async def pick_premise(llm: LLMInteraction, rng: random.Random | None = None) -> str:
    """No premise given? Let the LLM pick one -- seeded with random words from *us*."""
    words = (rng or random.Random()).sample(INSPIRATION, 3)
    response = await llm.query_template("premise", {"words": words})
    data = response.json()  # tolerant JSON parsing (code fences, small syntax slips)
    return str(data["premise"])
# --8<-- [end:pick_premise]


# --8<-- [start:create_seed]
async def create_seed(llm: LLMInteraction, premise: str) -> WorldSeed:
    schema = WorldSeed.model_json_schema()
    response = await llm.query_template("genesis", {"premise": premise, "schema": schema})
    # parse() validates against the Pydantic model; on failure it sends the
    # validation error back to the model and asks again (max_retries times).
    return await response.parse(WorldSeed, max_retries=2)  # type: ignore[return-value]
# --8<-- [end:create_seed]


# --8<-- [start:build_world]
def build_world(seed: WorldSeed) -> World:
    """Deterministically turn a validated seed into a real World."""
    world = World(name=seed.world_name, premise=seed.premise,
                  player=Player(location="", status=dict(seed.player_status)))

    start = Place(id=world.unique_id(seed.start.name), name=seed.start.name,
                  description=seed.start.description, tags=seed.start.tags, visited=True)
    world.places[start.id] = start
    world.player.location = start.id

    for p in seed.nearby:
        place = Place(id=world.unique_id(p.name), name=p.name, description=p.description,
                      tags=p.tags, exits={p.back_label: start.id})
        world.places[place.id] = place
        start.exits[p.exit_label] = place.id

    for t in seed.things:
        thing = Thing(id=world.unique_id(t.name), name=t.name, description=t.description,
                      properties=t.properties, location=PLAYER if t.carried else start.id)
        world.things[thing.id] = thing

    for b in seed.beings:
        being = Being(id=world.unique_id(b.name), name=b.name, description=b.description,
                      personality=b.personality, disposition=b.disposition, location=start.id)
        world.beings[being.id] = being

    world.facts = [Fact(text=f, turn=0) for f in seed.facts]
    return world
# --8<-- [end:build_world]
