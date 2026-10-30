"""The world model: everything the game knows, stored as plain Pydantic data.

The world is the single source of truth. The LLM never "remembers" anything on
its own -- whatever it invents must be written here through tools, and every
turn the prompt is rebuilt from this data.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

PLAYER = "player"
HERE = "here"


# --8<-- [start:models]
class Place(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    exits: dict[str, str] = Field(default_factory=dict)  # exit label -> place id
    properties: dict[str, str] = Field(default_factory=dict)
    visited: bool = False


class Thing(BaseModel):
    id: str
    name: str
    description: str
    location: str  # "player", a place id, a being id, or another thing id
    properties: dict[str, str] = Field(default_factory=dict)


class Being(BaseModel):
    id: str
    name: str
    description: str
    personality: str
    disposition: str  # how they currently feel about the player
    location: str  # a place id, or "player" if they follow you around
    properties: dict[str, str] = Field(default_factory=dict)
    memory: list[str] = Field(default_factory=list)


class Player(BaseModel):
    location: str
    status: dict[str, str] = Field(default_factory=dict)  # open-ended: anything goes


class Fact(BaseModel):
    text: str
    tags: list[str] = Field(default_factory=list)
    turn: int = 0


class JournalEntry(BaseModel):
    turn: int
    action: str
    summary: str
    changes: list[str] = Field(default_factory=list)


class World(BaseModel):
    name: str
    premise: str
    turn: int = 0
    story_so_far: str = ""
    chronicled_turn: int = 0  # last turn the chronicler condensed into story_so_far
    player: Player
    places: dict[str, Place] = Field(default_factory=dict)
    things: dict[str, Thing] = Field(default_factory=dict)
    beings: dict[str, Being] = Field(default_factory=dict)
    facts: list[Fact] = Field(default_factory=list)
    journal: list[JournalEntry] = Field(default_factory=list)
# --8<-- [end:models]

    # -- ids & lookup -------------------------------------------------------

    # --8<-- [start:unique_id]
    def unique_id(self, name: str) -> str:
        """Turn a name into a readable, unused id: "Rusty Key" -> "rusty-key"."""
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "thing"
        candidate, n = base, 2
        while candidate in self.places or candidate in self.things or candidate in self.beings:
            candidate, n = f"{base}-{n}", n + 1
        return candidate
    # --8<-- [end:unique_id]

    def entity(self, entity_id: str) -> Place | Thing | Being:
        for table in (self.places, self.things, self.beings):
            if entity_id in table:
                return table[entity_id]
        raise ValueError(f"Unknown id '{entity_id}'. Use search_world to find the right id.")

    def kind(self, entity_id: str) -> str:
        if entity_id in self.places:
            return "place"
        if entity_id in self.things:
            return "thing"
        if entity_id in self.beings:
            return "being"
        raise ValueError(f"Unknown id '{entity_id}'.")

    # --8<-- [start:resolve_location]
    def resolve_location(self, ref: str) -> str:
        """Accept friendly aliases so the LLM doesn't need to juggle ids."""
        ref = ref.strip()
        if ref.lower() in (HERE, "", "current", "room"):
            return self.player.location
        if ref.lower() in (PLAYER, "me", "inventory", "you"):
            return PLAYER
        self.entity(ref)  # raises a helpful error if unknown
        return ref
    # --8<-- [end:resolve_location]

    @property
    def here(self) -> Place:
        return self.places[self.player.location]

    def contents(self, location: str) -> list[Thing]:
        return [t for t in self.things.values() if t.location == location]

    def beings_at(self, location: str) -> list[Being]:
        return [b for b in self.beings.values() if b.location == location]

    def place_of(self, entity_id: str) -> str:
        """Walk up containers until we reach a place (or the player's place)."""
        seen: set[str] = set()
        current = entity_id
        while current not in self.places:
            if current == PLAYER:
                return self.player.location
            if current in seen:
                raise ValueError(f"Containment loop at '{current}'.")
            seen.add(current)
            current = self.entity(current).location  # type: ignore[union-attr]
        return current

    def is_inside(self, entity_id: str, container_id: str) -> bool:
        """True if entity_id is (transitively) inside container_id."""
        current = container_id
        while current in self.things or current in self.beings:
            if current == entity_id:
                return True
            current = self.entity(current).location  # type: ignore[union-attr]
        return current == entity_id

    def location_name(self, location: str) -> str:
        if location == PLAYER:
            return "your inventory"
        return self.entity(location).name

    # -- search -------------------------------------------------------------

    # --8<-- [start:search]
    def search(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        """Tiny keyword search over everything in the world (no embeddings needed)."""
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        if not words:
            return []

        def score(name: str, tags: list[str], text: str) -> int:
            name_l, text_l, tags_l = name.lower(), text.lower(), " ".join(tags).lower()
            return sum(3 * (w in name_l) + 2 * (w in tags_l) + (w in text_l) for w in words)

        hits: list[tuple[int, dict[str, str]]] = []
        for p in self.places.values():
            hits.append((score(p.name, p.tags, p.description),
                         {"id": p.id, "kind": "place", "name": p.name}))
        for t in self.things.values():
            hits.append((score(t.name, [], t.description),
                         {"id": t.id, "kind": "thing", "name": t.name,
                          "where": self.location_name(t.location)}))
        for b in self.beings.values():
            hits.append((score(b.name, [], f"{b.description} {b.personality}"),
                         {"id": b.id, "kind": "being", "name": b.name,
                          "where": self.location_name(b.location)}))
        for f in self.facts:
            hits.append((score("", f.tags, f.text), {"kind": "fact", "text": f.text}))

        hits = [h for h in hits if h[0] > 0]
        hits.sort(key=lambda h: h[0], reverse=True)
        return [h[1] for h in hits[:limit]]
    # --8<-- [end:search]


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


# --8<-- [start:store]
class WorldStore:
    """One JSON file per world. Open it in an editor and watch it grow!"""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "world"
        return self.directory / f"{slug}.json"

    def save(self, world: World) -> Path:
        path = self.path(world.name)
        path.write_text(world.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> World:
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(f"No saved world called '{name}'.")
        return World.model_validate_json(path.read_text(encoding="utf-8"))

    def names(self) -> list[str]:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]
# --8<-- [end:store]
