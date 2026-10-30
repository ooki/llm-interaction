"""The Game: routes input, owns the current world, saves, and supports /undo."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from llm_interaction import LLMInteraction

from adventure import chronicler
from adventure.commands import render_inventory, render_look, render_map, render_me, run_command
from adventure.genesis import build_world, create_seed, pick_premise
from adventure.turn import run_turn
from adventure.world import PLAYER, JournalEntry, World, WorldStore

logger = logging.getLogger(__name__)

UNDO_LIMIT = 20


def _jsonable(obj: Any) -> Any:
    """Make SDK objects (pydantic models etc.) safe to send to the browser."""
    def default(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump(exclude_none=True)
        return vars(o) if hasattr(o, "__dict__") else str(o)

    return json.loads(json.dumps(obj, default=default))


class Game:
    def __init__(self, llm: LLMInteraction, store: WorldStore, rng: random.Random | None = None):
        self.llm = llm
        self.store = store
        self.rng = rng or random.Random()
        self.world: World | None = None
        self.last_debug: dict[str, Any] = {}
        self._undo: list[str] = []
        self._lock = asyncio.Lock()

    # --8<-- [start:handle]
    async def handle(self, text: str) -> dict[str, Any]:
        """Entry point for everything the player types."""
        async with self._lock:  # one action at a time
            text = text.strip()
            reply: dict[str, Any] = {"kind": "command", "text": "", "changes": []}
            try:
                if text.startswith("/"):
                    out = await run_command(self, text)
                    reply.update(text=out.text, changes=out.changes,
                                 client_action=out.client_action)
                elif not text:
                    reply["text"] = "Do something! Or type /help."
                elif self.world is None:
                    reply["text"] = "No world yet. Type /new (or /new <your premise>) to begin."
                else:
                    reply.update(await self._turn(text))
            except Exception as e:  # network/LLM trouble: tell the player, keep the game alive
                logger.exception("Error handling %r", text)
                reply = {"kind": "error", "text": f"The weave of magic falters: {e}", "changes": []}
            reply["state"] = self.state()
            reply["debug"] = self.last_debug
            return reply
    # --8<-- [end:handle]

    # --8<-- [start:turn]
    async def _turn(self, action: str) -> dict[str, Any]:
        assert self.world is not None
        self._checkpoint()
        try:
            result = await run_turn(self.llm, self.world, action, rng=self.rng)
        except Exception:
            self.undo()  # roll back anything the tools half-changed
            raise

        debug: dict[str, Any] = {
            "stop_reason": result.stop_reason,
            "tool_calls": result.tool_calls,
            "trace": result.trace,
            "prompt": result.prompt,
            "last_call": self.llm.last_call,
        }
        if chronicler.is_due(self.world):
            try:
                debug["chronicle"] = await chronicler.chronicle(self.llm, self.world)
            except Exception as e:  # the chronicle is a bonus; never lose a turn over it
                logger.warning("Chronicler failed: %s", e)
        self.save()
        self.last_debug = _jsonable(debug)
        return {"kind": "turn", "text": result.narration, "changes": result.changes}
    # --8<-- [end:turn]

    # -- world management ---------------------------------------------------

    async def new_world(self, premise: str = "") -> str:
        premise = premise.strip() or await pick_premise(self.llm, self.rng)
        seed = await create_seed(self.llm, premise)
        world = build_world(seed)
        base, n = world.name, 2
        while self.store.path(world.name).exists():
            world.name, n = f"{base} {n}", n + 1
        world.journal.append(JournalEntry(turn=0, action="(the adventure begins)",
                                          summary=seed.opening.split("\n")[0][:200]))
        self.world = world
        self._undo.clear()
        self.save()
        self.last_debug = _jsonable({"genesis": seed, "last_call": self.llm.last_call})
        return seed.opening

    def load(self, name: str) -> None:
        self.world = self.store.load(name)
        self._undo.clear()

    def load_latest(self) -> None:
        if names := self.store.names():
            self.load(names[0])

    def save(self) -> None:
        if self.world is not None:
            self.store.save(self.world)

    # --8<-- [start:undo]
    def _checkpoint(self) -> None:
        assert self.world is not None
        self._undo = [*self._undo, self.world.model_dump_json()][-UNDO_LIMIT:]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.world = World.model_validate_json(self._undo.pop())
        self.save()
        return True
    # --8<-- [end:undo]

    # -- side panels ----------------------------------------------------------

    def state(self) -> dict[str, Any]:
        world = self.world
        if world is None:
            return {"world": None, "worlds": self.store.names()}
        return {
            "world": world.name,
            "premise": world.premise,
            "turn": world.turn,
            "place": world.here.name,
            "look": render_look(world),
            "inventory": render_inventory(world),
            "me": render_me(world),
            "map": render_map(world),
            "counts": {
                "places": len(world.places),
                "things": len(world.things),
                "beings": len(world.beings),
                "facts": len(world.facts),
                "carried": len(world.contents(PLAYER)),
            },
            "can_undo": bool(self._undo),
        }
