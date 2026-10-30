"""Slash commands: instant, free, deterministic -- no LLM involved.

Anything the player types that starts with "/" lands here. Everything else is
handed to the Dungeon Master (the LLM). Because these commands read the saved
world directly, they double as proof that the LLM really persisted what it invented.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from adventure import dice
from adventure.world import PLAYER, World

if TYPE_CHECKING:
    from adventure.game import Game


@dataclass
class CommandOutput:
    text: str
    changes: list[str] = field(default_factory=list)
    client_action: str | None = None  # e.g. "toggle_debug", handled by the browser


@dataclass
class Command:
    name: str
    usage: str
    help: str
    fn: Callable[[Game, str], CommandOutput | Awaitable[CommandOutput]]
    needs_world: bool


COMMANDS: dict[str, Command] = {}


# --8<-- [start:registry]
def command(name: str, usage: str, help: str, needs_world: bool = True):
    """Register a slash command. The function gets (game, argument_string)."""

    def register(fn):
        COMMANDS[name] = Command(name, usage, help, fn, needs_world)
        return fn

    return register


async def run_command(game: Game, text: str) -> CommandOutput:
    name, _, arg = text.lstrip("/").partition(" ")
    cmd = COMMANDS.get(name.lower())
    if cmd is None:
        return CommandOutput(f"Unknown command '/{name}'. Type /help for the list.")
    if cmd.needs_world and game.world is None:
        return CommandOutput("No world loaded yet. Start one with /new or /load <name>.")
    result = cmd.fn(game, arg.strip())
    return await result if inspect.isawaitable(result) else result
# --8<-- [end:registry]


# ---------------------------------------------------------------------------
# Renderers (also used by the side panels)
# ---------------------------------------------------------------------------


def _props(props: dict[str, str]) -> str:
    return f" ({', '.join(f'{k}: {v}' for k, v in props.items())})" if props else ""


def _thing_lines(world: World, location: str, indent: str = "  ") -> list[str]:
    lines = []
    for t in world.contents(location):
        lines.append(f"{indent}- {t.name}{_props(t.properties)}")
        lines.extend(_thing_lines(world, t.id, indent + "    "))
    return lines


# --8<-- [start:render_look]
def render_look(world: World) -> str:
    here = world.here
    lines = [f"== {here.name} ==", textwrap.fill(here.description, 78)]
    if here.properties:
        lines.append(f"Notable:{_props(here.properties)}")
    if things := _thing_lines(world, here.id):
        lines += ["", "You see:", *things]
    if beings := world.beings_at(here.id):
        lines += ["", "Here:"] + [f"  - {b.name} ({b.disposition})" for b in beings]
    if here.exits:
        lines += ["", "Exits:"] + [
            f"  - {label} -> {world.places[pid].name if pid in world.places else '???'}"
            for label, pid in here.exits.items()
        ]
    return "\n".join(lines)
# --8<-- [end:render_look]


def render_inventory(world: World) -> str:
    lines = _thing_lines(world, PLAYER)
    followers = world.beings_at(PLAYER)
    out = ["You carry:", *(lines or ["  (nothing)"])]
    if followers:
        out += ["", "Following you:"] + [f"  - {b.name} ({b.disposition})" for b in followers]
    return "\n".join(out)


def render_me(world: World) -> str:
    status = world.player.status
    lines = [f"  - {k}: {v}" for k, v in status.items()] or ["  (nothing notable -- yet)"]
    return "\n".join(["About you:", *lines])


def render_map(world: World) -> str:
    lines = []
    for p in world.places.values():
        if not p.visited:
            continue
        marker = "  <- you are here" if p.id == world.player.location else ""
        lines.append(f"* {p.name}{marker}")
        for label, pid in p.exits.items():
            target = world.places.get(pid)
            name = (target.name if target.visited else "?") if target else "???"
            lines.append(f"    {label} -> {name}")
    unseen = sum(1 for p in world.places.values() if not p.visited)
    if unseen:
        lines.append(f"\n({unseen} place(s) known but not yet visited)")
    return "\n".join(lines) or "(no map yet)"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "/help", "List commands", needs_world=False)
def _help(game: Game, arg: str) -> CommandOutput:
    width = max(len(c.usage) for c in COMMANDS.values())
    lines = [f"  {c.usage.ljust(width)}  {c.help}" for c in COMMANDS.values()]
    return CommandOutput(
        "Commands (no LLM involved):\n" + "\n".join(lines)
        + "\n\nAnything else you type goes to the Dungeon Master. Try anything."
    )


@command("look", "/look", "Describe where you are (straight from the save file)")
def _look(game: Game, arg: str) -> CommandOutput:
    return CommandOutput(render_look(game.world))


@command("inv", "/inv", "What you carry")
def _inv(game: Game, arg: str) -> CommandOutput:
    return CommandOutput(render_inventory(game.world))


@command("me", "/me", "Your status: wounds, curses, forms, titles...")
def _me(game: Game, arg: str) -> CommandOutput:
    return CommandOutput(render_me(game.world))


@command("map", "/map", "Places you have visited and their exits")
def _map(game: Game, arg: str) -> CommandOutput:
    return CommandOutput(render_map(game.world))


@command("journal", "/journal [n]", "The story so far (last n turns, default 15)")
def _journal(game: Game, arg: str) -> CommandOutput:
    n = int(arg) if arg.isdigit() else 15
    world = game.world
    lines = [f"Story so far: {world.story_so_far}", ""] if world.story_so_far else []
    lines += [f"{j.turn:>3}. {j.action}\n     -> {j.summary}" for j in world.journal[-n:]]
    return CommandOutput("\n".join(lines) or "(nothing has happened yet)")


@command("facts", "/facts", "Everything the world knows to be true")
def _facts(game: Game, arg: str) -> CommandOutput:
    facts = game.world.facts
    return CommandOutput("\n".join(f"- {f.text}" for f in facts) or "(no facts yet)")


# --8<-- [start:roll_command]
@command("roll", "/roll <dice>", "Roll dice yourself, e.g. /roll 2d6+1", needs_world=False)
def _roll(game: Game, arg: str) -> CommandOutput:
    try:
        result = dice.roll(arg or "1d20", rng=game.rng)
    except ValueError as e:
        return CommandOutput(str(e))
    return CommandOutput(f"You roll {result.describe()}")
# --8<-- [end:roll_command]


@command("undo", "/undo", "Undo the last Dungeon Master turn")
def _undo(game: Game, arg: str) -> CommandOutput:
    if not game.undo():
        return CommandOutput("Nothing to undo.")
    return CommandOutput("Time folds back on itself...\n\n" + render_look(game.world))


@command("new", "/new [premise]", "Start a new world (random premise if empty)", needs_world=False)
async def _new(game: Game, arg: str) -> CommandOutput:
    opening = await game.new_world(arg)
    return CommandOutput(f"{opening}\n\n{render_look(game.world)}",
                         changes=[f"New world: {game.world.name}"])


@command("worlds", "/worlds", "List saved worlds", needs_world=False)
def _worlds(game: Game, arg: str) -> CommandOutput:
    names = game.store.names()
    current = game.store.path(game.world.name).stem if game.world else None
    lines = [f"  - {n}{'  (current)' if n == current else ''}" for n in names]
    return CommandOutput("Saved worlds:\n" + "\n".join(lines) if names else "No saved worlds yet.")


@command("load", "/load <name>", "Load a saved world", needs_world=False)
def _load(game: Game, arg: str) -> CommandOutput:
    if not arg:
        return CommandOutput("Usage: /load <name>  (see /worlds)")
    try:
        game.load(arg)
    except FileNotFoundError as e:
        return CommandOutput(str(e))
    return CommandOutput(f"Loaded '{game.world.name}'.\n\n{render_look(game.world)}")


@command("debug", "/debug", "Toggle the LLM debug drawer", needs_world=False)
def _debug(game: Game, arg: str) -> CommandOutput:
    return CommandOutput("Debug drawer toggled.", client_action="toggle_debug")
