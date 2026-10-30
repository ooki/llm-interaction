"""One free-text turn: build context -> agent loop with world tools -> journal."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from llm_interaction import LLMInteraction

from adventure.context import build_turn_context
from adventure.tools import WORLD_TOOLS, TurnLog
from adventure.world import JournalEntry, World

MAX_TOOL_CALLS = 30

FALLBACK_NARRATION = (
    "The world shimmers, as if reality itself lost its train of thought. "
    "(The Dungeon Master ran out of moves this turn -- whatever was saved still stands.)"
)


@dataclass
class TurnResult:
    narration: str
    changes: list[str]
    rolls: list[dict[str, Any]]
    stop_reason: str
    tool_calls: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    prompt: dict[str, str] = field(default_factory=dict)


# --8<-- [start:run_turn]
async def run_turn(
    llm: LLMInteraction, world: World, action: str, rng: random.Random | None = None
) -> TurnResult:
    log = TurnLog(rng=rng or random.Random())
    trace: list[dict[str, Any]] = []

    # --8<-- [start:on_tool_call]
    def on_tool_call(index: int, name: str, args: dict, result: Any) -> None:
        trace.append({"n": index + 1, "tool": name, "args": args, "result": result})
    # --8<-- [end:on_tool_call]

    variables = build_turn_context(world, action)
    result = await llm.agent_loop_template(
        "turn",
        variables,
        tools=WORLD_TOOLS,
        context=[world, log],  # matched by type to ToolContext[World] / ToolContext[TurnLog]
        max_tool_calls=MAX_TOOL_CALLS,
        on_tool_call=on_tool_call,
    )

    # stop_reason is "stop_tool" (narrate was called), "content" (the model just
    # answered in plain text) or "max_calls" (it ran out of tool calls).
    narration = log.narration or (result.final_content or "").strip() or FALLBACK_NARRATION

    world.turn += 1
    world.journal.append(JournalEntry(
        turn=world.turn,
        action=action,
        summary=log.summary or narration.split("\n")[0][:160],
        changes=log.changes,
    ))
    # --8<-- [end:run_turn]

    return TurnResult(
        narration=narration,
        changes=log.changes,
        rolls=log.rolls,
        stop_reason=result.stop_reason,
        tool_calls=result.tool_call_count,
        trace=trace,
        prompt={
            # llm.render() shows exactly what agent_loop_template sent.
            "system": llm.render("turn_system.jinja", variables),
            "user": llm.render("turn_user.jinja", variables),
        },
    )
