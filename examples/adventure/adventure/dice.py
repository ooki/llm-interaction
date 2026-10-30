"""Dice notation parser and roller.

LLMs are bad at being random -- ask one to "roll a d20" a hundred times and the
results will cluster suspiciously. So the *engine* rolls, and the LLM only
decides *what* to roll and narrates the outcome.

Supported notation: ``d20``, ``1d20+3``, ``2d6-1``, ``4d6kh3`` (keep highest 3),
``2d20kl1`` (keep lowest, i.e. disadvantage), and sums like ``1d8+1d6+2``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

MAX_DICE = 100
MAX_SIDES = 1000

_TERM = re.compile(r"([+-]?)\s*(?:(\d*)d(\d+)(?:(kh|kl)(\d+))?|(\d+))", re.IGNORECASE)


@dataclass
class DieGroup:
    notation: str
    rolls: list[int]
    kept: list[int]
    sign: int = 1

    @property
    def total(self) -> int:
        return self.sign * sum(self.kept)


@dataclass
class RollResult:
    notation: str
    groups: list[DieGroup] = field(default_factory=list)
    modifier: int = 0

    @property
    def total(self) -> int:
        return sum(g.total for g in self.groups) + self.modifier

    def describe(self) -> str:
        parts = []
        for g in self.groups:
            shown = ", ".join(str(r) if r in g.kept else f"~{r}~" for r in g.rolls)
            parts.append(f"{'-' if g.sign < 0 else ''}{g.notation} [{shown}]")
        if self.modifier:
            parts.append(f"{self.modifier:+d}")
        return f"{' '.join(parts)} = {self.total}"

    def to_dict(self) -> dict:
        return {
            "dice": self.notation,
            "rolls": [r for g in self.groups for r in g.rolls],
            "kept": [r for g in self.groups for r in g.kept],
            "modifier": self.modifier,
            "total": self.total,
        }


# --8<-- [start:roll]
def roll(notation: str, rng: random.Random | None = None) -> RollResult:
    """Parse dice notation and roll it. Raises ValueError on nonsense input."""
    rng = rng or random.Random()
    text = notation.replace(" ", "").lower()
    if not text:
        raise ValueError("Empty dice notation. Try something like '1d20+2'.")

    result = RollResult(notation=text)
    pos = 0
    for match in _TERM.finditer(text):
        if match.start() != pos or (pos > 0 and not match.group(1)):
            raise ValueError(f"Can't read dice notation '{notation}'. Try '1d20+2' or '2d6'.")
        pos = match.end()
        sign = -1 if match.group(1) == "-" else 1
        count_s, sides_s, keep_mode, keep_s, flat = match.groups()[1:]

        if flat is not None:
            result.modifier += sign * int(flat)
            continue

        count, sides = int(count_s or 1), int(sides_s)
        if not (1 <= count <= MAX_DICE and 2 <= sides <= MAX_SIDES):
            raise ValueError(f"Dice out of range in '{notation}' (max {MAX_DICE}d{MAX_SIDES}).")
        rolls = [rng.randint(1, sides) for _ in range(count)]
        kept = rolls
        if keep_mode:
            keep = int(keep_s)
            if not 1 <= keep <= count:
                raise ValueError(f"Can't keep {keep} of {count} dice in '{notation}'.")
            kept = sorted(rolls, reverse=keep_mode == "kh")[:keep]
        label = f"{count}d{sides}{keep_mode or ''}{keep_s or ''}"
        result.groups.append(DieGroup(notation=label, rolls=rolls, kept=list(kept), sign=sign))

    if pos != len(text):
        raise ValueError(f"Can't read dice notation '{notation}'. Try '1d20+2' or '2d6'.")
    return result
# --8<-- [end:roll]
