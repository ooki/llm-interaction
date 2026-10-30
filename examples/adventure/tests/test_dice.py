import random

import pytest

from adventure.dice import roll


@pytest.mark.parametrize(("notation", "lo", "hi"), [
    ("d20", 1, 20), ("1d20", 1, 20), ("1d20+3", 4, 23), ("2d6-1", 1, 11),
    ("4d6kh3", 3, 18), ("2d20kl1", 1, 20), ("1d8+1d6+2", 4, 16), (" 3 d 6 ", 3, 18),
    ("1d20-1d4", -3, 19),
])
def test_valid_notation_stays_in_range(notation, lo, hi):
    rng = random.Random(0)
    totals = {roll(notation, rng).total for _ in range(500)}
    assert min(totals) >= lo and max(totals) <= hi


def test_keep_highest_keeps_the_best():
    r = roll("4d6kh3", random.Random(1))
    group = r.groups[0]
    assert len(group.rolls) == 4 and len(group.kept) == 3
    assert sorted(group.kept) == sorted(group.rolls)[1:]


def test_modifier_and_total():
    r = roll("1d20+5", random.Random(3))
    assert r.modifier == 5
    assert r.total == r.groups[0].rolls[0] + 5
    assert r.describe().endswith(f"= {r.total}")


@pytest.mark.parametrize("bad", ["", "banana", "1d1", "0d6", "1d20++2", "101d6", "2d6kh3",
                                 "1d20*2"])
def test_invalid_notation_raises(bad):
    with pytest.raises(ValueError):
        roll(bad)


def test_rolls_are_actually_random():
    totals = {roll("1d20", random.Random(seed)).total for seed in range(200)}
    assert totals == set(range(1, 21))
