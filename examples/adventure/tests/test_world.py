import pytest

from adventure.world import PLAYER, Thing, World


def test_unique_id_is_readable_and_unique(world: World):
    assert world.unique_id("Rusty Key!") == "rusty-key"
    assert world.unique_id("Pewter Mug") == "pewter-mug"
    assert world.unique_id("mug") == "mug-2"  # "mug" is taken


def test_resolve_location_aliases(world: World):
    assert world.resolve_location("here") == "tavern"
    assert world.resolve_location("player") == PLAYER
    assert world.resolve_location("docks") == "docks"
    with pytest.raises(ValueError, match="search_world"):
        world.resolve_location("narnia")


def test_place_of_walks_up_containers(world: World):
    world.things["coin"] = Thing(id="coin", name="Coin", description=".", location="satchel")
    assert world.place_of("coin") == "tavern"  # coin -> satchel -> player -> tavern
    assert world.is_inside("satchel", "coin") is True
    assert world.is_inside("coin", "satchel") is False


def test_search_finds_things_beings_and_facts(world: World):
    from adventure.world import Fact

    world.facts.append(Fact(text="The barkeep waters down the ale.", tags=["tavern"]))
    hits = world.search("barkeep ale")
    assert {"kind": "fact", "text": "The barkeep waters down the ale."} in hits
    assert world.search("mug")[0]["id"] == "mug"
    assert world.search("xx") == []


def test_store_roundtrip(world: World, store):
    path = store.save(world)
    assert path.name == "test-world.json"
    loaded = store.load("Test World")
    assert loaded == world
    assert store.names() == ["test-world"]
    with pytest.raises(FileNotFoundError):
        store.load("nope")
