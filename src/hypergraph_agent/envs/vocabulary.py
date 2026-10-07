"""Public type vocabulary shared by every RecipeQuest world.

The vocabulary is a declared symbolic inductive bias: agents see the type and
kind of every fact, never the world's hidden prerequisite table.
"""

from __future__ import annotations

RESOURCES = (
    "ore", "fuel", "sand", "wood", "clay", "fiber",
    "stone", "oil", "salt", "wax", "resin", "flint",
)
FACILITIES = ("furnace", "mould", "loom", "kiln", "press", "forge")
ITEMS = (
    "ingot", "key", "glass", "plank", "brick", "rope", "blade", "lens",
    "gear", "lamp", "chain", "bell", "dye", "wheel", "hinge", "seal",
)

KIND_RESOURCE = 0
KIND_FACILITY = 1
KIND_ITEM = 2
NUM_KINDS = 3

TYPE_NAMES: tuple[str, ...] = (
    RESOURCES + tuple(f + "_ready" for f in FACILITIES) + ITEMS
)
NUM_TYPES = len(TYPE_NAMES)
BASE_TYPES: tuple[int, ...] = tuple(range(len(RESOURCES) + len(FACILITIES)))
ITEM_TYPES: tuple[int, ...] = tuple(range(len(BASE_TYPES), NUM_TYPES))


def kind_of(type_id: int) -> int:
    if type_id < len(RESOURCES):
        return KIND_RESOURCE
    if type_id < len(BASE_TYPES):
        return KIND_FACILITY
    return KIND_ITEM


def is_base(type_id: int) -> bool:
    return kind_of(type_id) != KIND_ITEM
