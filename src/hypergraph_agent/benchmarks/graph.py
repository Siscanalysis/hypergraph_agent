"""Neutral recipe graphs for external benchmarks.

A ``RecipeGraph`` is the common format every loader in this package produces:

* ``base``: elements available from the start (gatherable, no recipe needed);
* ``recipes``: ``effect <- AND(requires)``; several recipes for one effect are
  alternatives (OR across recipes, AND within one);
* ``unlocks`` (optional): an element granted, or made available, when a
  condition over other elements holds: ``all`` (every source held),
  ``k_of_n`` (at least ``k`` of the sources held) or ``progress`` (at least
  ``k`` elements held in total);
* ``kinds`` (optional): a label per element (``resource``, ``facility``,
  ``item``, ``unlockable``, ``secret``); the RecipeQuest converter maps base
  elements labelled ``facility`` to facility types;
* ``notes``: what the loader changed or dropped, in plain words.

Facts are Boolean and persistent, as in RecipeQuest: an element is held or
not. Quantities and consumption, where a source has them, are kept only as
metadata (``Recipe.amounts``).

Levels (``RecipeGraph.layers``) are breadth-first layers: base elements are at
level 0 and an element is at level L + 1 when, for the first time, one of its
recipes (or one of its unlocks) is satisfied by elements of levels at most L.
This is the minimal derivation depth (the max heuristic of planning).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property

UNLOCK_KINDS = ("all", "k_of_n", "progress")


@dataclass(frozen=True)
class Recipe:
    effect: str
    requires: tuple[str, ...]  # AND; sorted, unique
    label: str = ""  # source action or rule name
    amounts: tuple[tuple[str, int], ...] = ()  # quantities in the source (metadata only)

    def to_dict(self) -> dict:
        d = {"effect": self.effect, "requires": list(self.requires)}
        if self.label:
            d["label"] = self.label
        if self.amounts:
            d["amounts"] = {k: v for k, v in self.amounts}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Recipe":
        return Recipe(d["effect"], tuple(d["requires"]), d.get("label", ""),
                      tuple(sorted((k, int(v)) for k, v in d.get("amounts", {}).items())))


@dataclass(frozen=True)
class Unlock:
    target: str
    kind: str  # all | k_of_n | progress
    sources: tuple[str, ...] = ()
    k: int = 0
    label: str = ""

    def satisfied(self, held: set) -> bool:
        if self.kind == "all":
            return all(s in held for s in self.sources)
        if self.kind == "k_of_n":
            return sum(s in held for s in self.sources) >= self.k
        return len(held) >= self.k

    def to_dict(self) -> dict:
        d = {"target": self.target, "kind": self.kind, "sources": list(self.sources), "k": self.k}
        if self.label:
            d["label"] = self.label
        return d

    @staticmethod
    def from_dict(d: dict) -> "Unlock":
        return Unlock(d["target"], d["kind"], tuple(d.get("sources", ())), int(d.get("k", 0)),
                      d.get("label", ""))


@dataclass(frozen=True)
class RecipeGraph:
    name: str
    elements: tuple[str, ...]
    base: frozenset
    recipes: tuple[Recipe, ...]
    unlocks: tuple[Unlock, ...] = ()
    kinds: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    source: str = ""  # benchmark folder name under benchmarks/

    # ------------------------------------------------------------ structure
    @cached_property
    def _by_effect(self) -> dict:
        out: dict = {}
        for r in self.recipes:
            out.setdefault(r.effect, []).append(r)
        return out

    def recipes_for(self, element: str) -> list[Recipe]:
        return list(self._by_effect.get(element, ()))

    def kind_of(self, element: str) -> str:
        return dict(self.kinds).get(element, "base" if element in self.base else "item")

    @property
    def derived(self) -> list[str]:
        """Elements with at least one recipe that are not base."""
        return [e for e in self.elements if e not in self.base and e in self._by_effect]

    @property
    def unlock_targets(self) -> list[str]:
        return sorted({u.target for u in self.unlocks})

    @cached_property
    def _layers(self) -> dict:
        return {}

    @cached_property
    def memo(self) -> dict:
        """Per-instance cache for derived structures (not part of equality)."""
        return {}

    def layers(self, use_unlocks: bool = True) -> dict[str, int]:
        """Breadth-first level of every reachable element (base = 0)."""
        if use_unlocks not in self._layers:
            self._layers[use_unlocks] = self._compute_layers(use_unlocks)
        return dict(self._layers[use_unlocks])

    def _compute_layers(self, use_unlocks: bool) -> dict[str, int]:
        level = {b: 0 for b in self.base}
        held = set(level)
        depth = 0
        while True:
            new = set()
            for r in self.recipes:
                if r.effect not in held and all(x in held for x in r.requires):
                    new.add(r.effect)
            if use_unlocks:
                for u in self.unlocks:
                    if u.target not in held and u.satisfied(held):
                        new.add(u.target)
            if not new:
                return level
            depth += 1
            for e in new:
                level[e] = depth
            held |= new

    def validate(self) -> None:
        known = set(self.elements)
        if len(known) != len(self.elements):
            raise ValueError(f"{self.name}: duplicate element names")
        if not self.base <= known:
            raise ValueError(f"{self.name}: base elements missing from elements")
        for r in self.recipes:
            if r.effect not in known or not set(r.requires) <= known:
                raise ValueError(f"{self.name}: recipe {r} names an unknown element")
            if not r.requires:
                raise ValueError(f"{self.name}: recipe for {r.effect} has no requirement")
            if r.effect in r.requires:
                raise ValueError(f"{self.name}: recipe for {r.effect} requires its own effect")
            if tuple(sorted(set(r.requires))) != r.requires:
                raise ValueError(f"{self.name}: requirements of {r.effect} are not sorted and unique")
        for u in self.unlocks:
            if u.kind not in UNLOCK_KINDS:
                raise ValueError(f"{self.name}: unknown unlock kind {u.kind!r}")
            if u.target not in known or not set(u.sources) <= known:
                raise ValueError(f"{self.name}: unlock {u} names an unknown element")

    # ---------------------------------------------------------- serializing
    def to_dict(self) -> dict:
        return {
            "format": "recipe_graph/1",
            "name": self.name,
            "source": self.source,
            "elements": list(self.elements),
            "base": sorted(self.base),
            "recipes": [r.to_dict() for r in self.recipes],
            "unlocks": [u.to_dict() for u in self.unlocks],
            "kinds": {k: v for k, v in self.kinds},
            "notes": list(self.notes),
        }

    @staticmethod
    def from_dict(d: dict) -> "RecipeGraph":
        if d.get("format") != "recipe_graph/1":
            raise ValueError("not a recipe_graph/1 document")
        g = RecipeGraph(d["name"], tuple(d["elements"]), frozenset(d["base"]),
                        tuple(Recipe.from_dict(r) for r in d["recipes"]),
                        tuple(Unlock.from_dict(u) for u in d.get("unlocks", ())),
                        tuple(sorted(d.get("kinds", {}).items())), tuple(d.get("notes", ())),
                        d.get("source", ""))
        g.validate()
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=False, ensure_ascii=False)


def make_graph(name: str, base, recipes, unlocks=(), kinds=None, notes=(), source: str = "",
               elements=None) -> RecipeGraph:
    """Build a validated graph: requirements sorted and unique, duplicate
    recipes merged, recipes that require their own effect or produce a base
    element dropped (counted in ``notes``), deterministic order."""
    base = frozenset(base)
    notes = list(notes)
    seen: dict = {}
    self_loops = for_base = duplicates = 0
    for r in recipes:
        req = tuple(sorted(set(r.requires)))
        if r.effect in req:
            self_loops += 1
            continue
        if r.effect in base:
            for_base += 1
            continue
        key = (r.effect, req)
        if key in seen:
            duplicates += 1
            continue
        seen[key] = Recipe(r.effect, req, r.label, tuple(sorted(r.amounts)))
    if self_loops:
        notes.append(f"dropped {self_loops} recipe(s) that require their own product")
    if for_base:
        notes.append(f"dropped {for_base} recipe(s) that produce a base element (base elements are given)")
    if duplicates:
        notes.append(f"merged {duplicates} duplicate recipe(s) (same product and requirements)")
    rec = sorted(seen.values(), key=lambda r: (r.effect, r.requires, r.label))
    names = set(base) | {r.effect for r in rec} | {x for r in rec for x in r.requires}
    names |= {u.target for u in unlocks} | {s for u in unlocks for s in u.sources}
    if elements is not None:
        names |= set(elements)
    kinds = tuple(sorted((kinds or {}).items()))
    g = RecipeGraph(name, tuple(sorted(names)), base, tuple(rec),
                    tuple(sorted(unlocks, key=lambda u: (u.target, u.kind, u.sources))),
                    kinds, tuple(notes), source)
    g.validate()
    return g


def arity_histogram(graph: RecipeGraph) -> dict[int, int]:
    return dict(sorted(Counter(len(r.requires) for r in graph.recipes).items()))
