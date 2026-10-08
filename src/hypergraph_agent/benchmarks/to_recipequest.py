"""RecipeQuest worlds from a recipe graph (evaluator side: a world holds hidden truth).

What the conversion keeps and what it changes
---------------------------------------------
A RecipeQuest recipe is ``effect <- item inputs (public) + true base subset
(hidden) of a public pool of base facts``. A converted world is the closure of
one or more goal elements of the graph:

* Items and base facts. Base elements of the graph are base facts (gathered or
  activated in one step). Derived elements are items. Elements granted by an
  unlock are base facts too (``unlocks="as_base"``, the default): RecipeQuest
  has no counting or threshold unlocks, so the condition is dropped; with
  ``unlocks="drop"`` they are unavailable and goals that need them fail.
* Requirements. For every recipe of an item, its requirements that are items
  become the public item inputs and its requirements that are base facts
  become the hidden true base subset. Both are defined by the data.
* Recipes without a base requirement. RecipeQuest needs a nonempty hidden
  subset (the walkers' hypothesis class is nonempty subsets of the pool), so a
  recipe whose requirements are all items gets its shallowest item
  requirement (lowest level, then name) promoted to a base fact: that element
  becomes gatherable in this world and its own recipes are no longer used.
  Promotions are listed in ``ConvertedWorld.promoted``; goals are never
  promoted.
* Alternatives. At most ``max_alternatives`` recipes per item are kept, the
  shallowest first (level of the recipe = 1 + the largest level among its
  requirements, then the sum of levels, then names). A recipe that would
  close a cycle (an element needed, through the kept recipes, to make itself)
  is dropped, so the structure is acyclic and the reference solver stays
  exact. Distinct alternatives that differ only in their hidden part need
  distinct pools; if no distinct public signature is found the alternative is
  dropped.
* Levels. An item's level is its breadth-first level in the converted world
  (base facts at 0), so a level counts crafting steps above the base facts
  of that world, not of the source graph.
* Pools. A recipe's pool is its true base subset plus decoys drawn (seeded)
  from the other base facts of the world; when the world has fewer base facts
  than ``pool_size``, the remaining decoys are vocabulary base types with no
  meaning in the data (``decoy:<type>``). Recipes with more than ``max_true``
  base requirements are dropped if the item keeps another recipe; otherwise
  the conversion fails.
* Vocabulary. RecipeQuest has 16 item types and 18 base types (12 resources,
  6 facilities). Items and base facts are assigned to types by a seeded
  permutation; base elements labelled ``facility`` or ``station`` in the graph
  get facility types while they last. A closure with more than 16 items or 18
  base facts cannot be converted (``ConversionError``).

Hidden truth lives only in ``World.recipes[*].true_base``, exactly as in
generated worlds, so ``envs.public_schema.public_view`` hides it in the
``unknown_prerequisites`` profile. The element names are kept in
``ConvertedWorld.names`` (evaluator side, for reports); agents see type ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..envs.generator import (
    PROFILES, Recipe as RQRecipe, TaskConfig, World, WorldConfig, derive_seed, make_task, stable_hash,
)
from ..envs.vocabulary import BASE_TYPES, ITEM_TYPES, KIND_FACILITY, TYPE_NAMES, kind_of
from .graph import RecipeGraph

FACILITY_KINDS = ("facility", "station")


class ConversionError(ValueError):
    """The requested closure does not fit RecipeQuest's semantics or vocabulary."""


@dataclass(frozen=True)
class ConvertedWorld:
    world: World
    graph: str
    goals: tuple[str, ...]
    names: tuple[tuple[int, str], ...]  # vocabulary type id -> element name (evaluator side)
    promoted: tuple[str, ...]  # derived elements turned into base facts
    dropped: tuple[str, ...]  # recipes left out, with the reason
    decoy_types: tuple[int, ...]  # pool members with no meaning in the data
    params: tuple[tuple[str, object], ...]
    closure_items: int = 0  # derived elements in the goals' closure before any promotion

    @property
    def kept_share(self) -> float:
        """Share of the goals' original closure that remains items (the rest
        was promoted to base facts or became unnecessary)."""
        return len(self.world.levels) / self.closure_items if self.closure_items else 1.0

    def name_of(self, type_id: int) -> str:
        return dict(self.names).get(type_id, f"decoy:{TYPE_NAMES[type_id]}")

    def type_of(self, name: str) -> int:
        return {n: t for t, n in self.names}[name]

    @property
    def items(self) -> list[str]:
        return [self.name_of(t) for t, _ in self.world.levels]

    def task(self, goal: str, seed: int = 0, cfg: TaskConfig | None = None, namespace: str = "benchmark"):
        """A RecipeQuest task for ``goal`` (default config: hidden prerequisites,
        no distractor base facts, as in the study R families)."""
        cfg = cfg or TaskConfig(profile="unknown_prerequisites", n_distractor_base=0)
        return make_task(self.world, seed, cfg, namespace, goal=self.type_of(goal))


def hypotheses(pool_size: int, max_true: int) -> int:
    return sum(math.comb(pool_size, k) for k in range(1, min(max_true, pool_size) + 1))


def _prepare(g: RecipeGraph, unlocks: str, max_alternatives):
    key = ("recipequest_prepare", unlocks, max_alternatives)
    if key not in g.memo:
        g.memo[key] = _prepare_uncached(g, unlocks, max_alternatives)
    return g.memo[key]


def _prepare_uncached(g: RecipeGraph, unlocks: str, max_alternatives):
    """Levels, given elements and the kept recipes per derived element.

    Recipes are ranked shallowest first. A recipe whose requirements all have
    a lower level than its effect is always kept (these recipes alone form an
    acyclic structure); any other recipe is kept only if it does not close a
    cycle through the recipes kept so far (an element would then be needed to
    make itself). At most ``max_alternatives`` recipes per element."""
    level = g.layers(use_unlocks=unlocks == "as_base")
    given = set(g.base) | (set(g.unlock_targets) if unlocks == "as_base" else set())
    rank = {}
    for e in g.derived:
        if e in given or e not in level:
            continue
        ok = [r for r in g.recipes_for(e) if all(x in level for x in r.requires)]
        ok.sort(key=lambda r: (1 + max(level[x] for x in r.requires), sum(level[x] for x in r.requires),
                               r.requires))
        rank[e] = ok
    chosen = {e: [r for r in rs if all(level[x] < level[e] for x in r.requires)] for e, rs in rank.items()}
    dropped = []

    def needs(src: str, target: str) -> bool:  # does making ``src`` need ``target``?
        seen, stack = set(), [src]
        while stack:
            x = stack.pop()
            if x == target:
                return True
            if x in seen or x in given:
                continue
            seen.add(x)
            stack.extend(y for r in chosen.get(x, ()) for y in r.requires)
        return False

    for e in sorted(rank, key=lambda x: (level[x], x)):
        for r in rank[e]:
            if r in chosen[e]:
                continue
            if max_alternatives is not None and len(chosen[e]) >= max_alternatives:
                break
            if any(needs(x, e) for x in r.requires):
                dropped.append(f"{e} <- {'+'.join(r.requires)}: would need {e} to make {e}")
            else:
                chosen[e].append(r)
        chosen[e].sort(key=rank[e].index)
        if max_alternatives is not None:
            for r in chosen[e][max_alternatives:]:
                dropped.append(f"{e} <- {'+'.join(r.requires)}: beyond max_alternatives")
            chosen[e] = chosen[e][:max_alternatives]
    return level, frozenset(given), {e: tuple(rs) for e, rs in chosen.items()}, tuple(dropped)


def _levels(items: set, bases: set, chosen: dict) -> dict:
    """Breadth-first levels of the converted structure (base facts at 0)."""
    level = {b: 0 for b in bases}
    depth = 0
    while True:
        new = {e for e in items if e not in level
               and any(all(x in level for x in r.requires) for r in chosen[e])}
        if not new:
            return level
        depth += 1
        level.update({e: depth for e in new})


def _closure(goals, chosen: dict, base: set) -> tuple[set, set]:
    items, bases, stack = set(), set(), list(goals)
    while stack:
        e = stack.pop()
        if e in base:
            bases.add(e)
            continue
        if e in items:
            continue
        items.add(e)
        for r in chosen[e]:
            stack.extend(r.requires)
    return items, bases


def recipequest_world(graph: RecipeGraph, goals, *, pool_size: int = 6, max_true: int = 3,
                      max_alternatives: int | None = 2, seed: int = 0, unlocks: str = "as_base",
                      max_items: int = len(ITEM_TYPES), max_base: int = len(BASE_TYPES)) -> ConvertedWorld:
    """Convert the closure of ``goals`` (element names) into a RecipeQuest world."""
    goals = tuple(sorted(set([goals] if isinstance(goals, str) else goals)))
    if unlocks not in ("as_base", "drop"):
        raise ValueError("unlocks must be 'as_base' or 'drop'")
    if not 1 <= max_true <= pool_size <= len(BASE_TYPES):
        raise ValueError("need 1 <= max_true <= pool_size <= 18")
    level, given, kept, dropped_all = _prepare(graph, unlocks, max_alternatives)
    for gl in goals:
        if gl not in level:
            raise ConversionError(f"{gl!r} is not reachable in {graph.name}")
        if gl in given:
            raise ConversionError(f"{gl!r} is a base element and cannot be a goal")
    chosen = {e: list(rs) for e, rs in kept.items()}

    # promotion loop: every kept recipe needs at least one base requirement
    base, promoted = set(given), []
    original = len(_closure(goals, chosen, base)[0])
    while True:
        items, _ = _closure(goals, chosen, base)
        todo = None
        for e in sorted(items, key=lambda x: (level[x], x)):
            for r in chosen[e]:
                if not any(x in base for x in r.requires):
                    cands = sorted((x for x in r.requires if x not in goals), key=lambda x: (level[x], x))
                    if not cands:
                        raise ConversionError(f"{e} <- {'+'.join(r.requires)} needs only goals")
                    todo = cands[0]
                    break
            if todo:
                break
        if todo is None:
            break
        base.add(todo)
        promoted.append(todo)

    # recipes with too many base requirements
    dropped = []
    for e in sorted(items):
        keep = [r for r in chosen[e] if sum(x in base for x in r.requires) <= max_true]
        for r in chosen[e]:
            if r not in keep:
                dropped.append(f"{e} <- {'+'.join(r.requires)}: more than {max_true} base requirements")
        if not keep:
            raise ConversionError(f"every recipe of {e} has more than {max_true} base requirements")
        chosen[e] = keep
    items, bases = _closure(goals, chosen, base)
    promoted = [p for p in promoted if p in bases]
    if len(items) > max_items:
        raise ConversionError(f"the closure has {len(items)} items (at most {max_items})")
    if len(bases) > max_base:
        raise ConversionError(f"the closure has {len(bases)} base facts (at most {max_base})")

    # recipes of the closure left out earlier, then levels of the converted world
    dropped = [d for d in dropped_all if d.split(" <- ")[0] in items] + dropped
    conv = _levels(items, bases, chosen)

    # vocabulary assignment
    rng = np.random.default_rng(derive_seed("benchmark_world", graph.name, goals, seed, pool_size,
                                            max_true, max_alternatives, unlocks))
    item_order = sorted(items, key=lambda x: (conv[x], x))
    item_types = [int(t) for t in rng.permutation(ITEM_TYPES)[:len(item_order)]]
    fac_types = [int(t) for t in rng.permutation([b for b in BASE_TYPES if kind_of(b) == KIND_FACILITY])]
    res_types = [int(t) for t in rng.permutation([b for b in BASE_TYPES if kind_of(b) != KIND_FACILITY])]
    type_of = dict(zip(item_order, item_types))
    for b in sorted(bases):
        wants_fac = graph.kind_of(b) in FACILITY_KINDS
        first, second = (fac_types, res_types) if wants_fac else (res_types, fac_types)
        type_of[b] = (first or second).pop(0)
    spare = sorted(set(BASE_TYPES) - {type_of[b] for b in bases})
    base_types = sorted(type_of[b] for b in bases)

    # recipes with pools
    rq, decoys_used = [], set()
    for e in item_order:
        sigs: dict = {}
        variant = 0
        for r in chosen[e]:
            inputs = tuple(sorted(type_of[x] for x in r.requires if x in items))
            true_base = tuple(sorted(type_of[x] for x in r.requires if x in bases))
            made = None
            for _ in range(16):
                others = [t for t in base_types if t not in true_base]
                others = [int(t) for t in rng.permutation(others)] if others else []
                need = pool_size - len(true_base)
                extra = others[:need]
                if need > len(extra):  # too few base facts in the world: decoys without data meaning
                    extra += [int(t) for t in rng.choice(spare, need - len(extra), replace=False)]
                pool = tuple(sorted(true_base + tuple(extra)))
                cand = RQRecipe(type_of[e], inputs, pool, true_base, variant)
                if all(cand.signature(p) not in sigs.get(p, ()) for p in PROFILES):
                    made = cand
                    break
            if made is None:
                dropped.append(f"{e} <- {'+'.join(r.requires)}: no distinct public signature")
                continue
            for p in PROFILES:
                sigs.setdefault(p, set()).add(made.signature(p))
            decoys_used |= {t for t in made.pool if t not in base_types}
            rq.append(made)
            variant += 1
    sizes = [len(r.true_base) for r in rq]
    weights = tuple(max(sizes.count(k), 0) / len(sizes) for k in range(1, max_true + 1))
    n_levels = max(conv[e] for e in items)
    by_item = {}
    for r in rq:
        by_item[r.effect] = by_item.get(r.effect, 0) + 1
    cfg = WorldConfig(
        n_levels=n_levels, items_per_level=max(1, len(ITEM_TYPES) // n_levels), pool_size=pool_size,
        max_true=max_true, arity_weights=weights,
        p_alternative=round(sum(v > 1 for v in by_item.values()) / len(by_item), 4),
        p_extra_item_input=round(sum(len(r.item_inputs) > 1 for r in rq) / len(rq), 4))
    params = (("pool_size", pool_size), ("max_true", max_true), ("max_alternatives", max_alternatives),
              ("seed", seed), ("unlocks", unlocks))
    key = "B" + stable_hash(("benchmark_world", graph.name, goals, params))[:8]
    world = World(key, seed, cfg, tuple(sorted((type_of[e], conv[e]) for e in items)), tuple(rq))
    names = tuple(sorted((t, n) for n, t in type_of.items() if n in items or n in bases))
    return ConvertedWorld(world, graph.name, goals, names, tuple(promoted), tuple(dropped),
                          tuple(sorted(decoys_used)), params, original)


def sink_goals(graph: RecipeGraph, unlocks: str = "as_base", max_alternatives: int | None = 2) -> list[str]:
    """Reachable derived elements that no kept recipe requires: the goals whose
    joint closure is the whole (kept) graph."""
    level, given, kept, _ = _prepare(graph, unlocks, max_alternatives)
    used = {x for rs in kept.values() for r in rs for x in r.requires}
    return sorted(e for e in kept if e not in used)


def convertible_goals(graph: RecipeGraph, **kw) -> list[str]:
    """Derived elements whose own closure converts (each as a single goal),
    in order of level, then name."""
    level = graph.layers(use_unlocks=kw.get("unlocks", "as_base") == "as_base")
    out = []
    for e in sorted((e for e in graph.derived if e in level), key=lambda x: (level[x], x)):
        try:
            recipequest_world(graph, [e], **kw)
        except ConversionError:
            continue
        out.append(e)
    return out
