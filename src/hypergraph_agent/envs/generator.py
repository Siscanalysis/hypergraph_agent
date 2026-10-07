"""World and task generation.

This module is on the evaluator side of the information boundary: a ``World``
holds each recipe's hidden true base prerequisites. Learning code receives only
``public_schema.PublicTaskSpec`` built by ``public_schema.public_view``.

Structure of a world
--------------------
Item types are arranged in levels 1..L. Each item has a primary recipe and,
with probability ``p_alternative``, an alternative recipe (OR across recipes,
AND within one). A level-l recipe consumes one item from level l-1 (a public
input) plus a hidden nonempty subset (size at most ``max_true``) of a public
pool of ``pool_size`` base facts. The pool is sampled before, and
independently of, the hidden subset.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .vocabulary import BASE_TYPES, ITEM_TYPES, TYPE_NAMES, is_base

PROFILE_KNOWN = "known_structure"
PROFILE_UNKNOWN = "unknown_prerequisites"
PROFILES = (PROFILE_KNOWN, PROFILE_UNKNOWN)

WORLD_MODES = ("per_task", "pool", "shared")


def stable_hash(obj) -> str:
    """Deterministic short hash of a JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def derive_seed(*parts) -> int:
    """Independent 63-bit seed from a tuple of labels (namespace seeding)."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class WorldConfig:
    n_levels: int = 8
    items_per_level: int = 2
    pool_size: int = 6
    max_true: int = 3
    arity_weights: tuple[float, ...] = (0.2, 0.5, 0.3)
    p_alternative: float = 0.25
    p_extra_item_input: float = 0.0

    def __post_init__(self):
        if self.n_levels * self.items_per_level > len(ITEM_TYPES):
            raise ValueError("not enough item types for n_levels * items_per_level")
        if not 1 <= self.max_true <= self.pool_size <= len(BASE_TYPES):
            raise ValueError("need 1 <= max_true <= pool_size <= number of base types")
        if len(self.arity_weights) != self.max_true:
            raise ValueError("arity_weights must have max_true entries")


@dataclass(frozen=True)
class Recipe:
    effect: int
    item_inputs: tuple[int, ...]
    pool: tuple[int, ...]
    true_base: tuple[int, ...]  # hidden in the unknown_prerequisites profile
    variant: int = 0

    def required(self) -> frozenset[int]:
        return frozenset(self.item_inputs) | frozenset(self.true_base)

    def signature(self, profile: str) -> str:
        """Type-level public signature; identifies a recipe across tasks."""
        items = ",".join(TYPE_NAMES[i] for i in sorted(self.item_inputs))
        if profile == PROFILE_KNOWN:
            base = ",".join(TYPE_NAMES[b] for b in sorted(self.true_base))
            return f"{TYPE_NAMES[self.effect]}<=[{items}]+{{{base}}}"
        pool = ",".join(TYPE_NAMES[b] for b in sorted(self.pool))
        return f"{TYPE_NAMES[self.effect]}<=[{items}]+pool{{{pool}}}"


@dataclass(frozen=True)
class World:
    world_key: str  # public opaque label used only as a belief scope
    seed: int
    config: WorldConfig
    levels: tuple[tuple[int, int], ...]  # (item type, level)
    recipes: tuple[Recipe, ...]

    def level_of(self, item: int) -> int:
        return dict(self.levels)[item]

    def items_at(self, level: int) -> list[int]:
        return sorted(it for it, lv in self.levels if lv == level)

    def recipes_for(self, item: int) -> list[Recipe]:
        return [r for r in self.recipes if r.effect == item]

    @property
    def n_levels(self) -> int:
        return max(lv for _, lv in self.levels)


def _sample_inputs(rng, by_level, level, cfg) -> tuple[int, ...]:
    if level == 1:
        return ()
    inputs = {int(rng.choice(by_level[level - 1]))}
    if level > 2 and rng.random() < cfg.p_extra_item_input:
        lower = [it for lv in range(1, level - 1) for it in by_level[lv]]
        inputs.add(int(rng.choice(lower)))
    return tuple(sorted(inputs))


def _sample_recipe(rng, item, inputs, cfg, variant) -> Recipe:
    pool = tuple(sorted(int(b) for b in rng.choice(BASE_TYPES, cfg.pool_size, replace=False)))
    weights = np.asarray(cfg.arity_weights, dtype=float)
    k = 1 + int(rng.choice(cfg.max_true, p=weights / weights.sum()))
    true_base = tuple(sorted(int(b) for b in rng.choice(pool, k, replace=False)))
    return Recipe(item, inputs, pool, true_base, variant)


def make_world(seed: int, cfg: WorldConfig, world_key: str | None = None) -> World:
    rng = np.random.default_rng(seed)
    n_items = cfg.n_levels * cfg.items_per_level
    items = [int(i) for i in rng.permutation(ITEM_TYPES)[:n_items]]
    levels = {items[i]: 1 + i // cfg.items_per_level for i in range(n_items)}
    by_level: dict[int, list[int]] = {}
    for it, lv in levels.items():
        by_level.setdefault(lv, []).append(it)
    for lv in by_level:
        by_level[lv].sort()

    recipes: list[Recipe] = []
    for item in sorted(levels, key=lambda it: (levels[it], it)):
        level = levels[item]
        primary = _sample_recipe(rng, item, _sample_inputs(rng, by_level, level, cfg), cfg, 0)
        recipes.append(primary)
        if rng.random() < cfg.p_alternative:
            for _ in range(16):
                alt = _sample_recipe(rng, item, _sample_inputs(rng, by_level, level, cfg), cfg, 1)
                if all(alt.signature(p) != primary.signature(p) for p in PROFILES):
                    recipes.append(alt)
                    break
    key = world_key or "W" + stable_hash(("world", seed))[:8]
    return World(key, seed, cfg, tuple(sorted(levels.items())), tuple(recipes))


@dataclass(frozen=True)
class TaskConfig:
    profile: str = PROFILE_KNOWN
    depth_min: int = 1
    depth_max: int = 4
    n_distractor_rules: int = 1
    n_distractor_base: int = 1
    p_init_base: float = 0.0
    p_init_item: float = 0.0
    failure_prob: float = 0.0
    budget_factor: float = 1.0
    budget_slack: int = 2
    max_budget: int = 256
    # "all": every fact is observed; "goal_only": base facts and the goal are
    # observed, intermediate items and the outcome of crafting them are not
    observe_items: str = "all"

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}")
        if not 0.0 <= self.failure_prob < 1.0:
            raise ValueError("failure_prob must be in [0, 1)")
        if self.observe_items not in ("all", "goal_only"):
            raise ValueError("observe_items must be 'all' or 'goal_only'")


@dataclass(frozen=True)
class Task:
    """A complete task including private truth. Evaluator side only."""

    task_key: str
    namespace: str
    seed: int
    world: World
    profile: str
    goal: int  # type id
    depth: int
    fact_types: tuple[int, ...]  # storage order
    fact_keys: tuple[str, ...]
    rules: tuple[Recipe, ...]  # storage order
    rule_keys: tuple[str, ...]
    initial_true: frozenset[int]  # type ids
    budget: int
    failure_prob: float
    observe_items: str = "all"

    def fact_index(self, type_id: int) -> int:
        return self.fact_types.index(type_id)

    def closure_items(self) -> set[int]:
        return _closure(self.world, self.goal)


def _closure(world: World, goal: int) -> set[int]:
    closure: set[int] = set()
    stack = [goal]
    while stack:
        it = stack.pop()
        if it in closure:
            continue
        closure.add(it)
        for r in world.recipes_for(it):
            stack.extend(r.item_inputs)
    return closure


def _opaque_keys(rng, prefix: str, n: int) -> tuple[str, ...]:
    codes = rng.choice(16 ** 6, size=n, replace=False)
    return tuple(f"{prefix}{int(c):06x}" for c in codes)


def public_upper_bound(n_base_absent: int, n_items_absent: int) -> int:
    """Constructive plan length from public counts: gather every base fact,
    craft every item once in level order, submit."""
    return n_base_absent + n_items_absent + 1


def make_task(
    world: World,
    seed: int,
    cfg: TaskConfig,
    namespace: str,
    goal: int | None = None,
    depth: int | None = None,
) -> Task:
    rng = np.random.default_rng(seed)
    if goal is None:
        if depth is None:
            hi = min(cfg.depth_max, world.n_levels)
            lo = min(cfg.depth_min, hi)
            depth = int(rng.integers(lo, hi + 1))
        goal = int(rng.choice(world.items_at(depth)))
    depth = world.level_of(goal)

    closure = _closure(world, goal)
    rules = [r for r in world.recipes if r.effect in closure]
    candidates = [
        r for r in world.recipes
        if r.effect not in closure and set(r.item_inputs) <= closure
    ]
    if candidates and cfg.n_distractor_rules > 0:
        picks = rng.permutation(len(candidates))[: cfg.n_distractor_rules]
        rules += [candidates[int(i)] for i in sorted(picks)]
    items = closure | {r.effect for r in rules}
    bases = {b for r in rules for b in r.pool}
    spare = [b for b in BASE_TYPES if b not in bases]
    if spare and cfg.n_distractor_base > 0:
        n = min(cfg.n_distractor_base, len(spare))
        bases |= {int(b) for b in rng.choice(spare, n, replace=False)}

    fact_types = sorted(items | bases)
    fact_types = tuple(int(fact_types[i]) for i in rng.permutation(len(fact_types)))
    rule_order = rng.permutation(len(rules))
    rules_t = tuple(rules[int(i)] for i in rule_order)

    initial = set()
    for ft in fact_types:
        p = cfg.p_init_base if is_base(ft) else cfg.p_init_item
        if ft != goal and p > 0 and rng.random() < p:
            initial.add(ft)

    n_base_absent = sum(1 for ft in fact_types if is_base(ft) and ft not in initial)
    n_item_absent = sum(1 for ft in fact_types if not is_base(ft) and ft not in initial)
    upper = public_upper_bound(n_base_absent, n_item_absent)
    budget = min(cfg.max_budget, math.ceil(cfg.budget_factor * upper) + cfg.budget_slack)

    task_key = "T" + stable_hash(("task", world.world_key, namespace, seed))[:10]
    return Task(
        task_key=task_key,
        namespace=namespace,
        seed=seed,
        world=world,
        profile=cfg.profile,
        goal=goal,
        depth=depth,
        fact_types=fact_types,
        fact_keys=_opaque_keys(rng, "f", len(fact_types)),
        rules=rules_t,
        rule_keys=_opaque_keys(rng, "r", len(rules_t)),
        initial_true=frozenset(initial),
        budget=budget,
        failure_prob=cfg.failure_prob,
        observe_items=cfg.observe_items,
    )


@dataclass(frozen=True)
class TaskStreamConfig:
    """Deterministic, namespaced stream of tasks.

    world_mode:
      per_task: a fresh world for every task (P1).
      pool:     ``n_worlds`` persistent worlds, visited round-robin (P2).
      shared:   one world with fixed mechanics for every namespace (P3).
    """

    namespace: str
    base_seed: int
    world_mode: str = "per_task"
    n_worlds: int = 0
    shared_world_seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    goal_levels: tuple[int, ...] | None = None  # restrict goals to these levels
    repeat_index: int | None = None  # overfit diagnostic: every draw returns this task

    def __post_init__(self):
        if self.world_mode not in WORLD_MODES:
            raise ValueError(f"unknown world_mode {self.world_mode!r}")
        if self.world_mode == "pool" and self.n_worlds <= 0:
            raise ValueError("pool mode needs n_worlds > 0")

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return stable_hash(self.to_dict())


class TaskStream:
    def __init__(self, cfg: TaskStreamConfig):
        self.cfg = cfg
        self._worlds: dict[int, World] = {}

    def world_index(self, i: int) -> int:
        if self.cfg.world_mode == "per_task":
            return i
        if self.cfg.world_mode == "pool":
            return i % self.cfg.n_worlds
        return 0

    def world(self, widx: int) -> World:
        if widx not in self._worlds:
            c = self.cfg
            if c.world_mode == "shared":
                seed = c.shared_world_seed
                key = "W" + stable_hash(("shared", seed))[:8]
            else:
                seed = derive_seed("world", c.base_seed, c.namespace, widx)
                key = "W" + stable_hash(("world", c.base_seed, c.namespace, widx))[:8]
            self._worlds[widx] = make_world(seed, c.world, key)
        return self._worlds[widx]

    def task(self, i: int) -> Task:
        c = self.cfg
        if c.repeat_index is not None:
            i = c.repeat_index
        world = self.world(self.world_index(i))
        tseed = derive_seed("task", c.base_seed, c.namespace, i)
        goal = None
        if c.goal_levels:
            rng = np.random.default_rng(derive_seed("goal", c.base_seed, c.namespace, i))
            levels = [lv for lv in c.goal_levels if lv <= world.n_levels]
            goal = int(rng.choice(world.items_at(int(rng.choice(levels)))))
        return make_task(world, tseed, c.task, c.namespace, goal=goal)

    def manifest(self, n: int) -> dict:
        entries = []
        for i in range(n):
            t = self.task(i)
            entries.append({
                "index": i, "task_key": t.task_key, "world_key": t.world.world_key,
                "goal": TYPE_NAMES[t.goal], "depth": t.depth,
                "n_facts": len(t.fact_types), "n_rules": len(t.rules), "budget": t.budget,
            })
        body = {"stream": self.cfg.to_dict(), "entries": entries}
        return {**body, "manifest_hash": stable_hash(body)}
