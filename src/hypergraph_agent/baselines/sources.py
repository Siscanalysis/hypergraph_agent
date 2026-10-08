"""Task sources and references of study X (evaluator side).

A source yields world blocks: the consecutive episodes of one world, in
world-major order, as ``TaskItem``s. The wrappers pass ``TaskItem.task`` only to
the environment's reset; learners never receive it. References (privileged)
are computed here for the rows of evaluation runs and are never shown to any
agent.

RecipeQuest streams use the generator of the walker studies in pool mode: the
task at stream index ``e * n_worlds + w`` is episode ``e`` of world ``w``, so a
test stream with ``n_worlds`` worlds and 16 episodes per world is exactly the
world-major order of ``evaluation.walkers.eval_order``, and the repository's
agents can play the same tasks through ``evaluation.walkers.play``. A test
seed ``s`` plays base seed ``base_seed + 1000 s`` (``eval_world_offset_by_seed``
of study R), so seeds play distinct worlds. Training sources draw worlds
``rank, rank + n_envs, rank + 2 n_envs, ...`` from a pool of 10**9 worlds of
the training namespace, so parallel environments of one run never share a
world and no world recurs within any planned training budget.

TechTree streams use ``TechStream``: worlds depend on (namespace, seed, world
index) and the world config; test seed ``s`` is the stream seed ``s``, as in
studies U and L.

External benchmarks (extension point): a setting with ``loader:
"package.module:function"`` takes its world blocks from that function instead
of the generator. The function is called as ``loader(setting, namespace, seed,
world_index, episodes_per_world)`` and returns the tasks of one world block in
episode order: ``envs.generator.Task`` objects for the RecipeQuest family,
``techtree.generator.TechTask`` objects for TechTree, or None when the
benchmark has no world of that index (a finite test set ends there). Everything
downstream (wrappers, references, runner, analysis) is unchanged, provided the
tasks fit the fixed layouts (RecipeQuest: the type vocabulary and at most
``setting.max_variants`` recipes per item type).
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field

from ..envs.generator import TaskConfig, TaskStream, TaskStreamConfig, WorldConfig, derive_seed
from ..techtree.generator import TechStream, TechStreamConfig, TechWorldConfig

TRAIN_POOL = 10 ** 9
SEED_STRIDE = 1000


@dataclass(frozen=True)
class TaskItem:
    task: object  # evaluator side: passed only to the environment's reset
    world_key: str
    world_episode: int
    order: int  # RecipeQuest stream index; TechTree position in the stream order
    dyn_seed: int
    meta: dict = field(default_factory=dict)  # evaluator-side bookkeeping for rows


def rq_world_config(setting: dict) -> WorldConfig:
    w = dict(setting.get("world", {}))
    if "arity_weights" in w:
        w["arity_weights"] = tuple(w["arity_weights"])
    return WorldConfig(**w)


def rq_stream_config(setting: dict, namespace: str, base_seed: int, n_worlds: int) -> TaskStreamConfig:
    return TaskStreamConfig(namespace, base_seed, "pool", n_worlds, 0, rq_world_config(setting),
                            TaskConfig(**setting.get("tasks", {})))


def tt_world_config(setting: dict) -> TechWorldConfig:
    w = dict(setting.get("world", {}))
    if "concepts_per_level" in w:
        w["concepts_per_level"] = tuple(w["concepts_per_level"])
    return TechWorldConfig(**w)


def tt_stream_config(setting: dict, namespace: str, seed: int, n_worlds: int,
                     episodes_per_world: int) -> TechStreamConfig:
    s = dict(setting.get("stream", {}))
    if "goal_levels" in s:
        s["goal_levels"] = tuple(s["goal_levels"])
    return TechStreamConfig(namespace=namespace, seed=seed, n_worlds=n_worlds,
                            episodes_per_world=episodes_per_world, world=tt_world_config(setting), **s)


def load_function(spec: str):
    """``"package.module:function"`` -> the function (external benchmark loaders)."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"a loader is given as 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)


class BlockSource:
    """World blocks of one stream, in the order of ``worlds`` (finite or infinite)."""

    def __init__(self, family: str, stream_cfg, worlds, episodes_per_world: int, seed: int = 0,
                 loader=None, setting: dict | None = None, namespace: str | None = None):
        if family not in ("recipequest", "techtree"):
            raise ValueError(f"unknown family {family!r}")
        self.family, self.cfg, self.E, self.seed = family, stream_cfg, episodes_per_world, seed
        self.loader, self.setting, self.namespace = loader, setting, namespace
        self._worlds = iter(worlds)
        self.blocks = 0

    def _external(self, w: int) -> list[TaskItem] | None:
        tasks = self.loader(self.setting, self.namespace, self.seed, w, self.E)
        if tasks is None:
            return None
        out = []
        for e, t in enumerate(tasks):
            i = w * self.E + e
            meta = {"budget": t.budget, "world_index": w}
            if self.family == "recipequest":
                meta["depth"] = t.depth
            out.append(TaskItem(t, t.world.world_key, e, i, self._dyn(i), meta))
        return out

    def _dyn(self, i: int) -> int:
        """Dynamics seed of the task at order ``i``, as ``evaluation.walkers.play`` derives it."""
        return derive_seed("dynamics", self.seed, i) % (2 ** 31)

    def next_block(self) -> list[TaskItem] | None:
        w = next(self._worlds, None)
        if w is None:
            return None
        if self.loader is not None:
            block = self._external(w)
            self.blocks += block is not None
            return block
        self.blocks += 1
        out = []
        # a fresh stream object per block: worlds are not revisited, so no world cache grows over training
        if self.family == "recipequest":
            stream, n = TaskStream(self.cfg), self.cfg.n_worlds
            for e in range(self.E):
                i = e * n + w
                t = stream.task(i)
                out.append(TaskItem(t, t.world.world_key, e, i, self._dyn(i),
                                    {"stream_index": i, "depth": t.depth, "budget": t.budget}))
        else:
            stream = TechStream(self.cfg)
            for e in range(self.E):
                t = stream.task(w, e)
                out.append(TaskItem(t, t.world.world_key, e, w * self.E + e, 0,
                                    {"world_index": w, "goal_level": t.world.levels[t.goal],
                                     "budget": t.budget}))
        return out


def _source(family, setting, namespace, stream_seed, label_seed, worlds, n_worlds, episodes):
    if setting.get("loader"):
        return BlockSource(family, None, worlds, episodes, label_seed, load_function(setting["loader"]),
                           setting, namespace)
    if family == "recipequest":
        cfg = rq_stream_config(setting, namespace, stream_seed, n_worlds)
    else:
        cfg = tt_stream_config(setting, namespace, stream_seed, n_worlds, episodes)
    return BlockSource(family, cfg, worlds, episodes, label_seed)


def training_source(family: str, setting: dict, namespace: str, base_seed: int, train_seed: int,
                    rank: int, n_envs: int, episodes_per_world: int) -> BlockSource:
    seed = base_seed + SEED_STRIDE * train_seed
    return _source(family, setting, namespace, seed, seed, itertools.count(rank, n_envs), TRAIN_POOL,
                   episodes_per_world)


def evaluation_source(family: str, setting: dict, namespace: str, base_seed: int, test_seed: int,
                      n_worlds: int, episodes_per_world: int) -> BlockSource:
    """The stream one test seed plays: ``n_worlds`` worlds x ``episodes_per_world`` episodes."""
    stream_seed = base_seed + SEED_STRIDE * test_seed if family == "recipequest" else test_seed
    return _source(family, setting, namespace, stream_seed, test_seed, range(n_worlds), n_worlds,
                   episodes_per_world)


def stream_budget(family: str, setting: dict, namespace: str, base_seed: int, test_seed: int,
                  n_worlds: int, episodes_per_world: int) -> int:
    """Sum of the task budgets of one evaluation stream (generator only): no run playing it can
    use more interactions."""
    src = evaluation_source(family, setting, namespace, base_seed, test_seed, n_worlds, episodes_per_world)
    total = 0
    while (block := src.next_block()) is not None:
        total += sum(it.meta["budget"] for it in block)
    return total


def reference(family: str, item: TaskItem) -> tuple[int | None, bool]:
    """PRIVILEGED: optimal length of the task and whether it is exact (evaluator rows only)."""
    if family == "recipequest":
        from ..envs.reference_solver import reference_solve
        ref = reference_solve(item.task)
    else:
        from ..techtree.generator import reference_solve
        ref = reference_solve(item.task)
    return ref.length, ref.exact
