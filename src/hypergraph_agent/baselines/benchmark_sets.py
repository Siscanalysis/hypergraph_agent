"""Benchmark-derived RecipeQuest test sets of study X (evaluator side).

The external-loader hook of ``sources`` (``setting.loader:
"hypergraph_agent.baselines.benchmark_sets:load"``) with the parameters in
``setting.benchmark``. Worlds come from published recipe graphs through the
converters of ``hypergraph_agent.benchmarks`` (``to_recipequest``), unchanged:

- ``worlds: whole``: the whole kept graph as one world (the closure of its sink
  goals), one world per conversion seed ``conversion_seed + i`` for
  ``i < n_worlds``. Conversion seeds change the vocabulary types and the decoys
  of the pools, not the hidden structure, so these worlds share one hidden
  structure (Crafter, the Craft world of policy sketches).
- ``worlds: per_goal``: one world per goal, the closure of that goal, for
  ``n_worlds`` goals drawn without replacement (seed ``goal_sample_seed``) from
  the goals whose closure converts, sorted; conversion seed
  ``conversion_seed``. Closures of different goals overlap within one graph
  (MSGI Mining, Little Alchemy 2, Freeciv).

Every episode of a world asks for an item drawn uniformly among the world's
items (seeded by namespace, test seed, world and episode); tasks follow the
setting's ``tasks`` (hidden prerequisites, goal-only observation, no distractor
base facts, failure probability 0, budget two steps above the public
brute-force bound, as in study R). ``records`` gives the conversion record of
every world (promoted elements, dropped recipes, share of the closure kept),
which the analysis reports with every result.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..envs.generator import TaskConfig, derive_seed, make_task

BENCHMARK_DEFAULTS = {"graph": None, "worlds": "whole", "n_worlds": 1, "conversion_seed": 0,
                      "goal_sample_seed": 0, "pool_size": 6, "max_true": 3, "max_alternatives": 2}
LOADER = "hypergraph_agent.baselines.benchmark_sets:load"


def params(setting: dict) -> dict:
    raw = setting.get("benchmark") or {}
    unknown = set(raw) - set(BENCHMARK_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown setting.benchmark keys {sorted(unknown)}")
    p = {**BENCHMARK_DEFAULTS, **raw}
    if p["graph"] is None or p["worlds"] not in ("whole", "per_goal"):
        raise ValueError("setting.benchmark needs a graph and worlds 'whole' or 'per_goal'")
    return p


@lru_cache(maxsize=None)
def _graph(name: str):
    from ..benchmarks import loaders
    return loaders.load(name)


def _conversion(p: dict) -> dict:
    return {k: p[k] for k in ("pool_size", "max_true", "max_alternatives")}


@lru_cache(maxsize=None)
def goal_sample(graph: str, n: int, seed: int, pool_size: int, max_true: int, max_alternatives: int) -> tuple:
    from ..benchmarks.to_recipequest import convertible_goals
    goals = convertible_goals(_graph(graph), pool_size=pool_size, max_true=max_true,
                              max_alternatives=max_alternatives)
    if n >= len(goals):
        return tuple(goals)
    rng = np.random.default_rng(derive_seed("x_benchmark_goals", graph, seed))
    return tuple(sorted(str(g) for g in rng.choice(goals, n, replace=False)))


@lru_cache(maxsize=None)
def _converted(graph: str, worlds: str, n: int, conversion_seed: int, goal_seed: int, pool_size: int,
               max_true: int, max_alternatives: int, world_index: int):
    from ..benchmarks.to_recipequest import recipequest_world, sink_goals
    if not 0 <= world_index < n:
        return None
    g = _graph(graph)
    kw = {"pool_size": pool_size, "max_true": max_true, "max_alternatives": max_alternatives}
    if worlds == "whole":
        return recipequest_world(g, sink_goals(g, max_alternatives=max_alternatives),
                                 seed=conversion_seed + world_index, **kw)
    goals = goal_sample(graph, n, goal_seed, pool_size, max_true, max_alternatives)
    if world_index >= len(goals):
        return None
    return recipequest_world(g, [goals[world_index]], seed=conversion_seed, **kw)


def converted(setting: dict, world_index: int):
    """The ``ConvertedWorld`` of a world index, or None beyond the set."""
    p = params(setting)
    return _converted(p["graph"], p["worlds"], p["n_worlds"], p["conversion_seed"], p["goal_sample_seed"],
                      p["pool_size"], p["max_true"], p["max_alternatives"], world_index)


def load(setting: dict, namespace: str, seed: int, world_index: int, episodes: int):
    """Tasks of one world block (``sources`` loader interface)."""
    cw = converted(setting, world_index)
    if cw is None:
        return None
    items = sorted(t for t, _ in cw.world.levels)
    cfg = TaskConfig(**setting.get("tasks", {}))
    tasks = []
    for e in range(episodes):
        rng = np.random.default_rng(derive_seed("x_benchmark_goal", namespace, seed, world_index, e))
        goal = int(rng.choice(items))
        task_seed = derive_seed("x_benchmark_task", namespace, seed, world_index, e)
        tasks.append(make_task(cw.world, task_seed, cfg, namespace, goal=goal))
    return tasks


def records(setting: dict) -> list[dict]:
    """Conversion record of every world of the set (evaluator side, for reports)."""
    out, w = [], 0
    while (cw := converted(setting, w)) is not None:
        out.append({"world_index": w, "world_key": cw.world.world_key, "graph": cw.graph,
                    "goals": list(cw.goals),
                    "params": dict(cw.params), "items": len(cw.world.levels),
                    "recipes": len(cw.world.recipes), "closure_items": cw.closure_items,
                    "kept_share": round(cw.kept_share, 4), "promoted": list(cw.promoted),
                    "dropped": list(cw.dropped), "decoy_types": len(cw.decoy_types)})
        w += 1
    return out


def summary(recs: list[dict]) -> dict:
    """Means over the worlds of a set."""
    if not recs:
        return {"worlds": 0}
    return {"worlds": len(recs), "items_mean": float(np.mean([r["items"] for r in recs])),
            "promoted_mean": float(np.mean([len(r["promoted"]) for r in recs])),
            "dropped_mean": float(np.mean([len(r["dropped"]) for r in recs])),
            "kept_share_mean": float(np.mean([r["kept_share"] for r in recs])),
            "decoy_types_mean": float(np.mean([r["decoy_types"] for r in recs]))}
