"""Graph statistics of the external benchmarks and of the generated families.

``python -m hypergraph_agent.benchmarks.stats`` prints the statistics of every
benchmark whose data is present; ``--write`` also writes
``benchmarks/<name>/stats.json`` and ``benchmarks/generated_families.json``;
``--export DIR`` writes each loaded graph in the neutral JSON format. Evaluator
side: it uses the reference solvers. No environment interaction is spent.

Four groups of statistics per graph:

* ``graph``: elements, base, recipes, AND arity, alternatives per derived
  element, depth (breadth-first level), reuse (how often derived elements are
  requirements of other recipes) and the size of the shallowest derivation of
  each element (derived elements in the tree that uses every element's
  shallowest recipe; an upper bound on the crafts of the cheapest plan);
* ``concepts``: the TechTree-like concept DAG (``to_techtree.concept_stats``);
* ``techtree_fit``: how much of the graph obeys TechTree's level rules
  (``to_techtree.techworld_from_graph``) for the linear and doubling rules,
  with base elements as level-1 concepts (combo length 1) or with level-1
  combos of two base elements (combo length 2);
* ``recipequest``: RecipeQuest worlds converted from the graph
  (``to_recipequest``), one per goal for a seeded sample of the convertible
  goals, and, when it fits the vocabulary, one world holding the whole graph.
  Task statistics follow the definitions of the study R family table
  (docs/studies/R_replication.md): optimal length, hidden recipes per task
  (every rule, including the distractor), alternative recipes needed by the
  goal, hypothesis entropy (sum over hidden recipes of log2 of the class
  size: nonempty subsets of at most 3 pool members), crafts of the optimal
  plan, the brute-force length (cheapest plan when every pool member is
  required) and the budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ..envs.derivations import StructRecipe, enumerate_derivations, plan_cost
from ..envs.generator import TaskConfig, WorldConfig, derive_seed, make_task, make_world
from ..envs.reference_solver import reference_solve
from ..envs.vocabulary import KIND_FACILITY, TYPE_NAMES, kind_of
from ..techtree.generator import TechWorldConfig, make_world as make_techworld
from . import loaders, sources
from .graph import Recipe, RecipeGraph, make_graph
from .to_recipequest import ConversionError, convertible_goals, hypotheses, recipequest_world, sink_goals
from .to_techtree import concept_dag, concept_stats, techworld_concepts, techworld_from_graph

MAX_TRUE = 3
TASK_CFG = TaskConfig(profile="unknown_prerequisites", n_distractor_base=0, observe_items="goal_only")
RECIPEQUEST_SAMPLE = 40


def _r(x: float) -> float:
    return round(float(x), 4)


def _hist(xs) -> dict:
    return {str(k): v for k, v in sorted(Counter(xs).items())}


def _summary(xs) -> dict:
    xs = list(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": _r(np.mean(xs)), "median": _r(np.median(xs)), "min": _r(min(xs)),
            "max": _r(max(xs))}


# ------------------------------------------------------------------- graph
def shallowest(graph: RecipeGraph, level: dict) -> dict:
    out = {}
    for e in graph.derived:
        if e not in level:
            continue
        rs = [r for r in graph.recipes_for(e)
              if all(x in level and level[x] < level[e] for x in r.requires)]
        rs.sort(key=lambda r: (max(level[x] for x in r.requires), sum(level[x] for x in r.requires),
                               r.requires))
        out[e] = rs[0] if rs else None
    return out


def graph_stats(graph: RecipeGraph) -> dict:
    level = graph.layers()
    derived = graph.derived
    reach = [e for e in derived if e in level]
    rec = graph.recipes
    arity = [len(r.requires) for r in rec]
    alts = [len(graph.recipes_for(e)) for e in derived]
    is_derived = set(derived)
    uses = Counter(x for r in rec for x in r.requires)
    first = shallowest(graph, level)
    memo: dict = {}

    def tree(e):
        if e not in memo:
            r = first.get(e)
            s = {e}
            if r is not None:
                for x in r.requires:
                    if x in is_derived and x in first:
                        s |= tree(x)
            memo[e] = s
        return memo[e]

    sizes = [len(tree(e)) for e in sorted(reach, key=lambda x: level[x])]
    return {
        "elements": len(graph.elements),
        "base": len(graph.base),
        "derived": len(derived),
        "reachable_derived": len(reach),
        "unlocks": len(graph.unlocks),
        "unlock_kinds": _hist(u.kind for u in graph.unlocks),
        "unlockable_elements": sum(1 for e in graph.unlock_targets if e not in is_derived),
        "recipes": len(rec),
        "and_arity": _summary(arity),
        "and_arity_histogram": _hist(arity),
        "recipes_per_derived": _summary(alts),
        "share_derived_with_alternatives": _r(sum(a > 1 for a in alts) / max(1, len(alts))),
        "depth": _summary([level[e] for e in reach]),
        "depth_histogram": _hist(level[e] for e in reach),
        "share_recipes_with_derived_requirement": _r(
            sum(any(x in is_derived for x in r.requires) for r in rec) / max(1, len(rec))),
        "share_recipes_only_derived_requirements": _r(
            sum(all(x in is_derived for x in r.requires) for r in rec) / max(1, len(rec))),
        "share_derived_reused": _r(sum(1 for e in derived if uses[e]) / max(1, len(derived))),
        "uses_per_derived": _summary([uses[e] for e in derived]),
        "shallowest_derivation_size": _summary(sizes),
    }


def world_graph(world, name: str) -> RecipeGraph:
    """The true structure of a RecipeQuest world as a recipe graph."""
    base = {TYPE_NAMES[b] for r in world.recipes for b in r.true_base}
    recipes = [Recipe(TYPE_NAMES[r.effect], tuple(TYPE_NAMES[x] for x in r.item_inputs + r.true_base))
               for r in world.recipes]
    kinds = {b: "facility" for b in base if kind_of(TYPE_NAMES.index(b)) == KIND_FACILITY}
    return make_graph(name, base, recipes, kinds=kinds)


# -------------------------------------------------------------- RecipeQuest
def brute_force_length(task) -> int:
    recipes = [StructRecipe(r.effect, r.item_inputs, r.pool) for r in task.rules]
    ds, _, _ = enumerate_derivations(recipes, frozenset(task.initial_true), task.goal)
    return plan_cost(ds[0])


def task_stats(task) -> dict:
    ref = reference_solve(task)
    closure = task.closure_items()
    return {
        "optimal": ref.length, "exact": ref.exact, "depth": task.depth, "budget": task.budget,
        "hidden": len(task.rules),
        "alternatives": sum(r.variant >= 1 and r.effect in closure for r in task.rules),
        "entropy": sum(math.log2(hypotheses(len(r.pool), MAX_TRUE)) for r in task.rules),
        "plan_crafts": sum(k.startswith("craft:") for k in ref.plan),
        "brute_force": brute_force_length(task),
    }


def world_stats(world) -> dict:
    rec = world.recipes
    per_item = Counter(r.effect for r in rec)
    return {
        "items": len(world.levels),
        "levels": world.n_levels,
        "recipes": len(rec),
        "base_facts": len({b for r in rec for b in r.pool}),
        "share_items_with_alternative": _r(sum(v > 1 for v in per_item.values()) / len(per_item)),
        "item_inputs_per_recipe": _r(np.mean([len(r.item_inputs) for r in rec])),
        "true_base_per_recipe": _r(np.mean([len(r.true_base) for r in rec])),
        "pool_per_recipe": _r(np.mean([len(r.pool) for r in rec])),
    }


def _task_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"tasks": 0}
    opt = [r["optimal"] for r in rows]
    return {
        "tasks": len(rows),
        "all_exact": all(r["exact"] for r in rows),
        "depth_histogram": _hist(r["depth"] for r in rows),
        "optimal_length": _summary(opt),
        "hidden_recipes": _summary([r["hidden"] for r in rows]),
        "alternative_recipes": _r(np.mean([r["alternatives"] for r in rows])),
        "entropy_bits": _r(np.mean([r["entropy"] for r in rows])),
        "plan_crafts": _r(np.mean([r["plan_crafts"] for r in rows])),
        "headroom": _r(np.mean([r["brute_force"] / r["optimal"] for r in rows])),
        "budget": _summary([r["budget"] for r in rows]),
        "brute_force_within_budget": all(r["brute_force"] <= r["budget"] for r in rows),
    }


def _mean_dicts(ds: list[dict]) -> dict:
    keys = [k for k, v in ds[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {k: _r(np.mean([d[k] for d in ds])) for k in keys}


def recipequest_stats(graph: RecipeGraph, sample: int = RECIPEQUEST_SAMPLE, seed: int = 0) -> dict:
    goals = convertible_goals(graph)
    level = graph.layers()
    reach = [e for e in graph.derived if e in level]
    rng = np.random.default_rng(derive_seed("benchmark_stats", graph.name, seed))
    picked = sorted(rng.choice(goals, size=min(sample, len(goals)), replace=False).tolist()) if goals else []
    worlds, rows, promoted, dropped, kept = [], [], [], [], []
    for i, g in enumerate(picked):
        cw = recipequest_world(graph, [g], seed=seed)
        worlds.append(world_stats(cw.world))
        promoted.append(len(cw.promoted))
        kept.append(cw.kept_share)
        dropped.append(len(cw.dropped))
        rows.append(task_stats(cw.task(g, seed=i, cfg=TASK_CFG)))
    out = {
        "convertible_goals": len(goals),
        "reachable_derived": len(reach),
        "share_convertible": _r(len(goals) / max(1, len(reach))),
        "sampled_worlds": len(picked),
        "per_goal_worlds": _mean_dicts(worlds) if worlds else {},
        "promoted_per_world": _summary(promoted),
        "kept_share_of_closure": _summary(kept),
        "dropped_recipes_per_world": _summary(dropped),
        "per_goal_tasks": _task_summary(rows),
    }
    sinks = sink_goals(graph)
    try:
        cw = recipequest_world(graph, sinks, seed=seed)
    except ConversionError as e:
        out["whole_graph_world"] = {"converted": False, "reason": str(e)}
    else:
        items = cw.items
        trows = [task_stats(cw.task(it, seed=i, cfg=TASK_CFG)) for i, it in enumerate(sorted(items))]
        out["whole_graph_world"] = {"converted": True, "goals": len(cw.goals), "promoted": len(cw.promoted),
                                    "kept_share_of_closure": _r(cw.kept_share),
                                    "dropped_recipes": len(cw.dropped), "world": world_stats(cw.world),
                                    "tasks_one_per_item": _task_summary(trows)}
    return out


def techtree_fit(graph: RecipeGraph) -> dict:
    out = {}
    for rule in ("linear", "doubling"):
        for k in (1, 2):
            try:
                out[f"{rule}_combo{k}"] = techworld_from_graph(graph, rule, combo_length=k).report
            except ConversionError as e:
                out[f"{rule}_combo{k}"] = {"rule": rule, "combo_length": k, "fitted_derived": 0,
                                           "coverage": 0.0, "reason": str(e)}
    return out


def benchmark_stats(benchmark: str) -> dict:
    graphs = {}
    for name in loaders.graphs_of(benchmark):
        g = loaders.load(name)
        graphs[name] = {"graph": graph_stats(g), "concepts": concept_stats(concept_dag(g)),
                        "techtree_fit": techtree_fit(g), "recipequest": recipequest_stats(g),
                        "notes": list(g.notes)}
    return {"benchmark": benchmark, "generated_by": "python -m hypergraph_agent.benchmarks.stats --write",
            "graphs": graphs}


# -------------------------------------------------------- generated families
FAMILIES = {
    # study R families (docs/studies/R_replication.md, configs/replication/f1.yaml, f2.yaml)
    "F1": (WorldConfig(), (1, 3)),
    "F2": (WorldConfig(p_alternative=0.6, pool_size=8, max_true=3, p_extra_item_input=0.25), (3, 5)),
}
TECH_FAMILIES = {
    # study L worlds (configs/layers/l1_main.yaml) at reuse depth 0, 1, 2;
    # study U (configs/unlock/u_main.yaml) at composite length 3
    "TechTree-L-d0": TechWorldConfig(n_primitives=3, n_levels=3, concepts_per_level=(3, 3, 3), combo_length=2,
                                     length_rule="linear", reuse_depth=0),
    "TechTree-L-d1": TechWorldConfig(n_primitives=3, n_levels=3, concepts_per_level=(3, 3, 3), combo_length=2,
                                     length_rule="linear", reuse_depth=1),
    "TechTree-L-d2": TechWorldConfig(n_primitives=3, n_levels=3, concepts_per_level=(3, 3, 3), combo_length=2,
                                     length_rule="linear", reuse_depth=2),
    "TechTree-U-k3": TechWorldConfig(n_primitives=4, n_slots=3, n_unlocks=1, unlock_level=1, n_levels=1,
                                     concepts_per_level=(2,), combo_length=3),
}


def family_stats(n_worlds: int = 50, tasks_per_world: int = 4) -> dict:
    out = {}
    for fam, (wcfg, (lo, hi)) in FAMILIES.items():
        graphs, worlds, rows = [], [], []
        tcfg = TaskConfig(profile="unknown_prerequisites", depth_min=lo, depth_max=hi, n_distractor_base=0,
                          observe_items="goal_only")
        for i in range(n_worlds):
            w = make_world(derive_seed("benchmark_family", fam, i), wcfg)
            g = world_graph(w, f"{fam}-{i}")
            graphs.append(graph_stats(g))
            worlds.append(world_stats(w))
            for j in range(tasks_per_world):
                seed = derive_seed("benchmark_family_task", fam, i, j)
                rows.append(task_stats(make_task(w, seed, tcfg, "stats")))
        out[fam] = {"worlds": n_worlds,
                    "graph_mean": {k: v for k, v in _flat_means(graphs).items()},
                    "world_mean": _mean_dicts(worlds),
                    "tasks": _task_summary(rows)}
    for fam, tcfg in TECH_FAMILIES.items():
        cs = [concept_stats(techworld_concepts(make_techworld(derive_seed("benchmark_family", fam, i), tcfg)))
              for i in range(n_worlds)]
        out[fam] = {"worlds": n_worlds, "concepts_mean": _flat_means(cs)}
    return out


def _flat_means(ds: list[dict]) -> dict:
    """Mean of every scalar and of every ``mean``/``max`` entry of summaries."""
    out = {}
    for k, v in ds[0].items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = _r(np.mean([d[k] for d in ds]))
        elif isinstance(v, dict) and "mean" in v:
            out[k + "_mean"] = _r(np.mean([d[k]["mean"] for d in ds if d[k].get("n")]))
            out[k + "_max"] = _r(np.mean([d[k]["max"] for d in ds if d[k].get("n")]))
    return out


# ---------------------------------------------------------------------- CLI
def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Statistics of the external benchmarks and generated families")
    ap.add_argument("--write", action="store_true", help="write benchmarks/<name>/stats.json")
    ap.add_argument("--benchmarks", nargs="*", default=list(loaders.BENCHMARKS))
    ap.add_argument("--families", action="store_true", help="also the generated families")
    ap.add_argument("--export", metavar="DIR", help="write each loaded graph as neutral JSON to DIR")
    args = ap.parse_args(argv)
    for b in args.benchmarks:
        if not loaders.available(b):
            print(f"{b}: data absent or unverified; run python benchmarks/{b}/fetch.py")
            continue
        doc = benchmark_stats(b)
        if args.write:
            (sources.folder(b) / "stats.json").write_text(dumps(doc), encoding="utf-8", newline="\n")
            print(f"wrote benchmarks/{b}/stats.json")
        else:
            print(dumps(doc))
        if args.export:
            out = Path(args.export)
            out.mkdir(parents=True, exist_ok=True)
            for name in loaders.graphs_of(b):
                (out / f"{name}.json").write_text(loaders.load(name).to_json() + "\n", encoding="utf-8",
                                                  newline="\n")
    if args.families:
        doc = {"generated_by": "python -m hypergraph_agent.benchmarks.stats --families --write",
               "families": family_stats()}
        if args.write:
            (sources.BENCHMARKS_DIR / "generated_families.json").write_text(dumps(doc), encoding="utf-8",
                                                                          newline="\n")
            print("wrote benchmarks/generated_families.json")
        else:
            print(dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
