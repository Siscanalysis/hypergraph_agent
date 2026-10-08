"""TechTree-like structures from a recipe graph (evaluator side).

Two outputs, depending on how well the data fits TechTree's shape.

``concept_dag`` (always): every reachable derived element is a concept, its
parents are the requirements of its shallowest recipe (the first way it can be
discovered), primitives are the base elements. ``concept_stats`` measures the
compositional depth of discoveries: the reuse depth of a concept is 0 when no
parent is itself a concept and 1 + the largest reuse depth of its concept
parents otherwise. In a recipe graph every derived element is built from its
parents, so every concept counts as compositional; in a generated TechTree
world only the concepts drawn as compositions of their parents do (flat
concepts are gated by their parents but do not reuse their sequences).

``techworld_from_graph`` (when the data has that shape): a TechTree world
whose concepts, parents and levels come from the data. TechTree concepts are
secret sequences of primitives whose lengths are public per level: a level-1
concept is a combo of ``combo_length`` primitives, and under the ``linear``
rule a level-l concept has two distinct parents at levels l-1 and 1 (both at
level l-1 under ``doubling``) and its sequence is the concatenation of theirs.
The converter assigns each derived element the lowest TechTree level at which
one of its recipes has exactly that form, and leaves out every element that
never fits. With ``combo_length=2`` a level-1 concept is a recipe whose
requirements, counted with their amounts, are two primitives; with
``combo_length=1`` the base elements themselves are the level-1 concepts (one
primitive each), so a recipe "earlier discovery + base element" can fit the
linear rule (in Little Alchemy the most frequent of the shapes TechTree
allows). What the data does not provide
and the conversion fixes: the concatenation order (TechTree hides it; here the
higher-level parent comes first, ties by name, and combo symbols are sorted)
and, for an element with several fitting recipes, which one is its parent pair
(the shallowest; a pair already used by another concept of the level is
skipped, because two concepts of a level cannot share a sequence). Every
concept above level 1 is compositional (``reuse_depth = levels - 1``); there
are no unlock slots. Levels stop where the deepest window space exceeds
TechTree's limit for its elimination agents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..envs.generator import derive_seed, stable_hash
from ..techtree.generator import MAX_WINDOW_SPACE, TechTask, TechWorld, TechWorldConfig
from .graph import RecipeGraph
from .to_recipequest import ConversionError


# ---------------------------------------------------------------- concept DAG
@dataclass(frozen=True)
class ConceptDAG:
    name: str
    primitives: tuple[str, ...]
    concepts: tuple[str, ...]
    parents: tuple[tuple[str, ...], ...]  # per concept, in the order of ``concepts``
    levels: tuple[int, ...]
    compositional: tuple[bool, ...]


def _ranked(graph: RecipeGraph, e: str, level: dict) -> list:
    rs = [r for r in graph.recipes_for(e) if all(x in level for x in r.requires)]
    return sorted(rs, key=lambda r: (1 + max(level[x] for x in r.requires),
                                     sum(level[x] for x in r.requires), r.requires))


def concept_dag(graph: RecipeGraph) -> ConceptDAG:
    level = graph.layers()
    concepts = sorted((e for e in graph.derived if e in level), key=lambda x: (level[x], x))
    parents = []
    for e in concepts:
        ok = [r for r in _ranked(graph, e, level) if all(level[x] < level[e] for x in r.requires)]
        parents.append(ok[0].requires if ok else ())
    prims = tuple(sorted(e for e in level if e not in set(concepts)))
    return ConceptDAG(graph.name, prims, tuple(concepts), tuple(parents),
                      tuple(level[e] for e in concepts), tuple(True for _ in concepts))


def techworld_concepts(world: TechWorld) -> ConceptDAG:
    """The concept DAG of a generated TechTree world: level-1 concepts have the
    primitives of their combo as parents, others their two parent concepts."""
    names = world.keys
    parents = []
    for c, lv in enumerate(world.levels):
        if lv == 1:
            parents.append(tuple(sorted({f"p{s}" for s in world.sequences[c]})))
        else:
            parents.append(tuple(names[p] for p in world.parents[c]))
    prims = tuple(f"p{s}" for s in range(world.config.n_primitives))
    return ConceptDAG(world.world_key, prims, names, tuple(parents), world.levels,
                      tuple(lv >= 2 and comp for lv, comp in zip(world.levels, world.compositional)))


def _mean(xs) -> float:
    return round(float(np.mean(xs)), 4) if len(xs) else 0.0


def _hist(xs) -> dict:
    return {str(k): v for k, v in sorted(Counter(xs).items())}


def concept_stats(dag: ConceptDAG) -> dict:
    idx = {c: i for i, c in enumerate(dag.concepts)}
    concept_parents = [[p for p in ps if p in idx] for ps in dag.parents]
    reuse: dict = {}
    for i in sorted(range(len(dag.concepts)), key=lambda j: dag.levels[j]):
        cps = concept_parents[i]
        reuse[i] = 1 + max(reuse[idx[p]] for p in cps) if cps and dag.compositional[i] else 0
    children = Counter(p for cps in concept_parents for p in cps)
    slots = [len(ps) for ps in dag.parents]
    n = len(dag.concepts)
    # local shape of two-parent concepts at level l >= 2 (primitives at level 0):
    # with the deeper parent at level l-1, the other parent is a primitive, a
    # level-1 concept (TechTree's linear rule), also at level l-1 (doubling
    # rule) or in between; "other" when the deeper parent is not at l-1
    lv = {c: dag.levels[i] for i, c in enumerate(dag.concepts)}
    shape = Counter()
    for i, ps in enumerate(dag.parents):
        if len(ps) != 2 or dag.levels[i] < 2:
            continue
        a, b = sorted((lv.get(p, 0) for p in ps), reverse=True)
        if a != dag.levels[i] - 1:
            shape["other"] += 1
        elif b == 0:
            shape["primitive"] += 1
        elif b == 1:  # includes level 2, where the linear and doubling rules coincide
            shape["level_1"] += 1
        elif b == a:
            shape["level_l-1"] += 1
        else:
            shape["between"] += 1
    two = sum(shape.values())
    return {
        "concepts": n,
        "primitives": len(dag.primitives),
        "levels": max(dag.levels) if n else 0,
        "level_histogram": _hist(dag.levels),
        "parents_per_concept": _mean(slots),
        "parents_histogram": _hist(slots),
        "concept_parent_share": round(sum(len(c) for c in concept_parents) / max(1, sum(slots)), 4),
        "share_with_concept_parent": round(sum(1 for c in concept_parents if c) / max(1, n), 4),
        "reuse_depth_mean": _mean(list(reuse.values())),
        "reuse_depth_max": max(reuse.values()) if reuse else 0,
        "reuse_depth_histogram": _hist(reuse.values()),
        "share_reused_as_parent": round(sum(1 for c in dag.concepts if children[c]) / max(1, n), 4),
        "children_per_concept_mean": _mean([children[c] for c in dag.concepts]),
        "children_per_concept_max": max((children[c] for c in dag.concepts), default=0),
        "two_parent_concepts_above_level_1": two,
        "other_parent_shares": {k: round(v / two, 4) for k, v in sorted(shape.items())} if two else {},
    }


# ------------------------------------------------------------ TechTree worlds
@dataclass(frozen=True)
class TechConversion:
    world: TechWorld
    graph: str
    names: tuple[str, ...]  # element name per concept index (evaluator side)
    primitives: tuple[str, ...]  # element name per primitive symbol
    report: dict

    def concept(self, name: str) -> int:
        return self.names.index(name)


def _multiset(r) -> list[str]:
    amounts = dict(r.amounts)
    return sorted(x for x in r.requires for _ in range(amounts.get(x, 1)))


def fit_levels(graph: RecipeGraph, rule: str = "linear", combo_length: int = 2,
               max_levels: int | None = None):
    """Concept -> (TechTree level, parents in concatenation order, sequence)
    for every derived element that fits; also the primitives."""
    if rule not in ("linear", "doubling"):
        raise ValueError("rule must be 'linear' or 'doubling'")
    level = graph.layers()
    prims = sorted(graph.base)
    sym = {p: i for i, p in enumerate(prims)}
    P = len(prims)
    cfg_len = (lambda lv: combo_length * (lv if rule == "linear" else 2 ** (lv - 1)))
    derived = sorted((e for e in graph.derived if e in level), key=lambda x: (level[x], x))
    fit: dict = {}
    lv = 1
    if combo_length == 1:  # the base elements themselves are the level-1 concepts
        fit = {p: (1, (), (sym[p],)) for p in prims}
        lv = 2
    while True:
        if max_levels is not None and lv > max_levels:
            break
        if P ** cfg_len(lv) > MAX_WINDOW_SPACE:
            break
        used, new = set(), {}
        for e in derived:
            if e in fit:
                continue
            for r in _ranked(graph, e, level):
                if lv == 1:
                    ms = _multiset(r)
                    if len(ms) != combo_length or not all(x in sym for x in ms):
                        continue
                    seq, par = tuple(sorted(sym[x] for x in ms)), ()
                else:
                    if len(r.requires) != 2 or not all(x in fit for x in r.requires):
                        continue
                    want = sorted((lv - 1, 1 if rule == "linear" else lv - 1))
                    a, b = sorted(r.requires, key=lambda x: (-fit[x][0], x))
                    if sorted((fit[a][0], fit[b][0])) != want:
                        continue
                    seq, par = fit[a][2] + fit[b][2], (a, b)
                if seq in used:
                    continue
                used.add(seq)
                new[e] = (lv, par, seq)
                break
        if not new:
            break
        fit.update(new)
        lv += 1
    return fit, prims


def techworld_from_graph(graph: RecipeGraph, rule: str = "linear", combo_length: int = 2,
                         max_levels: int | None = None, max_per_level: int | None = None,
                         seed: int = 0) -> TechConversion:
    fit, prims = fit_levels(graph, rule, combo_length, max_levels)
    if not fit:
        raise ConversionError(f"no element of {graph.name} fits a TechTree level-1 combo of "
                              f"{combo_length} primitives")
    # optional cap per level: keep the concepts with most fitted descendants
    if max_per_level is not None:
        desc = Counter()
        for e in sorted(fit, key=lambda x: -fit[x][0]):
            for p in fit[e][1]:
                desc[p] += 1 + desc[e]
        kept: dict = {}
        for lv in range(1, max(v[0] for v in fit.values()) + 1):
            elig = [e for e in fit if fit[e][0] == lv and all(p in kept for p in fit[e][1])]
            for e in sorted(elig, key=lambda x: (-desc[x], x))[:max_per_level]:
                kept[e] = fit[e]
        fit = kept
    if not any(e not in graph.base for e in fit):
        raise ConversionError(f"no derived element of {graph.name} fits the {rule} rule")
    order = sorted(fit, key=lambda x: (fit[x][0], x))
    index = {e: i for i, e in enumerate(order)}
    n_levels = max(fit[e][0] for e in order)
    counts = tuple(sum(1 for e in order if fit[e][0] == lv) for lv in range(1, n_levels + 1))
    cfg = TechWorldConfig(n_primitives=len(prims), n_slots=0, n_unlocks=0, unlock_level=1, n_levels=n_levels,
                          concepts_per_level=counts, combo_length=combo_length, length_rule=rule,
                          reuse_depth=n_levels - 1)
    rng = np.random.default_rng(derive_seed("benchmark_techworld", graph.name, rule, combo_length, seed))
    keys = tuple(f"c{int(c):06x}" for c in rng.choice(16 ** 6, size=len(order), replace=False))
    world = TechWorld(
        world_key="B" + stable_hash(("benchmark_techworld", graph.name, rule, combo_length, max_levels,
                                     max_per_level, seed))[:8],
        seed=seed, config=cfg, keys=keys, levels=tuple(fit[e][0] for e in order),
        sequences=tuple(fit[e][2] for e in order),
        parents=tuple(tuple(index[p] for p in fit[e][1]) for e in order),
        compositional=tuple(fit[e][0] >= 2 for e in order), unlocks=(), entries=())
    reachable = graph.layers()
    derived = [e for e in graph.derived if e in reachable]
    fitted = sum(1 for e in order if e not in graph.base)
    report = {"rule": rule, "combo_length": combo_length, "primitives": len(prims),
              "derived": len(derived), "fitted_derived": fitted,
              "coverage": round(fitted / max(1, len(derived)), 4),
              "concepts_per_level": list(counts), "levels": n_levels}
    return TechConversion(world, graph.name, tuple(order), tuple(prims), report)


def techtask(conv: TechConversion, goal: str, budget: int = 100, namespace: str = "benchmark",
             episode: int = 0) -> TechTask:
    key = "T" + stable_hash(("benchmark_techtask", conv.world.world_key, goal, namespace, episode))[:10]
    return TechTask(key, namespace, conv.world, conv.concept(goal), budget, 0, episode)
