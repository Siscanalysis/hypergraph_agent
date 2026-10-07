"""Bounded reference solvers. Evaluator side only.

These functions read hidden prerequisites. Nothing they return (plans,
lengths, eligibility) may enter learning code; an import-boundary test checks
that agent, representation, topology, skill and training modules never import
this module.

Exactness argument for ``reference_solve`` (monotone, deterministic, unit
costs, no consumption): any successful plan crafts every item of some
derivation tree of the goal at least once and gathers every base fact required
by the recipes it used, and the plan "gather the union of base facts, craft the
derivation bottom-up, submit" achieves exactly that cost. Enumerating all
consistent derivations (one recipe per crafted item) therefore yields the
optimum. With more derivations than ``max_derivations`` the result is a
constructive upper bound and is marked inexact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product

from .generator import Task
from .public_schema import build_actions
from .vocabulary import is_base


@dataclass(frozen=True)
class ReferenceResult:
    status: str  # optimal | upper_bound | unsolvable
    exact: bool
    length: int | None  # optimal length, or best constructive upper bound
    lower_bound: int
    upper_bound: int | None
    plan: tuple[str, ...]  # action keys
    work: int


Derivation = tuple[frozenset, frozenset]  # ({(item, rule_idx)}, {base types})


def _derive(task: Task, item: int, cap: int, memo: dict, stats: dict) -> list[Derivation]:
    if item in memo:
        return memo[item]
    if item in task.initial_true:
        memo[item] = [(frozenset(), frozenset())]
        return memo[item]
    out: dict[tuple, Derivation] = {}
    for ridx, recipe in enumerate(task.rules):
        if recipe.effect != item:
            continue
        own_bases = frozenset(b for b in recipe.true_base if b not in task.initial_true)
        child_sets = [_derive(task, x, cap, memo, stats) for x in recipe.item_inputs]
        for combo in product(*child_sets):
            choice: dict[int, int] = {item: ridx}
            bases = set(own_bases)
            consistent = True
            for pairs, cb in combo:
                for it, r in pairs:
                    if choice.setdefault(it, r) != r:
                        consistent = False
                        break
                if not consistent:
                    break
                bases |= cb
            stats["work"] += 1
            if not consistent:
                continue
            d = (frozenset(choice.items()), frozenset(bases))
            out[(d[0], d[1])] = d
    ranked = sorted(out.values(), key=lambda d: (len(d[0]) + len(d[1]), sorted(d[0]), sorted(d[1])))
    if len(ranked) > cap:
        stats["truncated"] = True
        ranked = ranked[:cap]
    memo[item] = ranked
    return ranked


def _lower_crafts(task: Task, item: int, memo: dict) -> int:
    if item in memo:
        return memo[item]
    if item in task.initial_true:
        memo[item] = 0
        return 0
    best = None
    for recipe in task.rules:
        if recipe.effect == item:
            inner = max((_lower_crafts(task, x, memo) for x in recipe.item_inputs), default=0)
            best = inner if best is None else min(best, inner)
    memo[item] = (best + 1) if best is not None else 10 ** 6
    return memo[item]


def _plan_from(task: Task, derivation: Derivation) -> tuple[str, ...]:
    actions = build_actions(task)
    by_fact = {a.fact: a.key for a in actions if a.fact is not None}
    by_rule = {a.rule: a.key for a in actions if a.rule is not None}
    pairs, bases = derivation
    plan = [by_fact[task.fact_index(b)] for b in sorted(bases, key=task.fact_index)]
    level = task.world.level_of
    for it, ridx in sorted(pairs, key=lambda p: (level(p[0]), task.fact_index(p[0]))):
        plan.append(by_rule[ridx])
    plan.append("submit")
    return tuple(plan)


def reference_solve(task: Task, max_derivations: int = 4096) -> ReferenceResult:
    stats = {"work": 0, "truncated": False}
    derivations = _derive(task, task.goal, max_derivations, {}, stats)
    lower = 1 + _lower_crafts(task, task.goal, {})
    if not derivations:
        return ReferenceResult("unsolvable", True, None, lower, None, (), stats["work"])
    best = derivations[0]
    length = len(best[0]) + len(best[1]) + 1
    exact = not stats["truncated"]
    return ReferenceResult(
        status="optimal" if exact else "upper_bound",
        exact=exact,
        length=length,
        lower_bound=length if exact else min(lower, length),
        upper_bound=length,
        plan=_plan_from(task, best),
        work=stats["work"],
    )


def bfs_shortest(task: Task, max_expansions: int = 200_000) -> tuple[int | None, str, int]:
    """Breadth-first search over fact configurations (deterministic dynamics).

    Returns (length including the final submit, status, expansions).
    """
    start = frozenset(task.initial_true)
    if task.goal in start:
        return 1, "optimal", 0
    bases = [ft for ft in task.fact_types if is_base(ft)]
    frontier = deque([(start, 0)])
    seen = {start}
    expansions = 0
    while frontier:
        state, dist = frontier.popleft()
        expansions += 1
        if expansions > max_expansions:
            return None, "limit", expansions
        successors = [state | {b} for b in bases if b not in state]
        for recipe in task.rules:
            if recipe.effect not in state and recipe.required() <= state:
                successors.append(state | {recipe.effect})
        for nxt in successors:
            if task.goal in nxt:
                return dist + 2, "optimal", expansions
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
    return None, "unsolvable", expansions
