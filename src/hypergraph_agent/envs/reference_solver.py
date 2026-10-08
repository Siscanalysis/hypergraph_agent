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

from .derivations import (
    Derivation, StructRecipe, crafting_order, enumerate_derivations, lower_bound_crafts, plan_cost,
)
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


def true_structure(task: Task) -> list[StructRecipe]:
    return [StructRecipe(r.effect, r.item_inputs, r.true_base) for r in task.rules]


def _plan_from(task: Task, recipes, derivation: Derivation) -> tuple[str, ...]:
    actions = build_actions(task)
    by_fact = {a.fact: a.key for a in actions if a.fact is not None}
    by_rule = {a.rule: a.key for a in actions if a.rule is not None}
    pairs, bases = derivation
    plan = [by_fact[task.fact_index(b)] for b in sorted(bases, key=task.fact_index)]
    plan += [by_rule[ridx] for _, ridx in crafting_order(pairs, recipes)]
    plan.append("submit")
    return tuple(plan)


def reference_solve(task: Task, max_derivations: int = 4096, held=None) -> ReferenceResult:
    """Optimal plan from ``held``, the facts already true (default: the task's
    initial state). The evaluator passes the current true state to replan
    after a failed step under noise."""
    recipes = true_structure(task)
    held = frozenset(task.initial_true if held is None else held)
    derivations, truncated, work = enumerate_derivations(recipes, held, task.goal, max_derivations)
    lower = 1 + lower_bound_crafts(recipes, held, task.goal)
    if not derivations:
        return ReferenceResult("unsolvable", True, None, lower, None, (), work)
    best = derivations[0]
    length = plan_cost(best)
    exact = not truncated
    return ReferenceResult(
        status="optimal" if exact else "upper_bound",
        exact=exact,
        length=length,
        lower_bound=length if exact else min(lower, length),
        upper_bound=length,
        plan=_plan_from(task, recipes, best),
        work=work,
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
