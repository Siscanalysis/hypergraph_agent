"""Optimal plans for a monotone AND/OR recipe structure.

This module contains no task data. The caller passes a structure, a list of
``StructRecipe(effect, item inputs, base requirement)``, the facts already
held and the goal. The evaluator passes the true structure (reference
solver); an agent passes its own hypothesis (hypergraph walker).

For monotone, deterministic, unit-cost dynamics the cheapest plan crafts every
item of one consistent derivation (one recipe per crafted item) once, gathers
the union of their base requirements once and then submits, so enumerating
consistent derivations yields the optimum. More than ``cap`` derivations per
item makes the result a constructive upper bound (``truncated``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class StructRecipe:
    effect: int
    items: tuple[int, ...]
    bases: tuple[int, ...]


Derivation = tuple[frozenset, frozenset]  # ({(item, recipe index)}, {base types to acquire})


def enumerate_derivations(recipes, held: frozenset, goal: int, cap: int = 4096):
    """Consistent derivations of ``goal``, cheapest first.

    Returns (derivations, truncated, work)."""
    memo: dict[int, list] = {}
    stats = {"work": 0, "truncated": False}

    def derive(item: int, stack: frozenset) -> list:
        if item in memo:
            return memo[item]
        if item in held:
            return [(frozenset(), frozenset())]
        if item in stack:
            # item inputs are public and levelled, so structures here are acyclic;
            # this guard only rules out infinite recursion on malformed input
            return []
        out: dict[tuple, Derivation] = {}
        for ridx, r in enumerate(recipes):
            if r.effect != item:
                continue
            own = frozenset(b for b in r.bases if b not in held)
            children = [derive(x, stack | {item}) for x in r.items]
            for combo in product(*children):
                choice = {item: ridx}
                bases = set(own)
                ok = True
                for pairs, cb in combo:
                    for it, rr in pairs:
                        if choice.setdefault(it, rr) != rr:
                            ok = False
                            break
                    if not ok:
                        break
                    bases |= cb
                stats["work"] += 1
                if ok:
                    d = (frozenset(choice.items()), frozenset(bases))
                    out[d] = d
        ranked = sorted(out.values(), key=lambda d: (len(d[0]) + len(d[1]), sorted(d[0]), sorted(d[1])))
        if len(ranked) > cap:
            stats["truncated"] = True
            ranked = ranked[:cap]
        memo[item] = ranked
        return ranked

    ds = derive(goal, frozenset())
    return ds, stats["truncated"], stats["work"]


def crafting_order(pairs: frozenset, recipes) -> list[tuple[int, int]]:
    """Order the crafts of a derivation so every input is crafted first."""
    choice = dict(pairs)
    order, seen = [], set()

    def visit(it: int):
        if it in seen or it not in choice:
            return
        seen.add(it)
        for x in recipes[choice[it]].items:
            visit(x)
        order.append((it, choice[it]))

    for it in sorted(choice):
        visit(it)
    return order


def lower_bound_crafts(recipes, held: frozenset, goal: int) -> int:
    memo: dict[int, int] = {}

    def lb(item: int, stack: frozenset) -> int:
        if item in held:
            return 0
        if item in memo:
            return memo[item]
        best = None
        for r in recipes:
            if r.effect == item and item not in stack:
                inner = max((lb(x, stack | {item}) for x in r.items), default=0)
                best = inner if best is None else min(best, inner)
        memo[item] = (best + 1) if best is not None else 10 ** 6
        return memo[item]

    return lb(goal, frozenset())


def plan_cost(derivation: Derivation) -> int:
    return len(derivation[0]) + len(derivation[1]) + 1
