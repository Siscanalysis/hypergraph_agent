"""Study PG arms on RecipeQuest: a policy graph over public states (public information only).

The agent sees what every walker sees: the executor's public spec and observations. It never reads
the task object, which it only passes to the executor's reset.

A node is (goal type, the set of fact types observed present, the craft actions made since that
set last changed). Under goal-only observation crafting an intermediate item leaves the observation
unchanged, so the third part, a memory of the agent's own recent crafts, is what tells the state
before an invisible craft from the state after it; it is reset whenever the observation changes.
Actions are identified across the tasks of a world by public, stable keys: gathering or activating
a fact type, and crafting a recipe signature (with an occurrence index when a task lists one
signature twice). The actions of a node are the task's gather, activate and craft actions, as for
the random agent of study X; ``wait`` is never chosen and ``submit`` is pressed as soon as the goal
is observed held.

Distributions, credit (loop erasure of each successful episode in node space, then one count per
(node, action) left on the path) and the three arms (``pg`` predictive Thompson sampling,
``pg_uniform`` the uniform control, ``pg_greedy`` an action of maximal count) are those of the
TechTree arms; the environment-independent core is shared with ``techtree.policygraph``.

Common random numbers: the uniform variates of an episode come from a stream seeded by the
episode's dynamics seed (the same for every arm of a run seed, since the runner derives it from the
seed and the stream position) and a fixed tag, so the three arms act identically in a world until
its first success. Graphs are kept per world and dropped with it.
"""

from __future__ import annotations

import numpy as np

from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from ..techtree.policygraph import MODES, LoopErasedPath, WorldGraph
from ..training.budget import BudgetExhausted

PG_STRATEGIES: tuple[str, ...] = tuple(MODES)  # pg, pg_uniform, pg_greedy
ALPHA0 = 1.0
STREAM_TAG = 0x5047  # separates the agent's stream from anything else seeded by the dynamics seed


def action_keys(spec) -> tuple[list[int], list[tuple]]:
    """(action indices, stable keys) of the task's gather, activate and craft actions."""
    idx, keys, seen = [], [], {}
    for i, a in enumerate(spec.actions):
        if a.kind in (GATHER, ACTIVATE):
            key = ("gather" if a.kind == GATHER else "activate", spec.facts[a.fact].type_id)
        elif a.kind == CRAFT:
            sig = spec.rules[a.rule].signature
            seen[sig] = seen.get(sig, -1) + 1
            key = ("craft", sig, seen[sig])
        else:
            continue
        idx.append(i)
        keys.append(key)
    return idx, keys


def observed(spec, obs) -> frozenset:
    return frozenset(spec.facts[i].type_id for i, p in enumerate(obs.present) if p)


class RecipePolicyGraph:
    def __init__(self, strategy: str, rng: np.random.Generator, epsilon: float = 0.0):
        if strategy not in PG_STRATEGIES:
            raise ValueError(f"unknown policy-graph strategy {strategy!r}")
        self.strategy, self.mode, self.epsilon = strategy, MODES[strategy], epsilon
        self.rng = rng  # unused: the episode streams give common random numbers across arms
        self.worlds: dict[str, WorldGraph] = {}

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        ex.reset(task, dyn_seed)
        spec = ex.spec
        if abs(spec.failure_prob - self.epsilon) > 1e-12:
            raise ValueError(f"agent epsilon {self.epsilon} differs from the task's declared "
                             f"failure probability {spec.failure_prob}")
        g = self.worlds.setdefault(spec.world_key, WorldGraph(ALPHA0))
        g.episodes += 1
        rng = np.random.default_rng([int(dyn_seed), STREAM_TAG])
        idx, keys = action_keys(spec)
        actions = tuple(keys)
        index_of = dict(zip(keys, idx))
        submit = spec.action_index()["submit"]
        goal = spec.facts[spec.goal].type_id
        seen = observed(spec, ex.obs)
        crafts: frozenset = frozenset()
        node = (goal, seen, crafts)
        path = LoopErasedPath()
        decisions = informed = 0
        status = None
        while not ex.terminated:
            if ex.obs.present[spec.goal]:
                a, key = submit, None
            else:
                key, had = g.choose(node, actions, self.mode, float(rng.random()))
                a = index_of[key]
                decisions += 1
                informed += had
                path.visit(node, key)
            try:
                if not ex.meter.can_charge(1):
                    raise BudgetExhausted("policy-graph budget")
                ex.primitive(a)
            except BudgetExhausted:
                status = "budget_exhausted"
                break
            if key is None:
                continue
            now = observed(spec, ex.obs)
            if now != seen:
                crafts = frozenset()
            elif key[0] == "craft":
                crafts = crafts | {key}
            nxt = (goal, now, crafts)
            g.observe(node, key, nxt)
            seen, node = now, nxt
        credited = g.credit(path) if ex.success else 0
        return {"success": ex.success, "status": status or ex.termination, "primitive_length": ex.n_primitive,
                "noop_or_invalid": ex.n_noop, "manager_decisions": ex.n_primitive, "skill_calls": 0,
                "pg_decisions": decisions, "pg_informed": informed, "pg_path": len(path.steps),
                "pg_credited": credited, **{f"pg_{k}": v for k, v in g.counters().items()}}


def make_policy_graph_agent(strategy: str, rng: np.random.Generator, epsilon: float = 0.0):
    return RecipePolicyGraph(strategy, rng, epsilon)
