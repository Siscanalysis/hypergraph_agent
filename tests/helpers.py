"""Hand-built worlds, tasks and scripted policies for deterministic tests.

Everything here is a test fixture. Scripted policies exercise mechanics only
and are never learned results.
"""

from __future__ import annotations

import torch

from hypergraph_agent.envs.generator import (
    PROFILE_KNOWN, Recipe, Task, TaskConfig, TaskStream, TaskStreamConfig, World, WorldConfig,
)
from hypergraph_agent.envs.vocabulary import TYPE_NAMES
from hypergraph_agent.training.budget import BudgetMeter, SessionLedger

T = {name: i for i, name in enumerate(TYPE_NAMES)}


def recipe(effect, items=(), pool=(), true=(), variant=0):
    pool = tuple(sorted(T[p] for p in pool)) if pool else tuple(sorted(T[t] for t in true))
    return Recipe(T[effect], tuple(sorted(T[i] for i in items)), pool,
                  tuple(sorted(T[t] for t in true)), variant)


def world(recipes, levels, key="Wtest", seed=0):
    return World(key, seed, WorldConfig(n_levels=max(levels.values()), items_per_level=2),
                 tuple(sorted((T[k], v) for k, v in levels.items())), tuple(recipes))


def task(w, goal, facts, initial=(), budget=20, profile=PROFILE_KNOWN, failure_prob=0.0,
         order=None, rule_order=None, key="Ttest", prefix=""):
    fact_types = tuple(T[f] for f in facts)
    rule_ids = list(range(len(w.recipes)))
    if rule_order is not None:
        rule_ids = [rule_ids[i] for i in rule_order]
    rules = tuple(w.recipes[i] for i in rule_ids)
    if order is not None:
        fact_types = tuple(fact_types[i] for i in order)
    return Task(
        task_key=key, namespace="train", seed=0, world=w, profile=profile, goal=T[goal],
        depth=w.level_of(T[goal]), fact_types=fact_types,
        fact_keys=tuple(f"{prefix}f{TYPE_NAMES[t]}" for t in fact_types),
        rules=rules,
        rule_keys=tuple(f"{prefix}r{i}_{TYPE_NAMES[w.recipes[i].effect]}" for i in rule_ids),
        initial_true=frozenset(T[f] for f in initial), budget=budget, failure_prob=failure_prob,
    )


def chain_world():
    """ingot <= {ore, fuel, furnace_ready}; key <= ingot + {mould_ready};
    alternative key <= ingot + {wax}."""
    return world(
        [recipe("ingot", (), ("ore", "fuel", "furnace_ready", "sand", "wood", "clay"),
                ("ore", "fuel", "furnace_ready")),
         recipe("key", ("ingot",), ("mould_ready", "wax", "sand", "wood", "clay", "fiber"),
                ("mould_ready",)),
         recipe("key", ("ingot",), ("wax", "salt", "sand", "wood", "clay", "fiber"), ("wax",), 1)],
        {"ingot": 1, "key": 2})


CHAIN_FACTS = ("ore", "fuel", "furnace_ready", "sand", "wood", "clay", "mould_ready", "wax",
               "fiber", "salt", "ingot", "key")


def small_stream(namespace="train", seed=3, profile=PROFILE_KNOWN, depth=(1, 2), mode="per_task",
                 n_worlds=0, failure_prob=0.0, pool_size=4, levels=3):
    return TaskStream(TaskStreamConfig(
        namespace, seed, mode, n_worlds, 0,
        WorldConfig(n_levels=levels, items_per_level=2, pool_size=pool_size, max_true=2,
                    arity_weights=(0.5, 0.5)),
        TaskConfig(profile=profile, depth_min=depth[0], depth_max=depth[1],
                   failure_prob=failure_prob, n_distractor_base=0)))


def meter(cap=10_000, kind="adaptive"):
    led = SessionLedger(None)
    return BudgetMeter(led, "test", "diagnostics", cap, kind=kind)


class ScriptedPolicy:
    """scripted_fixture: picks the first available candidate whose key starts
    with one of ``prefs`` (in order); ``wait`` otherwise. Not a learned policy."""

    hidden = 4
    label = "scripted_fixture"

    def __init__(self, prefs):
        self.prefs = list(prefs)

    def initial_state(self):
        return torch.zeros(self.hidden)

    def step(self, struct, present, mem, h):
        keys = list(struct.cand_keys)
        choice = None
        for p in self.prefs:
            for i, k in enumerate(keys):
                if k.startswith(p) and not _done(struct, present, i):
                    choice = i
                    break
            if choice is not None:
                break
        if choice is None:
            choice = keys.index("wait") if "wait" in keys else 0
        logits = torch.full((len(keys),), -20.0)
        logits[choice] = 20.0
        return logits, torch.tensor(0.0), h, None


def _done(struct, present, i):
    kind, idx = int(struct.cand_node_kind[i]), int(struct.cand_node_idx[i])
    if kind == 1:
        return bool(present[idx])
    if kind == 2:
        return bool(present[int(struct.rule_effect[idx])])
    return False
