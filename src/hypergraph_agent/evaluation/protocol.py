"""Evaluation tracks.

``frozen``     weights, library, contracts and topology are fixed; no evidence
               is recorded or committed. Within-episode recurrence and context
               gates operate normally.
``inference``  weights and library fixed; public evidence updates task/world
               scoped beliefs between episodes of the same world under the
               declared interaction budget. New worlds start from the prior.
``continual``  cross-task adaptation with parameter updates and admissions:
               not implemented in this version (raises).

Evaluation tasks come from a fixed base seed, identical for every arm and
seed (paired design). Reference lengths are computed here, on the evaluator
side, and are never passed to agents.
"""

from __future__ import annotations

import copy

import numpy as np

from ..envs.generator import TaskStream
from ..envs.recipequest import RecipeQuestEnv
from ..envs.reference_solver import reference_solve
from ..skills.executor import Executor
from ..topology.snapshot import TopologyManager
from ..training.budget import BudgetExhausted
from ..training.rollout import run_episode

TRACKS = ("frozen", "inference", "continual")


def evaluate(policy, arm: dict, stream_cfg, *, n_tasks: int, track: str, meter, action_gen,
             dyn_rng: np.random.Generator, greedy: bool = False, episodes_per_world: int = 1,
             topo: TopologyManager | None = None, epsilon: float = 0.0, topo_cfg: dict | None = None,
             library=None, lib_snapshot=None, live_contracts: bool = False, gamma: float = 1.0,
             purpose: str = "reporting_eval", on_episode=None) -> list[dict]:
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    if track == "continual":
        raise NotImplementedError("the continual cross-task adaptation track is not implemented")
    stream = TaskStream(stream_cfg)
    order = list(range(n_tasks))
    if stream_cfg.world_mode == "pool" and episodes_per_world > 1:
        nw = stream_cfg.n_worlds
        order = [e * nw + w for w in range(nw) for e in range(episodes_per_world)][:n_tasks]

    uses_topology = arm.get("topology", "none") != "none"
    if uses_topology:
        topo_cfg = topo_cfg or {}
        if topo is not None:
            ev_topo = TopologyManager(epsilon, topo_cfg.get("max_size", 3),
                                      topo_cfg.get("credible_mass", 0.9), topo.mode)
            ev_topo.load_state_dict(copy.deepcopy(topo.state_dict()))
        else:
            mode = arm["topology"]
            ev_topo = TopologyManager(epsilon, topo_cfg.get("max_size", 3),
                                      topo_cfg.get("credible_mass", 0.9), mode)
    else:
        ev_topo = None

    env = RecipeQuestEnv()
    results = []
    for k, i in enumerate(order):
        task = stream.task(i)
        snap = ev_topo.snapshot(task.world.world_key) if ev_topo is not None else None
        ex = Executor(env, meter, purpose, action_gen,
                      topo=ev_topo if (track == "inference" and ev_topo is not None) else None,
                      topo_snapshot=snap, graph_mode=arm["graph_mode"],
                      use_posterior=arm.get("use_posterior", True), library=library,
                      lib_snapshot=lib_snapshot, live_contracts=live_contracts)
        if not meter.can_charge(1):
            break
        rec = run_episode(policy, ex, task, int(dyn_rng.integers(0, 2 ** 31 - 1)),
                          record=False, greedy=greedy, gamma=gamma)
        if track == "inference" and ev_topo is not None:
            ev_topo.commit(reason="evaluation_inference")
        ref = reference_solve(task)
        row = {
            **rec.summary(), "order": k, "stream_index": i, "track": track, "greedy": greedy,
            "reference_length": ref.length, "reference_exact": ref.exact,
            "reference_lower_bound": ref.lower_bound,
            "budget": task.budget, "n_facts": len(task.fact_types), "n_rules": len(task.rules),
            "calls": rec.call_log,
        }
        results.append(row)
        if on_episode is not None:
            on_episode(row)
        if rec.status == "budget_exhausted":
            break
    return results


__all__ = ["evaluate", "TRACKS", "BudgetExhausted"]
