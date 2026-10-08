"""The repository's agents on study X's streams (evaluator side).

RecipeQuest agents run through ``evaluation.walkers.play`` exactly as in the
walker studies (``focused_sample``, ``maximal`` and the privileged
``reference``), plus ``random``, defined here: uniform over the task's gather,
activate and craft actions, submitting as soon as the goal is observed held
(the RecipeQuest counterpart of TechTree's ``RandomAgent``). TechTree agents
run through ``techtree.study.play`` with the agent module of study U
(``explorers``) or L (``layered``) and the common random numbers of those
studies (one agent seed per test seed, shared by every arm).

Rows are normalized to the row format of study X (``row``), which learners'
evaluation runs share.
"""

from __future__ import annotations

import numpy as np

from ..envs.generator import derive_seed
from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from ..training.budget import BudgetExhausted
from .sources import evaluation_source

ADAPTIVE_AGENTS = ("focused_sample", "pooled", "isolated", "remember", "promote", "sham_promote", "adaptive",
                   "none")
REPORTING_AGENTS = ("maximal", "reference", "random", "oracle", "oracle_library")


class RandomRecipeAgent:
    """Uniform over the task's gather, activate and craft actions; submits as soon as the goal is
    observed held. Public information only (the executor's public spec and observations)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        ex.reset(task, dyn_seed)
        spec = ex.spec
        submit = spec.action_index()["submit"]
        moves = [i for i, a in enumerate(spec.actions) if a.kind in (GATHER, ACTIVATE, CRAFT)]
        status = None
        while not ex.terminated:
            a = submit if ex.obs.present[spec.goal] else int(self.rng.choice(moves))
            try:
                if not ex.meter.can_charge(1):
                    raise BudgetExhausted("random agent budget")
                ex.primitive(a)
            except BudgetExhausted:
                status = "budget_exhausted"
                break
        return {"success": ex.success, "status": status or ex.termination, "primitive_length": ex.n_primitive,
                "noop_or_invalid": ex.n_noop, "manager_decisions": ex.n_primitive, "skill_calls": 0}


def agent_kind(agent: str) -> str:
    if agent in ADAPTIVE_AGENTS:
        return "adaptive"
    if agent in REPORTING_AGENTS:
        return "reporting"
    raise KeyError(f"unknown agent {agent!r}")


def row(base: dict, *, arm: str, kind: str, split: str, test_seed: int, **extra) -> dict:
    """A row of study X: one task played by one arm."""
    return {"arm": arm, "kind": kind, "split": split, "test_seed": test_seed,
            "unit": f"s{test_seed}:{base['world_key']}", "world_key": base["world_key"],
            "world_episode": base["world_episode"], "order": base["order"], "success": bool(base["success"]),
            "status": base["status"], "primitive_length": base["primitive_length"],
            "reference_length": base["reference_length"], "reference_exact": base["reference_exact"],
            "task_budget": base.get("task_budget", base.get("budget")), **extra}


class _ListStream:
    """The ``task(i)`` interface of ``TaskStream`` over the items of an evaluation source."""

    def __init__(self, items):
        self._tasks = {it.order: it.task for it in items}

    def task(self, i: int):
        return self._tasks[i]


def _items(cfg: dict, split: str, test_seed: int) -> list:
    sp, setting = cfg[split], cfg["setting"]
    src = evaluation_source(setting["family"], setting, sp["namespace"], sp["base_seed"], test_seed,
                            sp["n_worlds"], sp["episodes_per_world"])
    items = []
    while (block := src.next_block()) is not None:
        items += block
    return items


def play_recipequest(cfg: dict, agent: str, test_seed: int, split: str, meter) -> list[dict]:
    """The stream of one test seed in world-major order; ``play`` derives each task's dynamics seed from
    (test seed, item order) exactly as the wrappers do."""
    from ..config import DEFAULTS as RQ_DEFAULTS
    from ..evaluation.walkers import make_walker, play
    items = _items(cfg, split, test_seed)
    wc = {**RQ_DEFAULTS["walker"], **cfg["agent_params"]}
    epsilon = items[0].task.failure_prob if items else 0.0
    if agent == "random":
        a = RandomRecipeAgent(np.random.default_rng(derive_seed("x_random", test_seed) % 2 ** 32))
    elif agent == "reference":
        a = None
    else:
        a = make_walker(agent, wc, (test_seed, agent, "x"), epsilon)
    purpose = "reporting_eval" if agent_kind(agent) == "reporting" else "exploration"
    rows = play(a, agent, _ListStream(items), [it.order for it in items], meter, test_seed, None, purpose, {})
    extra = ("replans", "evals", "contradictions", "pool_candidates", "omitted_candidates")
    return [row(r, arm=agent, kind="agent", split=split, test_seed=test_seed, depth=r["depth"],
                stream_index=r["stream_index"], **{k: r[k] for k in extra if k in r}) for r in rows]


def play_techtree(cfg: dict, agent: str, test_seed: int, split: str, meter) -> list[dict]:
    from ..techtree.explorers import make_explorer
    from ..techtree.generator import reference_solve
    from ..techtree.layered import make_layered
    from ..techtree.study import play
    setting = cfg["setting"]
    tasks = [it.task for it in _items(cfg, split, test_seed)]
    refs = [reference_solve(t) for t in tasks]
    study = setting["agent_study"]
    rng = np.random.default_rng(derive_seed("techtree_agent", study, test_seed) % 2 ** 32)
    factory = make_explorer if study == "U" else make_layered
    a = None if agent == "reference" else factory(agent, rng)
    rows = play(a, agent, tasks, refs, meter, None, {}, ())
    return [row(r, arm=agent, kind="agent", split=split, test_seed=test_seed, goal_level=r["goal_level"],
                world_steps_before=r["world_steps_before"], fired=r.get("fired", []),
                discoveries_new=r.get("discoveries_new")) for r in rows]
