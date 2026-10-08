"""Walker study runner (evaluator side).

For every seed, arm (strategy) and evaluation variant, a walker plays the
declared evaluation stream in world-major order, so beliefs build up within
each unseen world over its episodes and start from the prior in the next one.
Rows record the evaluator's reference length (never shown to agents).

The ``learned`` arm first collects evidence in training worlds with the
collector strategy (physical interactions, charged to a collector run and
logically to the arm), trains the edit policy offline on search problems from
that evidence, and is then evaluated like every other arm. A paired offline
benchmark of the walks themselves is run on held-out evidence.

``reference`` is privileged: it executes the evaluator's optimal plan.

Every walker is given the failure probability of the stream it plays (the
evaluation variant's). With ``eval_world_offset_by_seed`` each seed plays its
own evaluation worlds (base seed + 1000 x seed), so worlds are independent
units across seeds; noise variants of one seed play the same worlds.
"""

from __future__ import annotations

import copy
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..agents.edit_policy import EditPolicy, benchmark_search, search_problems, train_edit_policy
from ..agents.walker import STRATEGIES, Walker
from ..config import eval_stream_config, train_stream_config
from ..envs.generator import TaskStream, derive_seed
from ..envs.recipequest import RecipeQuestEnv
from ..envs.reference_solver import reference_solve
from ..skills.executor import Executor
from ..training.budget import BudgetExhausted, BudgetMeter, SessionLedger
from ..training.run import RunContext
from .metrics import aggregate

NON_ADAPTIVE = ("maximal", "reference")
LEARNED = ("learned", "learned_sample")
EVAL_SEED_STRIDE = 1000


def _true_state(ex: Executor) -> frozenset:
    """PRIVILEGED: the facts that are actually true, read from the environment."""
    return frozenset(f for f, p in ex.env._present.items() if p)


def reference_episode(ex: Executor, task, dyn_seed: int) -> dict:
    """PRIVILEGED: executes the optimal plan computed from the hidden rules.
    A step that leaves the true state unchanged has failed (every step of an
    optimal plan changes it); the reference then replans from the current
    true state, so under noise it retries the failed step at once."""
    ex.reset(task, dyn_seed)
    idx = ex.spec.action_index()
    plan = list(reference_solve(task).plan)
    status, replans = None, 0
    while not ex.terminated:
        if not ex.meter.can_charge(1):
            status = "budget_exhausted"
            break
        key = plan.pop(0) if plan else "submit"
        before = _true_state(ex)
        try:
            ex.primitive(idx[key])
        except BudgetExhausted:
            status = "budget_exhausted"
            break
        if not ex.terminated and (not plan or _true_state(ex) == before):
            plan = list(reference_solve(task, held=_true_state(ex)).plan)
            replans += 1
    return {"success": ex.success, "status": status or ex.termination, "primitive_length": ex.n_primitive,
            "noop_or_invalid": ex.n_noop, "manager_decisions": ex.n_primitive, "skill_calls": 0,
            "replans": replans, "evals": 0}


def eval_order(stream_cfg, n_tasks: int, episodes_per_world: int) -> list[int]:
    if stream_cfg.world_mode == "pool" and episodes_per_world > 1:
        nw = stream_cfg.n_worlds
        return [e * nw + w for w in range(nw) for e in range(episodes_per_world)][:n_tasks]
    return list(range(n_tasks))


def variant_config(cfg: dict, seed: int, variant: dict) -> dict:
    """The configuration one seed plays under one evaluation variant: the
    variant's task overrides, the seed's shared world
    (``shared_world_offset_by_seed``) and the seed's own evaluation worlds
    (``eval_world_offset_by_seed``)."""
    wc = cfg["walker"]
    vcfg = copy.deepcopy(cfg)
    vcfg["eval"]["tasks"] = {**vcfg["eval"]["tasks"], **variant.get("tasks", {})}
    if wc["shared_world_offset_by_seed"]:
        vcfg["train"]["shared_world_seed"] = cfg["train"]["shared_world_seed"] + seed
    if wc["eval_world_offset_by_seed"]:
        vcfg["eval"]["base_seed"] = cfg["eval"]["base_seed"] + EVAL_SEED_STRIDE * seed
    return vcfg


def stream_budget(cfg: dict, seed: int, variant: dict) -> int:
    """Sum of the task budgets of the evaluation stream one run plays: an upper
    bound on its interactions, since no episode exceeds its budget (generator
    only, no environment interaction)."""
    stream_cfg = eval_stream_config(variant_config(cfg, seed, variant))
    stream = TaskStream(stream_cfg)
    order = eval_order(stream_cfg, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"])
    return sum(stream.task(i).budget for i in order)


def make_walker(strategy: str, wc: dict, seed_key, epsilon: float, policy=None):
    """An ``agents.walker`` strategy, or else an ``agents.markov`` agent (study
    M), whose module is imported only when one of its strategies is asked for."""
    rng = np.random.default_rng(derive_seed("walker", *seed_key) % (2 ** 32))
    if strategy not in STRATEGIES:
        from ..agents.markov import MARKOV_STRATEGIES, make_markov_agent
        if set(MARKOV_STRATEGIES) & set(STRATEGIES):
            raise ValueError(f"Markov strategies shadow walker strategies: "
                             f"{sorted(set(MARKOV_STRATEGIES) & set(STRATEGIES))}")
        return make_markov_agent(strategy, rng, wc, epsilon)
    return Walker(strategy, rng,
                  epsilon=epsilon, max_evals=wc["max_evals"], temperature=wc["temperature"],
                  restart_after=wc["restart_after"], exact_cap=wc["exact_cap"], plan_cap=wc["plan_cap"],
                  policy=policy, consistent_moves=wc["consistent_moves"], omit_prob=wc["omit_prob"])


def load_policy(wc: dict, seed: int):
    """Reuse an edit policy trained in an earlier run (``walker.policy_from``,
    a glob with ``{seed}``); returns it with that run's collection cost and id."""
    matches = sorted(glob.glob(wc["policy_from"].format(seed=seed)))
    if len(matches) != 1:
        raise FileNotFoundError(f"policy_from matched {len(matches)} files for seed {seed}")
    path = Path(matches[0])
    policy = EditPolicy(wc["policy"]["hidden"])
    policy.load_state_dict(torch.load(path, weights_only=True))
    man = json.loads((path.parent / "manifest.json").read_text())
    return policy, man.get("policy_training"), man["interactions"]["physical"], man["run_id"]


def play(agent, strategy: str, stream, order: list[int], meter, seed: int, ctx: RunContext | None,
         purpose: str, extra: dict) -> list[dict]:
    env = RecipeQuestEnv()
    rows, per_world = [], {}
    for k, i in enumerate(order):
        if meter.remaining <= 0:
            break
        task = stream.task(i)
        ex = Executor(env, meter, purpose, torch.Generator().manual_seed(seed))
        dyn = derive_seed("dynamics", seed, i) % (2 ** 31)
        r = reference_episode(ex, task, dyn) if strategy == "reference" else agent.run_episode(ex, task, dyn)
        wk = task.world.world_key
        per_world[wk] = per_world.get(wk, -1) + 1
        ref = reference_solve(task)
        row = {**r, "order": k, "stream_index": i, "world_key": wk, "world_episode": per_world[wk],
               "task_key": task.task_key, "depth": task.depth, "budget": task.budget,
               "reference_length": ref.length, "reference_exact": ref.exact, **extra}
        rows.append(row)
        if ctx is not None:
            ctx._append("eval.jsonl", row)
        if r["status"] == "budget_exhausted":
            break
    return rows


def run_walker_study(cfg: dict, ledger: SessionLedger, runs_dir: str) -> list[dict]:
    wc = cfg["walker"]
    alloc = cfg["run"]["allocation"]
    variants = wc["eval_variants"] or [{"name": "default", "tasks": {}}]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for seed in cfg["run"]["seeds"]:
        policy, policy_stats, collector_used, collector_id = None, None, 0, None
        if any(a["strategy"] in LEARNED for a in cfg["arms"]):
            if wc["policy_from"]:
                policy, policy_stats, collector_used, collector_id = load_policy(wc, seed)
            else:
                policy, policy_stats, collector_used, collector_id, _ = _train_policy(
                    cfg, ledger, runs_dir, seed, stamp, train_stream_config(cfg, seed).task.failure_prob)
        for variant in variants:
            vcfg = variant_config(cfg, seed, variant)
            stream_cfg = eval_stream_config(vcfg)
            epsilon = stream_cfg.task.failure_prob  # the walker knows the noise of the stream it plays
            if wc["warmup_interactions"] and train_stream_config(vcfg, seed).task.failure_prob != epsilon:
                raise ValueError("warm-up and evaluation streams must declare the same failure probability")
            order = eval_order(stream_cfg, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"])
            for arm in cfg["arms"]:
                strategy = arm["strategy"]
                run_id = f"{cfg['run']['name']}-{arm['id']}-{variant['name']}-s{seed}-{stamp}"
                ctx = RunContext(runs_dir, run_id, {**vcfg, "arm": arm, "variant": variant},
                                 phase=cfg["run"]["phase"], arm=arm["id"], seed=seed,
                                 parent_run=collector_id if strategy in LEARNED else None)
                kind = "reporting" if strategy in NON_ADAPTIVE else "adaptive"
                cap = wc["per_run_cap"]
                ledger.register_run(run_id, alloc, cap, {"strategy": strategy, "variant": variant["name"]})
                if strategy in LEARNED and collector_used:
                    ledger.logical_charge(run_id, collector_id, collector_used, "evidence collected to train the edit policy")
                meter = BudgetMeter(ledger, run_id, alloc, cap, kind=kind)
                agent = None if strategy == "reference" else make_walker(
                    strategy, wc, (seed, arm["id"], variant["name"]), epsilon, policy)
                warm = 0
                try:
                    if wc["warmup_interactions"] and strategy not in NON_ADAPTIVE:
                        tstream = TaskStream(train_stream_config(vcfg, seed))
                        wm = BudgetMeter(ledger, run_id, alloc, wc["warmup_interactions"], kind="adaptive")
                        play(agent, strategy, tstream, list(range(10_000)), wm, seed, None, "exploration", {})
                        wm.flush()
                        warm = wm.used
                    rows = play(agent, strategy, TaskStream(stream_cfg), order, meter, seed, ctx,
                                "reporting_eval" if kind == "reporting" else "exploration",
                                {"strategy": strategy, "variant": variant["name"], "failure_prob": epsilon})
                finally:
                    meter.flush()
                per_task = stream_cfg.world_mode == "per_task" and cfg["eval"]["episodes_per_world"] > 1
                summary = _summarize_rows(rows, cfg["eval"]["episodes_per_world"] if per_task else None)
                if policy is not None and strategy == wc["benchmark_source"] and agent is not None:
                    # paired offline comparison of the walks on held-out evidence (no interactions)
                    summary["search_benchmark"] = run_search_benchmark(cfg, policy, agent.worlds.values(), seed)
                    ctx.event({"type": "search_benchmark", **summary["search_benchmark"]})
                ctx.finish("completed", stop_reason="tasks_done" if len(rows) == len(order) else "interaction_cap",
                           interactions={"physical": meter.used + warm, "warmup": warm, "evaluation": meter.used,
                                         "kind": kind,
                                         "logical_policy_evidence": collector_used if strategy in LEARNED else 0},
                           policy_training=policy_stats if strategy in LEARNED else None,
                           policy_source=collector_id if strategy in LEARNED else None,
                           eval_summary=summary, session_budget=ledger.summary())
                results.append({"run_id": run_id, "arm": arm["id"], "strategy": strategy,
                                "variant": variant["name"], "seed": seed, **summary})
                print(f"{run_id}: success {summary['all']['success_rate']:.2f} "
                      f"cost/opt {summary['cost_ratio']:.2f} n={summary['all']['n']}", flush=True)
    return results


def _summarize_rows(rows: list[dict], block: int | None = None) -> dict:
    """Run summary. Episodes are indexed within their world, or, with
    ``block`` (a fresh world per task), by position within consecutive blocks
    of ``block`` tasks, which then replace worlds."""
    pos = (lambda r: r["order"] % block) if block else (lambda r: r["world_episode"])
    unit = (lambda r: f"block{r['order'] // block}") if block else (lambda r: r["world_key"])
    by_ep: dict = {}
    for r in rows:
        by_ep.setdefault(pos(r), []).append(r)
    total = sum(r["primitive_length"] for r in rows)
    opt = sum(r["reference_length"] for r in rows if r["reference_length"])

    def ratio(rs):
        o = sum(x["reference_length"] or 0 for x in rs)
        return sum(x["primitive_length"] for x in rs) / o if o else float("nan")

    half = (max((pos(r) for r in rows), default=0) + 1) // 2
    late = [r for r in rows if pos(r) >= half]
    worlds = sorted({unit(r) for r in rows})
    return {
        "all": aggregate(rows),
        "cost_ratio": total / opt if opt else float("nan"),
        "half_split_episode": half,
        "cost_ratio_early": ratio([r for r in rows if pos(r) < half]),
        "cost_ratio_late": ratio(late),
        "success_late": float(np.mean([r["success"] for r in late])) if late else float("nan"),
        "cost_ratio_late_by_world": {w: ratio([r for r in late if unit(r) == w]) for w in worlds},
        "by_world_episode": {str(k): {"n": len(v), "success": float(np.mean([x["success"] for x in v])),
                                      "cost_ratio": float(sum(x["primitive_length"] for x in v)
                                                          / max(1, sum(x["reference_length"] or 0 for x in v)))}
                             for k, v in sorted(by_ep.items())},
        "search_evaluations": int(sum(r.get("evals", 0) for r in rows)),
        "replans": int(sum(r.get("replans", 0) for r in rows)),
    }


def _train_policy(cfg, ledger, runs_dir, seed, stamp, epsilon):
    wc = cfg["walker"]
    alloc = cfg["run"]["allocation"]
    cid = f"{cfg['run']['name']}-collector-s{seed}-{stamp}"
    ctx = RunContext(runs_dir, cid, cfg, phase=cfg["run"]["phase"], arm="collector", seed=seed)
    ledger.register_run(cid, alloc, wc["collector_interactions"], {"strategy": wc["collector"]})
    meter = BudgetMeter(ledger, cid, alloc, wc["collector_interactions"])
    collector = make_walker(wc["collector"], wc, (seed, "collector"), epsilon)
    tstream_cfg = train_stream_config(cfg, seed)
    order = eval_order(tstream_cfg, tstream_cfg.n_worlds * wc["collector_episodes_per_world"],
                       wc["collector_episodes_per_world"])
    try:
        rows = play(collector, wc["collector"], TaskStream(tstream_cfg), order, meter, seed, ctx,
                    "exploration", {"strategy": wc["collector"], "variant": "collection"})
    finally:
        meter.flush()
    problems = search_problems(collector.worlds.values())
    torch.manual_seed(derive_seed("edit_policy", seed) % (2 ** 31))
    policy = EditPolicy(wc["policy"]["hidden"])
    stats = train_edit_policy(policy, problems, iters=wc["policy"]["iters"], max_evals=wc["max_evals"],
                              temperature=wc["temperature"], restart_after=wc["restart_after"],
                              lr=wc["policy"]["lr"], rng=np.random.default_rng(derive_seed("train_policy", seed)),
                              log=ctx.event)
    torch.save(policy.state_dict(), ctx.dir / "edit_policy.pt")
    ctx.finish("completed", interactions={"physical": meter.used, "kind": "adaptive"},
               collected={"episodes": len(rows), "worlds": len(collector.worlds), "problems": len(problems)},
               policy_training=stats, session_budget=ledger.summary())
    return policy, stats, meter.used, cid, collector


def run_search_benchmark(cfg: dict, policy, evidence_worlds, seed: int) -> dict:
    wc = cfg["walker"]
    problems = search_problems(evidence_worlds)
    return benchmark_search(problems, ("uniform", "focused", "learned"), policy=policy,
                            starts=wc["benchmark_starts"], seed=seed, max_evals=wc["max_evals"],
                            temperature=wc["temperature"], restart_after=wc["restart_after"])
