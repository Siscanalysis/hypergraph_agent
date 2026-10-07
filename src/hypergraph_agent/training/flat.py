"""Trainer for agents choosing primitive operations (smoke, P1, P2).

One batch:
    freeze the topology snapshot of every world
    collect complete episodes until ``batch_steps`` primitive interactions
    PPO update on exactly those episodes (snapshot ids are checked)
    commit public evidence to the dependency model (structural edits, if any,
    happen here, outside the update)
Arms with the same seed see the same task sequence and dynamics seeds.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..agents.policy import ActorCritic
from ..config import eval_stream_config, ppo_config, train_stream_config
from ..envs.generator import TaskStream
from ..envs.recipequest import RecipeQuestEnv
from ..evaluation.metrics import aggregate
from ..evaluation.protocol import evaluate
from ..skills.executor import Executor
from ..topology.snapshot import TopologyManager
from .budget import BudgetMeter, SessionLedger
from .ppo import ppo_update
from .rollout import run_episode
from .run import RunContext
from .seeding import RNGStreams


def make_topology(cfg: dict, arm: dict) -> TopologyManager | None:
    if arm["topology"] == "none":
        return None
    t = cfg["topology"]
    eps = cfg["train"]["tasks"]["failure_prob"]
    return TopologyManager(eps, t["max_size"], t["credible_mass"], arm["topology"])


def run_id_for(cfg: dict, arm: dict, seed: int) -> str:
    return f"{cfg['run']['name']}-{arm['id']}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"


def collect_batch(policy, env, meter, streams, stream, task_i, topo, arm, ppo, batch_steps, ctx,
                  batch, purpose="exploration"):
    records, steps = [], 0
    while steps < batch_steps and meter.remaining > 0:
        task = stream.task(task_i)
        task_i += 1
        snap = topo.snapshot(task.world.world_key) if topo is not None else None
        ex = Executor(env, meter, purpose, streams.actions, topo=topo, topo_snapshot=snap,
                      graph_mode=arm["graph_mode"], use_posterior=arm["use_posterior"])
        rec = run_episode(policy, ex, task, streams.next_seed("dynamics"), gamma=ppo.gamma)
        records.append(rec)
        steps += rec.n_primitive
        ctx.episode({"batch": batch, **rec.summary()})
        if rec.status == "budget_exhausted":
            break
    return records, steps, task_i


def log_batch(ctx, batch, records, steps, meter, stats, topo, deltas, n_events_seen):
    ctx.event({"type": "batch", "batch": batch, "steps": steps, "used": meter.used,
               "episodes": len(records),
               "success_rate": float(np.mean([r.success for r in records])) if records else None,
               "mean_primitive_length": float(np.mean([r.n_primitive for r in records])) if records else None,
               **{f"ppo_{k}": v for k, v in stats.items()}})
    alphas = [a for r in records for a in r.alpha_means]
    if alphas:
        ctx.event({"type": "context_weight", "batch": batch, "alpha_mean": float(np.mean(alphas)),
                   "alpha_std": float(np.std(alphas)), "n": len(alphas)})
    for d in deltas:
        ctx.event(d.to_dict())
    if topo is not None:
        for e in topo.events[n_events_seen:]:
            ctx.event(e)
        return len(topo.events)
    return n_events_seen


def train_flat_arm(cfg: dict, arm: dict, seed: int, ledger: SessionLedger, runs_dir: str) -> dict:
    torch.set_num_threads(cfg["run"]["threads"])
    run_id = run_id_for(cfg, arm, seed)
    allocation = cfg["run"]["allocation"]
    budget = cfg["budget"]
    ctx = RunContext(runs_dir, run_id, {**cfg, "arm": arm}, phase=cfg["run"]["phase"],
                     arm=arm["id"], seed=seed)
    ledger.register_run(run_id, allocation, budget["per_run_interactions"],
                        {"phase": cfg["run"]["phase"], "arm": arm["id"], "seed": seed})
    meter = BudgetMeter(ledger, run_id, allocation, budget["per_run_interactions"])

    streams = RNGStreams(seed)
    streams.seed_torch_init()
    policy = ActorCritic(arm["encoder"], **cfg["model"])
    ppo = ppo_config(cfg)
    opt = torch.optim.Adam(policy.parameters(), lr=ppo.lr)
    topo = make_topology(cfg, arm)
    stream_cfg = train_stream_config(cfg, seed)
    stream = TaskStream(stream_cfg)
    env = RecipeQuestEnv()
    ctx.write_manifest(
        allocation=allocation, run_cap=meter.cap, parameters=policy.parameter_count(),
        recurrent_state=policy.hidden, train_stream_hash=stream_cfg.config_hash(),
        regimes={"profile": stream_cfg.task.profile, "world_mode": stream_cfg.world_mode,
                 "reward": "terminal_success_only", "masks": "none_beyond_syntax",
                 "curriculum": "none", "graph_mode": arm["graph_mode"],
                 "topology": arm["topology"], "use_posterior": arm["use_posterior"]},
        rng_streams=streams.state_dict())

    batch, task_i, n_seen, stop = 0, 0, 0, None
    try:
        while True:
            if ctx.elapsed > cfg["run"]["wallclock_s"]:
                stop = "wallclock_cap"
                break
            if meter.remaining <= 0:
                stop = "interaction_cap"
                break
            records, steps, task_i = collect_batch(policy, env, meter, streams, stream, task_i, topo,
                                                   arm, ppo, budget["batch_steps"], ctx, batch)
            if not records:
                stop = "interaction_cap"
                break
            expected = lambda r: ((topo.snapshot(r.world_key).snapshot_id if topo else None), None)
            stats = ppo_update(policy, opt, records, ppo, streams.np["minibatch"], expected)
            deltas = topo.commit() if topo is not None else []
            n_seen = log_batch(ctx, batch, records, steps, meter, stats, topo, deltas, n_seen)
            batch += 1
    finally:
        meter.flush()

    ckpt = {"policy": policy.state_dict(), "policy_config": policy.config, "arm": arm,
            "topology": topo.state_dict() if topo is not None else None,
            "rng": streams.state_dict(), "task_index": task_i, "batches": batch,
            "used": meter.used, "config": cfg}
    torch.save(ckpt, ctx.dir / "checkpoint.pt")
    n_deltas = len(topo.deltas) if topo is not None else 0

    # non-adaptive development evaluation (reporting budget)
    ev = cfg["eval"]
    rep = BudgetMeter(ledger, run_id + ":eval", allocation, budget["reporting_eval_per_run"],
                      kind="reporting")
    eval_streams = RNGStreams(seed, "eval")
    rows = evaluate(policy, arm, eval_stream_config(cfg), n_tasks=ev["n_tasks"], track=ev["track"],
                    meter=rep, action_gen=eval_streams.actions, dyn_rng=eval_streams.np["dynamics"],
                    greedy=ev["greedy"], episodes_per_world=ev["episodes_per_world"],
                    topo=topo if ev["track"] == "frozen" else None,
                    epsilon=cfg["train"]["tasks"]["failure_prob"], topo_cfg=cfg["topology"],
                    gamma=ppo.gamma, on_episode=lambda row: ctx._append("eval.jsonl", row))
    rep.flush()
    summary = {"eval": aggregate(rows), "eval_by_depth": aggregate(rows, by="depth")}
    ctx.finish("completed", stop_reason=stop, batches=batch,
               interactions={"adaptive_physical": meter.used, "by_purpose": dict(meter.by_purpose),
                             "reporting_eval": rep.used, "logical_pretraining_charge": 0},
               topology_deltas=n_deltas, checkpoint="checkpoint.pt",
               eval_summary=summary, session_budget=ledger.summary())
    return {"run_id": run_id, "arm": arm["id"], "seed": seed, **summary}
