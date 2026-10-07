"""Phase 3: shared pretraining, then forked repertoire x revision arms.

Per seed block:
  pretraining (executed once, charged logically to every arm):
    manager exploration with primitives (dependency revision on)
    one discovery round (practice, validation, admission/rejection)
    manager training with the resulting library
  fork the identical checkpoint into each arm with its own post-fork RNG streams:
    F0 fixed library, frozen dependency structure, frozen contracts
    F1 fixed library, evidence-driven dependency revision, live contracts
    G0 growing library (further discovery rounds), frozen structure and contracts
    G1 growing library, revision, live contracts
  evaluate every arm on held-out longer compositions (frozen track)

Within a manager batch the library, controllers and topology snapshot are
fixed; structural edits and admissions happen between batches only.
"""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from ..agents.policy import ActorCritic
from ..config import eval_stream_config, ppo_config, train_stream_config
from ..envs.generator import TaskConfig, TaskStream, derive_seed
from ..envs.recipequest import RecipeQuestEnv
from ..evaluation.metrics import aggregate
from ..evaluation.protocol import evaluate
from ..skills import discovery as disc
from ..skills.executor import Executor
from ..skills.library import SkillLibrary
from ..topology.snapshot import TopologyManager
from .budget import BudgetMeter, SessionLedger
from .ppo import ppo_update
from .rollout import run_episode
from .run import RunContext
from .seeding import RNGStreams

SKILL_DEFAULTS = {
    "explore_before_discovery": 3000, "discovery_budget": 4500, "max_proposals": 8,
    "max_trained_per_round": 3, "max_admissions_per_round": 2, "max_admitted": 24, "max_depth": 3,
    "practice_budget": 1200, "practice_timeout": 12, "practice_batch_steps": 200,
    "validation_episodes": 10, "bc_epochs": 20, "controller_encoder": "gated",
    "admission": {"profile": "diagnostic", "min_episodes": 8, "min_success": 0.5,
                  "min_primitive_steps": 2},
    "arm_discovery_after": 2000, "arm_discovery_budget": 2500, "max_recent_episodes": 400,
    "val_goal_levels": [2, 3, 4], "val_tasks": 20,
    # learned: closed-loop controllers; macro / macro_retry: matched open-loop control
    "mode": "learned", "macro_retries": 3,
}


def skill_cfg(cfg: dict) -> dict:
    out = copy.deepcopy(SKILL_DEFAULTS)
    for k, v in cfg.get("skills", {}).items():
        if k not in SKILL_DEFAULTS:
            raise KeyError(f"unknown skills key {k!r}")
        out[k] = {**out[k], **v} if isinstance(v, dict) else v
    return out


class P3State:
    def __init__(self, manager, opt, topo, library, streams, task_i=0, recent=None, round_idx=0):
        self.manager, self.opt, self.topo, self.library = manager, opt, topo, library
        self.streams, self.task_i, self.recent, self.round_idx = streams, task_i, recent or [], round_idx

    def fork(self, label: str) -> "P3State":
        manager = copy.deepcopy(self.manager)
        opt = torch.optim.Adam(manager.parameters(), lr=self.opt.param_groups[0]["lr"])
        opt.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        topo = TopologyManager(self.topo.epsilon, self.topo.max_size, self.topo.credible_mass, self.topo.mode)
        topo.load_state_dict(copy.deepcopy(self.topo.state_dict()))
        library = copy.deepcopy(self.library)
        library._cache = {}
        return P3State(manager, opt, topo, library, self.streams.fork(label), self.task_i,
                       list(self.recent), self.round_idx)

    def hashes(self) -> dict:
        h = disc.controller_hash(self.manager.state_dict())
        return {"manager": h, "library_id": self.library.snapshot().library_id,
                "topology": {s: snap.snapshot_id for s, snap in self.topo._snapshots.items()}}


def manager_batch(state: P3State, cfg, meter, ctx, env, stream, batch, purpose, live_contracts,
                  use_library=True):
    ppo = ppo_config(cfg)
    lib_snap = state.library.snapshot() if use_library else None
    records, steps = [], 0
    while steps < cfg["budget"]["batch_steps"] and meter.remaining > 0:
        task = stream.task(state.task_i)
        state.task_i += 1
        snap = state.topo.snapshot(task.world.world_key)
        ex = Executor(env, meter, purpose, state.streams.actions, topo=state.topo, topo_snapshot=snap,
                      graph_mode="active", use_posterior=True, library=state.library,
                      lib_snapshot=lib_snap, live_contracts=live_contracts)
        rec = run_episode(state.manager, ex, task, state.streams.next_seed("dynamics"), gamma=ppo.gamma)
        records.append(rec)
        steps += rec.n_primitive
        ctx.episode({"batch": batch, "purpose": purpose, **rec.summary(), "calls": rec.call_log})
        if rec.status == "budget_exhausted":
            break
    if not records:
        return []
    expected = lambda r: (state.topo.snapshot(r.world_key).snapshot_id,
                          lib_snap.library_id if lib_snap is not None else None)
    stats = ppo_update(state.manager, state.opt, records, ppo, state.streams.np["minibatch"], expected)
    deltas = state.topo.commit()
    calls = [c for r in records for c in r.call_log]
    ctx.event({"type": "batch", "batch": batch, "purpose": purpose, "steps": steps, "used": meter.used,
               "episodes": len(records), "success_rate": float(np.mean([r.success for r in records])),
               "mean_primitive_length": float(np.mean([r.n_primitive for r in records])),
               "mean_manager_decisions": float(np.mean([r.n_decisions for r in records])),
               "skill_calls": len([c for c in calls if c["depth"] == 1]),
               "skill_call_success": sum(1 for c in calls if c["status"] == "target_success"),
               "library_id": lib_snap.library_id if lib_snap is not None else None,
               **{f"ppo_{k}": v for k, v in stats.items()}})
    for d in deltas:
        ctx.event(d.to_dict())
    state.recent = (state.recent + records)[-skill_cfg(cfg)["max_recent_episodes"]:]
    return records


def discovery_round(state: P3State, cfg, meter, ctx, stream, budget: int, live_contracts: bool,
                    seed: int) -> dict:
    sk = skill_cfg(cfg)
    ppo = ppo_config(cfg)
    lib = state.library
    n_events = len(lib.events)
    frags = disc.mine_fragments(state.recent, max_level=lib.max_depth)
    cands = disc.propose(frags, lib, state.round_idx, sk["max_proposals"])
    world = stream.world(0)
    task_cfg = stream.cfg.task
    practice_tasks = disc.TargetTasks(world, task_cfg, "practice", seed)
    val_tasks = disc.TargetTasks(world, task_cfg, "skillval", seed)
    start = meter.used
    per_cand = sk["practice_budget"] + sk["validation_episodes"] * sk["practice_timeout"]
    trained = admitted = 0
    ctx.event({"type": "discovery_round", "round": state.round_idx, "fragments": len(frags),
               "successful_fragments": sum(f.success for f in frags), "proposals": len(cands)})
    for cand in cands:
        if trained >= sk["max_trained_per_round"] or meter.used - start + per_cand > budget \
                or meter.remaining < per_cand:
            lib.defer(cand.cand_id, "not trained: per-round training or budget cap")
            continue
        if sk["mode"] != "learned":
            admitted += _macro_candidate(state, cand, frags, lib, sk, val_tasks, meter, ctx,
                                         admitted < sk["max_admissions_per_round"])
            trained += 1
            continue
        torch.manual_seed(derive_seed("controller", seed, cand.cand_id) % (2 ** 31))
        controller = ActorCritic(sk["controller_encoder"], **cfg["model"])
        cand.status = "training"
        lib_snap = lib.snapshot()
        env = RecipeQuestEnv()

        def ex_factory():
            return Executor(env, meter, "skill_practice", state.streams.actions, topo=None,
                            graph_mode="active", library=lib, lib_snapshot=lib_snap,
                            live_contracts=live_contracts)

        if sk["bc_epochs"] > 0:
            cand.bc_updates = disc.behaviour_clone(controller, cand, frags, state.recent, ex_factory,
                                                   "active", state.topo, True, lib_snap,
                                                   epochs=sk["bc_epochs"], lr=ppo.lr)
        log = ctx.event
        ps = disc.practice(controller, cand, practice_tasks, meter, state.streams, state.topo, "active",
                           True, lib, lib_snap, live_contracts, sk["practice_timeout"],
                           sk["practice_budget"], sk["practice_batch_steps"], ppo, log)
        val = disc.validate(controller, cand, val_tasks, meter, state.streams, state.topo, "active",
                            True, lib, lib_snap, live_contracts, sk["practice_timeout"],
                            sk["validation_episodes"])
        cand.practice_cost, cand.validation_cost = ps["interactions"], val["interactions"]
        cand.validation = {k: v for k, v in val.items() if k != "episodes"}
        ctx.event({"type": "skill_validation", "cand_id": cand.cand_id, "skill_key": cand.skill_key,
                   "bc_updates": cand.bc_updates, "practice": ps, **cand.validation,
                   "episodes": val["episodes"]})
        trained += 1
        ok, reason = disc.admission_decision(val, sk["admission"])
        if ok and admitted >= sk["max_admissions_per_round"]:
            ok, reason = False, "per-round admission cap"
        if ok:
            spec_probe = next((r.spec for r in reversed(state.recent)
                               if cand.target_type in r.spec.fact_by_type()), None)
            snap = state.topo.snapshot(world.world_key)
            spec = disc.build_spec(cand, controller, lib, sk["practice_timeout"], val, ps, snap,
                                   spec_probe, sk["admission"]["profile"])
            lib.admit(cand.cand_id, spec, controller.state_dict(), controller.config)
            admitted += 1
        else:
            lib.reject(cand.cand_id, reason)
    state.topo.commit(reason="after_discovery_round")
    for e in lib.events[n_events:]:
        ctx.event(e)
    state.round_idx += 1
    return {"round": state.round_idx - 1, "proposals": len(cands), "trained": trained,
            "admitted": admitted, "interactions": meter.used - start}


def _macro_candidate(state, cand, frags, lib, sk, val_tasks, meter, ctx, may_admit: bool) -> int:
    """Matched-macro control: same mined data, open-loop execution, same criteria."""
    steps = disc.macro_steps(cand, frags)
    if steps is None:
        lib.reject(cand.cand_id, "no primitive-only successful fragment for a macro")
        return 0
    placeholder = ActorCritic("incidence", d=8, hidden=8)  # never executed for macro skills
    retries = sk["macro_retries"] if sk["mode"] == "macro_retry" else 0
    spec = disc.SkillSpec(
        cand.skill_key, lib.next_version(cand.skill_key), 1, cand.target_type, sk["practice_timeout"],
        (), disc.controller_hash(placeholder.state_dict()), (), 0.0, float(len(steps)),
        json.dumps({"cand_id": cand.cand_id, "proposer": "heuristic_fragment_mining",
                    "controller": f"{sk['mode']} (open loop, not learned)"}),
        0, "{}", admission_profile=sk["admission"]["profile"],
        macro=tuple((int(k), int(t)) for k, t in steps), macro_retries=retries)
    val = disc.validate_macro(spec, val_tasks, meter, state.streams, state.topo, "active",
                              sk["validation_episodes"])
    cand.validation_cost = val["interactions"]
    cand.validation = {k: v for k, v in val.items() if k != "episodes"}
    ctx.event({"type": "skill_validation", "cand_id": cand.cand_id, "skill_key": cand.skill_key,
               "controller": sk["mode"], **cand.validation, "episodes": val["episodes"]})
    ok, reason = disc.admission_decision(val, sk["admission"])
    if ok and not may_admit:
        ok, reason = False, "per-round admission cap"
    if not ok:
        lib.reject(cand.cand_id, reason)
        return 0
    from dataclasses import replace
    spec = replace(spec, success_est=val["success_rate"],
                   duration_est=val["mean_tau_success"] or float(spec.timeout),
                   training_cost=val["interactions"],
                   validation=json.dumps(cand.validation))
    lib.admit(cand.cand_id, spec, placeholder.state_dict(), placeholder.config)
    return 1


def _evaluate_arm(state, cfg, arm_id, live, ledger, run_id, seed, ctx, namespace, goal_levels, n_tasks):
    ev = dict(cfg["eval"])
    sc = copy.deepcopy(cfg)
    sc["eval"]["goal_levels"] = goal_levels
    stream_cfg = eval_stream_config(sc, namespace)
    rep = BudgetMeter(ledger, f"{run_id}:eval-{namespace}", cfg["run"]["allocation"],
                      cfg["budget"]["reporting_eval_per_run"] // 2, kind="reporting")
    es = RNGStreams(seed, f"eval/{arm_id}/{namespace}")
    arm = {"graph_mode": "active", "topology": state.topo.mode, "use_posterior": True}
    rows = evaluate(state.manager, arm, stream_cfg, n_tasks=n_tasks, track="frozen", meter=rep,
                    action_gen=es.actions, dyn_rng=es.np["dynamics"], greedy=ev["greedy"],
                    topo=state.topo, epsilon=state.topo.epsilon, topo_cfg=cfg["topology"],
                    library=state.library, lib_snapshot=state.library.snapshot(),
                    live_contracts=live, on_episode=lambda row: ctx._append(f"eval_{namespace}.jsonl", row))
    rep.flush()
    return rows, rep.used


def run_phase3(cfg: dict, ledger: SessionLedger, runs_dir: str) -> list[dict]:
    torch.set_num_threads(cfg["run"]["threads"])
    sk = skill_cfg(cfg)
    budget = cfg["budget"]
    alloc = cfg["run"]["allocation"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for seed in cfg["run"]["seeds"]:
        c = copy.deepcopy(cfg)
        c["train"]["shared_world_seed"] = cfg["train"]["shared_world_seed"] + seed
        stream_cfg = train_stream_config(c, seed)
        stream = TaskStream(stream_cfg)
        env = RecipeQuestEnv()

        # ---------------- shared pretraining (physical, once per seed block)
        pre_id = f"{cfg['run']['name']}-pretrain-s{seed}-{stamp}"
        pre_ctx = RunContext(runs_dir, pre_id, c, phase="p3", arm="pretrain", seed=seed)
        ledger.register_run(pre_id, alloc, budget["pretraining_interactions"], {"phase": "p3", "arm": "pretrain"})
        meter = BudgetMeter(ledger, pre_id, alloc, budget["pretraining_interactions"])
        streams = RNGStreams(seed, "p3")
        streams.seed_torch_init()
        manager = ActorCritic(cfg["arms"][0].get("encoder", "gated"), **cfg["model"])
        opt = torch.optim.Adam(manager.parameters(), lr=ppo_config(cfg).lr)
        topo = TopologyManager(stream_cfg.task.failure_prob, cfg["topology"]["max_size"],
                               cfg["topology"]["credible_mass"], "revise")
        library = SkillLibrary(sk["max_admitted"], sk["max_depth"])
        state = P3State(manager, opt, topo, library, streams)
        pre_ctx.write_manifest(allocation=alloc, run_cap=meter.cap, train_stream_hash=stream_cfg.config_hash(),
                               regimes={"profile": stream_cfg.task.profile, "world_mode": "shared",
                                        "failure_prob": stream_cfg.task.failure_prob,
                                        "skill_practice_objective": "public target predicate, reward 1"},
                               skills_config=sk)
        batch, stop = 0, None
        wall = cfg["run"]["wallclock_s"]
        try:
            while meter.used < sk["explore_before_discovery"] and meter.remaining > 0:
                if pre_ctx.elapsed > wall:
                    stop = "wallclock_cap"
                    break
                manager_batch(state, c, meter, pre_ctx, env, stream, batch, "pretraining", True,
                              use_library=False)
                batch += 1
            if stop is None:
                rnd = discovery_round(state, c, meter, pre_ctx, stream, sk["discovery_budget"], True, seed)
                pre_ctx.event({"type": "discovery_summary", **rnd})
            while stop is None and meter.remaining > 0:
                if pre_ctx.elapsed > wall:
                    stop = "wallclock_cap"
                    break
                if not manager_batch(state, c, meter, pre_ctx, env, stream, batch, "pretraining", True):
                    break
                batch += 1
            stop = stop or "interaction_cap"
        finally:
            meter.flush()
        pre_dir = pre_ctx.dir
        library.save(pre_dir / "library")
        torch.save({"manager": manager.state_dict(), "manager_config": manager.config,
                    "topology": topo.state_dict(), "rng": streams.state_dict(), "task_index": state.task_i},
                   pre_dir / "checkpoint.pt")
        fork_hashes = state.hashes()
        pre_used = meter.used
        pre_ctx.finish("completed", stop_reason=stop, batches=batch, fork_hashes=fork_hashes,
                       interactions={"adaptive_physical": pre_used, "by_purpose": dict(meter.by_purpose)},
                       library={"version": library.version, "admitted": [list(s.ref) for s in library.available()],
                                "candidates": len(library.candidates)},
                       session_budget=ledger.summary())

        # ---------------- forked arms
        for arm in cfg["arms"]:
            arm_state = state.fork(arm["id"])
            assert arm_state.hashes()["manager"] == fork_hashes["manager"]
            run_id = f"{cfg['run']['name']}-{arm['id']}-s{seed}-{stamp}"
            ctx = RunContext(runs_dir, run_id, {**c, "arm": arm}, phase="p3", arm=arm["id"], seed=seed,
                             parent_run=pre_id)
            ledger.register_run(run_id, alloc, budget["per_run_interactions"], {"phase": "p3", "arm": arm["id"]})
            ledger.logical_charge(run_id, pre_id, pre_used, "shared P3 pretraining prefix")
            m = BudgetMeter(ledger, run_id, alloc, budget["per_run_interactions"])
            revision, growth = bool(arm.get("revision")), bool(arm.get("growth"))
            if not revision:
                arm_state.topo.freeze_structure()
            ctx.write_manifest(allocation=alloc, run_cap=m.cap, fork_hashes=fork_hashes,
                               arm_definition={"growth": growth, "revision": revision,
                                               "live_contracts": revision},
                               logical_pretraining_charge=pre_used)
            b, did_discovery, arm_stop = 0, False, "interaction_cap"
            try:
                while m.remaining > 0:
                    if ctx.elapsed > wall:
                        arm_stop = "wallclock_cap"
                        break
                    if growth and not did_discovery and m.used >= sk["arm_discovery_after"]:
                        rnd = discovery_round(arm_state, c, m, ctx, stream, sk["arm_discovery_budget"],
                                              revision, seed)
                        ctx.event({"type": "discovery_summary", **rnd})
                        did_discovery = True
                        continue
                    if not manager_batch(arm_state, c, m, ctx, env, stream, b, "exploration", revision):
                        break
                    b += 1
            finally:
                m.flush()
            arm_state.library.save(ctx.dir / "library")
            torch.save({"manager": arm_state.manager.state_dict(), "manager_config": arm_state.manager.config,
                        "topology": arm_state.topo.state_dict(), "arm": arm, "rng": arm_state.streams.state_dict(),
                        "config": c},
                       ctx.dir / "checkpoint.pt")
            test_rows, rep_t = _evaluate_arm(arm_state, c, arm["id"], revision, ledger, run_id, seed, ctx,
                                             "test", cfg["eval"]["goal_levels"], cfg["eval"]["n_tasks"])
            val_rows, rep_v = _evaluate_arm(arm_state, c, arm["id"], revision, ledger, run_id, seed, ctx,
                                            "val", sk["val_goal_levels"], sk["val_tasks"])
            summary = {"eval_test": aggregate(test_rows), "eval_test_by_depth": aggregate(test_rows, by="depth"),
                       "eval_val": aggregate(val_rows)}
            lib = arm_state.library
            ctx.finish("completed", stop_reason=arm_stop, batches=b,
                       interactions={"adaptive_physical": m.used, "by_purpose": dict(m.by_purpose),
                                     "logical_pretraining_charge": pre_used,
                                     "all_in_logical": pre_used + m.used,
                                     "reporting_eval": rep_t + rep_v},
                       library={"version": lib.version, "admitted": [list(s.ref) for s in lib.available()],
                                "candidates": {cid: {"key": cr.skill_key, "status": cr.status,
                                                     "reason": cr.reason}
                                               for cid, cr in lib.candidates.items()}},
                       topology_deltas=len(arm_state.topo.deltas), eval_summary=summary,
                       session_budget=ledger.summary())
            results.append({"run_id": run_id, "arm": arm["id"], "seed": seed, "parent": pre_id,
                            "all_in_logical": pre_used + m.used, **summary})
            print(f"{run_id}: test success {summary['eval_test'].get('success_rate', float('nan')):.3f} "
                  f"val success {summary['eval_val'].get('success_rate', float('nan')):.3f}", flush=True)
    return results
