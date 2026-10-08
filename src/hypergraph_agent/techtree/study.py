"""TechTree study runner (evaluator side).

For every seed, variant and arm, an agent plays its stream of
``n_worlds x episodes_per_world`` tasks in world-major order: each world is
visited for consecutive episodes, so agents with memory amortize discovery
within a world, and the next world starts from nothing. Each seed has its own
worlds; worlds are the independent units of the analysis. Every arm of a seed
and variant sees the same tasks. A variant may run a subset of the arms.

Agents receive only ``env.agent_interface``'s ``PublicEnv`` and opaque task
tokens. Rows record the privileged reference length and whether the goal needs
an unlocked slot (never shown to agents). Oracle arms (``oracle``,
``oracle_library``) and the phase-level ``reveal_levels`` receive links read
from the hidden world here, through ``Explorer.reveal``: a declared, identical
supply of discoveries, not an information leak through the environment.
``reference`` replays the evaluator's optimal plan.

Common random numbers: every arm and variant of a seed builds its agent from
the same seed, and explorers derive their per-world symbol preference from a
value drawn first, so paired contrasts are not inflated by unrelated
tie-breaking.

Interactions: arms that adapt from evidence are charged as ``adaptive``
(purpose ``exploration``); non-adaptive arms (``random``, the oracles,
``reference``) as ``reporting``. A run's cap is the sum of its episode
budgets, so no run is cut short. Every run records ``run_hash``, a hash of
what determines its rows (study, name, world, stream, variant, arm, seed).

Repeats (``check_repeats``): a measurement config refuses to start when any of
its runs already exists complete with the same run hash, unless ``resume``, which
skips complete runs and repeats only missing or incomplete ones; a resume is
refused when an incomplete run it would repeat was made by other code (other
loaded package code or commit), because the analysis would reject that rerun.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..envs.generator import derive_seed
from ..training.budget import BudgetMeter, SessionLedger
from ..training.run import LOADED_CODE_SHA256, RunContext, process_source
from .config import merged, run_hash, stream_config, variant_arms, variants
from .env import TechTreeEnv, agent_interface
from .explorers import Episode, make_explorer
from .generator import TechStream, needs_unlock, reference_solve, revealed_links
from .layered import make_layered

NON_ADAPTIVE = ("random", "oracle", "oracle_library", "reference")
ORACLES = ("oracle", "oracle_library")


def make_agent(study: str, arm: str, rng: np.random.Generator):
    return make_explorer(arm, rng) if study == "U" else make_layered(arm, rng)


def make_ledger(cfg: dict) -> SessionLedger:
    led = cfg["ledger"]
    ledger = SessionLedger(cfg["run"]["ledger"], dict(led["allocations"]), led["adaptive_cap"],
                           led["reporting_cap"])
    if cfg["run"]["allocation"] not in ledger.state["allocations"]:
        raise KeyError(f"allocation {cfg['run']['allocation']!r} is not in {cfg['run']['ledger']}")
    return ledger


def run_cap(cfg: dict, variant: dict) -> int:
    _, s = merged(cfg, variant)
    return s["n_worlds"] * s["episodes_per_world"] * s["episode_budget"]


def plan(cfg: dict) -> dict:
    """Runs, caps and totals of a config; touches nothing."""
    seeds = cfg["run"]["seeds"]
    rows, adaptive, reporting, n_runs = [], 0, 0, 0
    for v in variants(cfg):
        w, s = merged(cfg, v)
        arms = variant_arms(cfg, v)
        cap = run_cap(cfg, v)
        n_rep = sum(a in NON_ADAPTIVE for a in arms)
        adaptive += cap * (len(arms) - n_rep) * len(seeds)
        reporting += cap * n_rep * len(seeds)
        n_runs += len(arms) * len(seeds)
        rows.append({"variant": v["name"], "arms": arms, "per_run_cap": cap, "worlds_per_seed": s["n_worlds"],
                     "episodes_per_world": s["episodes_per_world"], "episode_budget": s["episode_budget"],
                     "namespace": s["namespace"], "reveal_levels": s["reveal_levels"],
                     "factors": factors(w, s)})
    return {"study": cfg["run"]["study"], "name": cfg["run"]["name"], "allocation": cfg["run"]["allocation"],
            "ledger": cfg["run"]["ledger"], "seeds": seeds, "arms": cfg["arms"], "variants": rows,
            "runs": n_runs, "worlds_per_variant": len(seeds) * rows[0]["worlds_per_seed"] if rows else 0,
            "caps": {"adaptive_total": adaptive, "reporting_total": reporting},
            "ledger_on_creation": cfg["ledger"]}


def factors(world: dict, stream: dict) -> dict:
    return {"combo_length": world["combo_length"], "reuse_depth": world["reuse_depth"],
            "n_unlocks": world["n_unlocks"], "unlock_visibility": stream["unlock_visibility"],
            "signal": stream["signal"], "reveal_levels": list(stream["reveal_levels"])}


def reference_episode(env, token, meter, ref) -> dict:
    """PRIVILEGED: replays the evaluator's shortest plan."""
    ep = Episode(env, token, meter)
    for a in ref.plan:
        if ep.done:
            break
        ep.step(a)
    return ep.row()


def play(agent, arm: str, tasks: list, refs: list, meter, ctx, extra: dict, reveal_levels) -> list[dict]:
    env, token_for = agent_interface(TechTreeEnv())
    rows, world_steps = [], defaultdict(int)
    for i, (task, ref) in enumerate(zip(tasks, refs)):
        if meter.remaining <= 0:
            break
        wk = task.world.world_key
        if agent is not None and arm in ORACLES:
            agent.reveal(wk, revealed_links(task.world))
        elif agent is not None and reveal_levels:
            agent.reveal(wk, revealed_links(task.world, reveal_levels))
        t0 = time.perf_counter()
        token = token_for(task)
        if arm == "reference":
            r = reference_episode(env, token, meter, ref)
        else:
            r = agent.run_episode(env, token, meter)
        row = {**r, "order": i, "world_index": task.world_index, "world_key": wk,
               "world_episode": task.world_episode, "task_key": task.task_key,
               "goal_level": task.world.levels[task.goal], "budget": task.budget,
               "reference_length": ref.length, "reference_exact": ref.exact,
               "needs_unlock": needs_unlock(task.world, task.goal), "world_steps_before": world_steps[wk],
               "episode_s": round(time.perf_counter() - t0, 5), **extra}
        world_steps[wk] += r["primitive_length"]
        rows.append(row)
        if ctx is not None:
            ctx._append("eval.jsonl", row)
        if r["status"] == "budget_exhausted":
            break
    return rows


def check_repeats(cfg: dict, runs_dir: str, resume: bool) -> set[str]:
    """Run hashes to skip; raises when the config may not start (see the module docstring)."""
    from .analysis import collect_runs
    existing = collect_runs(Path(runs_dir), cfg["run"]["name"])
    done = {r["run_hash"] for r in existing if r["complete"]}
    planned = {run_hash(cfg, v, a, seed) for seed in cfg["run"]["seeds"] for v in variants(cfg)
               for a in variant_arms(cfg, v)}
    if cfg["run"]["requires_flag"] and not resume and planned & done:
        raise RuntimeError(f"{len(planned & done)} runs of this measurement config already exist complete; "
                           f"use --resume to repeat only missing or incomplete runs")
    if resume:
        code = [LOADED_CODE_SHA256, process_source(Path(__file__).resolve().parents[3])["commit"]]
        stale = [r["run_id"] for r in existing
                 if r["run_hash"] in planned - done and not r["complete"] and r["code"] != code]
        if stale:
            raise RuntimeError(f"incomplete runs {stale} were made by other code; a rerun would be refused "
                               f"by the analysis and needs a dated amendment")
    return done if resume else set()


def run_study(cfg: dict, ledger: SessionLedger, runs_dir: str, resume: bool = False) -> list[dict]:
    study, alloc = cfg["run"]["study"], cfg["run"]["allocation"]
    done = check_repeats(cfg, runs_dir, resume)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for seed in cfg["run"]["seeds"]:
        for variant in variants(cfg):
            arms = [a for a in variant_arms(cfg, variant) if run_hash(cfg, variant, a, seed) not in done]
            if not arms:
                continue
            scfg, reveal = stream_config(cfg, variant, seed)
            stream = TechStream(scfg)
            tasks = [stream.task(w, e) for w, e in stream.order()]
            refs = [reference_solve(t) for t in tasks]
            w, s = merged(cfg, variant)
            for arm in arms:
                rh = run_hash(cfg, variant, arm, seed)
                run_id = f"{cfg['run']['name']}-{arm}-{variant['name']}-s{seed}-{stamp}"
                ctx = RunContext(runs_dir, run_id, {**cfg, "variant": variant, "arm": arm},
                                 phase=f"techtree_{study}", arm=arm, seed=seed)
                ctx.write_manifest(run_hash=rh, variant=variant["name"], expected_tasks=len(tasks))
                kind = "reporting" if arm in NON_ADAPTIVE else "adaptive"
                cap = run_cap(cfg, variant)
                ledger.register_run(run_id, alloc, cap, {"arm": arm, "variant": variant["name"], "kind": kind,
                                                         "run_hash": rh})
                meter = BudgetMeter(ledger, run_id, alloc, cap, kind=kind)
                # common random numbers: every arm and variant of a seed starts from the same agent seed
                rng = np.random.default_rng(derive_seed("techtree_agent", study, seed) % 2 ** 32)
                agent = None if arm == "reference" else make_agent(study, arm, rng)
                extra = {"arm": arm, "variant": variant["name"], "seed": seed, **factors(w, s)}
                t0 = time.perf_counter()
                try:
                    rows = play(agent, arm, tasks, refs, meter, ctx, extra, reveal)
                except Exception as exc:
                    meter.flush()
                    ctx.finish("failed", error=repr(exc), interactions={"physical": meter.used, "kind": kind})
                    raise
                finally:
                    meter.flush()
                summary = summarize_rows(rows, cfg["analysis"]["late_from"] or s["episodes_per_world"] // 2)
                summary["wallclock_s"] = round(time.perf_counter() - t0, 2)
                stop = "tasks_done" if len(rows) == len(tasks) else "interaction_cap"
                ctx.finish("completed", stop_reason=stop,
                           interactions={"physical": meter.used, "kind": kind}, eval_summary=summary,
                           session_budget=ledger.summary())
                results.append({"run_id": run_id, "arm": arm, "variant": variant["name"], "seed": seed,
                                "run_hash": rh, **summary})
                print(f"{run_id}: success {summary['success']:.2f} steps/ep {summary['steps_mean']:.1f} "
                      f"late cost/opt {summary['cost_ratio_late']:.2f} n={summary['n']} "
                      f"({summary['wallclock_s']:.1f}s)", flush=True)
    return results


def summarize_rows(rows: list[dict], late_from: int) -> dict:
    if not rows:
        return {"n": 0, "success": float("nan"), "steps_mean": float("nan"), "cost_ratio_late": float("nan")}

    def ratio(rs):
        ref = sum(r["reference_length"] for r in rs)
        return sum(r["primitive_length"] for r in rs) / ref if ref else float("nan")

    late = [r for r in rows if r["world_episode"] >= late_from]
    early = [r for r in rows if r["world_episode"] < late_from]
    return {
        "n": len(rows), "success": float(np.mean([r["success"] for r in rows])),
        "steps_mean": float(np.mean([r["primitive_length"] for r in rows])),
        "steps_total": int(sum(r["primitive_length"] for r in rows)),
        "cost_ratio": ratio(rows), "cost_ratio_early": ratio(early), "cost_ratio_late": ratio(late),
        "success_late": float(np.mean([r["success"] for r in late])) if late else float("nan"),
        "late_from": late_from, "episode_s_mean": float(np.mean([r["episode_s"] for r in rows])),
        "failure_status": {s: sum(1 for r in rows if not r["success"] and r["status"] == s)
                           for s in sorted({r["status"] for r in rows if not r["success"]})},
    }


def results_path(cfg: dict) -> Path:
    return Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
