"""Runner of study X (evaluator side): training, evaluation and agent runs.

Jobs
----
- ``train``: one learner and training seed. The learner trains on worlds of
  the training namespace (``sources.training_source``, ``train.n_envs``
  environments in one process) and saves the acting policy at every budget of
  ``train.budgets``; it stops at the largest, so it executes exactly that many
  interactions, all charged ``adaptive`` (purpose ``exploration``).
- ``eval``: one checkpoint (learner, training seed, budget), one split
  (``test`` at every budget; ``validation``, when declared, at the largest
  budget only) and one action mode. The frozen policy plays every seed's
  stream of the split in world-major order; references are computed here for
  the rows. Charged ``reporting``, except for a recurrent learner in trial mode,
  whose state adapts within a world: ``adaptive``.
- ``agent``: one repository agent and test seed on the same test stream
  (``agents``); adaptive agents are charged ``adaptive``, the others
  ``reporting``. Agents named in ``validation_agents`` (study XS) also play every
  seed of the validation stream.

A learner with ``shaping`` (study XS) trains with the potential-based term of
``shaping.py`` added to its rewards (one ``Shaper`` per environment, at the
learner's discount); its evaluation runs are unchanged.

Every run has a run directory (``training.run.RunContext``: manifest with
provenance, config, ``eval.jsonl`` rows, ``events.jsonl``), is registered in the
ledger with a cap, and charges every primitive step to a ``BudgetMeter``. A
training run's cap is its largest budget; an evaluation run's cap is the sum of
the task budgets of the streams it plays, so no cap can cut a run short.

Repeats: a job whose hash (``config.train_hash``, ``eval_hash``,
``agent_hash``) already has a completed run is skipped; a measurement config
(``run.requires_flag``) refuses to start when any of its planned jobs already
has a completed run, unless ``resume``. Runs are deterministic given the code
and the hash; an incomplete run is repeated as a new run.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..envs.generator import derive_seed
from ..training.budget import BudgetMeter, SessionLedger
from ..training.run import RunContext
from .agents import agent_kind, play_recipequest, play_techtree, row
from .config import agent_hash, eval_hash, train_hash
from .encoders import RQLayout, TTLayout
from .learners import (
    RECURRENT, Actor, build_model, checkpoint_callback, library_versions, load_model, rnd_vec_env,
)
from .shaping import learner_gamma, make_shaper
from .sources import evaluation_source, reference, stream_budget, training_source, tt_world_config
from .wrappers import SourceExhausted, make_gym

PHASE = "baselines_X"
SPLITS = ("test", "validation")


# ------------------------------------------------------------- planning
def layout_for(cfg: dict):
    s = cfg["setting"]
    if s["family"] == "recipequest":
        return RQLayout(s["max_variants"])
    w = tt_world_config(s)
    return TTLayout(sum(w.concepts_per_level), w.n_levels, w.n_primitives + w.n_slots,
                    max(w.length(lv) for lv in range(1, w.n_levels + 1)))


def split_seeds(cfg: dict, split: str) -> list[int]:
    return list(cfg[split]["seeds"]) if cfg[split]["namespace"] else []


def stream_sum(cfg: dict, split: str, seed: int) -> int:
    sp = cfg[split]
    return stream_budget(cfg["setting"]["family"], cfg["setting"], sp["namespace"], sp["base_seed"], seed,
                         sp["n_worlds"], sp["episodes_per_world"])


def eval_kind(learner: dict) -> str:
    return "adaptive" if learner["algo"] in RECURRENT and learner["trial"] else "reporting"


def jobs(cfg: dict, learners=None, seeds=None, agents=None) -> dict:
    """Every job of a config (optionally a subset), with its hash and cap."""
    keep_l = set(learners) if learners else None
    keep_s = set(seeds) if seeds is not None else None
    sums = {(sp, s): stream_sum(cfg, sp, s) for sp in SPLITS for s in split_seeds(cfg, sp)}
    out = {"train": [], "eval": [], "agent": []}
    top = max(cfg["train"]["budgets"])
    evaluation_only = bool(cfg["run"]["checkpoints_from"])
    for ln in cfg["learners"]:
        if keep_l is not None and ln["id"] not in keep_l:
            continue
        for seed in cfg["train"]["seeds"]:
            if keep_s is not None and seed not in keep_s:
                continue
            if not evaluation_only:  # an evaluation-only config uses checkpoints_from's training runs
                out["train"].append({"learner": ln, "seed": seed, "hash": train_hash(cfg, ln, seed),
                                     "cap": top, "kind": "adaptive"})
            for split in SPLITS:
                if not split_seeds(cfg, split):
                    continue
                budgets = (cfg["test"]["budgets"] or cfg["train"]["budgets"]) if split == "test" else [top]
                cap = sum(sums[(split, s)] for s in split_seeds(cfg, split))
                for b in budgets:
                    for mode in ln["modes"]:
                        out["eval"].append({"learner": ln, "seed": seed, "budget": b, "split": split,
                                            "mode": mode, "hash": eval_hash(cfg, ln, seed, b, split, mode),
                                            "train_hash": train_hash(cfg, ln, seed), "cap": cap,
                                            "kind": eval_kind(ln)})
    for agent in cfg["agents"]:
        if agents is not None and agent not in agents:
            continue
        for s in split_seeds(cfg, "test"):
            out["agent"].append({"agent": agent, "seed": s, "split": "test", "cap": sums[("test", s)],
                                 "hash": agent_hash(cfg, agent, s), "kind": agent_kind(agent)})
    for agent in cfg.get("validation_agents") or ():
        if agents is not None and agent not in agents:
            continue
        for s in split_seeds(cfg, "validation"):
            out["agent"].append({"agent": agent, "seed": s, "split": "validation",
                                 "cap": sums[("validation", s)],
                                 "hash": agent_hash(cfg, agent, s, "validation"), "kind": agent_kind(agent)})
    return out


def plan(cfg: dict, **subset) -> dict:
    js = jobs(cfg, **subset)
    totals = defaultdict(int)
    for kind in js:
        for j in js[kind]:
            totals[j["kind"]] += j["cap"]
    return {
        "study": "X", "name": cfg["run"]["name"], "setting": cfg["setting"],
        "allocation": cfg["run"]["allocation"], "ledger": cfg["run"]["ledger"],
        "learners": [ln["id"] for ln in cfg["learners"]], "agents": cfg["agents"],
        "train": cfg["train"], "test": cfg["test"], "validation": cfg["validation"],
        "jobs": {k: len(v) for k, v in js.items()},
        "stream_budget_sums": {f"{sp}-s{s}": stream_sum(cfg, sp, s)
                               for sp in SPLITS for s in split_seeds(cfg, sp)},
        "caps": {"adaptive_total": totals["adaptive"], "reporting_total": totals["reporting"],
                 "training_total": sum(j["cap"] for j in js["train"]),
                 "learner_eval_total": sum(j["cap"] for j in js["eval"]),
                 "agent_total": sum(j["cap"] for j in js["agent"])},
        "ledger_on_creation": cfg["ledger"],
    }


def missing_caps(cfg: dict) -> dict:
    """Caps of the planned jobs that have no completed run, by ledger kind (``adaptive``, ``reporting``),
    with the training jobs' share and the number of jobs (study XS's budget check before a launch)."""
    done = completed(cfg["run"]["runs_dir"], cfg["run"]["name"])
    out = {"adaptive": 0, "reporting": 0, "training": 0, "jobs": 0, "training_jobs": 0}
    for kind_name, js in jobs(cfg).items():
        for j in js:
            if j["hash"] in done:
                continue
            out[j["kind"]] += j["cap"]
            out["jobs"] += 1
            if kind_name == "train":
                out["training"] += j["cap"]
                out["training_jobs"] += 1
    return out


def make_ledger(cfg: dict) -> SessionLedger:
    led = cfg["ledger"]
    ledger = SessionLedger(cfg["run"]["ledger"], dict(led["allocations"]), led["adaptive_cap"],
                          led["reporting_cap"])
    if cfg["run"]["allocation"] not in ledger.state["allocations"]:
        raise KeyError(f"allocation {cfg['run']['allocation']!r} is not in {cfg['run']['ledger']}")
    return ledger


# ---------------------------------------------------------- run records
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def existing_runs(runs_dir: str | Path, name: str) -> list[dict]:
    """Manifests of study X runs of config ``name`` (complete or not), oldest first."""
    out = []
    for man in sorted(Path(runs_dir).glob(f"{name}-*/manifest.json")):
        m = json.loads(man.read_text())
        if not str(m.get("phase", "")).startswith(PHASE) or m.get("x_name") != name:
            continue
        out.append({**m, "dir": str(man.parent), "complete": m.get("status") == "completed"
                    and m.get("stop_reason") == "done"})
    return sorted(out, key=lambda m: (m.get("start_time", ""), m["run_id"]))


def completed(runs_dir, name) -> dict:
    """job hash -> earliest completed run manifest."""
    done = {}
    for m in existing_runs(runs_dir, name):
        if m["complete"] and m["job_hash"] not in done:
            done[m["job_hash"]] = m
    return done


def _context(cfg, run_id, job, kind_name, parent=None) -> RunContext:
    arm = job["learner"]["id"] if "learner" in job else job["agent"]
    ctx = RunContext(cfg["run"]["runs_dir"], run_id, {**cfg, "job": {k: v for k, v in job.items()}},
                     phase=f"{PHASE}_{kind_name}", arm=arm, seed=job["seed"], parent_run=parent)
    ctx.write_manifest(x_name=cfg["run"]["name"], job_hash=job["hash"], job_kind=kind_name,
                       libraries=library_versions(), setting=cfg["setting"]["name"])
    return ctx


def _finish(ctx, ledger, meter, kind, t0, **extra):
    meter.flush()
    ctx.finish("completed", stop_reason="done", interactions={"physical": meter.used, "kind": kind,
                                                              "by_purpose": dict(meter.by_purpose)},
               loop_seconds=round(time.perf_counter() - t0, 2), session_budget=ledger.summary(), **extra)


# ------------------------------------------------------------ training
def make_venv(cfg: dict, learner: dict, seed: int, meter):
    from stable_baselines3.common.vec_env import DummyVecEnv
    fam, setting, tr = cfg["setting"]["family"], cfg["setting"], cfg["train"]
    layout = layout_for(cfg)

    def factory(rank):
        def build():
            src = training_source(fam, setting, tr["namespace"], tr["base_seed"], seed, rank, tr["n_envs"],
                                  tr["episodes_per_world"])
            extra = {}
            if learner.get("shaping") is not None:  # study XS: one shaper per environment
                extra["shaping"] = make_shaper(learner["shaping"], learner_gamma(learner["params"]))
            return make_gym(fam, src, layout, trial=learner["trial"], step_penalty=learner["step_penalty"],
                            meter=meter, purpose="exploration", **extra)
        return build

    venv = DummyVecEnv([factory(r) for r in range(tr["n_envs"])])
    if learner["rnd"] is not None:
        venv = rnd_vec_env(venv, seed=derive_seed("x_rnd", setting["name"], seed) % 2 ** 31, **learner["rnd"])
    return venv


def model_seed(cfg: dict, seed: int) -> int:
    """Common random numbers: every learner of a training seed starts from the same library seed."""
    return derive_seed("x_learner", cfg["setting"]["name"], seed) % 2 ** 31


def run_train(cfg: dict, job: dict, ledger: SessionLedger, stamp: str) -> dict:
    ln, seed = job["learner"], job["seed"]
    run_id = f"{cfg['run']['name']}-train-{ln['id']}-s{seed}-{stamp}"
    ctx = _context(cfg, run_id, job, "train")
    alloc = cfg["run"]["allocation"]
    ledger.register_run(run_id, alloc, job["cap"], {"learner": ln["id"], "kind": "adaptive", "job": "train",
                                                     "job_hash": job["hash"]})
    meter = BudgetMeter(ledger, run_id, alloc, job["cap"], kind="adaptive", flush_every=4096)
    venv = make_venv(cfg, ln, seed, meter)
    model = build_model(ln["algo"], venv, ln["params"], model_seed(cfg, seed))
    checkpoints = []
    t0 = time.perf_counter()

    def save(budget, m, steps):
        path = ctx.dir / f"checkpoint_{budget}.zip"
        m.save(str(path))
        checkpoints.append({"budget": budget, "interactions": int(steps), "path": path.name,
                            "seconds": round(time.perf_counter() - t0, 2)})
        ctx.event({"type": "checkpoint", **checkpoints[-1]})

    cb = checkpoint_callback(cfg["train"]["budgets"], save, log=ctx.event)
    try:
        model.learn(total_timesteps=job["cap"], callback=cb)
    except Exception as exc:
        meter.flush()
        ctx.finish("failed", error=repr(exc), interactions={"physical": meter.used, "kind": "adaptive"})
        raise
    seconds = time.perf_counter() - t0
    wrappers = venv.venv.envs if ln["rnd"] is not None else venv.envs
    executed = sum(e.interactions for e in wrappers)
    if not executed == meter.used == model.num_timesteps == job["cap"]:
        raise RuntimeError(f"interaction count mismatch: wrappers {executed}, meter {meter.used}, "
                           f"library {model.num_timesteps}, budget {job['cap']}")
    extra = {"rnd": dict(venv.stats)} if ln["rnd"] is not None else {}
    _finish(ctx, ledger, meter, "adaptive", t0, checkpoints=checkpoints, training_seconds=round(seconds, 2),
            steps_per_second=round(meter.used / seconds, 1), model_seed=model_seed(cfg, seed),
            tasks_trained=sum(e.tasks_done for e in wrappers), **extra)
    return {"run_id": run_id, "dir": str(ctx.dir), "checkpoints": checkpoints, "interactions": meter.used,
            "seconds": round(seconds, 2)}


# ---------------------------------------------------------- evaluation
def run_eval(cfg: dict, job: dict, train_run: dict, ledger: SessionLedger, stamp: str) -> dict:
    ln, seed, split, mode, budget = job["learner"], job["seed"], job["split"], job["mode"], job["budget"]
    ckpt = Path(train_run["dir"]) / f"checkpoint_{budget}.zip"
    run_id = f"{cfg['run']['name']}-eval-{ln['id']}-s{seed}-b{budget}-{split}-{mode[:3]}-{stamp}"
    ctx = _context(cfg, run_id, job, "eval", parent=train_run["run_id"])
    alloc, kind = cfg["run"]["allocation"], job["kind"]
    ledger.register_run(run_id, alloc, job["cap"], {"learner": ln["id"], "kind": kind, "job": "eval",
                                                     "job_hash": job["hash"]})
    meter = BudgetMeter(ledger, run_id, alloc, job["cap"], kind=kind)
    model = load_model(ln["algo"], ckpt)
    model.set_random_seed(derive_seed("x_eval", cfg["setting"]["name"], seed, budget, split, mode) % 2 ** 31)
    actor = Actor(model, ln["algo"], deterministic=mode == "deterministic")
    purpose = "exploration" if kind == "adaptive" else "reporting_eval"
    rows = []
    t0 = time.perf_counter()
    try:
        _play_split(cfg, ln, seed, split, mode, budget, actor, meter, purpose, rows, ctx)
    except Exception as exc:
        meter.flush()
        ctx.finish("failed", error=repr(exc), interactions={"physical": meter.used, "kind": kind})
        raise
    _finish(ctx, ledger, meter, kind, t0, summary=summarize(rows, cfg), checkpoint=str(ckpt))
    return {"run_id": run_id, "rows": len(rows), "interactions": meter.used}


def _play_split(cfg, ln, seed, split, mode, budget, actor, meter, purpose, rows, ctx):
    fam, layout, sp = cfg["setting"]["family"], layout_for(cfg), cfg[split]
    for s in sp["seeds"]:
        src = evaluation_source(fam, cfg["setting"], sp["namespace"], sp["base_seed"], s, sp["n_worlds"],
                                sp["episodes_per_world"])
        counter = iter(range(10 ** 9))

        def on_end(item, res, s=s, counter=counter):
            length, exact = reference(fam, item)
            r = row({**res, "world_key": item.world_key, "world_episode": item.world_episode,
                     "order": next(counter), "reference_length": length, "reference_exact": exact,
                     "budget": item.meta["budget"]},
                    arm=ln["id"], kind="learner", split=split, test_seed=s, group=ln["group"],
                    algo=ln["algo"], train_seed=seed, budget=budget, mode=mode,
                    invalid_actions=res["invalid_actions"])
            rows.append(r)
            ctx._append("eval.jsonl", r)

        env = make_gym(fam, src, layout, trial=ln["trial"], meter=meter, purpose=purpose, on_task_end=on_end)
        while True:
            try:
                obs, _ = env.reset()
            except SourceExhausted:
                break
            actor.reset()
            done = False
            while not done:
                obs, _, done, _, _ = env.step(actor.act(obs, env.action_masks()))


def run_agent(cfg: dict, job: dict, ledger: SessionLedger, stamp: str) -> dict:
    agent, s, split = job["agent"], job["seed"], job["split"]
    run_id = f"{cfg['run']['name']}-agent-{agent}-{split}-s{s}-{stamp}"
    ctx = _context(cfg, run_id, job, "agent")
    alloc, kind = cfg["run"]["allocation"], job["kind"]
    ledger.register_run(run_id, alloc, job["cap"], {"agent": agent, "kind": kind, "job": "agent",
                                                     "job_hash": job["hash"]})
    meter = BudgetMeter(ledger, run_id, alloc, job["cap"], kind=kind)
    t0 = time.perf_counter()
    play = play_recipequest if cfg["setting"]["family"] == "recipequest" else play_techtree
    try:
        rows = play(cfg, agent, s, split, meter)
    except Exception as exc:
        meter.flush()
        ctx.finish("failed", error=repr(exc), interactions={"physical": meter.used, "kind": kind})
        raise
    for r in rows:
        ctx._append("eval.jsonl", r)
    _finish(ctx, ledger, meter, kind, t0, summary=summarize(rows, cfg))
    return {"run_id": run_id, "rows": len(rows), "interactions": meter.used}


def summarize(rows: list[dict], cfg: dict) -> dict:
    if not rows:
        return {"n": 0}
    late_from = cfg["analysis"]["late_from"]
    late = [r for r in rows if r["world_episode"] >= late_from]

    def ratio(rs):
        ref = sum(r["reference_length"] or 0 for r in rs)
        return sum(r["primitive_length"] for r in rs) / ref if ref else float("nan")

    return {"n": len(rows), "success": float(np.mean([r["success"] for r in rows])),
            "success_late": float(np.mean([r["success"] for r in late])) if late else float("nan"),
            "cost_ratio": ratio(rows), "cost_ratio_late": ratio(late),
            "steps_late": float(np.mean([r["primitive_length"] for r in late])) if late else float("nan")}


# ----------------------------------------------------------- orchestration
def chain(cfg: dict, train_job: dict, eval_jobs: list[dict], stamp: str, done: dict) -> list[dict]:
    """Train one (learner, seed) unless a completed run exists (``done`` maps job hashes to completed
    runs, including those of ``checkpoints_from``), then evaluate its checkpoints."""
    ledger = make_ledger(cfg)
    out = []
    tr = done.get(train_job["hash"])
    if tr is None:
        if cfg["run"]["checkpoints_from"]:
            raise RuntimeError(f"no completed training run of {train_job['learner']['id']} seed "
                               f"{train_job['seed']} in {cfg['run']['checkpoints_from']}; "
                               f"run that config first")
        res = run_train(cfg, train_job, ledger, stamp)
        tr = {"run_id": res["run_id"], "dir": res["dir"]}
        out.append({"job": "train", **res})
    else:
        tr = {"run_id": tr["run_id"], "dir": tr["dir"]}
    for j in eval_jobs:
        if j["hash"] not in done:
            out.append({"job": "eval", **run_eval(cfg, j, tr, ledger, stamp)})
    return out


def _worker(args):
    import torch
    torch.set_num_threads(1)
    from ..training.run import process_source
    process_source(Path(__file__).resolve().parents[3])
    kind, payload = args
    if kind == "chain":
        return chain(*payload)
    cfg, job, stamp = payload
    return [{"job": "agent", **run_agent(cfg, job, make_ledger(cfg), stamp)}]


def run_study(cfg: dict, resume: bool = False, workers: int = 1, stamp: str | None = None,
              failures_out: list | None = None, on_result=None, **subset) -> list[dict]:
    """Runs the missing jobs. A failing job is logged and the others continue; the failures go to
    ``failures_out`` when given (the CLI then exits non-zero), else a RuntimeError is raised at the end.
    ``on_result`` receives each finished task's results as they arrive."""
    js = jobs(cfg, **subset)
    done = completed(cfg["run"]["runs_dir"], cfg["run"]["name"])
    planned = {j["hash"] for k in js for j in js[k]}
    if cfg["run"]["requires_flag"] and not resume and planned & set(done):
        raise RuntimeError(f"{len(planned & set(done))} jobs of this measurement config already have "
                           f"completed runs; use --resume to run only the missing ones")
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    if cfg["run"]["checkpoints_from"]:  # training runs of the other config
        done = {**completed(cfg["run"]["runs_dir"], cfg["_source_name"]), **done}
    pairs = {}
    for e in js["eval"]:
        key = (e["learner"]["id"], e["seed"])
        pairs.setdefault(key, {"learner": e["learner"], "seed": e["seed"], "hash": e["train_hash"],
                               "evals": []})["evals"].append(e)
    for tj in js["train"]:
        pairs.setdefault((tj["learner"]["id"], tj["seed"]), {**tj, "evals": []})
    tasks = []
    for pair in pairs.values():
        tj = {k: v for k, v in pair.items() if k != "evals"}
        tj.setdefault("cap", max(cfg["train"]["budgets"]))
        tj.setdefault("kind", "adaptive")
        if tj["hash"] in done and all(e["hash"] in done for e in pair["evals"]):
            continue
        tasks.append(("chain", (cfg, tj, pair["evals"], stamp, done)))
    tasks += [("agent", (cfg, j, stamp)) for j in js["agent"] if j["hash"] not in done]
    results, failures = [], []

    def failed(task, exc_text):
        kind, payload = task
        job = payload[1]
        what = job["learner"]["id"] if "learner" in job else job["agent"]
        failures.append({"task": kind, "arm": what, "seed": job["seed"], "error": exc_text})
        print(f"[X] {kind} {what} seed {job['seed']} failed:\n{exc_text}", file=sys.stderr, flush=True)

    if workers <= 1:
        for t in tasks:
            try:
                got = _worker(t)
            except Exception:
                failed(t, traceback.format_exc())
                continue
            results += got
            if on_result:
                on_result(got)
    else:  # one failing job never cancels the others; results are kept as they arrive
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    got = fut.result()
                except Exception:
                    failed(futures[fut], traceback.format_exc())
                    continue
                results += got
                if on_result:
                    on_result(got)
    if failures_out is not None:
        failures_out.extend(failures)
    elif failures:
        raise RuntimeError(f"{len(failures)} of {len(tasks)} jobs failed: {failures}")
    return results


def results_path(cfg: dict) -> Path:
    return Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
