"""Study X2 trace collection (evaluator side; charges the study ledger) and the
study's own config schema (``configs/inference_baselines/``).

A config declares evidence policies and trace sets. Every (set, seed, policy)
is one run: the policy plays the set's tasks of that seed in world-major
order (16 episodes per world, beliefs carried within a world) and every
world's logs, public recipe descriptions, the recipes each episode's task
registered and, evaluator only, the hidden requirements are written to
``<run>/evidence/`` (the record of study M, plus the set's name and
observation mode).

Policies (fixed; the logs are what all methods read):

``maximal``        the brute-force node (every candidate gathered; no inference), charged
                   to the reporting cap as in studies M and R
``focused_sample`` the sampling walker (focused walk, then moves among consistent nodes)
``random_omit``    every candidate omitted independently with ``omit_prob`` at every
                   replan, ignoring the evidence (the walker study's control policy)

Sets: ``generated`` (family F1 of the walker study, items observable or
goal-only, every seed its own worlds) and ``benchmark`` (worlds converted from
a published recipe graph by ``benchmarks.to_recipequest``; one world per
conversion seed of the whole graph, or one per sampled goal; each episode asks
for an item drawn uniformly among the world's items). Nothing here reads
hidden requirements except ``world_records``, which stores them for the
evaluator.
"""

from __future__ import annotations

import copy
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from ..agents.walker import Walker
from ..config import DEFAULTS, eval_stream_config
from ..envs.generator import TaskConfig, TaskStream, derive_seed, make_task
from ..evaluation.markov_study import (
    LoggedMaximal, info_from_record, log_from_record, select_runs, world_records,
)
from ..evaluation.walkers import NON_ADAPTIVE, eval_order, make_walker, play, variant_config
from ..training.budget import BudgetMeter, SessionLedger
from ..training.run import RunContext

POLICIES = ("focused_sample", "random_omit", "maximal")
DEFAULT_VARIANT = {"name": "default", "tasks": {}}
RUN_KEYS = {"name", "phase", "allocation", "ledger", "runs_dir", "requires_flag"}
LEDGER_KEYS = {"allocations", "adaptive_cap", "reporting_cap"}
WALKER_DEFAULTS = {k: DEFAULTS["walker"][k] for k in ("max_evals", "temperature", "restart_after",
                                                       "exact_cap", "plan_cap")}
WALKER_DEFAULTS |= {"consistent_moves": 50, "omit_prob": 0.663}
SET_KEYS = {
    "generated": {"name", "kind", "observe_items", "namespace", "base_seed", "seeds", "n_worlds",
                  "episodes_per_world"},
    "benchmark": {"name", "kind", "graph", "worlds", "n_worlds", "namespace", "seeds", "episodes_per_world",
                  "observe_items", "conversion_seed", "goal_sample_seed"},
}
SET_DEFAULTS = {"generated": {"base_seed": 10_000},
                "benchmark": {"conversion_seed": 0, "goal_sample_seed": 0}}
TOP_KEYS = {"run", "ledger", "policies", "walker", "sets", "analysis"}
CONVERSION = {"pool_size": 6, "max_true": 3, "max_alternatives": 2}


# ================================================================== config

def load(path: str | Path) -> dict:
    """An X2 config, validated (unknown keys are errors)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return validate(raw, str(path))


def validate(raw: dict, source: str = "<dict>") -> dict:
    cfg = copy.deepcopy(raw)
    if set(cfg) - TOP_KEYS:
        raise KeyError(f"unknown config keys {sorted(set(cfg) - TOP_KEYS)}")
    run = cfg.setdefault("run", {})
    if set(run) - RUN_KEYS:
        raise KeyError(f"unknown run keys {sorted(set(run) - RUN_KEYS)}")
    run.setdefault("runs_dir", "runs")
    run.setdefault("requires_flag", False)
    run.setdefault("phase", run.get("name", "x2"))
    if set(cfg.get("ledger", {})) != LEDGER_KEYS:
        raise KeyError(f"ledger needs exactly {sorted(LEDGER_KEYS)}")
    if run["allocation"] not in cfg["ledger"]["allocations"]:
        raise KeyError(f"allocation {run['allocation']!r} is not declared in the ledger")
    pol = cfg.setdefault("policies", list(POLICIES))
    if not pol or set(pol) - set(POLICIES):
        raise ValueError(f"policies must be a nonempty subset of {POLICIES}")
    wc = cfg.setdefault("walker", {})
    if set(wc) - set(WALKER_DEFAULTS):
        raise KeyError(f"unknown walker keys {sorted(set(wc) - set(WALKER_DEFAULTS))}")
    cfg["walker"] = {**WALKER_DEFAULTS, **wc}
    names = set()
    dev = run["allocation"] == "dev"
    for s in cfg.setdefault("sets", []):
        kind = s.get("kind")
        if kind not in SET_KEYS:
            raise ValueError(f"set kind must be one of {sorted(SET_KEYS)}")
        if set(s) - SET_KEYS[kind]:
            raise KeyError(f"unknown keys of set {s.get('name')!r}: {sorted(set(s) - SET_KEYS[kind])}")
        for k, v in SET_DEFAULTS[kind].items():
            s.setdefault(k, v)
        missing = SET_KEYS[kind] - set(s)
        if missing:
            raise KeyError(f"set {s.get('name')!r} misses {sorted(missing)}")
        if s["observe_items"] not in ("all", "goal_only"):
            raise ValueError("observe_items must be 'all' or 'goal_only'")
        if kind == "benchmark" and s["worlds"] not in ("whole", "per_goal"):
            raise ValueError("benchmark worlds must be 'whole' or 'per_goal'")
        if s["name"] in names:
            raise ValueError(f"duplicate set name {s['name']!r}")
        names.add(s["name"])
        if dev != s["namespace"].endswith("_dev"):
            raise ValueError(f"set {s['name']!r}: development namespaces end with '_dev', and only "
                             f"the allocation 'dev' may use them")
    cfg.setdefault("analysis", {})
    cfg["_source"] = source
    return cfg


def make_ledger(cfg: dict) -> SessionLedger:
    led = cfg["ledger"]
    return SessionLedger(cfg["run"]["ledger"], dict(led["allocations"]), led["adaptive_cap"],
                         led["reporting_cap"])


# ================================================================= streams

def generated_config(s: dict) -> dict:
    """The walker-study config of a generated set: family F1 (Stage D
    distribution, depth 1-3, no distractor base facts, deterministic) with the
    set's observation mode, pool worlds, every seed its own worlds."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg["train"].update(world_mode="pool", n_worlds=8)
    cfg["train"]["tasks"].update(profile="unknown_prerequisites", depth_min=1, depth_max=3,
                                 n_distractor_base=0, observe_items=s["observe_items"])
    cfg["eval"].update(namespace=s["namespace"], base_seed=s["base_seed"], track="inference",
                       n_worlds=s["n_worlds"], episodes_per_world=s["episodes_per_world"],
                       n_tasks=s["n_worlds"] * s["episodes_per_world"])
    cfg["walker"]["eval_world_offset_by_seed"] = True
    return cfg


@lru_cache(maxsize=None)
def _graph(name: str):
    from ..benchmarks import loaders
    return loaders.load(name)


@lru_cache(maxsize=None)
def goal_sample(graph: str, n: int, seed: int) -> tuple:
    from ..benchmarks.to_recipequest import convertible_goals
    goals = convertible_goals(_graph(graph), **CONVERSION)
    if n >= len(goals):
        return tuple(goals)
    rng = np.random.default_rng(derive_seed("x2_benchmark_goals", graph, seed))
    return tuple(sorted(str(g) for g in rng.choice(goals, n, replace=False)))


@lru_cache(maxsize=None)
def converted_world(graph: str, worlds: str, n: int, conversion_seed: int, goal_seed: int, index: int):
    from ..benchmarks.to_recipequest import recipequest_world, sink_goals
    g = _graph(graph)
    if worlds == "whole":
        return recipequest_world(g, sink_goals(g, max_alternatives=CONVERSION["max_alternatives"]),
                                 seed=conversion_seed + index, **CONVERSION)
    goals = goal_sample(graph, n, goal_seed)
    return recipequest_world(g, [goals[index]], seed=conversion_seed, **CONVERSION)


class BenchmarkStream:
    """Tasks of a benchmark set for one seed, indexed in world-major order."""

    def __init__(self, s: dict, seed: int):
        self.s, self.seed, self.E = s, seed, s["episodes_per_world"]
        self.cfg = TaskConfig(profile="unknown_prerequisites", n_distractor_base=0,
                              observe_items=s["observe_items"])
        self.namespace = s["namespace"]
        self._tasks: dict = {}

    def n_worlds(self) -> int:
        if self.s["worlds"] == "whole":
            return self.s["n_worlds"]
        return min(self.s["n_worlds"], len(goal_sample(self.s["graph"], self.s["n_worlds"],
                                                       self.s["goal_sample_seed"])))

    def world(self, w: int):
        s = self.s
        return converted_world(s["graph"], s["worlds"], s["n_worlds"], s["conversion_seed"],
                               s["goal_sample_seed"], w).world

    def task(self, i: int):
        if i not in self._tasks:
            w, e = divmod(i, self.E)
            world = self.world(w)
            items = sorted(t for t, _ in world.levels)
            rng = np.random.default_rng(derive_seed("x2_benchmark_goal", self.namespace, self.seed, w, e))
            goal = int(rng.choice(items))
            tseed = derive_seed("x2_benchmark_task", self.namespace, self.seed, w, e)
            self._tasks[i] = make_task(world, tseed, self.cfg, self.namespace, goal=goal)
        return self._tasks[i]


def stream_for(s: dict, seed: int):
    """(stream, world-major order) of a set for one seed."""
    if s["kind"] == "generated":
        cfg = generated_config(s)
        scfg = eval_stream_config(variant_config(cfg, seed, DEFAULT_VARIANT))
        return TaskStream(scfg), eval_order(scfg, cfg["eval"]["n_tasks"], s["episodes_per_world"])
    st = BenchmarkStream(s, seed)
    return st, list(range(st.n_worlds() * s["episodes_per_world"]))


def run_budget(s: dict, seed: int) -> int:
    """Sum of the task budgets of one run's stream: an upper bound on its
    interactions (no episode exceeds its budget; generator only)."""
    stream, order = stream_for(s, seed)
    return int(sum(stream.task(i).budget for i in order))


# ================================================================== agents

class LoggedRandomOmit(Walker):
    """``random_omit`` with its episode logs kept: the episode loop of an
    episodic walker, the node choice of ``random_omit`` (same random stream,
    so the same actions)."""

    def __init__(self, rng, omit_prob: float, **kw):
        if not 0.0 <= omit_prob < 1.0:
            raise ValueError("omit_prob must be in [0, 1)")
        super().__init__("focused_sample", rng, omit_prob=omit_prob, **kw)
        self.label = "walker:random_omit"

    def choose_node(self, w, log, held=frozenset()):
        return self._omit_node(w), {"evals": 0}


def evidence_agent(policy: str, wc: dict, seed_key):
    rng = np.random.default_rng(derive_seed("walker", *seed_key) % (2 ** 32))
    if policy == "maximal":
        return LoggedMaximal(rng, plan_cap=wc["plan_cap"])
    if policy == "random_omit":
        return LoggedRandomOmit(rng, wc["omit_prob"], plan_cap=wc["plan_cap"])
    if policy == "focused_sample":
        return make_walker("focused_sample", wc, seed_key, 0.0)
    raise ValueError(f"unknown policy {policy!r}")


def kind_of(policy: str) -> str:
    return "reporting" if policy in NON_ADAPTIVE else "adaptive"


# ============================================================== collection

def plan(cfg: dict) -> dict:
    """Dry run: every run with its stream budget (its cap and worst case) and
    the totals against the ledger declaration; generator only."""
    runs, adaptive, reporting = [], 0, 0
    for s in cfg["sets"]:
        for seed in s["seeds"]:
            b = run_budget(s, seed)
            stream, order = stream_for(s, seed)
            for p in cfg["policies"]:
                runs.append({"set": s["name"], "seed": seed, "policy": p, "kind": kind_of(p),
                             "tasks": len(order), "worlds": len(order) // s["episodes_per_world"], "cap": b})
                if kind_of(p) == "adaptive":
                    adaptive += b
                else:
                    reporting += b
    led = cfg["ledger"]
    alloc = cfg["run"]["allocation"]
    return {"allocation": alloc, "ledger": cfg["run"]["ledger"], "runs": runs, "n_runs": len(runs),
            "worst_case_adaptive": adaptive, "worst_case_reporting": reporting,
            "allocation_covers_worst_case": adaptive <= led["allocations"][alloc],
            "reporting_cap_declared": led["reporting_cap"], "adaptive_cap_declared": led["adaptive_cap"]}


def world_labels(stream, order) -> dict:
    """{world key: index of the world in play order}."""
    out: dict = {}
    for i in order:
        out.setdefault(stream.task(i).world.world_key, len(out))
    return out


class AllowanceShort(RuntimeError):
    """The ledger's remaining allowance does not cover a run's stream sum."""


def run_config(s: dict, policy: str, seed: int, wc: dict) -> dict:
    """What a run's config hash covers: its content, never a file path."""
    return {"study": "x2", "set": s, "policy": policy, "seed": seed, "walker": wc,
            "arm": {"id": policy, "strategy": policy}}


def completed_run(runs_dir: str | Path, prefix: str, config_hash: str, code: str,
                  commit: str | None) -> str | None:
    """The id of a completed run under ``prefix`` with this config content,
    commit and loaded code, if there is one (``rank`` accepts evidence from one
    (commit, code) only)."""
    for d in sorted(Path(runs_dir).glob(f"{prefix}*")):
        if not ((d / "manifest.json").exists() and (d / "evidence").is_dir()):
            continue
        m = json.loads((d / "manifest.json").read_text())
        src = m.get("source") or {}
        if (m.get("config_hash") == config_hash and m.get("status") == "completed"
                and m.get("stop_reason") == "tasks_done"
                and src.get("loaded_code_sha256") == code and src.get("commit") == commit):
            return m.get("run_id", d.name)
    return None


def collect(cfg: dict, ledger: SessionLedger, stamp: str | None = None, only_sets=None,
            resume: bool = False) -> list[dict]:
    """Play every run of the config, charge the ledger and write the evidence.
    With ``resume``, a (set, policy, seed) that already has a completed run with
    the same config content, commit and loaded code is skipped. A run starts
    only if the ledger's remaining allowance of its kind covers its stream sum
    (``AllowanceShort`` otherwise: amend the ledger first)."""
    from ..envs.generator import stable_hash
    from ..training.run import LOADED_CODE_SHA256, process_source
    commit = process_source(Path(__file__).resolve().parents[3]).get("commit")
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    run, wc = cfg["run"], cfg["walker"]
    out = []
    for s in cfg["sets"]:
        if only_sets and s["name"] not in only_sets:
            continue
        for seed in s["seeds"]:
            cap = run_budget(s, seed)
            for policy in cfg["policies"]:
                rc = run_config(s, policy, seed, wc)
                prefix = f"{run['name']}-{s['name']}-{policy}-s{seed}-"
                if resume:
                    done_id = completed_run(run["runs_dir"], prefix, stable_hash(rc), LOADED_CODE_SHA256,
                                            commit)
                    if done_id:
                        out.append({"run_id": done_id, "set": s["name"], "policy": policy, "seed": seed,
                                    "skipped": True, "complete": True})
                        print(f"{done_id}: completed earlier with the same config, commit and code, skipped",
                              flush=True)
                        continue
                kind = kind_of(policy)
                left = ledger.remaining(run["allocation"], kind)
                if left < cap:
                    raise AllowanceShort(f"{prefix}{stamp} needs up to {cap} {kind} interactions and the "
                                         f"ledger has {left} left: amend the ledger first")
                stream, order = stream_for(s, seed)
                labels = world_labels(stream, order)
                run_id = f"{prefix}{stamp}"
                ctx = RunContext(run["runs_dir"], run_id, rc, phase=run["phase"], arm=policy, seed=seed)
                ledger.register_run(run_id, run["allocation"], cap, {"set": s["name"], "policy": policy})
                meter = BudgetMeter(ledger, run_id, run["allocation"], cap, kind=kind)
                agent = evidence_agent(policy, wc, (s["name"], seed, policy, "x2"))
                t0 = time.time()
                try:
                    rows = play(agent, policy, stream, order, meter, seed, ctx,
                                "reporting_eval" if kind == "reporting" else "exploration",
                                {"strategy": policy, "set": s["name"]})
                finally:
                    meter.flush()
                meta = {"run_id": run_id, "seed": seed, "strategy": policy, "set": s["name"],
                        "set_kind": s["kind"], "observe_items": s["observe_items"],
                        "namespace": s["namespace"]}
                recs = world_records(agent, stream, order, rows, meta)
                (ctx.dir / "evidence").mkdir()
                for rec in recs:
                    rec["world_index"] = labels[rec["world_key"]]
                    rec["world_label"] = f"{s['name']}:s{seed}:w{rec['world_index']}"
                    path = ctx.dir / "evidence" / f"w{rec['world_index']:03d}.json"
                    path.write_text(json.dumps(rec), encoding="utf-8")
                done = len(rows) == len(order)
                ctx.finish("completed", stop_reason="tasks_done" if done else "interaction_cap",
                           interactions={"physical": meter.used, "kind": kind}, evidence_worlds=len(recs),
                           seconds_per_task=round((time.time() - t0) / max(1, len(rows)), 4),
                           config_file=cfg.get("_source"), session_budget=ledger.summary())
                out.append({"run_id": run_id, "set": s["name"], "policy": policy, "seed": seed,
                            "tasks": len(rows), "complete": done, "interactions": meter.used,
                            "worlds": len(recs)})
                print(f"{run_id}: {len(rows)} tasks, {meter.used} interactions, {len(recs)} worlds",
                      flush=True)
    return out


# ================================================================ evidence

def load_evidence(runs_dir: str | Path, prefix: str) -> tuple[list[dict], dict]:
    """World records of the runs ``<runs_dir>/<prefix>*``, one run per (set,
    policy, seed): the earliest, or the earliest later completed run with the
    same config hash, commit and loaded code when it did not complete. Every
    run folder with a manifest and a config counts, so a crashed or killed
    run (no evidence, status not completed) is listed and replaced, or
    reported as an incomplete choice. Returns (records, selection), where the
    selection lists the reruns used, the ignored runs, the chosen runs without
    complete evidence and the (commit, code) pairs of the chosen runs."""
    runs = []
    for d in sorted(Path(runs_dir).glob(f"{prefix}*")):
        if not ((d / "manifest.json").exists() and (d / "config.json").exists()):
            continue
        man = json.loads((d / "manifest.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("study") != "x2":
            continue
        src = man.get("source") or {}
        complete = (man.get("status") == "completed" and man.get("stop_reason") == "tasks_done"
                    and (d / "evidence").is_dir())
        runs.append({"run_id": man.get("run_id", d.name), "dir": d, "seed": man.get("seed"),
                     "strategy": f"{cfg['set']['name']}/{cfg['policy']}", "start": man.get("start_time", ""),
                     "config_hash": man.get("config_hash"), "commit": src.get("commit"),
                     "code": src.get("loaded_code_sha256"), "complete": complete})
    chosen, sel = select_runs(runs)
    sel["incomplete_chosen"] = sorted(r["run_id"] for r in chosen if not r["complete"])
    sel["provenance"] = sorted({(r["commit"], r["code"]) for r in chosen}, key=str)
    sel["chosen"] = sorted(r["run_id"] for r in chosen)
    out = []
    for r in sorted(chosen, key=lambda x: (x["strategy"], x["seed"])):
        if not (r["dir"] / "evidence").is_dir():
            continue
        for f in sorted((r["dir"] / "evidence").glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["infos"] = {i["signature"]: info_from_record(i) for i in rec["infos"]}
            rec["logs"] = [log_from_record(x) for x in rec["logs"]]
            rec["complete"] = r["complete"]
            out.append(rec)
    return out, sel
