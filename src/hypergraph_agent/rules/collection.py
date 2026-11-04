"""Study RL trace collection (evaluator side; charges the study ledger) and
the study's own config schema (``configs/rule_learners/``), built on study
X2's collection machinery (``inference_baselines.collection``), which this
module imports and never changes.

RL collects logs of family F1 (X2's generated sets: study R's Stage D
distribution, goals of depth 1-3, no distractor base facts, items observable)
in variants that X2's schema cannot express, with X2's three evidence
policies (``maximal``, ``focused_sample``, ``random_omit``) and X2's walker
settings. A set has X2's generated-set keys plus the variant's options:

``failure_prob``  probability that an eligible state-changing action fails and
                  leaves the state unchanged (slips; ``envs.generator.TaskConfig``);
                  every policy is given it, as every walker of the repository is
                  given the failure probability of the stream it plays (default 0)
``pool_size``, ``max_true``, ``arity_weights``
                  the generator's options (``envs.generator.WorldConfig``): the pool of
                  candidates of every recipe, the largest requirement and the
                  probability of each requirement size 1..max_true (defaults 6, 3 and
                  (0.2, 0.5, 0.3), X2's)

With the defaults a set's tasks and the policies' actions are X2's for the
same namespace and seed (tested). Namespaces start with ``rl_``, and those of
development end with ``_dev`` (allocation ``dev`` only). A run's config hash
covers its content (study, set, policy, seed, walker settings), not the file
it came from. Nothing here reads hidden requirements except
``markov_study.world_records``, which stores them, evaluator only, in each
evidence file.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..config import eval_stream_config
from ..envs.generator import TaskConfig, TaskStream, WorldConfig, derive_seed
from ..evaluation.markov_study import (
    LoggedMaximal, info_from_record, log_from_record, select_runs, world_records,
)
from ..evaluation.walkers import eval_order, make_walker, play, variant_config
from ..inference_baselines import collection as x2
from ..training.budget import BudgetMeter, SessionLedger
from ..training.run import RunContext

STUDY = "rl"
POLICIES = x2.POLICIES
WALKER_DEFAULTS = dict(x2.WALKER_DEFAULTS)
TOP_KEYS = {"run", "ledger", "policies", "walker", "sets"}
SET_KEYS = {"name", "kind", "observe_items", "namespace", "base_seed", "seeds", "n_worlds",
            "episodes_per_world", "failure_prob", "pool_size", "max_true", "arity_weights"}
SET_DEFAULTS = {"kind": "generated", "observe_items": "all", "base_seed": 10_000, "failure_prob": 0.0,
                "pool_size": 6, "max_true": 3, "arity_weights": [0.2, 0.5, 0.3]}
AllowanceShort = x2.AllowanceShort


# ================================================================== config

def load(path: str | Path) -> dict:
    """An RL config, validated (unknown keys are errors)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return validate(raw, str(path))


def validate(raw: dict, source: str = "<dict>") -> dict:
    cfg = copy.deepcopy(raw)
    if set(cfg) - TOP_KEYS:
        raise KeyError(f"unknown config keys {sorted(set(cfg) - TOP_KEYS)}")
    run = cfg.setdefault("run", {})
    if set(run) - x2.RUN_KEYS:
        raise KeyError(f"unknown run keys {sorted(set(run) - x2.RUN_KEYS)}")
    run.setdefault("runs_dir", "runs")
    run.setdefault("requires_flag", False)
    run.setdefault("phase", run.get("name", STUDY))
    if set(cfg.get("ledger", {})) != x2.LEDGER_KEYS:
        raise KeyError(f"ledger needs exactly {sorted(x2.LEDGER_KEYS)}")
    if run["allocation"] not in cfg["ledger"]["allocations"]:
        raise KeyError(f"allocation {run['allocation']!r} is not declared in the ledger")
    pol = cfg.setdefault("policies", list(POLICIES))
    if not pol or set(pol) - set(POLICIES):
        raise ValueError(f"policies must be a nonempty subset of {POLICIES}")
    wc = cfg.setdefault("walker", {})
    if set(wc) - set(WALKER_DEFAULTS):
        raise KeyError(f"unknown walker keys {sorted(set(wc) - set(WALKER_DEFAULTS))}")
    cfg["walker"] = {**WALKER_DEFAULTS, **wc}
    dev, names = run["allocation"] == "dev", set()
    for s in cfg.setdefault("sets", []):
        if set(s) - SET_KEYS:
            raise KeyError(f"unknown keys of set {s.get('name')!r}: {sorted(set(s) - SET_KEYS)}")
        for k, v in SET_DEFAULTS.items():
            s.setdefault(k, copy.deepcopy(v))
        missing = SET_KEYS - set(s)
        if missing:
            raise KeyError(f"set {s.get('name')!r} misses {sorted(missing)}")
        if s["kind"] != "generated" or s["observe_items"] != "all":
            raise ValueError("RL sets are generated worlds with items observable (the rule learners need "
                             "every craft outcome)")
        world_of(s)  # the generator's own checks (sizes, weights)
        TaskConfig(failure_prob=float(s["failure_prob"]))
        if s["name"] in names:
            raise ValueError(f"duplicate set name {s['name']!r}")
        names.add(s["name"])
        if not s["namespace"].startswith("rl_"):
            raise ValueError(f"set {s['name']!r}: RL namespaces start with 'rl_'")
        if dev != s["namespace"].endswith("_dev"):
            raise ValueError(f"set {s['name']!r}: development namespaces end with '_dev', and only "
                             f"the allocation 'dev' may use them")
    cfg["_source"] = source
    return cfg


def make_ledger(cfg: dict) -> SessionLedger:
    return x2.make_ledger(cfg)


def declaration(cfg: dict) -> dict:
    """The ledger declaration of a config (allocations and caps)."""
    led = cfg["ledger"]
    return {"allocations": dict(led["allocations"]), "adaptive_cap": led["adaptive_cap"],
            "reporting_cap": led["reporting_cap"]}


def ledger_declaration(path: str | Path) -> dict | None:
    """The declaration an existing ledger file holds (from its creation or its
    last logged amendment), or None when the file does not exist yet."""
    p = Path(path)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"allocations": dict(s["allocations"]), "adaptive_cap": s["adaptive_cap"],
            "reporting_cap": s["reporting_cap"]}


def world_of(s: dict) -> WorldConfig:
    return WorldConfig(pool_size=int(s["pool_size"]), max_true=int(s["max_true"]),
                       arity_weights=tuple(float(w) for w in s["arity_weights"]))


# ================================================================= streams

def set_config(s: dict) -> dict:
    """X2's walker-study config of a generated set with the set's variant options."""
    cfg = x2.generated_config(s)
    cfg["world"].update(pool_size=int(s["pool_size"]), max_true=int(s["max_true"]),
                        arity_weights=[float(w) for w in s["arity_weights"]])
    cfg["train"]["tasks"]["failure_prob"] = float(s["failure_prob"])
    return cfg


def stream_for(s: dict, seed: int):
    """(stream, world-major order) of a set for one seed."""
    cfg = set_config(s)
    scfg = eval_stream_config(variant_config(cfg, seed, x2.DEFAULT_VARIANT))
    return TaskStream(scfg), eval_order(scfg, cfg["eval"]["n_tasks"], s["episodes_per_world"])


def run_budget(s: dict, seed: int) -> int:
    """Sum of the task budgets of one run's stream (generator only)."""
    stream, order = stream_for(s, seed)
    return int(sum(stream.task(i).budget for i in order))


def evidence_agent(policy: str, wc: dict, seed_key, epsilon: float):
    """X2's evidence policies, given the stream's failure probability (with 0,
    the agents of ``inference_baselines.collection.evidence_agent``)."""
    rng = np.random.default_rng(derive_seed("walker", *seed_key) % (2 ** 32))
    if policy == "maximal":
        return LoggedMaximal(rng, epsilon=epsilon, plan_cap=wc["plan_cap"])
    if policy == "random_omit":
        return x2.LoggedRandomOmit(rng, wc["omit_prob"], plan_cap=wc["plan_cap"], epsilon=epsilon)
    if policy == "focused_sample":
        return make_walker("focused_sample", wc, seed_key, epsilon)
    raise ValueError(f"unknown policy {policy!r}")


# ============================================================== collection

def plan(cfg: dict) -> dict:
    """Dry run: every run with its stream budget (its cap and worst case) and
    the totals against the ledger declaration; generator only."""
    runs, adaptive, reporting = [], 0, 0
    for s in cfg["sets"]:
        for seed in s["seeds"]:
            b = run_budget(s, seed)
            _, order = stream_for(s, seed)
            for p in cfg["policies"]:
                kind = x2.kind_of(p)
                runs.append({"set": s["name"], "seed": seed, "policy": p, "kind": kind, "tasks": len(order),
                             "worlds": len(order) // s["episodes_per_world"], "cap": b})
                adaptive += b if kind == "adaptive" else 0
                reporting += b if kind == "reporting" else 0
    led, alloc = cfg["ledger"], cfg["run"]["allocation"]
    found = ledger_declaration(cfg["run"]["ledger"])
    return {"allocation": alloc, "ledger": cfg["run"]["ledger"], "runs": runs, "n_runs": len(runs),
            "worst_case_adaptive": adaptive, "worst_case_reporting": reporting,
            "largest_run": {k: max((r["cap"] for r in runs if r["kind"] == k), default=0)
                            for k in ("adaptive", "reporting")},
            "allocation_covers_worst_case": adaptive <= led["allocations"][alloc],
            "reporting_cap_declared": led["reporting_cap"], "adaptive_cap_declared": led["adaptive_cap"],
            "ledger_file_declaration": found,
            "ledger_file_matches": found is None or found == declaration(cfg)}


def run_config(s: dict, policy: str, seed: int, wc: dict) -> dict:
    """What a run's config hash covers: its content, never a file path."""
    return {"study": STUDY, "set": s, "policy": policy, "seed": seed, "walker": wc,
            "arm": {"id": policy, "strategy": policy}}


def collect(cfg: dict, ledger: SessionLedger, stamp: str | None = None, only_sets=None,
            resume: bool = False, only_seeds=None) -> list[dict]:
    """Play every run of the config, charge the ledger and write the evidence
    (X2's ``collect`` with RL's streams and policies). With ``resume``, a
    (set, policy, seed) that already has a completed run with the same config
    content, commit and loaded code is skipped. A run starts only if the
    ledger's remaining allowance of its kind covers its stream sum
    (``AllowanceShort`` otherwise: amend the ledger first). ``only_sets`` and
    ``only_seeds`` restrict a session (sessions may run concurrently: the
    ledger file is locked)."""
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
            if only_seeds is not None and seed not in only_seeds:
                continue
            cap = run_budget(s, seed)
            for policy in cfg["policies"]:
                rc = run_config(s, policy, seed, wc)
                prefix = f"{run['name']}-{s['name']}-{policy}-s{seed}-"
                if resume:
                    done_id = x2.completed_run(run["runs_dir"], prefix, stable_hash(rc), LOADED_CODE_SHA256,
                                               commit)
                    if done_id:
                        out.append({"run_id": done_id, "set": s["name"], "policy": policy, "seed": seed,
                                    "skipped": True, "complete": True})
                        print(f"{done_id}: completed earlier with the same config, commit and code, skipped",
                              flush=True)
                        continue
                kind = x2.kind_of(policy)
                left = ledger.remaining(run["allocation"], kind)
                if left < cap:
                    raise AllowanceShort(f"{prefix}{stamp} needs up to {cap} {kind} interactions and the "
                                         f"ledger has {left} left: amend the ledger first")
                stream, order = stream_for(s, seed)
                labels = x2.world_labels(stream, order)
                run_id = f"{prefix}{stamp}"
                ctx = RunContext(run["runs_dir"], run_id, rc, phase=run["phase"], arm=policy, seed=seed)
                ledger.register_run(run_id, run["allocation"], cap, {"set": s["name"], "policy": policy})
                meter = BudgetMeter(ledger, run_id, run["allocation"], cap, kind=kind)
                eps = float(s["failure_prob"])
                agent = evidence_agent(policy, wc, (s["name"], seed, policy, STUDY), eps)
                t0 = time.time()
                try:
                    rows = play(agent, policy, stream, order, meter, seed, ctx,
                                "reporting_eval" if kind == "reporting" else "exploration",
                                {"strategy": policy, "set": s["name"], "failure_prob": eps})
                finally:
                    meter.flush()
                meta = {"run_id": run_id, "seed": seed, "strategy": policy, "set": s["name"],
                        "set_kind": s["kind"], "observe_items": s["observe_items"],
                        "namespace": s["namespace"], "failure_prob": eps, "pool_size": s["pool_size"],
                        "max_true": s["max_true"], "arity_weights": list(s["arity_weights"])}
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
                            "worlds": len(recs), "seconds": round(time.time() - t0, 2)})
                print(f"{run_id}: {len(rows)} tasks, {meter.used} interactions, {len(recs)} worlds, "
                      f"{time.time() - t0:.1f}s", flush=True)
    return out


# ================================================================ evidence

def load_evidence(runs_dir: str | Path, prefix: str) -> tuple[list[dict], dict]:
    """World records of RL's runs ``<runs_dir>/<prefix>*``, one run per (set,
    policy, seed) chosen as X2 chooses (``markov_study.select_runs``: the
    earliest, or the earliest later completed run with the same config hash,
    commit and loaded code when it did not complete); every run folder with a
    manifest and a config counts. Returns (records, selection) with the reruns
    used, the ignored runs, the chosen runs without complete evidence and the
    (commit, code) pairs of the chosen runs."""
    runs = []
    for d in sorted(Path(runs_dir).glob(f"{prefix}*")):
        if not ((d / "manifest.json").exists() and (d / "config.json").exists()):
            continue
        man = json.loads((d / "manifest.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("study") != STUDY:
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


X2_SETS = ("gen_obs",)
X2_BENCH_SETS = ("bench_crafter", "bench_psketch_craft", "bench_msgi_mining", "bench_little_alchemy_2",
                 "bench_freeciv_classic")


def load_x2_evidence(runs_dir: str | Path, prefix: str,
                     sets=X2_SETS + X2_BENCH_SETS) -> tuple[list[dict], dict]:
    """X2's world records through X2's own loader (read only), restricted to
    ``sets`` (by default ``gen_obs`` and X2's five benchmark sets, which RL
    reports descriptively only); X2's logs are deterministic (slip probability 0)."""
    recs, sel = x2.load_evidence(runs_dir, prefix)
    out = [r for r in recs if r.get("set") in set(sets)]
    for r in out:
        r.setdefault("failure_prob", 0.0)
    return out, sel
