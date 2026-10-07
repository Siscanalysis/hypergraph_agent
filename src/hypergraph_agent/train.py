"""Train agents from a YAML config.

    python -m hypergraph_agent.train --config configs/smoke.yaml
    python -m hypergraph_agent.train --config configs/study.yaml --dry-run

Every command shares the persistent session ledger named in the config, so
separate phase commands and restarts draw on one interaction allowance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import eval_stream_config, load_config, train_stream_config
from .training.budget import SessionLedger


def make_ledger(cfg: dict) -> SessionLedger:
    b = cfg["budget"]
    kw = {}
    if b["ledger_allocations"]:
        kw["allocations"] = dict(b["ledger_allocations"])
    if b["ledger_adaptive_cap"]:
        kw["adaptive_cap"] = b["ledger_adaptive_cap"]
    if b["ledger_reporting_cap"]:
        kw["reporting_cap"] = b["ledger_reporting_cap"]
    return SessionLedger(cfg["run"]["ledger"], **kw)


def plan(cfg: dict) -> dict:
    run, budget = cfg["run"], cfg["budget"]
    arms = [a["id"] for a in cfg["arms"]]
    seeds = list(run["seeds"])
    per_run = budget["per_run_interactions"]
    pre = budget.get("pretraining_interactions", 0)
    n_runs = len(arms) * len(seeds)
    physical = n_runs * per_run + len(seeds) * pre
    logical_per_arm = per_run + pre
    ts = train_stream_config(cfg, seeds[0] if seeds else 0)
    es = eval_stream_config(cfg)
    return {
        "phase": run["phase"], "allocation": run["allocation"], "arms": arms, "seeds": seeds,
        "train_split": {"namespace": "train", "world_mode": ts.world_mode, "n_worlds": ts.n_worlds,
                        "depth": [ts.task.depth_min, ts.task.depth_max],
                        "goal_levels": ts.goal_levels, "profile": ts.task.profile,
                        "failure_prob": ts.task.failure_prob},
        "eval_split": {"namespace": es.namespace, "base_seed": es.base_seed,
                       "n_tasks": cfg["eval"]["n_tasks"], "track": cfg["eval"]["track"],
                       "depth": [es.task.depth_min, es.task.depth_max], "goal_levels": es.goal_levels},
        "regimes": {"reward": "terminal success only (P3 skill practice adds a logged public "
                              "target-predicate objective)",
                    "masks": "padding only; no feasibility masks", "gamma": cfg["ppo"]["gamma"]},
        "interactions": {"per_run_adaptive_cap": per_run, "shared_pretraining_per_seed": pre,
                         "physical_adaptive_total": physical,
                         "logical_all_in_per_arm": logical_per_arm,
                         "reporting_eval_per_run_cap": budget["reporting_eval_per_run"],
                         "reporting_eval_total_cap": budget["reporting_eval_per_run"] * n_runs},
        "skills": cfg.get("skills", {}),
        "output": str(Path(run["runs_dir"]) / f"{run['name']}-<arm>-s<seed>-<time>"),
        "ledger": run["ledger"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.train")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the plan and costs, run nothing")
    p.add_argument("--allow-full-study", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", help="override the configured seeds")
    p.add_argument("--arms", nargs="*", help="run only these arm ids")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seeds:
        cfg["run"]["seeds"] = args.seeds
    if args.arms:
        cfg["arms"] = [a for a in cfg["arms"] if a["id"] in set(args.arms)]
        if not cfg["arms"]:
            p.error("no configured arm matches --arms")
    summary = plan(cfg)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    if cfg["run"]["requires_flag"] == "allow_full_study" and not args.allow_full_study:
        p.error("this is a full-study preset: pass --allow-full-study to launch it, or --dry-run")

    ledger = make_ledger(cfg)
    runs_dir = cfg["run"]["runs_dir"]
    phase = cfg["run"]["phase"]
    results = []
    if phase in ("smoke", "p1", "p2", "diagnostic"):
        from .training.flat import train_flat_arm
        for seed in cfg["run"]["seeds"]:
            for arm in cfg["arms"]:
                res = train_flat_arm(cfg, arm, seed, ledger, runs_dir)
                results.append(res)
                ev = res["eval"]
                print(f"{res['run_id']}: eval success {ev.get('success_rate', float('nan')):.3f} "
                      f"(n={ev.get('n', 0)})", flush=True)
    elif phase == "p3":
        from .training.phase3 import run_phase3
        results = run_phase3(cfg, ledger, runs_dir)
    else:
        p.error(f"unknown phase {phase!r}")
    out = Path(runs_dir) / f"{cfg['run']['name']}-results.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    print(json.dumps({"session_budget": ledger.summary()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
