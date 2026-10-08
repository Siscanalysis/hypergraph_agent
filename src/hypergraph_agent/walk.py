"""Run a hypergraph-walker study from a YAML config.

    python -m hypergraph_agent.walk --config configs/walker_stage_a.yaml --dry-run
    python -m hypergraph_agent.walk --config configs/walker_stage_a.yaml
    python -m hypergraph_agent.walk --config configs/replication/f1.yaml --seeds 0 1

The dry run also sums the task budgets of every (seed, variant) evaluation
stream from the generator alone: no run can use more interactions than that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents.walker import STRATEGIES
from .config import eval_stream_config, load_config
from .train import make_ledger


def plan(cfg: dict) -> dict:
    from .evaluation.walkers import NON_ADAPTIVE, stream_budget, variant_config
    wc = cfg["walker"]
    arms = [(a["id"], a["strategy"]) for a in cfg["arms"]]
    variants = wc["eval_variants"] or [{"name": "default", "tasks": {}}]
    seeds = cfg["run"]["seeds"]
    es = eval_stream_config(cfg)
    learned = any(s in ("learned", "learned_sample") for _, s in arms) and not wc["policy_from"]
    n_runs = len(arms) * len(variants) * len(seeds)
    sums = {f"s{seed}-{v['name']}": stream_budget(cfg, seed, v) for seed in seeds for v in variants}
    capped = sum(min(wc["per_run_cap"], s) for s in sums.values())
    n_adaptive = sum(s not in NON_ADAPTIVE for _, s in arms)
    return {
        "phase": cfg["run"]["phase"], "allocation": cfg["run"]["allocation"], "arms": arms,
        "variants": [{"name": v["name"], "failure_prob":
                      eval_stream_config(variant_config(cfg, seeds[0], v)).task.failure_prob} for v in variants],
        "seeds": seeds,
        "eval": {"namespace": es.namespace, "world_mode": es.world_mode, "n_worlds": es.n_worlds,
                 "episodes_per_world": cfg["eval"]["episodes_per_world"], "n_tasks": cfg["eval"]["n_tasks"],
                 "profile": es.task.profile, "observe_items": es.task.observe_items,
                 "depth": [es.task.depth_min, es.task.depth_max],
                 "base_seed_by_seed": {s: variant_config(cfg, s, variants[0])["eval"]["base_seed"] for s in seeds}},
        "caps": {"per_run": wc["per_run_cap"], "warmup_per_run": wc["warmup_interactions"],
                 "collector_per_seed": wc["collector_interactions"] if learned else 0,
                 "max_physical_total": n_runs * (wc["per_run_cap"] + wc["warmup_interactions"])
                 + (len(seeds) * wc["collector_interactions"] if learned else 0),
                 "stream_budget_sums": sums,
                 "per_run_cap_covers_every_stream": wc["per_run_cap"] >= max(sums.values()),
                 "worst_case_adaptive": n_adaptive * capped,
                 "worst_case_reporting": (len(arms) - n_adaptive) * capped},
        "search": {k: wc[k] for k in ("max_evals", "temperature", "restart_after", "exact_cap",
                                      "consistent_moves", "omit_prob")},
        "ledger": cfg["run"]["ledger"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.walk")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", help="run only these seeds (e.g. one process per seed)")
    p.add_argument("--arms", nargs="*", help="run only these arm ids")
    args = p.parse_args(argv)
    from .training.run import process_source
    process_source(Path(__file__).resolve().parents[2])
    cfg = load_config(args.config)
    if not args.dry_run and any(a["strategy"] not in STRATEGIES + ("reference",) for a in cfg["arms"]):
        # study M agents: refuse unset or invalid parameters before any run is registered, checked on the
        # whole config (before --arms), so no arm of a config with an unset Markov parameter can run
        from .agents.markov import markov_params
        try:
            markov_params(cfg["walker"]["markov"])
        except (KeyError, ValueError) as exc:
            p.error(f"walker.markov: {exc}")
    if args.seeds:
        cfg["run"]["seeds"] = args.seeds
    if args.arms:
        cfg["arms"] = [a for a in cfg["arms"] if a["id"] in set(args.arms)]
        if not cfg["arms"]:
            p.error("no configured arm matches --arms")
    if any(a["strategy"] is None for a in cfg["arms"]):
        p.error("every walker arm needs a strategy")
    summary = plan(cfg)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    if any(a["strategy"] == "random_omit" for a in cfg["arms"]) and cfg["walker"]["omit_prob"] is None:
        p.error("random_omit needs walker.omit_prob")
    from .evaluation.walkers import run_walker_study
    ledger = make_ledger(cfg)
    results = run_walker_study(cfg, ledger, cfg["run"]["runs_dir"])
    out = Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    print(json.dumps({"session_budget": ledger.summary()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
