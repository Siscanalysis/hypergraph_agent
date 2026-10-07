"""Run a hypergraph-walker study from a YAML config.

    python -m hypergraph_agent.walk --config configs/walker_stage_a.yaml --dry-run
    python -m hypergraph_agent.walk --config configs/walker_stage_a.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import eval_stream_config, load_config
from .train import make_ledger


def plan(cfg: dict) -> dict:
    wc = cfg["walker"]
    arms = [(a["id"], a["strategy"]) for a in cfg["arms"]]
    variants = [v["name"] for v in wc["eval_variants"]] or ["default"]
    seeds = cfg["run"]["seeds"]
    es = eval_stream_config(cfg)
    learned = any(s == "learned" for _, s in arms)
    n_runs = len(arms) * len(variants) * len(seeds)
    return {
        "phase": cfg["run"]["phase"], "allocation": cfg["run"]["allocation"], "arms": arms,
        "variants": wc["eval_variants"] or [{"name": "default"}], "seeds": seeds,
        "eval": {"namespace": es.namespace, "world_mode": es.world_mode, "n_worlds": es.n_worlds,
                 "episodes_per_world": cfg["eval"]["episodes_per_world"], "n_tasks": cfg["eval"]["n_tasks"],
                 "profile": es.task.profile, "observe_items": es.task.observe_items,
                 "failure_prob": es.task.failure_prob},
        "caps": {"per_run": wc["per_run_cap"], "warmup_per_run": wc["warmup_interactions"],
                 "collector_per_seed": wc["collector_interactions"] if learned else 0,
                 "max_physical_total": n_runs * (wc["per_run_cap"] + wc["warmup_interactions"])
                 + (len(seeds) * wc["collector_interactions"] if learned else 0)},
        "search": {k: wc[k] for k in ("max_evals", "temperature", "restart_after", "exact_cap")},
        "ledger": cfg["run"]["ledger"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.walk")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    from .training.run import process_source
    process_source(Path(__file__).resolve().parents[2])
    cfg = load_config(args.config)
    if any(a["strategy"] is None for a in cfg["arms"]):
        p.error("every walker arm needs a strategy")
    summary = plan(cfg)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
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
