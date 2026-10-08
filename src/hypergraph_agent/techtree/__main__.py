"""Run a TechTree study (U or L) from a YAML config.

    python -m hypergraph_agent.techtree --config configs/unlock/u_dev.yaml --dry-run
    python -m hypergraph_agent.techtree --config configs/unlock/u_dev.yaml
    python -m hypergraph_agent.techtree --config configs/layers/l1_main.yaml --allow-measurement

A measurement config runs whole: ``--arms`` and ``--variants`` subsets are
refused together with ``--allow-measurement``, and it refuses to start when its
runs already exist complete, unless ``--resume``, which repeats only missing or
incomplete runs (the analysis then uses such a rerun and lists it, provided the
code is unchanged).

Results: run directories under ``runs/`` and ``runs/<name>-results.jsonl``;
analyse with ``python -m hypergraph_agent.techtree.analysis --config <same>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.techtree")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the plan and budget totals, run nothing")
    p.add_argument("--variants", nargs="*", help="run only these variants (development only)")
    p.add_argument("--arms", nargs="*", help="run only these arms (development only)")
    p.add_argument("--resume", action="store_true", help="skip runs that already exist complete")
    p.add_argument("--allow-measurement", action="store_true")
    args = p.parse_args(argv)
    if args.allow_measurement and (args.arms or args.variants):
        p.error("a measurement config runs whole: --arms/--variants cannot be combined with "
                "--allow-measurement")
    from ..training.run import process_source
    process_source(Path(__file__).resolve().parents[3])
    from .config import load_config, variant_arms, variants
    from .study import make_ledger, plan, results_path, run_study
    cfg = load_config(args.config)
    if args.arms:
        keep = set(args.arms)
        cfg["variants"] = [{**v, "arms": [a for a in variant_arms(cfg, v) if a in keep]}
                           for v in variants(cfg)]
        cfg["arms"] = [a for a in cfg["arms"] if a in keep]
        if not cfg["arms"]:
            p.error("no configured arm matches --arms")
    if args.variants:
        cfg["variants"] = [v for v in variants(cfg) if v["name"] in set(args.variants)]
        if not cfg["variants"]:
            p.error("no configured variant matches --variants")
    summary = plan(cfg)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    if cfg["run"]["requires_flag"] and not args.allow_measurement:
        p.error("this is a measurement config: pass --allow-measurement to launch it, or --dry-run")
    ledger = make_ledger(cfg)
    results = run_study(cfg, ledger, cfg["run"]["runs_dir"], resume=args.resume)
    with open(results_path(cfg), "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    print(json.dumps({"ledger": ledger.summary()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
