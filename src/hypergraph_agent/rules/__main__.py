"""Study RL commands (docs/studies/RL_rule_learners.md).

    python -m hypergraph_agent.rules collect --config configs/rule_learners/rl_dev.yaml --dry-run
    python -m hypergraph_agent.rules collect --config configs/rule_learners/rl_dev.yaml [--resume]
    python -m hypergraph_agent.rules collect --config configs/rule_learners/rl_main.yaml \\
        --allow-measurement [--resume] [--sets ooc] [--seeds 0 1]
    python -m hypergraph_agent.rules rank --prefix rl-main- --x2-prefix x2-main- \\
        --out artifacts/rule_learners [--workers 4] [--resume]

``collect`` plays the evidence policies and charges the study ledger
(``runs/RL-ledger.json``); with ``--resume`` it skips every (set, policy,
seed) that already has a completed run with the same config content, commit
and code, and it refuses any run whose stream sum exceeds the ledger's
remaining allowance. ``rank`` reads RL's evidence and study X2's (read only,
through X2's loader; only X2's set ``gen_obs``) and interacts with no
environment; it refuses evidence whose chosen runs come from more than one
(commit, code) within either study, caches every row under ``--cache`` (RL's
own folder, never X2's) and, with ``--resume``, reuses the rows of the same
code, options and library versions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _parser():
    import argparse
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.rules")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="trace collection (charges the ledger)")
    c.add_argument("--config", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--sets", nargs="*", help="only these sets (default: all)")
    c.add_argument("--seeds", type=int, nargs="*", help="only these seeds (default: all)")
    c.add_argument("--resume", action="store_true",
                   help="skip runs completed earlier with the same config content, commit and code")
    c.add_argument("--allow-measurement", action="store_true",
                   help="required by configs with run.requires_flag (measurement)")
    r = sub.add_parser("rank", help="offline benchmark of every learner (no interaction)")
    r.add_argument("--runs", default="runs")
    r.add_argument("--prefix", required=True, help="RL's runs, e.g. rl-main-")
    r.add_argument("--x2-runs", default=None, help="folder of X2's runs (default: --runs)")
    r.add_argument("--x2-prefix", required=True, help="X2's runs, e.g. x2-main-")
    r.add_argument("--out", default="artifacts/rule_learners")
    r.add_argument("--stem", default="rl_benchmark")
    r.add_argument("--cache", default="runs/rl_cache", help="row cache (not tracked; never X2's cache)")
    r.add_argument("--resume", action="store_true", help="reuse cached rows of the same key")
    r.add_argument("--checkpoints", type=int, nargs="*", default=None)
    r.add_argument("--workers", type=int, default=1, help="processes (per world)")
    r.add_argument("--expected-worlds", type=int, default=60, help="protocol: 60 per set (seeds 0-4)")
    r.add_argument("--x2-artifact", default="artifacts/inference_baselines/x2_benchmark.json",
                   help="X2's summary for check V4 (it applies only when it covers the X2 runs read)")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from . import collection as col
    if args.cmd == "collect":
        cfg = col.load(args.config)
        summary = col.plan(cfg)
        if args.dry_run:
            print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1))
            for x in summary["runs"]:
                print(f"  {x['set']:<8} s{x['seed']} {x['policy']:<15} {x['kind']:<9} "
                      f"worlds {x['worlds']:>3} tasks {x['tasks']:>4} cap {x['cap']}")
            return 0
        if cfg["run"]["requires_flag"] and not args.allow_measurement:
            print("this config is a measurement config: pass --allow-measurement")
            return 2
        if not summary["allocation_covers_worst_case"]:
            print("the allocation does not cover the worst case of these runs; refusing")
            return 2
        if not summary["ledger_file_matches"]:
            print(f"refused: {cfg['run']['ledger']} declares {summary['ledger_file_declaration']} and the "
                  f"config {col.declaration(cfg)}: amend the ledger (or the config) first")
            return 2
        from ..training.run import process_source
        process_source(Path(__file__).resolve().parents[3])
        ledger = col.make_ledger(cfg)
        try:
            res = col.collect(cfg, ledger, only_sets=args.sets, resume=args.resume, only_seeds=args.seeds)
        except col.AllowanceShort as exc:
            print(f"refused: {exc}")
            return 2
        out = Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
        with open(out, "a", encoding="utf-8") as fh:
            for x in res:
                fh.write(json.dumps(x, default=str) + "\n")
        print(json.dumps({"session_budget": ledger.summary()}, indent=1))
        return 0

    from . import analysis as an
    rl, rl_sel = col.load_evidence(args.runs, args.prefix)
    x2, x2_sel = col.load_x2_evidence(args.x2_runs or args.runs, args.x2_prefix)
    if not rl or not x2:
        print(f"no evidence: {len(rl)} RL world records under {args.prefix!r}, {len(x2)} X2 records under "
              f"{args.x2_prefix!r}")
        return 1
    for name, sel in (("RL", rl_sel), ("X2", x2_sel)):
        if len(sel["provenance"]) > 1:
            print(f"refused: {name}'s chosen runs come from more than one (commit, code): "
                  f"{sel['provenance']}")
            return 2
    if Path(args.cache).resolve() == (Path(args.x2_runs or args.runs) / "x2_cache").resolve():
        print("refused: the cache folder is X2's")
        return 2
    worlds = x2 + rl
    artifact = None
    if args.x2_artifact and Path(args.x2_artifact).exists():
        artifact = json.loads(Path(args.x2_artifact).read_text(encoding="utf-8"))
    t0 = time.time()
    summary = an.rank_benchmark(
        worlds, checkpoints=tuple(args.checkpoints or an.CHECKPOINTS), workers=args.workers,
        expected_worlds=args.expected_worlds, cache_dir=args.cache, resume=args.resume,
        x2_artifact=artifact, x2_chosen=x2_sel["chosen"],
        progress=lambda rec, k: print(f"{rec['set']} {rec['strategy']} {rec.get('world_label')} @{k}: "
                                      f"{time.time() - t0:.0f}s", flush=True))
    summary["runs"] = {"rl": rl_sel, "x2": x2_sel}
    summary["incomplete_worlds"] = [w.get("world_label") for w in worlds if not w["complete"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.stem}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    md = an.markdown(summary)
    (out / f"{args.stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
