"""Study X2 commands (docs/studies/X2_inference_baselines.md).

    python -m hypergraph_agent.inference_baselines collect \\
        --config configs/inference_baselines/dev.yaml --dry-run
    python -m hypergraph_agent.inference_baselines collect --config configs/inference_baselines/dev.yaml
    python -m hypergraph_agent.inference_baselines collect \\
        --config configs/inference_baselines/x2_main.yaml --allow-measurement [--resume]
    python -m hypergraph_agent.inference_baselines rank --prefix x2-main- \\
        --out artifacts/inference_baselines [--resume] [--arms-hash-seeds 1 2 3 4]

``collect`` plays the evidence policies and charges the study ledger; with
``--resume`` it skips every (set, policy, seed) that already has a completed
run with the same config content, commit and code, and it refuses any run
whose stream sum exceeds the ledger's remaining allowance. ``rank`` reads
recorded evidence and interacts with no environment; it refuses evidence
from more than one (commit, code), caches every row and ARMS result under
``--cache`` and, with ``--resume``, reuses the entries of the same code,
options and library versions. ``--arms-hash-seeds`` adds the exploratory
check of ARMS under other string hash seeds, written to a separate file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _parser():
    import argparse
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.inference_baselines")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="trace collection (charges the ledger)")
    c.add_argument("--config", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--sets", nargs="*", help="only these sets (default: all)")
    c.add_argument("--resume", action="store_true",
                   help="skip runs completed earlier with the same config content and code")
    c.add_argument("--allow-measurement", action="store_true",
                   help="required by configs with run.requires_flag (measurement)")
    r = sub.add_parser("rank", help="offline benchmark of every method (no interaction)")
    r.add_argument("--runs", default="runs")
    r.add_argument("--prefix", required=True)
    r.add_argument("--out", default="artifacts/inference_baselines")
    r.add_argument("--stem", default="x2_benchmark")
    r.add_argument("--cache", default="runs/x2_cache", help="row and ARMS cache (not tracked)")
    r.add_argument("--resume", action="store_true", help="reuse cached rows and ARMS results of the same key")
    r.add_argument("--checkpoints", type=int, nargs="*", default=None)
    r.add_argument("--arms-workers", type=int, default=1, help="ARMS worker processes")
    r.add_argument("--workers", type=int, default=1, help="processes for the other methods (per world)")
    r.add_argument("--expected-test-worlds", type=int, default=36, help="protocol: 36 (seeds 2-4)")
    r.add_argument("--expected-selection-worlds", type=int, default=24, help="protocol: 24 (seeds 0-1)")
    r.add_argument("--arms-hash-seeds", type=int, nargs="*", default=None,
                   help="exploratory: ARMS on the selection worlds under these string hash seeds "
                        "(written to <stem>_arms_hash_seeds.json)")
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
                print(f"  {x['set']:<16} s{x['seed']} {x['policy']:<15} {x['kind']:<9} "
                      f"worlds {x['worlds']:>3} tasks {x['tasks']:>4} cap {x['cap']}")
            return 0
        if cfg["run"]["requires_flag"] and not args.allow_measurement:
            print("this config is a measurement config: pass --allow-measurement")
            return 2
        if not summary["allocation_covers_worst_case"]:
            print("the allocation does not cover the worst case of these runs; refusing")
            return 2
        from ..training.run import process_source
        process_source(Path(__file__).resolve().parents[3])
        ledger = col.make_ledger(cfg)
        try:
            res = col.collect(cfg, ledger, only_sets=args.sets, resume=args.resume)
        except col.AllowanceShort as exc:
            print(f"refused: {exc}")
            return 2
        out = Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
        with open(out, "a", encoding="utf-8") as fh:
            for x in res:
                fh.write(json.dumps(x, default=str) + "\n")
        print(json.dumps({"session_budget": ledger.summary()}, indent=1))
        return 0

    from . import benchmark as bm
    worlds, runs_used = col.load_evidence(args.runs, args.prefix)
    if not worlds:
        print(f"no X2 evidence under {args.runs} with prefix {args.prefix!r}")
        return 1
    if len(runs_used["provenance"]) > 1:
        print(f"refused: the chosen runs come from more than one (commit, code): {runs_used['provenance']}")
        return 2
    t0 = time.time()
    summary = bm.rank_benchmark(
        worlds, checkpoints=tuple(args.checkpoints or bm.CHECKPOINTS), arms_workers=args.arms_workers,
        workers=args.workers, expected_test_worlds=args.expected_test_worlds,
        expected_selection_worlds=args.expected_selection_worlds, cache_dir=args.cache, resume=args.resume,
        progress=lambda rec, k: print(f"{rec['set']} {rec['strategy']} {rec['world_label']} @{k}: "
                                      f"{time.time() - t0:.0f}s", flush=True))
    summary["seconds"] = round(time.time() - t0, 1)
    summary["runs"] = runs_used
    summary["incomplete_worlds"] = [w["world_label"] for w in worlds if not w["complete"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.stem}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    md = bm.markdown(summary)
    (out / f"{args.stem}.md").write_text(md, encoding="utf-8")
    print(md)
    bad = {s: n for s, n in summary["arms_status_totals"].items() if s in ("timeout", "worker_exit") and n}
    if bad:
        print(f"WARNING: ARMS jobs without a model because of the wall clock or a dead worker: {bad} "
              f"(cached; reused by --resume)")
    if args.arms_hash_seeds:
        check = bm.arms_seed_check(worlds, summary, args.arms_hash_seeds, arms_workers=args.arms_workers)
        (out / f"{args.stem}_arms_hash_seeds.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
        print(json.dumps(check, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
