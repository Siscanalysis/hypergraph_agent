"""Study M commands (docs/studies/M_markov_ranking.md).

    python -m hypergraph_agent.markov_study collect --config configs/markov/m1_collect.yaml --dry-run
    python -m hypergraph_agent.markov_study collect --config configs/markov/m1_collect.yaml
    python -m hypergraph_agent.markov_study rank --prefix markov-m1- --expected-test-worlds 36
    python -m hypergraph_agent.markov_study twins --out artifacts/markov
    python -m hypergraph_agent.markov_study factorial --prefix markov-m2 --expected-worlds 50

``collect`` plays the evidence arms and charges the study ledger; ``rank``,
``twins`` and ``factorial`` read recorded runs or constructed worlds and
interact with no environment. The M2 runs themselves go through
``python -m hypergraph_agent.walk --config configs/markov/m2.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _write(out: Path, stem: str, summary: dict, md: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if md is not None:
        (out / f"{stem}.md").write_text(md, encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.markov_study")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="M1 evidence collection (charges the ledger)")
    c.add_argument("--config", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--seeds", type=int, nargs="*")
    r = sub.add_parser("rank", help="M1 offline ranking benchmark")
    r.add_argument("--runs", default="runs")
    r.add_argument("--prefix", default="markov-m1-")
    r.add_argument("--out", default="artifacts/markov")
    r.add_argument("--stem", default="m1_ranking")
    r.add_argument("--checkpoints", type=int, nargs="*", default=None)
    r.add_argument("--starts", type=int, default=4)
    r.add_argument("--sweeps", type=int, default=40)
    r.add_argument("--burn", type=int, default=10)
    r.add_argument("--exact-work", type=int, default=2_000_000)
    r.add_argument("--selection-seeds", type=int, nargs="*", default=None, help="default: 0 1")
    r.add_argument("--test-seeds", type=int, nargs="*", default=None, help="default: 2 3 4")
    r.add_argument("--expected-test-worlds", type=int, default=36, help="protocol: 36 (seeds 2-4)")
    t = sub.add_parser("twins", help="structural and behavioural twins (constructed worlds)")
    t.add_argument("--out", default="artifacts/markov")
    t.add_argument("--restart", type=float, default=0.15)
    f = sub.add_parser("factorial", help="M2 factorial endpoints from recorded walker runs")
    f.add_argument("--runs", default="runs")
    f.add_argument("--prefix", default="markov-m2")
    f.add_argument("--out", default="artifacts/markov")
    f.add_argument("--stem", default="m2_factorial")
    f.add_argument("--expected-worlds", type=int, default=50, help="protocol: 50")
    args = p.parse_args(argv)

    from .evaluation import markov_study as ms
    if args.cmd == "collect":
        from .config import load_config
        from .train import make_ledger
        from .training.run import process_source
        process_source(Path(__file__).resolve().parents[2])
        cfg = load_config(args.config)
        if args.seeds:
            cfg["run"]["seeds"] = args.seeds
        summary = ms.collect_plan(cfg)
        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return 0
        ledger = make_ledger(cfg)
        results = ms.collect(cfg, ledger, cfg["run"]["runs_dir"])
        out = Path(cfg["run"]["runs_dir"]) / f"{cfg['run']['name']}-results.jsonl"
        with open(out, "a", encoding="utf-8") as fh:
            for x in results:
                fh.write(json.dumps(x, default=str) + "\n")
        print(json.dumps({"session_budget": ledger.summary()}, indent=1))
        return 0
    if args.cmd == "rank":
        worlds, runs_used = ms.load_evidence(args.runs, args.prefix)
        if not worlds:
            print(f"no evidence under {args.runs} with prefix {args.prefix!r}")
            return 1
        t0 = time.time()
        summary = ms.rank_benchmark(
            worlds, checkpoints=tuple(args.checkpoints or ms.CHECKPOINTS),
            selection_seeds=tuple(ms.SELECTION_SEEDS if args.selection_seeds is None
                                  else args.selection_seeds),
            test_seeds=tuple(ms.TEST_SEEDS if args.test_seeds is None else args.test_seeds),
            expected_test_worlds=args.expected_test_worlds,
            posterior_kw={"starts": args.starts, "sweeps": args.sweeps, "burn": args.burn,
                          "exact_work": args.exact_work},
            progress=lambda rec, k: print(f"{rec['strategy']} s{rec['seed']} {rec['world_key']} @{k}: "
                                          f"{time.time() - t0:.0f}s", flush=True))
        summary["seconds"] = round(time.time() - t0, 1)
        summary["runs"] = runs_used
        summary["incomplete_worlds"] = [f"{w['run_id']}:{w['world_key']}" for w in worlds
                                        if not w["complete"]]
        _write(Path(args.out), args.stem, summary, ms.markdown_m1(summary))
        print(ms.markdown_m1(summary))
        return 0
    if args.cmd == "twins":
        summary = {"structural": ms.structural_twins(args.restart),
                   "behavioural": ms.behavioural_twins(args.restart)}
        _write(Path(args.out), "twins", summary, None)
        print(json.dumps(summary, indent=1, default=str))
        return 0
    runs, runs_used = ms.load_runs(args.runs, args.prefix)
    if not runs:
        print(f"no runs with prefix {args.prefix!r} in {args.runs}")
        return 1
    summary = ms.factorial(runs, args.expected_worlds)
    summary["runs"] = runs_used
    _write(Path(args.out), args.stem, summary, ms.markdown_m2(summary))
    print(ms.markdown_m2(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
