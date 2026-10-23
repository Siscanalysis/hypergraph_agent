"""Command line of study X.

    python -m hypergraph_agent.baselines run --config configs/baselines/rq_f1_dev.yaml --dry-run
    python -m hypergraph_agent.baselines run --config configs/baselines/rq_f1_dev.yaml --learners mppo
    python -m hypergraph_agent.baselines run --config configs/baselines/rq_f1_main.yaml --allow-measurement \
        --workers 8
    python -m hypergraph_agent.baselines analyze --config configs/baselines/rq_f1_main.yaml
    python -m hypergraph_agent.baselines apply-tuning --config configs/baselines/rq_f1_main.yaml

``run`` trains every (learner, training seed) of the config unless a completed
run exists, evaluates its checkpoints, then runs the repository's agents on
the test stream. ``--learners``, ``--seeds`` and ``--agents`` select a subset
(``--agents`` with no name runs no agent); ``--workers`` runs jobs in parallel
processes (one torch thread each); a measurement config needs
``--allow-measurement`` and refuses to start when any planned job already has a
completed run, unless ``--resume``. ``analyze`` writes the endpoints to
``artifacts/baselines/<name>/`` (or ``--out``). ``apply-tuning`` writes the tuning
selection into a measurement config (``tuning.py``); a measurement run refuses
unless its learners reproduce that record. ``run`` exits with status 1 when any
job failed (the other jobs still run).

Study XS: ``analyze --config configs/baselines/xs_rq.yaml`` (a study file with the
single top-level key ``xs``) writes the combined analysis of its member configs
(``xs.py``); each member config is run on its own with ``run``. ``check-budget``
(a study file or a config) exits with status 1 unless the ledger still covers the
caps of every planned job without a completed run (``xs.budget_check``); study
XS's driver runs it before every launch or resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.baselines")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--dry-run", action="store_true", help="print jobs, caps and stream budgets; run nothing")
    r.add_argument("--learners", nargs="*", help="only these learner ids")
    r.add_argument("--seeds", type=int, nargs="*", help="only these training seeds")
    r.add_argument("--agents", nargs="*", help="only these repository agents (none when given empty)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--allow-measurement", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--config", required=True)
    a.add_argument("--out", default=None)
    t = sub.add_parser("apply-tuning")
    t.add_argument("--config", required=True)
    b = sub.add_parser("check-budget")
    b.add_argument("--config", required=True)
    args = p.parse_args(argv)

    from ..training.run import process_source
    process_source(Path(__file__).resolve().parents[3])
    if args.cmd == "apply-tuning":
        from .tuning import TuningError, apply_tuning
        try:
            print(json.dumps(apply_tuning(args.config), indent=1))
        except TuningError as exc:
            print(f"apply-tuning refused: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "check-budget":
        from .xs import budget_check
        res = budget_check(args.config)
        print(json.dumps(res, indent=1))
        if not res["ok"]:
            print(f"check-budget refused: the ledger has {res['left']} left for allocation "
                  f"{res['allocation']!r}, the jobs without a completed run need up to {res['need']}; "
                  f"further budget only through a dated amendment", file=sys.stderr)
            return 1
        return 0
    from .xs import is_study
    if is_study(args.config):  # study XS: the combined analysis of member configs
        if args.cmd != "analyze":
            p.error("a study file is only analysed; run each of its member configs")
        from .xs import write_study_report
        summary = write_study_report(args.config, args.out)
        print(json.dumps({"out": args.out or str(Path("artifacts") / "baselines" / summary["name"]),
                          "best_learner": summary["selection"]["best_learner"],
                          "tests": {t["name"]: t.get("verdict") for t in summary["tests"]}}, indent=1))
        return 0
    from .config import load_config
    cfg = load_config(args.config)
    if args.cmd == "analyze":
        from .analysis import write_report
        out = args.out or str(Path("artifacts") / "baselines" / cfg["run"]["name"])
        summary = write_report(cfg, out)
        print(json.dumps({"out": out, "tests": {t["name"]: t.get("verdict") for t in summary["tests"]}},
                         indent=1))
        return 0

    from .runner import make_ledger, plan, results_path, run_study
    subset = {"learners": args.learners or None, "seeds": args.seeds,
              "agents": None if args.agents is None else set(args.agents)}
    if args.learners is not None and not set(args.learners) <= {ln["id"] for ln in cfg["learners"]}:
        p.error("unknown learner id in --learners")
    summary = plan(cfg, **subset)
    if args.dry_run:
        print(json.dumps(summary, indent=2, default=str))
        return 0
    if cfg["run"]["requires_flag"] and not args.allow_measurement:
        p.error("this is a measurement config: pass --allow-measurement to launch it, or --dry-run")
    if cfg["run"]["requires_flag"]:
        from .tuning import TuningError, verify_tuning
        try:
            verify_tuning(cfg)
        except TuningError as exc:
            p.error(f"{exc} (a measurement run needs the selection written by apply-tuning)")
    make_ledger(cfg)  # creates the ledger file with the declared caps if it does not exist
    out = results_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)

    def append(results):  # each task's results as it finishes, so an interruption loses none
        with open(out, "a", encoding="utf-8") as fh:
            for res in results:
                fh.write(json.dumps(res, default=str) + "\n")

    failures = []
    results = run_study(cfg, resume=args.resume, workers=args.workers, failures_out=failures,
                        on_result=append, **subset)
    print(json.dumps({"runs": len(results), "failed": len(failures), "ledger": make_ledger(cfg).summary()},
                     indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
