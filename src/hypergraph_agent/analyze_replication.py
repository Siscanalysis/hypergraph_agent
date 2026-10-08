"""Frozen endpoints of study R from recorded runs.

    python -m hypergraph_agent.analyze_replication --runs runs --prefix rep- --out artifacts/replication

Writes ``replication_summary.json``, ``replication_summary.md`` and
``replication_cost_by_episode.png`` (see ``evaluation.replication``). A
contrast needs ``--expected-units`` paired worlds (blocks in F0) to count as
complete: 50 in the protocol; pass a smaller number for development runs.
"""

from __future__ import annotations

import argparse
import sys

from .evaluation.replication import N_BOOT, write_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.analyze_replication")
    p.add_argument("--runs", default="runs", help="directory holding the run directories")
    p.add_argument("--prefix", default="rep-", help="run-name prefix of the study's runs")
    p.add_argument("--out", default="artifacts/replication")
    p.add_argument("--expected-units", type=int, default=50,
                   help="paired worlds (blocks in F0) a contrast needs to count as complete")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)
    summary = write_report(args.runs, args.prefix, args.out, args.expected_units, args.n_boot,
                           not args.no_figure)
    if not summary["cells"]:
        print(f"no runs with prefix {args.prefix!r} in {args.runs}")
        return 1
    fmt = lambda k: "-" if not k.get("ci") else f"{k['estimate']:.3f} [{k['ci'][0]:.3f}, {k['ci'][1]:.3f}]"
    for label, c in summary["cells"].items():
        for name, k in c["contrasts"].items():
            print(f"{label} ({c['role']}) {name}: {k['n_units']} {c['units']}, late difference {fmt(k)} "
                  f"{'descriptive' if c['role'] == 'control' else k['verdict']}; D {fmt(k['did'])}")
        print(f"{label}: break-even episode {c['break_even_episode']}")
    for eps, a in summary["amortization"].items():
        print(f"amortization at {eps} ({a['role']}): supported {a['supported']}")
    print(f"primary: {summary['primary_verdict']}; general: {summary['general']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
