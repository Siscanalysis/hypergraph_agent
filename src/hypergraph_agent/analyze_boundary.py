"""Study R2 analyses (docs/studies/R2_boundary.md).

    python -m hypergraph_agent.analyze_boundary families --out artifacts/boundary
    python -m hypergraph_agent.analyze_boundary endpoints --runs runs --prefix r2- --out artifacts/boundary

``families`` computes the family statistics and the W2 guarantee of the
measurement configs in ``configs/boundary`` (every file not starting with
``dev_``) from the generator, the reference solver and the brute-force plan; it
interacts with no environment. ``endpoints`` reads recorded runs and writes
``boundary_summary.json``, ``boundary_summary.md`` and
``boundary_cost_by_episode.png`` (see ``evaluation.boundary``). A contrast
needs ``--expected-units`` paired worlds to count as complete: 50 in the
protocol; pass a smaller number for development runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluation.boundary import families_markdown, write_families, write_report
from .evaluation.replication import N_BOOT


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.analyze_boundary")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("families", help="family statistics from the generator (no interaction)")
    f.add_argument("--configs", nargs="*", help="configs to describe (default: the measurement configs)")
    f.add_argument("--out", default=None, help="directory for boundary_families.json and .md")
    e = sub.add_parser("endpoints", help="endpoints from recorded runs")
    e.add_argument("--runs", default="runs", help="directory holding the run directories")
    e.add_argument("--prefix", default="r2-", help="run-name prefix of the study's runs")
    e.add_argument("--out", default="artifacts/boundary")
    e.add_argument("--expected-units", type=int, default=50, help="paired worlds a contrast needs to count as complete")
    e.add_argument("--n-boot", type=int, default=N_BOOT)
    e.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "families":
        root = Path(__file__).resolve().parents[2] / "configs" / "boundary"
        paths = args.configs or sorted(str(x) for x in root.glob("*.yaml") if not x.name.startswith("dev_"))
        if not paths:
            print("no configs")
            return 1
        stats = write_families(paths, args.out)
        print(families_markdown(stats), end="")
        bad = [k for k, s in stats.items() if not s["guarantee"]["holds"]]
        if bad:
            print(f"W2 guarantee fails in: {', '.join(bad)}")
            return 1
        return 0

    summary = write_report(args.runs, args.prefix, args.out, args.expected_units, args.n_boot, not args.no_figure)
    if not summary["families"]:
        print(f"no runs with prefix {args.prefix!r} in {args.runs}")
        return 1
    fmt = lambda k: "-" if not k.get("ci") else f"{k['estimate']:.3f} [{k['ci'][0]:.3f}, {k['ci'][1]:.3f}]"
    for fam, c in summary["families"].items():
        for name, k in c["contrasts"].items():
            print(f"{fam} ({c['role']}, {c['kind']}) {name}: {k['n_units']} worlds, late difference {fmt(k)} "
                  f"{c['reading']}; DiD {fmt(k['did'])}; break-even {c['break_even_episode']}")
    for fam, c in summary["families"].items():
        if not c["contrasts"]:
            print(f"{fam} ({c['role']}, {c['kind']}): {c['reading']} (no usable bounds)")
    for label, k in summary["slack_contrasts"].items():
        print(f"{label}: change of the late difference {fmt(k) if k['paired'] else 'not paired'} {k['reading']}")
    print(f"primary {summary['primary_family']}: {summary['primary_reading']}; "
          f"depth against headroom: {summary['depth_against_headroom']['reading']}")
    for c in summary["caveats"]:
        print(f"caveat: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
