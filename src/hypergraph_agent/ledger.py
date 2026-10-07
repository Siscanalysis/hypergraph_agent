"""Inspect or initialize the session budget ledger.

    python -m hypergraph_agent.ledger show --ledger runs/ledger.json
    python -m hypergraph_agent.ledger init --ledger runs/ledger.json --allocations '{"p1": 27000}'
    python -m hypergraph_agent.ledger record --ledger runs/ledger.json --run-id X \
        --allocation p1 --adaptive 100 --reporting 10 --note "..."

``record`` enters interactions that were executed outside the ledger (for
example development runs before the protocol was frozen) so the session total
stays complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .training.budget import ADAPTIVE_CAP, REPORTING_CAP, SessionLedger


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.ledger")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("show", "init", "record"):
        s = sub.add_parser(name)
        s.add_argument("--ledger", default="runs/ledger.json")
    sub.choices["init"].add_argument("--allocations", required=True, help="JSON object")
    sub.choices["init"].add_argument("--adaptive-cap", type=int, default=ADAPTIVE_CAP)
    sub.choices["init"].add_argument("--reporting-cap", type=int, default=REPORTING_CAP)
    r = sub.choices["record"]
    r.add_argument("--run-id", required=True)
    r.add_argument("--allocation", required=True)
    r.add_argument("--adaptive", type=int, default=0)
    r.add_argument("--reporting", type=int, default=0)
    r.add_argument("--note", default="")
    args = p.parse_args(argv)

    path = Path(args.ledger)
    if args.cmd == "init":
        if path.exists():
            p.error(f"{path} exists; a ledger is never re-initialized")
        led = SessionLedger(path, json.loads(args.allocations), args.adaptive_cap, args.reporting_cap)
        led.save()
    elif args.cmd == "record":
        led = SessionLedger(path)
        led.register_run(args.run_id, args.allocation, args.adaptive, {"external": True, "note": args.note})
        if args.adaptive:
            led.record(args.run_id, args.allocation, args.adaptive, {"external": args.adaptive}, "adaptive")
        if args.reporting:
            led.record(args.run_id, args.allocation, args.reporting, {"external": args.reporting}, "reporting")
    led = SessionLedger(path)
    print(json.dumps({"summary": led.summary(), "runs": {k: {"used": v.get("used"), "allocation": v.get("allocation")}
                                                         for k, v in led.state["runs"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
