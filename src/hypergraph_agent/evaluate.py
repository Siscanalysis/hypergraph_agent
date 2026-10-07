"""Evaluate a saved agent (or a labelled baseline) on a declared track.

    python -m hypergraph_agent.evaluate --checkpoint runs/<run> --config configs/transfer_frozen.yaml
    python -m hypergraph_agent.evaluate --agent random --config configs/diagnostic_shallow.yaml

All environment interactions are charged to the session ledger: as
non-adaptive reporting evaluation by default, or as adaptive
``evaluation_selection`` when ``--selection`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .agents.baselines import RandomPolicy
from .checkpoints import load_agent
from .config import eval_stream_config, load_config
from .evaluation.metrics import aggregate
from .evaluation.protocol import evaluate
from .training.budget import BudgetMeter, SessionLedger
from .training.seeding import RNGStreams


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.evaluate")
    p.add_argument("--config", required=True, help="config whose eval section defines the track")
    p.add_argument("--checkpoint", help="run directory or checkpoint.pt")
    p.add_argument("--agent", choices=["learned", "random"], default="learned")
    p.add_argument("--selection", action="store_true",
                   help="this evaluation selects something: charge it as adaptive budget")
    p.add_argument("--max-interactions", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    ev = cfg["eval"]
    library = lib_snapshot = topo = None
    if args.agent == "learned":
        if not args.checkpoint:
            p.error("--checkpoint is required for a learned agent")
        policy, arm, topo, library, saved = load_agent(args.checkpoint)
        label = f"learned:{args.checkpoint}"
        saved_train = saved.get("train", {})
        if cfg["train"]["world_mode"] == "shared" and saved_train.get("world_mode") == "shared":
            # familiar mechanics: evaluate in the world the agent was trained in
            cfg["train"]["shared_world_seed"] = saved_train["shared_world_seed"]
            print(f"using the checkpoint's shared world seed {saved_train['shared_world_seed']}")
        if library is not None:
            lib_snapshot = library.snapshot()
    else:
        policy = RandomPolicy()
        arm = dict(cfg["arms"][0])
        label = "random"
    ledger = SessionLedger(cfg["run"]["ledger"])
    kind = "adaptive" if args.selection else "reporting"
    cap = args.max_interactions or cfg["budget"]["reporting_eval_per_run"]
    run_id = f"eval-{cfg['run']['name']}-{args.agent}-{time.strftime('%Y%m%d-%H%M%S')}"
    meter = BudgetMeter(ledger, run_id, cfg["run"]["allocation"], cap, kind=kind)
    streams = RNGStreams(0, f"evaluate/{label}")
    try:
        rows = evaluate(policy, arm, eval_stream_config(cfg), n_tasks=ev["n_tasks"], track=ev["track"],
                        meter=meter, action_gen=streams.actions, dyn_rng=streams.np["dynamics"],
                        greedy=ev["greedy"], episodes_per_world=ev["episodes_per_world"],
                        topo=topo if ev["track"] == "frozen" else None,
                        epsilon=cfg["train"]["tasks"]["failure_prob"], topo_cfg=cfg["topology"],
                        library=library, lib_snapshot=lib_snapshot,
                        live_contracts=bool(arm.get("revision", False)),
                        purpose="evaluation_selection" if args.selection else "reporting_eval")
    except NotImplementedError as exc:
        print(f"not_implemented: {exc}")
        return 2
    finally:
        meter.flush()
    out_dir = Path(cfg["run"]["runs_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    summary = {"agent": label, "track": ev["track"], "interactions": meter.used, "kind": kind,
               "summary": aggregate(rows), "by_depth": aggregate(rows, by="depth")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
