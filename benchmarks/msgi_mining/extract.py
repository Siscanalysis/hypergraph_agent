"""Rebuild ``mining_graph.json`` from the subtask-graph files of the MSGI repository.

The Mining domain of Sohn et al. (ICLR 2020) stores its subtask graphs as
PyTorch pickles (``torch.save`` of lists of dicts with ``ANDmat``, ``ORmat``,
``rmag``, ``trind``, ``W_a``, ``W_o``). This script

1. downloads ``full_mining.pkl`` (and, with ``--tasks``, the train/eval task
   files) into ``data/`` with ``fetch.py`` and verifies their SHA-256;
2. loads them with ``torch.load(weights_only=True)`` and an explicit allowlist
   of the NumPy reconstruction functions the pickles reference, so no other
   code in the pickle can run;
3. reads every graph as rules ``subtask <- OR of AND-nodes`` (a negative entry
   in ``ANDmat`` would be a negated precondition; Mining has none, and the
   script refuses graphs that have one), checks that all graphs of the file
   are sub-graphs of one rule set (the same rule for every subtask a graph
   contains) and writes that rule set with the subtask names, operations and
   objects of ``grid-world/environment/mining.py``.

    python benchmarks/msgi_mining/extract.py            # rewrite mining_graph.json
    python benchmarks/msgi_mining/extract.py --check    # compare with the committed file
    python benchmarks/msgi_mining/extract.py --tasks    # also data/tasks_<file>.json

Requires torch and numpy (dependencies of this repository).
"""

from __future__ import annotations

import argparse
import codecs
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "e665861f2d08f41b7dad16588447203e5010145a"
OPERATIONS = {4: "pickup", 5: "transform"}  # action index of the operation in mining.py
# grid-world/environment/mining.py at COMMIT (MIT, Copyright (c) 2020 Hyunjae Woo):
# object_list (oid -> name) and subtask_list (name, (operation, oid)), in id order.
OBJECTS = ("workspace", "furnace", "tree", "stone", "grass", "pig", "coal", "iron", "silver", "gold",
           "diamond", "jeweler", "lumbershop")
SUBTASKS = (
    ("Cut wood", (4, 2)), ("Get stone", (4, 3)), ("Get string", (4, 4)),
    ("Make firewood", (5, 12)), ("Make stick", (5, 12)), ("Make arrow", (5, 12)), ("Make bow", (5, 12)),
    ("Make stone pickaxe", (5, 0)), ("Hit pig", (4, 5)),
    ("Get coal", (4, 6)), ("Get iron ore", (4, 7)), ("Get silver ore", (4, 8)),
    ("Light furnace", (5, 1)),
    ("Smelt iron", (5, 1)), ("Smelt silver", (5, 1)), ("Bake pork", (5, 1)),
    ("Make iron pickaxe", (5, 0)), ("Make silverware", (5, 0)),
    ("Get gold ore", (4, 9)), ("Get diamond ore", (4, 10)),
    ("Smelt gold", (5, 1)), ("Craft silver diamond earrings", (5, 11)), ("Craft iron diamond rings", (5, 11)),
    ("Make goldware", (5, 1)), ("Craft gold diamond necklace", (5, 12)), ("Make electrum bracelet", (5, 1)),
)
TASK_FILES = tuple(f"{s}_mining_{i}" for s in ("train1", "eval1") for i in (1, 2, 3, 4))


def load_graphs(path: Path) -> list:
    import numpy as np
    import torch
    try:
        from numpy._core import multiarray as ma
    except ImportError:  # numpy < 2
        from numpy.core import multiarray as ma
    allow = [(ma._reconstruct, "numpy.core.multiarray._reconstruct"), (np.ndarray, "numpy.ndarray"),
             (np.dtype, "numpy.dtype"), codecs.encode]
    allow += [type(np.dtype(t)) for t in ("float64", "float32", "float16", "int64", "int32", "int16",
                                          "int8", "uint8", "uint16", "bool")]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, weights_only=True)


def rules_of(graph) -> dict[int, list[list[int]]]:
    """subtask id -> list of AND-nodes (sorted subtask ids); [] for a subtask
    without preconditions."""
    A = graph["ANDmat"].numpy()
    O = graph["ORmat"].numpy()
    ids = [int(t) - 1 for t in graph["trind"].tolist()]
    if (A < 0).any():
        raise ValueError("negated preconditions are not supported by this extraction")
    out = {}
    for s in range(A.shape[1]):
        ands = [j for j in range(O.shape[1]) if O[s, j] != 0]
        out[ids[s]] = sorted(sorted(ids[k] for k in range(A.shape[1]) if A[j, k] != 0) for j in ands)
    return out


def canonical(graphs) -> dict[int, list[list[int]]]:
    rules: dict = {}
    for g in graphs:
        for s, r in rules_of(g).items():
            if rules.setdefault(s, r) != r:
                raise ValueError(f"subtask {SUBTASKS[s][0]!r} has different rules in different graphs")
    return rules


def document(graphs) -> dict:
    rules = canonical(graphs)
    if sorted(rules) != list(range(len(SUBTASKS))):
        raise ValueError("the graphs do not cover every Mining subtask")
    return {
        "format": "msgi_subtask_graph/1",
        "domain": "Mining (Sohn, Woo, Choi and Lee, ICLR 2020)",
        "source": {
            "repository": "https://github.com/srsohn/msgi",
            "commit": COMMIT,
            "file": "grid-world/environment/data/task_graph_mining/full_mining.pkl",
            "names": "grid-world/environment/mining.py",
            "license": "MIT, Copyright (c) 2020 Hyunjae Woo (see LICENSE in this folder)",
        },
        "objects": list(OBJECTS),
        "subtasks": [{"id": i, "name": n, "operation": OPERATIONS[op], "object": OBJECTS[oid]}
                     for i, (n, (op, oid)) in enumerate(SUBTASKS)],
        "rules": [{"subtask": s, "or": rules[s]} for s in sorted(rules)],
        "check": {"graphs_in_file": len(graphs),
                  "every_graph_is_a_subgraph_of_these_rules": True},
    }


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed mining_graph.json")
    ap.add_argument("--tasks", action="store_true", help="also export the train/eval task files to data/")
    args = ap.parse_args(argv)
    fetch = [sys.executable, str(HERE / "fetch.py")] + (["--optional"] if args.tasks else [])
    if subprocess.call(fetch) != 0:
        return 1
    text = dumps(document(load_graphs(HERE / "data" / "full_mining.pkl")))
    target = HERE / "mining_graph.json"
    if args.check:
        same = target.exists() and target.read_text(encoding="utf-8").replace("\r\n", "\n") == text
        print("mining_graph.json reproduces" if same else "mining_graph.json differs")
        if not same:
            return 1
    else:
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {target.name}")
    if args.tasks:
        reference = canonical(load_graphs(HERE / "data" / "full_mining.pkl"))
        for name in TASK_FILES:
            graphs = load_graphs(HERE / "data" / f"{name}.pkl")
            for g in graphs:
                for s, r in rules_of(g).items():
                    if reference[s] != r:
                        raise ValueError(f"{name}: subtask {SUBTASKS[s][0]!r} differs from mining_graph.json")
            doc = {"file": f"{name}.pkl", "commit": COMMIT,
                   "note": "subtask ids per task (graph) and the reward magnitude of each listed subtask; "
                           "every rule equals the rule of mining_graph.json",
                   "tasks": [{"subtasks": [int(t) - 1 for t in g["trind"].tolist()],
                              "rewards": [round(float(x), 6) for x in g["rmag"]]} for g in graphs]}
            out = HERE / "data" / f"tasks_{name}.json"
            out.write_text(dumps(doc), encoding="utf-8", newline="\n")
            print(f"wrote data/{out.name} ({len(graphs)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
