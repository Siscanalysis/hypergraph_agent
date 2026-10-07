"""Episode aggregation. Failures are always counted; success-only path
lengths are reported next to failure statistics, never alone."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def aggregate(rows: list[dict], by: str | None = None) -> dict:
    if by is not None:
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(by)].append(r)
        return {str(k): aggregate(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    n = len(rows)
    if n == 0:
        return {"n": 0}
    succ = [r for r in rows if r["success"]]
    out = {
        "n": n,
        "success_rate": len(succ) / n,
        "mean_primitive_length_all": float(np.mean([r["primitive_length"] for r in rows])),
        "mean_manager_decisions_all": float(np.mean([r["manager_decisions"] for r in rows])),
        "mean_noop_or_invalid": float(np.mean([r["noop_or_invalid"] for r in rows])),
        "mean_skill_calls": float(np.mean([r.get("skill_calls", 0) for r in rows])),
        "n_failures": n - len(succ),
        "failure_status": dict(sorted(_count(r["status"] for r in rows if not r["success"]).items())),
    }
    if succ:
        out["mean_primitive_length_success"] = float(np.mean([r["primitive_length"] for r in succ]))
        ratios = [r["primitive_length"] / r["reference_length"] for r in succ
                  if r.get("reference_length") and r.get("reference_exact")]
        if ratios:
            out["mean_length_over_optimal_success"] = float(np.mean(ratios))
    return out


def _count(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c
