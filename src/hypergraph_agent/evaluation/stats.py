"""Seed-block uncertainty for method contrasts.

The unit of replication is the independently trained seed block (for P3, the
shared-pretraining seed). Episodes within one trained model are nested inside
their block and are never treated as independent training runs. Contrasts are
paired by seed block; blocks missing a method are excluded and reported.
"""

from __future__ import annotations

import numpy as np


def _blocks(scores: dict, methods) -> tuple[list, list]:
    common = sorted(set.intersection(*(set(scores.get(m, {})) for m in methods))) if methods else []
    every = sorted(set().union(*(set(scores.get(m, {})) for m in methods)))
    missing = [b for b in every if b not in common]
    return common, missing


def _block_value(v, rng=None):
    """A block value is a scalar, or a list of task outcomes (resampled when rng is given)."""
    if np.isscalar(v):
        return float(v)
    arr = np.asarray(v, dtype=float)
    if rng is not None:
        arr = arr[rng.integers(0, len(arr), len(arr))]
    return float(arr.mean())


def contrast(scores: dict, weights: dict, n_boot: int = 2000, seed: int = 0,
             nested: bool = True, level: float = 0.95) -> dict:
    """Bootstrap a linear contrast sum_m w_m * J_m over paired seed blocks.

    ``scores[method][block]`` is a scalar or a list of per-task outcomes.
    With ``nested`` the bootstrap resamples blocks, then tasks within blocks.
    """
    methods = [m for m, w in weights.items() if w != 0]
    blocks, missing = _blocks(scores, methods)
    if not blocks:
        return {"estimate": None, "ci": None, "n_blocks": 0, "missing_blocks": missing}
    per_block = np.array([sum(w * _block_value(scores[m][b]) for m, w in weights.items() if w)
                          for b in blocks])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        vals = []
        for j in pick:
            b = blocks[j]
            vals.append(sum(w * _block_value(scores[m][b], rng if nested else None)
                            for m, w in weights.items() if w))
        boots[i] = np.mean(vals)
    a = (1 - level) / 2
    return {
        "estimate": float(per_block.mean()),
        "ci": [float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))],
        "per_block": {str(b): float(v) for b, v in zip(blocks, per_block)},
        "n_blocks": len(blocks), "missing_blocks": missing, "level": level,
        "note": "percentile seed-block bootstrap; few blocks give unreliable intervals",
    }


def difference(scores: dict, a: str, b: str, **kw) -> dict:
    return contrast(scores, {a: 1.0, b: -1.0}, **kw)


def interaction(scores: dict, g1="G1", g0="G0", f1="F1", f0="F0", **kw) -> dict:
    """(J_G1 - J_G0) - (J_F1 - J_F0)."""
    return contrast(scores, {g1: 1.0, g0: -1.0, f1: -1.0, f0: 1.0}, **kw)


def cluster_bootstrap_mean(values, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap of the mean of independent cluster values (one
    value per cluster, e.g. a paired per-world difference): clusters are
    resampled with replacement, nothing inside a cluster is resampled."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return {"estimate": None, "ci": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return {"estimate": float(v.mean()), "ci": [float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))],
            "n": len(v), "n_boot": n_boot, "seed": seed, "level": level}


def sign_test(values) -> dict:
    """Exact two-sided sign test of a zero median; zeros are dropped."""
    from math import comb
    v = [float(x) for x in values]
    neg, pos = sum(x < 0 for x in v), sum(x > 0 for x in v)
    n = neg + pos
    k = min(neg, pos)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"negative": neg, "positive": pos, "zero": len(v) - n, "p_two_sided": p}
