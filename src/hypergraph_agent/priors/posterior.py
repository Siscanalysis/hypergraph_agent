"""The exact posterior over hidden requirements under any per-recipe prior (study HP1).

Priors are objects with ``weights(pool)``: a nonnegative weight for every hypothesis of the recipe's
class, in the order of ``topology.inference.enumerate_hypotheses`` (``UniformPrior``, or
``priors.size.SizePrior``). The joint prior of a complete hypergraph is the product over recipes.

``joint_marginals`` (goal-only logs). Study M's posterior (``evaluation.markov_study``:
``posterior_marginals``) with weights: hypotheses holding a candidate eliminated by a success are
dropped, recipes are grouped by the logs they share, and each group is enumerated depth first,
pruned by the logs (deterministic replay, ``markov_study.consistent``) and memoized on the frontier,
with every completion counted with its prior weight. The enumeration and its pruning do not depend on
the weights, so one pass computes the marginals under several priors at once, and whether a group is
exact (within ``exact_work`` log checks) is the same for every prior. Beyond the work budget each
prior gets weighted Gibbs sampling among consistent nodes (four focused-walk starts, 40 sweeps after
10 of burn-in, Rao-Blackwellized), every prior from the same seed. With uniform weights the exact
groups equal study M's posterior (tested).

``factorized_marginals`` (items observable). Every informative attempt is a labelled example of one
recipe, so the posterior factorizes by recipe: the per-recipe updater ``DependencyModel`` with the
prior as its opt-in ``log_prior`` (study X2's form of the posterior for the uniform prior).

``pinned`` reads the recipes whose candidate marginals are all 0 or 1 (one hypothesis left). With
deterministic dynamics the set of consistent hypotheses does not depend on the prior, so neither does
being pinned. Only pins of exactly computed groups (``exact_sigs``) are certain: a sampled marginal of
0 or 1 means only that no chain visited another hypothesis.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..agents.markov import observations
from ..agents.walker import WorldState, local_walk
from ..topology.inference import DependencyModel, Evidence, enumerate_hypotheses

PIN_TOL = 1e-9


class UniformPrior:
    """The uniform prior over the class (today's posterior)."""

    label = "uniform"

    def __init__(self, max_size: int = 3):
        self.max_size = max_size

    def weights(self, pool) -> np.ndarray:
        n = len(enumerate_hypotheses(len(pool), min(self.max_size, len(pool))))
        return np.full(n, 1.0 / n)

    def describe(self) -> dict:
        return {"label": self.label}


def class_hypotheses(info, max_size: int = 3) -> list[frozenset]:
    n = len(info.pool)
    return [frozenset(info.pool[p] for p in h) for h in enumerate_hypotheses(n, min(max_size, n))]


def _union_groups(sigs, rel) -> list[list[str]]:
    parent = {s: s for s in sigs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, r in rel:
        for s in r[1:]:
            parent[find(s)] = find(r[0])
    comps = defaultdict(list)
    for s in sigs:
        comps[find(s)].append(s)
    return list(comps.values())


class _Budget(Exception):
    pass


def _exact(comp, hyps, W, idxs, rel, ok, work: int, polys: list | None = None) -> list[dict] | None:
    """Weighted exact candidate marginals of one group under every prior (rows
    of ``W[s]``), or None when the evidence is inconsistent or the enumeration
    needs more than ``work`` log checks. The order, frontier and memoization
    are those of study M's ``_exact``. With ``polys`` (a list), the group's size
    polynomial is appended to it: the number of consistent assignments with c1
    recipes of size 1 and c2 of size 2 (the others of size 3) at [c1, c2]."""
    done, at, order, done_at = set(), set(), [], []
    by_sig = {s: [i for i in idxs if s in rel[i][1]] for s in comp}
    while len(order) < len(comp):
        def gain(s):
            return sum(1 for i in idxs if i not in done and set(rel[i][1]) <= at | {s})
        s = max(sorted(set(comp) - at), key=lambda s: (gain(s), len(by_sig[s]), -len(hyps[s])))
        order.append(s)
        at.add(s)
        new = [i for i in idxs if i not in done and set(rel[i][1]) <= at]
        done.update(new)
        done_at.append(new)
    frontier, closed = [], set()
    for d in range(len(order)):
        frontier.append(sorted({s for i in idxs if i not in closed for s in rel[i][1] if s in order[:d]}))
        closed.update(done_at[d])
    cands = {s: sorted({b for h in hyps[s] for b in h}) for s in order}
    col = {s: {b: j for j, b in enumerate(cands[s])} for s in order}
    width = max(len(c) for c in cands.values())
    n_priors = W[order[0]].shape[0]
    track = polys is not None
    memo, node, spent = {}, {}, [0]

    def rec(d):
        n_sub = len(order) - d
        if d == len(order):
            leaf = np.ones((1, 1)) if track else None
            return np.ones(n_priors), np.zeros((n_priors, len(order), width)), leaf
        key = (d, tuple(node[s] for s in frontier[d]))
        if key in memo:
            return memo[key]
        s = order[d]
        n_all, m_all = np.zeros(n_priors), np.zeros((n_priors, len(order), width))
        p_all = np.zeros((n_sub + 1, n_sub + 1)) if track else None
        for j, h in enumerate(hyps[s]):
            node[s] = h
            spent[0] += max(1, len(done_at[d]))
            if spent[0] > work:
                raise _Budget
            if all(ok(node, i) for i in done_at[d]):
                n, m, pc = rec(d + 1)
                if n.any():
                    wj = W[s][:, j]
                    n_all += wj * n
                    m_all += wj[:, None, None] * m
                    for b in h:
                        m_all[:, d, col[s][b]] += wj * n
                    if track:  # a size-1 hypothesis raises c1, size 2 raises c2, size 3 neither
                        if len(h) == 1:
                            p_all[1:, :-1] += pc
                        elif len(h) == 2:
                            p_all[:-1, 1:] += pc
                        else:
                            p_all[:-1, :-1] += pc
        del node[s]
        memo[key] = (n_all, m_all, p_all)
        return n_all, m_all, p_all

    try:
        total, counts, poly = rec(0)
    except _Budget:
        return None
    if not total.all():
        return None
    if track:
        polys.append(poly)
    return [{(s, b): counts[p, d, col[s][b]] / total[p] for d, s in enumerate(order) for b in cands[s]}
            for p in range(n_priors)]


def _gibbs(comp, hyps, Wp, by_sig, ok, rng, infos, logs_c, sigs, max_size, starts, sweeps, burn, walk_evals):
    """Weighted Gibbs sampling among consistent nodes for one prior (weights ``Wp``)."""
    acc, chains, missing = defaultdict(float), 0, 0
    order = sorted(comp)
    for _ in range(starts):
        ws = WorldState("posterior", 0.0, max_size)
        ws.infos.update(infos)
        ws.hypotheses = {s: list(hyps[s]) for s in sigs}
        ws.evidence.logs = list(logs_c)
        start = {s: hyps[s][int(rng.integers(len(hyps[s])))] for s in sigs}
        node, st = local_walk(start, ws, None, "focused", rng, max_size, walk_evals, 0.5, 80)
        if st["violations"]:
            missing += 1
            continue
        chains += 1
        node = dict(node)
        chain, n = defaultdict(float), 0
        for sweep in range(burn + sweeps):
            for s in order:
                cur = node[s]
                good, gw = [], []
                for j, h in enumerate(hyps[s]):
                    node[s] = h
                    if all(ok(node, i) for i in by_sig[s]):
                        good.append(h)
                        gw.append(Wp[s][j])
                if not good:
                    good, gw = [cur], [1.0]
                gw = np.asarray(gw, dtype=float)
                tot = gw.sum()
                if sweep >= burn:
                    for h, x in zip(good, gw):
                        for b in h:
                            chain[(s, b)] += x / tot
                node[s] = good[int(rng.choice(len(good), p=gw / tot))]
            n += sweep >= burn
        for k, v in chain.items():
            acc[k] += v / n
    if not chains:
        return None, 0, missing
    return {(s, b): acc.get((s, b), 0.0) / chains for s in comp for b in infos[s].pool}, chains, missing


def joint_marginals(infos: dict, logs: list, priors: list, seed: int, *, max_size: int = 3,
                    exact_work: int = 2_000_000, starts: int = 4, sweeps: int = 40, burn: int = 10,
                    walk_evals: int = 4000, cache: dict | None = None,
                    polys: list | None = None) -> tuple[list[dict], dict]:
    """Candidate marginals P(b in B_r | logs) of every hidden recipe attempted
    in ``logs`` under each prior of ``priors`` (see the module docstring).
    ``cache`` keeps the replay checks between calls on the same logs. With
    ``polys`` (a list, requirements of at most three), the size polynomial of
    every exactly computed group is appended to it (``_exact``); sampled groups
    have none and are counted in ``poly_groups_left_out``. The diagnostics list
    the recipes of exactly computed groups in ``exact_sigs``: only their pins
    are certain."""
    if polys is not None and max_size != 3:
        raise ValueError("size polynomials assume requirements of at most three")
    from ..evaluation.markov_study import consistent, relevant
    sigs = sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})
    elim = {e for lg in logs for kind, _, x in observations(lg, infos) if kind == "success" for e in x}
    hyps, W = {}, {}
    for s in sigs:
        full = class_hypotheses(infos[s], max_size)
        keep = [i for i, h in enumerate(full) if not any((s, b) in elim for b in h)] or list(range(len(full)))
        hyps[s] = [full[i] for i in keep]
        rows = np.array([np.asarray(p.weights(infos[s].pool), dtype=float)[keep] for p in priors])
        W[s] = rows / np.maximum(rows.max(axis=1, keepdims=True), 1e-300)
    rel = [(lg, r) for lg in logs for r in [relevant(lg, infos)] if r]
    cache = {} if cache is None else cache

    def ok(node, i) -> bool:
        lg, r = rel[i]
        key = (i, tuple(node[s] for s in r))
        v = cache.get(key)
        if v is None:
            v = cache[key] = consistent(node, lg, infos)
        return v

    out = [dict() for _ in priors]
    diag = {"components": 0, "exact_recipes": 0, "sampled_recipes": 0, "prior_recipes": 0, "chains": 0,
            "chains_without_start": 0, "poly_groups_left_out": 0, "exact_sigs": []}
    for comp in _union_groups(sigs, rel):
        diag["components"] += 1
        idxs = [i for i, (_, r) in enumerate(rel) if r[0] in comp]
        res = _exact(comp, hyps, W, idxs, rel, ok, exact_work, polys)
        if res is not None:
            diag["exact_recipes"] += len(comp)
            diag["exact_sigs"] += sorted(comp)
        else:
            diag["poly_groups_left_out"] += polys is not None
            by_sig = {s: [i for i in idxs if s in rel[i][1]] for s in comp}
            res, sampled = [], True
            for p in range(len(priors)):
                rng = np.random.default_rng(seed)
                m, ch, miss = _gibbs(comp, hyps, {s: W[s][p] for s in sigs}, by_sig, ok, rng, infos,
                                     [rel[i][0] for i in idxs], sigs, max_size, starts, sweeps, burn,
                                     walk_evals)
                diag["chains"] += ch
                diag["chains_without_start"] += miss
                sampled &= m is not None
                res.append(m)
            diag["sampled_recipes"] += len(comp) if sampled else 0
        for p in range(len(priors)):
            m = res[p]
            if m is None:  # inconsistent evidence or no consistent start: the (filtered) prior
                diag["prior_recipes"] += len(comp)
                m = {}
                for s in comp:
                    w = W[s][p] / W[s][p].sum()
                    m.update({(s, b): float(sum(x for x, h in zip(w, hyps[s]) if b in h))
                              for b in infos[s].pool})
            out[p].update(m)
    for p in range(len(priors)):
        for s in sigs:
            for b in infos[s].pool:
                out[p].setdefault((s, b), 0.0)
    return out, diag


def factorized_marginals(infos: dict, logs: list, sigs, priors: list, *, max_size: int = 3,
                         polys: list | None = None) -> tuple[list, dict]:
    """Candidate marginals under each prior when items are observable: the
    per-recipe updater with the prior as ``log_prior`` (None for the uniform
    prior, so that arm is the updater exactly as study X2 runs it). With
    ``polys``, every recipe's size polynomial (its consistent hypotheses by
    size, in the layout of ``_exact``) is appended, from the first prior's
    updater, which must be the uniform one."""
    from ..inference_baselines.traces import craft_examples
    examples = craft_examples(logs, infos)
    out, diag = [], {}
    if polys is not None and (not priors or not isinstance(priors[0], UniformPrior) or max_size != 3):
        raise ValueError("size polynomials come from the uniform updater, requirements of at most three")
    for p in priors:
        dm = DependencyModel("hp", 0.0, max_size, log_prior=None if isinstance(p, UniformPrior) else
                             (lambda pool, hyps, p=p: np.log(np.maximum(p.weights(pool), 1e-300))))
        for s in sigs:
            dm.register_rule(s, infos[s].items, infos[s].pool)
        counts = {"informative": 0, "uninformative": 0, "contradiction": 0}
        for e in examples:
            if e.signature in dm.rules:
                counts[dm.apply(Evidence(f"{e.log_id}:{e.t}", "hp", e.signature, e.present, e.success))] += 1
        scores = {}
        for s in sigs:
            m = dm.rules[s].marginals()
            scores.update({(s, b): m[b] for b in infos[s].pool})
        if polys is not None and not out:
            for s in sigs:
                b = dm.rules[s]
                poly = np.zeros((2, 2))
                for h, x in zip(b.hypotheses, b.posterior()):
                    if x > 0:
                        poly[(1, 0) if len(h) == 1 else (0, 1) if len(h) == 2 else (0, 0)] += 1
                polys.append(poly)
        out.append(scores)
        diag = counts
    return out, {"form": "factorized", "exact_recipes": len(sigs), "exact_sigs": sorted(sigs), **diag}


def pinned(marginals: dict, infos: dict, sigs, tol: float = PIN_TOL) -> dict:
    """{signature: requirement} of the recipes whose candidate marginals are all
    within ``tol`` of 0 or 1 (exactly one hypothesis left)."""
    out = {}
    for s in sigs:
        vals = {b: marginals[(s, b)] for b in infos[s].pool}
        if all(v <= tol or v >= 1.0 - tol for v in vals.values()):
            req = frozenset(b for b, v in vals.items() if v >= 1.0 - tol)
            if req:
                out[s] = req
    return out
