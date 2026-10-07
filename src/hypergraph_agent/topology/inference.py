"""Interaction-based prerequisite inference over a finite hypothesis class.

Hypothesis class (declared prior and limitation)
-----------------------------------------------
For every recipe the public schema lists its item inputs (certain) and a pool
of candidate base facts. The hidden base requirement B is assumed to be a
nonempty subset of the pool with at most ``max_size`` elements (41 hypotheses
for a pool of six and ``max_size`` three). The prior is uniform over that class.

Likelihood (monotone, single-effect, known noise ``epsilon``)
-------------------------------------------------------------
Evidence is ``(public facts before, attempted recipe, effect appeared?)`` and is
recorded only when the effect was absent before the attempt::

    p(y=1 | B, x) = (1 - epsilon) * [item_inputs <= x and B <= x]

With ``epsilon == 0`` a success eliminates hypotheses requiring an absent fact
and a failure eliminates hypotheses fully present. With ``epsilon > 0`` a
failure only down-weights them. Evidence that every hypothesis assigns zero
probability is logged as a contradiction (out-of-class) and leaves the
posterior unchanged. Unknown noise, disjunctions inside one recipe and
partially observed prerequisites are not supported by this model.

This is a transparent hypothesis updater, not a neural topology learner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..envs.generator import stable_hash


def enumerate_hypotheses(pool_size: int, max_size: int) -> tuple[tuple[int, ...], ...]:
    """Nonempty subsets of pool positions, canonical order (size, lexicographic)."""
    hyps = []
    for k in range(1, max_size + 1):
        hyps.extend(combinations(range(pool_size), k))
    return tuple(hyps)


def prior_marginal(pool_size: int, max_size: int) -> float:
    hyps = enumerate_hypotheses(pool_size, max_size)
    return sum(1 for h in hyps if 0 in h) / len(hyps)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    scope_key: str
    signature: str
    present_types: frozenset  # public fact types present before the attempt
    y: bool  # the previously absent effect appeared


@dataclass
class RuleBelief:
    signature: str
    item_inputs: tuple[int, ...]  # type ids
    pool: tuple[int, ...]  # type ids, sorted
    hypotheses: tuple[tuple[int, ...], ...]
    log_post: np.ndarray
    evidence_ids: list = field(default_factory=list)
    n_informative: int = 0
    n_uninformative: int = 0
    n_contradictions: int = 0

    def posterior(self) -> np.ndarray:
        return np.exp(self.log_post)

    def marginals(self) -> dict[int, float]:
        post = self.posterior()
        out = {}
        for pos, t in enumerate(self.pool):
            out[t] = float(sum(p for p, h in zip(post, self.hypotheses) if pos in h))
        return out

    def entropy(self) -> float:
        post = self.posterior()
        nz = post[post > 0]
        return float(-(nz * np.log(nz)).sum())

    def support(self, tol: float = 1e-9) -> int:
        return int((self.posterior() > tol).sum())

    def credible_union(self, mass: float) -> frozenset:
        """Union of the smallest top-posterior set holding ``mass``
        (deterministic tie break by canonical hypothesis index)."""
        post = self.posterior()
        order = sorted(range(len(post)), key=lambda i: (-post[i], i))
        chosen, acc = [], 0.0
        for i in order:
            if post[i] <= 0:
                break
            chosen.append(i)
            acc += post[i]
            if acc >= mass - 1e-12:
                break
        return frozenset(self.pool[p] for i in chosen for p in self.hypotheses[i])

    def copy(self) -> "RuleBelief":
        return RuleBelief(self.signature, self.item_inputs, self.pool, self.hypotheses,
                          self.log_post.copy(), list(self.evidence_ids),
                          self.n_informative, self.n_uninformative, self.n_contradictions)


def _logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    if not np.isfinite(m):
        return -math.inf
    return float(m + np.log(np.exp(v - m).sum()))


class DependencyModel:
    """Beliefs for one scope (a world, a task, or a training prior)."""

    def __init__(self, scope_key: str, epsilon: float, max_size: int = 3):
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        self.scope_key = scope_key
        self.epsilon = epsilon
        self.max_size = max_size
        self.rules: dict[str, RuleBelief] = {}

    def config(self) -> dict:
        return {"epsilon": self.epsilon, "max_size": self.max_size, "prior": "uniform"}

    def config_hash(self) -> str:
        return stable_hash(self.config())

    def register_rule(self, signature: str, item_inputs, pool) -> RuleBelief:
        if signature in self.rules:
            return self.rules[signature]
        pool = tuple(sorted(pool))
        hyps = enumerate_hypotheses(len(pool), min(self.max_size, len(pool)))
        belief = RuleBelief(signature, tuple(sorted(item_inputs)), pool, hyps,
                            np.full(len(hyps), -math.log(len(hyps))))
        self.rules[signature] = belief
        return belief

    def register_task(self, spec) -> None:
        """Register every recipe with a hidden base requirement (public data only)."""
        for r in spec.rules:
            if r.known_base is None:
                self.register_rule(
                    r.signature,
                    [spec.facts[i].type_id for i in r.item_inputs],
                    [spec.facts[i].type_id for i in r.pool],
                )

    def likelihood(self, belief: RuleBelief, ev: Evidence) -> np.ndarray | None:
        """Per-hypothesis likelihood, or None when the evidence is uninformative."""
        x = ev.present_types
        if not set(belief.item_inputs) <= x:
            if ev.y:
                return np.zeros(len(belief.hypotheses))  # impossible under the class
            return None
        sat = np.array([all(belief.pool[p] in x for p in h) for h in belief.hypotheses])
        p1 = (1.0 - self.epsilon) * sat
        return p1 if ev.y else 1.0 - p1

    def apply(self, ev: Evidence, beliefs: dict[str, RuleBelief] | None = None) -> str:
        """Update one belief in place. Returns informative/uninformative/contradiction."""
        beliefs = self.rules if beliefs is None else beliefs
        if ev.signature not in beliefs:
            raise KeyError(f"evidence for unregistered recipe {ev.signature!r}")
        b = beliefs[ev.signature]
        lik = self.likelihood(b, ev)
        if lik is None:
            b.n_uninformative += 1
            return "uninformative"
        with np.errstate(divide="ignore"):
            new = b.log_post + np.log(lik)
        z = _logsumexp(new)
        if not np.isfinite(z):
            b.n_contradictions += 1
            b.evidence_ids.append(ev.evidence_id)
            return "contradiction"
        b.log_post = new - z
        b.n_informative += 1
        b.evidence_ids.append(ev.evidence_id)
        return "informative"


def evidence_from_transition(spec, rule_index: int, present_before, present_after,
                             evidence_id: str, scope_key: str) -> Evidence | None:
    """Build public evidence for a craft attempt; None if the effect was already held."""
    rule = spec.rules[rule_index]
    if rule.known_base is not None:
        return None  # nothing hidden to infer
    if present_before[rule.effect]:
        return None  # repeated acquisition cannot identify eligibility
    types = frozenset(spec.facts[i].type_id for i, p in enumerate(present_before) if p)
    return Evidence(evidence_id, scope_key, rule.signature, types, bool(present_after[rule.effect]))
