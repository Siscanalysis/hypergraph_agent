"""Requirement-size priors (study HP1, docs/studies/HP1_size_prior.md).

A size prior gives each requirement size k (1..max_size) a probability pi_k, spread uniformly over the
hypotheses of size k in a recipe's class (the nonempty subsets of its pool with at most max_size
members). The uniform class prior is the size prior pi_k = N_k / N, with N_k the number of hypotheses
of size k; for a pool of six and max_size three it is 6/41, 15/41 and 20/41 (about 0.15, 0.37, 0.49).

Empirical Bayes from pinned recipes (``learned_sizes``). A recipe is pinned when the evidence leaves
exactly one hypothesis (its requirement, and so its size, is known). With n_k pinned recipes of size k
and a Dirichlet(A u) hyperprior on the size probabilities, u the sizes implied by the uniform class
prior of the reference pool and A the hyperprior strength, the learned prior is the posterior mean
pi_k = (A u_k + n_k) / (A + n). Without pinned recipes it is the uniform class prior exactly. Which
recipes get pinned depends on their size, so these counts are a selected sample.

Empirical Bayes from all the evidence (``marginal_sizes``). The same hyperprior, updated with the
exact likelihood of every log: for each group of recipes the evidence couples, the number of
consistent assignments with c1, c2 and c3 recipes of sizes 1, 2 and 3 (a size polynomial,
``priors.posterior``) gives the likelihood sum_c n_c prod_k (pi_k / N_k)^c_k. The product over
groups is a polynomial in pi, so the posterior of pi is a mixture of Dirichlet distributions
Dir(A u + C) with weights a_C B(A u + C) / prod_k N_k^C_k, and its mean is exact. Pinned recipes are
the special case of one term; without information the mean is u exactly.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..topology.inference import enumerate_hypotheses, size_log_prior

REFERENCE_POOL = 6  # the pool size of every generated family (F1)
MAX_SIZE = 3


def uniform_sizes(pool_size: int = REFERENCE_POOL, max_size: int = MAX_SIZE) -> tuple[float, ...]:
    """Size probabilities implied by the uniform prior over the class."""
    n = [math.comb(pool_size, k) for k in range(1, min(max_size, pool_size) + 1)]
    return tuple(x / sum(n) for x in n) + (0.0,) * (max_size - len(n))


def learned_sizes(pinned_sizes, strength: float, pool_size: int = REFERENCE_POOL,
                  max_size: int = MAX_SIZE) -> tuple[float, ...]:
    """Posterior mean of the size probabilities under a Dirichlet(strength x
    uniform_sizes) hyperprior, from the sizes of pinned recipes."""
    if strength <= 0:
        raise ValueError("the hyperprior strength must be positive")
    u = uniform_sizes(pool_size, max_size)
    c = Counter(int(k) for k in pinned_sizes)
    if any(k < 1 or k > max_size for k in c):
        raise ValueError(f"pinned sizes must lie in 1..{max_size}")
    n = sum(c.values())
    return tuple((strength * u[k - 1] + c.get(k, 0)) / (strength + n) for k in range(1, max_size + 1))


def _log_convolve(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Product of two size polynomials (layout [c1, c2]) with log coefficients
    (-inf for a zero coefficient)."""
    out = np.full((la.shape[0] + lb.shape[0] - 1, la.shape[1] + lb.shape[1] - 1), -np.inf)
    for i, j in zip(*np.nonzero(np.isfinite(lb))):
        view = out[i:i + la.shape[0], j:j + la.shape[1]]
        np.logaddexp(view, lb[i, j] + la, out=view)
    return out


def _log_beta(alpha: np.ndarray) -> np.ndarray:
    """log B(alpha) over the last axis."""
    lg = np.vectorize(math.lgamma)
    return lg(alpha).sum(axis=-1) - lg(alpha.sum(axis=-1))


def marginal_sizes(polys, strength: float, pool_size: int = REFERENCE_POOL) -> tuple[float, ...]:
    """Posterior mean of the size probabilities (sizes 1..3) under a
    Dirichlet(strength x uniform_sizes) hyperprior and the exact likelihood
    given by the size polynomials ``polys`` (one per group; poly[c1, c2] counts
    the consistent assignments with c1 recipes of size 1, c2 of size 2 and the
    rest, ``poly.shape[0] - 1 - c1 - c2``, of size 3)."""
    if strength <= 0:
        raise ValueError("the hyperprior strength must be positive")
    u = np.array(uniform_sizes(pool_size, MAX_SIZE))
    alpha = strength * u
    total, n = np.zeros((1, 1)), 0
    for p in polys:
        p = np.asarray(p, dtype=float)
        if p.shape[0] != p.shape[1] or not (p >= 0).all() or p.sum() <= 0:
            raise ValueError("a size polynomial is square, nonnegative and nonzero")
        with np.errstate(divide="ignore"):
            lp = np.log(p)
        total = _log_convolve(total, lp - lp.max())
        n += p.shape[0] - 1
    c1, c2 = np.nonzero(np.isfinite(total))
    C = np.stack([c1, c2, n - c1 - c2], axis=1).astype(float)
    keep = C[:, 2] >= 0
    C, coef = C[keep], total[c1[keep], c2[keep]]
    N = np.array([math.comb(pool_size, k) for k in range(1, MAX_SIZE + 1)], dtype=float)
    logw = coef - (C * np.log(N)).sum(axis=1) + _log_beta(alpha + C) - _log_beta(alpha)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    mean = (w[:, None] * (alpha + C)).sum(axis=0) / (strength + n)
    return tuple(float(x) for x in mean)


class SizePrior:
    """Per-recipe prior over the class from size probabilities (sizes 1..max_size).
    ``weights(pool)`` is the normalized prior over ``enumerate_hypotheses``
    of the pool, the order every posterior in this package uses."""

    def __init__(self, sizes, label: str = "size", max_size: int = MAX_SIZE):
        self.sizes = tuple(float(x) for x in sizes)
        if len(self.sizes) != max_size or min(self.sizes) < 0 or sum(self.sizes) <= 0:
            raise ValueError("need one nonnegative probability per size")
        self.label, self.max_size = label, max_size
        self._cache: dict = {}

    def log_weights(self, pool) -> np.ndarray:
        n = len(pool)
        if n not in self._cache:
            hyps = enumerate_hypotheses(n, min(self.max_size, n))
            self._cache[n] = size_log_prior(self.sizes, hyps)
        return self._cache[n]

    def weights(self, pool) -> np.ndarray:
        lw = self.log_weights(pool)
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def __call__(self, pool, hypotheses) -> np.ndarray:
        """The ``topology.inference.DependencyModel`` hook: log prior per hypothesis."""
        return self.log_weights(pool)

    def describe(self) -> dict:
        return {"label": self.label, "sizes": [round(x, 6) for x in self.sizes]}
