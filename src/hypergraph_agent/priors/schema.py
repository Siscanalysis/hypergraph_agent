"""Lifted rules over public families and classes (study HP2, docs/studies/HP2_schema_priors.md).

The rules describe the opt-in schema worlds of ``envs.generator``: base facts belong to K classes,
recipes to F families (public: the family of a recipe's effect item, ``family_of``); a family-f
recipe requires one fact of each class of its family's class set C_f, drawn uniformly among the
pool's facts of that class, and every pool holds a fact of every class. A rule reads "a family-f
recipe requires one fact of each class in C_f". For every family the learners enumerate the class
sets C (nonempty, at most min(max_size, K) classes, uniform prior).

Data. For every attempted recipe the evidence settles some candidates (``Observation``): required
ones and eliminated ones (study M's sound pairwise update, ``agents.hybrid.observations``). The
requirement therefore lies in H_r = {h in the class : required <= h, h and eliminated disjoint}; a
pinned recipe is the case |H_r| = 1. The learners condition on these sets (the coarsened evidence:
any information beyond the settled candidates is not used), each recipe's requirement independent
given the schema, as the generator draws them.

``SchemaV1`` (public classes, the declared table ``public_class``): exact for these data.
P(D_f | C) = prod over the family's recipes of sum over h in H_r of P(h | C, pool), where
P(h | C, pool) = prod over c in C of 1 / |pool facts of class c| when h holds exactly one fact of
each class in C and nothing else, and 0 otherwise.

``SchemaV2`` (hidden classes): the class of a fact is a variable. clingo enumerates every class
assignment of the facts the evidence requires, together with the class sets of their families, such
that the required facts of a recipe have distinct classes, all in its family's set, and a pinned
requirement has exactly one fact of every class of its family. Each model (a labelling with its class
sets; every labelling has the same prior) is weighted by the likelihood of all the data, in which
the classes of facts the model does not fix are marginalized within each recipe's pool separately
(1/K each, conditioned on the pool holding every class): the declared approximation, which ignores
that such a fact keeps one class across pools. A refit whose enumeration exceeds ``max_models`` is
refused (an overflow, counted): the learner keeps its previous fit, and so does ``ModelAverage``.

``ModelAverage``: the schema model against the no-schema model, the uniform prior over the class (the
posterior's own prior), whose likelihood of the data is prod |H_r| / |class|. With the prior
probability ``model_prior`` of the schema model, the two marginal likelihoods give its posterior
weight w; the prior of a recipe is w x predictive(h) + (1 - w) x uniform(h), in log space. Data a
schema cannot produce give it likelihood 0 and weight 0.

This package holds public-information code only: the privileged prior of the true schema
(``OracleSchema``) lives on the evaluator side, in ``evaluation.hybrid_priors``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from ..envs.generator import public_class

MAX_SIZE = 3


@dataclass(frozen=True)
class Observation:
    """What the evidence settles about one recipe (public data only)."""

    key: tuple  # (world key, signature): one signature in two worlds is two recipes
    family: int
    pool: tuple[int, ...]
    required: frozenset
    eliminated: frozenset

    @property
    def pinned(self) -> bool:
        return len(self.required) + len(self.eliminated) == len(self.pool) and bool(self.required)

    def allowed(self, max_size: int = MAX_SIZE) -> tuple[frozenset, ...]:
        """H_r: the hypotheses of the class compatible with the settled candidates."""
        return _allowed(self.pool, self.required, self.eliminated, max_size)


@lru_cache(maxsize=200_000)
def _allowed(pool, required, eliminated, max_size) -> tuple[frozenset, ...]:
    free = [b for b in pool if b not in required and b not in eliminated]
    out = []
    for k in range(0, max_size - len(required) + 1):
        for extra in combinations(free, k):
            h = required | frozenset(extra)
            if h:
                out.append(h)
    return tuple(out)


def options(n_classes: int, max_size: int = MAX_SIZE) -> list[tuple[int, ...]]:
    """Every class set a family may draw from: nonempty, at most min(max_size, K) classes."""
    return [c for k in range(1, min(max_size, n_classes) + 1) for c in combinations(range(n_classes), k)]


def rainbow(pool, classes, cset) -> dict:
    """{requirement: probability} given the class of every pool fact (``classes``,
    aligned with ``pool``): one fact of each class in ``cset``, uniformly among
    the pool's facts of that class; empty when a class has no pool fact."""
    groups = [[b for b, c in zip(pool, classes) if c == k] for k in cset]
    if any(not g for g in groups):
        return {}
    p = 1.0 / math.prod(len(g) for g in groups)
    return {frozenset(x): p for x in product(*groups)}


def class_size(pool, max_size: int = MAX_SIZE) -> int:
    n = len(pool)
    return sum(math.comb(n, k) for k in range(1, min(max_size, n) + 1))


def _logsumexp(v) -> float:
    v = np.asarray(v, dtype=float)
    m = v.max() if len(v) else -math.inf
    if not np.isfinite(m):
        return -math.inf
    return float(m + math.log(np.exp(v - m).sum()))


def _log(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


class SchemaV1:
    """Exact posterior over every family's class set with the public class table."""

    def __init__(self, n_classes: int, n_families: int, max_size: int = MAX_SIZE):
        self.K, self.F, self.max_size = n_classes, n_families, max_size
        self.opts = options(n_classes, max_size)
        self.log_post = {f: np.zeros(len(self.opts)) for f in range(n_families)}
        self.log_marginal = 0.0
        self.n_obs = 0

    def classes(self, pool) -> tuple[int, ...]:
        return tuple(public_class(b, self.K) for b in pool)

    def fit(self, observations: list[Observation]) -> None:
        lik = {f: np.zeros(len(self.opts)) for f in range(self.F)}
        for o in observations:
            cls, allowed = self.classes(o.pool), o.allowed(self.max_size)
            if len(allowed) == class_size(o.pool, self.max_size):
                continue  # nothing settled: the same factor (1) under every class set
            for i, c in enumerate(self.opts):
                r = rainbow(o.pool, cls, c)
                lik[o.family][i] += _log(sum(r.get(h, 0.0) for h in allowed))
        prior = -math.log(len(self.opts))
        self.log_post = {f: v + prior for f, v in lik.items()}
        self.log_marginal = sum(_logsumexp(v) for v in self.log_post.values())
        self.n_obs = len(observations)

    def predictive(self, family: int, pool) -> dict:
        lp = self.log_post[family]
        if not np.isfinite(lp).any():
            return {}
        w = np.exp(lp - lp.max())
        w /= w.sum()
        out: dict = defaultdict(float)
        cls = self.classes(pool)
        for wi, cset in zip(w, self.opts):
            if wi > 0:
                for h, p in rainbow(pool, cls, cset).items():
                    out[h] += wi * p
        return dict(out)

    def summary(self) -> dict:
        top = {f: list(self.opts[int(np.argmax(lp))]) for f, lp in self.log_post.items()
               if np.isfinite(lp).any()}
        return {"observations": self.n_obs, "map_class_sets": top}


class SchemaV2:
    """Hidden classes: clingo enumeration of class assignments and class sets."""

    def __init__(self, n_classes: int, n_families: int, max_size: int = MAX_SIZE, max_models: int = 50_000):
        self.K, self.F, self.max_size, self.max_models = n_classes, n_families, max_size, max_models
        self.opts = options(n_classes, max_size)
        self.models: list[tuple[dict, dict]] = [({}, {})]  # (fact -> class, family -> class set)
        self.log_w = np.zeros(1)  # log posterior weight of every model (normalized)
        self.log_marginal = 0.0
        self.overflow = 0  # fits whose enumeration exceeded max_models (the last good models are kept)
        self.n_obs = 0
        self._key = None
        self._obs: list[Observation] = []
        self._by_family: dict = {}
        self._pattern_cache: dict = {}
        self._lik_memo: dict = {}

    def program(self, constraints) -> str:
        """``constraints``: (family, required facts, pinned) of every recipe with a
        required fact."""
        lines = [f"cls_id(0..{self.K - 1})."]
        facts = sorted({b for _, q, _ in constraints for b in q})
        lines += [f"fact({b})." for b in facts]
        for i, (f, q, pin) in enumerate(constraints):
            lines.append(f"con({i},{f}).")
            lines += [f"req({i},{b})." for b in sorted(q)]
            if pin:
                lines.append(f"pinned({i},{len(q)}).")
        lines += [
            "fam(F) :- con(_, F).",
            "1 { cls(X, C) : cls_id(C) } 1 :- fact(X).",
            f"1 {{ sch(F, C) : cls_id(C) }} {min(self.max_size, self.K)} :- fam(F).",
            ":- pinned(I, N), con(I, F), #count { C : sch(F, C) } != N.",
            ":- req(I, X), con(I, F), cls(X, C), not sch(F, C).",
            ":- req(I, X), req(I, Y), X < Y, cls(X, C), cls(Y, C).",
            "#show cls/2.", "#show sch/2.",
        ]
        return "\n".join(lines)

    def enumerate(self, constraints) -> list[tuple[dict, dict]] | None:
        """Every model, or None when there are more than ``max_models``."""
        import clingo
        ctl = clingo.Control(["0", "--warn=none"])
        ctl.add("base", [], self.program(constraints))
        ctl.ground([("base", [])])
        models: list[tuple[dict, dict]] = []
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                cls, sch = {}, defaultdict(list)
                for sym in m.symbols(shown=True):
                    a, b = (int(x.number) for x in sym.arguments)
                    if sym.name == "cls":
                        cls[a] = b
                    else:
                        sch[a].append(b)
                models.append((cls, {f: tuple(sorted(v)) for f, v in sch.items()}))
                if len(models) > self.max_models:
                    return None
        return models

    def pattern(self, known: tuple, cset) -> dict:
        """{positions: probability} for a pool whose facts have the classes
        ``known`` (None: every class with probability 1/K, conditioned on the
        pool holding every class) and the class set ``cset``. Depends only on
        the pattern, so it is cached across pools and fits."""
        key = (known, cset)
        if key not in self._pattern_cache:
            unknown = [i for i, c in enumerate(known) if c is None]
            pos = tuple(range(len(known)))
            out: dict = defaultdict(float)
            norm = 0
            for fill in product(range(self.K), repeat=len(unknown)):
                cls = list(known)
                for i, c in zip(unknown, fill):
                    cls[i] = c
                if len(set(cls)) < self.K:
                    continue
                norm += 1
                for h, p in rainbow(pos, cls, cset).items():
                    out[tuple(sorted(h))] += p
            self._pattern_cache[key] = {h: v / norm for h, v in out.items()} if norm else {}
        return self._pattern_cache[key]

    def _lik(self, o: Observation, known: tuple, cset) -> float:
        """log P(requirement in H_r | class set, the pool's known classes), memoized per fit."""
        key = (o.key, known, cset)
        if key not in self._lik_memo:
            index = {b: i for i, b in enumerate(o.pool)}
            pat = self.pattern(known, cset)
            self._lik_memo[key] = _log(sum(pat.get(tuple(sorted(index[b] for b in h)), 0.0)
                                           for h in o.allowed(self.max_size)))
        return self._lik_memo[key]

    def fit(self, observations: list[Observation]) -> None:
        constraints = sorted({(o.family, frozenset(o.required), o.pinned) for o in observations
                              if o.required}, key=lambda c: (c[0], sorted(c[1]), c[2]))
        key = tuple(constraints)
        if key != self._key:
            models = self.enumerate(constraints)
            if models is None:
                self.overflow += 1
                return
            self.models, self._key = models, key
        informative = [o for o in observations
                       if len(o.allowed(self.max_size)) < class_size(o.pool, self.max_size)]
        self._obs, self.n_obs = informative, len(informative)
        self._lik_memo = {}
        self._by_family = defaultdict(list)
        for o in informative:
            self._by_family[o.family].append(o)
        n_facts = len({b for _, q, _ in constraints for b in q})
        fixed = {f for f, _, _ in constraints}
        logs = np.empty(len(self.models))
        for m, (cls, sch) in enumerate(self.models):
            total = 0.0
            for f, obs in self._by_family.items():
                if f in fixed:
                    total += sum(self._lik(o, tuple(cls.get(b) for b in o.pool), sch[f]) for o in obs)
                else:
                    terms = [sum(self._lik(o, tuple(cls.get(b) for b in o.pool), c) for o in obs)
                             for c in self.opts]
                    total += _logsumexp(terms) - math.log(len(self.opts))
            logs[m] = total
        # prior of a model: K^-n for its labelling, 1/|options| for every fixed class set
        prior = -n_facts * math.log(self.K) - len(fixed) * math.log(len(self.opts))
        self.log_marginal = _logsumexp(logs) + prior if len(logs) else -math.inf
        z = _logsumexp(logs)
        self.log_w = logs - z if np.isfinite(z) else np.full(len(logs), -math.inf)

    def predictive(self, family: int, pool) -> dict:
        if not self.models or not np.isfinite(self.log_w).any():
            return {}
        obs = self._by_family.get(family, [])
        groups: dict = defaultdict(float)  # models that predict alike are added up first
        for wm, (cls, sch) in zip(np.exp(self.log_w), self.models):
            if wm > 0:
                data = None if family in sch else tuple(tuple(cls.get(b) for b in o.pool) for o in obs)
                groups[(tuple(cls.get(b) for b in pool), sch.get(family), data)] += wm
        out: dict = defaultdict(float)
        for (known, cset, data), wm in groups.items():
            if cset is not None:
                cw = [(cset, 1.0)]
            else:  # the class set given this model and the family's data
                lw = np.array([sum(self._lik(o, kn, c) for o, kn in zip(obs, data)) for c in self.opts])
                if not np.isfinite(lw).any():
                    continue
                p = np.exp(lw - lw.max())
                cw = list(zip(self.opts, p / p.sum()))
            for c, pc in cw:
                for pos, p in self.pattern(known, c).items():
                    out[frozenset(pool[i] for i in pos)] += wm * pc * p
        return dict(out)

    def summary(self) -> dict:
        return {"observations": self.n_obs, "models": len(self.models), "overflow": self.overflow}


class ModelAverage:
    """A schema learner averaged against the no-schema model (the uniform prior
    over the class)."""

    def __init__(self, learner, model_prior: float = 0.5, max_size: int = MAX_SIZE):
        if not 0.0 < model_prior < 1.0:
            raise ValueError("model_prior must lie in (0, 1)")
        self.learner, self.max_size = learner, max_size
        self.log_odds0 = math.log(model_prior) - math.log(1.0 - model_prior)
        self.log_none = 0.0
        self.version = 0
        self.observations: list[Observation] = []

    def fit(self, observations: list[Observation]) -> None:
        """Refit both models on ``observations``. When the learner refuses the
        refit (an overflow), the average keeps its previous fit as well, so the
        two marginal likelihoods always read the same data."""
        before = getattr(self.learner, "overflow", 0)
        self.learner.fit(list(observations))
        if getattr(self.learner, "overflow", 0) != before:
            return
        self.observations = list(observations)
        self.log_none = sum(math.log(len(o.allowed(self.max_size)))
                            - math.log(class_size(o.pool, self.max_size)) for o in self.observations)
        self.version += 1

    def log_odds(self) -> float:
        return self.log_odds0 + self.learner.log_marginal - self.log_none

    def weight(self) -> float:
        z = self.log_odds()
        if z == -math.inf:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

    def log_prior(self, family: int, pool, hyps) -> np.ndarray:
        """Log prior over ``hyps`` (frozensets of the pool's facts, the class order)."""
        z = self.log_odds()
        log_none = -math.log(len(hyps))
        if z == -math.inf:
            return np.full(len(hyps), log_none)
        log_w = -math.log1p(math.exp(-z)) if z > -700 else z
        log_1w = -math.log1p(math.exp(z)) if z < 700 else -z
        pred = self.learner.predictive(family, pool)
        with np.errstate(divide="ignore"):
            lp = np.log(np.array([pred.get(h, 0.0) for h in hyps]))
        return np.logaddexp(log_w + lp, log_1w + log_none)
