"""Study HP2 agents (docs/studies/HP2_schema_priors.md): the sampling walker with a learned prior.

Public information only: the executor's public spec, the agent's own episode logs, and the declared
public labels of schema worlds (``envs.generator.family_of``; in version 1 also ``public_class``).

Arms
----
``hybrid``         ``focused_sample`` whose sampler uses the learned prior (``priors.schema``):
                   recipes without evidence are drawn from it, and the consistent moves target the
                   prior times consistency with every log (``Walker(prior=...)``)
``rules_only``     plans on each recipe's most probable requirement under the learned prior, ties
                   drawn at random at every replan; no walk and no consistency check
``oracle_schema``  ``hybrid`` with the true schema as the prior (``OracleSchema``), built on the
                   evaluator side (``evaluation.hybrid_priors``), which reads the
                   world's hidden schema before each episode; the reference point

Data of the rules. At every replan, study M's pairwise update (``agents.markov.PairwiseEvidence``)
reads the logs of the learning scope (the current episode's included) and settles candidates:
eliminated by a success, forced by a failure formula, and the class bounds. With deterministic
dynamics and persistent facts every settled edge is implied by the evidence (sound, not complete).
Every attempted recipe with a settled candidate gives an ``Observation`` (its required and its
eliminated candidates); a recipe all of whose candidates are settled is pinned. The learners
(``priors.schema``) are refitted on these whenever they change.

Scope (``walker.hybrid.scope``). ``world``: the rules are learned from the current world's recipes
and start afresh in every world. ``stream``: from every world the agent has played in its stream,
the current one included.

Every arm assumes deterministic dynamics. Each episode row adds ``choose_seconds`` (time spent
choosing nodes, the prior update included), ``hp_observed`` (recipes in the scope with a settled
candidate), ``hp_pinned`` (pinned recipes in the scope), ``hp_schema_weight`` (the schema model's
weight at the last replan), and, with hidden classes, ``hp_models`` and ``hp_overflow``.
"""

from __future__ import annotations

import time

import numpy as np

from ..envs.generator import family_of
from ..priors.schema import MAX_SIZE, ModelAverage, Observation, SchemaV1, SchemaV2
from .walker import Walker

HP_STRATEGIES: tuple[str, ...] = ("hybrid", "rules_only", "oracle_schema")
PARAMS = {
    "scope": None,  # "world" or "stream": where the rules are learned (required)
    "classes": None,  # "public" (version 1) or "hidden" (version 2) (required)
    "n_classes": None,  # K, public (required)
    "n_families": None,  # F, public (required)
    "model_prior": 0.5,  # prior probability of the schema model against the no-schema model
    "oracle_floor": 0.01,  # oracle_schema: weight of the uniform prior mixed into the truth
    "max_models": 50_000,  # hidden classes: a refit with more models is refused, the last fit kept
}
REQUIRED = ("scope", "classes", "n_classes", "n_families")


def hybrid_params(cfg: dict | None) -> dict:
    """Validated parameters from the free-form ``walker.hybrid`` config key."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(PARAMS)
    if unknown:
        raise KeyError(f"unknown walker.hybrid keys {sorted(unknown)}")
    p = {**PARAMS, **cfg}
    missing = [k for k in REQUIRED if p[k] is None]
    if missing:
        raise ValueError(f"walker.hybrid needs {missing}")
    if p["scope"] not in ("world", "stream") or p["classes"] not in ("public", "hidden") \
            or int(p["n_classes"]) < 1 or int(p["n_families"]) < 1 or not 0 < p["model_prior"] < 1 \
            or not 0 < p["oracle_floor"] < 1 or int(p["max_models"]) < 1:
        raise ValueError(f"walker.hybrid out of range: {p}")
    return p


def observations(ws, logs, n_families: int, max_size: int = MAX_SIZE) -> list[Observation]:
    """The settled candidates of every recipe of world state ``ws`` that the
    pairwise update over ``logs`` settles at least one candidate of (sound under
    deterministic dynamics), in signature order."""
    from .markov import PairwiseEvidence
    pe = PairwiseEvidence(ws.infos, 2.0, max_size).observe_logs(logs)
    out = []
    for sig in ws.hidden():
        info = ws.infos[sig]
        req = frozenset(b for b in info.pool if (sig, b) in pe.req)
        elim = frozenset(b for b in info.pool if (sig, b) in pe.eliminated)
        if (req or elim) and len(req) <= max_size and not req & elim and len(elim) < len(info.pool):
            family = family_of(info.effect, n_families)
            out.append(Observation((ws.world_key, sig), family, info.pool, req, elim))
    return out


class PriorSource:
    """What a ``Walker`` prior provides (``draw``, ``log_prior``), with a cache
    of each recipe's prior vector keyed by a version."""

    def __init__(self, params: dict, max_size: int = MAX_SIZE):
        self.params, self.max_size = params, max_size
        self._cache: dict = {}

    def version(self, w) -> object:
        raise NotImplementedError

    def log_vector(self, w, sig) -> np.ndarray:
        raise NotImplementedError

    def update(self, agent, w, log) -> None:
        pass

    def counters(self, w) -> dict:
        return {}

    def vector(self, w, sig) -> tuple[np.ndarray, dict]:
        key, ver = (w.world_key, sig), self.version(w)
        hit = self._cache.get(key)
        if hit is None or hit[0] != ver:
            lp = np.asarray(self.log_vector(w, sig), dtype=float)
            p = np.exp(lp - lp.max())
            hit = (ver, p / p.sum(), dict(zip(w.hypotheses[sig], lp.tolist())))
            self._cache[key] = hit
        return hit[1], hit[2]

    def log_prior(self, w, sig, h) -> float:
        return self.vector(w, sig)[1][h]

    def draw(self, w, sig, rng: np.random.Generator) -> frozenset:
        p, _ = self.vector(w, sig)
        return w.hypotheses[sig][int(rng.choice(len(p), p=p))]


class LearnedPrior(PriorSource):
    """Lifted rules learned from the settled candidates of the recipes in the
    learning scope, averaged with the no-schema model; refitted whenever they
    change."""

    def __init__(self, params: dict, max_size: int = MAX_SIZE):
        super().__init__(params, max_size)
        self.scopes: dict[str, ModelAverage] = {}
        self._fitted: dict[str, tuple] = {}
        self._world_obs: dict = {}

    def scope_of(self, world_key: str) -> str:
        return "stream" if self.params["scope"] == "stream" else world_key

    def _learner(self) -> ModelAverage:
        K, F = int(self.params["n_classes"]), int(self.params["n_families"])
        if self.params["classes"] == "public":
            learner = SchemaV1(K, F, self.max_size)
        else:
            learner = SchemaV2(K, F, self.max_size, int(self.params["max_models"]))
        return ModelAverage(learner, self.params["model_prior"], self.max_size)

    def scope(self, w) -> ModelAverage:
        key = self.scope_of(w.world_key)
        if key not in self.scopes:
            self.scopes[key] = self._learner()
        return self.scopes[key]

    def _observations(self, ws, log) -> list[Observation]:
        stamp = (len(ws.evidence.logs), None if log is None else (log.log_id, len(log.steps)))
        hit = self._world_obs.get(ws.world_key)
        if hit is None or hit[0] != stamp:
            hit = (stamp, observations(ws, ws.evidence.all_logs(log), int(self.params["n_families"]),
                                       self.max_size))
            self._world_obs[ws.world_key] = hit
        return hit[1]

    def update(self, agent, w, log) -> None:
        avg = self.scope(w)
        key = self.scope_of(w.world_key)
        worlds = list(agent.worlds.values()) if self.params["scope"] == "stream" else [w]
        obs = [o for ws in worlds for o in self._observations(ws, log if ws is w else None)]
        fingerprint = tuple((o.key, o.required, o.eliminated) for o in obs)
        if self._fitted.get(key) != fingerprint:
            avg.fit(obs)
            self._fitted[key] = fingerprint

    def version(self, w) -> object:
        return self.scope(w).version

    def log_vector(self, w, sig) -> np.ndarray:
        info = w.infos[sig]
        return self.scope(w).log_prior(family_of(info.effect, int(self.params["n_families"])), info.pool,
                                       w.hypotheses[sig])

    def counters(self, w) -> dict:
        avg = self.scope(w)
        out = {"hp_observed": len(avg.observations), "hp_pinned": sum(o.pinned for o in avg.observations),
               "hp_schema_weight": round(avg.weight(), 6)}
        if isinstance(avg.learner, SchemaV2):
            out.update({"hp_models": len(avg.learner.models), "hp_overflow": avg.learner.overflow})
        return out


class HybridAgent(Walker):
    """``focused_sample`` with an opt-in prior (``hybrid``, ``oracle_schema``),
    or the prior's most probable requirements without a walk (``rules_only``)."""

    def __init__(self, arm: str, rng: np.random.Generator, params: dict, prior: PriorSource, *,
                 epsilon: float = 0.0, **walker_kw):
        if arm not in HP_STRATEGIES:
            raise ValueError(f"unknown HP2 strategy {arm!r}")
        if epsilon:
            raise ValueError("the HP2 agents assume deterministic dynamics")
        super().__init__("focused_sample", rng, epsilon=epsilon, prior=prior, **walker_kw)
        self.arm, self.params = arm, params
        self.label = f"hybrid:{arm}"
        self._choose_seconds = 0.0
        self._world = None

    def choose_node(self, w, log, held: frozenset = frozenset()) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        self._world = w
        self.prior.update(self, w, log)
        out = self.rules_node(w, log) if self.arm == "rules_only" else Walker.choose_node(self, w, log, held)
        self._choose_seconds += time.perf_counter() - t0
        return out

    def rules_node(self, w, log) -> tuple[dict, dict]:
        node = {}
        for sig in w.hidden():
            p, _ = self.prior.vector(w, sig)
            best = np.flatnonzero(p >= p.max() * (1.0 - 1e-9))
            node[sig] = w.hypotheses[sig][int(best[int(self.rng.integers(len(best)))])]
        w.node = node
        # diagnostic only (not search work): does the plan contradict logged evidence?
        return node, {"evals": 0, "violations": w.evidence.violations(node, log)}

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        self._choose_seconds = 0.0
        row = super().run_episode(ex, task, dyn_seed)
        extra = self.prior.counters(self._world) if self._world is not None else {}
        return {**row, "choose_seconds": round(self._choose_seconds, 6), **extra}


def walker_kwargs(wc: dict, epsilon: float) -> dict:
    return dict(epsilon=epsilon, max_evals=wc["max_evals"], temperature=wc["temperature"],
                restart_after=wc["restart_after"], exact_cap=wc["exact_cap"], plan_cap=wc["plan_cap"],
                consistent_moves=wc["consistent_moves"])


def make_hybrid_agent(strategy: str, rng, wc: dict, epsilon: float) -> HybridAgent:
    """``hybrid`` and ``rules_only``; ``oracle_schema`` needs the evaluator side
    (``evaluation.hybrid_priors.make_oracle_agent``)."""
    if strategy not in ("hybrid", "rules_only"):
        raise ValueError(f"strategy {strategy!r} is not built here")
    params = hybrid_params(wc.get("hybrid"))
    return HybridAgent(strategy, rng, params, LearnedPrior(params), **walker_kwargs(wc, epsilon))
