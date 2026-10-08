"""Pairwise Markov-graph agents with proximity ranking (study M,
docs/studies/M_markov_ranking.md).

Two representations of what is known about hidden requirements
---------------------------------------------------------------
``hyper``  the walker's own: complete dependency hypergraphs (nodes of the
           meta-graph of ``agents.walker``), each evaluated jointly against every
           logged episode.
``pair``   a weighted star graph: one node per fact and one per recipe, an edge
           (recipe r, candidate b) with weight w(r, b), the belief that b belongs
           to r's hidden requirement, and unit edges from a recipe to its item
           inputs and its effect. Only per-edge numbers are stored.

Pairwise evidence (online)
--------------------------
Observations are processed once, in the order they were made (episodes, then
steps), and never revisited. An observation is the outcome of a craft whose
effect is observed (with goal-only observation: a craft of the goal) and was
absent before:

* success at step t: the recipe fired, so every candidate absent at t is not
  required (eliminated). For each item input, if exactly one producer recipe
  was attempted before t, that producer fired by its last attempt before t
  and its absent candidates there are eliminated too (recursively); with
  several attempted producers the conclusion is disjunctive and nothing is
  eliminated.
* failure at step t: the observation is the monotone formula "the recipe was
  not eligible at t", the OR of its candidates absent at t and, for each item
  input, the AND over the input's attempted producers of the same formula at
  their last attempt before t (TRUE when no producer was attempted: the
  failure is then explained by the missing item). The formula is simplified
  with the edges settled so far. Already satisfied: explained, no update. An
  edge is forced when setting it false falsifies the formula; forced edges
  become required and the formula is simplified again. Each of the k open
  edges left in a formula that is still open receives blame 1/k.
* class bounds, applied to a recipe whenever one of its edges is settled: the
  last open candidate of a recipe with no required candidate is required;
  when ``max_size`` candidates are required the others are eliminated.

Weights: w(r, b) = 0 if eliminated, 1 if required, otherwise
(n0 p0 + s) / (n0 + s), with s the accumulated blame, p0 the prior marginal of
a candidate under the uniform prior over the hypothesis class (16/41 for a
pool of six and requirements of at most three) and n0 the prior strength. A
failure whose open edges are cleared later is not re-attributed: disjunctive
evidence is summarized at once into per-edge counts, which is what a pairwise
representation can keep.

Within an episode, both pair cells also keep per-edge marks: each failure of
the current episode is taken once, in order, and while the node leaves it
unexplained its highest-weight open candidate is marked as required until the
episode ends (``PairAgent.marks``). The pair representation thus carries
per-edge evidence across episodes and the same per-edge marks within one; the
hyper cells use joint consistency with every log throughout.

Personalized PageRank
---------------------
For a symmetric nonnegative weight matrix W, the walk P = D^-1 W (a node
without edges jumps to the seed distribution s) and restart probability a,
pi = a s + (1 - a) pi P, so pi = a s (I - (1 - a) P)^-1: the successor
representation of the walk with discount 1 - a, normalized. The agents score
candidate b of recipe r by its star-graph PageRank personalized at the recipe
node r (``ppr_seed: recipe``), at r's effect fact (``effect``) or at the
current goal, the goal of the latest log (``goal``: one score per fact, shared
by every recipe whose pool holds it). Restart 1.0 means no propagation: the
score is the pairwise weight itself (the limit a -> 1 of the ranking for the
recipe seed). M1 measures these scores and fixes the seed and the restart.

Strategies (all share the planner, beliefs and replanning of ``Walker``)
------------------------------------------------------------------------
``hyper_rank``        the focused walk of ``focused_sample`` to a consistent node and the
                      same ``consistent_moves`` Metropolis moves among consistent nodes;
                      plans on the visited node with the highest stationary PageRank (no
                      teleport: proportional to degree) in the graph of consistent nodes
                      spanned by the visited ones and (``rank_neighbours``) their
                      consistent neighbours, divided by its number of single-edit
                      neighbours in the whole class: the share of its neighbours that are
                      consistent (``central``)
``pair_sample``       an exact draw per recipe over the hypothesis class with
                      P(h) proportional to the product over members b of h of
                      odds(w(r, b)) / odds(p0): the pairwise weights combined with the
                      uniform class prior (eliminated candidates excluded, required ones
                      included); then the within-episode marks (``PairAgent.marks``)
``pair_rank``         deterministic: per recipe the required candidates, then open ones by
                      descending star-graph score (``ppr_scores``), up to
                      k = max(1, round(sum of weights)) candidates, at most max_size; then
                      the same within-episode marks
``hyper_sample_ppr``  ``focused_sample`` whose focused edits are proposed in proportion to
                      the star-graph score of the edit's candidate (``PPRProposal``)
                      instead of uniformly; the consistent moves are unchanged
Every strategy assumes deterministic dynamics. Each episode row also records
``choose_seconds``, the wall-clock time spent choosing nodes.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from ..envs.vocabulary import is_base
from ..topology.inference import prior_marginal
from .walker import Walker, apply_edit, local_walk, neighbours

MARKOV_STRATEGIES: tuple[str, ...] = ("hyper_rank", "pair_sample", "pair_rank", "hyper_sample_ppr")
PPR_SEEDS = ("recipe", "effect", "goal")
PARAMS = {
    "restart": 0.15,  # PPR restart probability in (0, 1]; 1.0: no propagation (fixed from M1)
    "ppr_seed": "recipe",  # the recipe node, its "effect" or the current "goal" (fixed from M1)
    "prior_strength": 2.0,  # n0 of the pairwise weights
    "rank_neighbours": True,  # hyper_rank: add the visited nodes' consistent neighbours
    "proposal_floor": 0.1,  # hyper_sample_ppr: every focused edit keeps this weight
}


def markov_params(cfg: dict | None) -> dict:
    """Validated parameters from the free-form ``walker.markov`` config key. A
    null ``restart`` or ``ppr_seed`` is a placeholder still to be set (study M
    writes the M1 selection into the M2 config); agents refuse it."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(PARAMS)
    if unknown:
        raise KeyError(f"unknown walker.markov keys {sorted(unknown)}")
    p = {**PARAMS, **cfg}
    unset = [k for k in ("restart", "ppr_seed") if p[k] is None]
    if unset:
        raise ValueError(f"walker.markov {unset} not set: write in the M1 selection before running")
    if not 0 < p["restart"] <= 1 or p["prior_strength"] <= 0 or not 0 <= p["proposal_floor"] <= 1 \
            or p["ppr_seed"] not in PPR_SEEDS:
        raise ValueError(f"walker.markov out of range: {p}")
    return p


# ------------------------------------------------------------------ formulas
# A formula is True, False, ("lit", (signature, base)), ("or", parts) or ("and", parts);
# a literal states that the base belongs to the recipe's requirement. Formulas are monotone.

def lit(sig: str, b: int) -> tuple:
    return ("lit", (sig, b))


def any_of(parts) -> object:
    out = []
    for p in parts:
        if p is True:
            return True
        if p is not False:
            out.append(p)
    if not out:
        return False
    return out[0] if len(out) == 1 else ("or", tuple(out))


def all_of(parts) -> object:
    out = []
    for p in parts:
        if p is False:
            return False
        if p is not True:
            out.append(p)
    if not out:
        return True
    return out[0] if len(out) == 1 else ("and", tuple(out))


def simplify(f, false=frozenset(), true=frozenset()):
    """Substitute known edges (``false``: eliminated, ``true``: required)."""
    if f is True or f is False:
        return f
    if f[0] == "lit":
        return False if f[1] in false else True if f[1] in true else f
    parts = [simplify(c, false, true) for c in f[1]]
    return any_of(parts) if f[0] == "or" else all_of(parts)


def literals(f) -> set:
    if f is True or f is False:
        return set()
    if f[0] == "lit":
        return {f[1]}
    return set().union(*(literals(c) for c in f[1]))


def literal_order(f) -> list:
    """Literals in depth-first order (the failed recipe's own candidates first)."""
    if f is True or f is False:
        return []
    if f[0] == "lit":
        return [f[1]]
    out = []
    for c in f[1]:
        out += [x for x in literal_order(c) if x not in out]
    return out


def holds(f, node: dict) -> bool:
    """Value of a formula when the requirements are those of ``node``."""
    if f is True or f is False:
        return f
    if f[0] == "lit":
        sig, b = f[1]
        return b in node.get(sig, ())
    vals = (holds(c, node) for c in f[1])
    return any(vals) if f[0] == "or" else all(vals)


# --------------------------------------------------------------- observations

def base_sets(log) -> list[frozenset]:
    """Base facts present before each step: the initial ones and every base
    gathered or activated earlier (a step recorded as failed does not count)."""
    obs = getattr(log, "base_obs", ())
    present = {f for f in log.initial if is_base(f)}
    out = []
    for t, (kind, x) in enumerate(log.steps):
        out.append(frozenset(present))
        if kind in (GATHER, ACTIVATE) and (t >= len(obs) or obs[t] is not False):
            present.add(x)
    return out


class EpisodeReplay:
    """Craft attempts of one logged episode with the base facts present at each
    step, and the logical content of its observations (deterministic dynamics,
    items absent at reset unless observed initially)."""

    def __init__(self, log, infos: dict):
        self.log, self.infos = log, infos
        self.bases = base_sets(log)
        self.crafts = [(t, x) for t, (k, x) in enumerate(log.steps) if k == CRAFT]

    def producers(self, item: int, before: int) -> dict:
        """{signature: last attempt before step ``before``} of the attempted
        recipes producing ``item``."""
        out = {}
        for t, s in self.crafts:
            if t < before and self.infos[s].effect == item:
                out[s] = t
        return out

    def not_eligible(self, sig: str, t: int):
        """Formula: recipe ``sig`` could not fire at its attempt at step ``t``."""
        info = self.infos[sig]
        if info.known is None:
            parts = [lit(sig, b) for b in info.pool if b not in self.bases[t]]
        else:
            parts = [any(b not in self.bases[t] for b in info.known)]
        parts += [self.absent(i, t) for i in info.items]
        return any_of(parts)

    def absent(self, item: int, t: int):
        """Formula: ``item`` was absent before step ``t`` (every attempted producer failed)."""
        if item in self.log.initial:
            return False
        prods = self.producers(item, t)
        if not prods:
            return True
        return all_of([self.not_eligible(s, t2) for s, t2 in sorted(prods.items())])

    def fired(self, sig: str, t: int, out: set) -> set:
        """Edges eliminated because ``sig`` fired by step ``t`` (single-producer recursion)."""
        info = self.infos[sig]
        if info.known is None:
            out.update((sig, b) for b in info.pool if b not in self.bases[t])
        for i in info.items:
            if i in self.log.initial:
                continue
            prods = self.producers(i, t)
            if len(prods) == 1:
                (s, t2), = prods.items()
                self.fired(s, t2, out)
        return out

    def observations(self) -> list[tuple]:
        """("success", t, eliminated edges) or ("failure", t, formula), in step
        order, for every craft whose effect is observed and was absent before."""
        out, held = [], set()
        for t, sig in self.crafts:
            eff = self.infos[sig].effect
            if eff == self.log.goal:
                before, after = any(self.log.goal_obs[:t]), bool(self.log.goal_obs[t])
            elif self.log.effect_obs[t] is not None:
                before, after = eff in held, bool(self.log.effect_obs[t])
            else:
                continue
            if after:
                held.add(eff)
            if before:
                continue
            if after:
                out.append(("success", t, frozenset(self.fired(sig, t, set()))))
            else:
                out.append(("failure", t, self.not_eligible(sig, t)))
        return out


def observations(log, infos: dict) -> list[tuple]:
    return EpisodeReplay(log, infos).observations()


class PairwiseEvidence:
    """Online pairwise (star-graph) weights; see the module docstring."""

    def __init__(self, infos: dict, prior_strength: float = 2.0, max_size: int = 3):
        self.infos, self.n0, self.max_size = infos, prior_strength, max_size
        self.eliminated: set = set()
        self.req: set = set()
        self.blame: dict = defaultdict(float)
        self.counts = {"successes": 0, "failures": 0, "explained": 0, "forced": 0, "blamed": 0,
                       "contradictions": 0}

    def prior(self, sig: str) -> float:
        n = len(self.infos[sig].pool)
        return prior_marginal(n, min(self.max_size, n))

    def observe_logs(self, logs) -> "PairwiseEvidence":
        for log in logs:
            for obs in observations(log, self.infos):
                self.observe(obs)
        return self

    def observe(self, obs: tuple) -> None:
        kind, _, x = obs
        if kind == "success":
            self.counts["successes"] += 1
            for e in sorted(x):
                self._settle(e, False)
            return
        self.counts["failures"] += 1
        f = simplify(x, self.eliminated, self.req)
        if f is True or f is False:
            self.counts["explained" if f is True else "contradictions"] += 1
            return
        forced = [e for e in sorted(literals(f)) if simplify(f, {e}) is False]
        if forced:
            self.counts["forced"] += 1
            for e in forced:
                self._settle(e, True)
            f = simplify(f, self.eliminated, self.req)
            if f is True or f is False:
                return
        self.counts["blamed"] += 1
        open_ = sorted(literals(f))
        for e in open_:
            self.blame[e] += 1.0 / len(open_)

    def _settle(self, e: tuple, value: bool) -> None:
        if e in (self.eliminated if value else self.req):
            self.counts["contradictions"] += 1
            return
        (self.req if value else self.eliminated).add(e)
        self._bounds(e[0])

    def _bounds(self, sig: str) -> None:
        pool = self.infos[sig].pool
        while True:
            open_ = [b for b in pool if (sig, b) not in self.eliminated and (sig, b) not in self.req]
            n_req = sum((sig, b) in self.req for b in pool)
            if n_req == 0 and len(open_) == 1:
                self.req.add((sig, open_[0]))
            elif n_req >= self.max_size and open_:
                self.eliminated.update((sig, b) for b in open_)
            else:
                return

    def weight(self, sig: str, b: int) -> float:
        if (sig, b) in self.eliminated:
            return 0.0
        if (sig, b) in self.req:
            return 1.0
        s = self.blame.get((sig, b), 0.0)
        return (self.n0 * self.prior(sig) + s) / (self.n0 + s)

    def weights(self) -> dict:
        return {(sig, b): self.weight(sig, b) for sig, info in self.infos.items() if info.known is None
                for b in info.pool}


# ------------------------------------------------------------------- PageRank

def personalized_pagerank(W: np.ndarray, seed, restart: float) -> np.ndarray:
    """pi = restart * seed + (1 - restart) * pi P with P = D^-1 W; a row without
    weight jumps to the seed distribution."""
    s = np.asarray(seed, dtype=float)
    s = s / s.sum()
    deg = W.sum(axis=1)
    P = np.where(deg[:, None] > 0, W / np.where(deg > 0, deg, 1.0)[:, None], s[None, :])
    return np.linalg.solve(np.eye(len(W)) - (1.0 - restart) * P.T, restart * s)


def star_graph(infos: dict, weight) -> tuple[list, dict, np.ndarray]:
    """Fact nodes (type ids) and recipe nodes (signatures); returns (nodes, index, W)."""
    facts = sorted({f for i in infos.values() for f in (i.effect, *i.items, *i.pool, *(i.known or ()))})
    nodes = facts + sorted(infos)
    idx = {n: k for k, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))

    def link(a, b, x):
        W[idx[a], idx[b]] += x
        W[idx[b], idx[a]] += x

    for sig, info in infos.items():
        link(sig, info.effect, 1.0)
        for i in info.items:
            link(sig, i, 1.0)
        if info.known is not None:
            for b in info.known:
                link(sig, b, 1.0)
            continue
        for b in info.pool:
            x = weight(sig, b)
            if x > 0:
                link(sig, b, x)
    return nodes, idx, W


def _seed(idx: dict, facts) -> np.ndarray:
    s = np.zeros(len(idx))
    for f in facts:
        if f in idx:
            s[idx[f]] = 1.0
    if s.sum() == 0:
        s[:] = 1.0
    return s


def candidate_ppr(idx: dict, W: np.ndarray, infos: dict, sigs, restart: float, seed_of) -> dict:
    """Score of candidate b for recipe r: its PageRank on the graph (idx, W)
    personalized uniformly at the nodes ``seed_of(r)`` (at every node when none
    of them is in the graph). Restart 1.0 (no propagation): the weight of the
    edge between r's seed and b, i.e. W[seed, b] summed over the seed nodes."""
    out, cache = {}, {}
    for sig in sigs:
        key = tuple(seed_of(sig))
        if key not in cache:
            s = _seed(idx, key)
            cache[key] = s @ W if restart >= 1.0 else personalized_pagerank(W, s, restart)
        pi = cache[key]
        for b in infos[sig].pool:
            out[(sig, b)] = float(pi[idx[b]]) if b in idx else 0.0
    return out


def star_seed(infos: dict, how: str, goals=()):
    """Personalization of the star-graph scores: the recipe node itself
    (``recipe``), its effect fact (``effect``) or the goal fact(s) (``goal``)."""
    if how == "recipe":
        return lambda sig: (sig,)
    if how == "effect":
        return lambda sig: (infos[sig].effect,)
    if how == "goal":
        return lambda sig: tuple(goals)
    raise ValueError(f"unknown PageRank seed {how!r}")


def star_scores(infos: dict, pe: "PairwiseEvidence", restart: float, seed: str, goals=(), sigs=None) -> dict:
    """The agents' star-graph scores with the pairwise weights of ``pe`` (every
    recipe with a hidden requirement unless ``sigs``). Restart 1.0: the
    pairwise weight w(r, b) for every seed (no propagation)."""
    sigs = sorted(s for s, i in infos.items() if i.known is None) if sigs is None else sigs
    if restart >= 1.0:
        return {(s, b): pe.weight(s, b) for s in sigs for b in infos[s].pool}
    _, idx, W = star_graph(infos, pe.weight)
    return candidate_ppr(idx, W, infos, sigs, restart, star_seed(infos, seed, goals))


def ppr_scores(infos: dict, pe: "PairwiseEvidence", params: dict, logs) -> dict:
    """The agents' star-graph scores per (recipe, candidate), seeded as
    ``ppr_seed`` says; the goal is that of the latest log (the current episode)."""
    return star_scores(infos, pe, params["restart"], params["ppr_seed"], [lg.goal for lg in logs[-1:]])


# --------------------------------------------------------------------- agents

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


class MarkovAgent(Walker):
    """Runs the ``Walker`` episode loop as an episodic walker (its logs are kept
    as evidence) and replaces the node choice (``select``). ``self.strategy``
    stays the underlying walker strategy; ``self.arm`` is the study-M strategy."""

    base_strategy = "focused_sample"

    def __init__(self, arm: str, rng: np.random.Generator, params: dict, *, epsilon: float = 0.0,
                 policy=None, **walker_kw):
        if epsilon:
            raise ValueError("the study-M agents assume deterministic dynamics")
        super().__init__(self.base_strategy, rng, epsilon=epsilon, policy=policy, **walker_kw)
        self.arm, self.params = arm, params
        self.label = f"markov:{arm}"
        self._choose_seconds = 0.0

    def choose_node(self, w, log, held: frozenset = frozenset()) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        out = self.select(w, log, held)
        self._choose_seconds += time.perf_counter() - t0
        return out

    def select(self, w, log, held: frozenset) -> tuple[dict, dict]:
        return Walker.choose_node(self, w, log, held)

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        self._choose_seconds = 0.0
        row = super().run_episode(ex, task, dyn_seed)
        return {**row, "choose_seconds": round(self._choose_seconds, 6)}

    def pairwise(self, w, log) -> PairwiseEvidence:
        return PairwiseEvidence(w.infos, self.params["prior_strength"], self.max_size).observe_logs(
            w.evidence.all_logs(log))


class HyperRank(MarkovAgent):
    def select(self, w, log, held: frozenset) -> tuple[dict, dict]:
        attempted = {s for lg in w.evidence.all_logs(log) for s in lg.attempted()}
        start = dict(w.node)
        for sig in w.hidden():  # as focused_sample: recipes without evidence redrawn from the prior
            if sig not in attempted:
                hyps = w.hypotheses[sig]
                start[sig] = hyps[int(self.rng.integers(len(hyps)))]
        node, st = local_walk(start, w, log, "focused", self.rng, self.max_size, self.max_evals,
                              self.temperature, self.restart_after)
        if st["violations"] == 0:
            node, rank = self.central(node, w, log)
            st = {**st, **rank, "evals": st["evals"] + rank["mix_evals"] + rank["rank_evals"]}
        w.node = node
        return node, st

    def central(self, node: dict, w, log) -> tuple[dict, dict]:
        """The consistent moves of ``mix_consistent`` (deterministic case), then
        the visited node with the largest share of consistent single-edit
        neighbours (ties: the most recently visited). The visited nodes and
        their consistent neighbours form a connected graph, on which PageRank
        without teleport, the stationary walk, is proportional to degree; the
        score is that PageRank divided by the node's number of neighbours in
        the whole class, which is not regular (a recipe with a pool of six has
        10, 14 and 12 neighbours at sizes 1, 2 and 3), so raw PageRank would
        favour size two. Teleport would add the same mass to every node and,
        after the division, favour small hypotheses."""
        ev = w.evidence
        e0 = ev.evaluations
        sigs = sorted({s for lg in ev.all_logs(log) for s in lg.attempted() if w.infos[s].known is None})
        key = lambda n: tuple(n[s] for s in sigs)  # noqa: E731
        cur = dict(node)
        nb = neighbours(cur, w.infos, sigs, self.max_size)
        visited = {key(cur): 0}
        members = {key(cur): cur}
        accepted = 0
        for m in range(self.consistent_moves):
            if not nb:
                break
            nxt = apply_edit(cur, nb[int(self.rng.integers(len(nb)))])
            if ev.violations(nxt, log) != 0:
                continue
            nb2 = neighbours(nxt, w.infos, sigs, self.max_size)
            if self.rng.random() < min(1.0, len(nb) / max(len(nb2), 1)):
                cur, nb = nxt, nb2
                accepted += 1
                visited[key(cur)] = m + 1
                members[key(cur)] = cur
        e1 = ev.evaluations
        if self.params["rank_neighbours"]:
            checked = set(members)
            for k in list(visited):
                for e in neighbours(members[k], w.infos, sigs, self.max_size):
                    u = apply_edit(members[k], e)
                    ku = key(u)
                    if ku not in checked:
                        checked.add(ku)
                        if ev.violations(u, log) == 0:
                            members[ku] = u
        deg = single_edit_degrees(list(members))
        n_class = {k: len(neighbours(members[k], w.infos, sigs, self.max_size)) for k in visited}
        score = {k: deg[k] / max(1, n_class[k]) for k in visited}
        best = max(visited, key=lambda k: (round(score[k], 12), visited[k]))
        return dict(members[best]), {"mix_evals": e1 - e0, "mix_accepted": accepted,
                                     "rank_evals": ev.evaluations - e1, "rank_visited": len(visited),
                                     "rank_members": len(members)}


def single_edit_degrees(keys: list) -> dict:
    """Degree of every hypothesis key in the graph of the keys, two keys
    adjacent when they differ in one recipe by a single edit (one candidate
    added or removed, or one swapped)."""
    deg = {k: 0 for k in keys}
    for j in range(len(keys[0]) if keys else 0):
        buckets = defaultdict(list)
        for k in keys:
            buckets[k[:j] + k[j + 1:]].append(k)
        for group in buckets.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    x, y = group[a][j], group[b][j]
                    d = len(x ^ y)
                    if d == 1 or (d == 2 and len(x) == len(y)):
                        deg[group[a]] += 1
                        deg[group[b]] += 1
    return deg


class PairAgent(MarkovAgent):
    """Both pair cells: a node from the pairwise weights (a class draw or the
    ranking), then the per-edge within-episode marks (``marks``), the same
    rule in both cells. Each episode row adds ``pair_marks`` (marks made),
    ``refuted_replans`` (replans whose node a failure of the same episode had
    refuted) and ``repeated_nodes`` (those that repeat a node already chosen
    and refuted in the episode, compared on the recipes of its failures)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._reset_episode(None)

    def _reset_episode(self, log_id) -> None:
        self._episode, self._marks, self._done, self._seen = log_id, [], 0, set()
        self._counts = {"pair_marks": 0, "refuted_replans": 0, "repeated_nodes": 0}

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        self._reset_episode(None)
        row = super().run_episode(ex, task, dyn_seed)
        return {**row, **self._counts}

    def select(self, w, log, held: frozenset) -> tuple[dict, dict]:
        pe = self.pairwise(w, log)
        rank = self.arm == "pair_rank"
        pi = ppr_scores(w.infos, pe, self.params, w.evidence.all_logs(log)) if rank else None
        node = {}
        for sig in w.hidden():
            pool = w.infos[sig].pool
            wt = {b: pe.weight(sig, b) for b in pool}
            live = [b for b in pool if wt[b] > 0] or list(pool)
            req = [b for b in live if wt[b] >= 1.0]
            k = max(1, len(req), _round_half_up(sum(wt.values())))
            k = min(k, max(self.max_size, len(req)), len(live))
            if rank:
                order = sorted(live, key=lambda b: (wt[b] < 1.0, -round(pi[(sig, b)], 12), -wt[b], b))
                node[sig] = frozenset(order[:k])
            else:
                node[sig] = self.draw(w.hypotheses[sig], wt, pe.prior(sig), live, k)
        st = {"evals": 0, **pe.counts}
        node = self.marks(node, w, log, pe)
        w.node = node
        # diagnostic only (not search work): does the plan contradict logged evidence?
        st["violations"] = w.evidence.violations(node, log)
        return node, st

    def draw(self, hyps, wt: dict, p0: float, live, k) -> frozenset:
        """One hypothesis of the class, P(h) proportional to the product over
        members b of odds(w_b) / odds(p0); zero if h holds an eliminated
        candidate or misses a required one."""
        r0 = p0 / (1.0 - p0)
        req = {b for b, x in wt.items() if x >= 1.0}
        p = np.array([0.0 if any(wt[b] <= 0 for b in h) or not req <= h
                      else math.prod(wt[b] / (1.0 - wt[b]) / r0 for b in h if wt[b] < 1.0) for h in hyps])
        if p.sum() <= 0:
            return frozenset(sorted(live, key=lambda b: (-wt[b], b))[:k])
        return hyps[int(self.rng.choice(len(hyps), p=p / p.sum()))]

    def marks(self, node: dict, w, log, pe: PairwiseEvidence) -> dict:
        """Per-edge within-episode marks. Edges marked earlier in the episode
        are put into the node. Then each failure of the episode not yet
        processed is taken once, in order: while the node leaves it
        unexplained (its formula is false under the node), its highest-weight
        open candidate not yet in the node (ties: the failed recipe's own
        candidates first) is marked as required for the rest of the episode
        and put into the node. A recipe already holding max_size members swaps
        out its lowest-weight member that is neither marked nor required (ties:
        the larger type id); if there is none, the mark is kept but not placed.
        Earlier failures are not checked again. Residual refutations are
        counted, not prevented."""
        if log is None:
            return node
        if log.log_id != self._episode:
            self._reset_episode(log.log_id)
        node = dict(node)
        for e in self._marks:
            node = self._include(node, e, pe)
        fails = [f for kind, _, f in observations(log, w.infos) if kind == "failure"]
        for f in fails[self._done:]:
            while not holds(f, node):
                open_ = [e for e in literal_order(f) if e not in pe.eliminated
                         and e[1] not in node.get(e[0], ()) and e not in self._marks]
                if not open_:
                    break
                e = max(open_, key=lambda x: (pe.weight(*x), -open_.index(x)))
                self._marks.append(e)
                self._counts["pair_marks"] += 1
                node = self._include(node, e, pe)
        self._done = len(fails)
        sigs = sorted({s for f in fails for s, _ in literals(f)})
        if any(not holds(f, node) for f in fails):
            key = tuple(node[s] for s in sigs)
            self._counts["refuted_replans"] += 1
            self._counts["repeated_nodes"] += int(key in self._seen)
            self._seen.add(key)
        return node

    def _include(self, node: dict, e: tuple, pe: PairwiseEvidence) -> dict:
        s, b = e
        cur = node.get(s, frozenset())
        if b in cur:
            return node
        if len(cur) < self.max_size:
            return {**node, s: cur | {b}}
        marked = {x for y, x in self._marks if y == s}
        outs = [o for o in cur if o not in marked and (s, o) not in pe.req]
        if not outs:
            return node
        o = min(outs, key=lambda o: (pe.weight(s, o), -o))
        return {**node, s: (cur - {o}) | {b}}


class PPRProposal:
    """Focused-edit proposals for the walk phase of a sampling walker. A focused
    edit on recipe r is proposed with probability proportional to
    floor + (1 - floor) q, where q(add b) = s(r, b), q(remove b) = 1 - s(r, b)
    and q(swap o for b) = (s(r, b) + 1 - s(r, o)) / 2; s(r, b) is the
    agents' star-graph score of candidate b for r (``ppr_scores``),
    divided by the largest such score in r's pool (0 for an eliminated edge)."""

    def __init__(self, params: dict, max_size: int = 3):
        self.params, self.max_size = params, max_size
        self._cache: tuple | None = None

    def scores(self, w, logs) -> dict:
        key = (id(w), tuple((lg.log_id, len(lg.steps)) for lg in logs))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        pe = PairwiseEvidence(w.infos, self.params["prior_strength"], self.max_size).observe_logs(logs)
        pi = ppr_scores(w.infos, pe, self.params, logs)
        out = {}
        for sig, info in w.infos.items():
            if info.known is not None:
                continue
            raw = {b: (pi[(sig, b)] if pe.weight(sig, b) > 0 else 0.0) for b in info.pool}
            top = max(raw.values())
            out.update({(sig, b): (v / top if top > 0 else 0.5) for b, v in raw.items()})
        self._cache = (key, out)
        return out

    def propose(self, cur, w, logs, focus, attempted, prev, rng, trace=None):
        keys = sorted(focus, key=lambda x: (x.signature, x.remove or -1, x.add or -1))
        if not keys:
            return None
        s = self.scores(w, logs)
        lo = self.params["proposal_floor"]
        phi = np.empty(len(keys))
        for i, e in enumerate(keys):
            if e.remove is None:
                q = s[(e.signature, e.add)]
            elif e.add is None:
                q = 1.0 - s[(e.signature, e.remove)]
            else:
                q = 0.5 * (s[(e.signature, e.add)] + 1.0 - s[(e.signature, e.remove)])
            phi[i] = lo + (1.0 - lo) * q
        return keys[int(rng.choice(len(keys), p=phi / phi.sum()))]


class HyperSamplePPR(MarkovAgent):
    """``focused_sample`` with ``PPRProposal`` in the walk phase (run as the
    walker's learned-proposal sampling strategy with this proposal)."""

    base_strategy = "learned_sample"


def make_markov_agent(strategy: str, rng, wc: dict, epsilon: float):
    if strategy not in MARKOV_STRATEGIES:
        raise ValueError(f"unknown Markov strategy {strategy!r}")
    params = markov_params(wc.get("markov"))
    kw = dict(epsilon=epsilon, max_evals=wc["max_evals"], temperature=wc["temperature"],
              restart_after=wc["restart_after"], exact_cap=wc["exact_cap"], plan_cap=wc["plan_cap"],
              consistent_moves=wc["consistent_moves"])
    if strategy == "hyper_rank":
        return HyperRank(strategy, rng, params, **kw)
    if strategy == "hyper_sample_ppr":
        return HyperSamplePPR(strategy, rng, params, policy=PPRProposal(params), **kw)
    return PairAgent(strategy, rng, params, **kw)
