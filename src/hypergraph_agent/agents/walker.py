"""Hypergraph walkers: agents that move on a graph of hypergraphs.

The meta-graph
--------------
A node is a complete dependency hypothesis: for every recipe with a hidden
base requirement (keyed by its public signature) one nonempty subset of its
public candidate pool, of size at most ``max_size``. Two nodes are adjacent
when they differ by one single-incidence edit inside that class (add, remove
or swap one pool member), the kind of edit a ``TopologyDelta`` records. The
graph is never materialized: neighbours are generated on demand.

A walker commits to a node, plans on it with the derivation planner (its own
hypothesis, never the hidden rules), acts, and moves when public evidence
contradicts the node. Moves cost no environment interaction; every hypothesis
evaluation is counted as search work.

Strategies
----------
``maximal``        plan on the full-pool node (every candidate required); no inference
``random_omit``    the full-pool node with every candidate omitted independently with
                   probability ``omit_prob``, redrawn at every replan; ignores the evidence
                   (an exploration-matched control for the sampling walkers)
``sample``         posterior sampling with the factorized updater (items observable);
                   a contradicting craft outcome triggers a resample
``optimistic``     the cheapest node still supported by the factorized posterior
                   (fewest base facts not yet held), ties to the higher posterior
``local_uniform``  Metropolis walk with uniformly proposed single edits
``local_focused``  proposals restricted to edits that address a violated observation
``learned``        proposals drawn from a trained edit policy (``agents.edit_policy``)
``exact``          enumeration of the attempted recipes' joint space when it is small;
                   the cheapest consistent node (reference for search quality)
``focused_sample`` / ``learned_sample``
                   the focused or learned walk until a consistent node is reached,
                   then ``consistent_moves`` Metropolis moves restricted to consistent
                   nodes (targeting a uniform draw among them); recipes without
                   evidence are redrawn from the prior at every replan
The strategies from ``local_uniform`` on use episode-level evidence and also
work when intermediate items are unobserved (``observe_items: goal_only``),
where the posterior no longer factorizes by recipe. They assume that items
start absent, and all but the sampling walkers assume deterministic dynamics.

Known failure noise
-------------------
With failure probability ``epsilon > 0`` the sampling walkers score a node by
the exact likelihood of the logged observations (``filter_states``: forward
filtering over the unobserved items, every eligible craft branching into
success and failure). An observed fact the node cannot produce is impossible
under it, while a predicted fact that did not appear is explained by a
failure, so the walk phase counts only the former as violations, and the
consistent moves target the posterior (uniform prior times likelihood)
instead of a uniform draw. Every walker's planning beliefs about unobserved
items are their filtered probabilities under its node, thresholded at 0.5,
and a visibly failed gather or activation triggers a replan. With
``epsilon == 0`` all of this reduces to the deterministic replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..envs.derivations import StructRecipe, crafting_order, enumerate_derivations
from ..envs.public_schema import ACTIVATE, CRAFT, GATHER, SUBMIT, PublicTaskSpec
from ..envs.vocabulary import is_base
from ..topology.inference import DependencyModel, enumerate_hypotheses, evidence_from_transition
from ..training.budget import BudgetExhausted

STRATEGIES = ("maximal", "sample", "optimistic", "local_uniform", "local_focused", "learned", "exact",
              "focused_sample", "learned_sample", "random_omit")
EPISODIC = ("local_uniform", "local_focused", "learned", "exact", "focused_sample", "learned_sample")
SAMPLING = {"focused_sample": "focused", "learned_sample": "learned"}


@dataclass(frozen=True)
class RecipeInfo:
    signature: str
    effect: int
    items: tuple[int, ...]
    pool: tuple[int, ...]
    known: tuple[int, ...] | None


@dataclass(frozen=True)
class Edit:
    signature: str
    remove: int | None
    add: int | None

    @property
    def op(self) -> str:
        if self.remove is None:
            return "add"
        return "remove" if self.add is None else "swap"


@dataclass
class EpisodeLog:
    log_id: int
    goal: int
    initial: frozenset
    steps: list = field(default_factory=list)  # (choice kind, base type | signature | None)
    goal_obs: list = field(default_factory=list)  # observed goal presence after each step
    effect_obs: list = field(default_factory=list)  # observed effect of a non-goal craft, None if unobserved
    base_obs: list = field(default_factory=list)  # gathered/activated base fact observed present after the step

    def attempted(self) -> tuple[str, ...]:
        return tuple(sorted({x for k, x in self.steps if k == CRAFT}))

    def goal_seen(self) -> bool:
        return any(self.goal_obs)

    def base_present(self, t: int) -> bool:
        """Whether the base fact gathered or activated at step ``t`` was observed
        present afterwards (false only after a failure under noise; base facts
        are always observable). Steps without a record count as present."""
        return t >= len(self.base_obs) or self.base_obs[t] is not False


@dataclass(frozen=True)
class CraftTrace:
    t: int
    signature: str
    eligible: bool
    missing: frozenset  # hypothesized base requirements absent at this step
    bases: frozenset  # base facts present at this step
    produced: bool


def apply_edit(node: dict, e: Edit) -> dict:
    cur = set(node[e.signature])
    if e.remove is not None:
        cur.discard(e.remove)
    if e.add is not None:
        cur.add(e.add)
    out = dict(node)
    out[e.signature] = frozenset(cur)
    return out


def neighbours(node: dict, infos: dict, sigs, max_size: int) -> list[Edit]:
    out = []
    for sig in sorted(sigs):
        cur, pool = node[sig], infos[sig].pool
        absent = [b for b in pool if b not in cur]
        if len(cur) < max_size:
            out += [Edit(sig, None, b) for b in absent]
        if len(cur) > 1:
            out += [Edit(sig, b, None) for b in sorted(cur)]
        out += [Edit(sig, o, i) for o in sorted(cur) for i in absent]
    return out


def simulate(node: dict, log: EpisodeLog, infos: dict):
    """Replay a logged episode under a hypothesis, every eligible craft
    succeeding and base facts following their observed outcomes. Returns
    (violations, crafts), where a violation is (step, "F") when the hypothesis
    predicts an observed fact (the goal, or a craft's observable effect) that
    was not observed, or (step, "U") for the converse. Under failure noise an
    "F" violation is explained by a failed craft, a "U" violation is
    impossible: a failure leaves the state unchanged, so it can only keep
    facts from appearing."""
    state = set(log.initial)
    viol, crafts = [], []
    for t, ((kind, x), g, seen) in enumerate(zip(log.steps, log.goal_obs, log.effect_obs)):
        if kind in (GATHER, ACTIVATE):
            if log.base_present(t):
                state.add(x)
        elif kind == CRAFT:
            info = infos[x]
            req = info.known if info.known is not None else node[x]
            missing = frozenset(b for b in req if b not in state)
            ok = not missing and all(i in state for i in info.items)
            produced = ok and info.effect not in state
            if produced:
                state.add(info.effect)
            crafts.append(CraftTrace(t, x, ok, missing, frozenset(b for b in state if is_base(b)), produced))
            if seen is not None and (info.effect in state) != seen:
                viol.append((t, "F" if info.effect in state else "U"))
                (state.add if seen else state.discard)(info.effect)
        pred = log.goal in state
        if pred != g:
            viol.append((t, "F" if pred else "U"))
            (state.add if g else state.discard)(log.goal)  # condition on the observation
    return viol, crafts


def _merged(pairs) -> dict:
    out: dict = {}
    for s, p in pairs:
        out[s] = out.get(s, 0.0) + p
    return out


def _condition(dist: dict, fact: int, value: bool, force: bool) -> tuple[dict, float]:
    kept = {s: p for s, p in dist.items() if (fact in s) == value}
    z = sum(kept.values())
    if z > 0:
        return {s: p / z for s, p in kept.items()}, math.log(z)
    if not force:
        return {}, -math.inf
    return _merged(((s | {fact}) if value else (s - {fact}), p) for s, p in dist.items()), 0.0


def filter_states(node: dict, log: EpisodeLog, infos: dict, epsilon: float,
                  force: bool = False) -> tuple[float, dict]:
    """Forward filter of a logged episode under a hypothesis and known failure
    probability ``epsilon``.

    A state is the set of present facts; base facts follow their observed
    outcomes, items may be hidden. Each craft that is eligible under ``node``
    and whose effect is absent branches into success (1 - epsilon) and failure
    (epsilon); after every step the states are conditioned on what was
    observed (the goal always, a craft's effect when items are observable) and
    identical states are merged. Returns (log-likelihood, {state: probability}),
    the log-likelihood being the sum of the log normalizers: the probability of
    the observed goal and effect sequence given the actions and the observed
    base outcomes (factors that do not depend on the node are left out). An
    impossible observation gives (-inf, {}) unless ``force``, which imposes the
    observation on every state instead (the replay convention of ``simulate``,
    used for planning beliefs). With ``epsilon == 0`` there is one state, the
    replay of ``simulate``, and the likelihood is 1 exactly when ``simulate``
    finds no violation."""
    dist = {frozenset(log.initial): 1.0}
    ll = 0.0
    for t, ((kind, x), g, seen) in enumerate(zip(log.steps, log.goal_obs, log.effect_obs)):
        if kind in (GATHER, ACTIVATE):
            if log.base_present(t):
                dist = _merged((s | {x}, p) for s, p in dist.items())
        elif kind == CRAFT:
            info = infos[x]
            req = info.known if info.known is not None else node[x]
            out = []
            for s, p in dist.items():
                if info.effect not in s and all(b in s for b in req) and all(i in s for i in info.items):
                    out.append((s | {info.effect}, p * (1.0 - epsilon)))
                    if epsilon > 0:
                        out.append((s, p * epsilon))
                else:
                    out.append((s, p))
            dist = _merged(out)
            if seen is not None:
                dist, z = _condition(dist, info.effect, seen, force)
                ll += z
                if not dist:
                    return -math.inf, {}
        dist, z = _condition(dist, log.goal, g, force)
        ll += z
        if not dist:
            return -math.inf, {}
    return ll, dist


class Evidence:
    """Episode logs of one world with memoized violation counts and, under
    failure noise (``epsilon > 0``), memoized log-likelihoods. Under noise only
    impossible observations ("U") count as violations, so a node has no
    violation exactly when its likelihood is positive."""

    def __init__(self, infos: dict, epsilon: float = 0.0):
        self.infos = infos
        self.epsilon = epsilon
        self.logs: list[EpisodeLog] = []
        self._cache: dict = {}
        self._ll_cache: dict = {}
        self.evaluations = 0

    @staticmethod
    def _key(node: dict, log: EpisodeLog) -> tuple:
        return log.log_id, len(log.steps), tuple(node.get(s) for s in log.attempted())

    def violations(self, node: dict, extra: EpisodeLog | None = None) -> int:
        self.evaluations += 1
        total = 0
        for log in self.logs + ([extra] if extra is not None else []):
            key = self._key(node, log)
            if key not in self._cache:
                viol = simulate(node, log, self.infos)[0]
                self._cache[key] = len(viol) if self.epsilon == 0 else sum(k == "U" for _, k in viol)
            total += self._cache[key]
        return total

    def loglik(self, node: dict, extra: EpisodeLog | None = None) -> float:
        """Log-likelihood of every logged observation under ``node`` (one
        hypothesis evaluation)."""
        self.evaluations += 1
        total = 0.0
        for log in self.all_logs(extra):
            key = self._key(node, log)
            if key not in self._ll_cache:
                self._ll_cache[key] = filter_states(node, log, self.infos, self.epsilon)[0]
            total += self._ll_cache[key]
            if total == -math.inf:
                break
        return total

    def all_logs(self, extra: EpisodeLog | None = None) -> list[EpisodeLog]:
        return self.logs + ([extra] if extra is not None else [])


def focused_edits(node: dict, logs, infos: dict, max_size: int, impossible_only: bool = False) -> dict:
    """Edits that address at least one violated observation, with the number
    of violations each addresses (WalkSAT-style focus). ``impossible_only``
    (failure noise): address only observed facts the node cannot produce."""
    counts: dict[Edit, int] = {}
    for log in logs:
        viol, crafts = simulate(node, log, infos)
        for t, kind in viol:
            if impossible_only and kind != "U":
                continue
            for c in crafts:
                if c.t > t or infos[c.signature].known is not None:
                    continue
                cur, pool = node[c.signature], infos[c.signature].pool
                if kind == "F" and c.produced:  # require something that was absent then
                    absent = [b for b in pool if b not in c.bases and b not in cur]
                    if len(cur) < max_size:
                        cands = [Edit(c.signature, None, b) for b in absent]
                    else:
                        cands = [Edit(c.signature, o, b) for o in sorted(cur) for b in absent]
                elif kind == "U" and not c.eligible and c.missing:  # drop what was missing then
                    present = [b for b in pool if b in c.bases and b not in cur]
                    cands = []
                    for b in sorted(c.missing):
                        if len(cur) > 1:
                            cands.append(Edit(c.signature, b, None))
                        cands += [Edit(c.signature, b, p) for p in present]
                else:
                    continue
                for e in cands:
                    counts[e] = counts.get(e, 0) + 1
    return counts


def random_node(w: "WorldState", sigs, rng: np.random.Generator, base: dict) -> dict:
    node = dict(base)
    for sig in sigs:
        hyps = w.hypotheses[sig]
        node[sig] = hyps[int(rng.integers(len(hyps)))]
    return node


def local_walk(start: dict, w: "WorldState", log, mode: str, rng: np.random.Generator, max_size: int,
               max_evals: int, temperature: float, restart_after: int, policy=None, trace=None):
    """Metropolis walk on the meta-graph until a node consistent with all logged
    observations is found or ``max_evals`` hypothesis evaluations are spent.
    Under failure noise consistent means positive likelihood (no impossible
    observation; ``Evidence.violations``).

    ``mode``: ``uniform`` (any neighbour), ``focused`` (an edit addressing a
    violated observation, uniform among those), ``learned`` (``policy`` picks
    among all neighbours). ``trace`` collects (features, choice) for training.
    Returns the best node and statistics.
    """
    ev = w.evidence
    start_evals = ev.evaluations
    logs = ev.all_logs(log)
    attempted = sorted({s for lg in logs for s in lg.attempted() if w.infos[s].known is None})
    cur = dict(start)
    v = ev.violations(cur, log)
    best, best_v = cur, v
    moves = restarts = stall = 0
    prev: Edit | None = None
    while v > 0 and ev.evaluations - start_evals < max_evals:
        if mode == "uniform":
            cands = neighbours(cur, w.infos, attempted, max_size)
            e = cands[int(rng.integers(len(cands)))] if cands else None
        else:
            focus = focused_edits(cur, logs, w.infos, max_size, ev.epsilon > 0)
            if mode == "focused":
                keys = sorted(focus, key=lambda x: (x.signature, x.remove or -1, x.add or -1))
                e = keys[int(rng.integers(len(keys)))] if keys else None
            elif mode == "learned":
                e = policy.propose(cur, w, logs, focus, attempted, prev, rng, trace)
            else:
                raise ValueError(f"unknown walk mode {mode!r}")
        if e is None:
            break
        nxt = apply_edit(cur, e)
        vn = ev.violations(nxt, log)
        if vn <= v or rng.random() < math.exp(-(vn - v) / temperature):
            cur, v, prev = nxt, vn, Edit(e.signature, e.add, e.remove)
            moves += 1
        stall = 0 if v < best_v else stall + 1
        if v < best_v:
            best, best_v = cur, v
        if stall >= restart_after and v > 0:
            cur = random_node(w, attempted, rng, cur)
            v = ev.violations(cur, log)
            restarts += 1
            stall = 0
    return best, {"evals": ev.evaluations - start_evals, "moves": moves, "restarts": restarts,
                  "violations": best_v}


def mix_consistent(node: dict, w: "WorldState", log, rng: np.random.Generator, max_size: int,
                   moves: int) -> tuple[dict, dict]:
    """Metropolis moves restricted to nodes consistent with every logged
    observation. Proposals are uniform over single edits of attempted recipes;
    a move from x to y is accepted with min(1, pi(y) |N(x)| / (pi(x) |N(y)|)),
    which corrects for unequal neighbourhood sizes, so the walk targets pi on
    the nodes connected to the start through nodes with pi > 0. With
    deterministic dynamics pi is uniform over the consistent nodes; under
    failure noise it is the posterior, uniform prior over the hypothesis class
    times the episode likelihood. A proposal with pi(y) = 0 is rejected without
    a random draw."""
    ev = w.evidence
    start = ev.evaluations
    noisy = ev.epsilon > 0
    attempted = sorted({s for lg in ev.all_logs(log) for s in lg.attempted() if w.infos[s].known is None})
    cur = dict(node)
    nb = neighbours(cur, w.infos, attempted, max_size)
    ll = ev.loglik(cur, log) if noisy and nb and moves > 0 else 0.0
    accepted = 0
    for _ in range(moves):
        if not nb:
            break
        nxt = apply_edit(cur, nb[int(rng.integers(len(nb)))])
        if noisy:
            ll_next = ev.loglik(nxt, log)
            if ll_next == -math.inf:
                continue
        elif ev.violations(nxt, log) != 0:
            continue
        nb2 = neighbours(nxt, w.infos, attempted, max_size)
        ratio = len(nb) / max(len(nb2), 1)
        accept = math.exp(min(0.0, ll_next - ll + math.log(ratio))) if noisy else min(1.0, ratio)
        if rng.random() < accept:
            cur, nb = nxt, nb2
            if noisy:
                ll = ll_next
            accepted += 1
    return cur, {"mix_evals": ev.evaluations - start, "mix_accepted": accepted}


class WorldState:
    def __init__(self, world_key: str, epsilon: float, max_size: int):
        self.world_key = world_key
        self.infos: dict[str, RecipeInfo] = {}
        self.dep = DependencyModel(world_key, epsilon, max_size)
        self.evidence = Evidence(self.infos, epsilon)
        self.node: dict[str, frozenset] = {}
        self.hypotheses: dict[str, list[frozenset]] = {}
        self.max_size = max_size

    def register(self, spec: PublicTaskSpec, rng: np.random.Generator):
        for r in spec.rules:
            if r.signature in self.infos:
                continue
            ty = lambda idx: tuple(spec.facts[i].type_id for i in idx)
            known = ty(r.known_base) if r.known_base is not None else None
            info = RecipeInfo(r.signature, spec.facts[r.effect].type_id, ty(r.item_inputs),
                              tuple(sorted(ty(r.pool))), known)
            self.infos[r.signature] = info
            if known is None:
                self.dep.register_rule(r.signature, info.items, info.pool)
                hyps = enumerate_hypotheses(len(info.pool), min(self.max_size, len(info.pool)))
                self.hypotheses[r.signature] = [frozenset(info.pool[p] for p in h) for h in hyps]
                # optimistic start: a single random candidate (cheapest plans first)
                self.node[r.signature] = frozenset({int(rng.choice(info.pool))})

    def hidden(self) -> list[str]:
        return sorted(s for s, i in self.infos.items() if i.known is None)


class Walker:
    def __init__(self, strategy: str, rng: np.random.Generator, *, epsilon: float = 0.0,
                 max_size: int = 3, max_evals: int = 400, temperature: float = 0.5,
                 restart_after: int = 80, exact_cap: int = 20_000, plan_cap: int = 4096,
                 policy=None, consistent_moves: int = 50, omit_prob: float | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        if strategy in ("learned", "learned_sample") and policy is None:
            raise ValueError("the learned strategies need an edit policy")
        if strategy == "random_omit" and not (omit_prob is not None and 0.0 <= omit_prob < 1.0):
            raise ValueError("random_omit needs an omission probability in [0, 1)")
        self.consistent_moves, self.omit_prob = consistent_moves, omit_prob
        self.strategy, self.rng, self.epsilon = strategy, rng, epsilon
        self.max_size, self.max_evals, self.temperature = max_size, max_evals, temperature
        self.restart_after, self.exact_cap, self.plan_cap = restart_after, exact_cap, plan_cap
        self.policy = policy
        self.worlds: dict[str, WorldState] = {}
        self._log_ids = 0
        self.label = f"walker:{strategy}"

    # ------------------------------------------------------------- choosing a node
    def world(self, key: str) -> WorldState:
        if key not in self.worlds:
            self.worlds[key] = WorldState(key, self.epsilon, self.max_size)
        return self.worlds[key]

    def _sample_node(self, w: WorldState) -> dict:
        node = {}
        for sig in w.hidden():
            post = w.dep.rules[sig].posterior()
            node[sig] = w.hypotheses[sig][int(self.rng.choice(len(post), p=post / post.sum()))]
        return node

    def _optimistic_node(self, w: WorldState, held: frozenset) -> dict:
        node = {}
        for sig in w.hidden():
            post = w.dep.rules[sig].posterior()
            hyps = w.hypotheses[sig]
            ok = [i for i, p in enumerate(post) if p > 1e-12]
            i = min(ok, key=lambda i: (len(hyps[i] - held), -post[i], i))
            node[sig] = hyps[i]
        return node

    def _omit_node(self, w: WorldState) -> dict:
        """Each candidate of each pool omitted independently with probability
        ``omit_prob`` = q; a draw that omits the whole pool keeps one candidate
        drawn uniformly (requirements are publicly known to be nonempty). The
        expected omitted fraction of a pool of n is therefore q - q**n / n."""
        node = {}
        for sig in w.hidden():
            pool = w.infos[sig].pool
            keep = [b for b, u in zip(pool, self.rng.random(len(pool))) if u >= self.omit_prob]
            node[sig] = frozenset(keep or [pool[int(self.rng.integers(len(pool)))]])
        return node

    def choose_node(self, w: WorldState, log: EpisodeLog | None,
                    held: frozenset = frozenset()) -> tuple[dict, dict]:
        s = self.strategy
        if s == "maximal":
            return {sig: frozenset(w.infos[sig].pool) for sig in w.hidden()}, {"evals": 0}
        if s == "random_omit":
            return self._omit_node(w), {"evals": 0}
        if s == "sample":
            return self._sample_node(w), {"evals": 0}
        if s == "optimistic":
            return self._optimistic_node(w, held), {"evals": 0}
        if s == "exact":
            node, st = self._exact(w, log)
            if node is not None:
                return node, st
            return self._local(w, log, "focused")  # space too large: declared fallback
        if s in SAMPLING:
            return self._local_sample(w, log, SAMPLING[s])
        return self._local(w, log, {"local_uniform": "uniform", "local_focused": "focused",
                                    "learned": "learned"}[s])

    def _local_sample(self, w: WorldState, log, mode: str) -> tuple[dict, dict]:
        logs = w.evidence.all_logs(log)
        attempted = {s for lg in logs for s in lg.attempted()}
        start = dict(w.node)
        for sig in w.hidden():  # recipes without evidence: a fresh draw from the prior
            if sig not in attempted:
                hyps = w.hypotheses[sig]
                start[sig] = hyps[int(self.rng.integers(len(hyps)))]
        node, st = local_walk(start, w, log, mode, self.rng, self.max_size, self.max_evals,
                              self.temperature, self.restart_after, self.policy)
        if st["violations"] == 0:
            node, mix = mix_consistent(node, w, log, self.rng, self.max_size, self.consistent_moves)
            st = {**st, **mix, "evals": st["evals"] + mix["mix_evals"]}
        w.node = node
        return node, st

    def _local(self, w: WorldState, log, mode: str) -> tuple[dict, dict]:
        best, st = local_walk(dict(w.node), w, log, mode, self.rng, self.max_size, self.max_evals,
                              self.temperature, self.restart_after, self.policy)
        w.node = best
        return best, st

    def _exact(self, w: WorldState, log) -> tuple[dict | None, dict]:
        ev = w.evidence
        logs = ev.all_logs(log)
        attempted = sorted({s for lg in logs for s in lg.attempted() if w.infos[s].known is None})
        size = math.prod(len(w.hypotheses[s]) for s in attempted) if attempted else 1
        if size > self.exact_cap:
            return None, {"evals": 0, "exact_skipped": size}
        start = ev.evaluations
        consistent = []
        for combo in product(*(w.hypotheses[s] for s in attempted)):
            node = dict(w.node)
            node.update(zip(attempted, combo))
            if ev.violations(node, log) == 0:
                consistent.append(node)
        if not consistent:
            return None, {"evals": ev.evaluations - start, "exact_inconsistent": True}
        # the cheapest consistent node (fewest required base facts), ties at random:
        # the ideal version of what the optimistic local walks approximate
        sizes = [sum(len(n[s]) for s in attempted) for n in consistent]
        cheapest = [n for n, k in zip(consistent, sizes) if k == min(sizes)]
        node = cheapest[int(self.rng.integers(len(cheapest)))]
        w.node = node
        return node, {"evals": ev.evaluations - start, "consistent": len(consistent), "violations": 0}

    # ----------------------------------------------------------------- planning
    def plan(self, spec: PublicTaskSpec, w: WorldState, node: dict, believed: frozenset) -> list[str] | None:
        recipes = []
        for r in spec.rules:
            info = w.infos[r.signature]
            bases = info.known if info.known is not None else tuple(sorted(node[r.signature]))
            recipes.append(StructRecipe(info.effect, info.items, bases))
        goal = spec.facts[spec.goal].type_id
        ds, _, _ = enumerate_derivations(recipes, believed, goal, self.plan_cap)
        if not ds:
            return None
        pairs, bases = ds[0]
        by_type = spec.fact_by_type()
        base_key = {spec.facts[a.fact].type_id: a.key for a in spec.actions if a.fact is not None}
        craft_key = {a.rule: a.key for a in spec.actions if a.rule is not None}
        keys = [base_key[b] for b in sorted(bases, key=lambda b: by_type[b])]
        keys += [craft_key[ridx] for _, ridx in crafting_order(pairs, recipes)]
        return keys + ["submit"]

    def believed(self, spec, w: WorldState, node: dict, log: EpisodeLog | None, obs) -> frozenset:
        """Facts the walker plans from: the observed ones plus, when items are
        hidden, the items whose probability under the node, filtered over the
        current episode, is at least 0.5 (with deterministic dynamics: the
        replay of the episode under the node)."""
        seen = {spec.facts[i].type_id for i, p in enumerate(obs.present) if p}
        if spec.items_observable or log is None:
            return frozenset(seen)
        _, dist = filter_states(node, log, w.infos, self.epsilon, force=True)
        mass: dict = {}
        for s, p in dist.items():
            for f in s:
                mass[f] = mass.get(f, 0.0) + p
        return frozenset({f for f, m in mass.items() if m >= 0.5} | seen)

    @staticmethod
    def record(log: EpisodeLog, spec: PublicTaskSpec, desc, after, goal_type: int) -> None:
        target = (spec.facts[desc.fact].type_id if desc.fact is not None else
                  spec.rules[desc.rule].signature if desc.rule is not None else None)
        log.steps.append((desc.kind, target))
        log.goal_obs.append(bool(after[spec.goal]))
        seen = None
        if desc.kind == CRAFT and spec.items_observable:
            eff = spec.rules[desc.rule].effect
            if spec.facts[eff].type_id != goal_type:
                seen = bool(after[eff])
        log.effect_obs.append(seen)
        log.base_obs.append(bool(after[desc.fact]) if desc.kind in (GATHER, ACTIVATE) else None)

    # ------------------------------------------------------------------ episode
    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        ex.reset(task, dyn_seed)
        spec = ex.spec
        w = self.world(spec.world_key)
        w.register(spec, self.rng)
        if self.strategy in EPISODIC and self.strategy not in SAMPLING and self.epsilon > 0:
            raise ValueError("episodic walkers other than the sampling walkers assume deterministic dynamics")
        if abs(spec.failure_prob - self.epsilon) > 1e-12:
            raise ValueError(f"walker epsilon {self.epsilon} differs from the task's declared "
                             f"failure probability {spec.failure_prob}")
        idx = spec.action_index()
        goal = spec.facts[spec.goal].type_id
        self._log_ids += 1
        log = EpisodeLog(self._log_ids, goal,
                         frozenset(spec.facts[i].type_id for i, p in enumerate(ex.obs.present) if p))
        plan: list[str] = []
        node: dict = {}
        stats = {"replans": 0, "evals": 0, "moves": 0, "restarts": 0, "contradictions": 0,
                 "unresolved": 0, "node_changes": 0, "base_failures": 0,
                 "pool_candidates": 0, "omitted_candidates": 0}
        task_hidden = [r.signature for r in spec.rules if r.known_base is None]
        status = None
        while not ex.terminated:
            if not plan:
                held = frozenset(spec.facts[i].type_id for i, p in enumerate(ex.obs.present) if p)
                new, st = self.choose_node(w, log, held)
                stats["node_changes"] += sum(1 for s in new if node and new[s] != node.get(s))
                node = new
                stats["replans"] += 1
                # candidates the chosen node omits among the task's hidden recipes (exploration rate)
                stats["pool_candidates"] += sum(len(w.infos[s].pool) for s in task_hidden)
                stats["omitted_candidates"] += sum(len(w.infos[s].pool) - len(node[s]) for s in task_hidden)
                stats["evals"] += st.get("evals", 0)
                stats["moves"] += st.get("moves", 0)
                stats["restarts"] += st.get("restarts", 0)
                stats["unresolved"] += int(st.get("violations", 0) > 0)
                plan = self.plan(spec, w, node, self.believed(spec, w, node, log, ex.obs)) or \
                    self.plan(spec, w, {s: frozenset(w.infos[s].pool) for s in w.hidden()},
                              self.believed(spec, w, node, log, ex.obs)) or ["wait"]
            key = plan.pop(0)
            desc = spec.actions[idx[key]]
            before = ex.obs.present
            try:
                if not ex.meter.can_charge(1):
                    raise BudgetExhausted("walker budget")
                ex.primitive(idx[key])
            except BudgetExhausted:
                status = "budget_exhausted"
                break
            self.record(log, spec, desc, ex.obs.present, goal)
            if desc.kind in (GATHER, ACTIVATE) and not ex.obs.present[desc.fact]:
                stats["base_failures"] += 1  # a visible failure under noise: replan
                plan = []
            elif desc.kind == CRAFT:
                rule = spec.rules[desc.rule]
                ev = evidence_from_transition(spec, desc.rule, before, ex.obs.present, "w", spec.world_key)
                if ev is not None:
                    w.dep.apply(ev)
                if spec.items_observable:
                    effect_seen = ex.obs.present[rule.effect]
                    if not effect_seen and not before[rule.effect]:
                        stats["contradictions"] += 1  # the node predicted this craft to work
                        plan = []
                elif rule.effect == spec.goal and not ex.obs.present[spec.goal]:
                    stats["contradictions"] += 1  # predicted goal did not appear
                    plan = []
            elif desc.kind == SUBMIT and not ex.terminated:
                plan = []
        if self.strategy in EPISODIC:
            w.evidence.logs.append(log)
        return {"success": ex.success, "status": status or ex.termination,
                "primitive_length": ex.n_primitive, "noop_or_invalid": ex.n_noop,
                "manager_decisions": ex.n_primitive, "skill_calls": 0, **stats}
