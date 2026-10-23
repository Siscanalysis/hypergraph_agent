"""Potential-based reward shaping of study XS (learner side; public information only).

Ng, Harada and Russell (ICML 1999): adding ``F = gamma * Phi(s') - Phi(s)`` to every reward, for
any function ``Phi`` of the state, leaves the optimal policies of the task unchanged. In an
episodic task ``Phi`` is 0 at every terminal state, so the shaped discounted return of a learner
episode is the environment's return minus ``Phi(s_0)``, a constant of the start state
(``Shaper``; tested). ``gamma`` must be the learner's own discount (``learner_gamma``).

Potentials (``POTENTIALS``), each in [0, 1] and computed only from the public spec and the public
observation of the current task, never from a hidden requirement, sequence, parent or unlock:

- ``rq_goal_closure`` (RecipeQuest). The goal's public closure is the set of facts reached from the
  goal through the item inputs of the recipes that produce them (recipes and item inputs are
  public). Its candidate base facts are the union of the public candidate pools of those recipes
  (the public base sets in the known-structure profile). ``Phi = base_weight * (share of those
  base facts observed held) + goal_weight * (goal observed held)``. Under goal-only observation
  intermediate items are reported absent, so they never enter.
- ``tt_levels`` (TechTree). ``Phi = concept_weight * (sum of the public levels of the concepts
  held) / (sum of the levels of every concept) + goal_weight * (goal held)``. Which concept is a
  key, a part of the goal or a distractor is hidden, so every held concept counts.

A wrapper with a ``Shaper`` adds ``F`` to the reward of every step of training; evaluation never
shapes. The end of a learner episode (a task, or a world block in trial mode) is the terminal
state: there ``Phi`` is 0; inside a trial the first state of the next task is the next state.
"""

from __future__ import annotations

import numpy as np

from ..envs.public_schema import PublicObservation, PublicTaskSpec
from ..techtree.env import TechObservation, TechTreeSpec

DEFAULT_GAMMA = 0.99  # the discount of PPO, MaskablePPO and RecurrentPPO (tested against the libraries)
POTENTIALS = {
    "rq_goal_closure": ("recipequest", {"base_weight": 0.5, "goal_weight": 0.5}),
    "tt_levels": ("techtree", {"concept_weight": 0.5, "goal_weight": 0.5}),
}


def normalize(shaping: dict | None) -> dict | None:
    """A learner's ``shaping`` entry with its potential's default weights filled in (None: no shaping)."""
    if shaping is None:
        return None
    if not isinstance(shaping, dict) or shaping.get("potential") not in POTENTIALS:
        raise ValueError(f"shaping needs a potential in {sorted(POTENTIALS)} (got {shaping!r})")
    weights = POTENTIALS[shaping["potential"]][1]
    extra = set(shaping) - {"potential", *weights}
    if extra:
        raise KeyError(f"unknown shaping keys {sorted(extra)} for potential {shaping['potential']!r}")
    out = {"potential": shaping["potential"], **weights,
           **{k: float(v) for k, v in shaping.items() if k != "potential"}}
    if any(not 0.0 <= out[k] <= 1.0 for k in weights) or sum(out[k] for k in weights) > 1.0:
        raise ValueError("shaping weights must lie in [0, 1] and sum to at most 1")
    return out


def family(shaping: dict) -> str:
    return POTENTIALS[shaping["potential"]][0]


def learner_gamma(params: dict) -> float:
    return float(params.get("gamma", DEFAULT_GAMMA))


def rq_closure_bases(spec: PublicTaskSpec) -> tuple[int, ...]:
    """Fact indices of the candidate base facts of the goal's public closure."""
    producers: dict[int, list[int]] = {}
    for j, r in enumerate(spec.rules):
        producers.setdefault(r.effect, []).append(j)
    seen: set[int] = set()
    stack, bases = [spec.goal], set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for j in producers.get(f, ()):
            r = spec.rules[j]
            stack.extend(r.item_inputs)
            bases.update(r.known_base if r.known_base is not None else r.pool)
    return tuple(sorted(bases))


class RQPotential:
    """``rq_goal_closure``: the closure's candidate base facts observed held, and the goal."""

    def __init__(self, base_weight: float, goal_weight: float):
        self.base_weight, self.goal_weight = base_weight, goal_weight
        self._spec = None

    def __call__(self, spec: PublicTaskSpec, obs: PublicObservation) -> float:
        if spec is not self._spec:
            self._spec, self._bases, self._goal = spec, rq_closure_bases(spec), spec.goal
        seen = obs.present
        share = sum(seen[b] for b in self._bases) / len(self._bases) if self._bases else 0.0
        return self.base_weight * share + self.goal_weight * float(seen[self._goal])


class TTPotential:
    """``tt_levels``: the public levels of the concepts held, and the goal."""

    def __init__(self, concept_weight: float, goal_weight: float):
        self.concept_weight, self.goal_weight = concept_weight, goal_weight
        self._spec = None

    def __call__(self, spec: TechTreeSpec, obs: TechObservation) -> float:
        if spec is not self._spec:
            self._spec = spec
            self._levels = np.asarray(spec.concept_levels, dtype=float)
        holds = np.asarray(obs.held, dtype=bool)
        share = float(self._levels[holds].sum() / self._levels.sum()) if self._levels.sum() else 0.0
        return self.concept_weight * share + self.goal_weight * float(obs.held[spec.goal])


class Shaper:
    """One environment's shaping state: ``reset`` at the start of a learner episode, ``step`` after
    every transition (``end``: the learner episode ended, so the next potential is 0)."""

    def __init__(self, potential, gamma: float):
        self.potential, self.gamma = potential, float(gamma)
        self.phi = 0.0

    def reset(self, spec, obs) -> float:
        self.phi = self.potential(spec, obs)
        return self.phi

    def step(self, spec, obs, end: bool) -> float:
        nxt = 0.0 if end else self.potential(spec, obs)
        f = self.gamma * nxt - self.phi
        self.phi = nxt
        return f


def make_shaper(shaping: dict, gamma: float) -> Shaper:
    s = normalize(shaping)
    weights = {k: v for k, v in s.items() if k != "potential"}
    cls = RQPotential if s["potential"] == "rq_goal_closure" else TTPotential
    return Shaper(cls(**weights), gamma)
