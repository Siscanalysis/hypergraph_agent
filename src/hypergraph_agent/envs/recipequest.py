"""RecipeQuest Gymnasium environment.

Semantics
---------
* Facts are Boolean, persistent and non-consumptive.
* Every primitive action is exactly one transition and consumes one unit of the
  task budget, whether or not it changes anything.
* ``gather``/``activate`` make a base fact true; ``craft`` makes its effect true
  when every true prerequisite of that recipe holds; ``submit`` succeeds when the
  goal fact holds; ``wait`` does nothing.
* With ``failure_prob > 0`` an eligible state-changing action fails with that
  probability and leaves the state unchanged (independent draws from the
  dynamics RNG, seeded through ``reset(seed=...)``).
* Reward is exactly 1 on successful submission, else 0. Exhausting the budget
  is an unsuccessful terminal (``termination == "deadline"``). The environment
  never truncates by itself; truncation is reserved for external interruption.

Observations are ``PublicObservation`` objects. ``info`` carries only public
outcome data.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

from .generator import Task
from .public_schema import (
    ACTIVATE, CRAFT, GATHER, KIND_NAMES, SUBMIT,
    PublicObservation, PublicTaskSpec, public_view,
)
from .vocabulary import TYPE_NAMES


class PublicObservationSpace(spaces.Space):
    """Variable-size structured observation; membership check only."""

    def __init__(self):
        super().__init__(shape=None, dtype=None)

    def contains(self, x) -> bool:
        return isinstance(x, PublicObservation)

    def sample(self, mask=None, probability=None):
        raise NotImplementedError("observations are produced by the environment")

    @property
    def is_np_flattenable(self) -> bool:
        return False


class RecipeQuestEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode
        self.observation_space = PublicObservationSpace()
        self.action_space = spaces.Discrete(1)
        self._task: Task | None = None
        self.public_spec: PublicTaskSpec | None = None
        self.total_steps = 0  # every primitive transition this instance executed
        self._done = True

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not options or "task" not in options:
            raise ValueError("reset requires options={'task': Task}")
        task: Task = options["task"]
        self._task = task
        self.public_spec = public_view(task)
        self.action_space = spaces.Discrete(len(self.public_spec.actions))
        self._present = {ft: ft in task.initial_true for ft in task.fact_types}
        self._budget_left = task.budget
        self._t = 0
        self._done = False
        self._last: tuple[int | None, bool | None] = (None, None)
        return self._obs(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step() after a terminal transition; call reset()")
        a = int(action)
        actions = self.public_spec.actions
        if not 0 <= a < len(actions):
            raise ValueError(f"action index {a} is not a syntactic action of this task")
        desc = actions[a]
        task = self._task
        self.total_steps += 1
        self._t += 1
        self._budget_left -= 1
        changed = False
        reward = 0.0
        terminated = False
        reason = None

        if desc.kind in (GATHER, ACTIVATE):
            ft = task.fact_types[desc.fact]
            if not self._present[ft] and self._eligible_succeeds():
                self._present[ft] = True
                changed = True
        elif desc.kind == CRAFT:
            recipe = task.rules[desc.rule]
            if (not self._present[recipe.effect]
                    and all(self._present[x] for x in recipe.required())
                    and self._eligible_succeeds()):
                self._present[recipe.effect] = True
                changed = True
        elif desc.kind == SUBMIT:
            if self._present[task.goal]:
                reward = 1.0
                terminated = True
                reason = "success"

        if not terminated and self._budget_left <= 0:
            terminated = True
            reason = "deadline"
        self._done = terminated
        self._last = (a, changed)
        return self._obs(), reward, terminated, False, {"termination": reason, "changed": changed}

    def render(self):
        if self.render_mode == "ansi":
            return render_public(self.public_spec, self._obs())
        return None

    # ------------------------------------------------------------ internals
    def _eligible_succeeds(self) -> bool:
        p = self._task.failure_prob
        return p <= 0.0 or self.np_random.random() >= p

    def _obs(self) -> PublicObservation:
        task = self._task
        present = tuple(self._present[ft] for ft in task.fact_types)
        return PublicObservation(present, self._budget_left, self._t, *self._last)

    def debug_view(self) -> str:
        """Evaluator-only rendering that shows hidden prerequisites."""
        task = self._task
        lines = [f"[EVALUATOR DEBUG VIEW - not an agent input] task {task.task_key}"]
        for recipe in task.rules:
            req = ",".join(TYPE_NAMES[x] for x in sorted(recipe.required()))
            lines.append(f"  {TYPE_NAMES[recipe.effect]} <= {{{req}}}")
        return "\n".join(lines)


def fact_name(spec: PublicTaskSpec, i: int) -> str:
    return TYPE_NAMES[spec.facts[i].type_id]


def action_name(spec: PublicTaskSpec, a: int) -> str:
    d = spec.actions[a]
    if d.fact is not None:
        return f"{KIND_NAMES[d.kind]} {fact_name(spec, d.fact)}"
    if d.rule is not None:
        return f"{KIND_NAMES[d.kind]} {fact_name(spec, spec.rules[d.rule].effect)}"
    return KIND_NAMES[d.kind]


def render_public(spec: PublicTaskSpec, obs: PublicObservation) -> str:
    lines = [
        f"task {spec.task_key}  world {spec.world_key}  profile {spec.profile}",
        f"goal: {fact_name(spec, spec.goal)}   step {obs.t}   budget left {obs.budget_left}",
    ]
    held = [fact_name(spec, i) for i, p in enumerate(obs.present) if p]
    lines.append("held: " + (", ".join(held) if held else "(nothing)"))
    lines.append("recipes:")
    for r in spec.rules:
        items = [fact_name(spec, i) for i in r.item_inputs]
        if r.known_base is not None:
            base = "{" + ",".join(fact_name(spec, i) for i in r.known_base) + "}"
        else:
            base = "some of {" + ",".join(fact_name(spec, i) for i in r.pool) + "}"
        need = " + ".join(x for x in (", ".join(items), base) if x)
        lines.append(f"  {fact_name(spec, r.effect):>6} <= {need}")
    if obs.last_action is not None:
        outcome = "changed" if obs.last_changed else "no change"
        lines.append(f"last: {action_name(spec, obs.last_action)} -> {outcome}")
    return "\n".join(lines)
