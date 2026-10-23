"""Gymnasium wrappers with fixed spaces for off-the-shelf learners (study X).

``RecipeQuestGym`` and ``TechTreeGym`` play the tasks of a ``sources``
block source through the repository's environments and expose a fixed
``Box`` observation (``encoders``) and a fixed ``Discrete`` action space with
an action mask (``action_masks()``, the interface of sb3-contrib's
``MaskablePPO``).

Information boundary. The wrapper sits on the evaluator side only in that it
holds task items: it passes ``TaskItem.task`` to the environment's reset and
nothing else of it reaches the learner. Observations are built from the
public spec, the public observation, the public ``info`` of a step and the
agent's own actions; the ``info`` returned to the learner carries only public
outcome data. TechTree is played through ``env.agent_interface`` (opaque task
tokens), RecipeQuest through ``RecipeQuestEnv``'s public spec and
observations.

Episodes. With ``trial=False`` one learner episode is one task; consecutive
learner episodes still walk through each world's episodes in order. With
``trial=True`` one learner episode is a whole world block (RL^2-style
trials): when a task ends the next task of the same world starts within the
same learner episode, so a recurrent policy can carry what it learned about
the world from one task to the next; the learner episode ends after the last
task of the block. The episode context (episode index within the world, first
step of a task, success of the previous task of the world) is part of every
observation.

Invalid actions. A masked action (a fact or recipe absent from the task, an
announced slot that is not yet active) chosen by a learner that does not use
masks executes ``wait`` instead: it costs one primitive step of the task budget
and one ledger interaction, as any no-op does, and ``info["invalid"]`` is set.
No other penalty is added.

Reward. The environment's reward (1 on successful submission, else 0) minus
``step_penalty`` per step (default 0: no shaping). With ``shaping`` (a
``shaping.Shaper``, study XS; default None) the potential-based term
``gamma * Phi(s') - Phi(s)`` is added and reported as ``info["shaping"]``; the
potential reads only the public spec and observation, and it is 0 when the
learner episode ends.

Ledger. Every primitive step is charged to ``meter`` (``BudgetMeter``) before it
is executed; a run cannot exceed its cap. ``on_task_end(item, result)`` is the
evaluator's hook to record a row when a task ends (results are public outcome
counts).
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..envs.recipequest import RecipeQuestEnv
from ..techtree.env import TechTreeEnv, agent_interface
from .encoders import EpisodeContext, RQLayout, TTLayout


class SourceExhausted(RuntimeError):
    """A finite (evaluation) source has no further world block."""


class _BlockGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, source, *, trial: bool = False, step_penalty: float = 0.0, meter=None,
                 purpose: str = "exploration", on_task_end=None, shaping=None):
        self.source, self.trial, self.step_penalty = source, trial, float(step_penalty)
        self.meter, self.purpose, self.on_task_end = meter, purpose, on_task_end
        self.shaping = shaping
        self._queue: deque = deque()
        self._item = None
        self._prev_success = False
        self.interactions = 0  # primitive steps executed by this wrapper
        self.tasks_done = 0

    # ------------------------------------------------------------ blocks
    def _next_item(self, new_block: bool):
        if new_block or not self._queue:
            block = self.source.next_block()
            if block is None:
                raise SourceExhausted("the task source has no further world block")
            self._queue = deque(block)
            self._prev_success = False
        self._item = self._queue.popleft()
        self._task_steps = self._invalid = 0
        return self._item

    def _context(self, first: bool) -> EpisodeContext:
        return EpisodeContext(self._item.world_episode, self.source.E, first, self._prev_success)

    def _charge(self):
        if self.meter is not None:
            self.meter.charge(1, self.purpose)
        self.interactions += 1
        self._task_steps += 1

    def _end_task(self, success: bool, status: str | None):
        self.tasks_done += 1
        self._prev_success = success
        if self.on_task_end is not None:
            self.on_task_end(self._item, {"success": bool(success), "status": status,
                                          "primitive_length": self._task_steps,
                                          "invalid_actions": self._invalid})

    # --------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._start(self._next_item(new_block=self.trial))
        if self.shaping is not None:
            self.shaping.reset(self.spec, self._obs)
        return self._observe(first=True), {}

    def _shaped(self, reward: float, info: dict, end: bool) -> float:
        """The reward with the shaping term (study XS); unchanged without shaping."""
        if self.shaping is None:
            return reward
        info["shaping"] = self.shaping.step(self.spec, self._obs, end)
        return reward + info["shaping"]

    def step(self, action):
        reward, terminated, info = self._act(int(action))
        reward -= self.step_penalty
        if terminated:
            self._end_task(info["termination"] == "success", info["termination"])
            info = {**info, "task_end": True, "task_success": info["termination"] == "success"}
            if self.trial and self._queue:
                self._start(self._next_item(new_block=False))
                return self._observe(first=True), self._shaped(reward, info, False), False, False, info
            return self._observe(first=False), self._shaped(reward, info, True), True, False, info
        info = {**info, "task_end": False}
        return self._observe(first=False), self._shaped(reward, info, False), False, False, info


class RecipeQuestGym(_BlockGym):
    def __init__(self, source, layout: RQLayout | None = None, **kw):
        super().__init__(source, **kw)
        self.layout = layout or RQLayout()
        self.env = RecipeQuestEnv()
        self.observation_space = spaces.Box(0.0, 1.0, (self.layout.dim,), np.float32)
        self.action_space = spaces.Discrete(self.layout.n_actions)

    def _start(self, item):
        self._obs, _ = self.env.reset(seed=item.dyn_seed, options={"task": item.task})
        self.spec = self.env.public_spec
        self._map = self.layout.action_map(self.spec)
        self._static = self.layout.static(self.spec)
        self._counts = np.zeros(self.layout.n_actions, dtype=np.float32)
        self._last = None

    def action_masks(self) -> np.ndarray:
        return self._map >= 0

    def _act(self, canon: int):
        invalid = not 0 <= canon < self.layout.n_actions or self._map[canon] < 0
        if invalid:
            canon = self.layout.wait
            self._invalid += 1
        self._charge()
        self._obs, reward, terminated, _, info = self.env.step(int(self._map[canon]))
        self._counts[canon] += 1
        self._last = canon
        return reward, terminated, {"termination": info["termination"], "changed": info["changed"],
                                    "invalid": invalid}

    def _observe(self, first: bool) -> np.ndarray:
        dyn = self.layout.dynamic(self.spec, self._obs, self._last, self._counts, self._context(first))
        return np.concatenate([self._static, dyn])


class TechTreeGym(_BlockGym):
    def __init__(self, source, layout: TTLayout, **kw):
        super().__init__(source, **kw)
        self.layout = layout
        self.penv, self._token_for = agent_interface(TechTreeEnv())
        self.observation_space = spaces.Box(0.0, 1.0, (layout.dim,), np.float32)
        self.action_space = spaces.Discrete(layout.n_actions)

    def _start(self, item):
        self._obs, _ = self.penv.reset(self._token_for(item.task))
        self.spec = self.penv.public_spec
        self.layout.check(self.spec)
        self._static = self.layout.static(self.spec)
        self._counts = np.zeros(self.layout.n_actions, dtype=np.float32)
        self._trace: list[int] = []
        self._last, self._fired = None, ()

    def action_masks(self) -> np.ndarray:
        return np.asarray(self._obs.available, dtype=bool)

    def _act(self, a: int):
        invalid = not 0 <= a < self.layout.n_actions or not self._obs.available[a]
        if invalid:
            a = self.spec.wait_action
            self._invalid += 1
        self._charge()
        self._obs, reward, terminated, _, info = self.penv.step(a)
        if a < self.spec.n_symbols:
            self._trace = (self._trace + [a])[-self.layout.max_len:]
        else:
            self._trace = []
        self._counts[a] += 1
        self._last, self._fired = a, tuple(info["fired"])
        return reward, terminated, {"termination": info["termination"], "fired": self._fired,
                                    "invalid": invalid}

    def _observe(self, first: bool) -> np.ndarray:
        dyn = self.layout.dynamic(self.spec, self._obs, self._trace, self._last, self._counts, self._fired,
                                  self._context(first))
        return np.concatenate([self._static, dyn])


def make_gym(family: str, source, layout=None, **kw) -> _BlockGym:
    if family == "recipequest":
        return RecipeQuestGym(source, layout, **kw)
    if family == "techtree":
        return TechTreeGym(source, layout, **kw)
    raise ValueError(f"unknown family {family!r}")
