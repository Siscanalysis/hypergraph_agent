"""TechTree Gymnasium environment and its public view.

Semantics
---------
* Actions: the primitives (``n_primitives`` base symbols, then ``n_slots``
  extra slots), ``wait`` and ``submit``. Every action is one transition and
  costs one unit of the episode budget.
* The primitive trace is the sequence of primitives pressed since the start of
  the episode or the last ``wait``/``submit`` (both clear it).
* Firing: after every primitive press, repeatedly (a fixpoint within the step)
  any concept that is not held, whose parents are all held and whose sequence
  equals the current trace suffix becomes held. Held concepts persist until
  the end of the episode; every episode starts with nothing held.
* Unlocks: when a key concept becomes held, its slot becomes active for the
  rest of the episode. With ``unlock_visibility == "announced"`` a slot is in
  the available-action mask only while active (pressing an unavailable action
  is an error). With ``"silent"`` every slot is listed and available from the
  start as a generic ``try`` action; pressing an inactive one does nothing
  except break the trace (it matches no sequence), and the agent is never told
  which slot became active.
* Intermediate signal (``signal``): the observation's ``progress`` flag is true
  when the trace suffix is a nonempty proper prefix of the sequence of some
  concept that is not held and whose parents are held; ``None`` when the signal
  is off.
* ``submit`` succeeds (reward 1, terminal) when the goal is held, otherwise it
  only clears the trace. Exhausting the budget is an unsuccessful terminal
  (``deadline``).

Public information (``TechTreeSpec``): the alphabet and action list, the
concept keys and levels, the level lengths and parent levels (declared rules
of the family), the goal, the budget, the visibility mode and whether the
signal is on. Hidden: sequences, parents, which concepts are compositional,
which concepts unlock which slot. ``info`` carries only public outcome data.

Agents hold only a public facade: ``agent_interface`` returns a ``PublicEnv``
(reset with an opaque ``TaskToken``, step, public spec; results are
observations and public info) whose environment and task registry live in
closures. Ordinary attribute access cannot reach private state; introspection
(through a closure or a stack frame, for example) is not prevented here, and is
excluded by a source check of the agent modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from .generator import TechTask


@dataclass(frozen=True)
class TechTreeSpec:
    task_key: str
    world_key: str
    n_base: int
    n_slots: int
    unlock_visibility: str
    signal: bool
    concept_keys: tuple[str, ...]
    concept_levels: tuple[int, ...]
    level_lengths: tuple[int, ...]  # index l-1: sequence length of level l
    parent_levels: tuple[tuple[int, ...], ...]  # index l-1: levels of the two parents
    goal: int
    budget: int
    action_keys: tuple[str, ...]

    @property
    def n_symbols(self) -> int:
        return self.n_base + self.n_slots

    @property
    def wait_action(self) -> int:
        return self.n_symbols

    @property
    def submit_action(self) -> int:
        return self.n_symbols + 1

    def length(self, concept: int) -> int:
        return self.level_lengths[self.concept_levels[concept] - 1]

    def concepts_at(self, level: int) -> list[int]:
        return [c for c, lv in enumerate(self.concept_levels) if lv == level]


@dataclass(frozen=True)
class TechObservation:
    held: tuple[bool, ...]
    available: tuple[bool, ...]  # per action
    budget_left: int
    t: int
    last_action: int | None
    progress: bool | None  # None when the intermediate signal is off


def public_view(task: TechTask) -> TechTreeSpec:
    w = task.world
    cfg = w.config
    slot_name = "try" if task.unlock_visibility == "silent" else "x"
    actions = tuple(f"p{i}" for i in range(cfg.n_primitives)) \
        + tuple(f"{slot_name}{j}" for j in range(cfg.n_slots)) + ("wait", "submit")
    return TechTreeSpec(
        task_key=task.task_key, world_key=w.world_key, n_base=cfg.n_primitives, n_slots=cfg.n_slots,
        unlock_visibility=task.unlock_visibility, signal=task.signal, concept_keys=w.keys,
        concept_levels=w.levels,
        level_lengths=tuple(cfg.length(lv) for lv in range(1, cfg.n_levels + 1)),
        parent_levels=tuple(cfg.parent_levels(lv) for lv in range(1, cfg.n_levels + 1)),
        goal=task.goal, budget=task.budget, action_keys=actions)


class _ObservationSpace(spaces.Space):
    def __init__(self):
        super().__init__(shape=None, dtype=None)

    def contains(self, x) -> bool:
        return isinstance(x, TechObservation)

    @property
    def is_np_flattenable(self) -> bool:
        return False


class TechTreeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode
        self.observation_space = _ObservationSpace()
        self.action_space = spaces.Discrete(1)
        self.public_spec: TechTreeSpec | None = None
        self.total_steps = 0
        self._done = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not options or "task" not in options:
            raise ValueError("reset requires options={'task': TechTask}")
        task: TechTask = options["task"]
        w = task.world
        self._task, self._world = task, w
        self.public_spec = spec = public_view(task)
        self.action_space = spaces.Discrete(spec.n_symbols + 2)
        self._P, self._S = spec.n_base, spec.n_symbols
        self._held = [False] * len(w.levels)
        self._active = [s < self._P for s in range(self._S)]
        self._slot_of_key = {key: slot for key, slot in w.unlocks}
        self._by_len: dict[int, dict[tuple, int]] = {}
        self._prefix: dict[tuple, list[int]] = {}
        for c, s in enumerate(w.sequences):
            self._by_len.setdefault(len(s), {})[s] = c
            for j in range(1, len(s)):
                self._prefix.setdefault(s[:j], []).append(c)
        self._maxlen = max(len(s) for s in w.sequences)
        self._trace: list[int | None] = []
        self._budget_left, self._t, self._last = task.budget, 0, None
        self._done = False
        return self._obs(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step() after a terminal transition; call reset()")
        a = int(action)
        spec = self.public_spec
        if not 0 <= a < self._S + 2:
            raise ValueError(f"action {a} is not an action of this task")
        if a < self._S and spec.unlock_visibility == "announced" and not self._active[a]:
            raise ValueError(f"action {spec.action_keys[a]} is not available")
        self.total_steps += 1
        self._t += 1
        self._budget_left -= 1
        reward, terminated, reason, fired = 0.0, False, None, []
        if a < self._S:
            self._trace.append(a if self._active[a] else None)
            del self._trace[: -self._maxlen]
            fired = self._fire()
        else:
            self._trace.clear()
            if a == self._S + 1 and self._held[self._task.goal]:
                reward, terminated, reason = 1.0, True, "success"
        if not terminated and self._budget_left <= 0:
            terminated, reason = True, "deadline"
        self._done = terminated
        self._last = a
        return self._obs(), reward, terminated, False, {"termination": reason, "fired": tuple(fired)}

    def _gated(self, c: int) -> bool:
        return all(self._held[p] for p in self._world.parents[c])

    def _fire(self) -> list[int]:
        fired, changed = [], True
        tr = self._trace
        while changed:
            changed = False
            for n, table in self._by_len.items():
                if len(tr) >= n:
                    c = table.get(tuple(tr[-n:]))
                    if c is not None and not self._held[c] and self._gated(c):
                        self._held[c] = True
                        fired.append(c)
                        changed = True
        for c in fired:
            if c in self._slot_of_key:
                self._active[self._slot_of_key[c]] = True
        return fired

    def _progress(self) -> bool | None:
        if not self._task.signal:
            return None
        tr = self._trace
        for j in range(1, min(len(tr), self._maxlen - 1) + 1):
            for c in self._prefix.get(tuple(tr[-j:]), ()):
                if not self._held[c] and self._gated(c):
                    return True
        return False

    def _obs(self) -> TechObservation:
        silent = self.public_spec.unlock_visibility == "silent"
        avail = tuple(s < self._P or silent or self._active[s] for s in range(self._S)) + (True, True)
        return TechObservation(tuple(self._held), avail, self._budget_left, self._t, self._last,
                               self._progress())

    def render(self):
        if self.render_mode != "ansi":
            return None
        spec, obs = self.public_spec, self._obs()
        held = [spec.concept_keys[c] for c, h in enumerate(obs.held) if h]
        return (f"task {spec.task_key} world {spec.world_key} goal {spec.concept_keys[spec.goal]} "
                f"step {obs.t} budget left {obs.budget_left}\nheld: {', '.join(held) or '(nothing)'}")


class TaskToken:
    """Opaque handle of a task: it carries nothing an agent can read."""

    __slots__ = ()


class PublicEnv:
    """All an agent can touch: reset with a task token, step, the public spec.
    The environment and the tasks live in closures made by ``agent_interface``."""

    __slots__ = ("_reset", "_step", "_spec")

    def __init__(self, reset, step, spec):
        object.__setattr__(self, "_reset", reset)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_spec", spec)

    def __setattr__(self, name, value):
        raise AttributeError("PublicEnv is read-only")

    def reset(self, token: TaskToken):
        return self._reset(token)

    def step(self, action: int):
        return self._step(action)

    @property
    def public_spec(self) -> TechTreeSpec:
        return self._spec()


def agent_interface(env: TechTreeEnv):
    """(``PublicEnv`` for agents, function task -> ``TaskToken`` for the runner)."""
    registry: dict[TaskToken, TechTask] = {}

    def token_for(task: TechTask) -> TaskToken:
        token = TaskToken()
        registry[token] = task
        return token

    def reset(token: TaskToken):
        return env.reset(options={"task": registry.pop(token)})

    return PublicEnv(reset, lambda action: env.step(action), lambda: env.public_spec), token_for
