"""Public task descriptors: the only task information learning code receives.

``public_view`` is the information boundary. In the ``unknown_prerequisites``
profile it never reads ``Recipe.true_base``; a canary test checks that two tasks
differing only in hidden prerequisites produce identical public views.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import PROFILE_KNOWN, Task
from .vocabulary import KIND_FACILITY, KIND_RESOURCE, is_base, kind_of

GATHER = 0
ACTIVATE = 1
CRAFT = 2
WAIT = 3
SUBMIT = 4
SKILL = 5  # agent-side choice kind; never an environment action
NUM_CHOICE_KINDS = 6
KIND_NAMES = ("gather", "activate", "craft", "wait", "submit", "skill")


@dataclass(frozen=True)
class PublicFact:
    key: str  # opaque, task-local
    type_id: int
    kind: int
    is_goal: bool


@dataclass(frozen=True)
class PublicRule:
    key: str  # opaque, task-local
    signature: str  # type-level public schema, stable across tasks of a world
    effect: int  # fact index
    item_inputs: tuple[int, ...]  # fact indices, always public
    known_base: tuple[int, ...] | None  # fact indices; None when hidden
    pool: tuple[int, ...]  # candidate base fact indices; empty when structure is known


@dataclass(frozen=True)
class ActionDescriptor:
    key: str
    kind: int
    fact: int | None = None
    rule: int | None = None


@dataclass(frozen=True)
class PublicTaskSpec:
    task_key: str
    world_key: str
    profile: str
    facts: tuple[PublicFact, ...]
    rules: tuple[PublicRule, ...]
    actions: tuple[ActionDescriptor, ...]
    goal: int  # fact index
    budget: int
    failure_prob: float  # declared noise of the known-noise profile
    items_observable: bool = True  # False: only base facts and the goal are observed

    def fact_by_type(self) -> dict[int, int]:
        return {f.type_id: i for i, f in enumerate(self.facts)}

    def action_index(self) -> dict[str, int]:
        return {a.key: i for i, a in enumerate(self.actions)}


@dataclass(frozen=True)
class PublicObservation:
    present: tuple[bool, ...]
    budget_left: int
    t: int
    last_action: int | None
    last_changed: bool | None


def build_actions(task: Task) -> tuple[ActionDescriptor, ...]:
    actions = []
    for i, (ft, key) in enumerate(zip(task.fact_types, task.fact_keys)):
        k = kind_of(ft)
        if k == KIND_RESOURCE:
            actions.append(ActionDescriptor(f"gather:{key}", GATHER, fact=i))
        elif k == KIND_FACILITY:
            actions.append(ActionDescriptor(f"activate:{key}", ACTIVATE, fact=i))
    for j, key in enumerate(task.rule_keys):
        actions.append(ActionDescriptor(f"craft:{key}", CRAFT, rule=j))
    actions.append(ActionDescriptor("wait", WAIT))
    actions.append(ActionDescriptor("submit", SUBMIT))
    return tuple(actions)


def public_view(task: Task) -> PublicTaskSpec:
    index = {ft: i for i, ft in enumerate(task.fact_types)}
    facts = tuple(
        PublicFact(key, ft, kind_of(ft), ft == task.goal)
        for ft, key in zip(task.fact_types, task.fact_keys)
    )
    rules = []
    for recipe, key in zip(task.rules, task.rule_keys):
        items = tuple(index[i] for i in recipe.item_inputs)
        if task.profile == PROFILE_KNOWN:
            known = tuple(index[b] for b in recipe.true_base)
            pool: tuple[int, ...] = ()
        else:
            known = None
            pool = tuple(index[b] for b in recipe.pool)
        rules.append(PublicRule(
            key=key,
            signature=recipe.signature(task.profile),
            effect=index[recipe.effect],
            item_inputs=items,
            known_base=known,
            pool=pool,
        ))
    return PublicTaskSpec(
        task_key=task.task_key,
        world_key=task.world.world_key,
        profile=task.profile,
        facts=facts,
        rules=tuple(rules),
        actions=build_actions(task),
        goal=index[task.goal],
        budget=task.budget,
        failure_prob=task.failure_prob,
        items_observable=task.observe_items == "all",
    )


def observable(task: Task, type_id: int) -> bool:
    return task.observe_items == "all" or is_base(type_id) or type_id == task.goal


def initial_observation(task: Task) -> PublicObservation:
    present = tuple(ft in task.initial_true and observable(task, ft) for ft in task.fact_types)
    return PublicObservation(present, task.budget, 0, None, None)
