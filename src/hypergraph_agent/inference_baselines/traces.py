"""Episode logs in the representations the inference methods need (public data only).

The evidence of study X2 is the walker's ``EpisodeLog``: the actions of an
episode, the observed goal after every step, the observed effect of every
non-goal craft (``None`` when intermediate items are unobserved) and whether
each gathered or activated base fact was observed present afterwards. With
items observable that record determines the observed state before and after
every step exactly (facts are persistent and every fact that can change is
either a base fact, always observed, or a craft effect, observed by the craft
itself), so it converts without loss into:

``observed_states``  one ``StateStep`` per primitive step (state before, action,
                     state after), the input of action-model learners;
``craft_examples``   one ``CraftExample`` per informative attempt of a hidden
                     recipe (its effect was absent before): the facts present
                     before and whether the effect appeared, the input of the
                     factorized updater and of MSGI's precondition inference;
``successful_steps`` the steps that changed the state, as executed actions of a
                     STRIPS trace (what macq's learners and SAM accept: they
                     have no representation of a failed attempt).

With goal-only observation the outcome of crafting an intermediate item is not
observed, so these representations do not exist; ``observed_states`` raises
``ItemsUnobserved``. Only the joint posterior and the pairwise weights work on
such logs (they read ``EpisodeLog`` directly).

Nothing here reads hidden requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..envs.public_schema import ACTIVATE, CRAFT, GATHER


class ItemsUnobserved(ValueError):
    """A craft of an intermediate item has no observed outcome (goal-only logs)."""


class InconsistentLog(ValueError):
    """The logged observations contradict persistent, deterministic facts."""


@dataclass(frozen=True)
class StateStep:
    t: int
    kind: int
    target: object  # base type id, recipe signature, or None (wait, submit)
    before: frozenset  # observed facts (type ids) before the step
    after: frozenset  # observed facts after the step

    @property
    def changed(self) -> bool:
        return self.after != self.before


@dataclass(frozen=True)
class CraftExample:
    log_id: int
    t: int
    signature: str
    present: frozenset  # observed facts before the attempt
    success: bool  # the previously absent effect appeared


def items_observable(log, infos: dict) -> bool:
    """Every craft of a non-goal item has an observed outcome."""
    return all(seen is not None for (kind, x), seen in zip(log.steps, log.effect_obs)
               if kind == CRAFT and infos[x].effect != log.goal)


def observed_states(log, infos: dict) -> list[StateStep]:
    """The observed state sequence of an episode logged with items observable."""
    state = set(log.initial)
    out = []
    for t, ((kind, x), goal_seen, seen) in enumerate(zip(log.steps, log.goal_obs, log.effect_obs)):
        before = frozenset(state)
        if kind in (GATHER, ACTIVATE):
            if log.base_present(t):
                state.add(x)
        elif kind == CRAFT:
            effect = infos[x].effect
            if effect == log.goal:
                outcome = bool(goal_seen)
            elif seen is None:
                raise ItemsUnobserved(f"log {log.log_id}, step {t}: craft outcome not observed")
            else:
                outcome = bool(seen)
            if effect in state and not outcome:
                raise InconsistentLog(f"log {log.log_id}, step {t}: a held fact disappeared")
            if outcome:
                state.add(effect)
        if bool(goal_seen) != (log.goal in state):
            raise InconsistentLog(f"log {log.log_id}, step {t}: goal observation does not follow")
        out.append(StateStep(t, kind, x, before, frozenset(state)))
    return out


def craft_examples(logs, infos: dict) -> list[CraftExample]:
    """Informative attempts of hidden recipes, in log and step order."""
    out = []
    for log in logs:
        for s in observed_states(log, infos):
            if s.kind != CRAFT or infos[s.target].known is not None:
                continue
            effect = infos[s.target].effect
            if effect in s.before:
                continue  # repeated acquisition: eligibility not identified
            out.append(CraftExample(log.log_id, s.t, s.target, s.before, effect in s.after))
    return out


def successful_steps(log, infos: dict) -> list[StateStep]:
    """Steps that changed the observed state (executed actions of a STRIPS
    trace); failed attempts, repeats, waits and submissions are dropped, which
    keeps the state sequence continuous because they changed nothing."""
    return [s for s in observed_states(log, infos) if s.changed]
