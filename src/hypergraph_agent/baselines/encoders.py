"""Fixed-size public encodings of RecipeQuest and TechTree (study X).

Off-the-shelf learners need one observation vector and one action set for
every task. The per-task public specs have opaque keys and a variable action
list, so this module maps them onto fixed layouts indexed by the public type
vocabulary (RecipeQuest) or by the public concept and symbol indices
(TechTree).

Information boundary: every function here receives only public objects
(``PublicTaskSpec``, ``PublicObservation``, ``TechTreeSpec``,
``TechObservation``, the public ``info`` of a step) and the agent's own action
history. It never sees a task, a world or the reference solver; a test scans
this module for such names, and canary tests check that two tasks differing
only in hidden structure give identical vectors under identical public
histories.

RecipeQuest layout (``RQLayout``)
---------------------------------
Actions, canonical by type: gather each resource type (12), activate each
facility type (6), craft slot (item type, v) for v < ``max_variants`` (16 x 2),
wait, submit: 52 actions. The recipes of a task that produce the same item
type are put in slots by the order of their public signatures (item inputs and
candidate pool), so the slot of a recipe is public. A recipe that is in a task
as the only recipe of its effect (a distractor) takes slot 0. A canonical
action whose fact or recipe is absent from the task is masked.

Observation (all entries in [0, 1]):

- static per task: facts present in the task by type; goal one-hot; for every
  craft slot, whether it exists, its item inputs by item type and its public
  base set by base type (the candidate pool when prerequisites are hidden, the
  true base set in the known-structure profile, where it is public); flags for
  the known-structure profile and for observable items; the declared failure
  probability; the task budget / 64;
- dynamic: observed present facts by type (intermediate items stay absent
  when only the goal is observed, as the environment reports them);
  budget left and steps taken as fractions of the budget; the outcome of the
  last action (changed, unchanged, unobserved or none); the last canonical
  action (one-hot); how often each canonical action was taken in this episode
  (count / 3, clipped); the episode context supplied by the wrapper (episode
  index within the world / episodes per world, first step of a task, success
  of the previous task of the world).

TechTree layout (``TTLayout``)
------------------------------
Actions: the task's own action list (primitives, slots, wait, submit), fixed
within a setting; the mask is the public available-action mask.

Observation: goal one-hot; public level of every concept (one-hot); the
level lengths; flags for the signal and silent unlocks; concepts held; the
available-action mask; the progress flag (0 when the signal is off); budget
left and steps taken as fractions of the budget; the agent's own primitive
trace since the last ``wait`` or ``submit`` (the last ``max_len`` presses,
one-hot per position, with an "empty" symbol); the last action (one-hot);
presses per action in this episode (count / 10, clipped); concepts that fired
at the last step; the episode context as above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.public_schema import (
    ACTIVATE, CRAFT, GATHER, SUBMIT, WAIT, PublicObservation, PublicTaskSpec,
)
from ..envs.vocabulary import BASE_TYPES, FACILITIES, ITEM_TYPES, NUM_TYPES, RESOURCES
from ..techtree.env import TechObservation, TechTreeSpec

N_CONTEXT = 3  # episode index within the world, first step of a task, previous task solved


@dataclass(frozen=True)
class EpisodeContext:
    """Public context of the current task, supplied by the wrapper."""

    episode_in_world: int = 0
    episodes_per_world: int = 1
    first_step: bool = True
    previous_success: bool = False

    def vector(self) -> np.ndarray:
        return np.array([self.episode_in_world / max(1, self.episodes_per_world),
                         float(self.first_step), float(self.previous_success)], dtype=np.float32)


# ------------------------------------------------------------------ RecipeQuest
class RQLayout:
    """Canonical action set and observation layout of RecipeQuest."""

    def __init__(self, max_variants: int = 2):
        self.V = max_variants
        self.n_res, self.n_fac, self.n_items = len(RESOURCES), len(FACILITIES), len(ITEM_TYPES)
        self.craft0 = self.n_res + self.n_fac
        self.wait = self.craft0 + self.n_items * self.V
        self.submit = self.wait + 1
        self.n_actions = self.submit + 1
        self.n_slots = self.n_items * self.V
        self.slot_width = 1 + self.n_items + len(BASE_TYPES)
        self.static_dim = 2 * NUM_TYPES + self.n_slots * self.slot_width + 4
        self.dynamic_dim = NUM_TYPES + 2 + 4 + 2 * self.n_actions + N_CONTEXT
        self.dim = self.static_dim + self.dynamic_dim

    def craft_index(self, item_type: int, variant: int) -> int:
        return self.craft0 + (item_type - ITEM_TYPES[0]) * self.V + variant

    def base_index(self, base_type: int) -> int:
        """Gather (resources) and activate (facilities) actions are indexed by base type."""
        return base_type

    def slots(self, spec: PublicTaskSpec) -> list[int]:
        """Craft slot of every recipe of the task, by the order of public signatures."""
        by_effect: dict[int, list[int]] = {}
        for j, r in enumerate(spec.rules):
            by_effect.setdefault(spec.facts[r.effect].type_id, []).append(j)
        slot = [0] * len(spec.rules)
        for effect, js in by_effect.items():
            if len(js) > self.V:
                raise ValueError(f"{len(js)} recipes produce one item type; the layout has {self.V} slots")
            for v, j in enumerate(sorted(js, key=lambda j: spec.rules[j].signature)):
                slot[j] = v
        return slot

    def action_map(self, spec: PublicTaskSpec) -> np.ndarray:
        """Canonical index -> task action index, or -1 when absent from the task."""
        out = np.full(self.n_actions, -1, dtype=np.int64)
        slot = self.slots(spec)
        for a, d in enumerate(spec.actions):
            if d.kind in (GATHER, ACTIVATE):
                out[self.base_index(spec.facts[d.fact].type_id)] = a
            elif d.kind == CRAFT:
                eff = spec.facts[spec.rules[d.rule].effect].type_id
                out[self.craft_index(eff, slot[d.rule])] = a
            elif d.kind == WAIT:
                out[self.wait] = a
            elif d.kind == SUBMIT:
                out[self.submit] = a
        return out

    def static(self, spec: PublicTaskSpec) -> np.ndarray:
        x = np.zeros(self.static_dim, dtype=np.float32)
        types = [f.type_id for f in spec.facts]
        x[types] = 1.0
        x[NUM_TYPES + spec.facts[spec.goal].type_id] = 1.0
        off = 2 * NUM_TYPES
        slot = self.slots(spec)
        for j, r in enumerate(spec.rules):
            eff = spec.facts[r.effect].type_id
            s = off + ((eff - ITEM_TYPES[0]) * self.V + slot[j]) * self.slot_width
            x[s] = 1.0
            for i in r.item_inputs:
                x[s + 1 + spec.facts[i].type_id - ITEM_TYPES[0]] = 1.0
            base = r.known_base if r.known_base is not None else r.pool
            for b in base:
                x[s + 1 + self.n_items + spec.facts[b].type_id] = 1.0
        tail = off + self.n_slots * self.slot_width
        x[tail] = float(spec.profile == "known_structure")
        x[tail + 1] = float(spec.items_observable)
        x[tail + 2] = float(spec.failure_prob)
        x[tail + 3] = min(1.0, spec.budget / 64.0)
        return x

    def dynamic(self, spec: PublicTaskSpec, obs: PublicObservation, last_canon: int | None,
                counts: np.ndarray, ctx: EpisodeContext) -> np.ndarray:
        x = np.zeros(self.dynamic_dim, dtype=np.float32)
        for i, p in enumerate(obs.present):
            if p:
                x[spec.facts[i].type_id] = 1.0
        o = NUM_TYPES
        x[o] = obs.budget_left / spec.budget
        x[o + 1] = obs.t / spec.budget
        o += 2
        outcome = 3 if obs.last_action is None else (2 if obs.last_changed is None else
                                                     0 if obs.last_changed else 1)
        x[o + outcome] = 1.0
        o += 4
        if last_canon is not None:
            x[o + last_canon] = 1.0
        o += self.n_actions
        x[o:o + self.n_actions] = np.minimum(counts, 3) / 3.0
        o += self.n_actions
        x[o:o + N_CONTEXT] = ctx.vector()
        return x


# --------------------------------------------------------------------- TechTree
class TTLayout:
    """Observation layout of one TechTree setting (fixed concept count, levels, alphabet)."""

    def __init__(self, n_concepts: int, n_levels: int, n_symbols: int, max_len: int):
        self.n_c, self.n_levels, self.n_symbols, self.max_len = n_concepts, n_levels, n_symbols, max_len
        self.n_actions = n_symbols + 2
        self.static_dim = n_concepts + n_concepts * n_levels + n_levels + 2
        self.dynamic_dim = (n_concepts + self.n_actions + 1 + 2 + max_len * (n_symbols + 1)
                            + (self.n_actions + 1) + self.n_actions + n_concepts + N_CONTEXT)
        self.dim = self.static_dim + self.dynamic_dim

    @classmethod
    def from_spec(cls, spec: TechTreeSpec) -> "TTLayout":
        return cls(len(spec.concept_keys), len(spec.level_lengths), spec.n_symbols, max(spec.level_lengths))

    def check(self, spec: TechTreeSpec) -> None:
        if (len(spec.concept_keys), len(spec.level_lengths), spec.n_symbols, max(spec.level_lengths)) != \
                (self.n_c, self.n_levels, self.n_symbols, self.max_len):
            raise ValueError("the task does not match the setting's fixed TechTree layout")

    def static(self, spec: TechTreeSpec) -> np.ndarray:
        x = np.zeros(self.static_dim, dtype=np.float32)
        x[spec.goal] = 1.0
        o = self.n_c
        for c, lv in enumerate(spec.concept_levels):
            x[o + c * self.n_levels + lv - 1] = 1.0
        o += self.n_c * self.n_levels
        x[o:o + self.n_levels] = np.asarray(spec.level_lengths, dtype=np.float32) / self.max_len
        o += self.n_levels
        x[o] = float(spec.signal)
        x[o + 1] = float(spec.unlock_visibility == "silent")
        return x

    def dynamic(self, spec: TechTreeSpec, obs: TechObservation, trace: list[int], last_action: int | None,
                counts: np.ndarray, fired: tuple[int, ...], ctx: EpisodeContext) -> np.ndarray:
        x = np.zeros(self.dynamic_dim, dtype=np.float32)
        x[:self.n_c] = np.asarray(obs.held, dtype=np.float32)
        o = self.n_c
        x[o:o + self.n_actions] = np.asarray(obs.available, dtype=np.float32)
        o += self.n_actions
        x[o] = float(bool(obs.progress))
        o += 1
        x[o] = obs.budget_left / spec.budget
        x[o + 1] = obs.t / spec.budget
        o += 2
        width = self.n_symbols + 1
        recent = trace[-self.max_len:]
        pad = self.max_len - len(recent)
        for p in range(self.max_len):
            sym = recent[p - pad] if p >= pad else self.n_symbols
            x[o + p * width + sym] = 1.0
        o += self.max_len * width
        x[o + (self.n_actions if last_action is None else last_action)] = 1.0
        o += self.n_actions + 1
        x[o:o + self.n_actions] = np.minimum(counts, 10) / 10.0
        o += self.n_actions
        for c in fired:
            x[o + c] = 1.0
        o += self.n_c
        x[o:o + N_CONTEXT] = ctx.vector()
        return x
