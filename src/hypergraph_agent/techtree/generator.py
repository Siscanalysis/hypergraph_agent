"""TechTree worlds, task streams and the privileged reference solver.

Evaluator side of the information boundary: a ``TechWorld`` holds every
concept's secret primitive sequence, its parents, whether it is a composition
of earlier concepts and which concept unlocks which primitive. Agents receive
only ``env.public_view(task)`` and observations; agent modules never import
this one (a test scans them).

Alphabet
--------
``n_primitives`` base primitives are always available. ``n_slots`` extra
primitive slots exist in every world (a public range); ``n_unlocks`` of them
are activated, within an episode, when their key concept becomes held. How an
activation is shown is a property of the task (``unlock_visibility``, see
``env``).

Concepts, levels and gating
---------------------------
Concepts are arranged in public levels 1..D. The level determines the public
sequence length: ``combo_length * l`` under the ``linear`` rule, whose two
parents are at levels l-1 and 1, and ``combo_length * 2**(l-1)`` under
``doubling``, whose parents are both at level l-1. In both rules a concept's
length is the sum of its parents' lengths, so a composition keeps it. Level-1
concepts have distinct secret combos of base primitives. A concept of level
l >= 2 has two distinct hidden parents at those declared levels and a hidden
concatenation order; ordered parent pairs are distinct within a level. A
concept can fire only while both parents are held (gating).

Reuse depth
-----------
With ``reuse_depth = d`` a concept of level l is compositional when
2 <= l <= d + 1: s(x) = s(first parent) + s(second parent). Otherwise it is
flat: a uniform sequence of base primitives of the same length, redrawn until
it is (i) different from every other concept's sequence and (ii) not the
concatenation of the sequences of any two distinct concepts whose lengths add
up to its own. Rule (ii) means that "compose two earlier discoveries" never
yields a flat concept. Concept keys, parents, concatenation orders, level-1
combos and unlock choices are drawn from streams that do not depend on d, and
every flat sequence from its own per-concept stream, so worlds with the same
seed differ across d only in the sequences of their levels above 1.

Co-firing is allowed: typing one concept's sequence may also fire another
concept whose sequence is a suffix of the trace and whose parents are held.
Every firing still reveals exactly its own sequence (the trace suffix of its
public length), so discoveries are never ambiguous. A rule forbidding it was
not adopted: with two-symbol combos it would make every flat sequence avoid
ending in a level-1 combo while every compositional one ends in one, a larger
difference between world kinds than the effect it removes.

Unlocks
-------
For each unlock a key concept (level 1, base primitives only) and an entry
concept at ``unlock_level`` are chosen. The entry's sequence is drawn like a
flat one except that exactly one position, uniform, holds the unlocked slot;
an entry is never compositional. Compositional descendants of an entry inherit
the slot. The exact reference (``reference_solve``) searches the joint space
of held concepts and trace suffixes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from ..envs.generator import derive_seed, stable_hash

LENGTH_RULES = ("linear", "doubling")
VISIBILITIES = ("announced", "silent")
GOAL_RULES = ("uniform", "unlock")
MAX_WINDOW_SPACE = 1 << 20


@dataclass(frozen=True)
class TechWorldConfig:
    n_primitives: int = 3
    n_slots: int = 0
    n_unlocks: int = 0
    unlock_level: int = 1
    n_levels: int = 3
    concepts_per_level: tuple[int, ...] = (3, 3, 3)
    combo_length: int = 2
    length_rule: str = "linear"
    reuse_depth: int = 0

    def __post_init__(self):
        if self.length_rule not in LENGTH_RULES:
            raise ValueError(f"length_rule must be one of {LENGTH_RULES}")
        if len(self.concepts_per_level) != self.n_levels or min(self.concepts_per_level) < 1:
            raise ValueError("concepts_per_level needs one positive count per level")
        if not 0 <= self.n_unlocks <= self.n_slots:
            raise ValueError("need 0 <= n_unlocks <= n_slots")
        if self.n_unlocks and not 1 <= self.unlock_level <= self.n_levels:
            raise ValueError("unlock_level must be a level of the world")
        if not 0 <= self.reuse_depth < max(self.n_levels, 1):
            raise ValueError("reuse_depth must be in [0, n_levels - 1]")
        n1 = self.concepts_per_level[0]
        entries_at_1 = self.n_unlocks if self.unlock_level == 1 else 0
        if n1 - entries_at_1 > self.n_primitives ** self.combo_length:
            raise ValueError("more level-1 combos than base sequences of that length")
        if self.n_unlocks and n1 - entries_at_1 < self.n_unlocks:
            raise ValueError("every unlock needs its own level-1 key")
        if self.n_unlocks and self.concepts_per_level[self.unlock_level - 1] < self.n_unlocks:
            raise ValueError("not enough concepts at unlock_level for the entries")
        for lv in range(2, self.n_levels + 1):
            a, b = self.parent_levels(lv)
            na, nb = self.concepts_per_level[a - 1], self.concepts_per_level[b - 1]
            pairs = na * (na - 1) if a == b else 2 * na * nb
            if self.concepts_per_level[lv - 1] > pairs:
                raise ValueError(f"level {lv} has more concepts than distinct parent pairs")
        symbols = self.n_primitives + self.n_slots
        if symbols ** self.length(self.n_levels) > MAX_WINDOW_SPACE:
            raise ValueError("the deepest window space is too large for the elimination agents")

    def length(self, level: int) -> int:
        return self.combo_length * (level if self.length_rule == "linear" else 2 ** (level - 1))

    def parent_levels(self, level: int) -> tuple[int, ...]:
        if level == 1:
            return ()
        return (level - 1, 1 if self.length_rule == "linear" else level - 1)


@dataclass(frozen=True)
class TechWorld:
    world_key: str  # public opaque label, a memory scope for agents
    seed: int
    config: TechWorldConfig
    keys: tuple[str, ...]  # public, opaque, independent of the hidden structure
    levels: tuple[int, ...]  # public
    sequences: tuple[tuple[int, ...], ...]  # hidden
    parents: tuple[tuple[int, ...], ...]  # hidden, in concatenation order
    compositional: tuple[bool, ...]  # hidden
    unlocks: tuple[tuple[int, int], ...]  # hidden: (key concept, unlocked slot symbol)
    entries: tuple[int, ...]  # hidden: concepts drawn with an unlocked slot
    attempt: int = 0

    def at(self, level: int) -> list[int]:
        return [c for c, lv in enumerate(self.levels) if lv == level]

    def ancestors(self, c: int) -> set[int]:
        """``c`` and every concept reachable through parents."""
        out, stack = set(), [c]
        while stack:
            x = stack.pop()
            if x not in out:
                out.add(x)
                stack.extend(self.parents[x])
        return out

    def slot_keys(self) -> dict[int, int]:
        return {slot: key for key, slot in self.unlocks}

    def requirements(self, c: int) -> set[int]:
        """Concepts that must be held to fire ``c`` from scratch: its ancestors
        and, transitively, the keys (and their ancestors) of every unlocked slot
        their sequences use."""
        keys = self.slot_keys()
        out = self.ancestors(c)
        while True:
            extra = set()
            for x in out:
                for s in self.sequences[x]:
                    if s in keys and keys[s] not in out:
                        extra |= self.ancestors(keys[s])
            if not extra - out:
                return out
            out |= extra


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed("techtree", *parts))


def _digits(code: int, n: int, base: int) -> tuple[int, ...]:
    out = []
    for _ in range(n):
        code, r = divmod(code, base)
        out.append(r)
    return tuple(reversed(out))


def make_world(seed: int, cfg: TechWorldConfig, world_key: str | None = None) -> TechWorld:
    for attempt in range(200):
        w = _attempt(seed, cfg, world_key or "W" + stable_hash(("techtree", seed))[:8], attempt)
        if w is not None:
            return w
    raise RuntimeError("could not draw a valid world; the configuration is too tight")


def _attempt(seed: int, cfg: TechWorldConfig, key: str, attempt: int) -> TechWorld | None:
    P, k = cfg.n_primitives, cfg.combo_length
    levels = [lv for lv in range(1, cfg.n_levels + 1) for _ in range(cfg.concepts_per_level[lv - 1])]
    n = len(levels)
    at = {lv: [c for c in range(n) if levels[c] == lv] for lv in range(1, cfg.n_levels + 1)}
    codes = _rng(seed, attempt, "keys").choice(16 ** 6, size=n, replace=False)
    keys = tuple(f"c{int(c):06x}" for c in codes)

    parents: list[tuple[int, ...]] = [()] * n
    rp = _rng(seed, attempt, "parents")
    for lv in range(2, cfg.n_levels + 1):
        la, lb = cfg.parent_levels(lv)
        pairs = sorted({(a, b) for a in at[la] for b in at[lb] if a != b}
                       | {(b, a) for a in at[la] for b in at[lb] if a != b})
        for x, j in zip(at[lv], rp.choice(len(pairs), size=len(at[lv]), replace=False)):
            parents[x] = pairs[int(j)]

    unlocks, entries = [], []
    if cfg.n_unlocks:
        ru = _rng(seed, attempt, "unlock")
        slots = ru.permutation(cfg.n_slots)[: cfg.n_unlocks]
        for j in range(cfg.n_unlocks):
            entry = int(ru.choice([c for c in at[cfg.unlock_level] if c not in entries]))
            used = set(entries) | {kk for kk, _ in unlocks} | {entry}
            key_c = int(ru.choice([c for c in at[1] if c not in used]))
            entries.append(entry)
            unlocks.append((key_c, P + int(slots[j])))

    seqs: list[tuple[int, ...] | None] = [None] * n
    base_l1 = [c for c in at[1] if c not in entries]
    codes = _rng(seed, attempt, "level1").choice(P ** k, size=len(base_l1), replace=False)
    for c, code in zip(base_l1, codes):
        seqs[c] = _digits(int(code), k, P)
    for (_, slot), e in zip(unlocks, entries):
        r = _rng(seed, attempt, "entry", e)
        length = cfg.length(levels[e])
        for _ in range(1000):
            s = [int(v) for v in r.integers(0, P, length)]
            s[int(r.integers(length))] = slot
            if tuple(s) not in seqs:
                seqs[e] = tuple(s)
                break
        else:
            return None

    # validity check independent of reuse_depth: every hypothetical composition is distinct
    full = list(seqs)
    for lv in range(2, cfg.n_levels + 1):
        for x in at[lv]:
            if x not in entries:
                full[x] = full[parents[x][0]] + full[parents[x][1]]
        if len({full[x] for x in at[lv]}) < len(at[lv]):
            return None

    comp = [2 <= levels[x] <= cfg.reuse_depth + 1 and x not in entries for x in range(n)]
    for lv in range(2, cfg.n_levels + 1):
        for x in at[lv]:
            if comp[x]:
                seqs[x] = seqs[parents[x][0]] + seqs[parents[x][1]]
        for x in at[lv]:
            if seqs[x] is None:
                s = _flat(seed, attempt, x, cfg.length(lv), P, seqs)
                if s is None:
                    return None
                seqs[x] = s
    for lv in at:
        if len({seqs[x] for x in at[lv]}) < len(at[lv]):
            return None
    return TechWorld(key, seed, cfg, keys, tuple(levels), tuple(seqs), tuple(parents), tuple(comp),
                     tuple(unlocks), tuple(entries), attempt)


def _flat(seed, attempt, x, length, P, seqs) -> tuple[int, ...] | None:
    assigned = [s for s in seqs if s is not None]
    forbidden = {s for s in assigned if len(s) == length}
    for i, a in enumerate(assigned):
        for j, b in enumerate(assigned):
            if i != j and len(a) + len(b) == length:
                forbidden.add(a + b)
    r = _rng(seed, attempt, "flat", x)
    for _ in range(10_000):
        s = tuple(int(v) for v in r.integers(0, P, length))
        if s not in forbidden:
            return s
    return None


# ----------------------------------------------------------------- tasks
@dataclass(frozen=True)
class TechTask:
    """A complete task including private truth. Evaluator side only."""

    task_key: str
    namespace: str
    world: TechWorld
    goal: int
    budget: int
    world_index: int
    world_episode: int
    unlock_visibility: str = "announced"
    signal: bool = False


@dataclass(frozen=True)
class TechStreamConfig:
    """``n_worlds`` worlds per seed, each visited for ``episodes_per_world``
    consecutive episodes (world-major order). Worlds and goals are derived from
    (namespace, seed, world index[, episode]) only."""

    namespace: str
    seed: int
    n_worlds: int = 2
    episodes_per_world: int = 12
    world: TechWorldConfig = field(default_factory=TechWorldConfig)
    goal_levels: tuple[int, ...] = (3,)
    goal_rule: str = "uniform"
    episode_budget: int = 100
    unlock_visibility: str = "announced"
    signal: bool = False

    def __post_init__(self):
        if self.goal_rule not in GOAL_RULES:
            raise ValueError(f"goal_rule must be one of {GOAL_RULES}")
        if self.unlock_visibility not in VISIBILITIES:
            raise ValueError(f"unlock_visibility must be one of {VISIBILITIES}")
        if not self.goal_levels or not set(self.goal_levels) <= set(range(1, self.world.n_levels + 1)):
            raise ValueError("goal_levels must be levels of the world")
        if self.goal_rule == "unlock" and not self.world.n_unlocks:
            raise ValueError("goal_rule 'unlock' needs n_unlocks > 0")

    def to_dict(self) -> dict:
        return asdict(self)


class TechStream:
    def __init__(self, cfg: TechStreamConfig):
        self.cfg = cfg
        self._worlds: dict[int, TechWorld] = {}

    def world(self, widx: int) -> TechWorld:
        if widx not in self._worlds:
            c = self.cfg
            seed = derive_seed("techtree_world", c.namespace, c.seed, widx)
            key = "W" + stable_hash(("techtree_world", c.namespace, c.seed, widx))[:8]
            self._worlds[widx] = make_world(seed, c.world, key)
        return self._worlds[widx]

    def order(self) -> list[tuple[int, int]]:
        return [(w, e) for w in range(self.cfg.n_worlds) for e in range(self.cfg.episodes_per_world)]

    def task(self, widx: int, episode: int) -> TechTask:
        c = self.cfg
        w = self.world(widx)
        rng = np.random.default_rng(derive_seed("techtree_goal", c.namespace, c.seed, widx, episode))
        pool = [x for x in range(len(w.levels)) if w.levels[x] in c.goal_levels]
        if c.goal_rule == "unlock":
            pool = [x for x in pool if needs_unlock(w, x)]
            if not pool:
                raise ValueError(f"world {w.world_key} has no goal that needs an unlock")
            goal = int(rng.choice(pool))
        else:
            lv = int(rng.choice(sorted(set(c.goal_levels))))
            goal = int(rng.choice(w.at(lv)))
        task_key = "T" + stable_hash(("techtree_task", c.namespace, c.seed, widx, episode))[:10]
        return TechTask(task_key, c.namespace, w, goal, c.episode_budget, widx, episode,
                        c.unlock_visibility, c.signal)


def needs_unlock(world: TechWorld, goal: int) -> bool:
    """PRIVILEGED: whether firing ``goal`` requires an unlocked slot somewhere."""
    P = world.config.n_primitives
    return any(s >= P for x in world.requirements(goal) for s in world.sequences[x])


def revealed_links(world: TechWorld, levels=None) -> dict:
    """PRIVILEGED: the discovered links an oracle supplies to every arm alike:
    concept -> (sequence, parents, ((unlocked slot, its key), ...)).
    ``levels=None`` reveals every concept."""
    keys = world.slot_keys()
    out = {}
    for c, lv in enumerate(world.levels):
        if levels is None or lv in levels:
            seq = world.sequences[c]
            out[c] = (seq, world.parents[c], tuple(sorted({(s, keys[s]) for s in seq if s in keys})))
    return out


# ------------------------------------------------------------- reference
@dataclass(frozen=True)
class TechReference:
    status: str  # optimal | upper_bound
    exact: bool
    length: int  # primitive presses plus the final submit
    plan: tuple[int, ...]  # action indices, ending with submit
    states: int


def reference_solve(task: TechTask, max_states: int = 500_000) -> TechReference:
    """PRIVILEGED: shortest action sequence that makes the goal held, then submits.

    Breadth-first search over (held concepts among the goal's requirements,
    longest trace suffix that is a prefix of one of their sequences). Concepts
    outside the requirements cannot affect the goal, and pressing an inactive
    slot, waiting or submitting early only breaks the trace, so the search over
    active primitives is exact. Above ``max_states`` a constructive plan (each
    requirement typed in full, in dependency order) is returned as an upper
    bound."""
    w = task.world
    P = w.config.n_primitives
    S = P + w.config.n_slots
    submit = S + 1
    rel = sorted(w.requirements(task.goal))
    bit = {c: 1 << i for i, c in enumerate(rel)}
    pmask = {c: sum(bit[p] for p in w.parents[c]) for c in rel}
    pattern = {w.sequences[c]: c for c in rel}
    prefixes = {s[:i] for s in pattern for i in range(len(s))}
    slot_bit = {slot: bit[key] for slot, key in w.slot_keys().items() if key in bit}
    goal_bit = bit[task.goal]

    def advance(mask: int, t: tuple[int, ...]) -> int:
        changed = True
        while changed:
            changed = False
            for i in range(len(t)):
                c = pattern.get(t[i:])
                if c is not None and not mask & bit[c] and mask & pmask[c] == pmask[c]:
                    mask |= bit[c]
                    changed = True
        return mask

    def shrink(t: tuple[int, ...]) -> tuple[int, ...]:
        i = 0
        while t[i:] not in prefixes:
            i += 1
        return t[i:]

    start = (0, ())
    prev: dict = {start: None}
    queue = deque([start])
    while queue and len(prev) <= max_states:
        state = queue.popleft()
        mask, suffix = state
        for sym in range(S):
            if sym >= P and not mask & slot_bit.get(sym, 0):
                continue
            t = suffix + (sym,)
            nmask = advance(mask, t)
            nxt = (nmask, shrink(t))
            if nxt in prev:
                continue
            prev[nxt] = (state, sym)
            if nmask & goal_bit:
                plan = []
                s = nxt
                while prev[s] is not None:
                    s, a = prev[s]
                    plan.append(a)
                plan = tuple(reversed(plan)) + (submit,)
                return TechReference("optimal", True, len(plan), plan, len(prev))
            queue.append(nxt)
    order = sorted(rel, key=lambda c: (w.levels[c], c not in {kk for kk, _ in w.unlocks}))
    plan = tuple(s for c in order for s in w.sequences[c]) + (submit,)
    return TechReference("upper_bound", False, len(plan), plan, len(prev))


def debug_view(world: TechWorld) -> str:
    """Evaluator-only rendering of the hidden structure."""
    lines = [f"[EVALUATOR DEBUG VIEW - not an agent input] world {world.world_key}"]
    for c, lv in enumerate(world.levels):
        kind = "entry" if c in world.entries else (
            "combo" if lv == 1 else "comp" if world.compositional[c] else "flat")
        par = ",".join(world.keys[p] for p in world.parents[c])
        lines.append(f"  L{lv} {world.keys[c]} {kind:5s} parents[{par}] seq {world.sequences[c]}")
    for key, slot in world.unlocks:
        lines.append(f"  unlock: {world.keys[key]} activates slot {slot}")
    return "\n".join(lines)
