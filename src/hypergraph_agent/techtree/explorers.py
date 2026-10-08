"""Elimination explorers for TechTree (public information only).

Agents hold only a public facade: a ``PublicEnv`` and opaque ``TaskToken``s
(``env.agent_interface``), with which they reset, step and read the public spec
and their observations. Ordinary attribute access cannot reach private state,
and introspection is excluded by a source check of this module and ``layered``
(``tests/test_techtree.py``: no reflective builtins, dunder, frame or traceback
attributes, and no private attributes of objects other than ``self``). One
engine serves every non-random arm of studies U and L; arms differ in a few
switches.

Hypotheses and elimination
--------------------------
For a level l the hypotheses are the windows (sequences of the level's public
length) over the agent's alphabet. A window is eliminated for every
undiscovered concept of level l when it was the trace suffix while the agent
could be sure that their gating held and no level-l concept fired. Sure
gating: every concept at the declared parent levels of l is held, or is
undiscovered at a closed level (no window of that level remains open over the
agent's alphabet, so that concept, and everything that needs it, is out of the
agent's reach). With the intermediate signal on, a false ``progress`` flag also
eliminates every window that starts with a suffix of the trace. A firing
reveals the concept's sequence exactly: the trace suffix of its public length.

Silent slots: a ``try`` press may have been inactive (an inert symbol), so a
test that involves one is stored with the held set at that press and counts as
an elimination only while the current held set is a subset of it; activations
are monotone in the held set. This is the prior "new facts enable new
actions": slot windows are tried only once some concept is held, and are
re-opened whenever the held context grows beyond the one they were tested in.

Search
------
The engine first makes sure gating holds for the lowest level that still has
undiscovered concepts below or at the goal's level (replaying known concepts),
then picks hypotheses. Window order: windows that use usable extra slots first
and among those the fewest slots (an announced slot once it appears; silent
slots once some concept is held), otherwise base windows. ``pooled`` agents
press the shortest extension of the current trace that ends in an open window
of the chosen set (one press per new window along a de Bruijn-like walk, a
fixed per-world symbol preference breaking ties); ``isolated`` agents test one
window at a time, in the lexicographic order of that same preference, and clear
the trace with ``wait`` before each window longer than one press, so no press
tests two candidates. For one-press windows both orders coincide, so the two
arms then act identically.

Replay
------
For each known concept the agent keeps the intersection of the held sets at
its firings (a superset of its preconditions) and the held sets at which
typing its sequence failed to fire it. To fire a concept it holds the cheapest
parent pair at the declared parent levels that is consistent with both, plus
the key of each unlocked slot in its sequence (announced: the concept that
fired when the slot appeared; silent: every remaining candidate), recursively,
and types the sequence, skipping any part already at the end of the trace.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .env import TechTreeSpec


class Episode:
    """One episode through the public interface; every action is charged to the meter."""

    def __init__(self, env, token, meter):
        self.obs, _ = env.reset(token)
        self.env, self.meter = env, meter
        self.spec: TechTreeSpec = env.public_spec
        self.purpose = "reporting_eval" if meter.kind == "reporting" else "exploration"
        self.held: frozenset = frozenset()
        self.trace: list[int] = []
        self.trace_ctx: list[frozenset] = []  # held set before each press of the trace
        self.fired: list[tuple[int, int, int]] = []  # (step, concept, level)
        self.steps = 0
        self.done = self.success = False
        self.status: str | None = None
        self._max = max(self.spec.level_lengths)

    def step(self, action: int) -> tuple[int, ...]:
        if not self.meter.can_charge(1):
            self.done, self.status = True, "budget_exhausted"
            return ()
        before = self.held
        self.meter.charge(1, self.purpose)
        obs, reward, terminated, _, info = self.env.step(action)
        self.steps += 1
        self.obs = obs
        if action < self.spec.n_symbols:
            self.trace.append(action)
            self.trace_ctx.append(before)
            if len(self.trace) > self._max:
                del self.trace[0], self.trace_ctx[0]
        else:
            self.trace.clear()
            self.trace_ctx.clear()
        fired = info["fired"]
        if fired:
            self.held = before | frozenset(fired)
            self.fired.extend((self.steps, c, self.spec.concept_levels[c]) for c in fired)
        if terminated:
            self.done, self.success, self.status = True, reward > 0, info["termination"]
        return fired

    def row(self) -> dict:
        goal_step = next((s for s, c, _ in self.fired if c == self.spec.goal), None)
        return {"success": bool(self.success), "status": self.status, "primitive_length": self.steps,
                "goal_first_step": goal_step, "fired": [list(f) for f in self.fired]}


def _digit_table(base: int, n: int) -> np.ndarray:
    codes = np.arange(base ** n)
    table = np.empty((base ** n, n), dtype=np.int16)
    for i in range(n - 1, -1, -1):
        table[:, i] = codes % base
        codes //= base
    return table


class WorldKnowledge:
    """What one agent has learned about one world, from public evidence only."""

    def __init__(self, spec: TechTreeSpec, rng: np.random.Generator):
        self.P, self.S = spec.n_base, spec.n_symbols
        self.slots = tuple(range(self.P, self.S))
        self.silent = spec.unlock_visibility == "silent" and spec.n_slots > 0
        self.level_of = spec.concept_levels
        self.lengths = spec.level_lengths
        self.parent_levels = spec.parent_levels
        self.n_levels = len(spec.level_lengths)
        self.at = {lv: spec.concepts_at(lv) for lv in range(1, self.n_levels + 1)}
        self.parent_pool = {lv: [c for pl in sorted(set(spec.parent_levels[lv - 1])) for c in self.at[pl]]
                            for lv in self.at}
        self.known: dict[int, tuple[int, ...]] = {}
        self.by_seq: dict[tuple[int, ...], int] = {}
        self.revealed: set[int] = set()
        self.cands: dict[int, frozenset] = {}
        self.bad: dict[int, list[frozenset]] = {}
        self.unlock_by: dict[int, frozenset] = {}
        self.comp = self.flat = 0
        self.sham: dict[int, tuple[int, ...]] = {}
        self.contexts: list[frozenset] = []
        self._ctx_id: dict[frozenset, int] = {}
        self.version = 0
        self._cache: dict = {}
        self.table, self.contains, self.extra_count = {}, {}, {}
        self.elim, self.cond, self.known_mask = {}, {}, {}
        for lv in self.at:
            n = self.lengths[lv - 1]
            t = _digit_table(self.S, n)
            self.table[lv] = t
            self.contains[lv] = np.stack([(t == s).any(axis=1) for s in range(self.S)])
            self.extra_count[lv] = (t >= self.P).sum(axis=1)
            self.elim[lv] = np.zeros(self.S ** n, dtype=bool)
            self.known_mask[lv] = np.zeros(self.S ** n, dtype=bool)
            if self.silent:
                self.cond[lv] = np.full(self.S ** n, -1, dtype=np.int32)
        pref = [int(s) for s in rng.permutation(self.S)]
        self.rank = {s: i for i, s in enumerate(pref)}
        ranks = np.array([self.rank[s] for s in range(self.S)], dtype=np.int64)
        # lexicographic window order under the symbol preference (isolated testing)
        self.order = {lv: (ranks[t] * self.S ** np.arange(t.shape[1] - 1, -1, -1)).sum(axis=1)
                      for lv, t in self.table.items()}

    # ------------------------------------------------------------- codes
    def code(self, seq) -> int:
        c = 0
        for s in seq:
            c = c * self.S + int(s)
        return c

    def digits(self, level: int, code: int) -> tuple[int, ...]:
        return tuple(int(s) for s in self.table[level][code])

    def alpha_mask(self, level: int, allowed: frozenset) -> np.ndarray:
        key = ("alpha", level, allowed)
        if key not in self._cache:
            banned = [s for s in range(self.S) if s not in allowed]
            self._cache[key] = ~self.contains[level][banned].any(axis=0) if banned \
                else np.ones(self.S ** self.lengths[level - 1], dtype=bool)
        return self._cache[key]

    def ctx_id(self, ctx: frozenset) -> int:
        if ctx not in self._ctx_id:
            self._ctx_id[ctx] = len(self.contexts)
            self.contexts.append(ctx)
        return self._ctx_id[ctx]

    # ---------------------------------------------------------- learning
    def _add_known(self, c: int, seq: tuple[int, ...]):
        self.known[c] = seq
        self.by_seq[seq] = c
        self.known_mask[self.level_of[c]][self.code(seq)] = True
        self.version += 1

    def reveal(self, c: int, seq, parents, slot_keys):
        """Oracle-supplied link: exact sequence, parents and slot keys."""
        if c in self.revealed:
            return
        seq = tuple(seq)
        if c not in self.known:
            self._add_known(c, seq)
        self.revealed.add(c)
        self.cands[c] = frozenset(parents) | frozenset(k for _, k in slot_keys)
        for slot, key in slot_keys:
            self.unlock_by[slot] = frozenset({key})

    def is_composition(self, c: int, seq: tuple[int, ...]) -> bool:
        """Whether ``seq`` is the concatenation of two other distinct known concepts."""
        for i in range(1, len(seq)):
            a, b = self.by_seq.get(seq[:i]), self.by_seq.get(seq[i:])
            if a is not None and b is not None and a != b and c not in (a, b):
                return True
        return False

    def record_firing(self, c: int, seq: tuple[int, ...], held_after: frozenset) -> bool:
        new = c not in self.known
        if new:
            if self.level_of[c] >= 2:
                if self.is_composition(c, seq):
                    self.comp += 1
                else:
                    self.flat += 1
            self._add_known(c, seq)
        cand = held_after - {c}
        if not any(s >= self.P for s in seq):
            cand &= frozenset(self.parent_pool[self.level_of[c]])
        self.cands[c] = self.cands[c] & cand if c in self.cands else cand
        return new

    def eliminate(self, level: int, window, ctx):
        code = self.code(window)
        first = next((i for i, s in enumerate(window) if s >= self.P), None)
        if first is None or not self.silent:
            self.elim[level][code] = True
        else:
            self.cond[level][code] = self.ctx_id(ctx[first])

    def eliminate_prefix(self, level: int, prefix, ctx):
        n = self.lengths[level - 1]
        span = self.S ** (n - len(prefix))
        lo = self.code(prefix) * span
        first = next((i for i, s in enumerate(prefix) if s >= self.P), None)
        if first is None or not self.silent:
            self.elim[level][lo:lo + span] = True
        else:
            self.cond[level][lo:lo + span] = self.ctx_id(ctx[first])

    def add_failure(self, c: int, held: frozenset):
        self.bad.setdefault(c, []).append(held - {c})

    def ensure_sham(self, rng: np.random.Generator):
        """One useless macro node per known concept: a random base sequence of the
        same length that is no known sequence."""
        taken = set(self.known.values()) | set(self.sham.values())
        for c, seq in self.known.items():
            if c not in self.sham:
                while True:
                    s = tuple(int(v) for v in rng.integers(0, self.P, len(seq)))
                    if s not in taken:
                        self.sham[c] = s
                        taken.add(s)
                        break

    # ------------------------------------------------------------ status
    def unknown_levels(self) -> list[int]:
        return [lv for lv, cs in self.at.items() if any(c not in self.known for c in cs)]

    def open_mask(self, level: int, held: frozenset, allowed: frozenset) -> np.ndarray:
        m = ~self.elim[level] & ~self.known_mask[level] & self.alpha_mask(level, allowed)
        if self.silent:
            valid = [i for i, ctx in enumerate(self.contexts) if held <= ctx]
            if valid:
                m &= ~np.isin(self.cond[level], valid)
        return m

    def dead(self, level: int, code: int, held: frozenset) -> bool:
        if self.elim[level][code] or self.known_mask[level][code]:
            return True
        cid = self.cond[level][code] if self.silent else -1
        return cid >= 0 and held <= self.contexts[cid]

    def closed(self, level: int, alphabet: frozenset) -> bool:
        """All concepts of the level known, or no window open over the alphabet
        (conditional silent eliminations count as open)."""
        if all(c in self.known for c in self.at[level]):
            return True
        return not (~self.elim[level] & ~self.known_mask[level] & self.alpha_mask(level, alphabet)).any()

    def sure(self, level: int, held: frozenset, alphabet: frozenset) -> bool:
        return all(c in held or (c not in self.known and self.closed(self.level_of[c], alphabet))
                   for c in self.parent_pool[level])

    def macro_mask(self, level: int, nodes: dict) -> np.ndarray:
        """Windows that are the concatenation of two distinct nodes' sequences."""
        key = ("macro", level, id(nodes), len(nodes), self.version)
        if key not in self._cache:
            n = self.lengths[level - 1]
            m = np.zeros(self.S ** n, dtype=bool)
            items = list(nodes.items())
            for a, sa in items:
                for b, sb in items:
                    if a != b and len(sa) + len(sb) == n:
                        m[self.code(sa + sb)] = True
            self._cache = {k: v for k, v in self._cache.items() if k[0] != "macro" or k[1] != level}
            self._cache[key] = m
        return self._cache[key]

    # ------------------------------------------------------------ replay
    def preconditions(self, c: int, held: frozenset, memo: dict) -> list[int]:
        lv, seq = self.level_of[c], self.known[c]
        out = list(self.parent_hypothesis(c, held, memo)) if lv >= 2 else []
        slots = {s for s in seq if s >= self.P}
        if slots:
            if self.silent:
                out += sorted(self.cands.get(c, frozenset()) - set(out))
            else:
                for s in sorted(slots):
                    out += sorted(self.unlock_by.get(s, frozenset()) - set(out))
        return [p for p in out if p != c and p in self.known]

    def parent_hypothesis(self, c: int, held: frozenset, memo: dict) -> tuple[int, ...]:
        lv = self.level_of[c]
        pool = self.cands.get(c, frozenset(self.parent_pool[lv])) & frozenset(self.parent_pool[lv])
        pool = [p for p in pool if p in self.known]
        la, lb = self.parent_levels[lv - 1]
        pairs = {tuple(sorted((p, q))) for p in pool for q in pool
                 if p != q and sorted((self.level_of[p], self.level_of[q])) == sorted((la, lb))}
        bad = self.bad.get(c, [])
        valid = [pr for pr in pairs if not any(set(pr) <= h for h in bad)]
        if not valid:
            return tuple(sorted(pool))
        best = min(valid, key=lambda pr: (sum(self.cost(p, held, memo) for p in pr), pr))
        seq, first = self.known[c], self.known[best[0]]
        return best if seq[:len(first)] == first else (best[1], best[0])

    def cost(self, c: int, held: frozenset, memo: dict) -> int:
        if c in held:
            return 0
        if c not in memo:
            memo[c] = len(self.known[c])  # provisional value guards against cycles
            pre = self.preconditions(c, held, memo)
            memo[c] = len(self.known[c]) + sum(self.cost(p, held, memo) for p in pre)
        return memo[c]

    def plan(self, targets, held: frozenset) -> list[int]:
        """Concepts to fire, in order, so that every target is held."""
        order, seen, memo = [], set(), {}

        def visit(c):
            if c in held or c in seen:
                return
            seen.add(c)
            for p in self.preconditions(c, held, memo):
                visit(p)
            order.append(c)

        for t in targets:
            visit(t)
        return order


@dataclass(frozen=True)
class Chunk:
    kind: str  # replay | search_primitive | search_macro | idle
    symbols: tuple[int, ...]
    target: int | None = None  # replay: the concept to fire
    level: int | None = None  # search: level and code of the tested window
    code: int | None = None


class Explorer:
    """Elimination explorer with switches for the arms of studies U and L.

    memory:    keep world knowledge across episodes (False: forget after each).
    use_extra: add slots to the alphabet (False: never press a slot).
    isolated:  test one window at a time with the trace cleared in between.
    """

    def __init__(self, rng: np.random.Generator, *, memory: bool = True, use_extra: bool = True,
                 isolated: bool = False):
        self.rng = rng
        # per-world symbol preference from a seed drawn first, so agents built from the same seed
        # share it whatever randomness they consume later (common random numbers across arms)
        self.pref_seed = int(rng.integers(2 ** 62))
        self.memory, self.use_extra, self.isolated = memory, use_extra, isolated
        self.worlds: dict[str, WorldKnowledge] = {}
        self._revealed: dict[str, dict] = {}

    # ---------------------------------------------------------- interface
    def reveal(self, world_key: str, links: dict):
        """Links supplied by an oracle (identical for every arm of a phase):
        concept -> (sequence, parents, ((slot, key), ...))."""
        self._revealed[world_key] = dict(links)

    def knowledge(self, spec: TechTreeSpec) -> WorldKnowledge:
        k = self.worlds.get(spec.world_key) if self.memory else None
        if k is None:
            k = WorldKnowledge(spec, np.random.default_rng([self.pref_seed, *spec.world_key.encode()]))
            if self.memory:
                self.worlds[spec.world_key] = k
        for c, (seq, parents, slot_keys) in sorted(self._revealed.get(spec.world_key, {}).items()):
            k.reveal(c, seq, parents, slot_keys)
        return k

    def run_episode(self, env, token, meter) -> dict:
        ep = Episode(env, token, meter)
        k = self.knowledge(ep.spec)
        before = {c for c in k.known if c not in k.revealed}
        comp0, flat0 = k.comp, k.flat
        stats: Counter = Counter()
        goal = ep.spec.goal
        while not ep.done:
            if goal in ep.held:
                ep.step(ep.spec.submit_action)
                stats["submit"] += 1
                continue
            self._execute(k, ep, self._decide(k, ep), stats)
        row = ep.row()
        new = [c for c in k.known if c not in k.revealed and c not in before]
        row.update({
            "replay_steps": stats["replay"],
            "search_steps": stats["search_primitive"] + stats["search_macro"],
            "macro_steps": stats["search_macro"], "primitive_search_steps": stats["search_primitive"],
            "idle_steps": stats["idle"], "discoveries_new": len(new),
            # discoveries made while replaying known concepts (incidental composition) or searching
            "discoveries_in_replay": stats["new_replay"], "discoveries_in_macro": stats["new_search_macro"],
            "discoveries_in_primitive": stats["new_search_primitive"], "known_total": len(k.known),
            "revealed_total": len(k.revealed), "compositional_new": k.comp - comp0,
            "flat_new": k.flat - flat0,
            "belief_compositional": (1 + k.comp) / (2 + k.comp + k.flat),
        })
        return row

    # ----------------------------------------------------------- choices
    def usable(self, k: WorldKnowledge, ep: Episode) -> frozenset:
        """Symbols the agent may use as hypotheses now."""
        base = frozenset(range(k.P))
        if not self.use_extra:
            return base
        if k.silent:
            return base | frozenset(k.slots) if ep.held else base
        return base | frozenset(s for s in k.slots if ep.obs.available[s])

    def alphabet(self, k: WorldKnowledge) -> frozenset:
        """The agent's full alphabet, for deciding that a level is closed."""
        base = frozenset(range(k.P))
        if not self.use_extra:
            return base
        return base | (frozenset(k.slots) if k.silent else frozenset(k.unlock_by))

    def _decide(self, k: WorldKnowledge, ep: Episode) -> Chunk:
        spec, held = ep.spec, ep.held
        if spec.goal in k.known:
            return self._replay(k, ep, k.plan([spec.goal], held)[0])
        alpha = self.alphabet(k)
        target = next((lv for lv in range(1, spec.concept_levels[spec.goal] + 1)
                       if not k.closed(lv, alpha)), None)
        if target is None:  # out of reach with this alphabet
            return Chunk("idle", (spec.wait_action,))
        need = [c for c in k.parent_pool[target] if c in k.known and c not in held]
        if self.use_extra and k.slots:
            if k.silent:
                need += [c for c in sorted(k.known) if c not in held and c not in need]
            else:
                for s, keys in sorted(k.unlock_by.items()):
                    if not ep.obs.available[s]:
                        need += [c for c in sorted(keys) if c not in held and c not in need]
        if need:
            return self._replay(k, ep, k.plan(need, held)[0])
        return self._search(k, ep, target)

    def _replay(self, k: WorldKnowledge, ep: Episode, c: int) -> Chunk:
        seq, tr = k.known[c], ep.trace
        overlap = 0
        for j in range(min(len(seq) - 1, len(tr)), 0, -1):
            if tuple(tr[-j:]) == seq[:j] and not (k.silent and any(s >= k.P for s in seq[:j])):
                overlap = j
                break
        return Chunk("replay", seq[overlap:], target=c)

    def primitive_mask(self, k: WorldKnowledge, level: int, open_: np.ndarray) -> np.ndarray:
        """Windows with usable slots first (fewest slots), otherwise base windows."""
        tiers = k.extra_count[level]
        if self.use_extra:
            extra = open_ & (tiers > 0)
            if extra.any():
                return open_ & (tiers == tiers[extra].min())
        return open_ & (tiers == 0)

    def choose(self, k: WorldKnowledge, ep: Episode, level: int, open_: np.ndarray):
        """Hypothesis set to test next (overridden by the layered arms)."""
        return "search_primitive", self.primitive_mask(k, level, open_)

    def _search(self, k: WorldKnowledge, ep: Episode, level: int) -> Chunk:
        open_ = k.open_mask(level, ep.held, self.usable(k, ep))
        kind, mask = self.choose(k, ep, level, open_)
        if not mask.any():
            mask = open_
        if not mask.any():
            return Chunk("idle", (ep.spec.wait_action,))
        if self.isolated:
            idx = np.flatnonzero(mask)
            code = int(idx[np.argmin(k.order[level][idx])])
            syms = k.digits(level, code)
            clear = (ep.spec.wait_action,) if ep.trace and len(syms) > 1 else ()
            return Chunk(kind, clear + syms, level=level, code=code)
        syms, code = self.extension(k, ep, level, mask)
        return Chunk(kind, syms, level=level, code=code)

    def extension(self, k: WorldKnowledge, ep: Episode, level: int, mask: np.ndarray, pick: bool = True):
        """Shortest continuation of the trace that ends in a window of ``mask``."""
        n, S, tr = k.lengths[level - 1], k.S, ep.trace
        for extra in range(max(1, n - len(tr)), n + 1):
            plen = n - extra
            pre = tr[len(tr) - plen:] if plen else []
            if k.silent and any(s >= k.P and ep.trace_ctx[len(tr) - plen + i] != ep.held
                                for i, s in enumerate(pre)):
                continue  # a stale slot press may have been inert
            lo = k.code(pre) * S ** extra
            idx = np.flatnonzero(mask[lo:lo + S ** extra])
            if idx.size:
                if not pick:
                    return extra
                j = min(idx, key=lambda s: k.rank[int(s)]) if extra == 1 else int(self.rng.choice(idx))
                code = lo + int(j)
                return k.digits(level, code)[plen:], code
        raise RuntimeError("no extension found for a nonempty hypothesis set")

    # --------------------------------------------------------- execution
    def _execute(self, k: WorldKnowledge, ep: Episode, chunk: Chunk, stats: Counter):
        for a in chunk.symbols:
            fired, new = self._press(k, ep, a)
            stats[chunk.kind] += 1
            stats["new_" + chunk.kind] += new
            if ep.done or fired:
                return
            if chunk.code is not None and a < ep.spec.n_symbols and k.dead(chunk.level, chunk.code, ep.held):
                return  # the window was eliminated on the way (intermediate signal)
        if chunk.kind == "replay" and chunk.target not in ep.held:
            k.add_failure(chunk.target, ep.held)

    def _press(self, k: WorldKnowledge, ep: Episode, a: int) -> tuple[tuple[int, ...], int]:
        """One action; returns the concepts that fired and how many were new discoveries."""
        if a >= ep.spec.n_symbols:
            ep.step(a)
            return (), 0
        held0, alpha, avail0 = ep.held, self.alphabet(k), ep.obs.available
        sure = [lv for lv in k.unknown_levels() if k.sure(lv, held0, alpha)]
        fired = ep.step(a)
        if ep.status == "budget_exhausted":
            return (), 0
        tr, ctx = ep.trace, ep.trace_ctx
        fired_windows, new = set(), 0
        for c in fired:
            n = ep.spec.length(c)
            seq = tuple(tr[-n:])
            new += k.record_firing(c, seq, ep.held)
            fired_windows.add((k.level_of[c], seq))
        for lv in sure:
            n = k.lengths[lv - 1]
            if len(tr) >= n and (lv, tuple(tr[-n:])) not in fired_windows:
                k.eliminate(lv, tr[-n:], ctx[-n:])
        if ep.spec.signal and ep.obs.progress is False:
            for lv in sure:
                for j in range(1, min(k.lengths[lv - 1] - 1, len(tr)) + 1):
                    k.eliminate_prefix(lv, tr[-j:], ctx[-j:])
        if self.use_extra and not k.silent and fired:
            for s in k.slots:
                if ep.obs.available[s] and not avail0[s]:  # appeared now: a fired concept is its key
                    keys = frozenset(fired)
                    k.unlock_by[s] = k.unlock_by[s] & keys if s in k.unlock_by else keys
        return fired, new


class RandomAgent:
    """Uniform over the available primitives; submits as soon as the goal is held."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def reveal(self, world_key: str, links: dict):
        pass

    def run_episode(self, env, token, meter) -> dict:
        ep = Episode(env, token, meter)
        spec = ep.spec
        while not ep.done:
            if spec.goal in ep.held:
                ep.step(spec.submit_action)
            else:
                avail = [s for s in range(spec.n_symbols) if ep.obs.available[s]]
                ep.step(int(self.rng.choice(avail)))
        return ep.row()


U_ARMS = {
    "pooled": {}, "isolated": {"isolated": True}, "nomem": {"memory": False},
    "blind": {"use_extra": False}, "oracle": {},
}


def make_explorer(name: str, rng: np.random.Generator):
    if name == "random":
        return RandomAgent(rng)
    if name not in U_ARMS:
        raise KeyError(f"unknown study U arm {name!r}")
    return Explorer(rng, **U_ARMS[name])
