"""Study P arms: promotion when discoveries are imperfect learned skills (public information only).

In study L a discovery was exact: once a firing had revealed a concept's
sequence, typing it always worked, so promoting it to an atom of new hypotheses
carried no execution risk and needed no admission gate. Here every known
concept is a skill whose controller reproduces its sequence imperfectly and
improves with practice. The environment, the evidence and the engine of
``explorers.Explorer`` are unchanged; only the agent's executions of known
sequences change.

Skill model
-----------
An execution of a known concept c is one call of c's controller: a replay (to
hold c for gating or as the goal), c's part of a macro test (which types two
known concepts back to back) or a practice run. The controller presses each
intended primitive correctly with probability 1 - e and otherwise a different
available primitive (uniform among them), with

    e = e0 / (1 + n / h)

after n earlier executions of c in this world: every attempt counts as
practice, whatever its outcome, and h (``half_practice``) is the number of
executions that halves the error. The agent sees which primitive it pressed,
so eliminations, firings and precondition evidence are always computed from
the actual trace and stay exact. A controller that slips stops (that call can
no longer type its sequence), and the engine decides again from the current
trace, which usually repeats the replay or the macro test. Outcomes: ``ok``
(every intended press made), ``slip`` (stopped by a wrong press), ``cut``
(stopped by the engine before its last press without a slip: a firing, or the
end of the episode). A replay or practice run that types its sequence exactly
and does not fire its concept records a precondition failure, as in L; a
slipped one records nothing, because the sequence was not typed.

Primitive windows, ``wait`` and ``submit`` are pressed exactly: each press is a
single listed action, not a learned multi-step controller, and keeping them
exact confines the imperfection to the object of the study (executing a
discovery), so the arms' primitive searches stay those of study L.

Common random numbers: the n-th execution of concept c in a world reads the
n-th block of a stream seeded by (controller seed, world key, c), and the
controller seed is derived from the engine's symbol-preference seed without
drawing from the agent's generator. Every arm and error level of a seed
therefore meets the same draws for the same execution, a press that slips at
some e0 also slips at every larger e0, and at e0 = 0 ``remember`` and
``promote`` make exactly the decisions of study L's arms of the same name.

Arms
----
Adaptive arms are named ``<base>_eXX``, with XX the initial error e0 in
hundredths (``promote_gate_e30``: e0 = 0.3):

- ``remember``: study L's ``remember`` with imperfect replays.
- ``promote``: study L's ``promote``; a macro test executes two skills.
- ``promote_gate``: as ``promote``, but a known concept becomes an atom of
  macro hypotheses only once admitted: after ``gate_m`` ``ok`` outcomes among
  its last ``gate_k`` ``ok`` or ``slip`` executions (of any kind). Admission is
  once and for all, as in the RecipeQuest library. When a search level has no
  open macro hypothesis over admitted atoms but would have one over all known
  concepts, the agent practises (types the sequence of) a known concept that
  is not admitted and appears in such a hypothesis: the one with the most
  ``ok`` outcomes in its window, then the shortest, then the lowest index.
  Practice presses are ordinary actions of the episode, charged with purpose
  ``skill_practice``. Replays never need admission.

``oracle_library`` is study L's (every concept supplied by the runner, exact
execution): a privileged anchor.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .explorers import Chunk, Explorer
from .layered import Promote

DEFAULT_PARAMS = {"half_practice": 10, "gate_m": 2, "gate_k": 3}
BASES = ("remember", "promote", "promote_gate")
REPORTING_ARMS: tuple[str, ...] = ()  # oracle_library and reference are reporting arms of the runner
SKILL_KINDS = ("replay", "search_macro", "practice")
OUTCOMES = {"ok": 1, "slip": 0, "cut": -1}
CONTROLLER_TAG = 51711  # separates the controller streams from the symbol preference


def parse_arm(name: str) -> tuple[str, float]:
    """``promote_gate_e30`` -> (``promote_gate``, 0.3)."""
    base, sep, code = name.rpartition("_e")
    if not sep or base not in BASES or len(code) != 2 or not code.isdigit():
        raise KeyError(f"unknown study P arm {name!r}")
    return base, int(code) / 100


def pair_mask(k, level: int, nodes: dict) -> np.ndarray:
    """Windows of ``level`` that concatenate two distinct nodes (uncached ``macro_mask``)."""
    n = k.lengths[level - 1]
    m = np.zeros(k.S ** n, dtype=bool)
    for a, sa in nodes.items():
        for b, sb in nodes.items():
            if a != b and len(sa) + len(sb) == n:
                m[k.code(sa + sb)] = True
    return m


@dataclass
class SkillRecord:
    n: int = 0  # executions so far, every outcome
    recent: list = field(default_factory=list)  # ok (True) / slip (False) of the last gate_k
    admitted: bool = False


class SkillMixin:
    """Imperfect controllers for every execution of a known sequence (module docstring)."""

    def configure(self, e0: float, half_practice: float, gate_m: int, gate_k: int):
        if not 0 <= e0 < 1 or half_practice <= 0 or not 1 <= gate_m <= gate_k:
            raise ValueError("need 0 <= e0 < 1, half_practice > 0 and 1 <= gate_m <= gate_k")
        self.e0, self.half = float(e0), float(half_practice)
        self.gate_m, self.gate_k = int(gate_m), int(gate_k)
        # derived from the preference seed, so the agent's own generator is not consumed
        self.ctrl_seed = int(np.random.default_rng([self.pref_seed, CONTROLLER_TAG]).integers(2 ** 62))
        self.records: dict[str, dict[int, SkillRecord]] = defaultdict(dict)
        self.streams: dict[tuple[str, int], np.random.Generator] = {}
        self.world_key = ""
        self.counts: Counter = Counter()
        self.log: list[list[int]] = []
        return self

    # ---------------------------------------------------------- records
    def record(self, wk: str, c: int) -> SkillRecord:
        recs = self.records[wk]
        if c not in recs:
            recs[c] = SkillRecord()
        return recs[c]

    def error(self, rec: SkillRecord) -> float:
        return self.e0 / (1 + rec.n / self.half)

    def admitted(self, wk: str, c: int) -> bool:
        rec = self.records[wk].get(c)
        return rec is not None and rec.admitted

    def draws(self, wk: str, c: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """The next block of c's controller stream: one block per execution, in order."""
        key = (wk, c)
        if key not in self.streams:
            self.streams[key] = np.random.default_rng([self.ctrl_seed, *wk.encode(), c])
        block = self.streams[key].random(2 * length)
        return block[:length], block[length:]

    def finish(self, k, wk: str, c: int, outcome: str, presses: int, kind: str):
        rec = self.record(wk, c)
        self.log.append([c, k.level_of[c], rec.n, OUTCOMES[outcome], presses, SKILL_KINDS.index(kind)])
        rec.n += 1
        if outcome != "cut":
            rec.recent = (rec.recent + [outcome == "ok"])[-self.gate_k:]
            if not rec.admitted and sum(rec.recent) >= self.gate_m:
                rec.admitted = True
        self.counts["executions"] += 1
        self.counts["executions_" + outcome] += 1

    # -------------------------------------------------------- episodes
    def knowledge(self, spec):
        self.world_key = spec.world_key
        return super().knowledge(spec)

    def run_episode(self, env, token, meter) -> dict:
        self.counts, self.log = Counter(), []
        row = super().run_episode(env, token, meter)
        c = self.counts
        recs = self.records[self.world_key]
        row.update({
            "e0": self.e0, "practice_steps": c["practice_steps"],
            "exec_steps": row["replay_steps"] + row["macro_steps"] + c["practice_steps"],
            "failed_exec_steps": c["failed_exec_steps"], "executions": c["executions"],
            "executions_ok": c["executions_ok"], "executions_slip": c["executions_slip"],
            "executions_cut": c["executions_cut"], "discoveries_in_practice": c["new_practice"],
            "skills_admitted": sum(r.admitted for r in recs.values()),
            # one entry per execution: [concept, level, n before, outcome 1/0/-1, presses, kind index]
            "skill_exec": self.log,
        })
        return row

    # ------------------------------------------------------- execution
    def atoms(self, k, wk: str) -> dict:
        """Known concepts usable inside macro hypotheses."""
        return k.known

    def segments(self, k, wk: str, chunk: Chunk) -> list[tuple[int, int]]:
        """(skill, first position in its sequence) for each skill the chunk executes, in order."""
        if chunk.kind in ("replay", "practice"):
            return [(chunk.target, len(k.known[chunk.target]) - len(chunk.symbols))]
        if chunk.kind != "search_macro":
            return []
        window = k.digits(chunk.level, chunk.code)
        nodes = self.atoms(k, wk)
        for i in range(1, len(window)):
            a, b = k.by_seq.get(window[:i]), k.by_seq.get(window[i:])
            if a is not None and b is not None and a != b and a in nodes and b in nodes:
                start = len(window) - len(chunk.symbols)
                return [(a, start), (b, 0)] if start < i else [(b, start - i)]
        raise RuntimeError("a macro window is not the concatenation of two atoms")

    def _execute(self, k, ep, chunk: Chunk, stats: Counter):
        segs = self.segments(k, ep.spec.world_key, chunk)
        if not segs:
            self._exact(k, ep, chunk, stats)
            return
        purpose = ep.purpose
        if chunk.kind == "practice" and purpose == "exploration":
            ep.purpose = "skill_practice"
        try:
            self._controlled(k, ep, chunk, stats, segs)
        finally:
            ep.purpose = purpose

    def _exact(self, k, ep, chunk: Chunk, stats: Counter):
        """Primitive windows, idle: pressed exactly, as ``Explorer`` does."""
        for a in chunk.symbols:
            fired, new = self._press(k, ep, a)
            stats[chunk.kind] += 1
            stats["new_" + chunk.kind] += new
            if ep.done or fired:
                return
            if chunk.code is not None and a < ep.spec.n_symbols and k.dead(chunk.level, chunk.code, ep.held):
                return

    def _controlled(self, k, ep, chunk: Chunk, stats: Counter, segs):
        wk, spec = ep.spec.world_key, ep.spec
        total = 0
        for c, start in segs:
            seq = k.known[c]
            e = self.error(self.record(wk, c))
            u, v = self.draws(wk, c, len(seq)) if e > 0 else (None, None)
            presses = 0
            for p in range(start, len(seq)):
                a, slip = seq[p], False
                if e > 0 and u[p] < e:
                    alts = [s for s in range(spec.n_symbols) if s != a and ep.obs.available[s]]
                    if alts:
                        a, slip = alts[int(v[p] * len(alts))], True
                fired, new = self._press(k, ep, a)
                presses += 1
                total += 1
                stats[chunk.kind] += 1
                stats["new_" + chunk.kind] += new
                if chunk.kind == "practice":
                    self.counts["practice_steps"] += 1
                    self.counts["new_practice"] += new
                if slip:
                    self.finish(k, wk, c, "slip", presses, chunk.kind)
                    self.counts["failed_exec_steps"] += total
                    return
                stop = ep.done or fired or (chunk.code is not None
                                            and k.dead(chunk.level, chunk.code, ep.held))
                if stop:
                    self.finish(k, wk, c, "ok" if p == len(seq) - 1 else "cut", presses, chunk.kind)
                    return
            self.finish(k, wk, c, "ok", presses, chunk.kind)
        if chunk.kind in ("replay", "practice") and chunk.target not in ep.held:
            k.add_failure(chunk.target, ep.held)  # the sequence was typed exactly


class SkilledRemember(SkillMixin, Explorer):
    pass


class SkilledPromote(SkillMixin, Promote):
    pass


class PromoteGate(SkillMixin, Promote):
    def atoms(self, k, wk: str) -> dict:
        return {c: s for c, s in k.known.items() if self.admitted(wk, c)}

    def choose(self, k, ep, level, open_):
        macro = pair_mask(k, level, self.atoms(k, ep.spec.world_key)) & open_
        if macro.any():
            return "search_macro", macro
        return "search_primitive", self.primitive_mask(k, level, open_)

    def practice_target(self, k, wk: str, level: int, open_: np.ndarray) -> int | None:
        """A known concept that is not admitted and would be an atom of an open macro hypothesis."""
        full = k.macro_mask(level, k.known) & open_
        cands = set()
        for code in np.flatnonzero(full):
            window = k.digits(level, int(code))
            for i in range(1, len(window)):
                a, b = k.by_seq.get(window[:i]), k.by_seq.get(window[i:])
                if a is not None and b is not None and a != b:
                    cands |= {x for x in (a, b) if not self.admitted(wk, x)}
        if not cands:
            return None
        return min(cands, key=lambda c: (-sum(self.record(wk, c).recent), len(k.known[c]), c))

    def _search(self, k, ep, level: int) -> Chunk:
        wk = ep.spec.world_key
        open_ = k.open_mask(level, ep.held, self.usable(k, ep))
        kind, mask = self.choose(k, ep, level, open_)
        if kind != "search_macro":
            c = self.practice_target(k, wk, level, open_)
            if c is not None:
                return Chunk("practice", self._replay(k, ep, c).symbols, target=c)
        if not mask.any():
            mask = open_
        if not mask.any():
            return Chunk("idle", (ep.spec.wait_action,))
        syms, code = self.extension(k, ep, level, mask)
        return Chunk(kind, syms, level=level, code=code)


ARM_CLASSES = {"remember": SkilledRemember, "promote": SkilledPromote, "promote_gate": PromoteGate}


def make_skilled(name: str, rng: np.random.Generator, params: dict | None = None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    unknown = set(p) - set(DEFAULT_PARAMS)
    if unknown:
        raise KeyError(f"unknown study P agent parameters {sorted(unknown)}")
    if name == "oracle_library":
        return Explorer(rng)  # study L's anchor: supplied links, exact execution
    base, e0 = parse_arm(name)
    return ARM_CLASSES[base](rng).configure(e0, p["half_practice"], p["gate_m"], p["gate_k"])


# ------------------------------------------------------------ row summaries
# nested {variant: {arm: ...}} like the analysis tables, so the analysis can store them as JSON
def execution_shares(rows: list[dict]) -> dict:
    """{variant: {arm: shares of all actions spent executing skills, practising and in executions
    stopped by a slip, and the share of executions that slipped}}; rows of the skilled arms only."""
    acc = defaultdict(Counter)
    for r in rows:
        if "exec_steps" not in r:
            continue
        a = acc[(r.get("variant"), r.get("arm"))]
        for key in ("primitive_length", "exec_steps", "practice_steps", "failed_exec_steps",
                    "executions", "executions_slip"):
            a[key] += r[key]
    out: dict = {}
    for (variant, arm), a in sorted(acc.items(), key=lambda kv: str(kv[0])):
        total = a["primitive_length"] or 1
        out.setdefault(variant, {})[arm] = {
            "exec_share": a["exec_steps"] / total, "practice_share": a["practice_steps"] / total,
            "failed_share": a["failed_exec_steps"] / total,
            "slip_rate": a["executions_slip"] / a["executions"] if a["executions"] else None}
    return out


def reliability_curve(rows: list[dict], bins=(0, 1, 3, 6, 10, 20, 40, 80)) -> dict:
    """{variant: {arm: {skill level: [bins]}}}: the share of ``ok`` among ``ok`` and ``slip``
    executions by bins of the execution index n (practice so far); ``cut`` executions are left out."""
    acc = defaultdict(lambda: np.zeros((len(bins), 2)))
    edges = np.array(bins)
    for r in rows:
        for _, level, n, outcome, _, _ in r.get("skill_exec", ()):
            if outcome >= 0:
                cell = acc[(r.get("variant"), r.get("arm"), level)]
                b = int(np.searchsorted(edges, n, side="right")) - 1
                cell[b] += (outcome, 1)
    out: dict = {}
    for (variant, arm, level), cell in sorted(acc.items(), key=lambda kv: str(kv[0])):
        out.setdefault(variant, {}).setdefault(arm, {})[int(level)] = [
            {"n_from": int(bins[i]), "executions": int(t), "ok_rate": float(s / t)}
            for i, (s, t) in enumerate(cell) if t]
    return out


# read by techtree.analysis for study P (stored under row_summaries in analysis.json)
ROW_SUMMARIES = {"execution_shares": execution_shares, "reliability_curve": reliability_curve}
