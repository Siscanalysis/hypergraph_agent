"""Study CA arms: atoms proposed by compressing the agent's own press sequences (public information only).

Agents hold only the public facade (``env.agent_interface``); this module is under the source check
of the TechTree agent modules (no reflective builtins, dunder, frame or traceback attributes, no
private attributes of objects other than ``self``; imports restricted, with ``stitch_core`` added
for this module: docs/studies/CA_compression_atoms.md, Implementation). The arms record their own
presses through ``PressLog``, a wrapper of their public environment handle, and read the engine's
evidence (known sequences, failure clauses, eliminated windows) through its public attributes.

Corpus
------
Every compression arm keeps, per world, a corpus of its own press sequences, successful and
failed, which persists across the episodes of a world and is dropped with it, as the engine's
memory is. Agent parameter ``corpus`` (one value per config):

- ``tests`` (the measurement's): every window of the agent's own trace that its evidence counts as
  a test. Successful press sequences are the sequences that fired, one per firing the agent
  observes (discoveries, replays and incidental firings alike: the trace suffix of the fired
  concept's public length). Failed press sequences are the windows that the engine's evidence newly
  marks as tested without firing in the episode (its eliminations, at every level), and the replays
  that did not fire their target (the replayed sequence, once per new failure clause of the
  engine).
- ``firings``: the successful press sequences and the failed replays only.
- ``episodes``: each episode's whole press sequence, split at ``wait`` and ``submit``, as one
  successful or failed sequence according to the episode's outcome.

Sequences of fewer than two presses are dropped. The compressors see the sequences only: neither
which concept fired, nor whether a sequence succeeded.

Compressors (between episodes)
------------------------------
At the start of every episode whose world corpus has grown since the last compression, the arm
compresses the whole world corpus and replaces its library of fragments; the library is fixed
during an episode (a sleep phase between episodes, as in wake-sleep library learning).

- ``stitch``: stitch_core (Stitch, Bowers et al., POPL 2023) on the corpus written as chains,
  ``(a1 (a0 (a2 e)))`` for the presses 1, 0, 2, with at most ``stitch_iterations`` abstractions of
  arity at most 1 and no metavariable or body in function position, stopping earlier when no
  abstraction compresses the corpus. An abstraction is then a contiguous fragment followed by a
  hole (the rest of a sequence) or by the end marker, possibly through earlier abstractions, which
  are inlined. Stitch's default requires every abstraction to be used in at least two corpus
  sequences. Each abstraction whose inlined body is a chain becomes one fragment: its presses.
- ``bpe``: byte-pair encoding (``bpe_fragments``): at most ``bpe_merges`` merges of the most
  frequent adjacent token pair (counted over the corpus, every adjacent pair), ties broken by the
  smaller pair, stopping when no pair occurs twice; every merged token is a fragment.

Both caps are declared vocabulary sizes: with the measurement's corpus they bind (study CA's
protocol). The rows record each compression's effort (``ca_compress_effort``: the abstractions
Stitch returned or the merges made) next to the cap (``ca_compress_cap``).

Fragments shorter than 2 or longer than the longest public sequence length are dropped, and repeats
are kept once, in the order found.

Arms
----
- ``promote_stitch``, ``promote_bpe``: study L's ``promote`` with the library's fragments in place of
  the known concepts: a search at level l first tests the concatenations of ordered pairs of distinct
  fragments whose lengths add up to the level's length, then primitive windows. Memory, evidence,
  replay and gating are the shared engine's, so discoveries are still remembered and replayed; only
  the atoms of new hypotheses come from compression instead of from the discovery events.
- ``enum_prim``, ``enum_atoms``, ``enum_frag``: the enumerating rule learner of the layer factor.
  Before every search it orders the open windows of the level by their description length, the
  fewest tokens whose concatenation spells the window, and tests the windows of the smallest value
  (pooled, by the engine's walk). Tokens: the primitives at layer 0 (``enum_prim``), plus the known
  concepts, the discovered atoms, at layer 1 (``enum_atoms``), plus the library's Stitch fragments
  at layer 2 (``enum_frag``). At layer 0 every window has the description length of its length, so
  the learner is study L's ``remember`` (tested).
- ``oracle_library``: study L's anchor (every concept supplied by the runner), which also reports
  the description lengths of the world's composites (below) and the world's concept sequences; its
  actions are L's (tested).
- The comparison arms of study L (``remember``, ``promote``, ``sham_promote``, ``random``) and
  study U's ``pooled`` (the same agent as ``remember`` in every world) are built unchanged.

Description lengths of a world (privileged, reported by ``oracle_library``)
---------------------------------------------------------------------------
For every concept c of level >= 2 (a composite: compositions and flat concepts alike), from the links
the runner supplies to the oracle arm: layer 0, the length of its sequence; layer 1, the fewest
tokens over the primitives and the sequences of the concepts of lower levels; layer 2, the same with
the fragments that Stitch (same settings) finds in the world's own concept sequences added, except a
fragment equal to c's sequence: Stitch's greedy result on the world's own sequences, each seen once.
A fourth value replaces Stitch's fragments by the parent residues (``parent_residues``): for every
concept whose sequence is one parent's sequence followed or preceded by a part as long as the other
parent's, that part when it is no concept's sequence. In a latent world at reuse depth 2 the residues
are exactly the hidden fragments (tested); elsewhere a flat concept that starts or ends with a
parent's sequence by chance adds a residue.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .explorers import Explorer, make_explorer
from .layered import Promote, make_layered

DEFAULT_PARAMS = {"corpus": "tests", "stitch_iterations": 50, "bpe_merges": 50}
CORPORA = ("firings", "tests", "episodes")
REPORTING_ARMS: tuple[str, ...] = ()  # oracle_library, random and reference are the runner's
COMPRESSORS = ("stitch", "bpe")


class CompressionError(RuntimeError):
    """Stitch's backend failed (a panic); the arms record it and go on with an empty library."""


def check_params(params: dict | None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    extra = set(p) - set(DEFAULT_PARAMS)
    if extra:
        raise KeyError(f"unknown study CA agent parameters {sorted(extra)}")
    if p["corpus"] not in CORPORA:
        raise ValueError(f"agent parameter corpus must be one of {CORPORA}")
    for name in ("stitch_iterations", "bpe_merges"):
        if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < 1:
            raise ValueError(f"agent parameter {name} must be an integer >= 1")
    return p


# ------------------------------------------------------------------ compressors
def encode(seq) -> str:
    """Presses as a chain: ``(a1 (a0 e))`` for 1, 0."""
    out = "e"
    for s in reversed(seq):
        out = f"(a{int(s)} {out})"
    return out


def parse_sexpr(text: str):
    """Nested lists of tokens."""
    toks = text.replace("(", " ( ").replace(")", " ) ").split()
    stack: list[list] = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0][0]


def chain_symbols(expr, bodies: dict, hole: tuple[int, ...] = ()) -> tuple[int, ...] | None:
    """The presses of a chain expression with earlier abstractions inlined, ``#0`` standing for the
    presses ``hole`` (nothing for an abstraction's own body); None when it is not a chain."""
    if expr == "#0":
        return hole
    if expr == "e":
        return ()
    if isinstance(expr, str):  # an arity-0 abstraction
        return chain_symbols(parse_sexpr(bodies[expr]), bodies) if expr in bodies else None
    if len(expr) != 2 or not isinstance(expr[0], str):
        return None
    head, rest = expr
    tail = chain_symbols(rest, bodies, hole)
    if tail is None:
        return None
    if head.startswith("a") and head[1:].isdigit():
        return (int(head[1:]),) + tail
    if head in bodies:  # an earlier arity-1 abstraction applied to the rest of the chain
        return chain_symbols(parse_sexpr(bodies[head]), bodies, tail)
    return None


def _keep(frags, max_len: int) -> list[tuple[int, ...]]:
    out = []
    for f in frags:
        if f is not None and 2 <= len(f) <= max_len and f not in out:
            out.append(tuple(f))
    return out


def stitch_fragments(corpus, iterations: int, max_len: int) -> list[tuple[int, ...]]:
    """Fragments of the abstractions Stitch finds in ``corpus`` (sequences of presses)."""
    return stitch_library(corpus, iterations, max_len)[0]


def stitch_library(corpus, iterations: int, max_len: int) -> tuple[list[tuple[int, ...]], int]:
    """(fragments, number of abstractions Stitch returned, at most ``iterations``)."""
    try:  # imported here, so the package imports without it (as study X's learner libraries)
        import stitch_core
    except ImportError as exc:
        raise ImportError("study CA's Stitch arms and description lengths need stitch-core 0.1.29") from exc
    programs = [encode(s) for s in corpus if len(s) >= 2]
    if len(programs) < 2:
        return [], 0
    try:
        res = stitch_core.compress(programs, iterations=iterations, max_arity=1,
                                   no_curried_metavars=True, no_curried_bodies=True)
    except stitch_core.StitchException as exc:
        raise CompressionError(str(exc)) from exc
    bodies = {a.name: a.body for a in res.abstractions}
    frags = _keep((chain_symbols(parse_sexpr(a.body), bodies) for a in res.abstractions), max_len)
    return frags, len(res.abstractions)


def bpe_fragments(corpus, merges: int, max_len: int) -> list[tuple[int, ...]]:
    """Byte-pair encoding of ``corpus``: the merged tokens, in merge order."""
    return bpe_library(corpus, merges, max_len)[0]


def bpe_library(corpus, merges: int, max_len: int) -> tuple[list[tuple[int, ...]], int]:
    """(fragments, number of merges made, at most ``merges``)."""
    seqs = [[(int(s),) for s in seq] for seq in corpus]
    out = []
    for _ in range(merges):
        counts = Counter(p for seq in seqs for p in zip(seq, seq[1:]))
        if not counts:
            break
        (a, b), n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if n < 2:
            break
        out.append(a + b)
        for i, seq in enumerate(seqs):
            merged, j = [], 0
            while j < len(seq):
                if j + 1 < len(seq) and seq[j] == a and seq[j + 1] == b:
                    merged.append(a + b)
                    j += 2
                else:
                    merged.append(seq[j])
                    j += 1
            seqs[i] = merged
    return _keep(out, max_len), len(out)


# ---------------------------------------------------------- description lengths
def description_lengths(table: np.ndarray, tokens) -> np.ndarray:
    """For every row of ``table`` (windows x positions) the fewest tokens whose concatenation spells
    it; single symbols are always tokens, so a window of length n has at most n."""
    rows, n = table.shape
    best = np.zeros((rows, n + 1), dtype=np.int32)
    toks = sorted({tuple(int(s) for s in t) for t in tokens if 2 <= len(t) <= n})
    for i in range(1, n + 1):
        best[:, i] = best[:, i - 1] + 1
        for t in toks:
            m = len(t)
            if m <= i:
                hit = (table[:, i - m:i] == np.array(t)).all(axis=1)
                best[hit, i] = np.minimum(best[hit, i], best[hit, i - m] + 1)
    return best[:, n]


def min_tokens(seq, tokens) -> int:
    """Description length of one sequence (``description_lengths`` of a single row)."""
    return int(description_lengths(np.array([tuple(seq)]), tokens)[0])


# ------------------------------------------------------------------------ agents
class PressLog:
    """The agent's own handle of the public environment, recording every action it takes and the
    concepts that the step reports as fired (public ``info``)."""

    def __init__(self, env):
        self.env = env
        self.steps: list[tuple[int, tuple[int, ...]]] = []

    @property
    def public_spec(self):
        return self.env.public_spec

    def reset(self, token):
        return self.env.reset(token)

    def step(self, action: int):
        out = self.env.step(action)
        self.steps.append((int(action), tuple(out[4]["fired"])))
        return out


class Recorder:
    """Per world: the corpus of the agent's own press sequences and the library compressed from it
    (module docstring, Corpus and Compressors). Mixed into an ``explorers.Explorer`` subclass."""

    compressor: str | None = None

    def configure(self, params: dict | None = None):
        self.params = check_params(params)
        self.corpus: dict[str, list[tuple[int, ...]]] = {}
        self.library: dict[str, dict[int, tuple[int, ...]]] = {}
        self.compressed_at: dict[str, int] = {}
        self.versions: Counter = Counter()
        self.effort: dict[str, int] = {}  # abstractions or merges of the compression behind the library
        self.world, self.max_len = "", 0
        self.bad0: dict = {}
        self.elim0: dict = {}
        self.counts: Counter = Counter()  # sequences added to the corpus in the current episode
        self.errors = 0
        return self

    def knowledge(self, spec):
        k = super().knowledge(spec)
        self.world, self.max_len = spec.world_key, max(spec.level_lengths)
        self.bad0 = {c: len(v) for c, v in k.bad.items()}
        self.elim0 = {lv: m.copy() for lv, m in k.elim.items()} if self.params["corpus"] == "tests" else {}
        corpus = self.corpus.setdefault(spec.world_key, [])
        if self.compressor and self.compressed_at.get(spec.world_key) != len(corpus):
            frags, self.effort[spec.world_key] = self.compress(corpus)
            self.library[spec.world_key] = dict(enumerate(frags))
            self.compressed_at[spec.world_key] = len(corpus)
            self.versions[spec.world_key] += 1
        return k

    def cap(self) -> int:
        """The compressor's declared vocabulary size (0 without a compressor)."""
        return {"stitch": self.params["stitch_iterations"], "bpe": self.params["bpe_merges"]}.get(
            self.compressor or "", 0)

    def compress(self, corpus) -> tuple[list[tuple[int, ...]], int]:
        """(fragments, effort): the abstractions Stitch returned or the merges made."""
        try:
            if self.compressor == "stitch":
                return stitch_library(corpus, self.params["stitch_iterations"], self.max_len)
            return bpe_library(corpus, self.params["bpe_merges"], self.max_len)
        except CompressionError:
            self.errors += 1  # reported in the rows; the library is then empty for the episode
            return [], 0

    def fragments(self) -> list[tuple[int, ...]]:
        return list(self.library.get(self.world, {}).values())

    def add(self, seq, outcome: str):
        if len(seq) >= 2:
            self.corpus[self.world].append(tuple(int(s) for s in seq))
            self.counts[outcome] += 1

    def record(self, log: PressLog, success: bool):
        """Adds the episode's press sequences to the world corpus (module docstring, Corpus)."""
        spec, k = log.public_spec, self.worlds[self.world]
        if self.params["corpus"] == "episodes":
            seq: list[int] = []
            for a, _ in log.steps + [(spec.wait_action, ())]:
                if a < spec.n_symbols:
                    seq.append(a)
                else:
                    self.add(seq, "success" if success else "failure")
                    seq = []
            return
        trace: list[int] = []
        for a, fired in log.steps:  # the environment's trace: primitives since the last wait or submit
            trace = trace + [a] if a < spec.n_symbols else []
            for c in fired:
                self.add(trace[-spec.length(c):], "success")
        for c, clauses in sorted(k.bad.items()):
            for _ in range(len(clauses) - self.bad0.get(c, 0)):
                self.add(k.known[c], "failure")
        for lv, before in sorted(self.elim0.items()):
            for code in np.flatnonzero(k.elim[lv] & ~before):
                self.add(k.digits(lv, int(code)), "failure")

    def run_episode(self, env, token, meter) -> dict:
        self.counts = Counter()
        log = PressLog(env)
        row = super().run_episode(log, token, meter)
        self.record(log, row["success"])
        lib = self.fragments()
        row.update({"ca_library": [list(f) for f in lib], "ca_library_size": len(lib),
                    "ca_corpus_size": len(self.corpus.get(self.world, [])),
                    "ca_corpus_success": self.counts["success"], "ca_corpus_failure": self.counts["failure"],
                    "ca_compress_errors": self.errors, "ca_compress_effort": self.effort.get(self.world, 0),
                    "ca_compress_cap": self.cap()})
        return row


class CompressedPromote(Recorder, Promote):
    """``promote`` whose atoms are the library's fragments (module docstring, Arms)."""

    def configure(self, params: dict | None = None, compressor: str = "stitch"):
        if compressor not in COMPRESSORS:
            raise ValueError(f"unknown compressor {compressor!r}")
        self.compressor = compressor
        self.masks: dict = {}
        return super().configure(params)

    def pair_mask(self, k, level: int) -> np.ndarray:
        """Windows that concatenate two distinct fragments (cached per library version)."""
        key = (self.world, level, self.versions[self.world])
        if key not in self.masks:
            n = k.lengths[level - 1]
            m = np.zeros(k.S ** n, dtype=bool)
            frags = self.fragments()
            for a in frags:
                for b in frags:
                    if a != b and len(a) + len(b) == n:
                        m[k.code(a + b)] = True
            self.masks = {kk: v for kk, v in self.masks.items() if kk[:2] != key[:2]}
            self.masks[key] = m
        return self.masks[key]

    def choose(self, k, ep, level, open_):
        macro = self.pair_mask(k, level) & open_
        if macro.any():
            return "search_macro", macro
        return Explorer.choose(self, k, ep, level, open_)


class Enumerator(Recorder, Explorer):
    """The enumerating rule learner at layer 0, 1 or 2 (module docstring, Arms)."""

    def configure(self, params: dict | None = None, layer: int = 0):
        if layer not in (0, 1, 2):
            raise ValueError("layer must be 0, 1 or 2")
        self.layer = layer
        self.compressor = "stitch" if layer == 2 else None
        self.dls: dict = {}
        return super().configure(params)

    def tokens(self, k) -> frozenset:
        toks = set(k.known.values()) if self.layer >= 1 else set()
        if self.layer >= 2:
            toks |= set(self.fragments())
        return frozenset(toks)

    def dl(self, k, level: int) -> np.ndarray:
        n = k.lengths[level - 1]
        toks = frozenset(t for t in self.tokens(k) if 2 <= len(t) <= n)
        key = (self.world, level)
        if key not in self.dls or self.dls[key][0] != toks:
            self.dls[key] = (toks, description_lengths(k.table[level], toks))
        return self.dls[key][1]

    def choose(self, k, ep, level, open_):
        base = self.primitive_mask(k, level, open_)
        if not base.any():
            return "search_primitive", base
        dl = self.dl(k, level)
        low = int(dl[base].min())
        kind = "search_macro" if low < k.lengths[level - 1] else "search_primitive"
        return kind, base & (dl == low)

    def run_episode(self, env, token, meter) -> dict:
        row = super().run_episode(env, token, meter)
        row["ca_layer"] = self.layer
        return row


class LayerOracle(Explorer):
    """Study L's ``oracle_library`` (every concept supplied by the runner) that also reports the
    description lengths of the world's composites at each layer (module docstring)."""

    def configure(self, params: dict | None = None):
        self.params = check_params(params)
        self.spec = None
        self.world_dl: dict[str, dict[int, list[int]]] = {}
        self.errors = 0
        return self

    def knowledge(self, spec):
        self.spec = spec
        return super().knowledge(spec)

    def supplied(self, spec) -> dict[int, tuple[int, ...]]:
        """The concept sequences the runner supplied for the world (privileged links)."""
        links = self._revealed.get(spec.world_key, {})
        return {c: tuple(int(s) for s in links[c][0]) for c in sorted(links)}

    def parents(self, spec) -> dict[int, tuple[int, ...]]:
        """The parents, in concatenation order, that the runner supplied with the links."""
        links = self._revealed.get(spec.world_key, {})
        return {c: tuple(links[c][1]) for c in sorted(links)}

    def layer_lengths(self, spec) -> dict[int, list[int]]:
        seqs = self.supplied(spec)
        if len(seqs) < len(spec.concept_levels):
            return {}
        lv = spec.concept_levels
        try:
            frags = stitch_fragments(list(seqs.values()), self.params["stitch_iterations"],
                                     max(spec.level_lengths))
        except CompressionError:
            self.errors += 1  # reported in the rows; layer 2 then equals layer 1 for the world
            frags = []
        residues = parent_residues(seqs, self.parents(spec))
        out = {}
        for c, s in seqs.items():
            if lv[c] >= 2:
                lower = {seqs[x] for x in seqs if lv[x] < lv[c]}
                out[c] = [len(s), min_tokens(s, lower), min_tokens(s, lower | {f for f in frags if f != s}),
                          min_tokens(s, lower | (residues - {s}))]
        return out

    def run_episode(self, env, token, meter) -> dict:
        row = super().run_episode(env, token, meter)
        spec = self.spec
        if spec.world_key not in self.world_dl:
            self.world_dl[spec.world_key] = self.layer_lengths(spec)
        dls = self.world_dl[spec.world_key]
        by_level: dict[str, list[float]] = {}
        for lvl in sorted({spec.concept_levels[c] for c in dls}):
            vals = [dls[c] for c in dls if spec.concept_levels[c] == lvl]
            by_level[str(lvl)] = [float(np.mean([v[i] for v in vals])) for i in range(4)]
        row.update({"ca_goal_dl": dls.get(spec.goal), "ca_world_dl": by_level,
                    "ca_compress_errors": self.errors,
                    "ca_world_sequences": [[spec.concept_levels[c], list(s)]
                                           for c, s in self.supplied(spec).items()]})
        return row


def parent_residues(seqs: dict, parents: dict) -> set[tuple[int, ...]]:
    """Parts of concept sequences that one parent's sequence leaves at the other parent's place,
    when they are no concept's sequence (module docstring, Description lengths)."""
    concepts, out = set(seqs.values()), set()
    for c, ps in parents.items():
        if len(ps) == 2:
            s, a, b = seqs[c], seqs[ps[0]], seqs[ps[1]]
            if len(s) == len(a) + len(b):
                for keeps, part in ((s[:len(a)] == a, s[len(a):]), (s[len(s) - len(b):] == b, s[:len(a)])):
                    if keeps and part not in concepts:
                        out.add(part)
    return out


# ------------------------------------------------------------------------ factory
L_ARMS = ("none", "remember", "promote", "sham_promote", "adaptive", "random")
ENUMERATORS = {"enum_prim": 0, "enum_atoms": 1, "enum_frag": 2}


def make_compression(name: str, rng: np.random.Generator, params: dict | None = None):
    """Study CA arms; study L's comparison arms and study U's ``pooled`` are built unchanged."""
    p = check_params(params)
    if name in ("promote_stitch", "promote_bpe"):
        return CompressedPromote(rng).configure(p, compressor=name.split("_")[1])
    if name in ENUMERATORS:
        return Enumerator(rng).configure(p, layer=ENUMERATORS[name])
    if name == "oracle_library":
        return LayerOracle(rng).configure(p)
    if name == "pooled":
        return make_explorer(name, rng)
    if name in L_ARMS:
        return make_layered(name, rng)
    raise KeyError(f"unknown study CA arm {name!r}")


# ------------------------------------------------------------------ row summaries
# nested {variant: {arm: ...}} like the analysis tables, so the analysis stores them as JSON
LAYER_ARMS = {"remember": 0, "enum_atoms": 1, "enum_frag": 2}


def _ranks(x) -> np.ndarray:
    """Ranks from 0, ties sharing the mean of their positions."""
    x = np.asarray(x, dtype=float)
    ranks = np.empty(len(x))
    ranks[np.argsort(x, kind="mergesort")] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        ranks[x == v] = ranks[x == v].mean()
    return ranks


def spearman(x, y) -> float | None:
    """Spearman rank correlation; None with fewer than three points or a constant variable."""
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _world_rows(rows: list[dict], arms=None) -> dict:
    """{(variant, seed, world key, arm): the rows of that world, in episode order}."""
    out: dict = {}
    for r in rows:
        if arms is None or r.get("arm") in arms:
            out.setdefault((r.get("variant"), r.get("seed", 0), r["world_key"], r.get("arm")), []).append(r)
    return {key: sorted(rs, key=lambda r: r["world_episode"]) for key, rs in sorted(out.items(), key=str)}


def _costs(rs: list[dict]) -> dict:
    """Actions to the first success in the world (censored at the world's total), and the mean actions
    per episode over all episodes and over the late half."""
    steps = [r["primitive_length"] for r in rs]
    first = next((i for i, r in enumerate(rs) if r["success"]), None)
    late_from = (max(r["world_episode"] for r in rs) + 1) // 2
    late = [r["primitive_length"] for r in rs if r["world_episode"] >= late_from]
    return {"first_success_steps": float(sum(steps if first is None else steps[:first + 1])),
            "steps_all": float(np.mean(steps)), "steps_late": float(np.mean(late))}


def _oracle(rows: list[dict], key: str) -> dict:
    """{(variant, seed, world key): the oracle row field ``key`` of that world}."""
    return {wk[:3]: rs[0][key] for wk, rs in _world_rows(rows, {"oracle_library"}).items() if rs[0].get(key)}


def layer_summary(rows: list[dict]) -> dict:
    """The layer factor: per variant the mean description lengths of the worlds' composites by level
    and layer (``world_dl``, from the oracle rows); per variant and layer learner (``remember`` at
    layer 0, ``enum_atoms`` at 1, ``enum_frag`` at 2) the mean, over worlds, of the description length
    of the world's composites of the top level at the learner's layer and of its costs
    (``learners``); every such world (``points``); and Spearman rank correlations between that
    description length and each cost over every world and variant, per learner and pooled."""
    oracle = _oracle(rows, "ca_world_dl")
    per_variant: dict = {}
    for (variant, _, _), dl in oracle.items():
        per_variant.setdefault(variant, []).append(dl)
    world_dl = {v: {lvl: [float(np.mean([d[lvl][i] for d in dls]))
                          for i in range(min(len(d[lvl]) for d in dls))]
                    for lvl in sorted(dls[0])} for v, dls in sorted(per_variant.items())}
    points = []
    for (variant, seed, wk, arm), rs in _world_rows(rows, set(LAYER_ARMS)).items():
        dl = oracle.get((variant, seed, wk))
        if dl:
            top = max(dl, key=int)
            points.append({"variant": variant, "seed": seed, "world_key": wk, "arm": arm,
                           "layer": LAYER_ARMS[arm], "dl": dl[top][LAYER_ARMS[arm]], **_costs(rs)})
    learners: dict = {}
    for p in points:
        learners.setdefault(p["variant"], {}).setdefault(p["arm"], []).append(p)
    learners = {v: {arm: {"layer": ps[0]["layer"], "n_worlds": len(ps),
                          **{k: float(np.mean([p[k] for p in ps]))
                             for k in ("dl", "first_success_steps", "steps_all", "steps_late")}}
                    for arm, ps in sorted(arms.items())} for v, arms in sorted(learners.items())}
    corr = {}
    groups = [*((arm, [p for p in points if p["arm"] == arm]) for arm in LAYER_ARMS), ("pooled", points)]
    for name, ps in groups:
        corr[name] = {"n": len(ps), **{k: spearman([p["dl"] for p in ps], [p[k] for p in ps])
                                       for k in ("first_success_steps", "steps_all", "steps_late")}}
    return {"world_dl": world_dl, "learners": learners, "spearman": corr, "points": points}


def _part_of(seq: tuple, frag: tuple) -> bool:
    n = len(frag)
    return any(seq[i:i + n] == frag for i in range(len(seq) - n + 1))


def library_summary(rows: list[dict]) -> dict:
    """{variant: {arm: ...}} for the arms that keep a library: means over worlds of the library at
    the end of the world (its size; the shares of its fragments that are one of the world's concept
    sequences, that are a contiguous part of at least two of them without being one, and the rest),
    of the corpus size, the largest count of compression errors, and the share of episodes whose
    library came from a compression that reached the compressor's cap (``cap_share``; episodes
    after a compression only)."""
    seqs = {key: [tuple(s) for _, s in val] for key, val in _oracle(rows, "ca_world_sequences").items()}
    groups: dict = {}
    for (variant, seed, wk, arm), rs in _world_rows(rows).items():
        last = rs[-1]
        if "ca_library" not in last:
            continue
        lib = [tuple(f) for f in last["ca_library"]]
        entry = {"size": len(lib), "corpus": last["ca_corpus_size"], "errors": last["ca_compress_errors"]}
        world = seqs.get((variant, seed, wk))
        if world is not None and lib:
            concept = sum(f in world for f in lib)
            shared = sum(f not in world and sum(_part_of(s, f) for s in world) >= 2 for f in lib)
            entry.update({"concept_share": concept / len(lib), "shared_share": shared / len(lib),
                          "other_share": (len(lib) - concept - shared) / len(lib)})
        done = [r for r in rs if r.get("ca_compress_effort", 0) > 0]
        if done and last.get("ca_compress_cap"):
            at_cap = sum(r["ca_compress_effort"] >= r["ca_compress_cap"] for r in done)
            entry["cap_share"] = at_cap / len(done)
        groups.setdefault(variant, {}).setdefault(arm, []).append(entry)
    out: dict = {}
    for variant, arms in sorted(groups.items()):
        for arm, es in sorted(arms.items()):
            keys = [k for k in ("size", "corpus", "concept_share", "shared_share", "other_share", "cap_share")
                    if any(k in e for e in es)]
            out.setdefault(variant, {})[arm] = {
                "n_worlds": len(es), "errors": max(e["errors"] for e in es),
                **{k: float(np.mean([e[k] for e in es if k in e])) for k in keys}}
    return out


ROW_SUMMARIES = {"layers": layer_summary, "libraries": library_summary}
