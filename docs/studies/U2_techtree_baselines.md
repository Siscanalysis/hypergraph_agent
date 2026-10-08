# Study U2: generic exploration baselines on TechTree

Status: draft protocol, not frozen. No measurement run has been made. Development
runs used only the dev allocation and the namespaces `techtree_u2_dev` and
`techtree_u2l_dev`; their outcomes (two worlds in one cell, one in another) were
seen before this draft was written and are reported under Implementation.

## Question and claim

Question: study U validated TechTree as a test of discovery cost with two
elimination explorers, a random agent and an oracle, and study L compared ways to
represent discoveries with the same engine. How do standard, generic exploration
methods compare with the repository's explorers on the same worlds, and where does
each family break down?

U2 adds hand-designed, non-learning baselines from the planning and exploration
literature: width-based novelty search IW(1) and IW(2), flat macro expansion
(every sequence up to the public composite length listed as an action, sampled
uniformly or enumerated) and an archive explorer in the spirit of Go-Explore.
Learning agents are added by a separate study (X). The primary claim is that the
repository's `pooled` explorer needs fewer actions to the first success than every
one of these baselines in U's primary cell (k = 3, no signal, announced unlocks).
A pass is expected: the baselines lack pooling, hypothesis elimination and the
explorers' declared priors. The informative parts are the sizes, and where each
family breaks down: IW(k) at composites longer than k + 1, flat macro expansion
through the size of its action set, random exploration from an archive through
the waiting time of a random stream. U2 says nothing about learning agents.

## Relation to earlier work

docs/PRIOR_ART.md Section 8.3 lists the neighbours this study implements as arms:
width-based novelty search (C11; Lipovetzky and Geffner 2012 define IW(k) and its
serialized version SIW), macro-operators (C14) and the combination lock of
rich-observation RL (C10), which is what a blind search faces here. The archive
explorer follows the return-then-explore scheme of Go-Explore (Ecoffet et al.,
not in the source registry of PRIOR_ART; to be added with a verified entry before
any manuscript). Study U (docs/studies/U_composites_unlocks.md) supplies the U
setting, the `pooled` arm and the oracle; its limits named width-based novelty
search and flat macro expansion as untested. Study L
(docs/studies/L_layered_discovery.md) supplies the phase L2 setting and the
`remember` and `promote` arms.

## Protocol (to be frozen)

### Settings

TechTree (`src/hypergraph_agent/techtree/`), the same worlds as U and L in fresh
namespaces:

- U setting (namespace `techtree_u2`, never used before): P = 4 base primitives,
  R = 3 slots, one key and one entry composite of length k, the goal always the
  entry, 100 actions per episode, 12 episodes per world (the U protocol). Cells:
  k2-off-ann, k2-off-sil, k3-off-ann, k3-off-sil, k3-on-ann, k3-on-sil (k = 2 or
  3; signal off or on; announced or silent unlocks).
- L setting (namespace `techtree_u2l`, never used before): the world of study L
  phase L2, with full discovery (nothing supplied): 3 base primitives, three
  levels of three concepts with lengths 2, 4 and 6, a uniform level-3 goal, 100
  actions per episode, 12 episodes per world. Cells l-d0 and l-d2 (reuse depth 0
  and 2).

The k = 2 signal cells are not run: there every arm other than `pooled` acts
identically with and without the signal (below), so they hold no contrast.

### Arms

| Arm | Kind | Description |
|---|---|---|
| `pooled` | adaptive | U's elimination explorer (comparison, U setting). |
| `remember`, `promote` | adaptive | L's arms (comparison, L setting). |
| `iw1`, `iw2` | adaptive | Width-based novelty search IW(1), IW(2). |
| `macro_uniform` | adaptive | Flat macro expansion, a uniform draw per decision. |
| `macro_sys` | adaptive | Flat macro expansion, each macro once per pass. |
| `archive` | adaptive | Return to the most promising held set, explore randomly. |
| `random` | reporting | Uniform available primitives, no memory (anchor). |
| `oracle`, `oracle_library` | reporting, privileged | Every composite supplied (anchors). |
| `reference` | reporting, privileged | Replays the evaluator's shortest plan. |

### Definitions of the baselines

All five baselines (`techtree/baselines.py`) are the shared engine of
`techtree/explorers.py` with its search step replaced. They keep the engine's
memory and evidence per world (exact discoveries, candidate preconditions,
failure clauses, which key announced which slot), its gating (before searching,
the agent holds the known concepts the engine would hold) and its replay planner,
and they replay the goal once it is known. They hold only the public facade and
pass the same source check as the other agent modules. Every action goes through
the metered episode, so every press is charged, including the presses that return
to a search node and every press of a macro. The *context* of a search is the
held set; within an episode it only grows, so a firing ends the current search
and the next one starts in the new context (a later episode resumes a context's
search when the gating leads back to the same held set). The *actions* of a
context are the primitives available in it (announced: base symbols and active
slots; silent: all seven symbols), with no prior over slots.

**IW(1), IW(2).** A node is a primitive path from the context root (held set,
empty trace). Its state is the held set and the last L presses since the last
clear, L the public maximal sequence length. Its atoms are the values of the
environment's state variables the agent observes or reconstructs: each held
concept, each available slot (announced), the progress flag (signal on) and, for
j = 1..L, the symbol at position j from the end of the trace. Breadth-first
search generates the children of a node in a successor order drawn at random per
pass; a generated node is novel if it makes some tuple of at most w atoms true for
the first time in the pass (w = 1 or 2), and only novel nodes are expanded. Every
generated node is executed, so a composite fires when its node is generated, which
needs only its parent to be novel: one pass of IW(w) tests every sequence of up to
w + 1 presses from the root. Without a simulator a node is reached along the
shortest path the public trace semantics allow: a node shorter than L needs the
exact trace since a clear (`wait` and its path, or the rest of its path when the
current trace is a prefix of it); a longer node is reached by the shortest suffix
of its last L symbols that completes the current trace (silent: only presses made
under the current held set are reused). Every press on the way tests windows like
any other. An interrupted node goes back to the front of its queue. Each context
keeps its own tree, as in serialized IW. When a pass ends without the goal, a new
pass starts with an empty novelty table and a new random order (randomized
restarts), which keeps the search going where one pass is incomplete.

**Flat macro expansion.** Every sequence of 1..B actions of the context, B = L, is
a macro action; a macro presses its symbols without clearing the trace, and a
firing interrupts it like any other chunk. `macro_uniform` draws a macro uniformly
from this expanded action set at every search decision, without memory of tried
macros. Such a draw picks a length in proportion to its number of sequences and
then a uniform sequence, so the presses form an i.i.d. uniform stream over the
context's actions, exactly as uniformly random primitives do (tested): listing
every sequence as an action gives a blind sampler nothing. `macro_sys` executes
each macro of its context once per pass, shortest first, in a random order within
each length; an interrupted macro goes back; an exhausted pass starts again with
new orders. Its pass holds A + A^2 + ... + A^B macros for A actions, all shorter
ones before any of length B: the action-space blow-up.

**Archive.** Cells are held sets. The archive of a world records every cell the
agent has been in and how often each was selected. At its first search decision
of an episode the agent selects the most promising cell: the most held concepts
(in TechTree holding a concept never prevents another from firing), ties by
fewest selections. It returns there by replaying known sequences with the shared
planner, then presses uniformly random available primitives until the episode
ends; new cells found on the way are added. It does not read the signal.

Agent parameters are declared in the configs (`agent: {trace_atoms: max_length,
macro_bound: max_length}`, that is L and B equal to the public maximal length) and
enter every run hash.

Identities, tested and used to drop runs that cannot differ: `macro_uniform`,
`macro_sys` and `archive` never read the signal; `iw2` acts identically with and
without it for k <= 3 (its breadth-first order up to depth 3 does not depend on
novelty, and every composite fires within depth 3); at k = 2, `iw1` and `iw2`
act identically (both test every 2-press sequence before going deeper). Hence
`iw1` runs at k = 2 only in k2-off-ann (as a check), and only `pooled` and `iw1`
run in the signal cells.

### Cells, worlds and seeds

The five baselines are `iw1`, `iw2`, `macro_uniform`, `macro_sys` and `archive`.

| Cell | Arms |
|---|---|
| k2-off-ann, k3-off-ann | pooled, the five baselines, random, oracle, reference |
| k2-off-sil | pooled, the baselines except iw1, random |
| k3-off-sil | pooled, the five baselines, random |
| k3-on-ann, k3-on-sil | pooled, iw1 |
| l-d0, l-d2 | remember, promote, the five baselines, random, oracle_library, reference |

Seeds 0-7, 4 worlds per seed: 32 worlds per cell, 12 episodes each. As in U, the
worlds of one k (and, in the L setting, the coupled worlds of the two depths) are
shared across cells, and every arm of a seed starts from the same agent seed and
per-world preference seed (common random numbers).

### Endpoints

As in U and L, per world: `first_success_steps` (actions up to and including the
first successful episode; censored at the stream total, 1,200, when no episode
succeeds; censored worlds are counted and reported) and `steps_late` (mean actions
per episode over episodes 6-11, failures at their full budget), with success
rates, cost ratios to the reference and discovery curves. Rows of the baselines
also record IW passes, generated, novel and pruned nodes, macros executed and
passes, and archive cell counts and selected cell sizes.

### Predictions (written before any measurement run)

From coverage counts. `pooled`, `random`, the oracle and the L arms are taken from
the U and L results (fresh worlds, so they are predictions too). For the blind
samplers (`macro_uniform` and `archive`, identical in distribution in the U
setting, where both hold the key cell), a specific k-gram in an i.i.d. stream over
A symbols takes about A^k presses: 4^k for the key and 5^k (silent: 7^k for both)
for the entry. `macro_sys` first runs all shorter macros, then each length-k macro
tests its own window and two random-like windows across its boundaries, which
finds a target after about 0.28 N macros of k presses for N = A^k (0.37 N of 2
presses at k = 2), less when the shorter macros already hit it. IW(2) and, at
k = 2, IW(1) are complete breadth-first searches to depth k whose navigation
costs about 2-3 presses per node and also tests windows; at k = 3 one pass of
IW(1) tests only the 3-press sequences whose middle press is the pass's first
successor (1/A of them) and the windows on its way, so it needs several passes.

Mean actions to the first success per world, U setting (about; c: censored worlds
expected):

| Cell | pooled | iw2 | iw1 | macro_sys | macro_uniform, archive | random | oracle |
|---|---|---|---|---|---|---|---|
| k2-off-ann | 18 | 50 | = iw2 | 42 | 45 | 50 | 4.9 |
| k2-off-sil | 28 | 95 | (= iw2) | 95 | 105 | 150 | |
| k3-off-ann | 67 | 270 | 800, c | 190 | 200 | 350 | 6.8 |
| k3-off-sil | 125 | 650 | most c | 650 | 720, c | 1,035, most c | |
| k3-on-ann | 31 | (= off) | at most off | (= off) | (= off) | (= off) | |
| k3-on-sil | 38 | (= off) | at most off | (= off) | (= off) | (= off) | |

Ratios to `pooled` at k3-off-ann: about 12 (iw1), 4 (iw2), 2.8 (macro_sys) and
3 (`macro_uniform`, `archive`). Late-half costs equal the reference for every arm
that has discovered both composites; `iw1` and the k3-off-sil samplers will not
have in some worlds.

L setting, mean actions per episode in the late half (L2 values for the L arms):

| Depth | remember | promote | iw1, iw2 | macro_sys | macro_uniform | archive | random | oracle library |
|---|---|---|---|---|---|---|---|---|
| d0 | 39 | 45 | 90-100 | 55-75 | 55-75 | 55-75 | 97 | 14 |
| d2 | 28 | 11 | 70-100 | 45-75 | 45-75 | 25-55 | 97 | 8 |

Reasons: one IW(2) pass is complete only to length 3, while the L concepts have
lengths 4 and 6 (and the passes there are long); the random streams need about
3^6 = 729 presses per specific level-3 sequence against a systematic walk of about
one new window per press; the archive holds every known concept, so at depth 2 its
return replays compose level-3 sequences incidentally.

### Primary claim and pass criterion

P1, in cell k3-off-ann, metric `first_success_steps`: `pooled` needs fewer actions
than each generic baseline. P1 is an intersection-union test of five paired
contrasts, `iw1 - pooled` (P1a), `iw2 - pooled` (P1b), `macro_uniform - pooled`
(P1c), `macro_sys - pooled` (P1d) and `archive - pooled` (P1e); each holds when its
95% world-cluster bootstrap CI lies above 0, and P1 holds iff all five hold. An
intersection-union test needs no multiplicity correction: the probability of
declaring P1 when some component is false is at most 0.05. A censored world counts
at the stream total, a lower bound of the baseline's cost, so censoring can only
make a component harder to pass. The single contrast `iw2 - pooled` was the
initial proposal; the intersection-union form avoids choosing the competitor.

U2 passes iff P1 and the validity checks hold (`analysis.verdict`, rule `all`):

- V1: the reference solves every task it plays (both settings);
- V2, V3: the oracle solves every task and never uses fewer actions than the
  reference (U setting);
- V4, V5: the same for the oracle library (L setting).

Falsification: some generic baseline is not shown to need more actions than
`pooled` at k = 3 (a component of P1 fails).

### Secondary contrasts (reported, no verdict on U2)

- S1, width: `iw1 - iw2` in k3-off-ann and k3-off-sil (expect > 0: one pass of
  IW(1) covers a seventh to a quarter of the 3-press sequences, one of IW(2) all).
- S2, implementation check: `iw1` equivalent to `iw2` in k2-off-ann (90% CI within
  +-1 action; identical by construction).
- S3, tree execution: `iw2 - pooled` grows from k2-off-ann to k3-off-ann
  (interaction > 0).
- S4, action-space blow-up: `macro_sys - pooled` grows from k = 2 to k = 3
  (off-ann, interaction > 0); `macro_sys` silent minus announced at k = 3 (> 0).
- S5, no gain for a blind sampler: `macro_uniform - archive` in k3-off-ann and
  k3-off-sil, expect the CI not entirely below 0 (the two are identical in
  distribution here; this checks the absence of a gain and claims no equivalence).
- S6, memory of the unlock chain: `random - archive` on `first_success_steps` in
  k3-off-ann and on `steps_late` in l-d2 (expect > 0).
- S7, P1 in the silent cell: the five components in k3-off-sil (expect > 0).
- S8, L setting: `remember` below the best baseline in l-d0 and `promote` below the
  best baseline in l-d2 on `steps_late` (test `within_margin` with margin 0: the
  upper 95% bound of the arm minus the baseline with the lowest mean is at most 0).
- S9, signal: `pooled` on minus off at k = 3, announced (expect < 0, U's S6 on
  fresh worlds); `iw1` on minus off (expect the CI not entirely above 0).
- Descriptive: tables per cell, censoring counts, ratios to `pooled`, discovery
  curves, IW passes and node counts, macros per pass, archive cell sizes; cell
  k2-off-sil and the late halves are read by no test.

### Statistics

As in U and L: worlds are the units; contrasts are paired by world (interactions
across k pair worlds by index, as in U's S3); 10,000 percentile bootstrap
resamples of worlds with a fixed seed (20261008), 95% intervals (90% for
equivalence); two-sided exact sign tests over worlds reported alongside. No
multiplicity correction for the secondaries, which carry no verdict.

### Incomplete or repeated runs

The rules of U and L apply unchanged (shared runner and analysis). Every (cell,
arm, seed) must have exactly one complete run; a cell with a missing or incomplete
run is flagged and every test that reads it has no verdict, and U2 has no verdict
if a run read by P1 or V1-V5 is incomplete. A second run of the same (cell, arm,
seed) is an error unless the earlier one is incomplete and the later one has the
same run hash and code (a deterministic rerun, used and listed). The runner refuses
to start a measurement config whose runs exist complete, unless `--resume`.

### Limits stated in advance

- The baselines are hand-written adaptations; others could fare differently. The
  choices that matter: IW's atoms (trace positions up to L), execution by the
  shortest path, per-context trees and randomized restarts; the macro bound
  B = L; archive cells as held sets with random exploration.
- The repository's explorers carry declared priors (slot windows first) that match
  the generator; the baselines have none. P1 measures the total effect of pooling,
  elimination and priors against generic search; a prior-free `pooled` is not run.
- No baseline uses the signal as a heuristic (best-first width search would); in
  the signal cells only `pooled` and `iw1` can differ from their off values.
- IW is run without a simulator, so returning to a node costs actions; with state
  restoring, its cost would be counted in simulator calls instead. U2 counts
  environment actions, the quantity of U and L.
- All baselines keep the engine's memory, gating and replay; without them (as a
  learner from scratch) they would cost more.
- A toy domain; one primary cell; no learning agents.

### Budget

Ledger `runs/U2-ledger.json` (created by the first development run): allocations
`dev` 20,000 and `u2_main` 1,574,400 adaptive; adaptive cap 1,594,400; reporting
cap 557,600 (shared by development and measurement reporting runs). From the dry
run of `configs/unlock2/u2_main.yaml` (440 runs):

| Item | Runs | Per-run cap | Cap total | Expected use |
|---|---|---|---|---|
| U setting, adaptive arms | 216 | 4,800 | 1,036,800 | about 350,000 |
| U setting, reporting arms | 64 | 4,800 | 307,200 | about 120,000 |
| L setting, adaptive arms | 112 | 4,800 | 537,600 | about 380,000 |
| L setting, reporting arms | 48 | 4,800 | 230,400 | about 90,000 |
| Development (used, exact) | 19 | | | 5,942 adaptive + 1,945 reporting |

A run's cap is the sum of its 48 episode budgets, so no run is cut short. Expected
runtime on one CPU core: about 5 minutes (development: 1-11 ms per episode in the
U setting, 5-23 ms in the L setting, the baselines and `remember` at the top of
these ranges, plus about 0.1 s of bookkeeping per run).

Commands (after the freeze):
`python -m hypergraph_agent.techtree --config configs/unlock2/u2_main.yaml --allow-measurement`,
then `python -m hypergraph_agent.techtree.analysis --config configs/unlock2/u2_main.yaml`.
A measurement config runs whole; `--resume` repeats only missing or incomplete
runs.

### Stopping rules

The measurement config is run once, completely, after the freeze, from a clean
checkout of the freeze commit (DECISIONS S1). No interim analysis and no change of
parameters, arms or endpoints after any measurement row exists. If a validity
check fails, U2 fails; any repeat other than a deterministic rerun of an
incomplete run needs a dated amendment.

## Implementation

- `techtree/baselines.py`: the five baselines and the study factory
  `make_baseline(arm, rng, params)`, which also builds U's and L's arms unchanged
  (registered for study U2 in `techtree/study.py`).
- Configs: `configs/unlock2/u2_main.yaml` (measurement draft),
  `configs/unlock2/u2_dev.yaml` (development).
- Tests: `tests/test_techtree_baselines.py`: novelty pruning in one IW pass (only
  novel nodes are expanded; IW(1) tests the 3-press sequences with the first
  successor in the middle, IW(2) all of them), the shortest-path execution and the
  silent-mode reuse rule, the signal and k = 2 identities above, enumeration and
  sampling of macros (including the i.i.d. stream), the archive's cell selection
  and return, that every press is charged, soundness of the shared evidence in the
  U and L settings, the factory and common random numbers, the source check, the
  configs and an end-to-end run with analysis.

### Development record (calibration, not results)

Development config: 1 seed; k3-off-ann with 2 worlds and l-d2 with 1 world, 6
episodes each; 19 runs, all complete, V1-V5 held. It used 5,942 adaptive and 1,945
reporting interactions of the U2 ledger. Actions to the first success at
k3-off-ann (two worlds, stream total 600): `pooled` 59, `iw1` 480 (one world
censored), `iw2` 318, `macro_sys` 134, `macro_uniform` 44, `archive` 189, `random`
280, oracle 7. With two worlds the waiting times of the blind samplers dominate
(`macro_uniform` was below `pooled` there); these are not estimates. In the L world
at depth 2, `promote` and `remember` solved their late episodes, `macro_sys` two
of three and `archive` one of three, while `iw1`, `iw2`, `macro_uniform` and
`random` solved none. Before these runs, ad hoc checks during implementation
played two worlds per U cell (k = 2 and 3, both visibilities) and per L depth on
the `unit` namespace with in-memory meters (36,740 interactions, not charged to
any ledger). They showed that IW(1) and IW(2) act identically at k = 2, which
corrected an initial expectation that IW(1) would fail beyond k = 1, and the
macro predictions were corrected analytically after the development runs (a
uniform macro draw yields an i.i.d. stream). The baseline code changed after the
development runs only in docstrings and parameter validation, with no change of
behaviour at the declared parameters.

## Results

No measurement run has been made.

## Verdict and limits

No verdict yet.
