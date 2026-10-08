# Study L: discoveries used as atoms of new hypotheses

Status: protocol frozen on 2026-10-08 by the commit that adds this file
(73e5602), before any measurement run. The measurement runs executed that
commit from a clean checkout and are reported in the Results section.
Development runs used only the dev allocation and the namespace `techtree_dev`.
An independent review ran the full measurement design once on the scratch
namespace `checker_dev` with in-memory ledgers (no measurement world touched);
its outcomes were seen before the precedence rule, the conditions of the
verdict, the per-depth arm lists and the fresh namespace of phase L2 were
fixed.

## Question and claim

Question: if knowledge is a graph and a discovery is a new link between parts
of it, does it help to use that discovery as an atom of new hypotheses (a node
of a higher layer that further search composes), beyond remembering it as a
literal link that is replayed when the same need recurs? And does any benefit
depend on how often the world's discoveries are built from earlier ones (reuse
depth)?

The topological framing alone is not testable (a rule list with the same
content carries the same information), so L tests an algorithm that uses it.
Agents share one engine, memory, evidence and planner and differ only in the
hypothesis sets offered to a new search. A benefit at reuse depth 2 is expected
by construction: the world's discoveries are then compositions of earlier ones,
and `promote` searches compositions first. The depth-0 control is also by
construction: flat sequences are never the concatenation of two concepts (rule
(ii) of the generator), so every macro test at depth 0 is a guaranteed miss.
The informative parts are the size of the effect, the sham control (as many
promoted nodes, useless ones), the dose-response over depth, how much
`remember` already gains by incidental composition, and whether the adaptive
arm detects the world kind.

## Relation to earlier work

docs/PRIOR_ART.md Section 8.4 positions this study. The closest mechanism is
substructure discovery by compression (K12), where replacing discovered
substructures yields a hierarchical description of a graph; promoting
recurring compositions to new primitives is library learning (K6), chunking
(K9) and macro-operators (C14), whose known cost is the utility problem (K11).
Novelty curves of the adjacent possible (K2, K3) arise for any agent that
unlocks monotonically, which is why the discovery curves carry a random null.
Only the controlled contrast of promoting against remembering, with sham
promotion and reuse depth, is a potential empirical contribution.

## Protocol (frozen)

### Environment

TechTree (`src/hypergraph_agent/techtree/`) with three levels of three concepts:

- 3 base primitives; no extra slots, no unlocks, no intermediate signal.
- Level-1 concepts have secret combos of 2 primitives. A concept of level l >= 2
  has two distinct hidden parents, at levels l-1 and 1 (declared public rule),
  and can fire only while both are held. Public lengths: 2, 4 and 6 (a
  concept's length is the sum of its parents' lengths).
- Firing: after every primitive press, repeatedly within the step, a concept
  that is not held, whose parents are held and whose sequence equals the trace
  suffix becomes held. `wait` and `submit` clear the trace.
- Factor reuse depth d: concepts of levels 2..d+1 are compositions,
  s(x) = s(first parent) + s(second parent) in a hidden order; the others are
  flat: uniform sequences of the same length that are not the concatenation of
  any two distinct concepts (rule (ii)). d = 0: nothing is built from earlier
  discoveries; d = 1: level 2 is; d = 2: levels 2 and 3 are (level 3 from a
  level-2 discovery). Keys, parents, orders and level-1 combos are drawn
  independently of d, so the three depths use the same worlds except for the
  sequences above level 1 (coupled worlds, paired across depths).
- Tasks: the goal is a level-3 concept, uniform; budget 100 actions per
  episode; 12 consecutive episodes per world. Mean reference lengths
  (evaluator-only exact search, computed offline on 2,400 scratch tasks per
  depth): 12.5 at d = 0, 10.5 at d = 1, 7 at d = 2.
- Phase L1 (oracle discoveries at level 1): the three level-1 links (sequence
  and parents) are supplied identically to every arm at the start of every
  episode. Only the level-1 input is held fixed; level-2 and level-3
  discoveries are each arm's own. Phase L2: nothing is supplied; every arm
  discovers level 1 itself (the real pipeline), on fresh worlds.

### Arms

| Arm | Kind | Hypotheses for a new search |
|---|---|---|
| `none` | adaptive | No memory between episodes (supplied links are re-supplied each episode). L1 only, at d = 0 and 2. |
| `remember` | adaptive | Discoveries are literal links, replayed when needed (gating, goal), never used inside a new hypothesis: primitive windows only. |
| `promote` | adaptive | Discoveries are atoms of new hypotheses: first the concatenations of ordered pairs of distinct known concepts whose lengths add up to the level's length, then primitive windows. |
| `sham_promote` | adaptive | As `promote`, with as many promoted nodes, each a random base sequence of the same length as a real known concept (useless as atoms). |
| `adaptive` | adaptive | As `promote`, but before each test draws q from Beta(1 + c, 1 + f), with c and f this world's discoveries that were or were not the concatenation of two known concepts (exact check), and tests a macro hypothesis when q / (M kM) >= (1 - q) / (W kW) (set sizes M, W; presses to the next open window kM, kW). |
| `oracle_library` | reporting, privileged | Every concept supplied from the start (upper bound). |
| `random` | reporting | Uniform primitives; L2 only, at d = 2: the null for discovery curves. |
| `reference` | reporting, privileged | Replays the evaluator's shortest plan. |

Shared engine (`techtree/explorers.py`): before searching a level, the agent
holds every concept of that level's parent levels (sure gating: then a window
that is typed and does not fire is eliminated for every undiscovered concept
of the level); a firing reveals the sequence exactly; replays hold the cheapest
parent pair consistent with the held sets at earlier firings and failures.
Agents hold only a public facade (`env.agent_interface`: reset by an opaque
task token, step, the public spec and their observations); ordinary attribute
access cannot reach private state, and introspection is excluded by a source
check of the agent modules.

Design decisions:

- No admission gate for promotion. In RecipeQuest P3 a skill is a learned
  controller and needs validation before admission; here a discovery is exact
  (the firing reveals the sequence), so there is nothing to validate and every
  discovery is promoted at once.
- Incidental composition. Replaying known concepts back to back types their
  concatenation, which the environment tests like any other window; in
  compositional worlds `remember` therefore finds some compositions without
  hypothesizing them. All arms use the same planner and replay order, so this
  is common to them; the contrast measures intentional composition beyond it.
  Every row records `discoveries_in_replay` (new discoveries made while
  replaying), and the analysis reports it per arm and depth.
- Steps, not cost ratios, as the primary metric: reference lengths differ
  between depths (7 against 10.5 and 12.5), which would rescale ratio
  differences.
- Common random numbers: every arm and depth of a seed starts from the same
  agent seed and per-world symbol preference.

### Namespaces, worlds and seeds

Phase L1: measurement namespace `techtree_l`; phase L2: `techtree_l2`, a fresh
confirmation sample; neither is used in development (namespace
`techtree_dev`). Seeds 0-7, 4 worlds per seed: 32 worlds per depth and phase,
12 episodes each.

### Endpoints

Per world: `steps_late`, the mean actions per episode over episodes 6-11
(failures count their full budget); `steps_all`; success and cost ratios early
and late; `first_success_steps`; `discoveries_new` and `discoveries_in_replay`;
discovery curves (distinct concepts discovered against actions, supplied
level-1 concepts excluded in L1).

### Predictions (written before any measurement run)

Coverage counts: level 2 has 81 windows (all three concepts found after about
61.5 tests) against 6 macro candidates (about 5.3 tests); level 3 has 729
windows (all three after about 548 tests, a given one after 365) against 18
macro candidates (about 14 tests). With about 75 new windows per episode after
the sure-gating replays, `remember` needs about 7 episodes at level 3, less
where incidental composition helps (d >= 1).

- d = 2: `promote` finds everything in the first one or two episodes; late
  half near the reference (about 8 actions per episode) against about 20-30 for
  `remember`.
- d = 0: every macro test fails (about 90 extra actions once per world, before
  the primitive search); `promote` is equal or slightly worse than `remember`.
- d = 1: the level-2 saving (about 45 actions) and the futile level-3 macro
  tests (about 60) roughly cancel for level-3 goals; the analytic expectation is
  a difference near 0. (The audit's prediction, promotion better at every
  depth >= 1, is tested as S3.)
- `sham_promote`: the overhead of `promote` at d = 0 at every depth, so no
  better than `remember`, and much better than `none`, because its replay
  memory is intact (the prediction "sham worse than no memory" applies to a
  variant whose sham nodes replace memory, which is not run).
- `adaptive`: close to `promote` at d = 2; at d = 0 it learns q from the flat
  level-2 discoveries and skips part of the level-3 macro tests; at d = 1 the
  compositional level-2 discoveries push it towards the futile macro tests.
- `none`: fails most level-3 episodes (about 100 actions per episode).
- L2 differs from L1 by the level-1 search (at most 10 windows per world) and
  by its fresh worlds.

### Primary contrast, falsification and verdict

Primary L1 (phase L1): `promote - remember` on `steps_late` at d = 2, paired by
world; holds if the 95% bootstrap CI lies below 0.

Falsification F1: the 90% CI of the same difference lies within +-10% of
`remember`'s mean `steps_late` at d = 2 (promotion equivalent to remembering).

Conditions for reading L1 as a representation effect: S1 (no benefit at d = 0:
the CI of `promote - remember` at d = 0 does not lie entirely below 0) and S5
(`promote - sham_promote` at d = 2 has a CI below 0).

Verdict (`analysis.verdict`, rule `precedence`), applied in this order:

0. If L1, F1 or a validity check (V1-V3) has no verdict because a run it reads
   is incomplete, the phase has no verdict. If L1 holds without F1 and S1 or S5
   has no verdict, the phase has no verdict.
1. A failed validity check (V1-V3) fails the phase.
2. L1 holds and F1 does not: pass if S1 and S5 hold, otherwise "confounded".
3. L1 and F1 both hold: the benefit is declared negligible (not a pass).
4. F1 holds and L1 does not: falsified (no benefit).
5. Neither holds: inconclusive.

Power note: with 32 worlds the equivalence test can only succeed if the
standard deviation of the per-world differences is small (about 10 actions or
less for a true difference near 0); it is a stated rule, not a likely outcome.

Phase L2 applies the same rule to its own primary L2, falsification F2 and
conditions S1 and S5 (`configs/layers/l2_main.yaml`).

### Secondary contrasts (reported; they qualify but do not decide the verdict)

- S2: the benefit is larger at d = 2 than at d = 0 (interaction, paired by
  world across depths; CI below 0).
- S3: `promote - remember` at d = 1, and the interaction d = 2 versus d = 1.
- S4: `sham_promote - remember` at d = 0 and d = 2: no benefit from useless
  nodes.
- S6: `sham_promote - none` at d = 2 (CI below 0, see predictions).
- S7: `adaptive` within 10% of the better of `remember` and `promote` at every
  depth (upper CI bound of the paired difference at most 10% of that arm's
  mean). Underpowered: the margin is a few actions per episode against
  per-world differences of tens of actions, so a failure is uninformative.
- S8: the primary contrast on `steps_all`.
- S9: `none - remember` at d = 0 (memory amortizes).
- Validity: V1 the reference and V2 the oracle library solve every task; V3 the
  oracle library never beats the reference (both phases).
- Descriptive: `discoveries_in_replay` per arm and depth (incidental
  composition); discovery curves with the random null (L2, d = 2). Cells read
  by no test, descriptive only: L1 `sham_promote` at d = 1; L2
  `sham_promote` at d = 0 and 1, and `random` at d = 2 (the curve null).
- Table notes: `discoveries_new` of `none` counts the same concepts again in
  every episode, so it is not comparable with the other arms; `discovered_late`
  is not meaningful for the reference and the oracle library.

### Statistics

Worlds are the units; contrasts are paired by world (and across depths for
interactions, since worlds are coupled); 10,000 percentile bootstrap resamples
of worlds with a fixed seed (20261008); 95% intervals (90% for equivalence);
two-sided exact sign tests over worlds reported alongside. No multiplicity
correction for secondaries.

### Incomplete or repeated runs

Every (depth, arm, seed) must have exactly one complete run: every world with
all 12 episodes and no stop by the session budget. A cell with a missing or
incomplete run is flagged and every test that reads it has no verdict; the
verdict then follows step 0 above. A second run of the same (depth, arm, seed)
is an error, unless the earlier run is incomplete and the later one has the
same run hash and was made by the same code (the same loaded package code and
commit, from the run manifests): a deterministic rerun, which is then used and
listed. A rerun after a code change is refused and needs a dated amendment. The
runner refuses to start a measurement config whose runs already exist complete,
unless `--resume`, which repeats only missing or incomplete runs and is itself
refused when the run to repeat was made by other code.

### Limits stated in advance

- A macro hypothesis is the concatenation of stored sequences, so L cannot
  separate promotion to a higher-layer node from composing remembered
  sequences: the audit's "unfolded remember" control (remember allowed to
  concatenate stored sequences) coincides with `promote` here.
- Both effects of interest are partly fixed by construction (d = 2 benefit,
  d = 0 miss); the result is about sizes and controls, not existence.
- Only hand-written elimination agents are tested; nothing is claimed about
  learning agents.

### Budget

Ledger `runs/L-ledger.json` (created by the first development run): allocations
`dev` 20,000, `l1_main` 576,000 and `l2_main` 576,000 adaptive; adaptive cap
1,300,000; reporting cap 650,000. From the dry runs:

| Item | Runs | Per-run cap | Cap total | Expected use |
|---|---|---|---|---|
| L1 adaptive arms (none at d = 0, 2; remember, promote, sham, adaptive) | 112 | 4,800 | 537,600 | about 280,000 |
| L1 reporting arms (oracle library, reference) | 48 | 4,800 | 230,400 | about 23,000 |
| L2 adaptive arms (remember, promote, sham, adaptive) | 96 | 4,800 | 460,800 | about 210,000 |
| L2 reporting arms (oracle library, reference; random at d = 2) | 56 | 4,800 | 268,800 | about 60,000 |
| Development (used, exact) | 13 | | | 8,402 adaptive + 965 reporting |

A run's cap is the sum of its 48 episode budgets, so no run is cut short.
Expected runtime per phase on one CPU core: about 10 minutes (about 7-10 ms per
explorer episode, about 20 ms for `none`, plus about 0.6 s of provenance
recording per run).

Commands (after the freeze):
`python -m hypergraph_agent.techtree --config configs/layers/l1_main.yaml --allow-measurement`,
then `python -m hypergraph_agent.techtree.analysis --config configs/layers/l1_main.yaml`;
L2 the same with `l2_main.yaml`. A measurement config runs whole (subsets are
refused); `--resume` repeats only missing or incomplete runs.

### Stopping rules

L1 is run once, completely, after the freeze; L2 after L1, without changes
prompted by L1's outcome other than dated amendments. No interim analysis, no
change of parameters, arms or endpoints after any measurement row exists.
Incomplete runs are handled as above. If a validity check fails, the phase
fails.

## Implementation

- `techtree/layered.py`: the L arms on top of `techtree/explorers.py`.
- `techtree/generator.py`: reuse depth, coupled worlds, the flat-sequence rule,
  supplied links (`revealed_links`, evaluator side, passed by the runner through
  `Explorer.reveal` to every arm alike).
- Configs: `configs/layers/l1_main.yaml`, `configs/layers/l2_main.yaml`
  (measurement drafts), `configs/layers/l_dev.yaml` (development).
- Tests: `tests/test_layered.py` (promotion against remembering at d = 2, exact
  composition checks at every depth, macro tests never finding a flat concept,
  sham nodes, the adaptive belief, supplied links, the oracle library, soundness
  of every arm's evidence) and the shared tests in `tests/test_techtree.py`
  (including the verdict rules and the run-selection rules).

### Development record (calibration, not results)

Development used the development config (phase L1, d = 0 and 2, 1 seed x 2
worlds x 12 episodes). In those worlds `steps_late` was 8.6 (`promote`) against
23.1 (`remember`) at d = 2 and 42.5 against 39.0 at d = 0; the oracle library
and the reference solved every task; at d = 0 both `remember` and `promote`
failed most of the first six episodes and solved most later ones, as the
coverage count predicts. Two worlds per arm are dominated by where each arm's
walk happens to meet the level-3 sequences (the `sham_promote` and `adaptive`
numbers at d = 0 differed from `remember` in both directions), which motivated
common random numbers and 32 worlds per depth. Sizes: 3 primitives and lengths
2/4/6 make level 3 searchable by primitive windows within several episodes (729
windows) in flat worlds, so the depth-0 control is not decided by a floor
effect, while the 18 macro candidates make promotion fast at d = 2. The
doubling length rule (lengths 2/4/8, implemented as an option) would give 6,561
level-3 windows and make flat worlds unsolvable for every arm. Development used
8,402 adaptive and 965 reporting interactions of the L ledger (one crashed
reference run, fixed before re-running, used none; the analysis replaces it by
its deterministic rerun). These runs used per-arm agent seeds; the measurement
code uses common random numbers, which changes only tie-breaking.

## Results

All 160 runs of `configs/layers/l1_main.yaml` and 152 runs of
`configs/layers/l2_main.yaml` completed every task (no incomplete cell, no
rerun); every manifest records commit 73e5602, a clean working tree and one
loaded-code hash. Analyses: `artifacts/techtree/l1-main/` and
`artifacts/techtree/l2-main/` (reproduced exactly from the frozen snapshot).
Measurement used 301,694 adaptive and 24,370 reporting interactions for L1
(caps 537,600 and 230,400) and 228,640 and 61,566 for L2 (caps 460,800 and
268,800). The summed per-run wall-clock time from the manifests, including run
bookkeeping and provenance recording, was 156.0 s (L1) and 128.1 s (L2), of
which the evaluation loops took 140.3 s and 113.8 s.

Mean actions per episode in the late half (episodes 6-11), 32 worlds per cell:

| Phase, depth | remember | promote | sham_promote | adaptive | none / random | oracle library | reference |
|---|---|---|---|---|---|---|---|
| L1, d = 0 | 45.0 | 47.2 | 44.3 | 45.3 | 97.2 (none) | 13.9 | 12.4 |
| L1, d = 1 | 37.0 | 33.1 | 40.8 | 34.1 | | 11.3 | 10.4 |
| L1, d = 2 | 31.2 | 10.1 | 36.3 | 10.2 | 91.8 (none) | 8.8 | 7.0 |
| L2, d = 0 | 39.1 | 44.8 | 43.9 | 40.7 | | 14.1 | 13.0 |
| L2, d = 1 | 36.8 | 36.3 | 37.1 | 33.8 | | 11.5 | 10.5 |
| L2, d = 2 | 27.7 | 10.7 | 30.3 | 10.4 | 96.9 (random) | 8.4 | 7.0 |

Pre-registered tests, `promote - remember` on late-half actions unless stated
(95% world-cluster bootstrap intervals; 90% for F1 and F2). Two-sided sign
tests are given for the primaries; those of every test are in `analysis.json`.

- L1 (primary, phase L1, d = 2): -21.0 [-27.5, -14.7]; 25 of 26 untied worlds
  favour `promote` (sign test p = 8.0e-7). Holds. F1: 90% interval [-26.5,
  -15.7] against the margin +-3.12; not equivalent.
- L2 (primary, phase L2, d = 2): -17.0 [-22.2, -11.9]; 21 of 22 untied worlds
  (p = 1.1e-5). Holds. F2: [-21.4, -12.7] against +-2.77; not equivalent.
- S1 (condition, d = 0, no benefit): L1 +2.3 [-1.6, 6.1]; L2 +5.7 [-0.3, 12.1].
  Holds in both phases: promotion showed no benefit without reuse (both point
  estimates are above 0).
- S5 (condition, `promote - sham_promote`, d = 2): L1 -26.2 [-33.3, -19.2]; L2
  -19.6 [-25.1, -14.3]. Holds in both phases.
- S2 (benefit larger at d = 2 than at d = 0): L1 -23.3 [-30.8, -16.0]; L2 -22.8
  [-28.8, -16.6]. Holds.
- S3 (d = 1): L1 -3.9 [-8.7, 0.8], fails; L2 -0.6 [-4.7, 4.2], fails. The L1
  interaction d = 2 versus d = 1, -17.1 [-25.3, -9.6], holds.
- S4 (`sham_promote - remember`, L1): d = 0 -0.6 [-4.4, 3.2]; d = 2 +5.2 [-0.4,
  11.0]. Both hold: no benefit from useless nodes (at d = 2 the point estimate
  is above 0).
- S6 (`sham_promote - none`, L1, d = 2): -55.5 [-62.1, -48.8]. Holds: sham
  promotion keeps the replay memory and is far better than no memory.
- S7 (`adaptive` within 10% of the better fixed arm at every depth): fails in
  both phases. L1: d = 0 +0.3 [-3.2, 4.1] against `remember`, margin 4.50,
  holds; d = 1 +1.0 [-2.2, 4.3] against `promote`, margin 3.31, fails; d = 2
  +0.1 [-0.1, 0.3], margin 1.01, holds. L2: d = 0 +1.6 [-3.4, 6.5] against
  `remember`, margin 3.91, fails; d = 1 -2.4 [-6.1, 1.1] against `promote`,
  margin 3.63, holds; d = 2 -0.2 [-0.8, 0.2], margin 1.07, holds.
- S8 (L1, whole stream, d = 2): -32.3 [-38.1, -26.7]. Holds.
- S9 (L1, `none - remember`, d = 0): +52.2 [46.6, 57.8]. Holds.
- V1-V3: in both phases the reference and the oracle library solved all 1,152
  tasks and the oracle library never beat the reference. Hold.
- Incidental composition (`discoveries_in_replay` per world, `remember`): L1
  1.34 of 5.59 discoveries at d = 2, 0.81 of 5.44 at d = 1, 0.06 of 5.28 at
  d = 0; L2 0.41 of 8.56, 0 of 8.53 and 0.06 of 8.66. Replays alone found about
  a quarter (24%) of `remember`'s discoveries at L1 depth 2, but only 4.7% at
  L2 depth 2.
- Random null (L2, d = 2): `random` held 8.25 of the 9 concepts at least once
  by the end of a world (`remember` 8.56) but solved 5.7% of late episodes, and
  10 of 32 worlds never; counts of concepts discovered barely separate it from
  the explorers, so the discovery curves are read for timing only.

## Verdict and limits

Both phases pass under the precedence rule: the primary holds, the
equivalence does not, and both conditions (no benefit at depth 0, promote below
sham at depth 2) hold; L2 confirms L1 on fresh worlds with real discovery. In
this toy world, when discoveries are built from earlier ones (depth 2), using
them as atoms of new hypotheses cut late-half cost by 67% (L1) and 61% (L2)
relative to remembering them as literal links (31.2 to 10.1 and 27.7 to 10.7
actions per episode), to within 1.4 and 2.3 actions of the oracle library. That
a benefit exists at depth 2 is expected by construction, and the depth-0
control is guaranteed by the generator's rule for flat sequences; the
informative results are the size of the benefit, the absence of any benefit
from as many useless promoted nodes, the lack of a clear benefit at depth 1
(consistent with the prediction that the level-2 saving and the futile level-3
composition tests roughly cancel), and the share of compositions that plain
replay finds incidentally (24% of `remember`'s discoveries at L1 depth 2, 4.7%
at L2 depth 2). The adaptive arm matched the better fixed arm in point estimate
(within 2.4 actions everywhere) but failed S7 in both phases; it did not
demonstrate that it detects the world kind within the pre-set margin.

Limits: a macro hypothesis is the concatenation of stored sequences, so these
results cannot separate promotion to a higher-layer node from composing
remembered sequences; the domain is a toy (three primitives, three levels of
three concepts, deterministic dynamics, exact discoveries); all arms are
hand-written elimination procedures, so nothing is shown about learning agents,
and nothing about intelligence or knowledge in general; L1 fixes only the
level-1 input; `remember` gains part of the composition benefit through replay;
S7 was underpowered by design; the review pilot's outcomes were seen before the
verdict rules were fixed.
