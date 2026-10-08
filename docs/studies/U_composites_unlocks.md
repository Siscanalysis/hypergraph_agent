# Study U: discovery cost of unlisted composites and unlocks

Status: protocol frozen on 2026-10-08 by the commit that adds this file
(73e5602), before any measurement run. The measurement runs executed that
commit from a clean checkout and are reported in the Results section.
Development runs used only the dev allocation and the namespace `techtree_dev`.
An independent review ran the full measurement design once on the scratch
namespace `checker_dev` with in-memory ledgers (no measurement world touched)
and obtained U1a = 179 [148, 210] with a ratio of 3.9; those outcomes were seen
before the +-25% band of U1b, the per-cell arm lists and the margin of S2 were
fixed.

## Question and claim

Question: when useful actions are not listed (they are secret short sequences
of listed primitive actions) and achievements unlock further primitives, what
does discovering them cost, and does TechTree measure that cost the way theory
says it should?

U is a mechanism and validity check of the test, not a finding about agents.
Its primary contrast is fixed by counting: because facts persist, a primitive
sequence tests every length-k window it contains, so an agent that tests
candidates one at a time with the trace cleared in between pays about k + 1
actions per candidate where a pooling agent pays about one. With 64 base
windows and 48 slot windows at k = 3, the analytic ratio is 3.8 (written before
any run). A pass shows that the environment implements pooled testing as the
coverage argument predicts (and that the reference and the oracle behave), so
TechTree is a valid instrument for measuring discovery cost under composite
length, intermediate signal and unlock visibility. It shows nothing about
learning agents: both explorers are hand-written elimination procedures with
declared priors.

The secondaries describe the crossing for the arms tested: with no signal,
`pooled` and `isolated` need about half of the candidate windows per discovery
(a statement about these two arms, not about every possible agent); a signal on
partial progress lowers their cost; a silent unlock costs more than an
announced one, because slot candidates must be tried without being told which
slot became active.

## Relation to earlier work

docs/PRIOR_ART.md Section 8.3 positions this study. The closest prior work is
the combination lock of rich-observation RL (one correct action of ten at each
of 100 steps, C10) and Alchemy (C1), which resamples a latent causal structure
per episode and supplies a Bayes-optimal ideal observer. The pooled-testing
argument is group testing (N6, Section 8.1); width-based novelty search (C11),
macro-operators (C14), announced changes of the action set (C15) and hypotheses
of the form precondition, action sequence, postcondition (C18) are the other
neighbours. The family itself (composite length crossed with an intermediate
signal and unlock visibility, with exact references) is, at most, a
methodological contribution.

## Protocol (frozen)

### Environment

TechTree (`src/hypergraph_agent/techtree/`), one level of concepts per world:

- P = 4 base primitives `p0..p3`, always available; R = 3 extra primitive
  slots (a public range).
- Two hidden composites of length k: a key (k base primitives) and an entry
  (k symbols, exactly one of them, at a uniform position, the slot that the key
  activates; the slot itself is drawn uniformly from the three). Holding the
  key activates its slot for the rest of the episode.
- Firing: after every primitive press, a concept that is not held and whose
  sequence equals the suffix of the primitive trace becomes held (the trace is
  cleared by `wait` and `submit`). Held concepts persist within the episode
  and are lost at its end.
- Task: the goal is always the entry; `submit` succeeds (reward 1) when it is
  held. Budget 100 actions per episode, one unit per action. A world is
  visited for 12 consecutive episodes; agents may keep what they learned within
  a world, and the next world starts from nothing.
- Factor k (composite length): 1, 2, 3.
- Factor signal: off, or on: the observation's `progress` flag is true when the
  trace suffix is a nonempty proper prefix of a composite that is not held.
  With k = 1 there is no proper prefix, so the signal cells at k = 1 are
  identical to the no-signal cells for every arm and are not run.
- Factor visibility: announced (the activated slot appears in the observation's
  available-action mask; slots are listed but unavailable until then) or silent
  (all three slots are listed and available from the start as generic `try`
  actions; an inactive one only breaks the trace; the agent is never told which
  one became active).

Public information: the action list, P and R, the two concepts with their
level and length k, the goal, the budget, the visibility mode and whether the
signal is on. Hidden: the sequences, which concept is the key, which slot it
activates. Agents hold only a public facade (`env.agent_interface`: reset by an
opaque task token, step, the public spec and their observations); ordinary
attribute access cannot reach private state, and introspection is excluded by a
source check of the agent modules. The exact reference (`reference_solve`,
evaluator only) is a breadth-first search over held concepts and trace
suffixes, checked against an independent full-state search in the tests.

### Arms

All non-random arms share one engine (`techtree/explorers.py`): hypothesis
elimination over windows, exact discoveries (a firing reveals the sequence as
the trace suffix of the public length), cross-episode memory per world, replay
of known composites. Declared priors: windows that use a usable extra slot are
tried first, with the fewest slots first ("new facts enable new actions": an
announced slot once it appears; silent slots once some concept is held, with
eliminations that involve a silent slot valid only within the held context in
which they were made).

| Arm | Kind | Description |
|---|---|---|
| `pooled` | adaptive | Presses the shortest continuation of the trace that ends in an untested window (a de Bruijn-like walk: about one new window per press), ties broken by a fixed per-world symbol preference. |
| `isolated` | adaptive | Tests one window at a time, in the lexicographic order of the same per-world symbol preference; before every window longer than one press it clears the trace with `wait`, so no press tests two candidates. The falsification arm. At k = 1 its order and actions coincide with `pooled` by construction. |
| `random` | reporting | Uniform over the available primitives; submits as soon as the goal is held. |
| `oracle` | reporting, privileged | The same engine given both composites, the key and its slot (the composite "listed as an action"); the discovery cost of an arm is its cost minus the oracle's. |
| `reference` | reporting, privileged | Replays the evaluator's shortest plan. |

Implemented but not in the measurement config: `nomem` (no memory between
episodes) and `blind` (never presses a slot; fails every task here by
construction).

### Cells, namespaces, worlds and seeds

10 cells: k1-off-ann, k1-off-sil, and `k{2,3}-{off,on}-{ann,sil}`. Every cell
runs `pooled` and `isolated`. The oracle and the reference run in the three
k?-off-ann cells only: for one k they act identically in all signal and
visibility cells (same worlds and tasks; the reference plan presses a slot only
after its key fired, which is valid and has the same effect under both
visibilities; the oracle only replays and never reads the signal), which a test
checks (`test_reference_and_oracle_are_identical_across_signal_and_visibility`).
`random` runs only in k3-off-ann and k3-off-sil, the cells its contrasts read.

Measurement namespace `techtree_u` (never used in development); development
namespace `techtree_dev`. Seeds 0-4, 6 worlds per seed: 30 worlds per cell, 12
episodes each. Worlds depend on (namespace, seed, world index) and on k only, so
the signal and visibility cells of one k share their worlds and tasks
(paired). Every arm of a seed starts from the same agent seed (common random
numbers).

### Endpoints

Per world (the unit):

- `first_success_steps`: actions in the world up to and including the first
  successful episode; censored at the stream total (1,200) when no episode
  succeeds (the number censored is reported).
- `steps_late`: mean actions per episode over episodes 6-11 (failures count
  their full budget); success rates and cost ratios (actions / reference
  actions), early and late; discovery curves.

### Predictions (written before any run)

Expected actions to the first success with no signal, from the coverage
argument (a target uniform among N candidates is found after (N + 1) / 2 tests;
`pooled` about one press per new window after k - 1 presses; `isolated` k + 1
actions per window for k >= 2; random waits about A^k presses for a given
k-gram over A available symbols):

| k | Visibility | pooled | isolated | isolated / pooled | random (about) |
|---|---|---|---|---|---|
| 1 | announced | 4.5 | 4.5 | 1.0 | 10 |
| 1 | silent | 5.5 | 5.5 | 1.0 | 15 |
| 2 | announced | 15 | 40 | 2.7 | 42 |
| 2 | silent | 23 | 64 | 2.8 | 99 |
| 3 | announced | 60 | 229 | 3.8 | 190 |
| 3 | silent | 108 | 421 | 3.9 | 687 (most worlds censored) |

The oracle and the reference need 2k + 1 actions or fewer (overlaps). Episode
boundaries add a few actions per extra episode. Further predictions: the signal
reduces the cost of `pooled` and `isolated` at k >= 2 (random does not read it);
silent costs more than announced for every arm (about R = 3 times as many slot
windows); late-half costs are equal for every non-random arm once the
composites are known.

### Primary contrast and pass criterion

U1, in cell k3-off-ann, passes if both hold:

- U1a: the 95% world-cluster bootstrap CI of `isolated - pooled` on
  `first_success_steps` lies above 0;
- U1b: the ratio of the mean `first_success_steps` of `isolated` to that of
  `pooled` (over the 30 worlds) lies within +-25% of the analytic 3.8, that is
  2.85 to 4.75. The value 3.8 was written before any run; the band was set
  after the review pilot. With 30 worlds the ratio's sampling coefficient of
  variation is about 10%, so the band is about 2.5 standard errors wide on
  each side.

U passes iff U1a, U1b and the validity checks hold (`analysis.verdict`, rule
`all`):

- V1: the reference solves every task it plays;
- V2: the oracle solves every task;
- V3: the oracle never uses fewer actions than the reference.

Falsification: no pooled advantage at k = 3 (U1a fails), or a ratio outside the
band (the environment does not implement pooled testing as predicted).

### Secondary contrasts (reported, no verdict on U)

- S1: `isolated - pooled` in k3-off-sil, k3-on-ann, k3-on-sil, k2-off-ann and
  k2-off-sil (expect > 0); S1r: the isolated / pooled ratio in k3-off-sil,
  k2-off-ann and k2-off-sil against +-25% bands around the analytic 3.9, 2.67
  and 2.78 (bands set after the review pilot).
- S2: `isolated` equivalent to `pooled` at k1-off-ann (90% CI within +-1
  action). The two arms act identically at k = 1 by construction, so S2 checks
  the implementation, not a hypothesis.
- S3: the pooled advantage grows from k = 2 to k = 3 (interaction, off-ann).
- S4: discovery cost `pooled - oracle` at k3-off-ann (expect > 0).
- S5: `random - pooled` at k3-off-ann (expect > 0).
- S6: signal on minus off for `pooled` and `isolated` at k = 3, announced
  (expect < 0).
- S7: silent minus announced for `pooled`, `isolated` and `random` at k = 3, no
  signal (expect > 0).
- S8: late-half `steps_late` of `isolated - pooled` at k3-off-ann (expect no
  difference below 0).
- Descriptive: ratios isolated / pooled per cell; discovery curves for every
  arm, random included. Cells k1-off-sil, k2-on-ann and k2-on-sil are read by
  no test and are descriptive only.

### Statistics

Worlds are the units. Contrasts are paired by world; 10,000 percentile
bootstrap resamples of worlds with a fixed seed (20261008), 95% intervals (90%
for equivalence); a two-sided exact sign test over worlds is reported next to
each interval. No correction for multiplicity is applied to the secondaries,
which carry no verdict.

### Incomplete or repeated runs

Every (cell, arm, seed) must have exactly one complete run: every world with
all 12 episodes and no stop by the session budget. A cell with a missing or
incomplete run is flagged, and every test that reads it has no verdict. U has
no verdict (it does not pass) if any run that U1 or V1-V3 read is incomplete:
`pooled` or `isolated` in k3-off-ann, or an oracle or reference run in
k1-off-ann, k2-off-ann or k3-off-ann. A second run of the same (cell, arm,
seed) is an error, unless the earlier run is incomplete and the later one has
the same run hash and was made by the same code (the same loaded package code
and commit, from the run manifests): a deterministic rerun, which is then used
and listed (`techtree/analysis.py`). A rerun after a code change is refused and
needs a dated amendment. The runner refuses to start a measurement config whose
runs already exist complete, unless `--resume`, which repeats only missing or
incomplete runs and is itself refused when the run to repeat was made by other
code.

### Limits stated in advance

- Only two elimination procedures, a random agent and two privileged arms are
  tested. Width-based novelty search (IW(k)) and flat macro expansion (every
  sequence up to length k listed as an action) are not implemented, so no claim
  is made about them or about agents in general.
- One key and one entry per world; the goal is always the entry, so late
  episodes are pure replay and the late half carries no contrast.
- The priors of the explorers (slot windows first, fewest slots first) match
  the generator (exactly one slot per entry); a mismatched prior would cost
  more.

### Budget

Ledger `runs/U-ledger.json` (created by the first development run): allocations
`dev` 20,000 and `u_main` 864,000 adaptive; adaptive cap 1,000,000; reporting
cap 1,400,000 (shared by development and measurement reporting runs). From the
dry run of `configs/unlock/u_main.yaml`:

| Item | Runs | Per-run cap | Cap total | Expected use |
|---|---|---|---|---|
| `pooled`, `isolated` (adaptive, 10 cells) | 100 | 7,200 | 720,000 | about 80,000 |
| `oracle`, `reference` (3 cells), `random` (2 cells) (reporting) | 40 | 7,200 | 288,000 | about 80,000 |
| Development (used, exact) | 33 | | | 3,581 adaptive + 2,849 reporting |

A run's cap is the sum of its 72 episode budgets, so no run is cut short.
Expected runtime on one CPU core: about 2-3 minutes (about 1.5-2.5 ms per
explorer episode, 8 ms per random episode, plus about 0.6 s of provenance
recording per run).

Command (after the freeze):
`python -m hypergraph_agent.techtree --config configs/unlock/u_main.yaml --allow-measurement`,
then `python -m hypergraph_agent.techtree.analysis --config configs/unlock/u_main.yaml`.
A measurement config runs whole (subsets are refused); `--resume` repeats only
missing or incomplete runs.

### Stopping rules

The measurement config is run once, completely, after the freeze. No interim
analysis and no change of parameters, arms or endpoints after any measurement
row exists. Incomplete runs are handled as above. If a validity check fails, U
fails; any repeat other than a deterministic rerun of an incomplete run needs a
dated amendment.

## Implementation

- `techtree/generator.py`: worlds, unlock layout, task streams, exact reference
  (evaluator side). `techtree/env.py`: Gymnasium environment, public view and
  the agent facade.
- `techtree/explorers.py`: the elimination engine and the U arms. Agents hold
  only the public facade; ordinary attribute access cannot reach private state,
  and introspection is excluded by a source check (an AST check of the agent
  modules forbids reflective builtins, dunder, frame and traceback attributes,
  and private attributes of objects other than `self`; their imports are
  restricted too).
- `techtree/study.py`, `techtree/__main__.py`: runner and CLI with `--dry-run`
  and `--resume`; `techtree/analysis.py`: run selection, per-world metrics,
  contrasts, verdicts, tables, figures.
- `techtree/config.py`: strict config schema (unknown keys rejected; the `dev`
  allocation may only use namespaces ending in `_dev`, and every other
  combination needs `run.requires_flag`).
- Configs: `configs/unlock/u_main.yaml` (measurement draft),
  `configs/unlock/u_dev.yaml` (development).
- Tests: `tests/test_techtree.py`.

### Development record (calibration, not results)

Development runs used the development config (namespace `techtree_dev`, 1
seed x 2 worlds x 8 episodes, cells with k = 2 and 3). They checked plumbing and
sizes only. In those 2 worlds per cell the reference and the oracle solved all
their tasks, the oracle never beat the reference, and actions to the first
success were close to the predictions in size (k3-off-ann: pooled 28, isolated
192; k3-off-sil: 94 and 450; k2-off-ann: 15 and 54; random solved 31% of the
k3-off-ann episodes); two worlds are too few for the ratio (6.9 there). P = 4
and R = 3 were chosen so that k = 3 gives 64 base windows (an isolated search of
several hundred actions, spanning several episodes) while k = 1 stays trivial;
12 episodes x 100 actions cover the slowest predicted cell (isolated, k = 3,
silent) with margin. Development used 3,581 adaptive and 2,849 reporting
interactions of the U ledger. The development runs were made with an earlier
version of the code (per-arm agent seeds except the random run, random window
order for `isolated`, oracle and reference in every cell); the measurement code
differs only in tie-breaking and in which cells run which arm.

## Results

All 140 runs of `configs/unlock/u_main.yaml` completed every task (no
incomplete cell, no rerun); every manifest records commit 73e5602, a clean
working tree and one loaded-code hash. Analysis: `artifacts/techtree/u-main/`
(reproduced exactly from the frozen snapshot). Measurement used 75,638
adaptive and 79,193 reporting interactions of the U ledger (caps 720,000 and
288,000). The summed per-run wall-clock time from the manifests, including run
bookkeeping and provenance recording, was 72.8 s, of which the evaluation loops
took 55.8 s.

Mean actions to the first success per world (30 worlds per cell; predictions
from the protocol in brackets):

| Cell | pooled | isolated | isolated / pooled | oracle = reference | random |
|---|---|---|---|---|---|
| k1-off-ann | 4.9 [4.5] | 4.9 [4.5] | 1.00 [1.0] | 3.0 | |
| k1-off-sil | 6.0 [5.5] | 6.0 [5.5] | 1.00 [1.0] | | |
| k2-off-ann | 17.8 [15] | 44.0 [40] | 2.47 [2.7] | 4.9 | |
| k2-off-sil | 27.9 [23] | 70.6 [64] | 2.53 [2.8] | | |
| k2-on-ann | 12.8 | 18.9 | 1.48 | | |
| k2-on-sil | 14.6 | 22.5 | 1.55 | | |
| k3-off-ann | 67.0 [60] | 272.8 [229] | 4.07 [3.8] | 6.8 | 348.0 [190] |
| k3-off-sil | 124.9 [108] | 514.0 [421] | 4.11 [3.9] | | 1035.2 [687, most worlds censored] |
| k3-on-ann | 31.4 | 48.7 | 1.55 | | |
| k3-on-sil | 38.4 | 56.6 | 1.47 | | |

Censoring: only `random` at k3-off-sil had censored worlds (20 of 30 never
succeeded within the 1,200 actions of the stream); no other arm or cell had
any, so every other entry, and S5 below, is exact.

Pre-registered tests (95% world-cluster bootstrap intervals; 90% for S2).
Two-sided sign tests are given for the primary; those of every test are in
`analysis.json`.

- U1a, `isolated - pooled` at k3-off-ann: 205.8 [174.3, 237.0], all 30 worlds
  positive (sign test p = 1.9e-9). Holds.
- U1b, ratio of means at k3-off-ann: 4.07 (bootstrap interval [3.64, 4.51]),
  inside the band 2.85-4.75. Holds.
- V1, V2, V3: the reference and the oracle solved all 1,080 tasks they played,
  and the oracle never used fewer actions than the reference (0 violations
  each). Hold.
- S1, `isolated - pooled`: k3-off-sil 389.1 [326.1, 449.3]; k3-on-ann 17.3
  [11.6, 23.2]; k3-on-sil 18.2 [11.3, 25.3]; k2-off-ann 26.2 [21.2, 31.2];
  k2-off-sil 42.7 [35.4, 50.3]. All hold. S1r ratios: k3-off-sil 4.11 [3.59,
  4.71] in 2.93-4.88; k2-off-ann 2.47 [2.22, 2.72] in 2.00-3.33; k2-off-sil
  2.53 [2.28, 2.80] in 2.09-3.48. All hold.
- S2, k1-off-ann: difference 0 [0, 0], equivalent within +-1 action (the two
  arms act identically at k = 1 by construction). Holds.
- S3, growth of the pooling advantage from k = 2 to k = 3: 179.6 [151.5,
  206.9]. Holds.
- S4, discovery cost `pooled - oracle` at k3-off-ann: 60.2 [53.5, 66.2]. Holds.
- S5, `random - pooled` at k3-off-ann: 281.0 [194.2, 372.3]. Holds.
- S6, signal on minus off at k = 3, announced: `pooled` -35.7 [-42.6, -28.5];
  `isolated` -224.1 [-258.6, -188.1]. Hold.
- S7, silent minus announced at k = 3, no signal: `pooled` 57.9 [43.9, 72.1];
  `isolated` 241.2 [194.3, 285.5]; `random` 687.2 [529.1, 832.1], a lower bound
  only through the 20 censored k3-off-sil `random` worlds. Hold.
- S8, late-half `isolated - pooled` at k3-off-ann: 0 [0, 0], no difference
  below 0. Holds: both arms had finished discovering by episode 4 in every
  k3-off-ann world.
- Descriptive: with the signal on, the isolated / pooled ratio is about 1.5 at
  k = 2 and 3, presumably because both arms then extend confirmed prefixes
  instead of covering whole windows (not tested). `random` solved 26.1% of the
  late episodes at k3-off-ann (26.7% early) and 3.3% at k3-off-sil; it does not
  improve within a world. In the late half every non-random arm matched the
  reference exactly (late cost ratio 1.00) in every cell except `isolated` at
  k3-off-sil (1.93), which was still discovering at episode 6 or later in 11 of
  30 worlds.

## Verdict and limits

U passes (`analysis.verdict`: U1a, U1b and V1-V3 hold). The pass shows what U
was designed to show and no more: TechTree implements pooled testing of
unlisted composites as the coverage argument predicts. The isolated / pooled
ratio grows with composite length (1.00, about 2.5, about 4.1 for k = 1, 2, 3)
and the observed ratios lie inside their bands; the oracle and the reference
behave as required. Absolute costs of `pooled` and `isolated` ran 8-22% above
the predictions, and `random` 83% above at k3-off-ann; the predictions ignored
episode boundaries (held composites are lost at the end of every episode) and
the presses needed to reach each next window, which may account for the
direction but was neither quantified in advance nor tested. TechTree is
therefore a valid instrument for measuring discovery cost as a function of
composite length under pooled and isolated testing. The effects of the
intermediate signal and of silent unlocks rest only on the directions of the
secondaries S6 and S7, which carry no verdict; both match the predictions.
Random search stayed near its per-episode success rate without improving. U
says nothing about learning agents or about intelligence: both explorers are
hand-written elimination procedures whose priors match the generator.

Limits: a toy domain (four base primitives, three slots, one key and one entry
per world, deterministic dynamics). The primary contrast is fixed by counting,
and its band, like the S1r bands, was set after the review pilot. Two
elimination procedures, a random agent and two privileged arms are the only
arms, so width-based novelty search, flat macro expansion and learned agents
remain untested. The explorers' priors (slot windows first, fewest slots first)
match the generator (exactly one slot per entry); a mismatched prior would cost
more. S2 holds by construction; S8 is an outcome (both arms had finished
discovering by episode 4 at k3-off-ann). Cells k1-off-sil, k2-on-ann and
k2-on-sil are descriptive only.
