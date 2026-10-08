# Study P: promotion when discoveries are imperfect learned skills

Status: draft protocol, not frozen. No measurement run exists. Development used
only the dev allocation and the namespace `techtree_p_dev`; it is reported in
the development record below. On 2026-10-08, before any measurement run and
before the independent review, the primary was switched from the gated to the
ungated contrast (Design decisions); the predictions were written before the
switch and are kept as written.

## Question and claim

Question: in study L a discovery was exact. A firing revealed the concept's
sequence, and typing that sequence again always worked, so promoting a
discovery to an atom of new hypotheses carried no execution risk and needed no
admission gate. Real skills are learned controllers that reproduce a discovery
imperfectly and improve with practice. Does promotion still pay when every
execution of a discovery can fail? Does an admission gate (a skill becomes an
atom of new hypotheses only after it has proved reliable) recover the benefit?
And does the utility problem appear, a promoted but unreliable skill that costs
more than it saves?

Claim under test (primary P1): at reuse depth 2 and initial skill error 0.3,
promotion (`promote_e30`, no admission gate) needs fewer actions per late
episode than remembering (`remember_e30`), whose replays are equally
imperfect. Whether an admission gate adds anything is a pre-registered
secondary (S1, S3, S4).

What holds by construction, and what is informative. At e0 = 0 `remember` and
`promote` make exactly the decisions of study L's arms (tested), so those cells
are a replication of L's phase L2 on fresh worlds. At depth 0 every macro test
is still a guaranteed miss. The informative parts are whether the depth-2
benefit survives execution errors that hit promotion's macro tests and
remembering's replays alike, whether the cost of useless promoted nodes at
depth 0 grows with their unreliability (the utility problem), and what the
gate buys and costs.

## Relation to earlier work

- Study L ([L_layered_discovery.md](L_layered_discovery.md)): with exact
  discoveries, promotion cut late-half actions by 61-67% at reuse depth 2, gave
  no benefit at depth 0, no clear benefit at depth 1, and useless promoted
  nodes did not help. L adopted no admission gate because a discovery was exact
  (its design decisions; docs/DECISIONS.md S5), and its limits note that only
  exact discoveries and hand-written elimination agents were tested.
- docs/PRIOR_ART.md Section 8.4: promoting recurring compositions to new
  primitives is library learning (K6), option learning by compression (K8),
  chunking (K9) and macro-operators in planning (C14, where macros are filtered
  before reuse); the known cost is the utility problem (K11), in which learned
  knowledge costs more to use than it saves. Section 8.5 notes that with learned
  controllers the P3 admission gate would apply. In L only the number of macro
  hypotheses could make promotion costly, never their execution, so K11 was
  untested in its usual form.
- RecipeQuest P3 (docs/METHODS.md Section 6; docs/DECISIONS.md D12-D16): skills
  are learned controllers admitted after held-out validation (at least 8
  validation episodes and success at least 0.5), and the matched macro control
  is an open-loop sequence validated under the same criteria. P carries the
  admission idea to TechTree with a controller model simple enough to analyse;
  no neural controller is trained.

Only the controlled contrast of gated and ungated promotion against
remembering, under one explicit skill-error model with a depth-0 control and an
e0 = 0 replication, is a potential empirical contribution. No novelty claim is
made for the skill model.

## Protocol (to be frozen)

### Environment

Study L's phase-L2 setting, unchanged: TechTree with 3 base primitives, three
levels of three concepts, public lengths 2, 4 and 6, no slots, unlocks or
intermediate signal; the goal is a level-3 concept, uniform; budget 100 actions
per episode; 12 consecutive episodes per world; nothing is supplied, every arm
discovers level 1 itself. Reuse depth 2 and depth 0 (control) on coupled worlds
(same namespace and seeds, so keys, parents, orders and level-1 combos agree
across depths). Depth 1 is not run: L found no clear effect there, and the
budget goes to the error levels.

### Skill model (agent side)

The environment, the evidence rules and the engine of `techtree/explorers.py`
are those of L. Only the agent's executions of known sequences change.

- Executions. An execution of a known concept c is one call of c's controller:
  a replay (holding c for sure gating or as the goal), c's part of a macro test
  (which types two known concepts back to back) or a practice run. A replay
  that continues a sequence already begun on the trace is an execution of the
  remaining presses.
- Errors and practice. The controller presses each intended primitive
  correctly with probability 1 - e and otherwise a different available
  primitive, uniform among them, with e = e0 / (1 + n / h) after n earlier
  executions of c in the current world. Every attempt counts as practice,
  whatever its outcome; h is the half-practice (the error halves after h
  executions). Skills are per world, like discoveries.
- Slips. The agent sees which primitive it pressed. A controller that slips
  stops, because that call can no longer type its sequence, and the engine
  decides again from the current trace (usually repeating the replay or the
  macro test). Outcomes: `ok` (every intended press made), `slip` (stopped by a
  wrong press), `cut` (stopped by the engine before its last press without a
  slip: a firing, or the end of the episode).
- Evidence. Eliminations, firings and candidate preconditions are computed
  from the actual trace, so they stay exact. A replay or practice run that types
  its sequence exactly and does not fire its concept records a precondition
  failure, as in L; a slipped one records nothing, since the sequence was not
  typed.
- Exact presses. Primitive windows, `wait` and `submit` are pressed exactly.
  Each is a single listed action, not a learned multi-step controller, and
  keeping them exact confines the imperfection to the object of the study, the
  execution of a discovery: the arms' primitive searches stay those of L, and
  the arms differ only in how much of their work runs through skills.
- Common random numbers. The n-th execution of c in a world reads the n-th
  block of a stream seeded by (controller seed, world key, c). The controller
  seed is derived from the engine's symbol-preference seed without drawing from
  the agent's generator. Every arm and error level of a seed therefore meets
  the same draws for the same execution, a press that slips at some e0 also
  slips at every larger e0, and at e0 = 0 `remember` and `promote` reproduce
  L exactly.

Parameters (`agent` block of the configs, part of every run hash):

- e0 in {0, 0.1, 0.3}, carried by the arm name (`_e00`, `_e10`, `_e30`).
  At e0 = 0.3 a fresh 6-press skill succeeds in 12% of its calls and a fresh
  2-press skill in 49%; at e0 = 0.1, in 53% and 81%.
- h = 10 (`half_practice`). In the development runs at e0 = 0.3 a skill was
  executed 3 to 27 times by the end of episode 5 and 3 to 39 times by the end
  of the world, so with h = 10 its late-half error lies between about three
  quarters and a fifth of e0: learning is visible within a world but incomplete
  where the primary metric is read. Fixed before any measurement run, not
  tuned.
- Gate: 2 `ok` among the last 3 `ok` or `slip` executions (`gate_m`, `gate_k`),
  admission once and for all. A majority of recent executions, comparable to the
  P3 rule (success at least 0.5) with a short window because a skill has few
  executions; cheap when skills are exact (two executions per skill, once per
  world). A gate that also demotes was not chosen: at e0 = 0.3 a level-2 skill's
  success rate stays near one half for most of a world, so admitted skills would
  flicker in and out of the macro set.

Expected presses for one successful execution under stop-on-slip,
C(L, e) = (1 - r^L) / ((1 - r) r^L) with r = 1 - e (exact: L):

| e0, executions so far | e | 2 presses | 4 presses | 6 presses |
|---|---|---|---|---|
| 0.1, n = 0 | 0.100 | 2.3 | 5.2 | 8.8 |
| 0.1, n = 10 | 0.050 | 2.2 | 4.6 | 7.2 |
| 0.3, n = 0 | 0.300 | 3.5 | 10.5 | 25.0 |
| 0.3, n = 10 | 0.150 | 2.6 | 6.1 | 11.0 |
| 0.3, n = 30 | 0.075 | 2.2 | 4.9 | 8.0 |

### Arms

| Arm | Kind | Summary |
|---|---|---|
| `remember_eXX` | adaptive | L's `remember`; replays are skill executions |
| `promote_eXX` | adaptive | L's `promote`; a macro test executes two skills |
| `promote_gate_eXX` | adaptive | `promote` with admitted atoms only, practising when needed |
| `oracle_library` | reporting, privileged | L's anchor, exact execution |
| `reference` | reporting, privileged | the evaluator's shortest plan |

- `remember_eXX`: primitive windows only, as in L; every replay (sure gating,
  the goal) is a skill execution at initial error XX/100.
- `promote_eXX`: macro hypotheses (concatenations of two distinct known
  concepts whose lengths add up to the level's length) first, then primitive
  windows, as in L; a macro test executes the two skills in turn.
- `promote_gate_eXX`: as `promote`, but only admitted skills are atoms of
  macro hypotheses. When a level has no open macro hypothesis over admitted
  atoms but would have one over all known concepts, the agent practises (types
  the sequence of) a known concept that is not admitted and appears in such a
  hypothesis: the one with the most `ok` outcomes in its window, then the
  shortest, then the lowest index. Practice presses are ordinary actions of the
  episode, charged with purpose `skill_practice`. Replays never need admission.
- `oracle_library`: every concept supplied by the runner and executed exactly,
  as in L (upper bound and validity anchor).

XX is 00, 10 or 30: nine adaptive arms and two reporting arms at each depth.

Not run, with reasons:

- `promote_retry`. With stop-on-slip and an immediate new decision, every arm
  already retries a failed execution from the current trace, so a retry arm
  coincides with `promote`; an open-loop variant that keeps pressing after a
  slip only adds wasted presses.
- `sham_promote`, `adaptive`, `none`, `random`: their questions were answered
  in L.
- An oracle library with imperfect execution at each e0 (see Limits).

Information boundary: `techtree/skills.py` holds only the public facade, like
L's agent modules; `tests/test_tt_skills.py` applies the same source check
(imports limited to the list of L's agent modules plus `layered`; no reflective
builtins, dunder, frame or traceback attributes, and no private attributes of
objects other than `self`).

Design decisions:

- Imperfection on the agent side only. The environment is unchanged and the
  agent always knows what it pressed, so evidence stays correct; this isolates
  execution cost from the noisy evidence studied in R.
- One gate rule, fixed in advance. Practice happens only when the gate blocks
  every open macro hypothesis; it counts as practice for the learning curve like
  any other execution.
- Steps, not cost ratios, as the primary metric (as in L).
- Common random numbers across arms, depths and error levels (above).
- Primary switched before any measurement run (2026-10-08). The first draft
  took the gated contrast (`promote_gate_e30 - remember_e30`) as P1. The
  question is whether promotion still pays when skills are imperfect, which
  the ungated contrast answers directly; the gate is one proposed remedy and
  its value is a separate question. P1 is now `promote_e30 - remember_e30`,
  with its falsification F1 and its depth-0 condition C1 on the same arms; the
  former P1 is secondary S1 and the former C1 is secondary S1b. The precedence
  rule and C2 are unchanged.

### Namespaces, worlds and seeds

Measurement namespace `techtree_p`, development namespace `techtree_p_dev`;
neither had been used by any run when this draft was written (checked against
every run directory on 2026-10-08), and development never uses the measurement
namespace. Seeds 0-7, 4 worlds per seed: 32 worlds per depth, 12 episodes
each. Every arm of a seed and depth plays the same tasks.

### Endpoints

Per world, as in L: `steps_late`, the mean actions per episode over episodes
6-11 (failures count their full budget); `steps_all`; success and cost ratios
early and late; `first_success_steps`; `discoveries_new` and
`discoveries_in_replay`.

New, per episode row: `exec_steps` (presses inside skill executions: replays,
macro tests and practice), `practice_steps`, `failed_exec_steps` (presses of
replays, macro tests and practice runs stopped by a slip, including the
slipped press), `executions` with their outcomes, `skills_admitted`, and
`skill_exec` (one entry per execution: concept, level, execution index n,
outcome, presses, kind). Reported per arm and depth: the shares of all actions
spent executing skills, practising and in failed executions
(`skills.execution_shares`), and per-skill reliability curves, the share of
`ok` among `ok` and `slip` executions by execution index and skill level
(`skills.reliability_curve`). The analysis sums `exec_steps`, `practice_steps`
and `failed_exec_steps` per world into its tables and stores both summaries
under `row_summaries` in `analysis.json`.

### Predictions (written before any measurement run)

The bullets below were written before the switch of the primary and are kept
as written. Their labels are those of the first draft: P1 was the gated
contrast at depth 2 (now S1), S1 the ungated one (now P1), C1 the gated
control at depth 0 (now S1b) and S10 the gated contrast on the whole stream.
The swapped labels carry the same predictions, so the bullets read the same
under the current labels: P1 and S1 both hold; the gated control at depth 0
holds (C1 then, S1b now), and the ungated contrast at depth 0, the current C1,
was predicted above 0 under S5, so C1 holds too; S10 holds for either arm.
The predictions for S13-S15, added with the switch, follow the bullets.

Cost model. A macro test of a level-3 hypothesis types a 4-press and a 2-press
skill, so at e0 = 0.3 its cost falls from about 25 presses for fresh skills to
about 8 after 30 executions (table above). Promotion's macro search is short
(about 5 level-2 and 14 level-3 tests to find everything at depth 2), so its
execution overhead is paid in the first two to four episodes of a world. But
`remember` executes skills too: before every search step above level 1 it holds
every concept of the parent levels (sure gating), about 18 presses per
searching episode when exact, and it keeps searching for longer. Execution
errors therefore slow both arms.

- e0 = 0 (C2, S6, S8): L2 reproduces on fresh worlds. Depth 2: `promote` about
  11 late actions against about 28 for `remember`; depth 0: `promote` 2-6 worse
  than `remember`. `promote_gate` pays about 20-40 practice presses once per
  world, mostly before the late half, so it equals `promote` in the late half.
- Depth 2, e0 = 0.3 (P1, S1, S3, S9): late-half `promote` and `promote_gate`
  about 13-17 (the goal plan of four or so skills executed with late errors of
  about 0.08-0.12), `remember` about 32-40 (costlier sure-gating replays leave
  fewer search presses per episode). P1 and S1 hold with differences of about
  -15 to -25. The gate recovers nothing at depth 2 (S3 fails), because
  promotion keeps its advantage: there is little to recover.
- Dose-response at depth 2 (C2, S2, S1 side by side). The audit's expectation
  is that the promotion advantage shrinks as e0 grows. The cost model predicts
  that the absolute advantage does not shrink (the estimates at e0 = 0, 0.1 and
  0.3 within about 5 actions of each other, or growing), because remembering's
  replays are executions as well and are more numerous; only the relative
  advantage shrinks slightly. The formal test of the difference across error
  levels needs an analysis extension (Implementation); until then it is
  reported descriptively.
- Depth 0, e0 = 0.3 (C1, S4, S5): every macro test is futile and now costs
  about 11-25 presses for a level-3 hypothesis instead of 6, about 200-450
  presses per world before the primitive search resumes. The utility problem
  appears here: `promote_e30` worse than `remember_e30` by about 5-15 late
  actions (S5 holds). `promote_gate_e30` delays the futile tests until its
  skills are admitted and runs them with lower error, at the price of practice:
  between the two, nearer to `promote`; S4 (gate better than `promote`) is
  predicted in direction only and is likely inconclusive. C1 holds (no benefit
  at depth 0). At e0 = 0.1 the same effect is smaller (S12, direction only).
- S7 (gated benefit larger at depth 2 than 0, e0 = 0.3) holds; S10 (P1 on the
  whole stream) holds; S11 (P1 at e0 = 0.1) holds.
- Execution shares at e0 = 0.3: roughly a third of all actions of `promote` and
  `promote_gate` at depth 2 lie in executions stopped by a slip, against about
  an eighth for `remember`; practice takes about 5-10% of the gated arm's actions.
- Validity: the reference and the oracle library solve every task, and the
  oracle library never beats the reference (it executes exactly, as in L).

Added with the switch (no new numbers; implied by the bullets above): S13, the
advantage at depth 2 shrinking from e0 = 0 to e0 = 0.3, is predicted to fail
(the absolute advantage does not shrink); S14, the depth-0 cost of promotion
growing from e0 = 0 (2-6 actions) to e0 = 0.3 (5-15 actions), is predicted in
direction only; S15, the ungated benefit larger at depth 2 than at depth 0 at
e0 = 0.3, holds.

### Primary contrast, falsification and verdict

Primary P1: `promote_e30 - remember_e30` on `steps_late` at depth 2, paired
by world; holds if the 95% bootstrap CI lies below 0.

Falsification F1: the 90% CI of the same difference lies within +-10% of
`remember_e30`'s mean `steps_late` at depth 2 (promotion equivalent to
remembering).

Conditions for reading P1 as an effect of promotion with reuse: C1 (no benefit
at depth 0: the CI of `promote_e30 - remember_e30` at depth 0 does not lie
entirely below 0) and C2 (study L reproduced on these worlds:
`promote_e00 - remember_e00` at depth 2 has a CI below 0).

Verdict (`analysis.verdict`, rule `precedence`), applied in this order:

0. If P1, F1 or a validity check (V1-V3) has no verdict because a run it reads
   is incomplete, the study has no verdict. If P1 holds without F1 and C1 or C2
   has no verdict, the study has no verdict.
1. A failed validity check fails the study.
2. P1 holds and F1 does not: pass if C1 and C2 hold, otherwise "confounded".
3. P1 and F1 both hold: the benefit is declared negligible (not a pass).
4. F1 holds and P1 does not: falsified (no benefit).
5. Neither holds: inconclusive.

Power note: as in L, with 32 worlds the equivalence test can only succeed if
the per-world differences vary little (a standard deviation of about 10
actions or less for a true difference near 0).

### Secondary contrasts (reported; they qualify but do not decide the verdict)

All on `steps_late`, paired by world, unless stated.

- S1: `promote_gate_e30 - remember_e30`, depth 2 (gated promotion pays; CI
  below 0). The primary of the first draft.
- S1b: `promote_gate_e30 - remember_e30`, depth 0 (no benefit of gated
  promotion without reuse; CI not entirely below 0). C1 of the first draft.
- S2: `promote_e10 - remember_e10`, depth 2 (CI below 0).
- S3: `promote_gate_e30 - promote_e30`, depth 2 (the gate recovers part of a
  lost benefit; CI below 0). Predicted to fail.
- S4: `promote_gate_e30 - promote_e30`, depth 0 (the gate reduces the cost of
  useless unreliable nodes; CI below 0). Predicted in direction only.
- S5: `promote_e30 - remember_e30`, depth 0 (the utility problem; CI above 0).
- S6: `promote_e00 - remember_e00`, depth 0 (L's control reproduced; CI not
  entirely below 0).
- S7: interaction, `promote_gate_e30 - remember_e30` at depth 2 minus depth 0
  (paired by world across depths; CI below 0).
- S8: `promote_gate_e00` equivalent to `promote_e00` at depth 2 within +-10%
  (the gate's late overhead when skills are exact; 90% CI). The margin is about
  one action, so a failure is weakly informative.
- S9a, S9b: `promote_e30 - promote_e00` and `remember_e30 - remember_e00` at
  depth 2 (the cost of imperfect skills for each arm; CI above 0).
- S10: P1 (`promote_e30 - remember_e30`, depth 2) on `steps_all`.
- S11: `promote_gate_e10 - remember_e10`, depth 2 (CI below 0).
- S12: `promote_e10 - remember_e10`, depth 0 (the utility problem at the lower
  error level; CI above 0). Predicted in direction only (about +3 to +8).
- S13: interaction across error levels at depth 2:
  (`promote_e30 - remember_e30`) minus (`promote_e00 - remember_e00`), paired
  by world (the advantage shrinks as skills become unreliable; CI above 0).
- S14: the same interaction at depth 0 (the cost of useless promoted nodes
  grows with their unreliability; CI above 0).
- S15: interaction across depths at e0 = 0.3, `promote_e30 - remember_e30` at
  depth 2 minus depth 0 (paired by world across depths; CI below 0).
- Validity: V1 the reference and V2 the oracle library solve every task; V3 the
  oracle library never beats the reference.
- Descriptive: the dose-response at depth 2 (estimates of C2, S2 and P1 side by
  side, tested by S13); execution, practice and failed-execution shares per arm and depth;
  reliability curves by skill level; `discoveries_in_replay` and
  `discoveries_in_practice` (incidental composition); `skills_admitted`. Cells
  read by no test, descriptive only: `promote_gate_e10` and `promote_gate_e00`
  at depth 0.

### Statistics

As in L: worlds are the units; contrasts are paired by world (and across depths
for the interaction, since worlds are coupled); 10,000 percentile bootstrap
resamples of worlds with a fixed seed (20261008); 95% intervals (90% for
equivalence); two-sided exact sign tests over worlds reported alongside. No
multiplicity correction for secondaries.

### Incomplete or repeated runs

As in L. Every (depth, arm, seed) must have exactly one complete run: every
world with all 12 episodes and no stop by the session budget. A cell with a
missing or incomplete run is flagged and every test that reads it has no
verdict; the verdict then follows step 0 above. A second run of the same
(depth, arm, seed) is an error, unless the earlier run is incomplete and the
later one has the same run hash (which includes the `agent` block) and was made
by the same code (the same loaded package code and commit, from the run
manifests): a deterministic rerun, which is then used and listed. A rerun after
a code change is refused and needs a dated amendment. The runner refuses to
start a measurement config whose runs already exist complete, unless
`--resume`, which repeats only missing or incomplete runs and is itself refused
when the run to repeat was made by other code.

### Limits stated in advance

- The skill model is a stylized controller (independent errors per press, a
  fixed hyperbolic learning curve, stop on a slip), not a trained policy; its
  parameters are set, not learned, and the results hold for this model only.
- Only the execution of known sequences is imperfect. Discovery itself stays
  exact and evidence never misleads; noisy evidence is study R's subject.
- A macro hypothesis is still the concatenation of stored sequences, so P
  inherits L's limit: promotion to a higher-layer node and composing remembered
  sequences cannot be separated.
- One gate rule (2 of the last 3, admission once, practice when needed) is
  tested. Other gates (separate validation episodes, demotion, admission by
  expected value) are not; a failure of this gate says nothing about gates in
  general.
- No oracle with imperfect execution is run. The oracle library executes
  exactly, so it bounds the cost of knowledge, not of execution, and the gap
  between an arm and the oracle library mixes missing discoveries with
  execution errors; the execution overhead is read only from the execution
  shares and the reliability curves.
- Depth 1 is not run; toy domain; hand-written elimination agents.
- The development look (one world per depth) and a mechanism smoke on two
  unit-test worlds per depth were seen before the predictions were written.

### Budget

Ledger `runs/P-ledger.json` (created by the first development run on
2026-10-08): allocations `dev` 20,000 and `p_main` 691,200 adaptive; adaptive
cap 711,200; reporting cap 173,600 (the measurement reporting caps plus 20,000
for development). From the dry runs:

| Item | Runs | Per-run cap | Cap total | Expected use |
|---|---|---|---|---|
| Adaptive arms (nine per depth, depths 0 and 2) | 144 | 4,800 | 691,200 | about 280,000-400,000 |
| Reporting arms (oracle library, reference; depths 0 and 2) | 32 | 4,800 | 153,600 | about 16,000 |
| Development (used, exact) | 12 | 1,200 | 14,400 | 3,917 adaptive + 508 reporting |

A run's cap is the sum of its 48 episode budgets, so no run is cut short.
Expected use: the development use per world (490 adaptive, 127 reporting)
scaled to 144 and 32 runs of 4 worlds gives about 282,000 adaptive and 16,000
reporting interactions; L2's use per run (2,382 adaptive) gives about 343,000
before the extra cost of the error levels. Expected runtime on one CPU core: about 3 minutes
(evaluation loops about 50 s at the development mean of 6.7 ms per adaptive
episode and under 1 ms per reporting episode, plus run bookkeeping and
provenance recording of under 1 s per run).

Commands (after the freeze):

```
python -m hypergraph_agent.techtree --config configs/skills/p_main.yaml --allow-measurement
python -m hypergraph_agent.techtree.analysis --config configs/skills/p_main.yaml
```

A measurement config runs whole (subsets are refused); `--resume` repeats only
missing or incomplete runs.

### Stopping rules

The measurement config is run once, completely, after the freeze. No interim
analysis, no change of parameters, arms or endpoints after any measurement row
exists. Incomplete runs are handled as above. If a validity check fails, the
study fails.

## Implementation

- `techtree/skills.py`: the skill model (`SkillMixin`: controller draws,
  per-world skill records, stop on slip, execution log), the arms
  (`SkilledRemember`, `SkilledPromote`, `PromoteGate`, built on L's `Explorer`
  and `Promote`), the factory `make_skilled(arm, rng, params)` registered for
  study P in `techtree/study.py`, and two row summaries declared in
  `ROW_SUMMARIES` (`execution_shares`, `reliability_curve`).
- Shared analysis code (generic, backward compatible): `techtree/analysis.py`
  sums the row keys listed in `ROW_SUMS` per world when every row of the world
  carries them (the three execution counts were added to L's two discovery
  counts), stores the results of a study module's `ROW_SUMMARIES` under
  `row_summaries`, and lets an `interaction` test read other arms in its low
  cell (optional `a_lo` and `b_lo`, default `a` and `b`; S13 and S14 use them).
  `techtree/config.py` accepts the two keys and checks that they name
  configured arms and that an interaction never compares a cell with itself.
  Re-running the analyses of studies U, L1 and L2 with this code reproduces
  their published `analysis.json` files byte for byte.
- Configs: `configs/skills/p_main.yaml` (measurement draft) and
  `configs/skills/p_dev.yaml` (development); `tests/test_hygiene.py` lists
  `skills` among the configuration folders with their own schema.
- Tests: `tests/test_tt_skills.py` (arm names and parameters; the learning
  curve and the per-press slip rate in play; a slip presses another available
  primitive, stops and records no failure clause, while an exact typing still
  records a true one; common random numbers across arms and error levels; e0 = 0
  reproduces L's `remember` and `promote` at depths 0-2; sound evidence for
  every arm at e0 = 0.3 and 0.6; every press charged and attributed, practice
  under `skill_practice`; the gate's bookkeeping, recomputed from the execution
  log, with atoms used only after admission and practice only when no admitted
  macro hypothesis is open; the row summaries; the agent-boundary source check;
  configs, ledger blocks, the run hash and an end-to-end run of study P with
  its analysis; the interaction low cell, its validation, and the row
  summaries being absent for studies U and L).
- Still to do in shared documents before the freeze: a row for P in
  `docs/studies/README.md` and the corresponding entries in README.md,
  docs/DECISIONS.md and docs/METHODS.md Section 12.

### Development record (calibration, not results)

Development ran `configs/skills/p_dev.yaml` once (12 runs: `remember_e30`,
`promote_e30`, `promote_gate_e30`, `promote_gate_e00`, the oracle library and
the reference at depths 0 and 2, one world of 12 episodes each). It used 3,917
adaptive and 508 reporting interactions of the P ledger (112 of the adaptive
ones were practice). Late-half actions per episode in that single world:

| Depth | remember_e30 | promote_e30 | promote_gate_e30 | promote_gate_e00 | oracle library | reference |
|---|---|---|---|---|---|---|
| 0 | 35.5 | 35.2 | 45.7 | 20.2 | 13.3 | 13.0 |
| 2 | 19.5 | 15.2 | 16.3 | 9.7 | 8.7 | 7.0 |

Late success was 0.83 for `promote_gate_e30` at depth 0 and 1.0 elsewhere, so
e0 = 0.3 does not put the task at the budget floor. Shares of all actions in
executions stopped by a slip at e0 = 0.3 and depth 2: 0.38 (`promote`), 0.32
(`promote_gate`) and 0.12 (`remember`); practice took 4-17% of the
gated arms' actions. One world per depth is dominated by where each arm's
search meets the level-3 sequences (`remember` at depth 2 found its goals
early in this world), so these numbers say nothing about the contrasts.
Earlier, a mechanism smoke on two worlds per depth of the unit-test namespace
`unit`, in memory and outside the ledger, used 11,539 interactions in 12 runs
of the skilled arms (outcomes seen) and 6,794 in 8 runs checking that e0 = 0
reproduces L (identity only). No parameter was changed after either look.
After the switch of the primary, the dev config's analysis section was
updated (the new P1, F1 and C1, and one interaction with `a_lo` to exercise
the new key) and the existing dev runs were analysed again; that used no
interactions.

## Results

Not run. The protocol is a draft.

## Verdict and limits

Pending. The limits stated in advance are listed in the protocol.
