# Study R: replication of the Stage D amortization result

Status: protocol frozen on 2026-10-08 by the commit that adds this file,
before any measurement run; the measurement runs execute that commit from a
clean checkout. The development checks listed under Implementation used only
the `dev` allocation and the `replication_dev` namespace (plus the Stage D
worlds for the reproduction check).

## 1. Question and claim

Stage D found that walkers which sample among the hypotheses consistent with
the evidence (`focused_sample`) become cheaper than the brute-force plan
(`maximal`, planning on every candidate prerequisite) over episodes 9-16 of a
world, after paying for exploration in the first episodes. Study R asks
whether that result holds at scale: on many more worlds, each an independent
unit, on a second task family that differs in structure, and under random
action failures.

The claim under test is narrow: in worlds whose tasks reuse recipes, sampling
among hypotheses weighted by the evidence makes later episodes cheaper than the
brute-force plan. Two controls probe the explanation rather than the effect: a
family without recipe recurrence (evidence can never be reused) and a random
omission control that explores at the sampler's rate but ignores the evidence.

## 2. Relation to earlier work

Stage D (docs/RESULTS.md, Section 4.4; protocol W-D in docs/EXPERIMENT_PLAN.md)
ran five worlds x 16 episodes and two seeds that shared those worlds. Over
episodes 9-16 `focused_sample` needed 1.55 and 1.44 steps per optimal step
against 1.70 for `maximal`; the learned edit policy did not contribute. The
worlds were the only independent units, so no interval was interpreted.
docs/ROADMAP.md item 6c proposes this replication with more worlds, seeds,
families and failure noise, and names sampling among consistent nodes as the
default to build on. Study R therefore drops `learned_sample` (trained policies
exist only for seeds 0 and 1, and the learned component did not help in Stages
C and D).

Prior work on noisy precondition learning, group testing and posterior
sampling, and how it shapes the controls below, is reviewed in
docs/PRIOR_ART.md, Section 8.1.

## 3. Protocol (frozen)

### Families

| Family | Worlds | Goals | Per run | Unit |
|---|---|---|---|---|
| F1 | the Stage D generator: 8 levels x 2 items, pools of 6, true requirement of 1-3 (41 hypotheses per recipe), alternative recipe with probability 0.25 | depth 1-3 | 10 worlds x 16 episodes | world |
| F2 | pools of 8 with true requirement of 1-3 (92 hypotheses per recipe), alternative recipe with probability 0.6, second item input with probability 0.25 | depth 3-5 | 10 worlds x 16 episodes | world |
| F0 | the F1 generator with a fresh world for every task (no recurrence) | depth 1-3 | 10 blocks x 16 tasks | block of 16 consecutive tasks |

Common to all: hidden prerequisites (`unknown_prerequisites`), intermediate
items unobserved (`observe_items: goal_only`), no distractor base facts,
budget = brute-force bound + 2. Each family has its own evaluation namespace
(`rep_f0`, `rep_f1`, `rep_f2`, never used before), so no task or world label is
shared across families. The namespaces were renamed from `replication_f0`,
`replication_f1` and `replication_f2` before the freeze because a pre-freeze
review run touched tasks of the earlier namespaces; its outputs were discarded
unread. Seed s plays evaluation base seed 10000 + 1000 s
(`eval_world_offset_by_seed`), so the five seeds play 50 distinct worlds
(blocks in F0) per family. The two noise levels of a seed play the same worlds
and tasks.

F2 was chosen from generator statistics alone, before any development run. It
changes the axes the amortization explanation depends on: hypothesis entropy
(pools of 8 instead of 6), identifiability (more alternative recipes, OR
structure) and blame ambiguity. A second item input makes a recipe need two
earlier items, so a derivation is a tree rather than a chain and a failed goal
can be blamed on more recipes; the generator supports it as is. The suggested
depth 4-6 was feasible but has about 9% less brute-force headroom than F1, which
would shrink any gain for a trivial reason; depth 3-5 keeps the headroom of F1
while more than tripling the hypothesis entropy per task. Pools shared between
recipes were not added: the generator has no such option, pools drawn
independently from 18 base types already overlap (about 2 shared candidates
for two pools of 6, 3.6 for two pools of 8), and with goal-only observation the
posterior is already coupled across recipes. Statistics over the 800
measurement tasks of each family (generator and reference solver only):

| | F0 | F1 | F2 |
|---|---|---|---|
| optimal length, mean (range) | 6.8 (3-12) | 6.7 (3-12) | 11.0 (6-17) |
| hidden recipes per task, mean (range) | 3.8 (2-11) | 3.7 (2-9) | 10.1 (4-18) |
| alternative recipes per task | 0.55 | 0.57 | 3.45 |
| hypothesis entropy per task, bits | 20.1 | 19.7 | 65.6 |
| distinct recipes per world (block) over 16 episodes | 60.0 | 8.9 | 17.5 |
| recurrence: share of a task's recipes seen earlier in its world, episodes 2-16 (9-16) | 0 (0) | 0.91 (0.98) | 0.95 (0.99) |
| headroom: brute-force plan / optimum, per-task mean (pooled) | 1.91 (1.84) | 1.91 (1.83) | 1.88 (1.84) |
| budget, mean (range) | 19.5 (13-27) | 19.4 (12-27) | 27.4 (22-31) |

Every reference is exact, no budget reaches the cap of 256, and the brute-force
plan fits every budget, so with failure probability 0 `maximal` cannot fail
(docs/DECISIONS.md W2). Under noise it can run out of budget.

### Arms

| Arm | Kind | Description |
|---|---|---|
| `focused_sample` | adaptive | the focused walk to a node that explains every observation, then 50 Metropolis moves among such nodes; under noise the moves target the posterior (Section 4) |
| `random_omit` | adaptive (F1, F2) | the full-pool node with every candidate omitted independently with probability q, redrawn at every replan; ignores the evidence, plans and replans like the other arms |
| `maximal` | reporting | the full-pool node; under noise it retries with the same beliefs and replanning |
| `reference` | reporting, privileged | the optimal plan from the hidden rules, replanned from the true state after a failed step |

q matches the sampler's exploration rate. The target is the fraction of pool
candidates omitted by `focused_sample`'s nodes, pooled over every replan of
episodes 9-16 in the development run of the same family at failure
probability 0: 658 of 1,014 (0.649) in F1 and 2,807 of 4,056 (0.692) in F2.
When a draw omits a whole pool, `random_omit` keeps one candidate drawn
uniformly (requirements are publicly nonempty), so the realized omitted
fraction of a pool of n is q - q^n / n. q solves that equation for the
target: q = 0.663 in F1 (n = 6, realized 0.649) and q = 0.699 in F2 (n = 8,
realized 0.692).

### Noise levels

Failure probability 0 and 0.1 (eval variants `eps0`, `eps10`). An eligible
gather, activation or craft fails with that probability and leaves the state
unchanged. The value is declared to every agent and used by it.

### Endpoints

For each unit and arm, the late-half cost ratio: primitive steps over optimal
steps, summed over episodes 9-16 of the world (positions 9-16 of the block in
F0); the early half is episodes 1-8. A failed episode counts with every step it
used, which is its whole budget, so under noise, where `maximal` can also fail,
the cost endpoint can be driven by success as well as by plan length. The
optimum is the deterministic reference length in every noise level. The
contrast is the paired per-unit difference `focused_sample` minus `maximal`.

**Primary**: F1 at failure probability 0, the mean paired late-half difference
over the 50 worlds. It passes if the upper end of the 95% world-cluster
bootstrap interval is below 0.

**Secondaries**, each with the same criterion and reported separately: F1 at
0.1, F2 at 0, F2 at 0.1. The replication is called general only if all four
F1 and F2 cells pass. Reported with each: the paired per-unit difference in
late-half success, with its interval.

**Amortization check (pre-registered).** For every unit, the
difference-in-differences D = [(`focused_sample` - `maximal`) late] -
[(`focused_sample` - `maximal`) early], with a world-cluster bootstrap
interval. The amortization explanation is supported if the interval of D lies
below 0 in F1 at failure probability 0 and includes 0 or lies above 0 in F0 at
failure probability 0. The same is reported at 0.1. The late-half difference
in F0 is reported descriptively.

**Random omission.** `focused_sample` minus `random_omit` in every F1 and F2
cell, with the same criterion. Passing shows that using the evidence beats
exploring at the same omission rate without it. It does not show that the
posterior weighting itself matters: a control that uses the evidence without
posterior weights, such as an adaptive group-testing elimination rule, is
outside this study.

**Descriptive**: the break-even episode of every cell, the first episode index
e such that the pooled per-episode cost ratio of `focused_sample` is at or
below that of `maximal` for every episode from e to 16, or "never" if there is
no such e (when cells are compared, "never" counts as later than any episode);
early-half and overall cost ratios; success over all and late episodes; cost
ratio by episode index; per-seed late ratios; omission rates; exact two-sided
sign tests on the per-unit late differences. As an order-of-magnitude
expectation, not a criterion: with failure probability 0.1 the advantage may
persist but break even later, roughly twice as late judging from the capacity
of the noisy observation channel; in F2 a later break-even than in F1 is
expected.

### Statistics

The unit is a world (a block in F0). Seeds change only the walker's random
stream and which worlds it plays; worlds are not shared across seeds. The
interval is a percentile bootstrap over units (10,000 resamples with
replacement, seed 20261008, the same resamples for every statistic of a
contrast), each unit's value held fixed. There is one primary contrast;
secondaries, the amortization check and the random omission contrasts carry
their own verdicts and no multiplicity correction, and generality requires all
four F1 and F2 cells. Analysis: `python -m hypergraph_agent.analyze_replication
--runs runs --prefix rep- --out artifacts/replication` (a contrast needs 50
paired units by default).

### Budget

Ledger `runs/R-ledger.json`, declared identically in every config of
`configs/replication/`. A run's cap is the largest sum of task budgets among
the ten (seed, variant) streams of its family, so no cap can cut a run short
(no episode exceeds its budget). The `rep` allocation covers the sum of the
stream budgets of all adaptive runs, and the reporting cap that of all
reporting runs, so the ledger cannot cut a run short either.

| Item | Runs | Cap per run | Worst case (sum of stream budgets) | Expected (development rates) |
|---|---|---|---|---|
| F0 `focused_sample` | 10 | 3,145 | 31,248 | 24,300 |
| F1 `focused_sample`, `random_omit` | 20 | 3,159 | 62,068 | 43,300 |
| F2 `focused_sample`, `random_omit` | 20 | 4,449 | 87,820 | 82,100 |
| **Adaptive, allocation `rep`** | 50 | | 181,136 (allocated 182,330) | 149,700 |
| F0 bounds | 20 | 3,145 | 62,496 | 29,200 |
| F1 bounds | 20 | 3,159 | 62,068 | 29,000 |
| F2 bounds | 20 | 4,449 | 87,820 | 53,800 |
| **Reporting** | 60 | | 212,384 (217,774 left of the 221,860 cap) | 111,900 |
| Development, allocation `dev` | 21 | | 10,000 allocated | 6,928 adaptive + 4,086 reporting used |

Adaptive cap 192,330 (`dev` 10,000 + `rep` 182,330); reporting cap 221,860.
Expected values scale the interactions per task of the development runs to
1,600 tasks per arm and family; the F0 reference was not run in development
and is estimated from the F1 rate. A rerun (below) is charged again; the
difference between expected and worst-case use leaves room for it.
Development rates put the whole study at about 20 minutes of computation on
one core (at most 0.6 s per task, F2 under noise); seeds can run as separate
processes (`--seeds`).

### Stopping rules

Every run plays its whole stream; the caps cannot bind. Exactly one kind of
rerun is allowed: a run that did not complete (an error or an interruption)
may be rerun with the same commit, config hash, seed, arm and variant. Runs
are deterministic, so the rerun replays the same stream; it is used and both
runs are listed. A completed run is never replaced, and any other repeat is
ignored and listed. A contrast that still needs an incomplete run, or has
fewer than 50 paired units, is reported as incomplete and does not pass.
Results are analyzed only after every run of a cell has finished; no arm,
parameter, family or criterion changes after the freeze.

### If the primary fails

The estimate, its interval, the per-world differences and the break-even
episode are reported, and the conclusion is that the Stage D advantage did not
replicate in F1 at scale. Secondaries and controls are still reported, without
any claim of generality, and the Stage D statement in the README is qualified
accordingly.

### Not included

Exact Thompson sampling by enumeration (infeasible with goal-only evidence
beyond small evidence sets), a misspecified failure probability, and an
adaptive group-testing elimination heuristic. These limit what a failure of the
controls can be attributed to.

## 4. Implementation

- Episode logs record whether each gathered or activated base fact was
  observed present; replays use the observed outcome instead of assuming
  success.
- `filter_states` (agents/walker.py) computes the exact likelihood of a logged
  episode under a hypothesis with known failure probability: a forward filter
  over the unobserved items in which every eligible craft whose effect is
  absent branches into success and failure, conditioned after every step on
  the goal (and on a craft's effect when items are observable), with identical
  states merged; the log-likelihood is the sum of the log normalizers.
  Likelihoods are memoized per log, prefix length and the hypothesis
  restricted to the log's attempted recipes. At failure probability 0 the
  likelihood is positive exactly when the replay finds no violation (tested on
  random hypotheses and logs), and under noise it equals the sum over all
  failure patterns (tested against brute-force enumeration).
- Under noise an observed fact the hypothesis cannot produce is impossible,
  while a predicted fact that did not appear is explained by a failure (a
  failure leaves the state unchanged). The walk phase therefore counts only
  impossible observations as violations and focuses its edits on them; the
  consistent moves target prior x likelihood with the neighbourhood-size
  correction (a sampling test on a 41-node space matches the posterior within
  0.05 total variation). At failure probability 0 they make the same moves and
  random draws as before (tested).
- Planning beliefs about hidden items are their filtered probabilities under
  the chosen node, thresholded at 0.5 (at failure probability 0 identical to
  the earlier replay, tested on every prefix of many logs); a visibly failed
  gather or activation clears the plan. `maximal` uses the same logic, which
  lets it recover from hidden failures instead of repeating the last craft.
  The other episodic walkers still refuse noisy dynamics.
- `random_omit` (agents/walker.py) with the keep-one rule (realized rate
  tested against q - q^n / n), and per-episode counts of omitted pool
  candidates for every walker.
- The privileged reference replans from the current true state after a step
  that left it unchanged (`reference_solve(task, held=...)`).
- Runner: a walker now receives the failure probability of the stream it
  plays (the evaluation variant's); before, it received the training stream's
  value. No recorded stage varied the failure probability through a variant,
  so no recorded result is affected. `eval_world_offset_by_seed` gives each
  seed its own worlds; the dry run reports per-stream budget sums; run
  summaries of fresh-world streams split halves by block position. Study M's
  agents (`agents/markov.py`) are imported only when one of their strategies
  is requested, and their names may not shadow walker strategies; study R
  imports and runs with that module absent or failing (tested).
- `evaluation/replication.py` and `python -m hypergraph_agent.analyze_replication`
  compute the endpoints above, apply the rerun rule and write a JSON summary,
  a Markdown table and a cost-by-episode figure.

Equivalence and development checks:

- At failure probability 0 every walker strategy made the same decisions as
  the previous code on 840 test-fixture episodes (42 strategy and stream
  combinations, in-memory ledgers).
- Reproduction (`configs/replication/dev_stage_d.yaml`): Stage D's
  `focused_sample` seed 0 on its 80 tasks matched the recorded run exactly in
  primitive length and success per episode (and in search evaluations,
  replans and moves); 942 interactions, as recorded.
- Smoke runs (`configs/replication/dev_f0.yaml`, `dev_f1.yaml`, `dev_f2.yaml`;
  two worlds or blocks x 16 episodes, seed 0, both noise levels) finished
  every task in every arm. These are development observations on two units,
  not evidence: F1 late ratios 1.61 for `focused_sample` against 2.02 for
  `maximal` (1.78 against 2.18 under noise); F2 2.21 against 1.96, with
  `focused_sample` solving 41% of the tasks (25% under noise), because one
  wrong guess on a chain of 3-5 recipes costs more than the budget's slack;
  F0 2.25 against 1.71. They suggest that F2 may not break even within 16
  episodes. F2 and every other setting were left as chosen before these runs;
  only q was set from them. The `random_omit` smoke runs used q = 0.65 and
  0.69, before the keep-one correction; the configs now hold 0.663 and 0.699.
- Runtime per task (development, one core): `focused_sample` 0.09 s (F1),
  0.44 s (F2) and 0.04 s (F0) at failure probability 0, and 0.11, 0.59 and
  0.05 s at 0.1; the other arms 0.01 s or less.

## 5. Results

Not yet run.

## 6. Verdict and limits

Not yet available.
