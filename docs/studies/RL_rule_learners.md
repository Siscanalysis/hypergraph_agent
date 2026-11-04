# Study RL: smallest consistent rules against the exact posterior

Status: protocol frozen on 2026-11-04 by the commit that freezes this file, before any measurement
run, after an independent review (ready to freeze after fixes, no code blocker; both solver
encodings matched brute force on 600 problems); the measurement runs from a clean worktree of that
commit. The ledger caps were raised to the declaration below by a logged amendment before the
freeze. This is experiment RB-A of docs/studies/RB_rule_builder_design.md (roadmap item 6i), a
follow-up of study X2. Development used
only the allocation `dev` of `runs/RL-ledger.json`, namespaces ending in `_dev` and study X2's
development logs; its outcomes were seen before the predictions below were written and are reported
under Implementation as calibration, not results. The measurement namespaces (`rl_slip`, `rl_ooc`)
have never been used, and no learner of this study has read X2's measurement logs.

## Question and claim

Question: does exact search for the smallest rule consistent with all successes and failures match
the repository's exact posterior on logs whose hidden requirements lie inside the posterior's
hypothesis class, and does it win where that class is misspecified (requirements larger than the
class allows) or the evidence is noisy (slips)?

Claim under test (one primary): on generated worlds whose requirements have 4 or 5 of their 8
candidates, outside the posterior's class of at most 3, after 16 episodes of random-omission
evidence, the inferred sets of the smallest consistent DNF have a higher F1 against the hidden
requirements than the posterior's, with four pre-stated outcomes (DNF better, posterior better,
equivalent within 0.05 F1, inconclusive). The in-class comparison on X2's logs and the slip variant
are secondary tests.

The study is offline, as X2: traces are collected once with X2's fixed policies, every learner reads
the same logs and the same public recipe descriptions, and no learner chooses actions. X2's verdict
stands as published; RL reads X2's logs and adds learners in its own protocol.

## Relation to earlier work

- Study X2 (docs/studies/X2_inference_baselines.md, frozen at 12372ad, measured 2026-10-28,
  results committed in e4a9878) compared the posterior with MSGI, macq's Observer and ARMS, and a
  SAM-style learner on identical logs. RL reads X2's logs of set `gen_obs` and, descriptively, of
  its five benchmark sets (read only, through X2's loader), scores every learner with X2's code and
  reuses X2's collection machinery for its new logs. X2's results were known before the predictions
  below were written (its measurement report was read on 2026-10-29, before they were committed): on
  X2's 36 test worlds the posterior had AUROC 0.908, F1 0.809 and acting cost ratio 1.19 with
  `focused_sample` evidence at 16 episodes, and Observer's conservative sets acted more cheaply
  open-loop (1.09; X2's S5 reversed). RL's in-class tests use all 60 `gen_obs` worlds, X2's
  selection worlds included.
- Group testing (docs/PRIOR_ART.md, N6; Sections 8.1 and 12). With items observed, inference for one
  recipe is group testing: an attempt fails exactly when its omitted candidates contain a required
  one. Slips make a failure possible without a missing requirement and never the reverse (N6's
  addition-noise model), so successes stay reliable and only failures are soft; RL's soft learner
  encodes exactly that. With one term, the smallest consistent rule is the smallest satisfying set
  (SSS) decoder of noiseless group testing (N6, Section 2.2), which rests on defectives being rare
  and breaks ties arbitrarily; RL's learner reports every tie instead, and out of class, where 4 or 5
  of 8 candidates are required, its premise fails. X2's Observer and SAM rule is COMP (N6, Section
  2.3), which never declares a required candidate absent, the reason its sets never made a plan fail
  in X2.
- Version spaces (W3, Mitchell 1982): the posterior's support is the version space of its class; the
  smallest consistent rule is one member of a wider class, chosen by size.
- Noisy rule and operator learning (N1, Pasula, Zettlemoyer and Kaelbling, JAIR 2007; N2, Mourão,
  Zettlemoyer, Petrick and Steedman, UAI 2012) learn preconditions under noise; MSGI (R4) decodes a
  CART into a sum of products, the convention X2 used to score rules; ARMS (IB2) is a weighted
  MAX-SAT learner of action models.
- Exact and classical rule learning (docs/PRIOR_ART.md, Section 12, E1 to E11, checked 2026-11-04):
  Yu, Ignatiev, Stuckey and Le Bodic, optimal decision sets and lists with SAT (E1), the closest
  exact method, which also measures a model's size by its literals, as RL's hard learner does;
  CORELS (E2); Bayesian rule sets (E3), a prior over rule sets with a MAP search, the role of RL's
  description-length prior (E11) and slip likelihood; RIPPER (E4), CART (E5) and FIGS (E6); clingo
  (E7), RC2 (E8) in PySAT (E9); genetic programming (E10), as implemented by gplearn.
- Mixed requirement sizes (some recipes inside the class, some outside) are the question of the
  design note's RB-E (arbitration between modules); RL uses sizes 4 or 5 only.

docs/PRIOR_ART.md records no comparison, on matched logs of one hidden-prerequisite task, of an
exact smallest-rule learner with an exact posterior under class misspecification and slips; that
is what this study adds.

## Protocol (frozen)

### Settings

| Set | Logs | Worlds (seeds x worlds) | Requirements | Slips | Role |
|---|---|---|---|---|---|
| `gen_obs` | X2's `x2-main-gen_obs-*`, read only | 0-4 x 12 (60) | 1-3 of 6 | none | in class |
| `slip` | new, namespace `rl_slip` | 0-4 x 12 (60) | 1-3 of 6 | 0.1 | noisy |
| `ooc` | new, namespace `rl_ooc` | 0-4 x 12 (60) | 4 or 5 of 8 | none | out of class |
| X2's benchmark sets | X2's `x2-main-bench_*`, read only | 90 worlds | 1-3 of 6 | none | descriptive |

Every generated set is family F1 as in X2 (study R's Stage D distribution: goals of depth 1-3, no
distractor base facts, items observable, 16 episodes per world in world-major order, seed s plays
base seed 10,000 + 1,000 s). The variants use the generator's existing options and change no module:
`slip` sets `TaskConfig.failure_prob` to 0.1 (an eligible state-changing action fails and leaves the
state unchanged; a failed gather is visible, a slipped craft looks like a missing requirement) and
keeps X2's pools of six and size weights 0.2, 0.5 and 0.3; `ooc` sets `WorldConfig.pool_size` to 8,
`max_true` to 5 and `arity_weights` to (0, 0, 0, 0.5, 0.5), so every requirement has exactly 4 or 5
of the 8 candidates. Pools of eight keep the full pool from being a near-perfect answer: the full
pool's F1 against a requirement of k of n candidates is 2k / (n + k), 0.80 to 0.91 for 4 or 5 of 6
but 0.67 to 0.77 for 4 or 5 of 8 (the first development check, with pools of six, showed the
degenerate case). With the default options and the same namespace and seed key, RL's streams and
policies reproduce X2's task by task and action by action (tested); RL's own runs use other
namespaces (`rl_`) and other seed keys (set, seed, policy, "rl"), so their worlds and actions are new.

Evidence policies, X2's: `maximal` (every candidate gathered, no inference; reporting),
`focused_sample` (the sampling walker of studies R, M and X2, class of at most three candidates)
and `random_omit` (each candidate omitted with probability 0.663 at every replan, ignoring the
evidence). Each policy is given the stream's failure probability, as every walker of the
repository is. In `ooc` worlds the walker's class cannot contain the truth: it walks to the node
with the fewest violations, acts on it, fails and replans, and its logs record those failed
attempts.

X2's five benchmark sets (Crafter, Craft, MSGI Mining, Little Alchemy 2, Freeciv) are read from X2's
logs at 16 episodes with `focused_sample` and `random_omit` evidence, without `gplearn` and without
new collection, and are reported descriptively only: no validity or secondary test reads them,
because the ids of converted decoys are not exchangeable with those of true candidates (X2's S2).

Alternative recipes logged without saying which recipe applied: not run. RecipeQuest has one craft
action per recipe and every log names the recipe of each attempt; relabelling attempts offline to
"the item appeared" would not give a function of the state (which recipe was attempted changes the
outcome), so no rule over the state is the truth. The variant needs an environment option (a craft
action per item that fires whichever recipe is eligible), which RL does not add.

### Learners (arms)

Every learner reads, per scored recipe (X2's: the hidden recipes attempted in the world's first k
logs), the same informative attempts (X2's `craft_examples`: the effect was absent before) whose
item inputs were all present: the pool candidates present before the attempt and whether the effect
appeared. The posteriors read X2's attempts directly (their likelihood ignores attempts without the
item inputs). The declared slip probability is the evidence's public failure probability (0 for X2's
logs and `ooc`).

| Learner | Code | Role |
|---|---|---|
| `posterior` | X2's `factorized_marginals`: uniform prior over subsets of at most 3 | reference |
| `posterior_eps` | the same updater with the declared slip probability | noise-matched |
| `posterior_full` | the same over every nonempty subset of the pool | class-wide control |
| `dnf` | every smallest monotone DNF consistent with all attempts (clingo 5.8.2) | primary arm |
| `dnf_rc2` | the same problem by weighted MaxSAT (python-sat's RC2) | cross-check (V1) |
| `dnf_soft` | every MAP DNF under slips (below); `dnf` without slips | noise arm |
| `ripper` | RIPPER, wittgenstein 0.3.5, defaults | classical |
| `cart` | CART, scikit-learn 1.9.1 `DecisionTreeClassifier`, defaults | classical |
| `figs` | FIGS, imodels 3.0.4 `FIGSClassifier`, defaults (12 rules) | classical |
| `gplearn` | Boolean symbolic regression, gplearn 0.4.3 `SymbolicClassifier` | exploratory |
| `random` | every candidate tied; its inferred set is the full pool | chance, full pool |

`posterior_full`'s uniform prior gives every candidate the prior marginal 32/63 with pools of six
and 128/255 with pools of eight, both above 0.5: without evidence its inferred set is the full pool,
and evidence must push a candidate below one half to drop it. It is a control for the class, not a
competitor tuned to it.

Rules. A rule is a monotone DNF over the recipe's pool: an OR of at most two nonempty AND terms of
candidates (positive literals only: a present fact can enable a craft, an absent one never does;
requirements are publicly nonempty). It covers an attempt when one of its terms is inside the
present candidates. `dnf`: every success covered, no failure covered, fewest literals. `dnf_soft`:
every success covered (an ineligible craft never succeeds), each covered failure explained as a slip;
cost = ln |pool| per literal (a description-length prior) + ln(1 / 0.1) per covered failure (the
slip likelihood), in integers scaled by 1,000 (1,792 and 2,303 with pools of six): the negative log
posterior up to a constant. Since ln 6 < ln 10, one literal costs less than one covered failure, so
a slip that a single decoy literal can separate from the successes is still fitted with that decoy;
the weights are declared, not tuned. Both solvers enumerate every optimal rule (clingo with
`--opt-mode=optN` and a symmetry break that gives each rule one answer set; RC2 by blocking each
optimum in every slot placement), so ties are reported, not broken. The enumeration cap of 5,000
cannot bind: with pools of six there are 1,414 rules without an absorbed term in all (63 single
terms, 1,351 pairs); with pools of eight (`ooc`, no slips) the true requirement is itself a
consistent rule, so an optimal rule has at most five literals, and there are 2,752 rules of at most
five literals (1,736 of five). A problem without a consistent rule (only possible with slips) gives
no rule. The number of recipes whose optimal rules include a disjunction of two terms is recorded
per cell.

Scores. A candidate's score under one rule is the share of the rule's terms that contain it (MSGI's
convention in X2); under several optimal rules, the mean over them. A learner without a rule for a
recipe has every candidate at 0 (tied), as X2's set learners.

Classical learners are fitted only when a recipe's attempts contain both outcomes (with successes
only they predict success everywhere: the empty term; with failures only, or none, there is no
rule) and are decoded into a rule in one way for all of them: the classifier's own decision on every
subset of the pool (64 or 256), the rule's terms being the minimal subsets predicted to succeed.
Every random choice is seeded per job (set, policy, seed, world, checkpoint, recipe, learner); FIGS,
which ignores its own seed, gets numpy's global generator seeded by the job for its fit and restored
afterwards (Implementation, decision 5). `gplearn` (population 200, 10 generations, tournament 20,
functions min, max, mul, add, sub, parsimony 0.01, initial depth 1-4) runs only in the primary cell
(`ooc`, `random_omit`, 16 episodes), because a fit takes seconds.

Pins (as installed and run; MIT unless stated): clingo 5.8.2, python-sat 1.9.dev15 (X2's pin),
wittgenstein 0.3.5, scikit-learn 1.9.1 (BSD-3), imodels 3.0.4, gplearn 0.4.3 (BSD-3), with numpy
2.5.3, pandas 3.1.0 and scipy 1.18.1: `pip install clingo==5.8.2 python-sat==1.9.dev15
wittgenstein==0.3.5 scikit-learn==1.9.1 imodels==3.0.4 gplearn==0.4.3`.

### Metrics

X2's, computed by X2's code (`inference_baselines.benchmark`: `inferred_set`, `set_metrics`,
`acting_metrics`; `evaluation.markov_study`: `auroc`, `gathering_cost`): for every world, evidence
source and checkpoint k in {4, 8, 12, 16} (16 only for X2's benchmark sets), AUROC of each scored
recipe's candidates (ties one half), the ranked gathering cost, the precision, recall and F1 of the
inferred set (candidates scoring at least 0.5, else the top-scoring tie group) and the open-loop
acting cost ratio and plan success of the node made of the inferred sets (deterministic execution,
also for `slip` worlds). A world's value is the mean over its scored recipes; acting is per task.
Also recorded: each learner's runtime, rule and fit statuses, the number of optimal rules and of
recipes with disjunctive optima, and the posterior's skipped contradictions.

### Primary contrast, falsification and verdict

Primary: `ooc`, `random_omit` evidence, 16 episodes, all 60 worlds: per-world F1 of `dnf` minus F1
of `posterior`. Decision (study M's four outcomes, as X2): "dnf better" if the lower end of the 95%
world-cluster bootstrap interval is above 0; "posterior better" if its upper end is below 0;
otherwise "equivalent" if the 90% interval lies inside (-0.05, 0.05); otherwise "inconclusive"; a
significant difference takes precedence. The claim is falsified by "equivalent" or "posterior
better". The minimum detectable difference (2.80 x SD / sqrt(60)) is reported from the observed SD.
`random_omit` ignores every learner's beliefs, so the logs are adapted to neither learner;
`focused_sample`, whose logs are adapted to the posterior's class, is the replication (O3). Acting
cost is not primary: it is open-loop and in-sample, and in X2 it reversed against F1.

Verdict: the decision, None when any of the 60 worlds is missing or comes from an incomplete run,
and "invalid" when a validity test fails (a validity test that does not apply, None, does not).

Margins: AUROC 0.03 (X2 and M1); F1 and acting cost ratio 0.05 (new). A world has about six to
nine scored recipes at 16 episodes; missing one candidate of a four-candidate requirement costs
that recipe 0.14 F1, so 0.05 is two to three such misses per world. 0.05 in acting cost ratio is 5%
of the optimal length.

### Validity tests

- V1: clingo and RC2 give the same optimal cost and the same set of optimal rules in every job, hard
  and soft, and no enumeration stops at the cap.
- V2: on `maximal` evidence of RL's sets and `gen_obs`, every checkpoint, every per-world AUROC of
  the posteriors, the exact learners, the classical learners and `random` is exactly 0.5 (nothing in
  those logs depends on the hidden requirement; `gplearn` does not run there).
- V3 (from the evidence files): every hidden requirement of `ooc` has 4 or 5 candidates and every
  other one of RL's sets and `gen_obs` 1 to 3; the `slip` logs contain failed attempts whose
  requirement and item inputs were all present, the deterministic sets none.
- V4: on `gen_obs`, RL's per-world AUROC, F1, acting cost ratio and plan success of `posterior` and
  `random`, every source and checkpoint, equal those in X2's summary
  (`artifacts/inference_baselines/x2_benchmark.json`) within 1e-9 (both NaN counts as equal). It
  applies when that summary covers the X2 runs RL read (the measurement); otherwise it is None.

### Secondary tests (each with its own criterion, no multiplicity correction)

All at 16 episodes on every world of the set; "above" and "below" mean the 95% interval of the
difference lies entirely above or below 0; a decision is the four-way rule with the metric's
margin. A test whose cell lacks a world or reads an incomplete run does not hold (None).

| Test | Cell | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| O1 | `ooc`, `random_omit` | `dnf` - `posterior` | acting cost ratio | below |
| O3 | `ooc`, `focused_sample` | `dnf` - `posterior` | F1 | decision "dnf better" |
| O4 | `ooc`, `random_omit` | `dnf` - `posterior_full` | F1 | decision "posterior_full better" |
| O5 | `ooc`, `random_omit` | `dnf` - `random` (the full pool) | F1 | above |
| E2 | `gen_obs`, `focused_sample` | `dnf` - `posterior` | F1 | decision "equivalent" |
| N2 | `slip`, `focused_sample` | `dnf_soft` - `dnf` | F1 | above |
| N3 | `slip`, `focused_sample` | `dnf_soft` - `posterior_eps` | F1 | decision "equivalent" |
| C1 | `gen_obs`, `focused_sample` | `dnf` - each of `ripper`, `cart`, `figs` | F1 | all three above |

Eight tests without correction: if none of their hypotheses were true and the tests were
independent, the chance that at least one holds anyway would be about 1 - 0.95^8, or 0.34 (each
interval test has a 5% error rate, each equivalence test about 5% at its margin), so a single
holding test is weak evidence on its own, and the verdict rests on the primary alone.

Descriptive contrasts (the same interval and four-way decision, no criterion):

| Name | Cell | Contrast | Metric |
|---|---|---|---|
| E1 | `gen_obs`, `focused_sample` | `dnf` - `posterior` | AUROC |
| E3 | `gen_obs`, `focused_sample` | `dnf` - `posterior` | acting cost ratio |
| E4 | `gen_obs`, `random_omit` | `dnf` - `posterior` | AUROC |
| O2 | `ooc`, `random_omit` | `dnf` - `posterior` | AUROC |
| N1 | `slip`, `focused_sample` | `dnf_soft` - `posterior` | F1 |
| N4 | `slip`, `focused_sample` | `posterior_eps` - `posterior` | AUROC |
| C2 | `ooc`, `random_omit` | `dnf` - each of `ripper`, `cart`, `figs` | F1 |

Also descriptive: every learner's table at every checkpoint (X2's benchmark sets at 16 episodes),
the contrasts of every learner with `posterior` and with `dnf`, `gplearn`'s values and runtime,
statuses and the numbers of optimal rules and of recipes with disjunctive optima.

History of the test list: the primary and the first fourteen tests were written before the first
development check, O4's criterion after a one-world scratch run, O5 after that check (Implementation,
decision 1). The review moved the primary to `random_omit` evidence and O1, O4 and O5 to its cell,
made O3 the `focused_sample` replication, kept eight tests and made the other seven descriptive (O2
and C2 moved to the primary cell); the out-of-class pools went from six to eight candidates.

### Predictions (written after the second development check)

Written after the development runs listed under Implementation (two worlds per cell, outcomes seen,
`rl-dev2` with pools of eight), after reading X2's measurement report (since committed in e4a9878)
and after the reviewer's trial values with requirements of 4 or 5 of 8 had been seen: `dnf` -
`posterior` F1 +0.058 with `focused_sample` and +0.031 with `random_omit` evidence, and `dnf` - full
pool F1 +0.054 and -0.085.

- Primary: "dnf better", estimate between +0.02 and +0.07 F1. Reason: out of class the posterior
  keeps at most three candidates (recall at most 0.75 or 0.6) and skips evidence that contradicts
  every hypothesis, while the smallest consistent rule grows to every candidate some failure forces
  in (development +0.038 [0.036, 0.040]).
- O3 holds (development +0.079 [0.056, 0.102]; the reviewer's trial +0.058).
- O1 holds: the posterior's sets make every plan fail out of class (plan success 0.00 in
  development), and the DNF's larger sets fail less often (development -0.135 [-0.264, -0.006]).
- O4 holds: a posterior over every subset of the pool is far better out of class (development F1
  0.808 against 0.569); the DNF's gain is a gain over a class that is too small, not over inference
  that knows the class, and its Occam bias is itself misspecified there.
- O5 does not hold: with random-omission evidence the full pool is ahead of the DNF on F1
  (development -0.170 [-0.225, -0.116]; the reviewer's trial -0.085).
- E2 does not hold: in class the posterior is ahead (development -0.024 [-0.033, -0.014]).
- N2 holds (development +0.114 [0.000, 0.228]: soft constraints repair most of the hard learner's
  loss under slips, where it can even have no consistent rule). N3 does not hold: the noise-matched
  posterior is ahead of the soft DNF (development -0.042 [-0.050, -0.033]).
- C1 holds (development: RIPPER +0.26, CART +0.08, FIGS +0.10 F1 below the DNF in class).
- V1 to V4 hold.
- Descriptive: E1 and E4 favour the posterior (development -0.037 and -0.060 AUROC), E3 is
  inconclusive or favours the posterior (+0.084 [-0.071, 0.239]), O2 is about 0 (+0.002), N1 favours
  the posterior (-0.042), N4 is small and positive (+0.009); in C2 the DNF is below RIPPER out of
  class (-0.18 F1), whose larger sets win on F1 there, and below or level with CART and FIGS;
  `gplearn` is below the DNF in AUROC in the primary cell (development 0.591 against 0.693); on X2's
  benchmark sets the posterior leads the DNF in AUROC by 0.005 to 0.02 and the two are close in F1.

### Statistics

The unit is the world; generated worlds are independent across seeds and sets. Every contrast is a
paired per-world difference; its mean gets a percentile world-cluster bootstrap interval (10,000
resamples, seed 20261008, X2's `_ci`), 95% unless stated, 90% for equivalence. One primary; the
secondary tests have their own criteria and no multiplicity correction. Runs: one per (set, policy,
seed), the earliest; a run that did not complete is replaced by the earliest later completed run
with the same config hash, commit and loaded code (`markov_study.select_runs`, as X2); replaced,
ignored and incomplete chosen runs are listed. The analysis refuses evidence whose chosen runs come
from more than one (commit, code) within RL's runs or within X2's.

### Budget and runtime

Worst cases are the sums of the task budgets of each run's stream (dry run, generator only); a run
starts only if the ledger's remaining allowance of its kind covers its stream sum (`AllowanceShort`
otherwise), so no run is cut short. `maximal` is charged as reporting, the other policies as
adaptive.

| Stage | Config | Runs | Adaptive worst case | Reporting worst case |
|---|---|---|---|---|
| Development | `rl_dev.yaml` (`rl-dev2`) | 6 | 2,764 | 1,382 |
| Measurement | `rl_main.yaml` | 30 | 78,210 | 39,105 |

Per-run sums: `slip` 3,715 / 3,783 / 3,883 / 3,723 / 3,630 and `ooc` 4,034 / 4,256 / 4,105 /
3,996 / 3,980 (seeds 0-4), per policy; the largest run is `ooc` seed 1 (4,256). Declaration of
`runs/RL-ledger.json`, identical in both configs: allocation `rl` 82,500 (78,210 + 4,256 = 82,466,
rounded up to the next hundred: one rerun of the largest run fits), `dev` 10,000, adaptive cap
92,500 (their sum), reporting cap 48,400 (39,105 + 4,256 + 5,000 for development = 48,361, rounded
up). The ledger file was created by the first development run with the first declaration (`rl`
78,600, adaptive cap 88,600, reporting cap 46,300, for out-of-class pools of six); the new
declaration is applied as a logged amendment at the freeze. `collect` refuses to run while the
ledger file's declaration differs from the config's. Development used 3,814 adaptive (`rl-dev`
1,876, `rl-dev2` 1,938) and 1,693 reporting (827 and 866) interactions.

Runtime from the development rates (a laptop with 8 logical cores, shared with other work):
collection about 35 to 40 minutes in one session, almost all of it the five `ooc` runs of
`focused_sample` (2.0 to 2.3 s per task, about 7 minutes per run); every other run takes seconds.
Analysis 0.4 to 1 s per world and checkpoint for the posteriors, the exact and the classical
learners (2,160 world-checkpoints and 180 benchmark world-cells, 15 to 40 minutes in one process)
plus 8 to 26 s per world for `gplearn` (60 worlds, 8 to 26 minutes). The measurement takes about
1 to 2 hours in one process and runs at night.

### Stopping and rerun rules

No interim analysis. A run stops only at the end of its stream. A run, or a rerun of a failed run,
starts only if the ledger covers its stream sum and the ledger file's declaration equals the
config's; otherwise a logged ledger amendment comes first. Collection runs as one session at night.
After an interruption `collect --resume` skips every (set, policy, seed) that already has a
completed run with the same config content, commit and code; every collection session runs from
the one snapshot of the freeze commit (the code hash depends on the checkout's line endings, as
X2's results record, so a second checkout of the same commit can count as other code). Should a
session ever be split, the sessions use disjoint `--seeds` (the ledger file is locked, so they may
run at once), a `--resume` without filters runs only after every session has exited, and a stale
`runs/RL-ledger.lock` older than a minute may be removed (a live holder keeps it for milliseconds).
The analysis (`rank`) runs once after all 30 collection runs; after an interruption `rank --resume`
reuses the cached rows of the same code, options and library versions (`runs/rl_cache`, never X2's
cache). A run that fails without a code change is handled by the rerun rule and recorded as an
amendment; a fault that needs a code fix means a new freeze and a fresh collection of every RL run
from the new snapshot. X2's logs are never written.

### Commands (from a clean snapshot, after review and freeze)

From the repository root, so that `runs/` (with X2's measurement runs) and `artifacts/` (with X2's
summary, read by V4) are the repository's; `S` is a clean worktree of the freeze commit, whose code
both commands run. The ledger amendment to the declaration above comes first.

```bash
git worktree add ../snapshots/RL-<hash> <hash>
S=../snapshots/RL-<hash>
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.rules collect \
    --config $S/configs/rule_learners/rl_main.yaml --allow-measurement --resume
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.rules rank --prefix rl-main- \
    --x2-prefix x2-main- --out artifacts/rule_learners --cache runs/rl_cache --resume \
    --x2-artifact artifacts/inference_baselines/x2_benchmark.json
```

`--resume` is harmless on a first run (nothing to reuse) and makes a rerun after an interruption
continue where it stopped.

### Limits stated in advance

- One family (F1), items observable. Out of class means requirements of 4 or 5 of 8 candidates; the
  out-of-class set therefore differs from the in-class sets in pool size as well, so contrasts are
  read within a set, not across sets. Mixed sizes (RB-E's question), larger requirements and
  disjunctive recipes are not tested.
- The DNF learner's class (at most two terms, any size) is wider than the truth needs (every
  recipe's rule is one conjunction), so its disjunctions can only cost; its Occam bias favours small
  requirements, which is wrong out of class. The soft learner's prior and the slip probability are
  declared, not tuned; the slip probability is public, as for the walkers.
- The two learners treat contradictions differently. X2's updater leaves the posterior unchanged by
  an attempt that contradicts every hypothesis in its class (it skips it, so the result depends on
  the order of the evidence, and out of class whole stretches of evidence are skipped). The hard DNF
  learner has no rule when the attempts admit none (all candidates tied, only with slips), and the
  soft learner prices each covered failure instead.
- `posterior_full` starts every candidate above one half (32/63, 128/255), so it favours large sets.
- The evidence policies are X2's; `focused_sample`'s logs are adapted to the posterior's class of at
  most three, also where that class cannot hold the truth; `random_omit` is the primary's evidence.
- Classical learners run with library defaults, are not tuned and are decoded through their minimal
  predicted successes; with few attempts a classifier generalizes arbitrarily outside them. `gplearn`
  is exploratory, stochastic and runs in the primary cell only.
- Acting cost is open-loop and deterministic: slips do not enter the acting simulation.
- X2's benchmark sets are descriptive; X2's results on `gen_obs` were public before this protocol;
  RL's learners have not read X2's measurement logs. The alternative-recipes variant is not run.

## Implementation

- `src/hypergraph_agent/rules/`: `dnf.py` (problems, the clingo and RC2 solvers, enumeration of every
  optimal rule, scores, agreement), `classical.py` (RIPPER, CART, FIGS and gplearn fitted, seeded per
  job and decoded into rules), `learners.py` (every learner on one world at one checkpoint, from X2's
  attempts, with the disjunctive-optima diagnostic), `collection.py` (the study's own config schema on
  X2's collection machinery: X2's generated sets with the variant options, X2's policies given the
  failure probability, resume, the allowance check, the ledger declaration check, RL's evidence
  loader and X2's loader restricted to `gen_obs` and its benchmark sets), `analysis.py` (X2's
  metrics, X2's row cache under RL's own key and folder, checkpoints per set, parallel worlds,
  decisions, validity tests V1 to V4, secondary tests and descriptive contrasts, report),
  `__main__.py` (commands `collect` and `rank`).
- `configs/rule_learners/`: `rl_dev.yaml` (run name `rl-dev2`), `rl_main.yaml` (own schema; the
  directory is listed in `tests/test_hygiene.py`).
- No X2 module, config or shared module is changed; X2's code hash (which covers the whole package)
  changes in this working tree, not in X2's frozen snapshot.
- `tests/test_rule_learners.py` (27 tests): a planted conjunction recovered by both solvers, every
  optimal rule reported and at least one term required, unsatisfiable problems, one answer set per
  rule, RC2 equal to clingo on 40 random hard and soft problems, the soft learner explaining slips
  and reducing to the hard program without them, the classical learners recovering a conjunction
  (seeded), FIGS's rule independent of numpy's global generator (whose state the fit restores),
  their decoding and one-class rules, the posterior variants (equal to X2's at slip probability 0)
  and the shared attempts, `posterior_full` equal to a brute-force posterior over every subset (with
  and without slips) and its prior marginal of 32/63, RL's default sets equal to X2's streams and
  its policies acting as X2's action by action, the variant sets' requirement sizes, pools and slips,
  config validation, one ledger declaration, the measurement plan within its declaration, the
  refusal of a ledger declared otherwise, collect resume, seed filter and the allowance refusal, the
  decision rule, world metrics through X2's scoring with `gplearn` keyed by (set, source,
  checkpoint), the checkpoints of X2's benchmark sets, V4 against X2's summary, parallel and
  sequential worlds identical, the pipeline end to end on X2's and RL's scratch logs with X2's
  Crafter set (descriptive cells only, no recipe names in the report, the refusal of X2's cache
  folder), the cache and the code stamp, run selection with a crashed run replaced and mixed code
  refused, and the line hygiene of RL's files.

Development commands (from the repository root; V4 needs X2's summary of the same runs, so X2's own
`rank` writes one for X2's development logs into a scratch folder with a scratch cache):

```bash
python -m hypergraph_agent.rules collect --config configs/rule_learners/rl_dev.yaml --dry-run
python -m hypergraph_agent.rules collect --config configs/rule_learners/rl_dev.yaml
python -m hypergraph_agent.inference_baselines rank --prefix x2-dev- --out ../scratch/x2_dev \
    --cache ../scratch/x2_cache --expected-selection-worlds 2
python -m hypergraph_agent.rules rank --prefix rl-dev2- --x2-prefix x2-dev- --out ../scratch/rl_dev2 \
    --cache runs/rl_cache --expected-worlds 2 --x2-artifact ../scratch/x2_dev/x2_benchmark.json
```

### Development record (calibration, not results)

First check (`rl-dev`, 2026-10-29; out-of-class pools of six): `rl_dev.yaml`, seed 0, two worlds per
variant, 16 episodes, every policy (namespaces `rl_slip_dev`, `rl_ooc_dev`): 6 runs, all completed,
1,876 adaptive and 827 reporting interactions; the `ooc` run of `focused_sample` took 72.6 s for 32
tasks. Analysed with X2's development logs of `gen_obs` (229 s in one process on a busy machine).
Out of class the full pool, which needs no inference, had a higher F1 (0.875) and a lower acting
cost (1.13) than both the DNF learner (0.812, 1.24) and the posterior (0.750, 1.34): with 4 or 5 of
6 candidates required, the full pool is almost the answer (decision 1, and the review's pools of
eight). Before it, two identical scratch runs (one world per variant, six episodes, in-memory
ledger, scratch folders; not charged) checked the pipeline; unit tests use in-memory ledgers and
scratch folders. The working tree then held other builders' uncommitted changes to shared modules
(`agents/walker.py`, `envs/generator.py`, `topology/inference.py`, `config.py` and TechTree modules,
recorded in each run's `dirty.patch`), documented as opt-in options whose defaults leave every draw
and decision unchanged; a replay of X2's development collection of `gen_obs` (every policy, in
memory) with that working tree reproduced X2's recorded logs, recipes and hidden requirements in all
six worlds. The measurement runs from a clean snapshot.

Second check (`rl-dev2`, 2026-11-04, after the review; out-of-class pools of eight, namespaces
`rl_slip_dev` and `rl_ooc8_dev`): 6 runs, all completed, 1,938 adaptive and 866 reporting
interactions, 74 s in all; the `slip` runs reproduced the first check's interactions exactly (317,
466 and 425), and the `ooc` run of `focused_sample` took 65.4 s for 32 tasks. X2's own `rank` on
X2's development logs (92 s, scratch folder and cache) gave the summary for V4. RL's analysis of
X2's development logs (`gen_obs` and, descriptively, Crafter, MSGI Mining and Little Alchemy 2)
and the six runs took 60 s in one process, `gplearn` 16 s of it. V1 held in all 507 hard and 159
soft jobs (no cap), V2 in all 240 per-world values, V3 with 15 slips in the `slip` logs, and V4 in
all 192 compared values (largest difference 0).

Values at 16 episodes, two worlds per cell (`random`'s inferred set is the full pool; F1 and acting
cost ratio):

| Learner | `ooc` `random_omit` F1 | acting | `ooc` `focused_sample` F1 | acting | `gen_obs` F1 | `slip` F1 |
|---|---|---|---|---|---|---|
| `posterior` | 0.531 | 1.61 | 0.547 | 1.62 | 0.861 | 0.790 |
| `posterior_eps` | 0.531 | 1.61 | 0.547 | 1.62 | 0.861 | 0.790 |
| `posterior_full` | 0.808 | 1.14 | 0.857 | 1.06 | 0.771 | 0.723 |
| `dnf` | 0.569 | 1.48 | 0.626 | 1.23 | 0.837 | 0.635 |
| `dnf_soft` | 0.569 | 1.48 | 0.626 | 1.23 | 0.837 | 0.749 |
| `ripper` | 0.747 | 1.48 | 0.644 | 1.44 | 0.577 | 0.598 |
| `cart` | 0.618 | 1.48 | 0.647 | 1.24 | 0.760 | 0.787 |
| `figs` | 0.650 | 1.44 | 0.647 | 1.24 | 0.737 | 0.639 |
| `gplearn` | 0.570 | 1.54 | - | - | - | - |
| `random` | 0.740 | 1.42 | 0.735 | 1.42 | 0.548 | 0.451 |

AUROC in the primary cell: `posterior` 0.691, `posterior_full` 0.731, `dnf` 0.693, `cart` 0.691,
`figs` 0.684, `ripper` 0.609, `gplearn` 0.591; in class (`gen_obs`, `focused_sample`) `posterior`
0.930 and `dnf` 0.894; under slips `posterior` 0.856, `posterior_eps` 0.865, `dnf` 0.725 and
`dnf_soft` 0.825. Plan success in the primary cell: `posterior` 0.00, `dnf` 0.09, `posterior_full`
1.00.

Tests on these worlds (intervals over two worlds are not informative): primary +0.038 [0.036,
0.040], "dnf better"; O1 -0.135 [-0.264, -0.006]; O3 +0.079 [0.056, 0.102]; O4 -0.238 [-0.295,
-0.182]; O5 -0.170 [-0.225, -0.116]; E2 -0.024 [-0.033, -0.014]; N2 +0.114 [0.000, 0.228]; N3 -0.042
[-0.050, -0.033]; C1 holds (RIPPER +0.261, CART +0.077, FIGS +0.101). Descriptive: E1 -0.037, E3
+0.084, E4 -0.060, O2 +0.002, N1 -0.042, N4 +0.009, C2 RIPPER -0.177, CART -0.049, FIGS -0.080.
No recipe had a disjunctive optimum in any cell; the most optimal rules of one recipe were 8
(`ooc`). X2's updater skipped 15 contradicting attempts in the primary cell and 27 with
`focused_sample` evidence; under slips the hard DNF had no consistent rule for 3 of 17 recipes. On
X2's benchmark sets (one or two worlds each) the posterior led the DNF in AUROC by 0.004 to 0.021,
and F1 was within 0.08 either way. Runtime per cell of two worlds at 16 episodes: posteriors 0.01 to
0.06 s, clingo 0.12 to 0.19 s, RC2 0.01 to 0.03 s, RIPPER, CART and FIGS at most 0.26 s.

Decisions that came from development and the review:

1. O5 added after the first check: out of class the full pool, which needs no inference, had a
   higher F1 and a lower acting cost than both the posterior and the DNF learner, so the primary
   alone could overstate what the DNF learner gains.
2. `collect --seeds` added: the walker is slow in `ooc` worlds, where no node of its class is
   consistent with the logs.
3. `gplearn` restricted with 200 programs and 10 generations: fits took 1.4 to 6 s and missed three-
   and four-candidate conjunctions in scratch timing runs, as the design note reported; after the
   review it runs in the primary cell only.
4. wittgenstein reports its warnings through `warnings.showwarning` directly; the learner silences
   them inside the fit only. A classifier is fitted only when both outcomes occur, so library
   behaviour on one class never decides a score.
5. FIGS (imodels 3.0.4) keeps its `random_state` but never passes it on: its split stumps are
   scikit-learn trees without a seed, which break ties between equally good splits (frequent with
   binary features and few attempts) with numpy's global generator, seeded from the operating
   system in every process. The test of parallel against sequential worlds failed on 2026-10-29
   with different FIGS values while other builders were editing the package, which at first hid
   the cause; on 2026-11-04, with no edit in progress, it failed in 3 of 3 runs, and with the job's
   seed fixed FIGS returned one rule under six global states and another under the other six of
   twelve. The learner now seeds that generator with the job's seed for the fit and restores it
   afterwards: FIGS then returned one rule under all twelve, the parallel test passed in 3 of 3
   runs, and a new test checks both. Run again after the fix, the first check's analysis reproduced
   all 72 per-world entries of every other learner exactly; FIGS changed in 13.
6. The review (2026-11-04): out-of-class pools of eight (the full pool's F1 2k / (n + k) made pools
   of six degenerate); the primary on `random_omit` evidence; eight tests and seven descriptive
   contrasts; `gplearn` in the primary cell; X2's benchmark sets as descriptive data; V4; the ledger
   declaration check in `collect`; the new declaration; python-sat pinned at 1.9.dev15; collection
   in one session at night.

## Amendments

None.

## Results

None yet: no measurement run exists.

## Verdict and limits

None yet.
