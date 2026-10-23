# Study X2: inference of hidden prerequisites against open-source structure-inference baselines

Status: protocol frozen on 2026-10-23 by the commit that freezes this file, before any
measurement run, after two independent reviews (their blockers, other findings and minor points
fixed); the measurement runs from a clean worktree of that commit. The ledger caps were raised
to the declaration below by a logged amendment before the freeze. The development
checks listed under Implementation used only the allocation `dev` of
`runs/X2-ledger.json` and namespaces ending in `_dev`; they are calibration,
not results. The measurement namespaces (`x2_gen`, `x2_bench_<graph>`) have
never been used.

## Question and claim

Question: given identical episode logs, how well does the repository's
inference of hidden prerequisites (the posterior over complete dependency
hypergraphs) recover the hidden requirements, compared with open-source
structure-inference methods that address the same problem from other
traditions: subtask-graph inference in meta reinforcement learning (MSGI) and
action-model learning from traces (macq's Observer and ARMS, and a SAM-style
safe learner)?

Claim under test (one pre-registered primary): on generated worlds with
items observable, after 16 episodes of the sampling walker, the posterior
ranks a recipe's candidates better than the best open-source baseline
(AUROC), with four pre-stated outcomes (posterior better, baseline better,
equivalent within 0.03 AUROC, inconclusive).

The study is offline (as study M's M1): traces are collected once with fixed
policies, then every method reads the same logs and the same public recipe
descriptions, and no method chooses the actions. The primary's evidence
comes from `focused_sample`, which plays nodes drawn approximately from the
posterior, so those logs are adapted to the posterior's hypothesis class:
the walker omits candidates the posterior finds unlikely, which produces the
informative failures that the posterior uses. The same contrast on
`random_omit` evidence (S1), a policy that ignores every method's beliefs,
is the method-independent check; `maximal` evidence (S2) is the
no-information control.

## Relation to earlier work

docs/PRIOR_ART.md records the methods: MSGI (R4; Sohn, Woo, Choi and Lee,
ICLR 2020) infers each subtask's precondition from (completion vector,
eligibility) pairs with a CART decoded into a sum of products; SAM learning
(A2; Juba, Le and Stern, KR 2021) learns safe conjunctive preconditions by
intersecting the states before successful executions; OHCAM (A3) keeps a
Bayesian posterior over action models. ARMS (Yang, Wu and Jiang, Artificial
Intelligence 171, 2007) learns STRIPS models from plan traces through a
weighted MAX-SAT over action, information and plan constraints; the macq
library (MIT; github.com/AI-Planning/macq) implements ARMS, SLAF, LOCM, AMDN
and a pre-state intersection learner ("Observer"). Study M compared the
posterior with pairwise (Markov) weights on goal-only logs (M1: equivalent
within 0.03 AUROC at 16 episodes). docs/PRIOR_ART.md records no comparison
of these traditions on matched logs of one hidden-prerequisite task; that
comparison is what this study adds.

What the methods can use differs by construction, and the study reports it
rather than removing it: the set learners read only successful executions
(a STRIPS trace has no failed action), MSGI reads labelled attempts without
a prior over requirement sizes, and the posterior and the pairwise weights
read successes and failures with the declared hypothesis class (nonempty
subsets of the six-candidate pool with at most three members, uniform prior).

## Protocol (frozen)

### Traces

Collection: `python -m hypergraph_agent.inference_baselines collect --config
configs/inference_baselines/x2_main.yaml --allow-measurement`. Every (set,
seed, policy) is one run; each run plays its set's tasks of that seed in
world-major order (16 episodes per world; a policy's beliefs carry over
within a world and start from the prior in the next) and writes, per world,
the episode logs, the public recipe descriptions, the recipes each episode's
task registered and, evaluator only, the hidden requirements
(`<run>/evidence/w<index>.json`, the record of study M plus the set and the
observation mode). A run's config hash covers its content (set, policy,
seed, walker settings), not the file it came from.

Policies (fixed; deterministic dynamics):

- `maximal`: the brute-force node, every candidate of every pool gathered, no
  inference (charged to the reporting cap, as in studies R and M);
- `focused_sample`: the sampling walker of studies R and M (focused walk,
  then 50 moves among the nodes consistent with every log); adaptive;
- `random_omit`: every candidate omitted independently with probability
  0.663 at every replan, ignoring the evidence (study R's control policy,
  with the value fixed there for F1); adaptive.

Sets (45 runs; worlds per seed in brackets):

| Set | Worlds | Observation | Seeds |
|---|---|---|---|
| `gen_obs` | family F1, namespace `x2_gen` (12) | items observable | 0-4 |
| `gen_goal` | the same worlds and tasks as `gen_obs` (12) | goal-only | 0-4 |
| `bench_crafter` | whole Crafter graph, conversion seeds 0-9 (10) | items observable | 0 |
| `bench_psketch_craft` | whole Craft world, conversion seeds 0-9 (10) | items observable | 0 |
| `bench_msgi_mining` | MSGI Mining, closures of 20 sampled goals (20) | items observable | 0 |
| `bench_little_alchemy_2` | Little Alchemy 2, closures of 30 sampled goals (30) | items observable | 0 |
| `bench_freeciv_classic` | Freeciv classic, closures of 20 sampled goals (20) | items observable | 0 |

Family F1 is study R's Stage D distribution (hidden prerequisites, goals of
depth 1-3, no distractor base facts, pools of six, at most three hidden base
facts); every seed plays its own worlds (base seed 10,000 + 1,000 x seed).
The two whole-graph sets each have one hidden structure (conversion seeds
change vocabulary types and pool decoys only).

Benchmark worlds are converted by `benchmarks.to_recipequest` with the
settings of study X (pools of six, at most three hidden base facts, at most
two alternatives per item); every episode asks for an item drawn uniformly
among the world's items. Little Alchemy 2 and Freeciv are read only from the
locally fetched data (`benchmarks/little_alchemy/fetch.py`,
`benchmarks/freeciv/fetch.py`); neither their data nor their element or
recipe names are committed, and the analysis reports opaque world labels
(`<set>:s<seed>:w<index>`) only (tested for both). Development used other
namespaces and, for Crafter and Little Alchemy 2, worlds that are not
measurement worlds (conversion seed 100, goal-sample seed 99). The two MSGI
Mining development worlds (goal-sample seed 99, conversion seed 0) are the
same converted worlds, with the same world keys, as the measurement worlds
`bench_msgi_mining:s0:w5` and `bench_msgi_mining:s0:w10` (MSGI Mining has 23
convertible goals); their development tasks and episodes differ (another
namespace), and their development outcomes were seen. S8 for MSGI Mining is
therefore also reported without these two worlds (pre-registered below).

### Methods

| Method | Code | Input | Runs on |
|---|---|---|---|
| `posterior` | this repository (joint posterior, marginals) | every logged episode | both modes |
| `pair_weight` | this repository (study M's online pairwise weights) | every logged episode | both modes |
| `msgi` | MSGI authors' code, srsohn/msgi `e665861`, MIT | informative attempts | items observable |
| `observer` | macq 0.3.11 (MIT), Observer, in process | STRIPS traces | items observable |
| `arms` | macq 0.3.11 (MIT), ARMS, worker process | STRIPS traces | items observable |
| `sam` | SAM-style safe learner, reimplemented here | successful steps | items observable |
| `random` | every candidate tied | none | both modes |

Third-party code as installed and pinned for the measurement: macq 0.3.11
(MIT) with nnf 0.4.1 (ISC) and python-sat 1.9.dev15 (MIT, a pre-release),
`pip install macq==0.3.11 nnf==0.4.1 python-sat==1.9.dev15`; torch
2.14.1+cpu for the vendored MSGI code (`third_party/msgi`, unchanged;
`SOURCE.md` records the commit, license and SHA-256 of each file).

`posterior`. With items observable every attempt's outcome is observed and
depends only on the attempted recipe's hypothesis and the observed state, so
the posterior factorizes by recipe; it is then computed exactly by the
per-recipe updater (`learners.factorized_marginals`, which applies
`topology.inference`), equal to study M's joint enumeration where that is
exact (tested). Development showed why: on items-observable logs the joint
enumeration joins most recipes of a world into one group and ran out of its
work budget (Gibbs fallback, minutes per world). With goal-only logs it is
study M's enumeration (`evaluation.markov_study.posterior_marginals`, exact
per connected group of recipes within 2,000,000 log checks, Gibbs beyond).

`msgi` (`inference_baselines/msgi_adapter.py`; `third_party/msgi/SOURCE.md`).
MSGI's full pipeline (`ILP.infer_graph`) needs the eligibility of every
subtask at every step, which MSGI's environment reveals and RecipeQuest does
not: here eligibility is observed only through an attempted craft. The
adapter runs the authors' `cart_train`, `decode_cart` and `simplify_Kmap` on
each recipe's informative attempts (the effect was absent before; label: it
appeared), with the recipe's pool candidates and item inputs as features, and
MSGI's two rules for subtasks that need no tree (eligible at every
observation: no precondition; never eligible: unknown). A candidate's score
is the share of the decoded AND terms in which it is a positive literal.
Faithful to the authors' code, `decode_cart` adds an AND term only at a node
whose split separates its examples perfectly; a pure positive leaf directly
under an impure split is dropped, and among equally good splits `cart_train`
keeps the lowest-index feature (features in type-id order).

`observer` and `arms` (`inference_baselines/macq_adapter.py`). A log
converts into a STRIPS trace when items are observable: every step that
changed the state is an executed action (`gather_t<type>`,
`activate_t<type>`, `craft_<k>`), every fact type is a fluent; failed
attempts, repeats, waits and submissions are dropped, since a STRIPS trace
cannot represent an action whose preconditions did not hold.

ARMS settings: `upper_bound` 3 (in macq, the number of relations at which an
action's precondition, add or delete list counts as fully learned, which
ends that action's MAX-SAT steps), the library defaults otherwise
(`min_support` 2, weights 110 and 100, threshold 0.6, defaults 30), every
fluent observed. ARMS runs in a worker process with `PYTHONHASHSEED=0` and
three documented run-time changes to the libraries, all in `arms_worker.py`:

1. Weights follow their clauses. The released macq 0.3.11 permutes the
   MAX-SAT weights: `ARMS.step3` builds the problem as
   `And(list(constraints_w_weights))`, whose clauses nnf keeps in a
   frozenset, and passes `list(constraints_w_weights.values())` in insertion
   order, so `to_wcnf` pairs the i-th clause of the frozenset with the i-th
   weight (macq/extract/arms.py lines 743-746, macq/utils/pysat.py lines
   41-45 and 65-68). X2 runs every soft clause with the weight `step3`
   assigns to it (`arms_worker.aligned_weights`; tested clause by clause);
   no as-released ARMS arm is run. In the review's scratch check, one F1 job
   had 1,254 of its 2,076 soft clauses carrying another clause's weight. In
   the development check below the released pairing would have moved the
   weight of 86,293 of the 138,957 soft clauses (62%). How much the fix
   changes ARMS depends on the tie-break (next paragraph).
2. Empty constraint groups. `ARMS._calculate_support_rates` computes
   `max([])` when the I3 or the plan constraint group is empty (no relation
   or no action pair reaches the minimum support), which raises; the guard
   returns no weights for that group, which is what the algorithm's
   definition gives (no constraints of that group).
3. Reproducibility. nnf names the auxiliary variables of its CNF conversion
   with `os.urandom` and hashes an AND or OR node with its class object,
   whose hash is its memory address; the clause order, the variable
   numbering and therefore the optimum the solver returns among equally
   weighted ones changed from process to process (development: three
   processes, three models for one job). The worker draws the names from a
   generator seeded per job and hashes nodes by class name (the problem is
   unchanged up to variable names and order); repeated jobs agree across
   processes (tested).

ARMS depends on the string hash seed, which orders Python's sets of fluents
and constraints and so decides ties among equally weighted MAX-SAT optima.
X2 reports ARMS for one arbitrary tie-break, hash seed 0. The review
reproduced the development numbers at seed 0 (mean AUROC on the two
development `gen_obs` worlds, 16 episodes, as released and with the weights
paired): 0.630 and 0.554 (`focused_sample`), 0.553 and 0.522
(`random_omit`), 0.549 and 0.562 (`maximal`). With the weights paired,
hash seeds 0, 1, 2 and 3 gave 0.554, 0.631, 0.573 and 0.751
(`focused_sample`) and 0.522, 0.546, 0.512 and 0.629 (`random_omit`). The
fix therefore moves ARMS by -0.08 to +0.32 depending on the seed, and at
seed 0 in opposite directions on the two worlds (0.476 to 0.548 and 0.783
to 0.561). Neither the measurement nor the fix is wrong: ARMS's result is
one draw among its tie-breaks. An exploratory check, pre-registered here,
reruns ARMS on the 24 selection worlds at 16 episodes under hash seeds 1 to
4 (`rank --arms-hash-seeds 1 2 3 4`, written to
`<stem>_arms_hash_seeds.json`, not used by any decision) and reports, per
seed and evidence source, ARMS's mean AUROC, the baseline that would then be
selected and whether "ARMS below Observer" would still hold.

ARMS repeats its MAX-SAT step until every action counts as fully learned; an
action that never does keeps the loop going (development: after its first
step one Little Alchemy 2 job solved the same 54-clause problem again and
again). A job therefore stops after 50 MAX-SAT steps, a deterministic limit,
with status `iteration_cap` and the model as learned so far (each action's
precondition after its last step); that Little Alchemy 2 job gives the same
model at limits of 2, 3, 10, 50 and 200 steps. A wall-clock limit of 120 s
per job, which then has no model, remains as a safety net. Stopped,
timed-out and failed jobs are counted per cell and in a status line of the
report; `rank` prints a warning when a job timed out or its worker exited,
and such results are cached and reused by `--resume` like any other.

`sam`. The precondition of a craft is the intersection of the states before
its successful executions; a recipe never executed successfully has no
model. On these traces it is the same rule as macq's Observer, and the two
are tested equal; `sam` is kept as an independent check of the macq
pipeline, not as a separate baseline.

Not run, and why:

- Set learners and MSGI on goal-only logs: the outcome of crafting an
  intermediate item is not observed, so neither STRIPS traces nor labelled
  attempts exist (`traces.ItemsUnobserved`). On `gen_goal` only `posterior`,
  `pair_weight` and `random` run.
- macq SLAF needs one trace (episodes joined by a reset action); on a 25-step
  check trace with every fluent observed it learned no precondition for the
  craft action (2.2 s; 10.4 s with the crafted item hidden). Not used.
- macq LOCM learns state machines of object sorts from action parameters;
  RecipeQuest actions are grounded and have no objects. macq AMDN is built
  for noisy, disordered parallel observations and needs a further package.
  Not used.
- OHCAM (A3): no implementation was located for this study; a
  reimplementation would be the repository's own posterior with another
  prior, not an independent baseline.
- MSGI's reward inference, exploration bonus and GRProp policy: not
  inference of preconditions.

### Metrics

For every world record, evidence source (policy) and checkpoint k in {4, 8,
12, 16}, every method reads the world's first k logs and the recipes
registered by then; the scored recipes are the hidden recipes attempted in
those logs (as in M1). Against the hidden requirement T of each scored
recipe:

- AUROC: the probability that a required candidate outscores a non-required
  one, ties one half (a set learner's scores are 1 inside its set, 0 outside;
  a recipe without a model has every candidate tied).
- Ranked gathering cost (study M; exploratory).
- Precision, recall and F1 of the inferred set S: for every method the
  candidates scoring at least 0.5, or, when none does, the top-scoring tie
  group (requirements are publicly nonempty); precision is 0 for an empty S.
  For marginal probabilities this is the decision with the fewest expected
  misclassified candidates; for a set learner it is its set, or the full
  pool without a model (SAM's conservative default).
- Acting cost of the inferred model, without interaction: on each of the
  world's recorded tasks (all its episodes, the same for every method and
  checkpoint), from the task's initial facts, the derivation planner's
  cheapest plan on the inferred node (S for the scored recipes, the full pool
  for the others) is executed against the hidden requirements; if the goal is
  not held at its end, the brute-force plan from the state reached completes
  it; plus one submission. World value: total length over total optimal
  length (the same planner on the hidden requirements, equal to the reference
  solver; tested), and the plan success rate (share of tasks whose own plan
  reached the goal). The plan is executed to its end before the fallback, so
  this charges the full cost of acting open-loop on a wrong model.

A world's value of a per-recipe metric is the mean over its scored recipes.

### Best open-source baseline

The open-source baselines are `msgi`, `observer` and `arms` (code of their
authors or of the macq library). For each evidence source, the best is the
one with the largest mean per-world AUROC on the 24 selection worlds (seeds
0-1) of `gen_obs` at 16 episodes (ties within 1e-6 in the order msgi,
observer, arms). The selection needs all 24 selection worlds from completed
runs; otherwise no baseline is selected for that source, and the primary
verdict and the tests that use the selected baseline are None. The number
of selection worlds is recorded. The test worlds (seeds 2-4) are used only
for the contrasts. Benchmark sets have no selection split; their contrasts
use the baseline selected on `gen_obs` for the same evidence source.

### Primary contrast

On the 36 test worlds (seeds 2-4) of `gen_obs`, `focused_sample` evidence,
16 episodes: per-world AUROC of `posterior` minus AUROC of the selected
baseline. Decision (as M1, margin 0.03 AUROC): "posterior better" if the
lower end of the 95% world-cluster bootstrap interval is above 0; "baseline
better" if its upper end is below 0; otherwise "equivalent" if the 90%
interval lies inside (-0.03, 0.03); otherwise "inconclusive". A significant
difference takes precedence over equivalence. The verdict needs a selected
baseline and all 36 test worlds from completed runs; otherwise it is None.
The minimum detectable difference (two-sided 5%, power 80%) is 2.80 x SD /
sqrt(36) of the per-world differences, reported from the observed SD.

### Secondary tests (each with its own criterion, no multiplicity correction)

All are computed by `benchmark.secondary_tests` and reported under
`secondary` in the artifact and in its markdown report. "Primary cell":
`gen_obs`, test worlds, `focused_sample` evidence, 16 episodes.

| Test | Cell | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| S1 | `gen_obs` test, `random_omit`, 16 | posterior - its baseline | AUROC | decision "posterior better" |
| S2 | generated sets, `maximal`, all | each method at chance | AUROC | see below |
| S3 | every cell | `observer` - `sam` | AUROC, F1, acting | largest difference 0 |
| S4 | primary cell | posterior - selected baseline | F1 | 95% interval above 0 |
| S5 | primary cell | posterior - selected baseline | acting cost ratio | 95% interval below 0 |
| S6 | primary cell | posterior - `pair_weight` | AUROC | decision "equivalent" |
| S7 | `gen_goal` test, `focused_sample`, 16 | posterior - `pair_weight` | AUROC | "equivalent" |
| S8 (5) | each benchmark set, `focused_sample`, 16 | posterior - its baseline | AUROC | interval above 0 |
| S8 (6th) | MSGI Mining without w5 and w10 | posterior - its baseline | AUROC | interval above 0 |

S2 holds when every per-world value of `posterior`, `pair_weight`, `msgi`,
`observer`, `sam` and `random` on `maximal` evidence (`gen_obs` and
`gen_goal`, every split and checkpoint) is exactly 0.5 and the 95% interval
of `arms` minus 0.5 (`gen_obs` test worlds, 16 episodes) contains 0. In S8
"its baseline" is the one selected for `focused_sample` evidence. "Above"
and "below" mean entirely above or below 0. "Decision" is the primary's
four-way rule (95% interval for a difference, 90% interval inside (-0.03,
0.03) for equivalence; S6 and S7 report the 90% interval); every interval
is a 95% interval unless stated. Each S8 test reads only its set's worlds,
and its interval treats them as independent although the
whole-graph sets share one hidden structure and per-goal closures overlap;
the S8 tests are descriptive of those fixed worlds.

### Predictions (written before any measurement run)

Written after the development runs listed under Implementation, whose
outcomes (two worlds per cell) were seen.

- P1 (primary): posterior better. Reason: the set learners use only the
  eliminations that successes give, which the posterior also uses, and
  ignore failures, which the posterior uses with the class prior; a recipe
  without a success has no set model at all; MSGI uses failures but has no
  prior over requirement sizes and few examples per recipe, and its rule
  "eligible at every observation: no precondition" discards the
  eliminations of successful attempts. Expected best baseline: `msgi` or
  `observer` (development, two worlds, selected `msgi`).
- S1: posterior better on `random_omit` evidence.
- S2: on `maximal` evidence every craft succeeds with every candidate
  present, so on the generated worlds `posterior`, `pair_weight`, `msgi`,
  `observer`, `sam` and `random` are exactly 0.5 AUROC at every checkpoint,
  and `arms` is 0.5 in expectation (nothing in the logs depends on the
  hidden requirement). Converted worlds are excluded (see Limits).
- S3: holds (a check of the macq pipeline).
- S4: holds. S5: holds.
- S6: equivalent within 0.03, as M1 found on goal-only logs; no direction is
  predicted (development, two worlds, had the pairwise weights ahead).
- S7: equivalent within 0.03 (a replication of M1's outcome on new worlds).
- S8: holds on every set, and on MSGI Mining without the two worlds seen in
  development (`bench_msgi_mining:s0:w5` and `:w10`).
- Exploratory: ARMS below Observer, at hash seed 0, the reported tie-break
  (at other seeds this may change: development gave ARMS up to 0.751 at seed
  3, against Observer 0.807, on two worlds; the hash-seed check reports it);
  the safe sets of `observer` and `sam` never make a plan fail (plan success
  1), at the price of gathering decoys; all checkpoints; ranked gathering
  cost; counts of MSGI statuses (tree, always, never, failed) and ARMS job
  statuses (ok, iteration_cap, error, timeout, worker_exit, no_traces).

### Statistics

The unit is the world; generated worlds are independent across seeds.
Every contrast is a paired per-world difference; its mean gets a percentile
world-cluster bootstrap interval (10,000 resamples, seed 20261008, the
bootstrap of study M), 95% unless stated. One primary contrast; the
secondary tests have their own criteria and no multiplicity correction.
Runs: one per (set, policy, seed), the earliest; a run that did not complete
(including a run killed before it wrote evidence: every run folder with a
manifest and a config counts) is replaced by the earliest later completed
run with the same config hash, commit and loaded code (runs are
deterministic); replaced, ignored and incomplete chosen runs are listed
(`collection.load_evidence`). The analysis refuses evidence whose chosen
runs come from more than one (commit, code).

### Budget

Worst cases are the sums of the task budgets of each run's stream (dry run,
generator only). A run's meter caps it at the smaller of its stream sum and
the ledger's remaining allowance of its kind (`training/budget.py`,
`BudgetMeter`), so a run could be cut short only if the allowance were
smaller than its stream sum; `collect` refuses to start such a run
(`AllowanceShort`), so no run is cut short.

| Stage | Config | Runs | Adaptive worst case | Reporting worst case |
|---|---|---|---|---|
| Development | `dev.yaml` | 15 | 4,540 | 2,270 |
| Measurement | `x2_main.yaml` | 45 | 116,118 | 58,059 |

Per-run sums: `gen_obs` and `gen_goal` 3,569 / 3,681 / 3,619 / 3,726 /
3,693 (seeds 0-4); `bench_crafter` 2,616, `bench_psketch_craft` 2,177,
`bench_msgi_mining` 4,628, `bench_little_alchemy_2` 7,461,
`bench_freeciv_classic` 4,601; per policy 58,059. The largest run of either
kind is a Little Alchemy 2 run (7,461).

Ledger `runs/X2-ledger.json`, declared identically in both configs, sized so
that one rerun of the largest run of each kind fits after the worst case:
allocation `x2` 123,700 (116,118 + 7,461 = 123,579, rounded up to the next
hundred), allocation `dev` 10,000, adaptive cap 133,700 (10,000 + 123,700),
reporting cap 67,100 (58,059 + 7,461 + 1,528 reporting interactions used in
development = 67,048, rounded up to the next hundred). The ledger file was created by the first development
run with the earlier declaration (`x2` 116,200, adaptive cap 126,200,
reporting cap 63,000); the raise is applied as a logged amendment at the
freeze, before any measurement run.

### Stopping rules

No interim analysis. A run stops only at the end of its stream. A run, or a
rerun of a failed run, starts only if the ledger's remaining allowance of its
kind covers its stream sum; otherwise a logged ledger amendment comes first.
After an interruption, `collect --resume` skips every (set, policy, seed)
that already has a completed run with the same config content, commit and
code; every collection session runs from the one snapshot of the freeze
commit. The analysis (`rank`) runs once after all 45 collection runs; after
an interruption `rank --resume` reuses the cached rows and ARMS results of
the same code, options and library versions. A run that fails without a
code change (for example a killed process) is handled by the rerun rule and
recorded as an amendment. A fault that needs a code fix means a new freeze
and a fresh collection of every run from the new snapshot, since `rank`
refuses evidence from more than one (commit, code).

### Limits stated in advance

- One task family for the primary (F1, deterministic dynamics, pools of six,
  at most three hidden requirements), the class that the posterior's prior
  matches; the generator draws requirement sizes with weights 0.2, 0.5 and
  0.3 while the prior is uniform over the class.
- The primary's evidence comes from a policy that samples from the
  posterior, so its logs are adapted to the posterior's hypothesis class;
  S1 (`random_omit`) is the method-independent check.
- The baselines are applied outside their home settings: MSGI expects
  eligibility of every subtask at every step and gets attempt outcomes;
  the action-model learners expect traces of executed actions and get the
  successful steps only. These are the adaptations that give them the same
  logs; other adaptations could score differently.
- The STRIPS conversion drops failed attempts, so the set learners cannot
  use the evidence that most separates candidates; this is a property of
  the representation, reported as such.
- The posterior and the pairwise weights use the declared hypothesis class
  (nonempty, at most three of the six candidates); the baselines use only
  that requirements are nonempty (the inferred-set rule). Baselines extended
  with the class could do better; that is not tested.
- ARMS runs with three library changes (weights paired with their clauses,
  empty constraint groups, reproducibility) and a step limit; its settings
  are the library's defaults except `upper_bound`, not tuned. It is reported
  for one arbitrary tie-break (string hash seed 0); on two development
  worlds its mean AUROC ranged from 0.554 to 0.751 (`focused_sample`) and
  from 0.512 to 0.629 (`random_omit`) over hash seeds 0 to 3, so its value,
  its rank among the baselines and the direction of the pairing fix are
  properties of that draw; the exploratory hash-seed check shows the spread
  on the selection worlds. Its wall-clock safety limit depends on machine
  load; timeouts are cached and counted.
- In converted worlds the decoys' type ids are not exchangeable with the true
  candidates' ids (decoys come from vocabulary types without data meaning),
  so a method that breaks ties by type id (MSGI's CART keeps the lowest-index
  feature among equal splits; ARMS's solver order follows variable names)
  can depart from chance for reasons unrelated to the evidence. S2 is stated
  for generated worlds only for this reason.
- Acting cost is evaluated open-loop on the recorded tasks (the logged
  episodes themselves, in-sample for the checkpoints that include them).

## Implementation

- `src/hypergraph_agent/inference_baselines/`: `traces.py` (logs to state
  steps, labelled attempts and STRIPS steps), `msgi_adapter.py`,
  `macq_adapter.py` and `arms_worker.py`, `learners.py` (factorized
  posterior, SAM-style learner, set scores), `collection.py` (the study's
  own config schema, benchmark streams, the logged `random_omit` policy,
  collection with resume and the allowance check, evidence loading),
  `benchmark.py` (scores, metrics, acting cost, selection, decisions,
  secondary tests, cache and resume, report), `provenance.py` (code hash,
  vendored-file hash, library versions, cache key), `__main__.py`
  (commands).
- `third_party/msgi/` (MIT; `SOURCE.md` with commit and SHA-256).
- `configs/inference_baselines/`: `dev.yaml`, `x2_main.yaml` (own schema;
  listed in `tests/test_hygiene.py`).
- `rank` caches every row (one per run, world and checkpoint) and every ARMS
  result under `--cache` (default `runs/x2_cache`, not tracked), in a folder
  keyed by the package's code hash, the vendored MSGI files, the options and
  the versions of python, macq, nnf, python-sat, torch and numpy; every
  worker stamps its code hash on its rows and ARMS results and a mismatch
  is refused; the summary records the commit, the code hash, the versions
  and the cache key (`provenance`) and the ARMS status totals; progress is
  printed as worlds finish. An interruption loses at most the worlds and
  ARMS jobs in flight. A reused row takes the completeness of the evidence
  it comes from now, not when it was cached.
- `tests/test_inference_baselines.py` (25 tests): the logged `random_omit`
  acting like `random_omit`, trace conversion and its refusal on goal-only
  logs, the factorized posterior equal to the joint enumeration, `sam` equal
  to macq's Observer and safe, MSGI recovering a conjunction and its two
  rules, the vendored files' checksums, ARMS reproducible across worker
  processes, every ARMS soft clause carrying the weight `step3` gave it, the
  step limit keeping a model (with `upper_bound` 99 a job stopped after 2
  steps has a precondition for every craft and the same model as after 3),
  the provenance code hash equal to the run manifests' hash, set scores and
  the inferred-set rule, acting cost (exact on the truth, brute force on
  full pools, failures on a wrong node), the planner's optimum equal to the
  reference solver, the decision rule and the baseline selection, parallel
  and sequential world metrics identical, config validation (namespaces,
  keys, one ledger declaration), the measurement plan within its
  allocation, benchmark streams, the whole pipeline end to end (including
  brute-force evidence at chance and no element names of Little Alchemy 2
  or Freeciv in the report, with worlds that are not measurement worlds),
  `collect --resume` and the allowance refusal, crashed runs listed and
  replaced and mixed (commit, code) refused, the rank cache and resume with
  the selection's completeness rule, the status totals, the exploratory
  hash-seed check and the code-stamp refusal, and line lengths.

Commands (from the repository root):

```bash
python -m hypergraph_agent.inference_baselines collect \
    --config configs/inference_baselines/x2_main.yaml --dry-run
python -m hypergraph_agent.inference_baselines rank --prefix x2-dev- \
    --out ../scratch/x2_dev --expected-selection-worlds 2 --resume
```

Measurement from a snapshot (after review and freeze), from the repository
root, so that `runs/` and `artifacts/` are the repository's; `S` is a clean
worktree of the freeze commit, whose code both commands run. The fetched
data is not in Git, so it is copied into the snapshot and checked first:

```bash
git worktree add ../snapshots/X2-<hash> <hash>
S=../snapshots/X2-<hash>
cp -r benchmarks/little_alchemy/data $S/benchmarks/little_alchemy/
cp -r benchmarks/freeciv/data $S/benchmarks/freeciv/
.venv/Scripts/python.exe $S/benchmarks/little_alchemy/fetch.py --check
.venv/Scripts/python.exe $S/benchmarks/freeciv/fetch.py --check
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.inference_baselines collect \
    --config $S/configs/inference_baselines/x2_main.yaml --allow-measurement --resume
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.inference_baselines rank \
    --prefix x2-main- --out artifacts/inference_baselines --workers 4 --arms-workers 4 --resume \
    --arms-hash-seeds 1 2 3 4
```

`--resume` is harmless on a first run (nothing to reuse) and makes a rerun
after an interruption continue where it stopped. `--arms-hash-seeds 1 2 3 4`
adds the exploratory check after the analysis (288 ARMS jobs: 24 selection
worlds, three evidence sources, four seeds; about 5 to 20 minutes) and
writes it to `artifacts/inference_baselines/x2_benchmark_arms_hash_seeds.json`;
it does not change the main artifact.

### Development record (calibration, not results)

Collection: allocation `dev`, seed 0, 16 episodes per world, every policy
(`configs/inference_baselines/dev.yaml`, stamp 20261022-173714): two
generated worlds per observation mode (namespace `x2_gen_dev`), one Crafter
world (conversion seed 100), two MSGI Mining worlds and one Little Alchemy 2
world (goal-sample seed 99); 15 runs, all completed. Ledger after
development: 3,185 adaptive (`dev`) and 1,528 reporting interactions; the
review fixes added no collection run. Collection took 0.2 to 4.0 s per run
(0.006 to 0.12 s per task).

Offline benchmark of the 15 runs, first version (permuted ARMS weights, no
step limit): 413 s on a machine shared with other jobs, one ARMS job of 72
stopped at the 120 s wall-clock limit. After the first review's fixes:
72.5 s as the summary records it (one process for the methods, two ARMS
workers, a less loaded machine); ARMS 70 jobs completed and 2 stopped at the
step limit, none timed out; the goal-only posterior exact for every recipe.
In the second review the same benchmark with `--workers 4` took 138 s
against 72.5 s in one process, with identical results: each world process
spends about 7 s on imports, more than a small development world takes.
Resume check: a run interrupted after 35 s had cached all 72 ARMS results
and 13 of the 24 world records; `rank --resume` reused them, computed the
other 11 and reproduced the uninterrupted run's tables and secondary tests
exactly.

Expected measurement runtime from these rates: collection about 15 minutes
(about 10,000 tasks); offline benchmark about 1 to 3 h with four ARMS
workers (1,800 ARMS jobs, 720 goal-only world-checkpoints, whose posterior
took up to about 100 s each in development), more on a loaded machine; the
exploratory hash-seed check adds about 5 to 20 minutes.

In-memory checks (unit tests and scratch runs with one world and up to
eight episodes per set, in-memory ledgers, scratch folders, development or
scratch namespaces) are not charged to the ledger.

Decisions that came from development and the review:

1. ARMS returned three different models for one job in three processes
   with `PYTHONHASHSEED` fixed (nnf's random auxiliary names and
   address-based node hashes); fixed in the worker and tested.
2. The joint enumeration on items-observable logs: one Crafter world with
   four episodes of `random_omit` evidence took 445 s and fell back to
   Gibbs for 6 of its 11 scored recipes (scratch run). With items observable
   the posterior is therefore computed in its factorized form, which is
   exact; its equality with the enumeration is tested where the enumeration
   is exact.
3. macq's `max([])` on an empty constraint group: guarded.
4. The review found the permuted ARMS weights; X2 runs the weights `step3`
   assigns (Methods). At string hash seed 0, the reported tie-break, the
   development check moved ARMS's AUROC on the `gen_obs` worlds at 16
   episodes from 0.630 to 0.554 (`focused_sample`) and from 0.553 to 0.522
   (`random_omit`), against Observer 0.807 and 0.724 and MSGI 0.848 and
   0.742 (two worlds each). The second review showed that this direction is
   a property of the seed (the fix moves ARMS by -0.08 to +0.32 over seeds 0
   to 3; Methods), which the exploratory hash-seed check addresses.
5. The one ARMS job that hit the 120 s limit repeated the same 54-clause
   MAX-SAT problem without end after its first step; the deterministic step
   limit (50 steps, model as learned so far) replaces the wall clock as the
   working limit.
6. Parallel computation of worlds (`--workers 4`) failed once with a broken
   process pool on the shared machine (worker processes ended with "the
   handle is invalid"); the benchmark recomputes in the parent process,
   deterministically, any world whose worker fails and reports the count
   (`parallel_fallbacks`); a test checks that parallel and sequential
   metrics are identical. The development benchmarks ran with one process.
7. With two development worlds, `msgi` was the selected baseline for
   `focused_sample` and `random_omit` evidence, and on `maximal` evidence
   every method except `arms` gave exactly 0.5 AUROC (prediction S2).

## Amendments

None.

## Results

None yet: no measurement run exists.

## Verdict and limits

None yet: written after the measurement, from the frozen analysis, against
the predictions and limits above.
