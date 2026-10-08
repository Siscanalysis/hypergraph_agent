# Study X: the repository's agents against open-source reinforcement-learning baselines

Status: draft protocol, not frozen. No tuning or measurement run has been
made. The development checks listed under Implementation used only the `dev`
allocation of `runs/X-ledger.json` and namespaces ending in `_dev`; they are
calibration, not results.

## Question and claim

Question: on the repository's environments, how do standard deep
reinforcement-learning learners from maintained open-source libraries, trained
across many worlds with the same public information, compare with the
repository's agents, which learn only within each world's stream of episodes?

The learners are Stable-Baselines3 PPO and sb3-contrib's MaskablePPO and
RecurrentPPO, plus MaskablePPO with a random-network-distillation curiosity
bonus; each trains on up to 1,000,000 interactions of fresh training worlds
and is then evaluated frozen on test worlds that the repository's agents also
play, task for task.

The claim under test is narrow: on RecipeQuest family F1 with goal-only
observation (the family of study R), the sampling walker `focused_sample`
acts more cheaply over the late half of each test world's stream (episodes
9-16) than the best of these learners after 1,000,000 training interactions.
The secondaries ask the same of the TechTree agents of studies U and L and,
out of distribution, on RecipeQuest worlds converted from five published
recipe graphs (Crafter, the Craft world of policy sketches, MSGI Mining, Little
Alchemy 2, Freeciv); the learning curves describe sample efficiency. The comparison is not
symmetric by design and the study does not hide it: the repository's agents
carry declared inductive biases (a hypothesis class of candidate subsets, a
derivation planner, window elimination with exact discoveries), while the
learners start without them and get training worlds from the same generator
instead. The study measures what a generic learner achieves from data alone
under declared budgets, not which kind of agent is better in general.

## Relation to earlier work

- The first session's own recurrent PPO policies, trained for 4,500
  interactions per run, solved 0 of 20 tasks in P1, 0-2 of 27 in P2 and at
  most 2 of 20 longer compositions in P3 (docs/RESULTS.md, Sections 4.1-4.3);
  the walkers then solved those tasks (Section 4.4). Those policies were the
  repository's own implementation at a small budget, so a reader can fairly ask
  whether the walkers were compared with a weak learner. X answers with
  maintained library implementations, library defaults, a declared tuning
  stage and budgets up to about 220 times the pilot's.
- Study R fixed the RecipeQuest design reused here (family F1, 10 worlds x 16
  episodes per seed, late half 9-16, world-cluster bootstrap) and measured
  `focused_sample` at 1.51 and `maximal` at 1.87 steps per optimal step in the
  late half on its 50 worlds. Studies U and L fixed the TechTree cells reused
  here (U: k3-off-ann; L: phase L2 at reuse depth 2) and their arms.
- benchmarks/README.md catalogues the external benchmarks, their licenses,
  their converters to RecipeQuest and how far the generated families
  represent them; X uses its RecipeQuest converters unchanged. A separate
  study (X2) compares structure-inference baselines (MSGI-style inference,
  action-model learning); X stays on acting agents and reinforcement-learning
  learners.
- docs/PRIOR_ART.md, Section 8.3: Alchemy (C1) resamples a latent causal
  structure in every episode and reports that the deep RL agents it tested
  showed no appreciable structure learning; exact sequences without an
  intermediate signal are a combination lock (C10).
- Learners and techniques (to be entered in docs/PRIOR_ART.md before the
  freeze): Stable-Baselines3 (Raffin et al., JMLR 22(268), 2021); PPO
  (Schulman et al., arXiv:1707.06347); invalid-action masking for policy
  gradients (Huang and Ontanon, FLAIRS 2022, arXiv:2006.14171); recurrent
  policies trained across trials as fast within-task learners, RL^2 (Duan et
  al., arXiv:1611.02779) and learning to reinforcement learn (Wang et al.,
  arXiv:1611.05763); random network distillation (Burda et al., ICLR 2019,
  arXiv:1810.12894).

## Protocol (draft, to be frozen)

### Settings

| Setting | Environment | Training namespace | Validation (selection) | Test | Late half |
|---|---|---|---|---|---|
| `rq_f1` | RecipeQuest F1: Stage D generator, hidden prerequisites (pools of 6, 41 hypotheses per recipe), goals at depth 1-3, intermediate items unobserved, no distractor base facts, failure probability 0 | `x_rq_train` | `x_rq_val`: 2 seeds x 10 worlds x 16 episodes | `x_rq_test`: 5 seeds x 10 worlds x 16 episodes (50 worlds) | episodes 9-16 |
| `tt_u_k3` | TechTree, study U cell k3-off-ann: 4 base primitives, 3 extra slots, a key of 3 base primitives that activates one slot, goal an entry of 3 symbols using it; no signal; announced unlocks; budget 100 | `x_tt_u_train` | `x_tt_u_val`: 2 seeds x 6 worlds x 12 | `x_tt_u_test`: 5 seeds x 6 worlds x 12 (30 worlds) | episodes 6-11 (from 0) |
| `tt_l2_d2` | TechTree, study L phase L2 at reuse depth 2: 3 base primitives, three levels of three concepts (lengths 2, 4, 6), levels 2 and 3 compositions, nothing supplied; budget 100 | `x_tt_l_train` | `x_tt_l_val`: 3 seeds x 4 worlds x 12 | `x_tt_l_test`: 8 seeds x 4 worlds x 12 (32 worlds) | episodes 6-11 (from 0) |

Benchmark test sets (out of distribution; evaluation only, `bench_*.yaml`):
RecipeQuest worlds converted from published recipe graphs by
`benchmarks.to_recipequest` (pools of 6, at most 3 hidden base facts and 2
alternatives per item), played with the F1 task settings (hidden
prerequisites, goal-only observation, no distractor base facts, budget two
steps above the public brute-force bound) for 16 episodes per world, each
episode's goal drawn uniformly among the world's items
(`baselines/benchmark_sets.py`). The learners are the `rq_f1` checkpoints at
1,000,000 interactions, trained only on generated F1 worlds, so these worlds
are out of distribution for them; the repository's agents play the same
tasks. Conversion record (means per world, from `benchmark_sets.records`):

| Set | Worlds | Kind | Items | Promoted elements | Dropped recipes | Share of the closure kept | Pool decoys without data meaning |
|---|---|---|---|---|---|---|---|
| `bench_crafter` | 10 | whole graph, one world per conversion seed | 16 | 1 (`wood`) | 5 | 0.94 | 0 |
| `bench_psketch_craft` | 10 | whole graph, one world per conversion seed | 11 | 0 | 0 | 1.00 | 0 |
| `bench_msgi_mining` | 20 | one goal's closure per world (20 of 23 goals) | 6.2 | 0 | 0 | 1.00 | 1.2 |
| `bench_little_alchemy_2` | 30 | one goal's closure per world (30 of 682 goals) | 5.2 | 6.0 | 4.3 | 0.14 | 2.6 |
| `bench_freeciv_classic` | 20 | one goal's closure per world (20 of 80 goals) | 4.9 | 3.7 | 0 | 0.49 | 3.5 |

The record (per world: goals, promoted elements, dropped recipes, kept share)
is part of every reported result: in Little Alchemy 2 only 14% of a goal's
original closure remains items (the rest is promoted to gatherable base
facts), so those worlds test the hidden base requirements of short chains,
not the deep discovery structure of the game. Whole-graph worlds of one
benchmark share one hidden structure (a conversion seed changes vocabulary
types and decoys only), and per-goal closures of one graph overlap, so the
worlds of a set are not independent draws of structure. Little Alchemy 2 and
Freeciv data are fetched (`benchmarks/<name>/fetch.py`, not committed);
results derived from them may be reported.

Test streams have the layout of the studies they come from: RecipeQuest seed
s plays base seed 10000 + 1000 s in pool mode (world-major order, exactly
`evaluation.walkers.eval_order`); TechTree seed s is the stream seed s. Every
namespace is fresh (never used by any earlier study or development run).
Training worlds of training seed s and environment k of a run are worlds k,
k + 8, k + 16, ... of the training namespace (base seed 30000 + 1000 s), each
visited for its whole block of episodes in order; no world recurs within a
run, training worlds never coincide with validation or test worlds, and
every learner of a training seed sees the same training worlds.

### Public information (wrappers)

`baselines/wrappers.py` and `baselines/encoders.py` give each setting a fixed
`Box` observation in [0, 1] and a fixed `Discrete` action set with a mask
(`action_masks()`):

- RecipeQuest (52 actions, 1,339 observation entries). Actions canonical by
  type: gather each resource type, activate each facility type, a craft slot
  for each (item type, recipe variant) with the variants of a task ordered by
  their public signatures, wait, submit; actions whose fact or recipe is absent
  from the task are masked. Observation: facts present in the task by type;
  goal one-hot; for every craft slot whether it exists, its item inputs by
  type and its public candidate pool by type (never the hidden requirement);
  observed present facts by type (intermediate items stay absent under
  goal-only observation, as the environment reports them); budget left and
  steps taken as fractions of the budget; the outcome of the last action
  (changed, unchanged, unobserved, none); the last action; per-action counts in
  the episode; the episode context (episode index within the world, first step
  of a task, previous task of the world solved).
- TechTree (9 actions and 69 entries in `tt_u_k3`; 5 and 105 in `tt_l2_d2`).
  Actions: the task's primitives, slots, wait, submit; the mask is the public
  available-action mask. Observation: goal one-hot; public concept levels and
  level lengths; flags for the signal and silent unlocks; held concepts; the
  available-action mask; the progress flag when the cell has one; budget left
  and steps taken; the agent's own primitive trace since the last wait or
  submit (its last 3 or 6 presses); the last action; per-action counts;
  concepts that fired at the last step; the episode context.
- Information boundary. Observations come only from the public spec, the
  public observation, the public `info` of a step and the agent's own actions;
  TechTree is played through `env.agent_interface` with opaque task tokens.
  Tests: twin RecipeQuest tasks whose hidden requirements differ give
  identical observation sequences and masks under any history that never
  crafts the goal (the only craft whose outcome is observed); TechTree twins
  with different hidden sequences or unlock slots give identical observations;
  the learner-side modules never name a hidden field; `info` passed to a
  learner holds only public outcome data.
- Episodes. Task mode: one learner episode per task (consecutive episodes
  still walk through each world's episodes in order). Trial mode (RL^2-style):
  one learner episode per world block, the next task of the world starting
  inside the same learner episode, so a recurrent policy can carry what it
  learned about the world.
- Invalid actions. A learner without masks that chooses a masked action
  executes `wait`: one primitive step of the task budget and one ledger
  interaction, as any no-op; no other penalty.
- Reward: the environment's (1 on successful submission, else 0); no shaping
  (`step_penalty` 0). The discount of the libraries (0.99) is the only pressure
  towards short episodes.

### Learners

| Learner (id) | Library class | Episodes | Masks | Settings | Declared deviations from the library defaults |
|---|---|---|---|---|---|
| `mppo` | sb3-contrib `MaskablePPO`, `MlpPolicy` | task | yes | all | `n_steps` 256 per environment (8 x 256 = 2,048 per rollout, the default rollout of one environment) |
| `ppo` | Stable-Baselines3 `PPO`, `MlpPolicy` | task | no (invalid: `wait`) | `rq_f1` | as `mppo` |
| `rppo` | sb3-contrib `RecurrentPPO`, `MlpLstmPolicy` | trial | no (invalid: `wait`) | all | one linear-ReLU layer of 64 features before the LSTM, one LSTM of 64 units shared by actor and critic, minibatches of 512 (see Implementation: the default ran at 3-10 interactions per second) |
| `mppo_rnd` | `MaskablePPO` + RND reward wrapper | task | yes | all | as `mppo`; RND bonus coefficient 0.01, target and predictor MLPs 128-128-64, Adam 1e-4, predictor trained every 128 vectorized steps |

PPO without masks runs only in `rq_f1`, where most of the 52 canonical actions
are absent from any task; in TechTree the mask covers at most the three
inactive announced slots (`tt_u_k3`) or nothing (`tt_l2_d2`, where MaskablePPO
and PPO coincide). DQN is implemented and passed a development smoke run but
is not in the measurement design: its default replay buffer of 10^6
RecipeQuest observations does not fit in memory, and a reduced buffer would
be a further deviation; an amendment may add it. Common to all: 8 environments
per run in one process, CPU, one torch thread per process, library seed
derived from (setting, training seed), identical for every learner of a
training seed. RND follows Burda et al. with one combined episodic reward and
value head instead of two value heads (`baselines/learners.py`).

### Tuning stage (pre-registered; development namespaces only)

Configs `configs/baselines/tune_*.yaml`, allocation `x_tune`. Each candidate
trains once (training seed 0) for 100,000 interactions on a `_dev` training
namespace and is evaluated in both action modes (deterministic, stochastic) on
a `_dev` evaluation namespace (`rq_f1`: 2 seeds x 10 worlds; `tt_u_k3`: 2 x 6;
`tt_l2_d2`: 3 x 4). Candidates per learner:

| Learner | Candidates (in tie-break order) |
|---|---|
| `mppo`, `ppo` | as declared above; `ent_coef` 0.01; `ent_coef` 0.01 with `net_arch` [256, 256] |
| `rppo` | as declared above; `ent_coef` 0.01; `ent_coef` 0.01 with `gamma` 0.999 (credit across the episodes of a trial) |
| `mppo_rnd` | RND coefficient 0.01; 0.1 |

Selection (`python -m hypergraph_agent.baselines analyze --config
configs/baselines/tune_<setting>.yaml`, field `tuning`): per learner, the
(candidate, mode) with the lowest mean of the setting's selection metric over
the tuning worlds at 100,000 interactions (late cost ratio for `rq_f1`, late
actions per episode for TechTree); ties go to the earlier candidate, then to
the deterministic mode. The selected parameters and mode are written into the
measurement configs by a dated amendment that also sets `run.tuning_record` to
the tuning analysis; the measurement CLI refuses a config whose
`tuning_record` is unset. Nothing else changes after the tuning outcome is
seen.

### Training and evaluation regimes

- Cross-world: per setting, every learner trains with training seeds 0 and 1
  for 1,000,000 interactions; the policies acting at the 10,000th, 100,000th
  and 1,000,000th training interaction are saved (`train.budgets`) and
  evaluated frozen on the whole test stream; the 1,000,000 checkpoint is also
  evaluated on the validation stream. A recurrent learner's state starts empty
  at the start of each test world (trial mode) and is never trained further.
- Learning curves: the test endpoints at 10,000, 100,000 and 1,000,000
  training interactions.
- Benchmark test sets: the `rq_f1` checkpoints at 1,000,000 interactions of
  both training seeds, in each learner's selected mode, on every benchmark set
  (`run.checkpoints_from: configs/baselines/rq_f1_main.yaml`, evaluation only),
  next to `focused_sample`, `maximal`, `reference` and `random`. The best
  learner is the one selected for `rq_f1` (on F1 validation worlds), not
  re-selected per set.
- The repository's agents on the same test streams, task for task:
  `focused_sample`, `maximal`, `reference` (privileged) and `random` (uniform
  over the task's gather, activate and craft actions, submitting once the goal
  is observed held) in `rq_f1`; `pooled`, `oracle` (privileged), `reference`
  and `random` in `tt_u_k3`; `remember`, `promote`, `oracle_library`
  (privileged), `reference` and `random` in `tt_l2_d2`. They play exactly as in
  studies R, U and L (same modules, parameters and common random numbers);
  they use no training worlds.

### Endpoints

Per test world (the unit), per arm (`baselines/analysis.py`): the late-half
cost ratio (primitive steps over optimal steps summed over the late half; a
failed episode counts every step it used, its whole budget), early-half and
overall cost ratios, the late minus early cost ratio (within-world
improvement), mean actions per episode (late, early, all), success rates, and
actions to the first success in the world (censored at the stream total). A
learner's value in a world is the mean over its two training seeds.
Interactions used for training (from the ledger), interactions each
repository agent spends per test world, and wall-clock times (training,
evaluation loops) are reported.

### Best learner

At 1,000,000 interactions, the learner with the lowest mean selection metric
over the validation worlds (`rq_f1`: late cost ratio; TechTree: late actions
per episode), each world's value averaged over the two training seeds; ties
go to the earlier learner in the config. The selection reads no test row.
`best_learner` in the tests below is that learner, its test value per world
the mean over its two training seeds.

### Primary contrast

X1, `rq_f1`, late cost ratio, `focused_sample` minus the best learner at
1,000,000 interactions, paired by test world (50 worlds). It passes if the
upper end of the 95% world-cluster bootstrap interval is below 0.

### Secondary contrasts (each with its own criterion, no multiplicity correction)

| Test | Setting | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| X1s | `rq_f1` | `focused_sample` - best learner | late success | interval not entirely below 0 |
| X2 | `rq_f1` | `maximal` - best learner | late cost ratio | interval not entirely above 0 (the learner does not beat the brute-force plan) |
| X3 | `rq_f1` | best learner - `random` | late cost ratio | upper end below 0 |
| XU1 | `tt_u_k3` | `pooled` - best learner | actions to the first success | upper end below 0 |
| XU2 | `tt_u_k3` | `pooled` - best learner | late actions per episode | upper end below 0 |
| XU3 | `tt_u_k3` | best learner - `random` | actions to the first success | upper end below 0 |
| XU4 | `tt_u_k3` | best learner - `random` | late actions per episode | upper end below 0 |
| XL1 | `tt_l2_d2` | `promote` - best learner | late actions per episode | upper end below 0 |
| XL2 | `tt_l2_d2` | `remember` - best learner | late actions per episode | upper end below 0 |
| XL3 | `tt_l2_d2` | best learner - `random` | late actions per episode | upper end below 0 |
| XB (5 tests) | each benchmark set | `focused_sample` - best learner | late cost ratio | upper end below 0 |

Each XB test reads only its set's worlds (10-30), and the interval treats
them as independent although they share structure (see Settings); a pass
describes those fixed worlds, not the benchmark's population of tasks.

Descriptive: per benchmark set, `maximal` minus the best learner and
`focused_sample` minus `maximal` on the late cost ratio (the latter is the
study R contrast on these worlds); the within-world improvement (late minus early cost ratio) of
the best learner and of `focused_sample` (X4, X4r, XU5, XL4); learning curves
of every learner against the repository's agents; per-training-seed values of
every contrast; the training interactions after which a learner first reaches
the late cost of `maximal` (`rq_f1`) or of `random` (TechTree), if any;
invalid actions chosen by unmasked learners.

### Predictions (written before any tuning or measurement run)

Written after the development runs listed under Implementation, in which no
learner solved any task at 8,000 or 16,000 training interactions.

- X1 passes by a wide margin. No learner reaches the late cost of `maximal`
  (1.87 in study R): without memory across episodes a learner cannot use the
  evidence of earlier episodes of a world, and the best memoryless strategy
  under a budget of two steps above the brute-force bound is the brute-force
  plan. Expected best-learner late cost ratio 2.0-2.8 with late success below
  0.95, against about 1.5 for `focused_sample`. X1s holds. X2 holds.
- X3 holds: by 1,000,000 interactions the best learner has learned to gather
  candidates and craft in order (random: about 3.1, success near 0, in
  development).
- The best learner is `mppo` or `mppo_rnd`; `ppo` and `rppo` are costlier at
  equal budgets. `rppo` in trial mode shows no within-world improvement whose
  interval excludes 0; the feedforward learners' late minus early cost ratio
  is within 0.05 of 0.
- Learning curves: at 10,000 interactions every learner is near `random`; most
  of the gain comes between 100,000 and 1,000,000.
- `tt_u_k3`: XU1, XU2, XU3 hold. The best learner's actions to the first
  success lie between 70 and 300 (`pooled` 67 in study U; `random` 348); its
  late actions per episode stay above 40 (it searches again in every episode),
  against about 7 for `pooled`. XU4 holds (late actions below those of
  `random`, about 85).
- `tt_l2_d2`: XL1 and XL2 hold; the best learner fails most late episodes
  (above 90 actions per episode; `remember` 27.7, `promote` 10.7 in study L);
  XL3 does not hold (the best learner is within a few actions of `random`,
  96.9 in study L).

Benchmark test sets (written before any run on them; one development run on
one Crafter world, listed under Implementation, was seen):

- XB holds in all five sets: frozen learners trained on F1 neither carry
  evidence across episodes nor know these structures. No learner's late cost
  ratio is below that of `maximal` on any set.
- `focused_sample` - `maximal` (descriptive): below 0 in Crafter, the Craft
  world and Freeciv (recurring recipes, no or few alternatives); not below 0
  in MSGI Mining (brute-force headroom about 1.3, little room to save) and
  Little Alchemy 2 (many alternatives, as in family F2, where the sampler did
  not break even).

### Fairness rules

1. Identical public information: the learners' observations are built from
   the same public spec and observations the repository's agents receive, plus
   their own action history, which those agents also keep. Feedforward
   learners keep no memory across episodes; the recurrent learner in trial
   mode can, as the repository's agents do within a world.
2. Identical test tasks: the same worlds, tasks, order and dynamics seeds for
   every arm (tested).
3. Declared training budgets: 10,000, 100,000 and 1,000,000 interactions on
   training worlds disjoint from validation and test worlds; every training
   interaction is charged to the ledger and checked against the library's own
   count (`run_train` refuses a mismatch).
4. Library defaults except the declared deviations; tuning only on `_dev`
   namespaces with the grid and rule above; the best learner chosen on
   validation worlds, never on test worlds.
5. Unmasked learners pay for invalid choices exactly as for any no-op.

### Statistics

The unit is a test world; worlds are independent across test seeds. Paired
per-world differences; the mean gets a percentile world-cluster bootstrap
interval (`evaluation.stats.cluster_bootstrap_mean`, 10,000 resamples, seed
20261008, 95%), with an exact two-sided sign test. Training seeds are not
resampled: a learner's per-world value is the mean over its two training
seeds, so the intervals do not include training-seed variability; the
contrast against each training seed is reported next to each test. A test
whose paired worlds are fewer than the test stream's worlds is "incomplete"
and does not hold.

### Budget and runtime

Ledger `runs/X-ledger.json` (created by the first development run with the
declaration shared by every config of `configs/baselines/`): allocations
`dev` 300,000, `x_tune` 4,500,000 and `x_main` 40,000,000 adaptive; adaptive
cap 44,800,000; reporting cap 5,000,000. Training is `adaptive`; frozen
evaluation is `reporting`, except for the recurrent learner in trial mode,
whose state adapts within a world (`adaptive`); `focused_sample`, `pooled`,
`remember` and `promote` are `adaptive`, the other agents `reporting`. A run's
cap is its training budget or the sum of the task budgets of the streams it
plays, so no cap binds. Worst cases from the dry runs (`run --dry-run`):

| Stage | Runs (train / eval / agent) | Training | Learner evaluation | Agents | Adaptive | Reporting |
|---|---|---|---|---|---|---|
| Tuning `rq_f1` | 11 / 22 / 0 | 1,100,000 | 138,424 | 0 | 1,137,752 | 100,672 |
| Tuning `tt_u_k3` | 8 / 16 / 0 | 800,000 | 230,400 | 0 | 886,400 | 144,000 |
| Tuning `tt_l2_d2` | 8 / 16 / 0 | 800,000 | 230,400 | 0 | 886,400 | 144,000 |
| Measurement `rq_f1` | 8 / 32 / 20 | 8,000,000 | 424,536 | 62,400 | 8,121,734 | 365,202 |
| Measurement `tt_u_k3` | 6 / 24 / 20 | 6,000,000 | 734,400 | 144,000 | 6,280,800 | 597,600 |
| Measurement `tt_l2_d2` | 6 / 24 / 40 | 6,000,000 | 777,600 | 192,000 | 6,336,000 | 633,600 |
| Development (used, exact) | 11 / 23 / 17, and 5 profiling runs | 168,576 | 17,150 | 5,150 | 175,853 | 15,023 |

| Benchmark sets (5 configs) | 0 / 40 / 20 | 0 | 175,872 | 87,936 | 65,952 | 197,856 |

Tuning needs 2,910,552 adaptive interactions at worst (allocation 4,500,000),
measurement 20,738,534 and the benchmark sets 65,952 (allocation
40,000,000); the allocations were declared
before the design was reduced to two training seeds and are upper bounds.

Throughput per process, measured in development on a shared 4-core laptop
(Intel i7-8550U, 8 hardware threads) that other jobs kept at 70-100% load, with
one to three development processes at a time (training interactions per
second, including the policy updates):

| Learner | `rq_f1` | `tt_u_k3` | `tt_l2_d2` |
|---|---|---|---|
| `mppo` | 419 | 301 | 299 |
| `ppo` | 396 | - | - |
| `mppo_rnd` | 186 | not measured (taken as 186) | not measured (taken as 186) |
| `rppo` (declared architecture) | 157 | 374 | 369 |
| `rppo` (library default architecture) | 3-10 | - | - |
| DQN (replay buffer 100,000) | 847 | - | - |

Frozen evaluation ran at about 250-650 interactions per second (one
environment). At these rates the measurement needs about 22 process-hours of
training and 2 of evaluation, about 3 hours of wall-clock time with 8 worker
processes (`run --workers 8`; the longest single job, `rppo` on `rq_f1`, about
1.8 hours); the tuning stage about 3.4 process-hours, under half an hour with 8
workers. The study is sized in interactions; the times are estimates for this
machine.

Commands (after the freeze):

```bash
python -m hypergraph_agent.baselines run --config configs/baselines/tune_rq_f1.yaml --workers 8
python -m hypergraph_agent.baselines analyze --config configs/baselines/tune_rq_f1.yaml
# (the same for tune_tt_u_k3 and tune_tt_l2_d2; then the dated amendment)
python -m hypergraph_agent.baselines run --config configs/baselines/rq_f1_main.yaml --allow-measurement --workers 8
python -m hypergraph_agent.baselines analyze --config configs/baselines/rq_f1_main.yaml
# (the same for tt_u_k3_main and tt_l2_d2_main)
python -m hypergraph_agent.baselines run --config configs/baselines/bench_crafter.yaml --allow-measurement --workers 8
python -m hypergraph_agent.baselines analyze --config configs/baselines/bench_crafter.yaml
# (the same for bench_psketch_craft, bench_msgi_mining, bench_little_alchemy_2, bench_freeciv_classic,
#  after rq_f1_main has completed; the fetch-only data first: python benchmarks/little_alchemy/fetch.py,
#  python benchmarks/freeciv/fetch.py)
```

### Stopping rules and reruns

Every job (a training run, an evaluation run, an agent run) runs once, to the
end of its budget or stream; caps cannot bind. A job with a completed run is
skipped, and a measurement config refuses to start when any of its jobs has a
completed run unless `--resume`, which runs only the missing ones. A run that
did not complete (an error or an interruption) is repeated as a new run with
the same job hash and code; runs are deterministic given the code, the job
hash and the seeds (tested for training and evaluation), so the repeat
replays the same stream; both runs are listed and the completed one is used.
A completed run is never replaced; later repeats are listed and ignored. The
tuning stage is analysed once, after all its runs of a setting completed;
each measurement setting is analysed once, after all its runs completed; the
benchmark sets run after every `rq_f1` measurement run has completed and are
analysed once each, after all their runs completed. No
interim analysis; no change of learners, parameters, settings, endpoints or
criteria after any measurement row exists, other than dated amendments.

### If the primary fails

The estimate, its interval, the per-world differences, the per-seed values and
the learning curves are reported, and the conclusion is that a standard
library learner trained on 1,000,000 interactions of other F1 worlds acts at
least as cheaply as `focused_sample` in the late half; the README's statements
contrasting the walkers with learned policies are then qualified accordingly.

### Limits stated in advance

- One budget scale (up to 1,000,000 interactions) and two training seeds per
  learner; larger budgets, other architectures (attention over the recipe
  structure, graph networks) and other algorithm families are untested.
- Training-seed variability is not in the intervals (two seeds; reported
  separately).
- The recurrent learner uses a reduced architecture for compute reasons;
  the declared alternative is the library default, which did not run at a
  usable speed.
- The fixed observation layouts are one public encoding among many; a learner
  could do better with another, for example one that summarizes the public
  evidence of earlier episodes of a world (which would move part of the
  repository's inference into the encoding, and is therefore not done).
- The tuning grid is small and tuning uses 100,000 interactions, a tenth of the
  largest budget.
- The benchmark sets are conversions: promoted elements, dropped recipes,
  pool decoys and the uniform choice of each episode's goal are part of what is
  tested, and the worlds of a set share structure. They are out-of-distribution
  tests for the learners only; the repository's agents need no training
  worlds, but their hypothesis class (requirements of 1-3 base facts within a
  pool of 6) is the converters' too.

### External benchmarks (loader interface)

The benchmark test sets above use the loader interface of `baselines/sources.py`,
which any further benchmark can use. A setting may take its worlds from an
external loader instead of the generator: `setting.loader: "package.module:function"`, called as
`loader(setting, namespace, seed, world_index, episodes_per_world)` and
returning the tasks of one world block (`envs.generator.Task` for RecipeQuest,
`techtree.generator.TechTask` for TechTree), or None when the benchmark has no
further world (`baselines/sources.py`). Wrappers, references, runner and
analysis need no change as long as the tasks fit the fixed layouts
(RecipeQuest: the type vocabulary and at most `setting.max_variants` recipes
per item type). Study X's loader is `baselines.benchmark_sets:load` with its
parameters in `setting.benchmark` (graph, whole-graph or per-goal worlds,
number of worlds, conversion and goal-sample seeds, pool size, hidden-set
size, alternatives). An evaluation-only config (`run.checkpoints_from`)
evaluates the checkpoints of another config's completed training runs and
takes that config's best-learner selection; a benchmark with its own training
worlds could instead be a training config of its own. Any further test set
gets its own config and fresh namespace and is added by a dated amendment
before its runs.

## Implementation

- `src/hypergraph_agent/baselines/encoders.py`: the fixed layouts
  (`RQLayout`, `TTLayout`) and the episode context; public objects only.
- `wrappers.py`: `RecipeQuestGym`, `TechTreeGym` (fixed spaces, masks, task and
  trial modes, invalid actions as `wait`, ledger charging, the evaluator's
  task-end hook).
- `sources.py` (evaluator side): world blocks of training, validation and test
  streams, stream budgets, references, the external-loader hook.
- `benchmark_sets.py` (evaluator side): the benchmark test sets (worlds from
  `benchmarks.to_recipequest`, episode goals, tasks) and their conversion
  records.
- `learners.py`: the library classes (imported lazily), the feature layer of
  the recurrent learner, the RND reward wrapper, the checkpoint callback and
  the frozen actor.
- `agents.py`: the repository's agents on the X streams (through
  `evaluation.walkers.play` and `techtree.study.play`) and the RecipeQuest
  random agent.
- `runner.py`: jobs, caps, run directories and manifests (provenance and
  library versions), repeats, parallel workers, and a ledger subclass that
  retries reads and replacements blocked by another process (on Windows a
  concurrent reader made `os.replace` fail in development).
- `analysis.py`: per-world metrics, learner averaging over training seeds,
  selection of the best learner and of tuning candidates, contrasts, learning
  curves, tables and a figure; `__main__.py`: `run` and `analyze`.
- Configs: `configs/baselines/*_dev.yaml` and `bench_dev.yaml` (development),
  `tune_*.yaml` (tuning stage), `*_main.yaml` (measurement drafts),
  `bench_*.yaml` (benchmark test sets, evaluation only), with their own schema
  (`baselines/config.py`); `tests/test_hygiene.py` lists `baselines/` among
  the config directories it does not load with the RecipeQuest schema.
- Tests: `tests/test_baselines.py` (layouts and masks, goal-only masking,
  information-boundary twins for both environments, a source scan of the
  learner-side modules, public `info`, invalid actions, trial mode, seeding
  and disjoint namespaces, identity with the walker and TechTree streams, the
  external-loader hook, config rules, an end-to-end run of every learner and
  agent with exact ledger accounting, repeat refusal, deterministic training,
  the RND bonus, the analysis, the selection tie rules, the retrying ledger,
  the benchmark sets and their conversion records, and an evaluation-only
  config that reuses another config's checkpoints and selection).

### Development record (calibration, not results)

Packages: stable-baselines3 2.9.0 and sb3-contrib 2.9.0 were added to the
project environment; they required no change of any installed package
(gymnasium 1.4.0, numpy 2.5.3, torch 2.14.1+cpu, Python 3.12.13).

Development used the `dev` allocation only: 175,853 adaptive and 15,023
reporting interactions (ledger, exact; 190,876 in all), in 56 runs on
namespaces ending in `_dev`:

- training: 155,264 interactions in 11 runs: `mppo`, `ppo`, `mppo_rnd`,
  DQN and the declared `rppo` on `rq_f1`, and `mppo` and the declared `rppo` on
  both TechTree settings, 16,000 each; a first `rppo` run on `rq_f1` with the
  library default architecture, stopped by hand after 10,240 recorded
  interactions (at about 10 per second; up to 511 further interactions may have
  run unflushed); a first `rppo` run on `tt_u_k3`, 1,024 interactions, stopped
  by the ledger's file-sharing error on Windows, which led to the retrying
  ledger;
- profiling: 13,312 interactions in 5 short runs of the training loop
  (RecurrentPPO with the default and two reduced architectures, MaskablePPO;
  one MaskablePPO profile with the default 2,048 steps per environment ran
  into its cap of 4,096 interactions, as designed);
- evaluation of the checkpoints at 8,000 and 16,000 interactions: 15,920
  interactions; agents: 4,447;
- the benchmark path (`bench_dev.yaml`: one Crafter world, conversion seed 0,
  16 episodes; the `rq_f1` development checkpoints at 16,000 interactions and
  the four agents): 1,933 interactions in 9 runs.

Outcomes, on one test seed with two RecipeQuest worlds or one TechTree world:
no learner solved any task at 8,000 or 16,000 interactions (every episode ran
to its deadline in deterministic mode); the repository's agents behaved as in
their studies (`focused_sample` late cost ratio 1.58 and `maximal` 1.97 over two
worlds, `random` 3.27; `pooled`, `promote` and the references at 7 actions per
late episode, `random` 91-97). On the Crafter world no learner solved a
task; `focused_sample` 1.50, `maximal` 1.66 and `random` 2.02 (late success
0.25) in the late half. These numbers size the study; they are not results.

## Results

None: the protocol is a draft.

## Verdict and limits

None yet.
