# Study XS: does a competent open-source learner change study X's picture?

Status: protocol frozen on 2026-10-23 by the commit that freezes this file, before any
measurement run; the measurement ran from a clean worktree of that commit (`XS_SNAPSHOT`, see
Budget and runtime) on 2026-10-26/27. Results below. The development checks listed under
Implementation used only the `dev`
allocation of `runs/XS-ledger.json` and namespaces ending in `_dev`; they are calibration, not
results.

## Question and claim

Question: study X compared the repository's agents with library learners that stayed at random
level. If the same library learner is given the conditions that X's verdict named as limiting
(a budget with room to explore, a denser reward and longer training), does it become competent,
and does the sampling walker `focused_sample` still act more cheaply than it?

The learner is sb3-contrib's MaskablePPO with study X's tuned settings for RecipeQuest family F1
(no new tuning, so the comparison isolates the three factors). It trains in a 2 x 2 design,
budget slack {2 (X's), 12} x reward {sparse (X's), potential-based shaping}, for up to 2,500,000
interactions of fresh training worlds with checkpoints at 300,000, 1,000,000 and 2,500,000, and
is evaluated frozen on fresh test worlds that the repository's agents also play, task for task,
at the same slack.

The claim under test (primary, XS1) is competence: the best of the four learner conditions,
chosen on validation worlds only, acts more cheaply than the `random` agent at its own slack over
the late half of each test world's stream (episodes 9-16). The key secondary (XS2, the analogue of
X's primary X1) asks whether `focused_sample` is still cheaper than that learner at the same
slack. The factor contrasts measure each factor's effect.

## Relation to earlier work

- Study X (docs/studies/X_open_source_baselines.md) found every learner at random level: the
  best RecipeQuest learner (MaskablePPO, tuned) reached a late cost ratio of 2.953 against 2.975
  for `random` and 1.517 for `focused_sample` (X1 holds, X3 does not), no learner improved within
  a world (X4), the masked learners reached 11-16% training success, and the unmasked ones
  received no reward at all. Its verdict names the sparse reward and a budget two steps above the
  brute-force bound as the likely limits. XS keeps X's learner, encodings, information boundary,
  tuning record, analysis code and statistics, and changes only those factors and the training
  length. Its slack-2, sparse condition at 1,000,000 interactions is X's `mppo` arm on fresh
  training and test worlds.
- Study R2 (docs/studies/R2_boundary.md) introduced the slack levels 6 and 12 of RecipeQuest
  (`train.tasks.budget_slack` in its configs, `setting.tasks.budget_slack` here; the budget is the
  public brute-force bound plus the slack, and levels of one family share worlds and tasks). At
  failure probability 0 `maximal` and the privileged reference play identical episodes at every
  slack; `focused_sample` reads the budget, so its episodes can differ.
- docs/PRIOR_ART.md, Section 8: Alchemy (C1, Section 8.3) resamples a latent causal structure in
  every episode and reports no appreciable structure learning by the deep RL agents it tested;
  exact sequences without an intermediate signal are a combination lock (C10). Potential-based
  shaping is the standard remedy for a sparse reward that keeps the task's optimal policies.
- Shaping (docs/PRIOR_ART.md, Section 9, entries S1 and S2): A. Y. Ng, D. Harada and S.
  Russell, Policy invariance under reward transformations: theory and application to reward
  shaping, ICML 1999, pp. 278-287: adding `F(s, s') = gamma * Phi(s') - Phi(s)` for any function
  `Phi` of the state leaves the optimal policies unchanged (and this form is the only one with
  that guarantee without further assumptions). M. Grześ, Reward shaping in episodic reinforcement
  learning, AAMAS 2017: in an episodic task the potential of every terminal state must be 0 for
  that guarantee. Both checked on 2026-10-22 (the ICML proceedings entry and the author's page;
  the AAMAS paper and slides on the author's page).

## Protocol (frozen)

### Setting

Environment: RecipeQuest F1 as in X (`rq_f1`: hidden prerequisites, pools of 6, goals at depth
1-3, intermediate items unobserved, no distractor base facts, failure probability 0).

| Member (config) | Slack | Training | Validation (selection) | Test (seeds x worlds) | Late half |
|---|---|---|---|---|---|
| `s2` (`xs_rq_s2.yaml`) | 2 | `xs_rq_train` | `xs_rq_val`: 2 seeds x 10 | `xs_rq_test`: 5 x 10 | 9-16 |
| `s12` (`xs_rq_s12.yaml`) | 12 | `xs_rq_train` | `xs_rq_val`: 2 x 10 | `xs_rq_test`: 5 x 10 | 9-16 |

Every world plays 16 episodes; the test stream has 50 worlds, the validation stream 20.
The two members share namespaces, worlds, tasks, dynamics seeds and training worlds; every task
budget is exactly 10 steps larger at slack 12 (tested). Every namespace is fresh (used by no
earlier study or development run; tested against every other config). Stream layouts are X's:
test seed s plays base seed 10000 + 1000 s in pool mode (world-major order); training worlds of
training seed s and environment k are worlds k, k + 8, ... of `xs_rq_train` (base seed 30000 +
1000 s), never validation or test worlds. Every learner of a training seed starts from the same
library seed (setting name `rq_f1` in both members), so the four conditions share their initial
networks and training worlds.

TechTree (X's cells `tt_u_k3` and `tt_l2_d2`) is not measured: the RecipeQuest design fills the
time budget (Budget and runtime). The TechTree potential is implemented and tested; adding those
settings, or MaskablePPO with RND, needs a dated amendment before their runs.

### Learner and conditions

| Learner id | Member | Slack | Reward |
|---|---|---|---|
| `s2_sparse` | `s2` | 2 | environment reward (1 on successful submission, else 0), as in X |
| `s2_shaped` | `s2` | 2 | the same plus potential-based shaping, potential `rq_goal_closure` |
| `s12_sparse` | `s12` | 12 | environment reward |
| `s12_shaped` | `s12` | 12 | environment reward plus shaping |

Every learner is sb3-contrib `MaskablePPO` (`MlpPolicy`) with X's tuned settings for `rq_f1`: the
choice of X's tuning group `mppo` (`mppo.wide`: `n_steps` 256 per environment, `ent_coef` 0.01,
`net_arch` [256, 256], other parameters the library defaults, discount 0.99) in the
deterministic action mode. The configs carry X's frozen tuning record
(`artifacts/baselines/x-rq-tune/tuning_summary.json`, sha256 `dd20417a...`) with
`run.tuning_groups`; before every measurement run the CLI checks the digest, recomputes X's choice
from the recorded scores and refuses unless every learner has exactly those settings and mode
(`tuning.verify_tuning`; shaping is the only free key). The digest `dd20417a...` is that of the
working copy inherited from X, which this Windows checkout writes with CRLF line endings
(`core.autocrlf=true`); the committed LF bytes hash to `06b6d1f1...`. So the check passes only in a
Windows checkout with `core.autocrlf=true`, and a clone with LF line endings (on Linux, for
example) refuses the measurement configs; the mechanism is X's and is unchanged. As in X: 8
environments per run in one process, CPU, one torch thread per process, task mode (one learner
episode per task), masks, no memory across episodes.

### Shaping (`baselines/shaping.py`)

Training rewards of a shaped learner are `r + gamma * Phi(s') - Phi(s)` with `gamma` the learner's
discount (0.99, tested against the libraries' defaults) and `Phi = 0` at the end of every learner
episode (success or deadline), so the shaped discounted return of an episode is the environment's
return minus `Phi(s_0)` (tested; 0 in F1, where nothing is held at the start). Evaluation never
shapes. The potential uses only what the wrapper already shows the learner:

- `rq_goal_closure`: `Phi = 0.5 x (share of the candidate base facts of the goal's public closure
  that are observed held) + 0.5 x (goal observed held)`. The closure follows the public item inputs
  of the recipes producing the goal and, recursively, their inputs; its candidate base facts are
  the union of those recipes' public candidate pools (never the hidden requirements); a
  distractor recipe outside the closure does not count. Intermediate items are unobserved under
  goal-only observation and never enter. The potential therefore rewards gathering every
  candidate of the closure (the brute-force plan) and crafting the goal; it says nothing about
  which candidates are required.
- `tt_levels` (TechTree, not measured here): `Phi = 0.5 x (sum of the public levels of the
  concepts held) / (sum of the levels of all concepts) + 0.5 x (goal held)`.

Tests: twin RecipeQuest tasks whose hidden requirements differ give identical shaping terms under
any history that never crafts the goal (the only craft whose outcome is observed), and TechTree
twins with different hidden sequences give identical shaping; the closure's bases equal the pools
of the evaluator's closure; the shaping module names no hidden field and imports only the public
schema modules; shaped returns telescope to `-Phi(s_0)` in task and trial mode; the environment's
reward is unchanged inside the shaped reward.

### Training and evaluation

- Training: each learner, training seeds 0 and 1, 2,500,000 interactions; the policies acting at
  the 300,000th, 1,000,000th and 2,500,000th interaction are saved and evaluated frozen on the
  whole test stream of their member (their slack); the 2,500,000 checkpoint also on the validation
  stream.
- The repository's agents on the test streams of both members, task for task, as in X:
  `focused_sample`, `maximal`, `reference` (privileged) and `random` (uniform over the task's
  gather, activate and craft actions, submitting once the goal is observed held). `random` also
  plays the validation stream of each member (`validation_agents`), for the selection.
- Arms of the combined analysis (`configs/baselines/xs_rq.yaml`, `baselines/xs.py`): learner ids;
  agents as `<agent>.<member>` (`random.s12`); `<learner>~random`, per world, the learner's
  metrics minus those of `random` at the learner's slack; `<arm>#<budget>`, an arm at another
  checkpoint; in tests `best_learner` and `<agent>.best` (the agent at the best learner's slack).

### Endpoints

Those of X, per test world (the unit) and arm: late-half cost ratio (primitive steps over optimal
steps summed over episodes 9-16; a failed episode counts every step it used, its whole budget),
early and overall cost ratios, late minus early cost ratio (within-world improvement), actions per
episode, success rates, actions to the first success. A learner's value in a world is the mean
over its two training seeds. Descriptive: learning curves over the checkpoints; training-task
success and invalid choices per 10,000 training interactions (MaskablePPO never chooses a masked
action, so the latter is 0 by construction and is reported as a check) and the shaping received;
per-training-seed values.

Cost ratios are comparable only at one slack: at slack 12 a failed episode costs 10 more steps.
Every contrast across slack levels is therefore taken against `random` at each arm's own slack.

### Selection of the best learner (validation only)

At 2,500,000 interactions, the learner with the lowest mean over the 20 validation worlds of its
late cost ratio minus that of `random` at its own slack on the same world (each world's learner
value the mean over the two training seeds). Ties go to the earlier learner (`s2` before `s12`,
sparse before shaped). The selection reads no test row. Unless every planned job of both members
has a completed run and every learner has a finite score, there is no best learner and every test
is "incomplete".

### Tests

Paired by test world (50 worlds), at 2,500,000 interactions unless stated; each test has its own
criterion, with no multiplicity correction. "Below" means the upper end of the 95% interval is
below 0, "above" that the lower end is above 0. Metric: late cost ratio unless stated.

| Test | Role | Contrast | Holds when |
|---|---|---|---|
| XS1 | primary | best learner - `random` (its slack) | below |
| XS1s | secondary | best learner - `random` (its slack), late success | above |
| XS2 | secondary (X1 analogue) | `focused_sample` (its slack) - best learner | below |
| XS3 | secondary | best learner - `maximal` (its slack) | below (beats the brute-force plan) |
| XS4 | secondary | best learner, late minus early cost ratio | below (within-world improvement) |
| XSF1 | factor | `s2_shaped` - `s2_sparse` | below (shaping helps at slack 2) |
| XSF2 | factor | `s12_shaped` - `s12_sparse` | below (shaping helps at slack 12) |
| XSF3 | factor | `s12_sparse~random` - `s2_sparse~random` | below (slack helps, sparse) |
| XSF4 | factor | `s12_shaped~random` - `s2_shaped~random` | below (slack helps, shaped) |
| XSF5 | factor | best learner at 2,500,000 - at 300,000 | below |
| XSF6 | factor | best learner at 2,500,000 - at 1,000,000 | below (training beyond X's budget helps) |

Descriptive (single arms, same criterion): each condition minus `random` at its slack (XSD1-4) and
`focused_sample`'s within-world improvement at the best learner's slack (XSD5); learning curves of
every condition against the agents; per-training-seed values of every contrast, reported with each
test (`by_training_seed`): the seeds of the learner `a` when `b` is an agent or absent
(`xs.seed_contrasts`; XS1, XS1s, XS3, XS4), the seeds of `b` when `b` is a learner, as in X; and
each condition's late cost ratio per seed (`per_training_seed` of the summary).

### Predictions (written before any measurement run)

Written after the development profile listed under Implementation (32,768 training interactions
per run, two test worlds), whose outcomes say nothing about the measurement budgets.

- XS1 holds. The best learner is `s12_shaped`; its late cost ratio lies at least 0.3 below that
  of `random` at slack 12, with late success between 0.5 and 0.9 (`random` at slack 12: about
  0.06, its value in the development profile). A shaped learner learns to gather the closure's
  candidates and craft in order; the extra 10 steps let it repair a wrong crafting order.
- XS1s holds. XS2 holds by a wide margin (estimate below -0.5): `focused_sample` uses the
  evidence of earlier episodes of a world, which a memoryless policy cannot, so the learner's
  best strategy is the brute-force plan. XS3 does not hold (no learner beats `maximal`). XS4 does
  not hold: the learners' late minus early cost ratio is within 0.05 of 0.
- XSF1 and XSF2 hold (shaping helps at both slacks). XSF3 and XSF4 hold (slack helps relative to
  `random`; more of the training tasks end in a reward). XSF5 holds; XSF6 does not (most of the
  gain comes before 1,000,000 interactions).
- `s2_sparse` (X's configuration) is within 0.05 of `random` at slack 2 at every checkpoint, as in
  X; `s2_shaped` beats `random` at slack 2 but stays above `maximal`.

### Fairness rules

X's rules 1-5 apply unchanged (identical public information, identical test tasks, declared
training budgets charged to the ledger and checked against the library's count, library defaults
except declared deviations and X's tuning, the best learner chosen on validation worlds only).
In addition: the potential reads only fields the learner already observes (tested); the agents
play at the learner's slack in every contrast; the four conditions share training worlds and
initial networks; no XS tuning.

### Statistics

X's: the unit is a test world; paired per-world differences; percentile world-cluster bootstrap
(`evaluation.stats.cluster_bootstrap_mean`, 10,000 resamples, seed 20261008, 95%) and an exact
two-sided sign test; training seeds are averaged, not resampled, and every test reports its
contrast for each training seed (see Tests). A test with fewer paired worlds than the 50 test
worlds is "incomplete".

### Budget and runtime

Ledger `runs/XS-ledger.json`, with the declaration shared by every `xs_*` member config:
allocations `dev` 600,000 and `xs_main` 40,100,000 adaptive; adaptive cap 40,700,000; reporting
cap 1,000,000. Training is `adaptive`, frozen evaluation `reporting`, `focused_sample` `adaptive`,
the other agents `reporting`; a run's cap is its training budget or the sum of the task budgets of
the streams it plays, so no cap binds. Dry runs (`run --dry-run`, worst cases):

| Member | Runs (train / eval / agent) | Training | Learner evaluation | Agents | Adaptive | Reporting |
|---|---|---|---|---|---|---|
| `s2` | 4 / 16 / 22 | 10,000,000 | 209,064 | 67,657 | 10,015,391 | 261,330 |
| `s12` | 4 / 16 / 22 | 10,000,000 | 317,864 | 102,857 | 10,023,391 | 397,330 |
| Measurement | 8 / 32 / 44 | 20,000,000 | 526,928 | 170,514 | 20,038,782 | 658,660 |

Margin for reruns. A run's meter fixes its cap when the run starts, and the ledger does not refuse
a charge beyond an allocation, so runs restarted at once after a failure could together overshoot
an allocation that only fits the plan. `xs_main` therefore covers the planned adaptive
interactions (20,038,782), a full rerun of every training job after a simultaneous failure
(20,000,000) and a rerun of the adaptive agent runs (38,782): 40,077,564, rounded up to
40,100,000; the adaptive cap is the sum of both allocations (600,000 + 40,100,000). The reporting
cap covers the planned 658,660, the 24,197 used in development and 317,143 to spare, more than
the 8 evaluation or agent runs that can be in progress at once (at most 8 x 23,391 = 187,128). The
development runs created the ledger with an earlier declaration (`xs_main` 26,000,000, adaptive
cap 26,600,000: the plan for a top budget of 3,000,000, with no rerun margin); its caps are raised
to the values above before the freeze and the change is logged in the ledger
(`caps_raised_before_measurement`).

Budget check (stopping rule). Before any launch or resume the driver runs `check-budget` on the
study file (`xs.budget_check`): it continues only if what the ledger has left for `xs_main` (the
allocation and the adaptive cap) and the reporting cap cover the caps of every job of both members
that has no completed run, training jobs first of all; otherwise it stops. Further budget comes only
through a dated amendment.

Runtime. Sizing uses X's measurement of this exact learner and setting on this machine
(MaskablePPO `mppo.wide` on `rq_f1`, 8 training processes at once: 148-149 training interactions
per second per process over 1,000,000 interactions, 6,718-6,746 s), not the single-process rate
(419 per second with X's default network). At that rate 2,500,000 interactions take 4.7 h per
process; the 8 training jobs run at once (both members, 4 workers each), and frozen evaluation
(about 920 interactions per second in X) and the agents add about 0.2 h: about 5 h in all, if
nothing else loads the machine. A top budget of 3,000,000 would need about 5.8 h with no margin.
The development profile (below) ran while other jobs kept the machine at 100% load and measured
the relative costs: shaping and slack add no measurable cost per interaction. Under such a load
the measurement runs proportionally longer (3.4 times slower training in the profile, which would
take the measurement to about 16 h); it stays correct and resume-safe. The estimate therefore
assumes that nothing else runs on the machine during the measurement.

Commands (after the freeze). One driver runs both members at once, then the analyses; it stops
at the first failure, logs to `runs/xs_overnight.log` and is resume-safe. The frozen copy of the
driver is run with the repository root as working directory (it holds `runs/` and `artifacts/`):

```bash
# from the repository root; XS_SNAPSHOT: a clean worktree of the frozen commit (code and configs)
git worktree add ../snapshots/XS-<commit> <commit>
export XS_SNAPSHOT=../snapshots/XS-<commit>
bash "$XS_SNAPSHOT/scripts/run_xs_overnight.sh"
```

`XS_SNAPSHOT` is exported on its own line: in `XS_SNAPSHOT=... bash "$XS_SNAPSHOT/..."` the shell
would expand `$XS_SNAPSHOT` before the assignment takes effect. The driver refuses (exit status 2)
when `XS_SNAPSHOT` is unset or `git -C "$XS_SNAPSHOT" status --porcelain` reports any change, and
logs the snapshot's commit (`git rev-parse HEAD`); the code and configs come only from the snapshot
(`--dev` runs the development members with this checkout's code instead, never as a fallback).
Its steps: remove a stale ledger lock; the budget check (exit status 3 when it fails);
`run --allow-measurement --resume --workers 4` for `xs_rq_s2.yaml` and `xs_rq_s12.yaml` at once;
`analyze` each member; `analyze` the study file `xs_rq.yaml` (selection and tests,
`artifacts/baselines/xs-rq/`).

### Stopping rules and reruns

X's rules apply: every job runs once to the end of its budget or stream; a job with a completed
run is skipped; a measurement config refuses to start when any of its jobs has a completed run
unless `--resume`; an incomplete run is repeated as a new run with the same job hash and code (runs
are deterministic given the code, the hash and the seeds) and both are listed; a completed run is
never replaced. An interruption can lose the work of up to 8 training jobs in progress (each
restarts from zero). Every launch and resume passes the budget check first (Budget and runtime);
when the ledger no longer covers the missing jobs the measurement stops, and further budget comes
only through a dated amendment. The analysis uses only complete data. No interim analysis; no
change of learners, conditions, potentials, budgets, endpoints or criteria after any measurement
row exists, other than dated amendments.

### If the primary fails

The estimate, interval, per-world differences, per-seed values and learning curves are reported,
and the conclusion is that MaskablePPO with X's tuning stays at random level on F1 even with ten
extra steps of budget, a potential-based shaping signal from public information and 2,500,000
training interactions; X's verdict (a comparison with learners at random level) then stands for
this wider range of conditions.

### Limits stated in advance

- One learner (MaskablePPO with X's tuning; no new tuning for the new conditions), one task family,
  two training seeds, one potential with fixed weights; other potentials could guide more (and
  could move part of the repository's inference into the reward).
- Training-seed variability is not in the intervals.
- The shaping rewards the brute-force plan (gather every candidate, craft, submit); it cannot tell
  a memoryless learner which candidates are required, so it cannot make a feedforward learner use
  the evidence of earlier episodes.
- At slack 12 cost ratios are higher for every arm that fails; contrasts across slack levels are
  taken against `random`, which is one choice of reference.
- TechTree and the benchmark sets are not measured.

## Implementation

- `src/hypergraph_agent/baselines/shaping.py`: the potentials (`rq_goal_closure`, `tt_levels`),
  the shaper (`gamma * Phi(s') - Phi(s)`, 0 at the end of a learner episode) and the discount;
  public objects only.
- `wrappers.py`: optional `shaping` (default None, X's behaviour unchanged); the shaping term is
  also reported as `info["shaping"]` and summed in the training log.
- `config.py`: learner key `shaping` (normalized; hashed only when set, so every X hash is
  unchanged, tested against X's completed runs), `validation_agents`, `run.tuning_groups`;
  `tuning.py`: verification of an inherited tuning record; `runner.py`: one shaper per training
  environment, agents on the validation stream, the caps of the jobs without a completed run
  (`missing_caps`).
- `xs.py`: the study file (`xs:` with members and analysis), relabelled agent arms, derived arms
  (`~random`, `#budget`), the selection against `random` at each learner's slack, the tests (X's
  `run_test`) with per-training-seed contrasts of the learner side (`seed_contrasts`), learning
  and training curves, report and figure, and the budget check (`budget_check`); `__main__.py`:
  `analyze` takes a study file, `check-budget` exits with status 1 when the ledger does not cover
  the missing jobs.
- Configs: `configs/baselines/xs_rq_s2.yaml`, `xs_rq_s12.yaml` (measurement members),
  `xs_rq.yaml` (study file), `xs_rq_s2_dev.yaml`, `xs_rq_s12_dev.yaml`, `xs_rq_dev.yaml`
  (development). `tests/test_baselines.py` now checks only X's configs (`xs_*` files have their
  own ledger and are checked in `tests/test_baselines_xs.py`).
- `scripts/run_xs_overnight.sh`: the unattended driver (requires a clean `XS_SNAPSHOT`, logs its
  commit, runs the budget check before any launch or resume; `--dev` for the development members).
- Tests (`tests/test_baselines_xs.py`): closure bases against the evaluator's closure, bounded
  potentials, hidden-structure twins for both environments, a source scan of the shaping module,
  telescoping in task and trial mode and in TechTree, unshaped wrappers unchanged, the libraries'
  discount, X's hashes unchanged, configs, one ledger whose caps cover every planned job and a
  full rerun of every training job, fresh namespaces, the inherited tuning record (and its refusal
  of a changed learner), slack members sharing tasks with budgets 10 apart, the selection against
  `random` at each learner's slack and its refusal of partial data, per-training-seed contrasts,
  member consistency, an end-to-end run of both members with shaped training and the combined
  report, the budget check (missing jobs only, refusal, read only), the driver's refusal without a
  clean snapshot, line lengths.
- Study X unchanged: after these edits the frozen X analyses (`rq_f1_main`, `tt_u_k3_main`,
  `tt_l2_d2_main` of the X snapshot) were recomputed with the current code and compared with
  `artifacts/baselines/x-*-main/summary.json` (Development record).

### Development record (calibration, not results)

Packages unchanged from X (stable-baselines3 2.9.0, sb3-contrib 2.9.0, torch 2.14.1+cpu, Python
3.12.13). Development used the `dev` allocation only: 262,928 adaptive and 24,197 reporting
interactions (287,125 in all; ledger, exact) on 2026-10-22, in 49 registered runs on the
namespaces `xs_rq_train_dev`, `xs_rq_val_dev` and `xs_rq_test_dev` (configs `xs_rq_s2_dev.yaml`,
`xs_rq_s12_dev.yaml`, `xs_rq_dev.yaml`):

- training: the four learners of the measurement, training seeds 0 and 1, 32,768 interactions
  each (262,144), all eight at once (both members, 4 workers each), as in the measurement, while
  about eight processes of other jobs kept the CPU at 100%: 43.3-44.0 training interactions per
  second per process (745-757 s per run), the same for shaped and sparse learners and at both
  slacks. X measured 148-149 per second for this learner with 8 processes of its own, so the
  external load slowed training about 3.4 times;
- evaluation of the checkpoints at 8,192 and 32,768 interactions on one test seed of 2 worlds x 16
  episodes, and at 32,768 on one validation seed of 2 worlds; the four agents on the test stream
  and `random` on the validation stream of each member; the combined analysis of `xs_rq_dev.yaml`
  (selection, derived arms, every test, curves and report) ran end to end;
- two process-pool breakdowns. After every training run had completed, the pools of both members
  lost a worker at the same moment during the validation evaluations; a second attempt lost one
  again in member `s2` (member `s12` completed). Each time the runner logged the failed jobs,
  marked the interrupted runs incomplete and exited non-zero; the missing jobs then completed when
  run again, once in a single process and once in a pool of 4 with nothing else started from the
  same terminal session. Both breakdowns happened as another Python process started from that
  session ended; memory was ample (9.5 of 32 GB free) and no crash was logged. The cause was not
  identified; the measurement should run from its own terminal, and the driver resumes after any
  interruption.

Outcomes (two test worlds, 32,768 interactions): every learner was within 0.04 of `random` at its
slack in the late half, and only `s12_shaped` solved a late test task (late success 0.03).
`random` had a late cost ratio of 2.93 at slack 2 and 4.29 at slack 12 (late success 0.06 at
both), `maximal` 2.00 and `focused_sample` 1.42 at both slacks. Training-task success between
8,192 and 32,768 interactions was 1.3-1.6% at slack 2 and 7.0-8.6% at slack 12; MaskablePPO chose
no invalid action. These numbers size the study; they are not results.

Study X unchanged: `summary.json` and `summary.md` of `x-rq-main`, `x-ttu-main` and `x-ttl-main`,
recomputed with the current code from the X snapshot's configs (commit 11a8c45 with its tuning
records), are byte-identical to the committed artifacts; every planned job hash of X's main,
tuning and benchmark configs checked (`rq_f1_main`, `tt_u_k3_main`, `tt_l2_d2_main`, `tune_rq_f1`,
`bench_crafter`) still maps to its completed run.

## Amendments

None.

## Results

All 84 planned jobs (8 training, 32 evaluation, 44 agent) executed the freeze commit 97ff3f8 from a
clean worktree (2026-10-26 20:00 to 2026-10-27 02:30) and completed; none was rerun, and every
training job reached its checkpoints at exactly 300,000, 1,000,000 and 2,500,000 interactions. The
measurement used 20,017,601 adaptive interactions (worst case 20,038,782) and 569,231 reporting
(658,660); development usage is unchanged and the ledger holds only the pre-freeze amendment. The
frozen analyses wrote `artifacts/baselines/xs-rq-s2/`, `xs-rq-s12/` and `xs-rq/` (the study
analysis). An independent recomputation from the raw rows reproduced every estimate, interval, sign
test and per-seed value exactly, and every analysis output regenerates byte for byte. The member
summaries reuse X's report format: their title reads "Study X" and their "best learner" is the
member's own, not the study's selection, which is in `xs-rq/`.

Selection (validation worlds; learner minus `random` at its slack): `s2_sparse` -0.00609,
`s2_shaped` -0.0196, `s12_sparse` -0.376, `s12_shaped` -0.241. The best learner is `s12_sparse`.

Test worlds (50), learners at 2,500,000 interactions; censored share in parentheses:

| Arm | Slack | Late cost ratio | Late success | Actions to the first success |
|---|---|---|---|---|
| `s2_sparse` | 2 | 2.96 | 0.125 | 147 (26%) |
| `s2_shaped` | 2 | 2.97 | 0.119 | 159 (31%) |
| `s12_sparse` | 12 | 4.01 | 0.481 | 56.5 (0%) |
| `s12_shaped` | 12 | 4.06 | 0.436 | 67.4 (1%) |
| `random` | 2 / 12 | 2.98 / 4.34 | 0.080 / 0.203 | 189 (32%) / 129 (6%) |
| `focused_sample` | 2 / 12 | 1.43 / 1.42 | 0.995 / 1.00 | 16.2 / 14.3 |
| `maximal` | 2 / 12 | 1.91 / 1.91 | 1.00 / 1.00 | 11.7 / 11.7 |

A late cost ratio is comparable only at one slack: if every late episode failed it would be 3.05 at
slack 2 and 4.65 at slack 12.

Tests (95% intervals; sign test as worlds lower / higher / equal):

| Test | Contrast | Estimate [95% CI] | Sign test | Holds |
|---|---|---|---|---|
| XS1 | `s12_sparse` - `random` (slack 12) | -0.332 [-0.432, -0.231] | 45/5/0, p 4.2e-9 | yes |
| XS1s | the same, late success | +0.279 [0.223, 0.335] | 4/45/1, p 8.2e-10 | yes |
| XS2 | `focused_sample` - `s12_sparse` | -2.59 [-2.76, -2.42] | 50/0/0 | yes |
| XS3 | `s12_sparse` - `maximal` | +2.10 [1.98, 2.22] | 0/50/0 | no |
| XS4 | `s12_sparse`, late - early | +0.0175 [-0.0991, 0.129] | 23/27/0, p 0.67 | no |
| XSF1 | `s2_shaped` - `s2_sparse` | +0.0120 [-0.0198, 0.0529] | 21/22/7, p 1.0 | no |
| XSF2 | `s12_shaped` - `s12_sparse` | +0.0538 [-0.0229, 0.126] | 17/33/0, p 0.033 | no |
| XSF3 | `s12_sparse~random` - `s2_sparse~random` | -0.311 [-0.420, -0.199] | 42/8/0 | yes |
| XSF4 | `s12_shaped~random` - `s2_shaped~random` | -0.269 [-0.379, -0.164] | 40/9/1 | yes |
| XSF5 | `s12_sparse`, 2,500,000 - 300,000 | -0.262 [-0.352, -0.176] | 41/9/0 | yes |
| XSF6 | `s12_sparse`, 2,500,000 - 1,000,000 | -0.0676 [-0.137, 0.0056] | 33/17/0, p 0.033 | no |

- Primary: XS1 holds in both training seeds (-0.297 and -0.367).
- Descriptive, each condition minus `random` at its slack: `s2_sparse` -0.0214 [-0.0619, 0.0162],
  `s2_shaped` -0.0094 [-0.0417, 0.0251], `s12_shaped` -0.278 [-0.377, -0.180]. At slack 2 (X's)
  no condition beats `random` at any checkpoint. `focused_sample` improves within a world (late
  minus early -0.606 [-0.671, -0.543]); no learner does.
- Competence: scoring a failed late episode 0 and an optimal one 1, `s12_sparse` reaches 0.176
  against 0.085 for `random`; it solves 0.929 of depth-1, 0.405 of depth-2 and 0.053 of depth-3
  late goals (`random`: 0.485, 0.098, 0.009).
- Shaping: no benefit at either slack. At slack 12 the sign test points against shaping (17 worlds
  lower, 33 higher) and the shaped learner's late success is 0.045 below the sparse one's
  [-0.088, -0.004]; that direction was not specified in advance and the criterion is the interval,
  so the finding is "no benefit".
- Training length: at slack 12, 74% of the gain from 300,000 to 2,500,000 interactions comes before
  1,000,000. XSF6 misses narrowly (upper end +0.0056; both training seeds improve): a small effect,
  not established.
- XS2 is measured at slack 12; at slack 2 the same contrast is about -1.52. XS's test worlds are
  not X's, so its numbers cannot be paired with X's.

Predictions (written before the measurement): 8 met, 1 met on its point estimate only, 5 not met.
Met: XS1 and XS1s hold; XS2 below -0.5; XS3 and XS4 do not hold; XSF3 and XSF4 hold, with more
training tasks rewarded at slack 12 (0.24-0.46 against 0.056-0.124); XSF5 holds and XSF6 does
not, most of the gain coming before 1,000,000; `s2_sparse` within 0.05 of `random` at every
checkpoint. Point estimate only: the best learner at least 0.3 below `random` (0.332; the interval
reaches -0.231). Not met: the best learner is `s12_sparse`, not `s12_shaped`; late success 0.481,
not 0.5-0.9; `random` at slack 12 reaches 0.203, not about 0.06 (the development profile misjudged
it by a factor of about 3); XSF1 and XSF2 do not hold; `s2_shaped` does not beat `random` at
slack 2.

## Verdict and limits

XS's primary holds: with 12 steps of budget slack instead of X's 2 and 2,500,000 training
interactions, MaskablePPO beats the random agent on fresh test worlds (-0.332 [-0.432, -0.231],
lower in 45 of 50 worlds). It does not change study X's picture. The learner stays far costlier
than the repository's agents at the same slack (late cost ratio 4.01 against 1.42 for
`focused_sample` and 1.91 for the brute-force plan), fails more than half of its late tasks, solves
almost no depth-3 goal and does not improve within a world. The potential-based shaped reward gave
no benefit at either slack, and at X's slack no condition beat the random agent at any checkpoint:
the gain comes from the slack.

Limits:
- One library learner (MaskablePPO with X's tuning), one potential and two slacks. A memoryless
  policy cannot use the evidence of earlier episodes of a world, which is what the repository's
  agents exploit. Study X's recurrent learner was never rewarded at slack 2 and was not retrained
  at slack 12.
- Training success is measured with a stochastic policy and the tests with deterministic actions.
- Actions to the first success are censored in 26-52% of the slack-2 worlds and are lower bounds.
- The predictions followed a development profile that misjudged `random` at slack 12.
