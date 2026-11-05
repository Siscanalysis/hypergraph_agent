# Study HP1: a learned requirement-size prior for the exact posterior

Status: protocol frozen on 2026-11-05 by the commit that freezes this file, before any measurement,
after an independent review (ready to freeze after fixes; its decisions are recorded under
Implementation); the analysis runs from a clean worktree of that commit. This is stage 1
of experiment RB-C in docs/studies/RB_rule_builder_design.md (roadmap item 6i): the simplest rule
over graph objects, a prior over requirement sizes, learned from the agent's own evidence. HP1 is
offline and reads existing logs only (study M's M1 collection and study X2's generated sets, both
read only, as their own run selection chooses them); it has no ledger and interacts with no
environment. Development read only the development logs of studies M and X2 (namespaces
`markov_dev` and `x2_gen_dev`); its outcomes were seen before the predictions below were written and
are reported under Implementation as calibration, not results. No method of this study has been run
on a measurement log: the dry run of the measurement config parsed the evidence files and scored
nothing.

## Question and claim

Question: the exact posterior over complete dependency hypergraphs uses a uniform prior over each
recipe's hypothesis class (the nonempty subsets of a pool of six candidates with at most three
members, 41 hypotheses), which implies requirement sizes 1, 2 and 3 with probabilities 6/41, 15/41
and 20/41 (about 0.15, 0.37 and 0.49), while the generator draws sizes with weights 0.2, 0.5 and
0.3. Does a size prior learned by empirical Bayes from the agent's own evidence, through the exact
likelihood of all of it across the worlds of a stream, make the posterior's inference of hidden
requirements better than the uniform prior, and how close do the learned sizes come to the
generator's weights? The design note's estimator, which learns the sizes from the recipes the
evidence has already pinned down, is the first secondary test.

Claim under test (one primary): with study M's M1 evidence (the sampling walker, 60 worlds), after 4
episodes, the posterior under the size prior learned across the stream from the exact likelihood of
all the evidence (`marginal_stream`) has a lower Brier score than the posterior under the uniform
prior, with study M's four pre-stated outcomes (marginal_stream better, uniform better, equivalent
within 0.005, inconclusive). The claim is falsified by "equivalent" or "uniform better".

The primary was switched before the freeze from the pinned-recipe estimate (`learned_stream`, the
design note's estimator) to this marginal-likelihood estimate, because of a structural diagnostic:
12 of the 13 pinned recipes of development had size 3, which follows from the class bound (three
forced candidates pin a recipe, while a smaller requirement needs every other candidate eliminated
by a success), so the pinned recipes are a sample selected by size. The switch did not follow the
arm's development result: `marginal_stream`'s development Brier score was worse than the uniform
prior's in two of the three goal-only cells at 4 episodes (Implementation). `learned_stream` minus
`uniform` is the first named secondary test (S1), predicted "uniform better". The generator's
weights (`oracle`, privileged: the bound in expectation) are compared throughout.

## Relation to earlier work

- Studies M, R and X2 name the gap this study measures. Study M's limits: "Every agent and the
  posterior use the uniform prior over the 41-hypothesis class, while the generator draws
  requirement sizes with weights 0.2, 0.5 and 0.3"; study R's: "one hypothesis prior (uniform over
  the class) is used throughout"; X2's limits: "Only the posterior and the pairwise weights use the
  class prior". HP1 reads the logs of M's M1 collection (frozen at b443d2c, measured 2026-10-08) and
  of X2's generated sets (frozen at 12372ad, measured 2026-10-28, results committed in e4a9878).
  Their results were public before this protocol: M1's posterior had AUROC 0.737 at 4 episodes and
  0.864 at 16 on its 36 test worlds, and X2's posterior AUROC 0.908 and F1 0.809 on its primary cell
  (items observable, 16 episodes). Neither depends on a size prior other than the uniform one.
- Overhypotheses (docs/PRIOR_ART.md, Section 13, where the records are verified): B2 (Kemp,
  Perfors and Tenenbaum, Developmental Science 10, 2007) learns, across many bags, a distribution
  over what a bag holds, whose counterpart here is HP1's size vector, and B1 (Kemp, Goodman and
  Tenenbaum, Cognitive Science 34, 2010) carries the idea to causal schemata that speed up the
  learning of new causal models. A distribution over requirement sizes, learned across recipes and
  worlds, is the simplest overhypothesis of this testbed; HP2 tests a richer one (schemas over
  families and classes).
- Group testing (docs/PRIOR_ART.md, N6, Sections 8.1 and 12): with items observed, inference for one
  recipe is group testing, and the smallest satisfying set decoder (N6, Section 2.2) rests on
  defectives being rare, a prior on small requirement sets. A3 (OHCAM) keeps a Bayesian posterior
  over action models with a simplicity prior. Study RL (Section 12; frozen 2026-11-04) measures a
  description-length prior (E11) in its soft rule learner. HP1 measures a learned prior in place of
  a declared one.
- Empirical Bayes and hierarchical priors (docs/PRIOR_ART.md, Section 13): B6 (Robbins, "An
  Empirical Bayes Approach to Statistics", Third Berkeley Symposium, 1956) poses the problem: a
  decision recurs with parameters drawn from a fixed but unknown distribution, which is estimated
  from the earlier units, from their parameters when those are known (HP1's pinned-recipe arms) and
  otherwise through the marginal distribution of their observations (the marginal-likelihood arms,
  the primary). B8 (Gelman, Carlin, Stern, Dunson, Vehtari and Rubin, Bayesian Data Analysis, third
  edition, 2013) gives the Dirichlet-multinomial update (Section 3.4, the form of the pinned-recipe
  estimate) and, separately, hierarchical models with hyperpriors (Sections 5.1 to 5.3, worked on a
  beta-binomial example), and it treats estimating the prior from the data, as HP1 does, as an
  approximation to the full hierarchical analysis (p. 104).
- Ignorability: B7 (Rubin, "Inference and missing data", Biometrika 63, 1976; restated in B8,
  Section 8.2) gives sufficient conditions under which likelihood and Bayesian inference may ignore
  the process that selected the data: the data are missing at random, and that process has
  parameters distinct from those of the data model; B8 (Section 8.4) adds that a sequential design
  is ignorable given the earlier outcomes it uses. HP1 argues that its evidence meets both: which
  recipes are attempted, and which outcomes are recorded, depend only on the policies of studies M
  and X2 (fixed functions of the observed history and of their own seeds), on public task draws and
  on the outcomes already observed, never on a hidden requirement except through those outcomes,
  and the policies' parameters are fixed apart from the size distribution. Under these premises,
  stated as an assumption under Limits, the likelihood of all the evidence (the primary arm's) may
  ignore how the evidence was collected, whereas the pinned-recipe arms condition on a subset
  selected by the recipes' own sizes.

docs/PRIOR_ART.md records no comparison, on matched logs of one hidden-prerequisite task, of an exact
posterior under a uniform class prior, a learned size prior and the generator's own size weights;
that is what this study adds. It claims no new method.

## Protocol (frozen)

### Settings

| Source | Logs (read only) | Observation | Evidence policy | Role |
|---|---|---|---|---|
| `m1_focused` | M's M1 collection | goal-only | `focused_sample` | primary |
| `m1_maximal` | the same runs | goal-only | `maximal` | no information (V1) |
| `x2_goal_focused` | X2's set `gen_goal` | goal-only | `focused_sample` | replication |
| `x2_goal_random` | the same set | goal-only | `random_omit` | policy-independent |
| `x2_goal_maximal` | the same set | goal-only | `maximal` | no information (V1) |
| `x2_obs_focused` | X2's set `gen_obs` | items observable | `focused_sample` | factorized posterior |
| `x2_obs_random` | the same set | items observable | `random_omit` | descriptive |
| `x2_obs_maximal` | the same set | items observable | `maximal` | no information (V1) |

M's M1 collection is the runs `markov-m1-*` (namespace `markov_sel`), X2's sets the runs
`x2-main-*` (namespace `x2_gen`); every source has seeds 0-4 with twelve worlds each (60 worlds).
Every source is family F1 (study R's Stage D distribution: pools of six, at most three hidden base
facts, sizes drawn with weights 0.2, 0.5 and 0.3, goals of depth 1-3, no distractor base facts,
deterministic dynamics), 16 episodes per world in world-major order, every seed its own worlds. A
stream is one run: one seed's twelve worlds in play order. The sources are read through the loaders
of the studies that collected them (`evaluation.markov_study.load_evidence`,
`inference_baselines.collection.load_evidence`), which choose one run per (policy, seed) by their
own rerun rules; every record's namespace must equal the source's (`load_source` refuses others).
X2's logs are used when every chosen run completed, the runs of a study come from one (commit,
code) and every source has its 60 worlds; otherwise the X2 secondary tests are None (incomplete).
The dry runs of 2026-10-29 and 2026-11-05 (the second on HEAD 7dacf57 with this working tree)
found every source complete: 5 runs and 60 worlds each, M's runs from b443d2c and X2's from
12372ad, one code identity each, no rerun used and no run ignored.

### Priors (arms)

A size prior gives each size k (1 to 3) a probability pi_k, spread uniformly over the hypotheses of
size k in a recipe's class; the joint prior is the product over recipes.

| Prior | Size probabilities | Learned from | Role |
|---|---|---|---|
| `uniform` | 6/41, 15/41, 20/41 (the class prior) | nothing | reference, today's posterior |
| `oracle` | 0.2, 0.5, 0.3 | the generator (privileged) | bound |
| `learned_world` | Dirichlet posterior mean | pinned recipes of the world's first k logs | within a world |
| `learned_stream` | the same | those and the pinned recipes of the stream's earlier worlds | S1 |
| `marginal_world` | Dirichlet mixture mean | all evidence of the world's first k logs | within a world |
| `marginal_stream` | the same | that and all evidence of the stream's earlier worlds | primary arm |

Hyperprior: Dirichlet(A u) on (pi_1, pi_2, pi_3), u the sizes implied by the uniform class prior and
A = 2 (fixed a priori, as study M's prior strength n0 = 2). Without data every learned prior equals
the uniform prior exactly (tested), so a learned arm can differ from `uniform` only through evidence.

- Pinned recipe: in a group of recipes computed exactly, the uniform posterior leaves exactly one
  hypothesis (every candidate marginal within 1e-9 of 0 or 1). With deterministic dynamics the set of
  consistent hypotheses does not depend on the prior, so neither does being pinned (tested). Pins
  read from sampled groups are no proof (a sampled marginal of 0 or 1 means only that no chain
  visited another hypothesis): they are reported apart and never used (tested). With n_k pinned
  recipes of size k, the learned prior is pi_k = (A u_k + n_k) / (A + n).
- Marginal likelihood: for each group of recipes the evidence couples, the enumeration of the
  posterior also counts the consistent assignments by size profile (c1, c2, c3 recipes of sizes 1, 2
  and 3: a size polynomial), so the likelihood of the evidence is sum over profiles of the count
  times prod_k (pi_k / N_k)^c_k, N_k the number of hypotheses of size k. The product over groups is a
  polynomial in pi; with the Dirichlet hyperprior the posterior of pi is a mixture of Dirichlet
  distributions and its mean is exact (`priors.size.marginal_sizes`; tested against numerical
  integration over the simplex). A pinned recipe is the case of a single term; groups that fall
  back to sampling (below) have no polynomial and are counted.
- Stream priors use every earlier world of the stream at its last checkpoint (16 episodes) and the
  current world's first k episodes; the first world of a stream has no earlier world.

Posterior under each prior (`priors.posterior`): goal-only logs, study M's exact enumeration (hypotheses
holding a candidate eliminated by a success dropped, recipes grouped by the logs they share,
depth-first enumeration pruned by deterministic replay and memoized on the frontier) with every
completion weighted by its prior; with uniform weights it equals M's posterior exactly (tested, and
V3). The enumeration and its pruning do not depend on the weights, so one pass serves several priors
and whether a group is computed exactly (within 2,000,000 log checks) is the same for every prior;
beyond that budget each prior gets weighted Gibbs sampling among consistent nodes (four focused-walk
starts, 40 sweeps after 10 of burn-in, Rao-Blackwellized), every prior from the same seed. Items
observable: the per-recipe updater (`DependencyModel`) with the prior as its opt-in `log_prior`, X2's
form of the posterior; with the uniform prior it is X2's computation exactly (tested). Every random
choice is seeded per (set, policy, seed, world, checkpoint).

### Metrics

For every world, source and checkpoint k in {4, 8, 12, 16}, every prior reads the world's first k
logs and the recipes registered by then; the scored recipes are the hidden recipes attempted in those
logs (as in M1 and X2). Per scored recipe, against its hidden requirement:

- Brier score: the mean over the recipe's candidates of (marginal minus indicator)^2, a proper
  scoring rule for the candidate marginals; lower is better.
- AUROC of the candidates' ranking (ties one half; study M's `auroc`).
- F1 of the inferred set (candidates with marginal at least 0.5, else the top-scoring tie group;
  X2's rule and code).
- Ranked gathering cost (study M's; exploratory).

A world's value is the mean over its scored recipes. Also recorded per world and checkpoint: the
number of pinned recipes and their sizes, the true sizes of the scored recipes (evaluator side, for
the selection diagnostic), every learned size vector, the numbers of exactly computed, sampled and
prior-only recipes and of groups without a polynomial, and the run time.

Why the Brier score is primary and AUROC is not. A size prior is exchangeable within a recipe. If
every failure clause that holds candidate b also holds candidate a, swapping b for a maps the
consistent hypotheses that hold b and not a one to one onto consistent hypotheses of the same size
that hold a and not b, so a ranks at least as high as b under every size prior. Size priors can
therefore reorder only candidates that are incomparable in this sense, so AUROC is nearly
insensitive to them (not immune: incomparable candidates can still swap); in development the
per-world AUROC of all six priors agreed exactly in every world and checkpoint. F1 responds to a
prior (the 0.5 threshold moves) but is not a proper score: in development the oracle's F1 was below
the uniform prior's in some cells. The Brier score of the marginals is the measure under which the
generator's own prior is optimal in expectation (the posterior mean minimizes the expected squared
error when the prior is the truth and the way the data were collected can be ignored), so `oracle`
is the bound in expectation over worlds, not in every sample of worlds.

### Primary contrast, falsification and verdict

Primary: `m1_focused`, 4 episodes, all 60 worlds: per-world Brier score of `marginal_stream` minus
that of `uniform`. Decision (study M's four outcomes): "marginal_stream better" if the 95%
world-cluster bootstrap interval lies entirely below 0; "uniform better" if it lies entirely above 0;
otherwise "equivalent" if the 90% interval lies inside (-0.005, 0.005); otherwise "inconclusive"; a
significant difference takes precedence over equivalence. The claim is falsified by "equivalent" or
"uniform better". The minimum detectable difference, 2.80 x SD / sqrt(60), is reported from the
observed SD of the per-world differences, and a bootstrap clustered by stream (five clusters) is
reported beside it as descriptive.

Fixed episode count: 4, the earliest checkpoint, where the evidence is thinnest, fixed before
development. Development does not support the reasons first given for it, that the prior matters
most there and that the bound shrinks with evidence: the oracle's Brier score was above the uniform
prior's at 4 episodes in all four sources (+0.0024 to +0.0073) and below it at 16 (-0.0015 to
-0.0029). The count stays at 4, because choosing another one now would follow those outcomes.

Margin: 0.005 Brier, about 2.5% of the uniform posterior's Brier score at 4 episodes (0.19 in
development) and three times the expected gap between the generator's weights and the uniform prior
without any evidence (0.0016, from the prior marginals 0.35 and 0.39 against a requirement share of
0.35). Development's SD of the per-world primary difference (0.0066 over two worlds; 0.0070 for the
first primary, `learned_stream`) gives a minimum detectable difference of about 0.0025 for 60
worlds, below the margin.

Power. The largest possible gain, the oracle's, is about 0.0016 without evidence, against a minimum
detectable difference of 0.0025 and a margin of 0.005: even the true prior would come out "better"
only about 40% of the time. "Equivalent" therefore means that size priors are practically
irrelevant at this margin, and the informative outcomes are harm (a learned prior worse than the
uniform one) and the size vectors the estimates recover.

Verdict (the vocabulary of studies X2 and RL): None when the primary is incomplete (a world missing,
a run incomplete, or a stream of its source failed in the analysis); otherwise "invalid" when a
validity test fails (a test that does not apply is None and invalidates nothing); otherwise the
decision.

### Validity tests

Each validity test holds, fails, or is None when it does not apply.

- V1: on brute-force evidence (`m1_maximal`, `x2_goal_maximal`, `x2_obs_maximal`), every checkpoint,
  every prior has a per-world AUROC of exactly 0.5 and every prior gives the same inferred sets
  (equal F1): nothing in those logs depends on the hidden requirement, nothing is eliminated or
  pinned, and an exchangeable prior ties every candidate. None without a brute-force source.
- V2: every pin of an exactly computed group equals the hidden requirement (soundness of the
  posterior and of the pinned test). Pins read from sampled groups, and how many of them are wrong,
  are reported beside it and do not enter the test. None when no recipe is pinned.
- V3: on M's logs the `uniform` arm reproduces study M's published per-world posterior AUROC
  (`artifacts/markov/m1_ranking.json`) within 1e-9 at every world and checkpoint that M computed
  exactly and HP1 computed exactly; None when no such pair exists.

### Secondary tests (each with its own criterion, no multiplicity correction)

All on every world of the source; "below" and "above" mean the 95% interval of the difference lies
entirely below or above 0; "equivalent" means the 90% interval lies inside (-margin, margin); a
decision is the four-way rule (for the Brier score lower is better). A test with a criterion
reports whether it holds (true or false), a decision test reports its decision, and either result is
None when the test is incomplete: its source lacks a world, reads an incomplete run, or lost a
stream in the analysis.

| Test | Source, episodes | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| S1 | `m1_focused`, 4 | `learned_stream` - `uniform` (the first primary) | Brier | decision, margin 0.005 |
| S2 | `m1_focused`, 4 | `oracle` - `uniform` (the bound) | Brier | below |
| S3 | `m1_focused`, 4 | `marginal_stream` - `learned_stream` | Brier | below |
| S4 | `m1_focused`, 4 | `learned_world` - `uniform` | Brier | equivalent, margin 0.005 |
| S5 | `m1_focused`, 4 | `learned_stream` - `uniform` | AUROC | equivalent, margin 0.01 |
| S6 | `m1_focused`, 4 | `learned_stream` - `uniform` | F1 | decision, margin 0.03 |
| S7 | `m1_focused`, 4 | `marginal_stream` - `uniform` | F1 | decision, margin 0.03 |
| S8 | `m1_focused`, 16 | `learned_stream` - `uniform` | Brier | decision, margin 0.005 |
| S9 | `x2_goal_focused`, 4 | `learned_stream` - `uniform` | Brier | decision, margin 0.005 |
| S10 | `x2_obs_focused`, 4 | `learned_stream` - `uniform` | Brier | decision, margin 0.005 |
| S11 | `x2_goal_random`, 4 | `learned_stream` - `uniform` | Brier | decision, margin 0.005 |
| S12 | `x2_goal_focused`, 4 | `marginal_stream` - `uniform` | Brier | below |

Twelve tests without correction: if none of their hypotheses were true and the tests were
independent, the chance that at least one holds anyway would be about 1 - 0.95^12, or 0.46, so a
single holding test is weak evidence on its own; the verdict rests on the primary alone.

Descriptive (no criterion): every prior's table of Brier score, AUROC, F1 and gathering cost at
every checkpoint and source; the mean learned size vectors; the sizes of the pinned recipes against
the true sizes of the scored recipes (the selection diagnostic); counts of pinned, sampled and
prior-only recipes, of pins read from sampled groups (and how many are wrong) and of groups without
a polynomial; the primary clustered by stream; run times.

### Predictions (written after the development checks)

Written on 2026-11-04 and revised on 2026-11-05 with the review (the switch of the primary and the
power statement), after the development checks of 2026-10-29 and their rerun of 2026-11-05 (two
worlds per source, outcomes seen; Implementation) and after the published results of studies M and
X2.

- Primary: "equivalent", estimate between -0.002 and +0.004 Brier; the next likeliest outcome is
  "uniform better". Reason: even the true prior gains about 0.0016 at most, below the minimum
  detectable difference (Power, above), and the marginal estimate is noisier than the truth: it
  follows the sizes of the requirements the logs constrain, which in development helped in some cells
  and hurt in others (+0.0050 [0.0003, 0.0096] here over two worlds, worse than the uniform prior in
  two of the three goal-only cells at 4 episodes).
- S1: "uniform better", estimate between +0.002 and +0.008 (the prediction of the first primary).
  Reason: almost every pinned recipe has size 3 (12 of 13 in development: three forced candidates pin
  a recipe through the class bound, while a smaller requirement needs every other candidate
  eliminated by a success), so the pinned estimate moves toward size 3 (about 0.7 to 0.8 after a
  stream's earlier worlds), away from the generator's 0.3, and the posterior becomes worse calibrated
  than under the uniform prior. In development: +0.0050 [0.000, 0.0099] over two worlds (one of them
  the first of its stream, without stream information).
- S2 does not hold (estimate between -0.002 and +0.002): the bound is about 0.0016 without evidence,
  below the minimum detectable difference, and in development the oracle's Brier score was above the
  uniform prior's at 4 episodes in all four sources (+0.0024 here).
- S3 holds (estimate between -0.006 and -0.001): the estimate from all the evidence, ignorable
  under the premises of B7, is better than the one selected by size.
- S4 holds: a world has almost no pinned recipe after 4 episodes (none in development), so the
  within-world prior equals the uniform prior.
- S5 holds with a difference of 0 up to rounding (development: 0 in every world and checkpoint).
- S6 and S7: "inconclusive" (F1 moves in both directions; development +0.069 [0.000, 0.138]).
- S8: "uniform better" (the pinned estimate is more extreme after 16 episodes; development +0.0127
  on average over two worlds).
- S9, S10 and S11: "uniform better" or "inconclusive"; S12 does not hold.
- V1, V2 and V3 hold.

### Statistics

The unit is the world; worlds are independent across seeds within a source, and M's and X2's worlds
are different worlds. Every contrast is a paired per-world difference; its mean gets a percentile
world-cluster bootstrap interval (10,000 resamples, seed 20261008, `evaluation.stats`), 95% unless
stated, 90% for equivalence; an exact two-sided sign test and the SD are reported with each. Runs
and records are the ones the collecting studies' loaders choose (one run per policy and seed, the
earliest; a run that did not complete replaced by the earliest later completed run with the same
config hash, commit and loaded code). One primary contrast; the secondary tests have their own
criteria. The worlds of a stream share their stream priors, so the 60-world interval is conditional
on the five streams; the primary is also reported with a bootstrap over the five stream means
(descriptive: five clusters cannot support a decision).

### Budget and runtime

No interaction and no ledger: HP1 reads finished logs. Dry run (`--dry-run`: manifests and evidence
files parsed, nothing scored; 2026-10-29, repeated on 2026-11-05 with the same figures, 26 s):

| Source | Runs | Worlds | Checkpoints | World-checkpoints | Commit of the runs |
|---|---|---|---|---|---|
| `m1_focused` | 5 | 60 | 4 | 240 | b443d2c |
| `m1_maximal` | 5 | 60 | 4 | 240 | b443d2c |
| `x2_goal_focused` | 5 | 60 | 4 | 240 | 12372ad |
| `x2_goal_random` | 5 | 60 | 4 | 240 | 12372ad |
| `x2_goal_maximal` | 5 | 60 | 4 | 240 | 12372ad |
| `x2_obs_focused` | 5 | 60 | 4 | 240 | 12372ad |
| `x2_obs_random` | 5 | 60 | 4 | 240 | 12372ad |
| `x2_obs_maximal` | 5 | 60 | 4 | 240 | 12372ad |
| all | 40 streams | 480 | | 1,920 | |

Runtime. Brute-force evidence has no constraint and items-observable evidence is factorized, so
the 720 goal-only world-checkpoints with `focused_sample` and `random_omit` evidence dominate. In
development (a machine shared with other work) the 64 world-checkpoints took 492 s of computation,
268 s of wall-clock time with three worker processes; most took under a second, two took about three
minutes (an M development world at 4 episodes, 179 s; an X2 goal-only world at 12 episodes, 164 s),
and a group beyond the exact budget costs a weighted Gibbs run per prior. Study M's own analysis of
the M1 logs (one pass of the posterior and every other M1 method) took 1,025 s. Estimate for the
measurement: 2 to 6 hours of computation in one process, 1 to 2 hours with four worker processes
(`--workers 4`, one stream per process); it runs at night.

### Stopping and rerun rules

No interim analysis. The analysis runs once, from a clean snapshot of the freeze commit, after the
dry run has confirmed every source complete. It is deterministic (seeded); streams are collected as
they finish and each is cached at once (written whole to a temporary file, then renamed, so a kill
mid-write leaves no partial cache file), and a stream that fails is recorded as a problem of its
source, which leaves that source's tests incomplete (None) without discarding the finished streams.
After an interruption or a failed stream, `--resume` reuses the cached streams of the same code and
config (`--cache runs/hp1_cache`, keyed by the loaded code and the config) and computes the rest. A
fault that needs a code fix means a new freeze and a fresh analysis of every stream; a result
computed before the fix is discarded and the event recorded as an amendment. The logs of studies M
and X2 are never written.

### Commands (from a clean snapshot, after review and freeze)

From the repository root, so that `runs/` (with M's and X2's measurement runs) and `artifacts/` (with
M's ranking summary, read by V3) are the repository's; `S` is a clean worktree of the freeze commit:

```bash
git worktree add ../snapshots/HP1-<hash> <hash>
S=../snapshots/HP1-<hash>
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.evaluation.hybrid_priors hp1 \
    --config $S/configs/hybrid_priors_offline/hp1_main.yaml --runs runs --dry-run
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.evaluation.hybrid_priors hp1 \
    --config $S/configs/hybrid_priors_offline/hp1_main.yaml --runs runs \
    --out artifacts/hybrid_priors --workers 4 --cache runs/hp1_cache --resume
```

The analysis writes `artifacts/hybrid_priors/hp1-main.json` (summary and every per-world row) and
`hp1-main.md`.

### Limits stated in advance

- One family (F1) and its generator; sizes are the only rule learned here (HP2 learns schemas). The
  learned priors assume the reference pool of six, which every F1 recipe has (checked per world).
- The pinned recipes are a selected sample: whether a recipe gets pinned depends on its size. The
  `learned_*` arms implement the design note's estimator as specified; the `marginal_*` arms, which
  use all the evidence instead (ignorable under the premises below), were added after development,
  and the primary was switched to them before the freeze (Question and claim).
- Power: the largest possible gain, the oracle's, is about 0.0016 without evidence, below the
  minimum detectable difference (0.0025) and the margin (0.005); even the true prior would come out
  "better" only about 40% of the time. An "equivalent" primary says that size priors are practically
  irrelevant at this margin, not that the learned prior recovers the truth; harm and the recovered
  size vectors are the informative outcomes.
- Pins come only from groups computed exactly. Dropping the sampled groups is itself a selection by
  complexity: recipes in large or tangled groups never reach the `learned_*` arms (the dropped pins
  are counted).
- The worlds of a stream share their stream priors, so the 60-world interval is conditional on the
  five streams (the bootstrap over streams is descriptive only).
- Empirical Bayes uses the same evidence twice: the logs that estimate a world's size prior are
  also the logs whose posterior it scores.
- The marginal-likelihood arms, the primary's included, ignore how the evidence was collected. That
  is assumed, argued under Relation to earlier work but not tested: B7's sufficient conditions hold
  if the policies and the task draws depend on hidden requirements only through observed outcomes
  and their parameters are distinct from the size distribution. The polynomials of sampled groups
  are missing as well, a selection by complexity that this argument does not cover.
- The hyperprior strength A = 2 and the episode count k = 4 were fixed in advance and never varied.
- The evidence comes from fixed policies of other studies. `focused_sample`'s logs are adapted to the
  uniform prior's class (its nodes favour size 3, which affects what gets pinned); `random_omit`
  (S11) is the policy-independent check.
- Size priors cannot reorder candidates that are comparable by their failure clauses, so AUROC is
  nearly insensitive to them by construction (incomparable candidates can still swap); F1 is not a
  proper score.
- The oracle is privileged, and it is a bound for the Brier score only in expectation over worlds.
- Offline: no acting. Whether a prior changes what a sampling walker does, and at what cost, is
  HP2's question.
- Groups beyond the exact budget are sampled per prior, have no size polynomial and give no pin;
  all three are counted.
- The results of M and X2 on these logs were public before this protocol; HP1's methods have not
  been run on their measurement logs.

## Implementation

- `src/hypergraph_agent/priors/size.py`: size priors (`SizePrior`, `uniform_sizes`), the pinned-recipe
  estimate (`learned_sizes`) and the exact marginal-likelihood estimate from size polynomials
  (`marginal_sizes`, a mixture of Dirichlet distributions computed in log space).
- `src/hypergraph_agent/priors/posterior.py`: the exact posterior under any per-recipe prior
  (`joint_marginals`: study M's enumeration with weights, several priors in one pass, size
  polynomials, weighted Gibbs beyond the budget, and the recipes of exactly computed groups in
  `exact_sigs`; `factorized_marginals`: X2's updater with the prior as `log_prior`), `UniformPrior`
  and `pinned`. The package holds public-information code only (it is one of the learning packages
  that `tests/test_information_boundary.py` checks); the generator's size weights live in the HP1
  configs, on the evaluator side.
- `src/hypergraph_agent/topology/inference.py` (opt-in hooks only): `prior_marginal(...,
  size_weights=None)`, `size_log_prior`, and `DependencyModel(..., log_prior=None)`; with the
  defaults nothing changes (tested, and the frozen-behaviour digests below).
- `src/hypergraph_agent/evaluation/hybrid_priors.py`, command `hp1`: the study's own config schema
  (`validate_hp1`: a development config, named `*-dev`, may read only namespaces ending in `_dev`,
  and no other config may read them; YAML booleans are not priors), the sources through M's and X2's
  loaders with the namespace and provenance checks, one stream per process (`stream_rows`, collected
  as they finish; a failing stream is recorded as a problem of its source), pins from exactly
  computed groups only, the metrics, the decision rule with the metric's orientation, validity and
  secondary tests in the vocabulary of studies X2 and RL, the bootstrap over streams, the report,
  the dry run, and the per-stream cache with resume.
- `configs/hybrid_priors_offline/`: `hp1_dev.yaml` (development logs only) and `hp1_main.yaml` (own
  schema; the directory is listed in `tests/test_hygiene.py`).
- `tests/test_hybrid_priors.py` (shared with study HP2): the weighted posterior equal to brute-force
  enumeration under the uniform and a size prior and to study M's posterior, the size polynomials
  equal to brute-force counts by size profile, the Gibbs fallback near the exact posterior, the
  factorized posterior equal to X2's and its polynomials, the size priors and both estimates (no data
  gives the uniform prior, pinned recipes the conjugate update, a polynomial the mean found by
  numerical integration), pinned recipes equal to the truth and independent of the prior, the
  `DependencyModel` prior hook and `size_log_prior`, the whole pipeline on a tiny in-memory
  collection (shallow worlds, scratch namespace), the decision rule in both orientations, the
  verdict vocabulary (a complete, an incomplete and an invalid result; tests that do not apply),
  pins taken only from exactly computed groups (sampled pins counted, never used), a failing stream
  that leaves its source incomplete while the finished stream is kept and cached, cache files
  written whole or not at all (a write killed midway), and config
  validation (development namespaces only, the measurement sources and the switched primary, YAML
  booleans refused, unknown keys refused); the frozen-behaviour tests and the line hygiene of the new
  files.

Development commands (from the repository root; outputs outside the repository):

```bash
python -m hypergraph_agent.evaluation.hybrid_priors hp1 \
    --config configs/hybrid_priors_offline/hp1_dev.yaml --runs runs --dry-run
python -m hypergraph_agent.evaluation.hybrid_priors hp1 \
    --config configs/hybrid_priors_offline/hp1_dev.yaml --runs runs --out ../scratch/hp1_dev \
    --workers 3 --cache ../scratch/hp1_dev_cache --resume
```

### Development record (calibration, not results)

Logs: study M's development collection (`markov-dev-m1-*`, namespace `markov_dev`: two worlds,
`focused_sample` and `maximal`, 16 episodes) and study X2's development logs of `gen_goal` and
`gen_obs` (namespace `x2_gen_dev`: two worlds per set and policy), read only. Two development
analyses on 2026-10-29: the first with four priors (`uniform`, the generator's weights, `learned_world`,
`learned_stream`; 284 s of wall-clock time with three worker processes, 590 s of computation), the
second with all six (268 s, 492 s), on a machine shared with other work.

Checks: the weighted enumeration with uniform weights equalled study M's posterior on all eight
world-checkpoints of M's development logs (largest difference 0); a size prior with the
uniform-implied sizes equalled the class prior (difference 0); V1 held on every brute-force cell
(AUROC exactly 0.5 and equal inferred sets under every prior) and V2 held (every pinned requirement
was the hidden one).

Values (mean of two worlds; Brier score, lower is better):

| Source @ episodes | uniform | oracle | learned world | learned stream | marginal world | marginal stream |
|---|---|---|---|---|---|---|
| `m1_focused` @ 4 | 0.1923 | 0.1947 | 0.1923 | 0.1972 | 0.1931 | 0.1972 |
| `m1_focused` @ 16 | 0.1095 | 0.1069 | 0.1177 | 0.1222 | 0.1140 | 0.1177 |
| `x2_goal_focused` @ 4 | 0.1072 | 0.1145 | 0.1072 | 0.1072 | 0.1091 | 0.1121 |
| `x2_goal_focused` @ 16 | 0.0977 | 0.0962 | 0.1026 | 0.1026 | 0.0971 | 0.0954 |
| `x2_obs_focused` @ 4 | 0.1141 | 0.1170 | 0.1122 | 0.1114 | 0.1138 | 0.1132 |
| `x2_obs_focused` @ 16 | 0.0689 | 0.0669 | 0.0775 | 0.0773 | 0.0742 | 0.0741 |
| `x2_goal_random` @ 4 | 0.1743 | 0.1787 | 0.1743 | 0.1730 | 0.1734 | 0.1728 |
| `x2_goal_random` @ 16 | 0.1253 | 0.1224 | 0.1300 | 0.1313 | 0.1299 | 0.1316 |
| `m1_maximal` @ 4 to 16 | 0.2438 | 0.2475 | 0.2438 | 0.2438 | 0.2438 | 0.2438 |

- AUROC was identical under all six priors in every world and checkpoint (difference 0).
- F1 moved in both directions: `m1_focused` at 4 episodes 0.456 under `uniform` and 0.525 under
  `learned_stream` and `marginal_stream`; `x2_goal_focused` at 4 episodes 0.847 under `uniform` and
  0.733 under `oracle`.
- Pinned recipes at 16 episodes: 12 of size 3 and 1 of size 2, against true sizes of the scored
  recipes of 28 of size 3, 33 of size 2 and 7 of size 1 (all sources). No world had a pinned recipe
  after 4 episodes. Mean learned sizes at 16 episodes: `learned_stream` (0.061, 0.152, 0.787) on M's
  logs and (0.110, 0.274, 0.616) on X2's goal-only logs; `marginal_stream` (0.079, 0.176, 0.745) and
  (0.087, 0.481, 0.433). M's two development worlds had mostly size-3 requirements (8 of the 14
  scored at 16 episodes), X2's mostly size 2, and the marginal estimate followed each.
- Tests on these worlds (intervals over two worlds are not informative), in the numbering of
  2026-10-29: the first primary (`learned_stream`, now S1) +0.0050 [0.000, 0.0099], "inconclusive",
  SD 0.0070; S1 (the oracle, now S2) +0.0024 [-0.0027, 0.0075]; S2 (`marginal_stream`, now the
  primary) +0.0050 [0.0003, 0.0096]; S5 0 [0, 0]; S6 +0.069 [0.000, 0.138]; S9 0 [0, 0].

Rerun on 2026-11-05 with the frozen code (exact pins only, the switched primary; 218 s of
wall-clock time with three worker processes): every per-world value and every learned size vector
equalled those of 2026-10-29. No development group was sampled, so no pin came from a sampled group
(26 pins over all checkpoints, all exact). The tests in the frozen numbering: primary (`marginal_stream`
minus `uniform`) +0.0050 [0.0003, 0.0096], "uniform better" on two worlds, SD 0.0066; S1
(`learned_stream`) +0.0050 [0.000, 0.0099], "inconclusive", SD 0.0070; S2 (the oracle) +0.0024
[-0.0027, 0.0075], does not hold; S5 holds (0); S6 "inconclusive" (+0.069); S9 "equivalent" (0);
V1 and V2 hold, and V3 is None (no published artifact covers development logs). The bootstrap over
streams has one cluster in development (one seed of M's development collection).

Decisions that came from development:

1. The primary metric is the Brier score, not AUROC (nearly insensitive to size priors by
   construction, as given under Metrics, and identical under every prior in development) and not F1
   (moves both ways, not a proper score). The design note named AUROC and F1 for this stage; both
   remain secondary (S5, S6, S7).
2. The `marginal_*` arms were added after the pinned sizes showed the selection (12 of 13 of size
   3); the pinned arms stay as the design note's estimator.
3. The generator's prior is named `oracle` (YAML reads `true` as a boolean, which the config schema
   now refuses).
4. The unit test of the whole pipeline uses shallow worlds (two levels, goals of depth 1-2) and a
   bound on the exact work: on full F1 worlds one world-checkpoint took minutes.

Decisions of the independent review (2026-11-05), made before the freeze:

5. The primary is `marginal_stream` minus `uniform`; `learned_stream` minus `uniform` becomes S1
   (Question and claim gives the reason, a structural diagnostic, and the arm's development result,
   which did not favour it). The Brier score, k = 4 and A = 2 were signed off.
6. The power statement (Primary contrast; Limits): at this margin "equivalent" means that size
   priors are practically irrelevant.
7. The vocabulary of results follows studies X2 and RL (holds, decision, None, "invalid").
8. Pins come only from exactly computed groups, for the `learned_*` arms and for V2; sampled pins are
   reported apart.
9. Streams are collected as they finish, a failing stream leaves its source incomplete, and the
   cache keeps every finished stream.
10. The justification of k = 4 was corrected against the development record, and k was kept.
11. The bootstrap over streams of the primary is reported as descriptive.
12. The generator's size weights were removed from `priors/`, which joined the learning packages of
    `tests/test_information_boundary.py`; the oracle's weights come from the configs (evaluator
    side).

Test runs with the changes (2026-11-05, HEAD 7dacf57 with this working tree): before the review's
fixes, `tests/test_hybrid_priors.py` 56 passed in 48 s of wall time (the pipeline test 1.8 s, 316 s
before decision 4) and the existing suites named in HP2's Implementation passed as well (296 tests,
none failed or skipped). After the fixes, `tests/test_hybrid_priors.py` 61 passed (59 s), and
`tests/test_information_boundary.py` 6, `tests/test_hygiene.py` 81 and `tests/test_walker.py` 33
passed; the dry run and the development analysis were repeated with the frozen code (Development
record).

Frozen behaviour (shared with HP2; docs/studies/HP2_schema_priors.md, Implementation): the opt-in
hooks leave every existing config, stream and agent exactly as before.

## Amendments

None.

## Results

None yet: no measurement run exists.

## Verdict and limits

None yet.
