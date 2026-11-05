# Study HP2: lifted rules over families and classes as a prior for the sampling walker

Status: protocol frozen on 2026-11-05 by the commit that freezes this file, before any measurement
run, after an independent review (ready to freeze after fixes; its decisions are recorded under
Implementation); the measurement runs from a clean worktree of that commit. This is stage 2
of experiment RB-C in docs/studies/RB_rule_builder_design.md (roadmap item 6i), the hybrid of idea
I4: the graph supplies the objects, rules are learned over them and fed back to the posterior as a
prior. It adds an opt-in generator option, `schema`, whose default leaves every existing world, task
and stream unchanged (Implementation). Development used only the allocation `dev` of
`runs/HP2-ledger.json` and namespaces ending in `_dev`; its outcomes were seen before the
predictions below were written and are reported under Implementation as calibration, not results.
The measurement namespaces (`hp2_v1w`, `hp2_v1s`, `hp2_v2w`, `hp2_v2s`, `hp2_none`) have never been
used.

## Question and claim

Question: where the hidden requirements of a world follow a shared structure over public objects (a
schema: recipes belong to families, base facts to classes, and each family's requirement takes one
fact from each of its classes), does the sampling walker get cheaper when its uniform prior over
each recipe's hypotheses is replaced by a prior that it learns as lifted rules from its own
evidence, averaged against a model without a schema? How much of the structure's value does the
learned prior capture (the true schema as prior is the reference point), does it still help when
the classes are hidden and must be inferred from the facts that the evidence requires, and what
does it cost where there is no schema?

Claim under test (one primary): in schema worlds with public class and family labels and a schema
per world (setting v1 world), the `hybrid` walker has a lower late-half cost ratio (episodes 9-16)
than the `posterior` walker (`focused_sample`). Falsification: equivalence within 10% of the
posterior's mean late cost ratio. Harm check: the same contrast on worlds without a schema, within
5% (a secondary test, H1).

## Relation to earlier work

- Learned structure over typed objects (docs/PRIOR_ART.md, Section 13, where the records are
  verified): B1 (Kemp, Goodman and Tenenbaum, Cognitive Science 34, 2010) learns causal schemata,
  types of objects and causal laws between types, that speed up the learning of new causal models;
  here families and classes are the types and a schema is a law between them. B3 (Kemp,
  Tenenbaum, Griffiths, Yamada and Ueda, "Learning Systems of Concepts with an Infinite Relational
  Model", AAAI-06) infers latent classes of objects from the relations they take part in, the
  problem of version 2, solved here by enumerating class assignments of the facts the evidence
  requires. B2 (Kemp, Perfors and Tenenbaum, Developmental Science 10, 2007) learns overhypotheses
  across many sets. B4 (Sohn and coauthors' MTSGI, UAI 2022; Section 5, item 8 of docs/PRIOR_ART.md),
  the design note's closest work in reinforcement learning, stores (subtask graph, trajectory) pairs
  from earlier tasks, retrieves the most similar one for a new task and combines it with the graph
  inferred from the new task's own trajectory, as a product of the two graphs' policies. B5 (Diuk,
  Cohen and Littman, ICML 2008) states dynamics over object classes.
- docs/PRIOR_ART.md: the walker is posterior sampling in the sense of W5, and HP2 changes the prior
  of the posterior it samples; Bayesian rule sets (E3) place a prior over rule sets; clingo (E7)
  enumerates the hidden-class models; OHCAM (A3) keeps a Bayesian posterior over action models with
  a simplicity prior.
- Model averaging and Bayes factors (docs/PRIOR_ART.md, Section 13): B9 (Kass and Raftery, "Bayes
  Factors", JASA 90, 1995) and B10 (Hoeting, Madigan, Raftery and Volinsky, "Bayesian Model
  Averaging: A Tutorial", Statistical Science 14, 1999, with a correction in 2000). A model average
  drops a wrong model only as fast as the marginal likelihoods separate, which with sparse, partial
  evidence can be slow (development; H1), and the marginal likelihood is highly sensitive to choices
  that are typically made arbitrarily (B8, Section 7.4), such as HP2's uniform prior over class sets.
- Repository studies: study R found the sampling walker (`focused_sample`) cheaper than the
  brute-force plan in the late half of family F1 (late cost ratio 1.51 against 1.87), and study M
  kept the hypergraph representation and posterior sampling against pairwise and ranking variants;
  their sampler, family and late-half endpoint are HP2's. HP1 measures the simplest learned prior,
  over requirement sizes, offline. Study RL searches each recipe's attempts for the smallest
  consistent rule with clingo and compares it with the posterior as an inference method; HP2 learns
  rules across recipes and feeds them to the posterior as a prior that the posterior's consistency
  check can overrule.

docs/PRIOR_ART.md already records a comparison of a learned cross-task prior with the same inference
without it: MTSGI against MSGI (B4 against R4), on hidden subtask graphs. HP2 differs in its
qualifiers: its prior is lifted (rules over public families and classes, averaged against a model
without a schema), it feeds an exact-consistency sampler, and it is measured against the true schema
as a reference and on worlds without a schema as a harm check, neither of which appears in MTSGI's
Section 6 (its appendix was not read). It claims no new method.

## Protocol (frozen)

### Schema worlds (the generator option)

`world.schema` in a walker config builds an `envs.generator.SchemaWorldConfig`; every other config
builds the plain `WorldConfig` and never reaches the schema code. With K = 3 classes and F = 4
families:

- Classes: the 18 base types (12 resources and 6 facilities) fall into three classes of six.
  Version 1 uses the declared public table (class = type id modulo 3); version 2 (`hidden_classes`)
  draws a balanced partition with the schema and hides it.
- Families: the 16 item types fall into four families of four (item index modulo 4, public); a
  recipe's family is that of its effect item, so alternative recipes of an item share it.
- Schema: each family gets a class set, its size drawn with the world's arity weights (0.2, 0.5 and
  0.3 for sizes 1, 2 and 3, as the default generator draws requirement sizes), then that many
  distinct classes uniformly.
- Requirements: a family-f recipe requires, for each class of f's set, one pool fact of that class,
  drawn uniformly among the pool's facts of that class. Pools are drawn uniformly among the pools of
  six base facts that hold a fact of every class (rejection sampling; about 85% of uniform pools
  qualify), so every requirement has exactly one fact per class of its family's set.
- Scope: `world` draws a schema (and in version 2 a partition) per world, from a random stream
  separate from the world's layout; `stream` draws one per task stream, shared by its worlds (with
  `eval_world_offset_by_seed` every seed is its own stream); `none` draws no schema: requirements
  follow the default rule (a size by the arity weights, then a uniform subset of the pool), so the
  public labels are decoys; pools are drawn as in schema worlds, so only the requirements differ.
- Everything else is family F1: eight levels of two items, pools of six, an alternative recipe with
  probability 0.25, goals of depth 1-3, intermediate items unobserved, no distractor base facts,
  deterministic dynamics, budget two steps above the public brute-force bound.

### Settings

| Setting | Config | Namespace | Seeds x worlds | Unit | Schema scope | Classes |
|---|---|---|---|---|---|---|
| v1 world (primary) | `hp2_v1_world.yaml` | `hp2_v1w` | 0-4 x 10 | world (50) | world | public |
| v1 stream | `hp2_v1_stream.yaml` | `hp2_v1s` | 0-9 x 4 | stream (10) | stream | public |
| v2 world | `hp2_v2_world.yaml` | `hp2_v2w` | 0-4 x 10 | world (50) | world | hidden |
| v2 stream | `hp2_v2_stream.yaml` | `hp2_v2s` | 0-9 x 4 | stream (10) | stream | hidden |
| none (harm check) | `hp2_none.yaml` | `hp2_none` | 0-4 x 10 | world (50) | none | public, decoys |

Every setting plays 16 episodes per world in world-major order; seed s plays evaluation base seed
10,000 + 1,000 s. In the stream settings the worlds of a stream share their schema, so the unit is
the stream (the mean of its worlds' values); four worlds per stream leave room for learning across
worlds and ten streams give ten units.

### Arms

| Arm | Strategy | Charged as | Node chosen at a replan |
|---|---|---|---|
| `posterior` | `focused_sample` | adaptive | focused walk, then 50 consistent moves; uniform class prior |
| `hybrid` | `hybrid` | adaptive | the same walker with the learned prior (below) |
| `rules_only` | `rules_only` | adaptive | each recipe's most probable requirement under the learned prior |
| `oracle_schema` | `oracle_schema` | adaptive | `hybrid` with the true schema as its prior (privileged) |
| `maximal` | `maximal` | reporting | the full-pool node |
| `reference` | `reference` | reporting | the evaluator's optimal plan (privileged) |

`posterior` is studies R and M's sampler with their settings (at most 400 evaluations per walk,
temperature 0.5, restart after 80, 50 consistent moves). `hybrid` changes two things only, through
the walker's opt-in prior (`Walker(prior=...)`): every recipe without evidence is drawn from the
prior at every replan, and the consistent moves (Metropolis moves over single edits of attempted
recipes, restricted to nodes consistent with every log) target the prior instead of the uniform
prior: a move from node x to node y is accepted with min(1, pi(y) |N(x)| / (pi(x) |N(y)|)), pi the
prior and N(x) the edits available at x. The focused walk to a consistent node is unchanged.
`rules_only` plans on the prior's most probable hypothesis of every recipe (ties drawn at random at
every replan) without a walk and without checking consistency: the rules alone, a control. It keeps
its logs, so its rules are learned as `hybrid`'s are. `oracle_schema` gives `hybrid` the true
schema (in version 2 also the true partition; in worlds without a schema the generator's size
prior), mixed with the uniform prior at 0.01 so that every hypothesis keeps a positive prior; its
privileged read of a world's hidden schema happens on the evaluator side
(`evaluation.hybrid_priors`), before each episode, as the reference reads the rules. It is a
reference point, not a bound: the walker acts on one sample, and a truer prior need not make it
cheaper.

### The learned prior (definitions)

Data. At every replan, study M's pairwise update (`agents.markov.PairwiseEvidence`) reads the logs
of the learning scope (the current episode's included) and settles candidates: eliminated by a
success, forced by a failure formula, and the class bounds. With deterministic dynamics and
persistent facts every settled candidate is implied by the evidence (sound, not complete; tested
against the truth). Every recipe with a settled candidate gives an observation: its requirement
lies in H_r, the hypotheses of the class that hold every required candidate and no eliminated one;
a recipe with every candidate settled is pinned (|H_r| = 1). The learners are refitted whenever the
observations change.

Rules. A rule reads "a family-f recipe requires one fact of each class in C_f". For every family the
learners consider the seven nonempty class sets of at most three classes, with a uniform prior.

- Version 1 (public classes), exact for these data: P(D_f | C) is the product over the family's
  observed recipes of the sum over h in H_r of P(h | C, pool), where P(h | C, pool) is the product
  over c in C of 1 / (the pool's facts of class c) when h holds exactly one fact of each class in C
  and nothing else, and 0 otherwise. The posterior over C and the predictive for a recipe follow.
- Version 2 (hidden classes): clingo (5.8.2, as study RL records) enumerates every class assignment
  of the facts the evidence requires together with the class sets of their families such that the
  required facts of a recipe have distinct classes, all in its family's set, and a pinned
  requirement has exactly as many classes as facts. A model's prior is K^-n for its n facts
  (classes independent and uniform, which ignores that the true partition is balanced) times 1/7
  for every class set it fixes. Each model is weighted by the likelihood of all observations, in
  which the classes of facts the model does not fix are marginalized within each pool separately
  (each class with probability 1/3, conditioned on the pool holding every class), a declared
  approximation that ignores that such a fact keeps one class across pools; the class set of a
  family without a required fact is summed out under its uniform prior. A refit with more than
  50,000 models is refused: the learner and the model average keep their previous fit, and the
  overflow is counted. A few refused refits are expected in version 2 (the review's version 2 trial
  had one, with 19.2 s for that node choice); a refused refit is tried again, enumerating anew,
  whenever the observations change (about 19 s each time), a known runtime cost.
- Model average: the schema model against the no-schema model, the uniform prior over the class
  (the posterior's own prior, likelihood the product of |H_r| / 41). With prior probability 0.5 for
  the schema model, the two marginal likelihoods give its weight w, and a recipe's prior is w times
  the schema predictive plus (1 - w) times the uniform prior, computed in log space. Data that a
  schema cannot produce give it likelihood 0 and weight 0.
- Learning scope (`walker.hybrid.scope`, declared per config to match the setting): `world` learns
  from the current world's recipes and starts afresh in every world (settings v1 world, v2 world,
  none); `stream` learns from every world of the stream played so far (the stream settings).

Fixed a priori: K = 3 and F = 4 (public), the uniform prior over class sets (which implies
requirement sizes 1, 2 and 3 with probabilities 3/7, 3/7 and 1/7), the model prior 0.5, the
oracle's floor 0.01, the model cap 50,000 and the walker's search settings. Each episode row
records `choose_seconds` (node choice including the prior update), `hp_observed` and `hp_pinned`
(observed and pinned recipes in the current fit), `hp_schema_weight` (w at the last replan) and,
with hidden classes, `hp_models` and `hp_overflow` (refused refits so far).

### Endpoints

Per world and arm (study PG's `unit_metrics`): the late cost ratio (primitive steps over the
reference's optimal steps, summed over episodes 9-16; a failed episode counts every step it used),
early and overall cost ratios, late success, steps to the first success, and search evaluations and
node-choice seconds per replan; for the learning arms, the observed and pinned recipes, the schema
weight and (hidden classes) the number of models at the world's end. In the stream settings a
unit's value of each is the mean of its worlds' values.

### Primary contrast, falsification and verdict

P1: v1 world, 50 worlds, `hybrid` - `posterior` in late cost ratio, paired by world: holds when the
upper end of the 95% world-cluster bootstrap interval is below 0. E1 (falsification by
equivalence): the 90% interval of the same difference lies within plus or minus 10% of
`posterior`'s mean late cost ratio. V1 (validity): the reference solves every task of the primary
setting; V1 is undetermined when a reference run of that setting is missing or incomplete.
Verdict (study PG's precedence): "invalid" when V1 fails; P1 without E1 is a "pass"; both is
"negligible"; E1 alone is "falsified"; neither is "inconclusive". The verdict is None when a run of
either arm is missing or incomplete, when the runs come from more than one code identity, when
fewer than 50 worlds pair up, or when V1 is undetermined.

### Secondary tests (each with its own criterion, no multiplicity correction)

"Upper end" and "lower end" are those of the 95% interval of the paired difference. A test whose
setting lacks a completed run for some seed, or pairs fewer units than its setting has, is
incomplete and does not hold.

| Test | Setting | Contrast (paired by unit) | Metric | Holds when |
|---|---|---|---|---|
| S1 | v1 world | `oracle_schema` - `posterior` | late cost ratio | upper end below 0 |
| S2 | v1 world | `hybrid` - `oracle_schema` | late cost ratio | descriptive |
| S3 | v1 world | `rules_only` - `hybrid` | late cost ratio | lower end above 0 |
| S4 | v1 world | `hybrid` - `posterior` | late success | lower end above -0.02 |
| S5 | v1 stream | `hybrid` - `posterior` | late cost ratio | upper end below 0 |
| S6 | v2 world | `hybrid` - `posterior` | late cost ratio | upper end below 0 |
| S7 | v2 stream | `hybrid` - `posterior` | late cost ratio | upper end below 0 |
| H1 (harm) | none | `hybrid` - `posterior` | late cost ratio | upper end below 5% of `posterior`'s mean |
| S8 | v1 world | `posterior` - `maximal` | late cost ratio | upper end below 0 |

Eight tests with a criterion: if none of their hypotheses were true and the tests were independent,
the chance that at least one holds anyway would be about 1 - 0.95^8, or 0.34; the verdict rests on
the primary alone. Descriptive: per setting and arm, the means over units of every endpoint above;
the failed episodes of every arm (the reference's and `maximal`'s included); and the worlds whose
hidden-class learner had refused a refit by their end.

### Predictions (written after the development checks)

Written on 2026-11-04, after the smoke checks and development runs of 2026-10-29 (two to four
worlds per setting, outcomes seen; Implementation). The development estimates rest on two to four
worlds and the smoke checks sometimes point the other way, so the ranges below are wide.

- Primary: "negligible" (P1 and E1 both hold), estimate between -0.10 and -0.04 (hybrid 3% to 7%
  cheaper in the late half); the next likeliest outcome is "falsified" (E1 alone, an estimate
  between -0.04 and +0.05). Reason: the schema weight can rise slowly within a world (in the v1
  development worlds it passed 0.95 by the end of episode 6 in two worlds and of episodes 11 and 15
  in the two that pinned no recipe), and until the class sets are learned the uniform prior over
  class sets favours small requirements (sizes 1, 2 and 3 with probabilities 3/7, 3/7 and 1/7),
  which costs failed attempts. Development: -0.096 [-0.198, 0.044] on four worlds; the second smoke round (two
  other worlds) had the hybrid dearer (1.84 against 1.63).
- S1 holds, estimate between -0.20 and -0.60 (development -0.267; smoke rounds -0.66 and -0.59): a
  known schema is worth a lot. S2 (descriptive): between +0.10 and +0.50 (development +0.171).
- S3 holds, estimate +0.7 or more (development +1.263, smoke +0.78): the rules alone, without the
  consistency check, fail often (late success 0.09 to 0.44 for `rules_only` in development).
- S4 does not hold, estimate between -0.04 and 0 (development 0.000, late success 1.00 for both;
  second smoke round 0.88 against 0.96): early small requirements cost the occasional failed
  episode.
- S5 holds, estimate between -0.10 and -0.30 (development -0.305, smoke -0.12): shared across a
  stream's worlds, the rules are learned once (the weight was 0.96 or more from the second world of
  every development stream on).
- S6 and S7 do not hold, estimates between -0.10 and +0.10 (development -0.026 and -0.034; smoke
  +0.21 in v2 worlds): with hidden classes, few required facts per family fix little of the
  partition (end weights 0.60 to 0.99 in the v2 development worlds).
- H1 does not hold, estimate between 0 and +0.20 (development +0.250 [-0.010, 0.604]; smoke -0.33):
  the model average gives a schema weight 0 only when some family's evidence contradicts every
  class set; until then random requirements can fit a per-family class set about as well as the
  uniform prior or better (end weights 0.93, 0.28, 1.00 and 0 in the four development worlds).
- S8 holds, estimate between -0.3 and -0.8 (development -0.721): the posterior beats the
  brute-force plan in schema worlds too, as in study R.
- V1 holds.

### Statistics

The unit is the world in the world-scope settings and the stream in the stream settings; worlds
(and streams) are independent across seeds. Every contrast is a paired per-unit difference; its
mean gets a percentile cluster bootstrap interval (10,000 resamples, seed 20261008,
`evaluation.stats.cluster_bootstrap_mean`), 95% unless stated, 90% for E1; an exact two-sided sign
test is reported with each. The stream settings have ten units: percentile intervals over ten units
undercover, so their sign tests are read alongside them. Runs: per setting, arm and seed the earliest
completed walker run; a run that did not complete is not used, and a seed without a completed run
makes its arm's tests incomplete. Every run of a setting must come from one code identity (commit and
loaded code), and the runs of each (setting, arm) from one config (the hash of the recorded run
config without the fields that differ between seeds and reruns of one arm: the per-seed base seed,
the seed list and the arm list, which a rerun of one (seed, arm) pair rewrites, and the config's file
path; the run's own arm is kept); otherwise the arms concerned have a problem and their tests are
incomplete. The analysis records the clingo version it ran with.

### Budget and runtime

Worst cases are the sums of the task budgets of each run's stream (dry run, generator only); every
per-run cap is the largest such sum of its config, so no cap can cut a run short. The four arms
other than `maximal` and `reference` are charged as adaptive.

| Config | Runs | Per-run cap | Stream budget sums, all seeds | Adaptive worst case | Reporting worst case |
|---|---|---|---|---|---|
| `hp2_v1_world.yaml` | 30 | 3,184 | 15,714 | 62,856 | 31,428 |
| `hp2_v1_stream.yaml` | 60 | 1,302 | 12,314 | 49,256 | 24,628 |
| `hp2_v2_world.yaml` | 30 | 3,111 | 15,397 | 61,588 | 30,794 |
| `hp2_v2_stream.yaml` | 60 | 1,267 | 12,185 | 48,740 | 24,370 |
| `hp2_none.yaml` | 30 | 3,218 | 15,808 | 63,232 | 31,616 |
| measurement, in all | 210 | | | 285,672 | 142,836 |
| development, five `hp2_dev_*.yaml` | 60 | 622 to 1,011 | | 29,992 | 14,996 |

Stream budget sums per seed (dry runs of 2026-11-05, which reproduced every figure above): v1 world
3,184, 3,167, 3,100, 3,181 and 3,082 (seeds 0-4); v2 world 3,071, 3,080, 3,038, 3,097 and 3,111;
none 3,203, 3,192, 3,059, 3,136 and 3,218; v1 stream 1,302, 1,173, 1,254, 1,225, 1,237, 1,225,
1,217, 1,230, 1,188 and 1,263 (seeds 0-9); v2 stream 1,227, 1,222, 1,267, 1,152, 1,193, 1,190,
1,251, 1,253, 1,267 and 1,163. Every per-run cap equals the largest sum of its config.

Ledger `runs/HP2-ledger.json`, declared identically in every config of `configs/hybrid_priors/`
(tested): allocation `hp2` 288,900 (285,672 + 3,218 for one rerun of the largest run = 288,890,
rounded up to the next hundred), allocation `dev` 31,100 (29,992 + 1,011, rounded up), adaptive cap
320,000 (their sum), reporting cap 161,100 (142,836 + 14,996 + 3,218 = 161,050, rounded up). The
ledger file was created by the first development run with this declaration; after development it
holds 18,544 adaptive (`dev`) and 7,348 reporting interactions.

Runtime from the development rates (a machine shared with other work): per run of 32 tasks (world
settings) or 48 (stream settings), `posterior` took 1.0 to 4.0 s, `hybrid` 1.0 to 11.5 s (at most
0.31 s per task), `rules_only` 2.5 to 12.1 s, `oracle_schema` 0.3 to 3.1 s, and `maximal` and
`reference` at most 0.4 s; the 60 development runs took 3 min 24 s in all. A measurement run has 160
tasks (world settings) or 64 (stream settings), so a world-scope config should take at most about
15 minutes and a stream-scope config about 10, about an hour for the five in sequence or about 15
minutes with the five configs as concurrent processes (they share the locked ledger file). The
review's trials of the `hybrid` arm took 30 to 64 s per 160-task run and 8 to 19 s per 64-task run
(Implementation). The hidden-class learner slows as its models grow (6,912 at most in development,
4,212 in the review's version 2 stream trial); the model cap bounds each enumeration, and every
refused refit costs about 19 s again whenever the observations change. The measurement runs at
night.

### Stopping and rerun rules

Every run plays its whole stream; no interaction cap can bind (the model cap of version 2 is a
different, declared limit of the learner). No interim analysis; no change of arms, parameters,
settings, endpoints or criteria after any measurement row exists, other than dated amendments.
Every run and every rerun uses the same snapshot, because the verdict needs one code identity. The
walker command has no resume: after an interruption only the missing (seed, arm) pairs are run
again, one command per pair with `--seeds` and `--arms` (runs are deterministic given the code), and
the analysis uses the earliest completed run of each. A stale `runs/HP2-ledger.lock` may be removed
(a live holder keeps it for milliseconds, and a waiting command gives up after a minute). The
analysis is run by hand, once, after every run of the five configs has been checked complete; no
script runs it automatically. A fault that needs a code fix means a new freeze and a fresh run of
every measurement config from the new snapshot; such a full rerun exceeds the `hp2` allocation and
needs a logged ledger amendment first.

### Commands (from a clean snapshot, after review and freeze)

From the repository root, which holds `runs/` (and the ledger) and `artifacts/`; `S` is a clean
worktree of the freeze commit:

```bash
git worktree add ../snapshots/HP2-<hash> <hash>
S=../snapshots/HP2-<hash>
for c in v1_world v1_stream v2_world v2_stream none; do
  PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.walk \
      --config $S/configs/hybrid_priors/hp2_$c.yaml --dry-run
done
# at night; the five configs may run as concurrent processes
for c in v1_world v1_stream v2_world v2_stream none; do
  PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.walk \
      --config $S/configs/hybrid_priors/hp2_$c.yaml &
done; wait
# once, after every run has completed
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.evaluation.hybrid_priors hp2 \
    --runs runs --root $S --out artifacts/hybrid_priors
```

The analysis writes `artifacts/hybrid_priors/hp2_summary.json` and prints every test.

### Limits stated in advance

- Gains by construction: in schema worlds a prior that knows the schema should help, and the oracle
  shows by how much. The informative parts are the gap between the learned prior and the oracle,
  the harm check, and version 2, where the classes come from the evidence.
- One schema generator: three classes, four families, one fact per class, pools that hold every
  class. Version 2 uses the covering-pool rule as generator-level knowledge, like K and F (its
  likelihood conditions every pool on holding every class); what it leaves unused is the
  probability that a pool covers every class, as evidence about the labels (weak: about 85% of
  uniform pools hold every class anyway).
- The rules read the settled candidates of study M's pairwise update, which are sound but not
  complete: evidence that only the joint posterior can resolve does not reach the rules. Version 2
  marginalizes unfixed classes per pool independently and gives class assignments an independent
  uniform prior. The uniform prior over class sets favours small requirements before any evidence
  (sizes 1, 2 and 3 with probabilities 3/7, 3/7 and 1/7).
- The learners treat the settled candidates as a random sample of the evidence, but whether a
  candidate gets settled depends on the requirement's size, so the Bayes factor can lean toward class
  sets whose requirements settle easily.
- The rules use the same evidence twice: the logs that fit the rules also feed the posterior that the
  rules' prior enters.
- The model average with prior 0.5 gives a wrong schema weight 0 only when some family's evidence
  contradicts every class set; otherwise the weight follows the Bayes factor, which on sparse
  evidence can favour a schema for random requirements (H1). A more cautious model prior was not
  tried, by design (Implementation, decision 2).
- The model cap can bind in version 2: a few refused refits are expected (one in the review's
  version 2 world trial), each leaving the previous fit in place until a later refit succeeds; how
  often it binds is reported (the worlds whose learner refused a refit).
- The agent is told its learning scope (world or stream), which matches the setting; learning
  whether schemas persist across worlds is not tested.
- The stream settings have ten units of four worlds each; their intervals are wider, and percentile
  intervals over ten units undercover (their sign tests are read alongside).
- `rules_only` is the rules alone, deliberately without recipe-level evidence; it is a control, not
  a competitive rule-based agent. It learns from its own logs, which often settle nothing: in
  development, seed 0 of the v1 world setting failed every episode at the deadline with no
  observation and weight 0.5 throughout. S3 therefore cannot isolate the value of the consistency
  check.
- The prior enters the walker only through draws for recipes without evidence and the target of
  the consistent moves. The hybrid's draws from the prior use its random stream differently, so the
  arms are not coupled at the decision level.
- One family (F1), deterministic dynamics, 16 episodes per world, goal-only observation.

## Implementation

- `src/hypergraph_agent/envs/generator.py` (the `schema` option only): `SchemaConfig`,
  `SchemaWorldConfig`, `WorldSchema`, the public labels `public_class` and `family_of`,
  `draw_schema`, covering pools, schema requirements, `make_world`'s branch for schema configs and
  `TaskStream.stream_schema`; `World` gains a hidden `schema` field (default None).
- `src/hypergraph_agent/config.py`: `world.schema` and `walker.hybrid` are accepted as optional keys
  that are absent from the defaults, so every other config loads to the same dictionary (tested);
  `world_config` builds a `SchemaWorldConfig` only when a schema is declared. The study changes this
  shared module minimally, because without these two optional keys the schema option could not be
  reached from walker configs.
- `src/hypergraph_agent/agents/walker.py` (opt-in hooks only): `Walker(prior=None)`, prior draws for
  recipes without evidence, and `mix_consistent(..., log_prior=None)`; with the defaults every draw
  and decision is unchanged.
- `src/hypergraph_agent/topology/inference.py` (opt-in hooks, documented with HP1):
  `size_log_prior`, which the oracle uses in worlds without a schema.
- `src/hypergraph_agent/priors/schema.py`: observations and their sets H_r, `rainbow`, `SchemaV1`
  (exact), `SchemaV2` (clingo enumeration, per-pool marginalization, model cap) and `ModelAverage`
  (which keeps its previous fit when the learner refuses a refit); public information only (the
  package is one of the learning packages that `tests/test_information_boundary.py` checks).
- `src/hypergraph_agent/agents/hybrid.py`: `hybrid_params` (the free-form `walker.hybrid` key), the
  settled-candidate observations, `LearnedPrior` (per scope, refitted when the observations
  change), `HybridAgent` (`hybrid`, `rules_only`) and `make_hybrid_agent`; public information only.
- `src/hypergraph_agent/evaluation/hybrid_priors.py`: the privileged `OracleSchema` (the true schema
  as a prior) and `OracleSchemaAgent`, and the HP2 analysis (command `hp2`; `--dev` for the
  development configs; one code identity per setting and one config per (setting, arm); the clingo
  version recorded).
- `src/hypergraph_agent/evaluation/walkers.py`: `make_walker` imports `agents.hybrid` (and, for
  `oracle_schema`, the evaluator module) only when one of its strategies is asked for, and refuses
  names that shadow walker, policy-graph or Markov strategies.
- `configs/hybrid_priors/`: five measurement configs and five development configs (`hp2_dev_*`), in
  the walker schema (loaded by `tests/test_hygiene.py` like every walker config).
- `tests/test_hybrid_priors.py` (shared with HP1): the frozen-behaviour digests below; schema worlds
  (one fact per class of the family, balanced partitions, every pool holding every class, the three
  scopes, determinism, validation, configs through the loader, a public view that never carries the
  schema); the prior hooks (`prior_marginal`, `size_log_prior`, the `DependencyModel` hook, a
  Metropolis target equal to prior times consistency, prior draws for recipes without evidence);
  the rule learners (`rainbow` and H_r, version 1's class set and marginal likelihood against brute
  force, a contradicted schema dropped to weight 0, version 2's clingo models equal to a brute-force
  enumeration and its predictive, the model average keeping its fit on an overflow, the oracle's
  prior); the agents (parameter validation, names, every arm playing schema worlds in both versions,
  the oracle refusing default worlds, settled candidates agreeing with the truth, noise refused);
  the configs (one ledger declaration, caps equal to the largest stream sums, worst cases within the
  allocations, generator and agent settings in agreement, development namespaces and allocation);
  the analysis end to end on a tiny run (the descriptive table, the overflow count, the clingo
  version, and V1 undetermined without the reference's run); the config identity (equal across seeds,
  config paths and a rerun of one (seed, arm) pair, different for another arm or a changed
  parameter); and the hygiene of the new files (line
  length, dashes, no hidden field read by agent-side code). `tests/test_information_boundary.py`
  includes `priors` among the learning packages.

Frozen behaviour (default off). The tests embed digests computed with the code of commit 8b06e2a
(no file they depend on changed up to 7dacf57: the commits in between only add study RL's package
and configs), and the working tree matched them every time they were recomputed on 2026-10-29:
every one of the 67 walker-schema configs loads to the same dictionary; the evaluation streams of
13 frozen configs (studies R, M, R2 and PG and walker Stage D; every task of every seed and variant,
with each stream's config hash and manifest hash) and X2's generated streams are generated
identically; and 21 agent cases (all ten walker strategies, the four Markov and three policy-graph
strategies, goal-only and items observable, with and without failure noise) produce identical
episode rows. Study R's committed analysis (`artifacts/replication/replication_summary.json`, `.md`
and `replication_cost_by_episode.png`) regenerated byte-identically with the working tree on
2026-10-29, and study R2's generator-only family statistics (13 families of 800 tasks) gave the
same Markdown and printed output with HEAD and with the working tree, the JSON differing only in
the recorded config path. Study RL's development record (2026-10-29) reports that a replay of X2's
development collection of `gen_obs` with this working tree reproduced X2's recorded logs, recipes
and hidden requirements. The changes made after 2026-10-29 touch only HP2's own modules and the
tests; on 2026-11-05 the frozen-behaviour tests passed again with them, as did the existing suites
(below).

Development commands (from the repository root):

```bash
for c in v1_world none v2_world v1_stream v2_stream; do
  python -m hypergraph_agent.walk --config configs/hybrid_priors/hp2_dev_$c.yaml
done
python -m hypergraph_agent.evaluation.hybrid_priors hp2 --runs runs --dev --out ../scratch/hp2_dev
```

### Development record (calibration, not results)

Smoke checks (2026-10-29, namespace `hp2_smoke_dev`, two worlds per setting, played with an
in-memory ledger; not charged). In the first round the learner read only pinned recipes: it pinned
no recipe in three of the four settings and one in the fourth, its schema weight stayed at 0.5
(0.23 in the fourth), and `hybrid`'s late cost ratios were 1.76 to 2.10 against 1.90 to 2.04 for
`posterior`, while `oracle_schema` reached 1.23 to 1.44 in the three schema settings (decision 1).
The second round used the learner of the development runs (only line wrapping changed before
them): the schema weight ended at 0.98 (v1 world), 0.98 (v1 stream), 0.82 (v2 world) and 0 (no
schema), and `hybrid` against `posterior` was 1.84 against 1.63 (v1 world), 1.59 against 1.71 (v1
stream), 1.66 against 1.45 (v2 world) and 1.94 against 2.27 (no schema).

Development runs (2026-10-29, 10:50 to 10:54 local time): the five `hp2_dev_*.yaml` configs, seeds
0-1, two worlds (world settings) or three (stream settings) of 16 episodes, every arm: 60 runs, all
completed, 18,544 adaptive and 7,348 reporting interactions, 3 min 24 s in all.

Late cost ratio (late success in brackets) per setting, four worlds or two streams:

| Arm | v1 world | v1 stream | v2 world | v2 stream | none |
|---|---|---|---|---|---|
| `posterior` | 1.372 (1.00) | 1.537 (0.98) | 1.392 (1.00) | 1.341 (1.00) | 1.352 (0.94) |
| `hybrid` | 1.276 (1.00) | 1.233 (1.00) | 1.365 (1.00) | 1.308 (0.98) | 1.602 (1.00) |
| `rules_only` | 2.539 (0.22) | 2.967 (0.12) | 2.569 (0.44) | 2.667 (0.27) | 2.771 (0.09) |
| `oracle_schema` | 1.105 (1.00) | 1.215 (1.00) | 1.049 (1.00) | 1.092 (1.00) | 1.423 (0.94) |
| `maximal` | 2.093 (1.00) | 2.026 (1.00) | 1.822 (1.00) | 2.063 (1.00) | 1.818 (1.00) |
| `reference` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Tests on these units (intervals over four worlds or two streams are not informative): P1 -0.096
[-0.198, 0.044] and E1 not inside (90% [-0.183, 0.024] against a band of 0.137): "inconclusive";
S1 -0.267 [-0.304, -0.215]; S2 +0.171 [0.089, 0.259]; S3 +1.263 [0.682, 1.685]; S4 0.000 [0.000,
0.000]; S5 -0.305 [-0.449, -0.161]; S6 -0.026 [-0.115, 0.067]; S7 -0.034 [-0.180, 0.112]; H1 +0.250
[-0.010, 0.604]; S8 -0.721 [-1.188, -0.369]; V1 held (the reference solved every task of every
setting).

Schema weight (at the last replan of each episode). In the v1 world-scope worlds it passed 0.95 by
the end of episode 6 in the two worlds of seed 1 (which had pinned four and two recipes by their
end) and of episodes 11 and 15 in the two of seed 0 (which pinned none), and ended at 0.98 to 1.00.
In the v2 world-scope worlds it ended at 0.60 to 0.99. In both stream settings it passed 0.95
within the first world or at the start of the second and stayed at 0.96 or more from the second
world on. In the four worlds without a schema it ended at 0.93, 0.28, 1.00 and 0: the last fell
from 0.81 to 0 in episode 14 (a contradiction), and the one at 1.00 had three pinned recipes, each
producible by a class set of its family. The hidden-class learner held at most 6,912 models (cap
50,000) and never overflowed. In the worlds without a schema, `oracle_schema` (which plans with the
generator's sizes there) was dearer than `posterior` (1.423 against 1.352 on four worlds).

Review checks (2026-11-05, run by the independent reviewer in memory, uncharged): the `hybrid` arm
only, on the namespaces `hp_review_v2s_dev` and `hp_review_v2w_dev`, 8,144 interactions in all; 30
to 64 s per 160-task run and 8 to 19 s per 64-task run. In the version 2 stream trial the learner
held at most 4,212 models and never overflowed; the version 2 world trial had one refused refit
(above 50,000 models), with 19.2 s for that node choice.

Decisions that came from development:

1. The rule learners read the settled candidates of every recipe with evidence, not only pinned
   recipes (the first smoke round).
2. The model prior stays at 0.5 although the development worlds without a schema showed harm (H1
   +0.25 on four worlds; the second smoke round showed a gain there): the averaging is the design's,
   and tuning it on four worlds would fit noise. The review kept it: H1 tests the harm in advance.
3. `config.py` accepts `world.schema` and `walker.hybrid` as optional keys absent from the defaults,
   and builds a `SchemaWorldConfig` only for configs that declare a schema.
4. The oracle is built on the evaluator side; agent-side code never reads a world's schema (tested).
5. The analysis reads the development configs with `--dev`.
6. Made while writing this protocol (2026-11-04), affecting no development run: when the
   hidden-class learner refuses a refit (an overflow), the model average keeps its previous fit
   too, so that both marginal likelihoods read the same data; the analysis reports every
   descriptive endpoint and the overflow count, and V1 is undetermined (verdict None) without the
   reference's runs of the primary setting. The development test values do not depend on these
   changes: rerun on 2026-11-05, the development analysis reproduced them exactly.

Decisions of the independent review (2026-11-05), made before the freeze:

7. Accepted as declared: the model prior 0.5 (decision 2), the approximations of version 2, the
   definition of `rules_only`, the settled candidates as data, the primary on version 1, the ten
   stream units as secondary settings (their sign tests read alongside the intervals), no further
   development worlds, and the change to `config.py`.
8. The limits were corrected and extended: version 2 uses the covering-pool rule as generator-level
   knowledge; settled candidates are not a random sample; the evidence is used twice; the model cap
   can bind; `rules_only` often learns nothing from its own logs.
9. The rerun rules require one snapshot for every run, a logged ledger amendment before a full
   rerun, and an analysis started by hand once every run is checked complete.
10. The analysis requires one config per (setting, arm) and records the clingo version.
11. The privileged `OracleSchema` moved from `priors/` to the evaluator side, and `priors` joined the
    learning packages of `tests/test_information_boundary.py`.
12. A refused refit is not memoized (the review's R1 was skipped): it is enumerated again, about
    19 s each time the observations change, a known runtime cost.

Test runs with the changes (2026-11-05, HEAD 7dacf57 with this working tree, Python 3.12.13):
`tests/test_hybrid_priors.py` 56 passed (48 s of wall time); the existing suites walker 33 passed,
Markov 37, replication 17, boundary 25, information boundary 6, hygiene 81, environment 22, policy
graph 23, inference baselines 25 and rule learners 27 (352 tests in all, none failed or skipped,
about 6 minutes of wall time). The one warning comes from X2's `inference_baselines/benchmark.py`
(a mean of an empty slice in a test of X2's decision rule) and is unrelated to these changes. The
development analysis, rerun with the current code into a scratch folder, reproduced every value of
its 2026-10-29 summary exactly (estimates, intervals, sign tests, verdicts, the late cost ratio and
late success tables, failed episodes); the new code only adds the descriptive fields and the
overflow count, zero in every setting.

After the review's fixes (2026-11-05): `tests/test_hybrid_priors.py` 61 passed (59 s), and
`tests/test_information_boundary.py` 6, `tests/test_hygiene.py` 81 and `tests/test_walker.py` 33
passed. The development analysis, rerun once more, again reproduced every value of its 2026-10-29
summary exactly, now with one config per (setting, arm) and clingo 5.8.2 recorded; the five dry runs
reproduced the figures of the morning byte for byte. No HP2 config changed.

## Amendments

None.

## Results

None yet: no measurement run exists.

## Verdict and limits

None yet.
