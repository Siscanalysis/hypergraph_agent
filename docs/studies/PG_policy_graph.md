# Study PG: a policy graph (public states with learned action distributions)

Status: protocol frozen on 2026-10-22 by the commit that freezes this file, before any
measurement run; the measurement runs that commit from a clean snapshot. Development used only
the `dev` allocation of `runs/PG-ledger.json` and namespaces ending in `_dev`; its outcomes were
seen before the predictions below were written and are reported under Implementation as
calibration, not results.

## Question and claim

docs/ROADMAP.md item 6h records the idea: map the state to a node of a graph (states or abstract
states, edges the observed transitions) whose node holds a learned probability distribution over
actions, and act by sampling from it. Question: does such a policy graph, built and trained only
from the agent's own episodes within a world, learn within a world; how much of that is the learned
distributions rather than the graph and its bookkeeping; and where does it stand against the
repository's model-based agents and against study X's library learners on the same worlds?

The arms are `pg` (learned distributions, Thompson sampling), `pg_uniform` (the same graph and
bookkeeping with uniform distributions: the control that isolates the learned distributions) and
`pg_greedy` (an action of maximal count). The claim under test is narrow: in TechTree, study U's
cell k3-off-ann, `pg` takes fewer actions per episode than `pg_uniform` in the late half of each
world's stream (episodes 6-11). The secondaries ask the same in study L's phase L2 at reuse
depth 2 and in RecipeQuest family F1 with goal-only observation, compare the policy graph with the
repository's agents (`pooled`, `remember`, `promote`, `focused_sample`, `maximal`), and pair it
world by world with study X's best learner in each setting.

The study measures one declared node map per environment and one credit rule. It does not test
learned abstractions, learned controller memory or generalization across worlds: a fresh world
starts with an empty graph, as every agent of the repository starts a world without knowledge.

## Relation to earlier work

- Finite-state controllers. Hansen (1998, "Solving POMDPs by Searching in Policy Space", UAI,
  pp. 211-219) represents a policy for a partially observable problem as a finite-state
  controller, a graph whose nodes choose actions and whose transitions follow observations, and
  improves it by search in policy space; Meuleau, Peshkin, Kim and Kaelbling (1999, "Learning
  Finite-State Controllers for Partially Observable Environments", UAI, pp. 427-436) learn
  stochastic controllers by gradient descent and note that memoryless policies are usually not
  enough under partial observability. PG's controller memory is not learned: a node is a fixed
  function of the public observation and a bounded memory of the agent's own actions (a history
  window), the graph grows from experience, and only the node distributions are learned.
- State abstraction. Li, Walsh and Littman (2006, "Towards a Unified Theory of State Abstraction
  for MDPs", ISAIM, pp. 531-539) classify abstractions by what they preserve. PG's node map keeps
  every distinction the public observation and the memory window make and merges nothing else; it
  is a declared design input, not an abstraction found from data.
- Go-Explore (Ecoffet, Huizinga, Lehman, Stanley and Clune, arXiv:1901.10995, 2019; updated as
  "First return, then explore", Nature 590, 580-586, 2021) keeps an archive of cells, returns to a
  promising cell and explores from it. Study U2's `archive` arm follows that scheme with held sets
  as cells. PG's graph is a cell graph with a learned action distribution per cell instead of a
  return rule; after one success, `pg_greedy` replays a whole successful path.
- Credit by loop erasure. The loop-erased random walk (Lawler 1980, "A self-avoiding random
  walk", Duke Mathematical Journal 47(3), 655-693) is the path a walk leaves once its loops are
  erased in chronological order; Wilson (1996, "Generating random spanning trees more quickly than
  the cover time", STOC, pp. 296-303) builds uniform spanning trees from such walks. When the walk
  is Markov in node space (TechTree: a node and an action determine the next node), the last-exit
  decomposition gives the action a loop-erased successful walk credits at a node u, the action of
  the walk's last exit from u: given the loop-erased path up to u, it goes to a neighbour v with
  probability proportional to p(u, v) times the probability that the walk from v reaches the goal
  before hitting u or an earlier node of that path (approximately, since an episode stops after
  its budget), so the counts point towards the goal even when the behaviour is uniformly random.
- Posterior sampling. Osband, Russo and Van Roy (2013; docs/PRIOR_ART.md, W5) sample a model per
  episode and act on it. PG draws a categorical per decision from a Dirichlet over actions, which
  for a single action is the posterior predictive (see the definitions).
- Study X (docs/studies/X_open_source_baselines.md): Stable-Baselines3 PPO and sb3-contrib
  learners trained on up to 1,000,000 interactions of other worlds stayed at random level (X3,
  XU3, XU4, XL3) and did not improve within a world (X4). PG is a learner of the opposite kind: no
  training worlds, learning only within the world it plays.
- Studies U, L and U2 supply the TechTree cells and the anchors; study R the RecipeQuest family F1.

## Protocol (frozen)

### Settings

| Setting | Namespace | Worlds | Episodes per world | Late half | Config |
|---|---|---|---|---|---|
| U k3-off-ann | `x_tt_u_test` | seeds 0-4 x 6 (30) | 12 | 6-11 | `configs/policygraph_tt/pg_u_main.yaml` |
| L2 depth 2 | `x_tt_l_test` | seeds 0-7 x 4 (32) | 12 | 6-11 | `configs/policygraph_tt/pg_l_main.yaml` |
| F1 goal-only | `x_rq_test` | seeds 0-4 x 10 (50) | 16 | 8-15 | `configs/policygraph_rq/pg_rq_main.yaml` |

- U k3-off-ann (TechTree, study U): 4 base primitives, 3 extra slots, a key of 3 base primitives
  that activates one slot, the goal an entry of 3 symbols using it; no signal; announced unlocks;
  budget 100.
- L2 depth 2 (TechTree, study L): 3 base primitives, three levels of three concepts (lengths 2, 4
  and 6), levels 2 and 3 compositions of earlier concepts, goal a level-3 concept, nothing
  supplied; budget 100.
- F1 goal-only (RecipeQuest, study R): hidden prerequisites (pools of 6), goals at depth 1-3,
  intermediate items unobserved, no distractor base facts, failure probability 0; seed s plays
  base seed 10000 + 1000 s in pool mode (world-major order); the budget of a task is two steps
  above its public brute-force bound.

The worlds are study X's test streams of the same settings, reused on purpose so that `pg` can be
paired with X's learners world by world (study X used them as completed measurement worlds; X's
results on them were public before this protocol was written; nothing of PG was tuned on them, and development
used fresh `_dev` namespaces). `tests/test_policygraph.py` checks that the PG configs reproduce
X's test tasks exactly: TechTree task keys, world keys and goals for every seed; RecipeQuest task
keys and stream positions (hence dynamics seeds) for every seed.

### Arms

| Setting | Policy-graph arms | Anchors |
|---|---|---|
| U k3-off-ann | `pg`, `pg_uniform`, `pg_greedy` | `pooled` (study U), `random`, `reference` (privileged) |
| L2 depth 2 | `pg`, `pg_uniform`, `pg_greedy` | `remember`, `promote` (study L), `random`, `reference` |
| F1 goal-only | `pg`, `pg_uniform`, `pg_greedy` | `focused_sample`, `maximal` (study R), `reference` |

The anchors are the modules and parameters of their studies (TechTree: built by the PG factory
through `make_explorer` and `make_layered`; RecipeQuest: `focused_sample` with 50 consistent moves,
as in R and X). `pg_uniform` has exactly the action distribution of the random agents (uniform
over the available primitives in TechTree, over the task's gather, activate and craft actions in
RecipeQuest, submitting as soon as the goal is held or observed held); TechTree also runs the
repository's `random` arm, RecipeQuest does not (study X's random agent lives in `baselines/`).

### The policy graph (definitions)

Node map (public information only; per world, dropped with it):

- TechTree: (goal, held concepts, the last m presses). m = `order`, by default the longest public
  sequence length minus 1 (U: 2; L2: 5), so that a node and its next press can spell any concept;
  the window is shorter at the start of an episode. The agents never press `wait`, so the window is
  the episode's presses so far.
- RecipeQuest: (goal type, the set of fact types observed present, the craft actions made since
  that set last changed). Under goal-only observation crafting an intermediate item does not change
  the observation, so the third part, a memory of the agent's own crafts that is reset whenever
  the observation changes, separates the state before an invisible craft from the state after it.
  Actions are identified across the tasks of a world by public stable keys: gather or activate a
  fact type, craft a recipe signature (with an occurrence index if a task lists a signature twice).

Both node maps include the goal: tasks of a world have different goals in L2 and F1, and a policy
for one goal is not a policy for another. The actions of a node are the available primitives
(TechTree: base symbols and announced active slots) or the task's gather, activate and craft
actions (RecipeQuest); `wait` is never chosen and `submit` is pressed as soon as the goal is held
(TechTree) or observed held (RecipeQuest), as by the random agents.

Distributions. Each node holds a Dirichlet over its actions with prior alpha0 = 1 (uniform) plus
success counts. Learning happens only after a successful episode: the episode's decisions are
loop-erased in node space (chronologically: a node visited again cuts the path back to its first
visit) and every (node, action) left on the loop-erased path gains one count. Every successful
episode counts, not only the shortest so far; failed episodes add nothing. Each node of the path
thus credits the action of its last visit, after which the episode reached the goal without
returning. The rule is needed (an argument, not a development measurement): without loop erasure
the counts of a combination lock stay uniform in expectation, because in a uniformly random
successful episode the node before the last press of a composite is left by wrong presses about
as often as the uniform rule predicts (every wrong press there is followed by a later visit).

Acting. `pg`: Thompson sampling with a fresh draw at every decision, a categorical drawn from the
node's Dirichlet and then an action drawn from it. A single action drawn that way has the marginal
probability (alpha0 + n_a) / sum_b (alpha0 + n_b), so the agent samples that predictive directly.
`pg_uniform`: uniform over the node's actions; it keeps the same graph and counts but never reads
them. `pg_greedy`: uniform over the actions of maximal count (all actions when the node has no
counts). All three map one uniform variate per decision to an action in the same way, and the
variates of an episode come from a stream shared by every arm of a seed (TechTree: the agent seed,
the world and the episode index within the world; RecipeQuest: the episode's dynamics seed, which
the runner derives from the seed and the stream position, with a fixed tag). The three arms
therefore act identically in a world until its first success (common random numbers; tested), and
their actions to the first success coincide by construction.

Edges. The observed transitions (node, action, next node) are counted per world; nothing reads
them except the reports (graph size).

### Endpoints

Per world (the unit), per arm: actions per episode in the late half (TechTree), the late cost
ratio (actions over reference actions summed over the late episodes; RecipeQuest), early and
overall values, success rates, and actions to the first success in the world (censored at the
stream total); failed episodes count every action they used. Reported: the policy-graph counters
per episode (decisions, decisions taken at a node with counts, the loop-erased path, credited
pairs) and the graph size at the end of each world (nodes, edges, nodes with counts, successes)
(`policygraph.graph_summary`, under `row_summaries` of the TechTree analysis); interactions from the
ledger and wall-clock times.

### Predictions (written before any measurement run, after the development runs)

- PG1 (primary) holds: `pg` - `pg_uniform` in late actions per episode between -30 and -55 (`pg`
  about 35-55, `pg_uniform` about 85-92); E1 (equivalence) does not hold; the verdict is "pass".
  Late success: `pg` above 0.8, `pg_uniform` about 0.3 (S7 holds).
- `pg_greedy` is cheaper still (about 10-25 late actions; S1 and S2 hold): one success suffices for
  it to replay a loop-erased path.
- The model-based agents stay far cheaper on fresh worlds: `pooled` needs about 6.8 late actions
  and about 70 actions to the first success against about 230-270 for the policy-graph arms (S3,
  S4 and S5 hold).
- S6 holds: `pg_uniform` is equivalent to `random` within 10% (both about 90 late actions).
- L2 depth 2: the graph is large (hundreds of nodes per world at order 5) and successes are rare,
  so `pg` hardly learns: SL1 estimate between -4 and 0 actions, holding at most narrowly; SL2 holds
  (`pg_greedy` about 10-20 actions cheaper than `pg_uniform`); SL3 and SL4 hold (`promote` about
  11, `remember` about 22-30 late actions against about 92 for `pg`); SL5 holds.
- F1 goal-only: successes are rare (late success of the uniform rule below 0.1) and the counts are
  goal-specific, so `pg` acts as `pg_uniform` almost everywhere: PR1 does not hold (estimate within
  0.03 of 0); PR2 holds narrowly or not (`pg_greedy` 0.03-0.10 cheaper); PR3, PR4 and PR5 hold
  (`focused_sample` about 1.5, `maximal` about 1.9 against about 2.9-3.0 for `pg`).
- Against study X's best learners: `pg` is cheaper in U k3-off-ann (about 40-55 late actions
  against 95.2 for `mppo_rnd`) and in L2 (about 92 against 98.5, since X's learner was worse than
  random there), so XT-U and XT-L hold; in F1 `pg` is at the level of the uniform rule (about
  2.95-3.0 against 2.953 for `mppo`), so XT-R does not hold.

### Primary test and verdict

PG1, U k3-off-ann, late actions per episode, `pg` - `pg_uniform`, paired by world (30 worlds):
passes when the upper end of the 95% world-cluster bootstrap interval is below 0. Falsification
E1 (equivalence): the 90% interval of the same difference lies within plus or minus 10% of
`pg_uniform`'s mean. Validity V1: the reference solves every task. Verdict (rule `precedence` of
`techtree.analysis.verdict`): a failed validity test fails the study; PG1 without E1 is a pass;
both is "negligible"; E1 alone is "falsified"; neither is "inconclusive".

### Secondary tests (each with its own criterion, no multiplicity correction)

| Test | Setting | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| S1 | U k3-off-ann | `pg_greedy` - `pg_uniform` | late actions | upper end below 0 |
| S2 | U k3-off-ann | `pg_greedy` - `pg` | late actions | upper end below 0 |
| S3 | U k3-off-ann | `pooled` - `pg` | late actions | upper end below 0 |
| S4 | U k3-off-ann | `pooled` - `pg` | actions to the first success | upper end below 0 |
| S5 | U k3-off-ann | `pooled` - `pg_greedy` | late actions | upper end below 0 |
| S6 | U k3-off-ann | `pg_uniform` vs `random` | late actions | 90% interval within +-10% of `random` |
| S7 | U k3-off-ann | `pg` - `pg_uniform` | late success | lower end above 0 |
| SL1 | L2 depth 2 | `pg` - `pg_uniform` | late actions | upper end below 0 |
| SL2 | L2 depth 2 | `pg_greedy` - `pg_uniform` | late actions | upper end below 0 |
| SL3 | L2 depth 2 | `promote` - `pg` | late actions | upper end below 0 |
| SL4 | L2 depth 2 | `remember` - `pg` | late actions | upper end below 0 |
| SL5 | L2 depth 2 | `pg_uniform` vs `random` | late actions | 90% interval within +-10% of `random` |
| PR1 | F1 | `pg` - `pg_uniform` | late cost ratio | upper end below 0 |
| PR2 | F1 | `pg_greedy` - `pg_uniform` | late cost ratio | upper end below 0 |
| PR3 | F1 | `focused_sample` - `pg` | late cost ratio | upper end below 0 |
| PR4 | F1 | `maximal` - `pg` | late cost ratio | upper end below 0 |
| PR5 | F1 | `focused_sample` - `pg` | actions to the first success | upper end below 0 |
| XT-U | U k3-off-ann | `pg` - X's best learner (`mppo_rnd`) | late actions | upper end below 0 |
| XT-L | L2 depth 2 | `pg` - X's best learner (`mppo_rnd`) | late actions | upper end below 0 |
| XT-R | F1 | `pg` - X's best learner (`mppo`) | late cost ratio | upper end below 0 |

Descriptive: the same three comparisons with `pg_greedy` in place of `pg`; every arm's table;
the graph counters; per-episode cost curves (the TechTree figures).

### Comparison with study X's learners

X's best learner in each setting is the one X selected on its validation worlds (`mppo_rnd` in
both TechTree settings, stochastic actions; `mppo` in F1, deterministic actions), at 1,000,000
training interactions; its value in a world is the mean over its two training seeds, as in X. The
rows come from X's evaluation runs (`runs/x-*-main-eval-<learner>-s<training seed>-b1000000-test-*`;
local run directories of the checkout where X ran, not committed). Before any contrast,
`evaluation.policygraph_study` checks that (i) those rows reproduce X's committed per-world values
of X's own test between one of its agents and the best learner (XU2, XL1, X1; checked in
development: largest difference 0 in all three settings), and (ii) every task in X's rows has the
same reference length in PG's rows of the same world and episode; a failed check aborts. X's
learners are not re-run.

### Statistics

The unit is a world; worlds are independent across seeds. Paired per-world differences; the mean
gets a percentile world-cluster bootstrap interval (10,000 resamples, seed 20261008, 95%; 90% for
the equivalence tests) and an exact two-sided sign test (`techtree.analysis` for the TechTree
tests; `evaluation.stats.cluster_bootstrap_mean` and `sign_test` in `evaluation.policygraph_study`
for the RecipeQuest and X tests). The common random numbers make the policy-graph arms identical
until each world's first success, so their paired differences come from learning alone.

### Incomplete or repeated runs

TechTree: every (variant, arm, seed) needs exactly one complete run; `--resume` repeats only
missing or incomplete runs, and the analysis uses such a deterministic rerun only when it has the
same run hash and the same code (`techtree.analysis.select_runs`). RecipeQuest: the walker CLI has
no resume; after an interruption only the missing (seed, arm) pairs are run again, one command per
pair with `--seeds` and `--arms` (the two options select their cross product; runs are
deterministic given the code), and the analysis uses per arm and seed the earliest completed run,
all from one code identity; incomplete runs are listed (`incomplete_runs`) and not used. A
test that reads an arm without a complete run for every seed, or with fewer paired worlds than
the stream's, is "incomplete" and never holds.

### Stopping rules

Every run plays its whole stream; no cap can bind (each run's cap is the sum of its task budgets).
Each config is analysed once, after all its runs completed. No interim analysis; no change of
arms, parameters, settings, endpoints or criteria after any measurement row exists, other than
dated amendments.

### Budget and runtime

One ledger for the study, `runs/PG-ledger.json`, created by the first development run with the
declaration shared by every PG config (TechTree `ledger` block and walker `budget` block, tested
identical): allocations `dev` 150,000, `pg_tt` 261,600 and `pg_rq` 62,400 adaptive; adaptive cap
474,000 (their sum); reporting cap 330,000. The TechTree runner charges `pg_uniform` (whose
actions never depend on evidence), `random` and `reference` as reporting; the walker runner
charges every arm except `maximal` and `reference` as adaptive, `pg_uniform` included. Worst cases
from the dry runs (every episode using its whole budget):

| Config | Runs | Adaptive | Reporting |
|---|---|---|---|
| `pg_u_main.yaml` | 30 | 108,000 | 108,000 |
| `pg_l_main.yaml` | 56 | 153,600 | 115,200 |
| `pg_rq_main.yaml` | 30 | 62,400 | 31,200 |
| measurement in all | 116 | 324,000 (`pg_tt` 261,600 + `pg_rq` 62,400) | 254,400 |
| `pg_dev.yaml` (development) | 26 | 67,200 | 57,600 |
| `pg_rq_dev.yaml` (development) | 12 | 9,652 | 4,826 |

The RecipeQuest run cap (3,257) is the largest task-budget sum of the five seeds' streams. Runtime
in development on a 4-core laptop shared with other jobs (one process): 26 TechTree runs in 3 min
43 s and 12 RecipeQuest runs in 1 min 36 s, most of it run bookkeeping (each TechTree run computed
for 0.2-2.9 s). Estimate for the measurement: about 5 min for `pg_u_main`, 10 min for
`pg_l_main` and 10 min for `pg_rq_main`, one process each (they may run concurrently).

### Commands (after the freeze)

From the repository root, which holds `runs/` (with study X's runs) and `artifacts/`; the code
and configs come from a clean worktree of the frozen commit:

```bash
git worktree add ../pg_snapshot <frozen commit>
S=../pg_snapshot; export PYTHONPATH=$S/src PYTHONUNBUFFERED=1; PY=.venv/Scripts/python.exe
$PY -m hypergraph_agent.techtree --config $S/configs/policygraph_tt/pg_u_main.yaml --allow-measurement
$PY -m hypergraph_agent.techtree --config $S/configs/policygraph_tt/pg_l_main.yaml --allow-measurement
$PY -m hypergraph_agent.walk --config $S/configs/policygraph_rq/pg_rq_main.yaml
# analyses, once each, after every run of the config completed
for c in u l; do
  $PY -m hypergraph_agent.techtree.analysis --config $S/configs/policygraph_tt/pg_${c}_main.yaml \
      --runs runs --out artifacts/techtree/pg-$c-main
done
$PY -m hypergraph_agent.evaluation.policygraph_study rq --runs runs --root $S --out artifacts/policygraph
$PY -m hypergraph_agent.evaluation.policygraph_study x --runs runs --x-runs runs --root $S \
    --out artifacts/policygraph
```

After an interruption: the TechTree commands again with `--resume`; the walker command once per
missing (seed, arm) pair, with `--seeds <seed> --arms <arm>`. A stale `runs/PG-ledger.lock` older than a
minute may be removed (a live holder keeps it for milliseconds).

### Limits stated in advance

- One node map per environment and one credit rule, declared and not tuned: no learned
  abstraction, no learned controller memory, no backoff to shorter windows, no generalization
  across goals or worlds. A different map (a goal-free node, a shorter window in L2) could learn
  more or less.
- Thompson sampling per decision equals sampling the posterior predictive; a draw kept for a whole
  episode (posterior sampling in the sense of Osband et al.) is untested.
- Learning starts only at a world's first success, which the policy-graph arms reach as fast as
  the uniform rule; the study measures what the graph does with successes, not exploration.
- Success-only credit ignores the information in failures, which the model-based agents use
  (eliminations, failure clauses); the comparison with them is not between equals.
- The worlds were study X's test worlds; their results were public before this protocol.
- X's learners were trained on other worlds and are compared frozen; `pg` trains only within the
  world it plays. The pairing with X describes these fixed worlds and checkpoints.

## Implementation

- `src/hypergraph_agent/techtree/policygraph.py`: the environment-independent core (`pick`,
  `LoopErasedPath`, `WorldGraph`), the TechTree arms (`PolicyGraph`), the factory
  `make_policygraph` (which also builds study U's and L's arms and anchors unchanged),
  `REPORTING_ARMS` and the row summary `graph_summary` (`ROW_SUMMARIES`). Registered as study `PG`
  in `techtree/study.py` (`STUDY_AGENTS`) and `techtree/config.py` (`STUDIES`). The module is under
  the source check of the TechTree agent modules (no reflective builtins, dunder, frame or traceback
  attributes, no private attributes of other objects; imports restricted).
- `src/hypergraph_agent/agents/policygraph.py`: the RecipeQuest arms (`RecipePolicyGraph`), which
  read only the executor's public spec and observations; `evaluation/walkers.make_walker` imports
  the module lazily for its strategies (`pg`, `pg_uniform`, `pg_greedy`) and refuses names that
  shadow walker or Markov strategies.
- `src/hypergraph_agent/evaluation/policygraph_study.py`: the RecipeQuest endpoints and the paired
  comparison with study X's learners (evaluator side; CLI subcommands `rq` and `x`).
- Configs: `configs/policygraph_tt/` (TechTree schema: `pg_u_main.yaml`, `pg_l_main.yaml`,
  `pg_dev.yaml`) and `configs/policygraph_rq/` (walker schema: `pg_rq_main.yaml`, `pg_rq_dev.yaml`);
  `tests/test_hygiene.py` lists `policygraph_tt` among the directories it does not load with the
  RecipeQuest schema.
- Tests: `tests/test_policygraph.py` (loop erasure, the action choice, the weights of each arm,
  parameter checks, identical behaviour of the three arms until the first success in both
  environments, the greedy replay of the loop-erased path, counts only from successes, node
  contents, the factory and the reporting arms, the source checks of both agent modules, the
  dispatch and name collisions, one ledger declaration, caps covering the allocations, flags and
  namespaces, X's test streams reproduced task for task, an end-to-end run and analysis, the
  analysis helpers, and line lengths).

### Development record (calibration, not results)

Development used 80,224 ledgered interactions in 38 runs (allocation `dev`; 41,335 adaptive and
38,889 reporting): `pg_dev.yaml`, seeds 0-1 x 4 worlds per cell (namespaces `pg_tt_u_dev`,
`pg_tt_l_dev`), every arm, 26 runs, 32,837 adaptive and 36,508 reporting; `pg_rq_dev.yaml`,
seeds 0-1 x 4 worlds (namespace `pg_rq_dev`), every arm, 12 runs, 8,498 adaptive and 2,381
reporting. Before them, two smoke runs of one world per cell with an in-memory ledger (10,418
interactions, not in the ledger) checked that the arms run.

Outcomes on 8 development worlds per cell (seen before the predictions were written):

| Cell | `pg` | `pg_uniform` | `pg_greedy` | Anchors |
|---|---|---|---|---|
| U k3-off-ann, late actions per episode | 44.4 | 87.2 | 15.0 | `pooled` 6.9, `random` 86.1 |
| U k3-off-ann, late success | 0.94 | 0.35 | 1.00 | `pooled` 1.00, `random` 0.35 |
| U k3-off-ann, actions to the first success | 260.6 | 260.6 | 260.6 | `pooled` 50.9, `random` 202.4 |
| L2 depth 2, late actions per episode | 92.8 | 94.6 | 79.4 | `promote` 11.3, `remember` 28.1, `random` 93.4 |
| F1, late cost ratio | 2.872 | 2.872 | 2.790 | `focused_sample` 1.407, `maximal` 1.848 |
| F1, actions to the first success | 164.9 | 164.9 | 164.9 | `focused_sample` 21.0, `maximal` 13.4 |

`pg` - `pg_uniform` in U k3-off-ann: -42.8 [-54.2, -31.9] over 8 worlds; in L2: -1.8 [-5.5, 0].
In F1 `pg` and `pg_uniform` acted identically in all 8 worlds. Graph size at the end of a world:
about 47 nodes and 190 edges in U (loop-erased paths of about 13 decisions; 38% of the decisions
at a node with counts), about 950 nodes in L2 (0.4% of the decisions at a node with counts).

The development runs used a node map whose window, with fewer than m presses in the episode,
kept fewer presses than declared: in L2 (m = 5) the last 2 after 3 presses and the last 1 after 4
(for k3, m = 2, it was already exact). The review before the freeze found it and the frozen code
keeps the last min(presses, m) presses. A check on 16 scratch L2 worlds gave about 990 nodes per
world and 96.3 late actions for both `pg` and `pg_uniform` under either map; the L2 figures above
come from the earlier map.

## Results

None yet: no measurement run exists.

## Verdict and limits

Pending the measurement.
