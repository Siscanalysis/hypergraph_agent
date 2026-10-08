# Study M: pairwise Markov graphs with proximity ranking

Status: protocol frozen on 2026-10-08 by the commit that adds this file
(b443d2c), before any measurement run. The M1 collection runs executed that
commit and the M2 runs executed the commit of amendment MA1 (95d008a), each
from a clean checkout; the analyses were run from the same snapshots, and
Results and Verdict below report them. Development runs used only the
allocation `dev` and the namespace `markov_dev`.

## Question and claim

Can an agent whose knowledge of hidden prerequisites is a graph of pairwise
Markov transitions, read with a PageRank-style proximity score, match the
hypergraph walker, which evaluates complete dependency hypergraphs jointly
against every logged episode? And which of the components (representation,
node selector, proximity-ordered search) reduce acting cost or search work
when they are combined?

The study makes two claims, each with one pre-registered primary test:

1. M1 (selection, offline). With identical evidence, the joint posterior over
   complete hypergraphs ranks a recipe's candidates better than personalized
   PageRank (PPR) on the evidence-weighted star graph (recipe and fact nodes);
   the decision has four pre-stated outcomes (posterior better, PPR better,
   equivalent within 0.03 AUROC, inconclusive). Pre-stated order: posterior (a)
   > star-graph PPR (c) >= fact-graph PPR (b) > random (e), with (b) near
   chance on evidence from the brute-force policy.
2. M2 (confirmation, online, fresh worlds). In a 2x2 factorial of
   representation {hyper, pair} and selector {sample, rank}, the hypergraph
   representation lowers late-episode acting cost.

Scope: one task family (F1 of the walker study: the Stage D distribution,
goal-only observation of items, chains of depth 1-3), deterministic dynamics.

## Relation to earlier work

docs/PRIOR_ART.md, Section 8.2, records the related work. A PageRank-style
proximity is not a new object: PPR with restart probability one minus the
discount is the normalized successor representation of the same chain (M1,
M4 there), which is policy-dependent (M3). Random walks on hypergraphs with
edge-independent vertex weights reduce to walks on a graph (M9), and several
hypergraph Laplacians are those of the clique or star expansion (M10); the
clique expansion loses conjunctions, the star expansion (one node per recipe,
a factor graph, M12) does not, but a walk on it still sums over paths, whereas
conjunctive reachability needs every tail of a hyperarc (M11) and is evaluated
by AND-OR propagation such as the additive heuristic (M13). The section also
notes that statistics learned from the agent's own trajectories inherit its
policy, so evidence from the brute-force node cannot separate required from
unrequired candidates. No checked work compares a proximity score with
posterior inference over conjunctive preconditions on matched evidence; that
comparison, the twins and the factorial are what this study adds, and the
reductions above fix the expected direction.

## Protocol (frozen)

### Setting

Family F1 (`configs/walker_stage_d.yaml`): profile `unknown_prerequisites`,
goal depth 1-3, no distractor base facts, `observe_items: goal_only`, failure
probability 0, pool-mode worlds with 16 episodes per world in world-major
order, so evidence accumulates within a world and starts from the prior in the
next one. Every seed plays its own worlds (evaluation base seed + 1000 x seed),
so worlds are independent units. World namespaces: `markov_dev` (development
only), `markov_sel` (M1), `markov_conf` (M2). Ledger `runs/M-ledger.json`,
declared identically in every `configs/markov/` file: allocations `dev`
6,000, `m1` 18,475 and `m2` 77,650 (adaptive cap 102,125), reporting cap
50,500.

### Representations

Hypothesis class (both representations): for each recipe, a nonempty subset of
its public pool of six candidate base facts with at most three members (41
hypotheses), uniform prior.

`hyper`: a node of the walker's meta-graph, one hypothesis for every recipe,
scored by deterministic replay of every logged episode (`agents/walker.py`).

`pair`: a weighted star graph with one node per fact and one per recipe, an
edge (recipe r, candidate b) of weight w(r, b) and unit edges from a recipe to
its item inputs and its effect. Weights are updated online, each observation
once, in the order observed, and never revisited (`agents/markov.py`). An
observation is a craft of the goal (the only craft outcome observed):

- Success at step t: every candidate of the goal recipe absent at t is
  eliminated (w = 0). For each item input with exactly one producer recipe
  attempted before t, that producer fired by its last attempt before t, and
  its candidates absent there are eliminated too, recursively. With several
  attempted producers nothing is eliminated.
- Failure at step t: the monotone formula "the goal recipe was not eligible
  at t": the OR of its candidates absent at t and, for each item input, the
  AND over the input's attempted producers of the same formula at their last
  attempt (true when no producer was attempted, which explains the failure).
  The formula is simplified with the edges settled so far: if satisfied, no
  update. An edge whose falsity would falsify the formula is forced and
  becomes required (w = 1), and the formula is simplified again; each of the k
  open edges left in a formula that is still open receives blame 1/k.
- Class bounds on a recipe whenever one of its edges is settled: the last open
  candidate of a recipe with no required candidate is required; three required
  candidates eliminate the rest.
- Open edges: w(r, b) = (n0 p0 + s) / (n0 + s), with s the accumulated blame,
  p0 = 16/41 the prior marginal and n0 = 2.

A failure whose open edges are cleared later is not re-attributed. This is the
point under test: disjunctive evidence is summarized at once into per-edge
numbers.

Within an episode both pair cells also use per-edge marks, the same rule in
both: each failure of the current episode is taken once, in order, and while
the node leaves it unexplained (its formula is false under the node), its
highest-weight open candidate not yet in the node is marked as required for
the rest of the episode (ties: the failed recipe's own candidates first). A
recipe that already has three members swaps out its lowest-weight member that
is neither marked nor required (ties: the larger type id). There is no search
over nodes and earlier failures are not checked again, so a node refuted by
its own episode can still be chosen; such replans are counted
(`refuted_replans`, and `repeated_nodes` when the same node, compared on the
recipes of the episode's failures, recurs). So the pair representation
carries per-edge evidence across episodes and the same per-edge marks within
an episode; the hyper cells use joint consistency with every log throughout.

Personalized PageRank with restart probability a and seed distribution s on a
symmetric weight matrix W: pi = a s + (1 - a) pi P with P = D^-1 W (a node
without edges jumps to s), so pi = a s (I - (1 - a) P)^-1, the normalized
successor representation with discount 1 - a. The agents' score of candidate b
for recipe r is its star-graph PPR personalized at the recipe node r, at r's
effect fact, or at the current goal (the goal of the latest log; one score per
fact, shared by every recipe whose pool holds it). Restart 1.0 means no
propagation: the score is the pairwise weight w(r, b) itself, which is the
limit a -> 1 of the ranking for the recipe seed. M1 measures exactly these
scores (the same function, on the logs and recipes an agent would hold) and
fixes the seed and the restart probability.

### Arms

All agents share the derivation planner, beliefs and replanning of `Walker`,
keep the same episode logs and differ only in how a node is chosen at a
replan. Every episode row of a Markov arm records `choose_seconds`, the
wall-clock time spent choosing nodes.

| Arm | Representation | Selector | Node chosen at a replan |
|---|---|---|---|
| `focused_sample` | hyper | sample | focused walk to a node consistent with every log, then 50 Metropolis moves among consistent nodes (unchanged existing strategy) |
| `hyper_rank` | hyper | rank | as `focused_sample`, but the visited consistent node with the largest share of consistent single-edit neighbours (see below) |
| `pair_sample` | pair | sample | per recipe, an exact draw over the hypothesis class with P(h) proportional to the product over members b of odds(w(r, b)) / odds(p0): the pairwise weights combined with the uniform class prior; eliminated candidates excluded, required ones included; then the within-episode marks |
| `pair_rank` | pair | rank | deterministic: required candidates first, then open ones by descending star-graph score, up to k = max(1, round(sum of the recipe's weights)), at most 3; then the same within-episode marks |
| `hyper_sample_ppr` | hyper | sample | `focused_sample` whose focused edits are proposed with probability proportional to 0.1 + 0.9 q, q(add b) = s(r, b), q(remove b) = 1 - s(r, b), q(swap o for b) = (s(r, b) + 1 - s(r, o)) / 2, s the star-graph score divided by the largest in r's pool (0 when eliminated); consistent moves unchanged, so the sampling target is unchanged |
| `maximal` | (anchor) | - | the full-pool node |
| `reference` | (anchor) | - | the evaluator's optimal plan (privileged) |

`hyper_rank`. The visited consistent nodes and all their consistent
neighbours form a connected graph (two nodes adjacent when they differ by one
single-candidate edit). On it, PageRank without teleport, the stationary walk,
is proportional to degree. The score divides it by the node's number of
neighbours in the whole hypothesis class, which is not regular (10, 14 and 12
neighbours at requirement sizes 1, 2 and 3 for a pool of six), so the score is
the share of a node's neighbours that are consistent. Raw PageRank would
favour size two for reasons of class geometry alone, and teleport would add
the same mass to every node and favour small hypotheses after the division.
The score keeps a slight tilt toward size three: a node that covers more of a
disjunction has a larger consistent share. Ties go to the most recently
visited node. The arm picks a central (consensus) consistent node rather than
a sample.

Recipes without evidence are drawn from the prior by the sampling arms (with
all weights at p0 the class draw is uniform over the class), ranked from prior
weights by `pair_rank`.

Fixed a priori: n0 = 2, p0 = 16/41, the full neighbourhood for `hyper_rank`,
the within-episode mark rule, proposal floor 0.1, and the
walker's search settings (400 evaluations per walk, temperature 0.5, restart
after 80, 50 consistent moves). Fixed by M1: the star-graph PPR seed and
restart probability (`walker.markov.ppr_seed`, `walker.markov.restart`). In
`configs/markov/m2.yaml` both are null placeholders, which the agents refuse;
`python -m hypergraph_agent.walk` checks the Markov parameters of the whole
config (before `--arms` filtering) and refuses any run of it, so no arm of
`m2.yaml` can run before the M1 selection is written in; a dry run, which
interacts with nothing, still works.

### M1: selection (offline ranking benchmark)

Collection (`python -m hypergraph_agent.markov_study collect --config
configs/markov/m1_collect.yaml`): seeds 0-4, twelve `markov_sel` worlds per
seed (60 worlds), 16 episodes per world; `focused_sample` (adaptive,
allocation `m1`) and `maximal` (the brute-force policy, reporting cap) on the
same worlds. Each world's episode logs, public recipe descriptions, the
recipes each episode's task registered and (evaluator only) the hidden
requirements are written to `<run>/evidence/`.

Rankings (`python -m hypergraph_agent.markov_study rank`, which expects 36 test
worlds by default; no interaction), for every recipe attempted in the world's first k
episodes, k in {4, 8, 12, 16}, with the recipes registered by then (what an
agent holds at a replan in episode k), separately for each evidence source:

- (a) `posterior`: marginal probability under the joint posterior over complete
  hypergraphs. Computed exactly for each connected group of recipes
  (depth-first enumeration pruned by the logs, with the counts of the remaining
  recipes memoized on the assigned recipes they still share an incomplete log
  with), within a budget of 2,000,000 log checks per group; beyond it, Gibbs
  sampling among consistent nodes from four focused-walk starts (40 sweeps
  after 10 of burn-in), Rao-Blackwellized. The number of exact and sampled
  recipes is reported.
- (b) `fact_ppr`: PPR personalized at the recipe's effect on the empirical fact
  chain, the clique expansion of the fact set of every episode that reached the
  goal (initial facts, bases gathered and effects crafted up to the goal, the
  goal; pair weight 1/(n - 1) per episode).
- (c) `star_ppr`: the agents' score, star-graph PPR with the online pairwise
  weights; the goal seed uses the goal of the latest logged episode, as an
  agent at a replan in that episode does.
- (d) `andor`: AND-OR factor score. One independent belief per edge with a
  fair-coin prior and a per-recipe factor allowing 1 to 3 required candidates
  (together exactly the uniform class prior); every failure kept as a factor
  over its monotone formula, with a literal that occurs more than once (an
  item input shared by alternative producers) summed over exactly; successes
  as eliminations, except that a success with several attempted producers of
  an input is dropped (a disjunction of negations); damped loopy sum-product.
  In -log space AND adds precondition costs as h_add does and OR is a soft
  minimum (h_add's minimum at zero temperature); it shares h_add's
  independence assumption. Any goal-distance propagation on the weighted graph
  would order a recipe's candidates by their edge weights, so this score is
  the AND-OR alternative that differs from (c) in how it combines
  observations: it keeps the disjunctions, as factors.
- (e) `random`: all candidates tied.
- Diagnostic: `pair_weight` (the weights without propagation; equal to (c) at
  restart 1.0).

Metrics against the hidden requirement of each recipe: AUROC (ties count one
half) and the ranked gathering cost, the expected number of candidates
gathered in score order (ties in random order) until the requirement is
covered. A world's value is the mean over its recipes with evidence.

Selection, on the 24 worlds of seeds 0-1 only: the PPR setting, seed {recipe,
effect, goal} x restart probability {0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0},
with the largest mean per-world AUROC of (c) at k = 16 on `focused_sample`
evidence (ties within 1e-6: the larger restart, then the seed in the order
recipe, effect, goal); (b) selects its restart the same way. The selected (c)
setting is written into `configs/markov/m2.yaml` before any M2 run and
recorded as an amendment. A selection at restart 1.0 means that propagation
adds nothing beyond the raw pairwise weights; a selection at 0.05 lies at the
edge of the grid and is reported as such.

Primary contrast, on the 36 worlds of seeds 2-4 only: per-world AUROC (a)
minus AUROC (c) at the selected setting, k = 16, `focused_sample` evidence,
with a pre-stated margin of 0.03 AUROC. Decision: "posterior better" if the
lower end of the 95% world-cluster bootstrap interval is above 0; "PPR better"
if its upper end is below 0; otherwise "equivalent" if the 90% interval lies
inside (-0.03, 0.03); otherwise "inconclusive". A significant difference takes
precedence over equivalence (it is reported with the equivalence flag).
Predicted: posterior better. An inconclusive result is weak evidence either
way; the minimum detectable difference (two-sided 5%, power 80%) is
2.80 x SD / sqrt(36) of the per-world differences, reported from the observed
SD (planning value: SD 0.05, about 0.023 AUROC, below the margin; development
gave 0.045 over eight pairs of world and checkpoint).

Secondary predictions, reported with intervals on the test worlds and not used
for the decision: (c) - (b) >= 0 and (b) - (e) > 0 on
`focused_sample` evidence at k = 16; on `maximal` evidence every score at
chance (exactly 0.5 for (a), (d) and the weights, since nothing is ever
omitted; 0.5 in expectation for any score that ignores the hidden
requirement). Exploratory: whether (d) lies between (a) and (c), and the ranked
gathering costs.

Twins (constructed worlds, no interaction; `markov_study twins`, unit tests):

- Structural: one recipe {ore, fuel, sand} -> ingot against three recipes
  {ore, fuel}, {fuel, sand}, {ore, sand} -> ingot. The unweighted clique
  expansions and their PPR are identical; the star expansions differ; the
  cheapest plans need 3 and 2 base facts (h_add of the goal 4 and 3).
- Behavioural: a chain ingot -> key whose worlds A (ingot needs ore, key needs
  oil) and B (ingot needs fuel, key needs stone) are observed under one
  schedule: two brute-force episodes (identical logs, every score identical),
  an omission of ore and stone (fails in both: identical observations), and a
  probe omitting stone (succeeds in A, fails in B). Expected: after the probe
  the posterior and (d) give the edge (ingot, ore) probability 1 in A and 16/41
  in B, while the pairwise weight (0.512 in both), the star-graph PPR order
  for every seed and the fact-chain order of the ingot recipe's candidates are
  identical in A and B.

### M2: confirmation (online)

Runs: `python -m hypergraph_agent.walk --config configs/markov/m2.yaml` (the
five arms above except the anchors, allocation `m2`) and `--config
configs/markov/m2_bounds.yaml` (`maximal`, `reference`, reporting cap). Seeds
0-4, ten `markov_conf` worlds per seed (50 worlds), 16 episodes per world, 160
tasks per run. M2 is run whatever M1's outcome; M1 only fixes the PPR settings
and states the predictions.

Endpoints per world (`python -m hypergraph_agent.markov_study factorial`, which
expects 50 worlds by default): late-half cost ratio (primitive steps over optimal
steps, summed over episodes 9-16), late-half success rate, search evaluations,
evaluations per replan, and seconds per replan (run wall-clock over replans
for every arm; choice time over replans for the Markov arms). Cell values
HS = `focused_sample`, HR = `hyper_rank`, PS = `pair_sample`, PR = `pair_rank`.

- Representation main effect (hyper - pair): ((HS - PS) + (HR - PR)) / 2.
- Selector main effect (rank - sample): ((HR - HS) + (PR - PS)) / 2.
- Interaction: (HR - HS) - (PR - PS).

Primary contrast (the only one in M2): the representation main effect on
late-half cost, 95% interval. Predicted negative (hyper cheaper); confirmed if
the upper end of the interval is below 0.

Selection rule (pre-registered): a level of a factor "brings more" if choosing
it over the other level lowers late-half cost with a 97.5% interval
(Bonferroni over the two main effects) excluding 0 and lowers late-half
success by at most 2 percentage points (point estimate of the success main
effect). With no level selected the result is "no component brings more" and
there is no winning combination. Otherwise the winning combination is the cell
with the lowest mean late-half cost among the cells that use every selected
level; it is reported as confirmed only if its paired 95% interval against the
runner-up (the next cheapest cell) lies below 0, and as descriptive otherwise.
With an incomplete run set the selected levels, the winning cell and its
comparison are null.

Secondary: the counts of residual refuted replans and repeated nodes in the
pair cells. `hyper_sample_ppr` - `focused_sample`, paired per world, on search
evaluations over all 16 episodes (predicted lower), evaluations per replan,
late-half cost and success (predicted unchanged), with seconds per replan
reported per arm (the PPR solve is not a hypothesis evaluation). Anchors are
reported per arm.

### Statistics

The unit is the world; worlds are independent across seeds, and a seed only
labels the agents' random streams. Every contrast is a paired per-world
difference; its mean gets a percentile world-cluster bootstrap interval
(10,000 resamples, seed 20261008, 95% unless stated). One primary contrast
per stage; everything else is descriptive.

Runs: one per arm and seed, the earliest. A run that did not complete is
replaced by the earliest later completed run with the same config hash, commit
and loaded code (runs are deterministic); both are listed, as are ignored
repeats. A verdict needs every run it uses to be complete and all expected
worlds (36 test worlds in M1, 50 in M2); otherwise it is None.

### Budget

Worst cases are the sums of the task budgets of the streams (generator only,
dry runs); no episode can exceed its budget, and every per-run cap is the
largest stream sum, so no run can be cut short.

| Stage | Config | Arms (kind) | Runs | Per-run cap | Worst case |
|---|---|---|---|---|---|
| Development | `dev.yaml`, `dev_collect.yaml`, one variant of `dev.yaml` | all (adaptive `dev`; `maximal` reporting) | 13 | 620 | allocation 6,000 |
| M1 | `m1_collect.yaml` | `focused_sample` (adaptive `m1`), `maximal` (reporting) | 5 + 5 | 3,750 | 18,475 adaptive, 18,475 reporting |
| M2 | `m2.yaml` | five arms (adaptive `m2`) | 25 | 3,148 | 77,650 adaptive |
| M2 anchors | `m2_bounds.yaml` | `maximal`, `reference` (reporting) | 10 | 3,148 | 31,060 reporting |

Stream budget sums per seed: M1 3,679 / 3,691 / 3,688 / 3,667 / 3,750 (total
18,475); M2 3,093 / 3,014 / 3,145 / 3,130 / 3,148 (total 15,530). Reporting
worst case 49,535 plus 427 used in development, within the 50,500 cap.

The ledger file `runs/M-ledger.json` was created by the first development run
with the earlier declaration; before any measurement run its caps were raised
to the declaration above by a logged amendment in the ledger (`m1` 6,180 to
18,475; adaptive cap 89,830 to 102,125; reporting cap 38,500 to 50,500;
development usage unchanged).

### Stopping rules

No interim analysis and no early stopping for results. A run stops only at the
end of its stream or at its cap (never binding by construction). M1 is analyzed
once, after all ten collection runs; its PPR selection is written into the M2
config before the first M2 run. A run that fails for a software fault is
handled by the rerun rule above and recorded as an amendment.

### Amendments

- MA1 (2026-10-08, after M1, before any M2 run). The frozen selection rule,
  applied to the 24 selection worlds (seeds 0-1) of the M1 collection, chose
  star-graph PPR with seed `recipe` and restart probability 1.0 (and, for the
  fact chain, seed `effect` and restart 1.0). Both are written into
  `configs/markov/m2.yaml`; nothing else changes. Restart 1.0 is the edge of the
  grid where the score equals the raw pairwise weights, so, as pre-stated,
  propagation adds nothing beyond the raw weights on these worlds; `pair_rank`
  and the proposals of `hyper_sample_ppr` therefore rank by the pairwise
  weights.
- The commit of MA1 left two tests failing that asserted the pre-MA1
  placeholders in `configs/markov/m2.yaml`; the next commit changed those tests
  only (they now blank the two settings themselves). No code or config used by
  a measurement run changed.

## Implementation

- `src/hypergraph_agent/agents/markov.py`: observation formulas
  (`EpisodeReplay`), the online pairwise learner (`PairwiseEvidence`), star
  graph and scores (`personalized_pagerank`, `star_graph`, `candidate_ppr`,
  `star_scores`, `ppr_scores`), the agents (`HyperRank` with
  `single_edit_degrees`, `PairAgent` with `draw` and `marks`, `PPRProposal`
  with `HyperSamplePPR`) and `make_markov_agent`, imported by
  `evaluation.walkers.make_walker` only when a Markov strategy is asked for;
  agent parameters come from `walker.markov`.
- `src/hypergraph_agent/evaluation/markov_study.py`: evidence collection
  (`collect`; `maximal` runs with its logs kept), run selection
  (`select_runs`), the M1 rankings, metrics and decision (`posterior_marginals`,
  `andor_marginals`, `fact_chain`, `rank_benchmark`, `m1_decision`), the twins,
  and the M2 analysis (`factorial`).
- `src/hypergraph_agent/walk.py`: the guard that checks `walker.markov` on the
  whole config before any run.
- `src/hypergraph_agent/markov_study.py`: command line (`collect`, `rank`,
  `twins`, `factorial`).
- `configs/markov/`: `dev.yaml`, `dev_collect.yaml`, `m1_collect.yaml`,
  `m2.yaml`, `m2_bounds.yaml`.
- `tests/test_markov.py` (37 tests): formulas, the pairwise update rule
  (including a forced edge with the rest of its formula still blamed), PPR as
  the normalized successor representation and restart 1.0 as the edge weight,
  single-edit degrees and the normalization that removes the class's size
  preference, replay agreement with `simulate`, the exact posterior against
  brute-force enumeration, Gibbs against the exact posterior, exact factor
  probabilities with repeated literals, the AND-OR prior and explaining away,
  M1 scores equal to the agents' scores, the M1 decision with its margin,
  metrics with ties, both twins, every arm playing tasks, the within-episode
  marks (one rule shared by both pair cells, the swap at three members, the
  reset per episode, the counters), the class draw of `pair_sample`, the PPR
  proposal, `LoggedMaximal` matching `maximal`, the collection and benchmark
  pipeline, the rerun rule, the factorial arithmetic, selection rule and null
  selection when incomplete, the M2 config refusing to run until M1 is written
  in (including through `walk.main` with `--arms`), the shared ledger
  declaration and line lengths.

Development record (allocation `dev`, namespace `markov_dev`, seed 0, two
worlds x 16 episodes; outcomes are not results). Thirteen runs: the five M2
arms (`dev.yaml`, stamp 20261008-103333), the M1 collection pair
(`dev_collect.yaml`), `pair_rank` and `hyper_sample_ppr` with restart 0.9 and
the recipe seed (a variant of `dev.yaml` whose config is stored in its run
directories), and the four revised arms after the first review (`dev.yaml`,
stamp 20261008-122545; `pair_rank` then used the node search later replaced by
the marks, and both pair cells were checked again with the marks in memory, as
in the unit tests, without interactions charged). Ledger after development:
5,079 adaptive (`dev`) and 427
reporting interactions. Runtime of the revised arms (one process, CPU), per
task and per replan (run wall-clock; choice time in brackets): `hyper_rank`
0.63 s and 0.25 s (0.25 s; about 1,700 hypothesis evaluations per replan for
the full consistent neighbourhood, so no restriction was needed),
`hyper_sample_ppr` 0.062 s and 0.023 s (0.021 s; 51 evaluations per replan),
`pair_rank` 0.031 s and 0.010 s (0.008 s), `pair_sample` 0.019 s and 0.007 s
(0.005 s); `focused_sample` 0.023 s per replan (52 evaluations). The offline
benchmark took about two minutes per two worlds and both sources. Decisions
that came from development and the independent review:

1. `pair_rank` repeated a node its own episode had refuted until the task
   deadline: observed in development run
   `markov-dev-pair_rank-default-s0-20261008-103333` (effect seed, restart
   0.15); the first review counted 227 repeats with the effect seed and 599
   with the goal seed at restart 0.15, 24 with the recipe seed at 0.15 and none
   with the recipe seed at restarts 0.5 to 1.0. A node search that removed
   them was rejected in the second review because it gave `pair_rank` alone
   joint within-episode reasoning (mixing the pair level and confounding the
   selector effect); both pair cells now use the per-edge marks above.
   Residual repeats remain and are counted: the reviewer measured 10, 0 and 0
   repeated nodes for the effect, goal and recipe seeds at restart 0.15 on 8
   worlds; an in-memory check on 8 other worlds x 12 tasks gave 0, 16 and 0
   for `pair_rank` (2, 18 and 0 refuted replans) and none for `pair_sample`.
2. The PPR seed is a selected parameter (recipe, effect or goal) together with
   the restart, and M1 measures exactly the agents' scores.
3. Gibbs estimates of the posterior (default settings) were within 0.04 of the
   exact marginals and within 0.025 in world AUROC, and exact computation is
   cheap with memoization, so (a) is exact whenever it fits the work budget
   (all recipes in development).
4. `hyper_rank` divides PageRank by the class degree (the share of consistent
   neighbours); `pair_sample` draws over the class; M1 is enlarged, split into
   selection and test worlds and decided with an equivalence margin; M2 has
   one primary contrast and a Bonferroni-corrected selection rule.

## Results

Evidence: `artifacts/markov/m1_ranking.{md,json}` and
`artifacts/markov/m2_factorial.{md,json}`, produced by the frozen analyses run
from the snapshots of the freeze commit (M1) and of the MA1 commit (M2).
Re-runs of both analyses from the same snapshots into a scratch folder
reproduced the artifacts exactly (every field except the elapsed time), and
the twins were recomputed from the same code. Intervals are 95%
world-cluster bootstrap intervals (10,000 resamples) unless stated.

### Runs and interactions

M1: ten collection runs (`focused_sample` and `maximal`, seeds 0-4, twelve
`markov_sel` worlds per seed, 192 tasks per run), all completed; no rerun. M2:
25 factorial runs and 10 anchor runs (seeds 0-4, ten `markov_conf` worlds per
seed, 160 tasks per run), all completed; 50 paired worlds; no rerun and no
ignored run.

Interactions of the measurement runs (`runs/M-ledger.json` minus development):
59,624 adaptive and 27,089 reporting. Adaptive: M1 `focused_sample` 10,866
(allocation `m1`, 18,475); M2 48,758 (allocation `m2`, 77,650):
`focused_sample` 9,385, `hyper_rank` 9,526, `pair_sample` 9,914, `pair_rank`
10,352, `hyper_sample_ppr` 9,581. Reporting: M1 `maximal` 11,535; M2 `maximal`
9,994 and `reference` 5,560. With development (5,079 adaptive, 427
reporting) the ledger holds 64,703 of 102,125 adaptive and 27,516 of 50,500
reporting interactions.

### M1: ranking benchmark (offline)

The posterior was exact for every recipe at 16 episodes (164 recipe-world
pairs on the selection worlds, 258 on the test worlds); at 4 to 12 episodes 6
to 30 recipe-world pairs per split and checkpoint were sampled. The AND-OR
factor score did not converge in 2 selection worlds at 12 episodes and in 2 at
16 episodes.

Selection (24 worlds of seeds 0-1, `focused_sample` evidence, 16 episodes;
mean per-world AUROC of the star-graph score):

| Seed | 0.05 | 0.15 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 (no propagation) |
|---|---|---|---|---|---|---|---|
| recipe | 0.778 | 0.821 | 0.847 | 0.861 | 0.867 | 0.870 | 0.879 |
| effect | 0.730 | 0.768 | 0.789 | 0.809 | 0.813 | 0.813 | 0.879 |
| goal | 0.629 | 0.607 | 0.605 | 0.615 | 0.611 | 0.610 | 0.879 |

The fact chain gave 0.629, 0.668, 0.689, 0.696, 0.696, 0.697 and 0.753 at the
same restarts. Both selections fell at restart 1.0, the edge of the grid where
the score is the raw edge weight (the three seeds tie there and the tie rule
picks `recipe`); every setting that propagates ranked worse than the raw
weights, and the goal seed by far the worst. As pre-stated, propagation added
nothing beyond the raw weights; in every table below the star-graph score
equals the pairwise weight (their difference is 0.000 [0.000, 0.000]
everywhere).

Primary (36 test worlds of seeds 2-4, 16 episodes, `focused_sample`
evidence): AUROC posterior minus AUROC star-graph score 0.009 [-0.004, 0.023];
90% interval [-0.002, 0.021], inside the margin (-0.03, 0.03). Decision:
**equivalent**. Observed SD of the per-world differences 0.044, minimum
detectable difference 0.020. The predicted outcome, "posterior better", was
not confirmed.

Test worlds, `focused_sample` evidence, AUROC by checkpoint:

| Episodes | (a) posterior | (d) AND-OR | (c) star-graph score | (b) fact chain | (e) random | (c) - (a) |
|---|---|---|---|---|---|---|
| 4 | 0.737 [0.712, 0.764] | 0.732 [0.704, 0.759] | 0.702 [0.672, 0.733] | 0.629 [0.598, 0.661] | 0.500 | -0.036 [-0.054, -0.017] |
| 8 | 0.783 [0.757, 0.810] | 0.775 [0.747, 0.805] | 0.771 [0.743, 0.800] | 0.665 [0.637, 0.696] | 0.500 | -0.012 [-0.027, 0.004] |
| 12 | 0.818 [0.800, 0.837] | 0.815 [0.796, 0.835] | 0.809 [0.789, 0.830] | 0.692 [0.667, 0.718] | 0.500 | -0.009 [-0.024, 0.005] |
| 16 | 0.864 [0.850, 0.880] | 0.864 [0.847, 0.882] | 0.855 [0.835, 0.875] | 0.738 [0.713, 0.763] | 0.500 | -0.009 [-0.023, 0.004] |

Secondary predictions (test worlds, 16 episodes):

- (c) - (b) >= 0: 0.116 [0.095, 0.138]; holds.
- (b) - (e) > 0: 0.238 [0.213, 0.263]; holds.
- Brute-force (`maximal`) evidence at chance: the posterior, the AND-OR score,
  the star-graph score and the weights are exactly 0.500 at every checkpoint
  in both splits; the fact chain gives 0.497 to 0.498 on the test worlds (its
  difference to random at 16 episodes -0.003 [-0.015, 0.009]) and 0.502 to
  0.514 on the selection worlds, where one checkpoint departs from chance
  (8 episodes: 0.014 [0.002, 0.027]); holds on the test worlds.

Exploratory: the AND-OR score equals the posterior at 16 episodes (posterior
minus AND-OR 0.000 [-0.006, 0.007]) and is not distinguishable from the
star-graph score there (AND-OR minus star 0.009 [-0.003, 0.022]); at 4
episodes it is above it (0.030 [0.015, 0.044]), so (d) lies between (a) and (c)
early and coincides with (a) late. Ranked gathering cost at 16 episodes (test
worlds; candidates gathered until the requirement is covered): posterior 2.88
[2.76, 3.01], AND-OR 2.88 [2.74, 3.01], star-graph score 2.94 [2.80, 3.09],
fact chain 3.36 [3.22, 3.50], random 4.60 [4.53, 4.67]; on brute-force
evidence 4.61 for every score (fact chain 4.62). On the selection worlds,
which chose the setting, the posterior led the star-graph score at 16 episodes
(star minus posterior -0.021 [-0.036, -0.005]); the test worlds decide.

Twins (constructed worlds): as predicted. Structural: equal clique expansions
and clique PPR, different star expansions, cheapest plans of 5 and 4 steps (3
and 2 base facts), h_add of the goal 4 and 3. Behavioural: identical scores
after the brute-force episodes and after the shared failure; after the probe
the posterior and the AND-OR score give the edge (ingot, ore) 1.000 in world A
and 0.390 in world B, while the pairwise weight is 0.512 in both and the
rankings of the ingot recipe's candidates by the star-graph score (every
seed) and by the fact chain are identical in A and B.

### M2: factorial (online)

Per arm, 50 worlds:

| Arm | Late cost ratio | Late success | All episodes | Evaluations / world | Evaluations / replan | Seconds / replan (run) | Choice seconds / replan |
|---|---|---|---|---|---|---|---|
| `focused_sample` (hyper, sample) | 1.46 | 1.00 | 1.69 | 2,071 | 52.2 | 0.0200 | - |
| `hyper_rank` (hyper, rank) | 1.49 | 0.98 | 1.72 | 69,998 | 1,737.2 | 0.3592 | 0.3436 |
| `pair_sample` (pair, sample) | 1.56 | 0.99 | 1.79 | 0 | 0.0 | 0.0074 | 0.0052 |
| `pair_rank` (pair, rank) | 1.68 | 0.99 | 1.87 | 0 | 0.0 | 0.0056 | 0.0035 |
| `hyper_sample_ppr` | 1.52 | 0.98 | 1.73 | 2,091 | 50.7 | 0.0231 | 0.0205 |
| `maximal` (anchor) | 1.83 | 1.00 | 1.82 | 0 | 0.0 | 0.0080 | - |
| `reference` (anchor) | 1.00 | 1.00 | 1.00 | 0 | 0.0 | - | - |

With the M1 selection (restart 1.0), `pair_rank` ranks by the pairwise weights
and `hyper_sample_ppr` weights its proposals by them (MA1).

Primary: representation main effect (hyper minus pair) on late-half cost
-0.146 [-0.194, -0.097]: **the hypergraph representation lowers late cost**
(confirmed); its effect on late success 0.000 [-0.009, 0.009].

Other effects: selector (rank minus sample) +0.077 [0.034, 0.118] on late
cost, -0.010 [-0.019, -0.003] on late success; interaction -0.079
[-0.172, 0.011] on late cost, -0.010 [-0.030, 0.007] on late success.

Selection rule (97.5% intervals): representation -0.146 [-0.202, -0.090],
success effect 0.000, so hyper brings more; selector +0.077 [0.028, 0.124],
which excludes 0 on the side where sampling is cheaper, and choosing sampling
changes late success by +0.010, so sample brings more. Selected levels: hyper
and sample. Winning cell `focused_sample` (1.46), descriptive: against the
runner-up `hyper_rank` (1.49) its paired difference is -0.037
[-0.094, 0.020].

Secondary, pair cells (episode rows of the runs): `pair_sample` made 98
within-episode marks, planned on a node refuted by its own episode in 9 of
2,068 replans and repeated no node; `pair_rank` made 4 marks, with 0 refuted
replans and 0 repeated nodes in 1,810 replans. Descriptive, not
pre-registered: the pair cells planned on a node that contradicted earlier
logged evidence in 1,048 of 2,068 replans (`pair_sample`) and 532 of 1,810
(`pair_rank`), against 0 of 1,979 for `focused_sample`, 2 of 1,970 for
`hyper_rank` and 0 of 2,060 for `hyper_sample_ppr`.

Secondary, `hyper_sample_ppr` minus `focused_sample` (50 worlds): search
evaluations per world +19.4 [-77.4, 113.6] (predicted lower: not confirmed);
per replan -1.53 [-2.18, -0.97]; late cost +0.068 [0.013, 0.126] and late
success -0.015 [-0.028, -0.005] (predicted unchanged: both changed, for the
worse). Seconds per replan 0.0231 against 0.0200 for `focused_sample` (choice
time 0.0205).

Anchors: the brute-force plan (`maximal`) costs 1.83 in the late half with
every task solved; the reference is 1.00 by construction. Both pair cells
stay below the brute-force plan (1.56 and 1.68), as do the hyper cells (1.46,
1.49 and 1.52). `hyper_rank` spent 69,998 hypothesis evaluations per world
(1,737.2 per replan, against 2,071 and 52.2 for `focused_sample`) and 0.3592 s
per replan (against 0.0200 s), without lowering cost.

## Verdict and limits

M1. The predicted "posterior better" was not confirmed: on the 36 test worlds
the joint posterior and the star-graph score are equivalent within the
pre-stated margin at 16 episodes (0.009 [-0.004, 0.023]; 90% [-0.002, 0.021]).
The posterior is ahead early (4 episodes: by 0.036 [0.017, 0.054] AUROC) and the
gap closes as evidence accumulates. PageRank-style proximity itself added
nothing: on the selection worlds every setting that propagates over the star
graph or the fact chain ranked worse than its own edge weights, so the
selected "PPR" is the raw pairwise weight. What ranks nearly as well as the
posterior is therefore the per-edge evidence (exact eliminations, forced
edges and 1/k blame), not the walk. The audit's order holds for the other
pairs: star-graph weights above the fact chain (0.116 [0.095, 0.138]), the fact
chain above random (0.238 [0.213, 0.263]), and brute-force evidence at chance
for every method on the test worlds. The twins came out as constructed: the
pairwise weights and the clique chain do not separate worlds that the joint
posterior and the AND-OR factor score separate.

M2. The hypergraph representation lowers late-half cost by 0.146
[-0.194, -0.097] steps per optimal step without losing success: confirmed.
Ranking instead of sampling raises cost (+0.077 [0.034, 0.118]) and slightly
lowers success. Under the pre-registered rule the components that bring more
are the hypergraph representation and posterior sampling, the combination
already in `focused_sample`, which is the cheapest cell (descriptively; its
lead over `hyper_rank` is -0.037 [-0.094, 0.020]). The Markov components do not
bring more: the pairwise representation is dearer on average over the two
selectors (cell means 1.56 and 1.68 against 1.46 and 1.49), ranking is dearer
than sampling on average over the two representations (ranking by the
pairwise weights in the pair cells, the consistent-share centrality in the
hyper cells; the interaction, -0.079 [-0.172, 0.011], does not separate the
two), and proximity-ordered proposals (`hyper_sample_ppr`) neither saved
evaluations per world nor kept acting cost, so the niche named by the audit,
proposals that save search without changing acting, is not supported. The
centrality choice also costs
about 34 times the hypothesis evaluations of `focused_sample` per world and
18 times the time per replan. Near-equal offline rankings did not translate
into equal acting: a pairwise agent cannot hold earlier disjunctive evidence
and planned on nodes that the logs already contradicted in 51% (`pair_sample`)
and 29% (`pair_rank`) of its replans. The pairwise agents nonetheless remain
cheaper than the brute-force plan (1.56 and 1.68 against 1.83, descriptive):
per-edge evidence is useful, joint evidence more so.

Limits. One task family (F1: goal-only observation, chains of depth 1-3,
deterministic dynamics); other families, noise and deeper chains are untested.
Every agent and the posterior use the uniform prior over the 41-hypothesis
class, while the generator draws requirement sizes with weights 0.2, 0.5 and
0.3. One pairwise update rule (online, n0 = 2, blame 1/k, the within-episode
marks) stands for the pair representation; other per-edge rules, and learned
chains of other kinds, could rank or act differently. The fact chain is one
clique construction (co-occurrence in episodes that reached the goal). The
restart grid ends at 1.0, where both selections fell, so the result is that
no propagation within the grid helped, not that none could. The equivalence
in M1 is relative to a margin of 0.03 AUROC with a minimum detectable
difference of 0.020 on 36 worlds, and on the selection split the posterior
was ahead (star minus posterior -0.021 [-0.036, -0.005]). The rank selector
is two specific rules, times per replan depend on the machine, and the counts
of refuted replans and of nodes contradicting the logs come from the episode
rows, not from a contrast with an interval.
