# Experiment plan (development protocol)

Frozen: 2026-10-07, before any pilot of the session ledger was run and before
any held-out evaluation. This is a local protocol, not an external
preregistration. Later changes are appended under "Amendments" with a date;
nothing above that section is edited after the freeze.

## 1. Question and scope

Research question (a hypothesis, not a claim): from public interaction
evidence, can an agent learn and revise a compact library of closed-loop
skills and conjunctive prerequisite relations, and does that improve transfer
to longer compositions beyond topology adaptation or skill discovery alone?

The development pilots below test whether the mechanisms run and whether any
learning occurs. With two seeds per contrast they cannot support comparative
claims; they exist to size and debug the full study.

## 2. Phases, arms and primary contrasts

| Phase | Information regime | Arms | Primary contrast | Secondary |
|---|---|---|---|---|
| P1 | `known_structure`, fresh world per task, deterministic | `set` (token attention), `incidence` (role-typed message passing, memory concatenated), `gated` (same plus memory/goal gate) | `gated - set` | `gated - incidence` (gate only) |
| P2 | `unknown_prerequisites`, 8 persistent training worlds, deterministic | `evidence_set` (no inference features), `fixed_supergraph` (all candidate edges + posterior features, structure frozen), `adaptive_incidence` (posterior credible-set edges + same features, structure revised) | `adaptive_incidence - fixed_supergraph` | `fixed_supergraph - evidence_set` |
| P3 | `unknown_prerequisites`, one shared world per seed block, failure probability 0.1 (public), random initial inventories | forked from one pretraining checkpoint: F0 fixed library / frozen structure, F1 fixed / revised, G0 growing / frozen, G1 growing / revised | `G1 - G0` on held-out longer compositions | `F1 - F0`, `G0 - F0`, exploratory interaction `(G1 - G0) - (F1 - F0)` |

The three phases isolate mechanisms; their numbers are not comparable with one
another.

## 3. Endpoints and evaluation sets

The endpoint is the success rate under the fixed external reward (1 on
successful submission, else 0) and the task budget computed from public counts.
Evaluation tasks use base seed 10000 for every arm and seed (paired design).
Policies are evaluated stochastically (sampling), the same way for every arm.

| Phase | Track | Evaluation set |
|---|---|---|
| P1 | frozen | namespace `val`, 30 tasks, depths 1-4 (held-out tasks from the training distribution) |
| P2 | within-task inference | namespace `val`, 5 unseen worlds x 6 consecutive episodes; beliefs start from the prior in each world and update between episodes |
| P3 | frozen | namespace `test`, 20 tasks with goals at levels 5-6 (longer than every training goal, levels 1-3); secondary: namespace `val`, 20 tasks at levels 2-4 |

Secondary metrics: primitive episode length (all episodes and successes),
manager decisions, no-op or invalid attempts, skill calls, failure statuses,
reference-plan ratio on exact references, topology edits, proposals,
admissions and rejections with reasons, all-in and physical interaction costs.

## 4. Budgets

All commands share `runs/ledger.json`. Allocations (adaptive primitive
interactions) after amendment A1 of `docs/DECISIONS.md`, applied before any
pilot ran:

| Allocation | Interactions |
|---|---:|
| Pre-plan development and tuning (already spent, recorded) | 37,000 |
| Smoke (P1 encoders, P2 path) | 2,500 |
| Shallow learning diagnostic | 4,600 |
| P1: 3 encoders x 2 seeds x 4,500 | 27,000 |
| P2: 3 variants x 2 seeds x 4,500 | 27,000 |
| P3: 2 pretraining prefixes x 9,000 + 4 arms x 2 seeds x 5,000 | 58,000 |
| Unallocated reserve | 3,900 |
| **Session cap** | **160,000** |

Reporting evaluation: 3,452 already spent during development; the remaining
16,548 are capped per run (P1 400, P2 500, P3 1,100 per arm, diagnostic 400
plus 400 for the random-agent reference). A run whose cap is reached stops and
records `interaction_cap`.

P3 per-arm all-in cost = 9,000 (shared prefix, charged logically to every arm)
+ up to 5,000 after the fork. Both the physical session ledger and the logical
per-arm cost are reported.

## 5. Fixed hyperparameters

Adam, learning rate 1e-3, 8 PPO epochs per batch, minibatches of 8 episodes,
256-step batches, clip 0.2, value coefficient 0.5, entropy coefficient 0.01,
gradient clipping 0.5, gamma 1 (finite-horizon success objective), GAE
lambda 0.95 per primitive step, no advantage normalization, model width 64,
3 weight-tied message-passing rounds, 2 attention layers. These were chosen on
depth-1 development tasks with the gated encoder only (disclosed in A1); every
arm uses them unchanged.

P3 discovery (diagnostic admission profile): at most 8 proposals, 3 trained
and 2 admitted per round; 1,200 practice interactions and 10 validation
episodes per trained candidate; 12-step practice timeout; admission requires at
least 8 validation episodes, success at least 0.5 and a mean of at least 2
primitive steps on successes. Admitted skills are `provisional`.

## 6. Required learning diagnostics and failure criteria

1. Shallow baseline learns: the `set` encoder trained on depth-1 tasks has a
   higher evaluation success than the random agent on the same 30 tasks.
   Failure is recorded if it does not.
2. Evidence-driven topology revision occurs: at least one
   `structural_incidence_edit` with evidence ids is logged in a P2
   `adaptive_incidence` run. Failure is recorded otherwise.
3. Learned skill admitted and used: in P3 pretraining at least one candidate
   with a trained (non-scripted) controller passes validation, is admitted, and
   is later invoked by the trained manager. If no candidate passes, the failure
   is recorded; thresholds are not lowered and no scripted skill is inserted.

## 7. Statistics

Per-seed success rates are reported for every run. Contrasts are paired by
seed block with a percentile seed-block bootstrap that resamples tasks within
blocks. With two blocks the intervals are not interpretable and no direction
is claimed. The full study (`configs/study.yaml`, not launched) uses ten seed
blocks for the primary contrast; that number is a starting design, to be
revised after an assessment of development variability.

## 8. Not run in this session

`configs/study.yaml`, `configs/ablations/*` and the transfer configs are
runnable but not launched. The continual-adaptation track is not implemented.

## Addendum W: hypergraph-walker study (2026-10-07)

A second, separately budgeted study: agents that move on a graph whose nodes
are complete dependency hypergraphs and whose edges are single-incidence edits
(docs/METHODS.md, Section 11). Ledger `runs/walker-ledger.json`: 40,000
interactions in total (32,000 adaptive, 8,000 reporting), of which 4,377
adaptive were spent on plumbing checks before this addendum was written and
are recorded as `pre_plan_development`.

### W-A: Stage A (frozen before any Stage A run)

Question: does choosing a node of the hypergraph graph from public evidence and
planning on it solve the pilot tasks with fewer interactions than the
brute-force node, and how does that compare with the pilot policies?

| Part | Tasks | Arms | Seeds |
|---|---|---|---|
| P2 tasks | the P2 pilot's evaluation set (5 unseen worlds x 6 episodes, beliefs from the prior per world) at budget factor 1.0 (as in the pilot) and 0.6 | `sample`, `optimistic`, `local_focused`; bounds `maximal`, `reference` (privileged) | walkers 0, 1, 2; bounds 0 |
| P3 worlds | the P3 pilot's shared world per seed block (failure probability 0.1), 1,500 interactions on training tasks (levels 1-3), then the pilot's 20 test tasks (levels 5-6) | `sample`, `optimistic`, `maximal`, `reference` | 0, 1 |
| P1 tasks | the P1 pilot's evaluation set (known structure) | planner on the supplied structure | 0 |

Endpoints: success rate; cost ratio = total primitive steps / total optimal
steps over all episodes of a run; the cost ratio by episode index within a
world. Primary contrast: `optimistic` versus `maximal` on the cost ratio of
the P2 tasks at budget factor 1.0. Secondary: `sample` versus `optimistic`;
`local_focused` (a local walk on the meta-graph) versus `optimistic` (exact
factorized inference), which asks whether the walk recovers what exact
inference finds when the space factorizes. Walkers are planners on a
hypothesized structure, not learned policies, so the comparison with the pilot
policies is across agent classes and is reported descriptively. Tasks are
fixed and shared by all arms; worlds are the only independent units (5 for P2,
2 for P3), so no interval is interpreted.

Failure criteria: if `optimistic` does not reach a lower cost ratio than
`maximal` at budget factor 1.0, the claim "walking the hypergraph graph saves
interactions on these tasks" fails for Stage A.

### W-B: Stage B (frozen after Stage A, before any Stage B run)

Profile: hidden prerequisites with intermediate items unobserved
(`observe_items: goal_only`), deterministic, persistent worlds. Hypotheses no
longer factorize by recipe, and the joint space of a world's attempted recipes
(about 41 to the power 6 after a few episodes) cannot be enumerated, so the
`exact` arm is not run (it would silently fall back to the focused walk);
exactness is covered by unit tests on small evidence only.

Per seed (0, 1): a `local_focused` collector plays 8 training worlds x 8
episodes (at most 1,500 interactions, charged logically to the learned arm);
the edit policy is trained on internal search problems from that evidence
(1,500 searches, at most 400 evaluations each); then `local_uniform`,
`local_focused` and `learned` each play 5 unseen worlds x 8 episodes (40 tasks,
budget factor 1.0). Bounds `maximal` and `reference` play the same tasks once.

Primary contrast: offline, on the held-out evidence of the `local_focused`
arm, paired start nodes (5 per problem): mean hypothesis evaluations to a
consistent node, `learned` versus `local_focused`. Secondary: the same for
`local_focused` versus `local_uniform`; online success and cost ratio of the
three walkers against the bounds. Failure criterion: if `learned` does not need
fewer evaluations than `local_focused`, the claim "a learned walk improves on
the focused heuristic" fails. The online contrasts are descriptive (5 worlds,
2 seeds).

### W-C: Stage C (frozen after Stage B, before any Stage C run)

Motivation from the recorded results: in Stage A, drawing a node from the
posterior beat taking the cheapest one; in Stage B every walker planned on the
first consistent node it reached (effectively the cheapest-guess rule), and the
learned policy's advantage was removing long failed searches.

Arms, on Stage B's 40 tasks and seeds 0 and 1, allocation `walker_b` (at most
800 interactions per run):

- `local_focused`: reproduction control. It must reproduce Stage B's per-seed
  success and cost ratio exactly; if it does, Stage B's recorded `learned` and
  `maximal` results serve as same-code baselines, otherwise only comparisons
  within Stage C are reported.
- `focused_sample`: the focused walk to a consistent node, then 50 Metropolis
  moves restricted to consistent nodes (uniform proposals with the
  neighbourhood-size correction); recipes without evidence redrawn from the
  prior at every replan.
- `learned_sample`: the same with the learned walk, reusing Stage B's trained
  edit policy of the same seed (no retraining, no new evidence collection; the
  original collection cost is charged logically).

The 50 moves were fixed before any run and are not tuned.

Primary contrast: `learned_sample` versus `maximal` (Stage B, 1.82) on the cost
ratio. It passes if `learned_sample` has a lower cost ratio than `maximal` in
both seeds. Secondary: `focused_sample` versus `local_focused`, `learned_sample`
versus Stage B `learned` (the effect of sampling), and `learned_sample` versus
`focused_sample`. Descriptive only (5 worlds, 2 seeds).

## Amendments

- 2026-10-07, A2 (post hoc, after the shallow diagnostic and P1 were
  evaluated): one additional shallow diagnostic with the `gated` encoder on the
  same evaluation tasks, funded by 3,400 unallocated interactions. Reported as
  post hoc; the failure of diagnostic 1 stands. Details in docs/DECISIONS.md.
- 2026-10-07, A3: run provenance is recorded once per process (code actually
  executed); earlier pilot manifests are interpreted as described in
  docs/DECISIONS.md.
