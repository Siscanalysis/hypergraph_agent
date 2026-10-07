# Results (development session of 2026-10-07)

Everything below comes from recorded runs; per-run manifests, configs, events
and evaluation episodes are in `artifacts/runs/`, contrasts and figures in
`artifacts/summary.json` and `artifacts/*.png`. Raw episode logs and
checkpoints stay outside the repository. With two seed blocks per contrast
**no comparative claim is made**: the pilots test mechanisms and size the
study.

## 1. Status

| Phase | Specified | Implemented | Unit tested | Smoke run | Pilot run | Scientifically evaluated |
|---|---|---|---|---|---|---|
| P1 fixed structure | yes | yes | yes | yes | yes (no learning detected) | no |
| P2 topology inference | yes | yes | yes | yes | yes | no |
| P3 skill discovery | yes | yes | yes | yes (before the protocol freeze) | yes | no |

Not implemented: continual-adaptation track, within-episode hard routing,
relation-local memory, goal-conditioned primitive-policy control,
library-transplant diagnostic, credible-set-mask-as-feature control, learned
proposals or termination (docs/DECISIONS.md, docs/ROADMAP.md).

Test suite: 144 deterministic tests pass (`python -m pytest -q`), covering the
items listed in docs/METHODS.md (environment semantics, information boundary,
equivariance and padding, incidence and hypergraph equivalence, topology
updates and rollback, executor semantics, library lifecycle, duration-aware
returns, likelihood reconstruction and stale-batch refusal, budget accounting,
transfer bindings, statistics fixtures, installation hygiene, walker replay
semantics against the environment, meta-graph neighbourhoods and searches).

## 2. Interaction budget

Session ledger (`python -m hypergraph_agent.ledger show`):

| Allocation | Adaptive primitive interactions |
|---|---:|
| Pre-plan development and tuning (outside the ledger, entered afterwards; DECISIONS A1) | 37,000 |
| Smoke | 2,400 |
| Diagnostics (`set` 4,000; post-hoc `gated` 4,000, A2) | 8,000 |
| P1 pilot | 27,000 |
| P2 pilot | 27,000 |
| P3 pilot (physical: 2 x 9,000 pretraining + 8 x 5,000 arms) | 58,000 |
| **Total / cap** | **159,400 / 160,000** |

Reporting evaluation: 16,066 of 20,000 (3,452 of them during pre-plan
development). Logical P3 cost per arm: 14,000 (9,000 shared prefix + 5,000).
Several evaluations stopped before their nominal task count when their
reporting cap was reached (P1: 20 of 30 tasks; P2: 27-28 of 30; diagnostic
`set`: 26 of 30); the stopping rule is the cap, identical for every arm.

## 3. Required learning diagnostics

**3.1 Shallow baseline learns: FAILED as pre-registered.** The `set` encoder,
trained for 4,000 interactions on depth-1 tasks, solved 1 of 26 evaluation
tasks; a random agent solved 2 of the same 26. Its training success stayed
between 0 and 0.17 per batch with flat entropy.

*Post hoc (A2).* The same diagnostic with the `gated` encoder (the encoder the
hyperparameters were tuned on) solved 22 of 30 tasks (18 of the 26 common
tasks), with training success rising from 0 to about 0.5 per batch and policy
entropy falling from 2.70 to 2.19. Successful episodes took 3.1 times the
optimal plan length on average. This shows that the shared training core can
learn shallow tasks; it was decided after seeing the P1 outcome, uses one seed,
and does not reverse the recorded failure.

**3.2 Evidence-driven topology revision: occurred.** Both P2
`adaptive_incidence` runs logged structural edits from public evidence: 34
edits removing 79 candidate incidences (seed 0) and 27 edits removing 58 (seed
1); no incidence was added back (deterministic profile). Seed 0 recorded 621
informative, 524 uninformative and 0 contradictory attempts. Example
(`artifacts/runs/p2-pilot-adaptive_incidence-s0-*/events.jsonl`): in world
`W2ec22df1`, version 1 to 2, the recipe
`lens<=[]+pool{fiber,stone,salt,resin,mould_ready,kiln_ready}` lost the
incidences `fiber` and `kiln_ready` on evidence `e51-e54, e76`, with posterior
entropy falling from 3.71 (uniform over 41 hypotheses) to 2.64. The
fixed-supergraph and evidence-only arms logged no structural edits, as
designed.

**3.3 Learned skill admitted and used: occurred (diagnostic profile).** In
each P3 pretraining run, one discovery round mined the agent's own
trajectories, proposed candidates, trained three controllers (behaviour
cloning on the agent's own fragments, then 1,200 interactions of PPO
practice), validated them on 10 separate episodes and applied the declared
criteria:

| Seed | Candidate | Validation success | Mean steps (successes) | Decision |
|---|---|---|---|---|
| 0 | `achieve[bell]/L1` | 10/10 | 5.0 | admitted (provisional) |
| 0 | `achieve[glass]/L1` | 8/10 | 5.5 | admitted (provisional) |
| 0 | `achieve[blade]/L1` | 0/10 | - | rejected: success 0.00 below 0.5 |
| 1 | `achieve[dye]/L1` | 9/10 | 4.2 | admitted (provisional) |
| 1 | `achieve[key]/L1` | 8/10 | 6.0 | admitted (provisional) |
| 1 | `achieve[gear]/L1` | 0/10 | - | rejected: success 0.00 below 0.5 |

Three further candidates per seed were deferred by the per-round training cap.
The trained manager then invoked the admitted skills during pretraining (seed
0: `bell` 87 calls, 40 reached their target; `glass` 27 calls, 14 reached it;
seed 1: `dye` 15 of 49, `key` 12 of 30). The event chain (source episodes,
candidate id, controller hash, validation episodes, admission, later
invocations) is in each pretraining run's `events.jsonl` and `library/`.

Level-2 skills: every growing-arm run (G0 and G1, both seeds) admitted one
level-2 skill (`achieve[bell]/L2` in seed 0, `achieve[dye]/L2` in seed 1;
validation 9/10 or 10/10). Each pins
the level-1 skill **with the same target**, and its learned controller does
call that child (15 to 46 nested calls per run), so nested execution was
exercised, but these are wrappers, not new compositions. The cause was a
mining flaw, fixed after the pilot (DECISIONS A4); no meaningful learned
composition was obtained in this session.

## 4. Pilots

### 4.1 P1 (known structure; 4,500 interactions per run)

| Arm | Parameters | Eval success seed 0 | Eval success seed 1 | Train success, first 3 to last 3 batches |
|---|---:|---|---|---|
| `set` | 138,946 | 0/20 | 0/20 | 0.05 to 0.02; 0.00 to 0.00 |
| `incidence` | 144,898 | 0/20 | 0/20 | 0.05 to 0.07; 0.00 to 0.00 |
| `gated` | 157,315 | 0/20 | 0/20 | 0.03 to 0.02; 0.00 to 0.03 |

No arm learned at this budget on the depth 1-4 mixture, so the primary
contrast (`gated - set` = 0.0) is a floor effect and carries no information
about the gate. The gated runs' mean context gate fell from 0.56 to 0.31 and
from 0.52 to 0.06 over training (logged as `context_weight`, uninterpreted).

### 4.2 P2 (hidden prerequisites; 4,500 interactions per run; inference track on unseen worlds)

| Arm | Eval seed 0 | Eval seed 1 | Structural edits |
|---|---|---|---|
| `evidence_set` | 0/27 | 0/27 | none (no inference module) |
| `fixed_supergraph` | 1/27 | 1/27 | none (structure frozen; same beliefs) |
| `adaptive_incidence` | 2/28 | 0/27 | 34; 27 |

Seed-block differences: adaptive minus fixed -0.001 (bootstrap interval
-0.075 to 0.089); fixed minus evidence-only 0.037 (0.000 to 0.093). With two
blocks and near-zero success these intervals are not interpretable; all arms
are close to the floor.

### 4.3 P3 (shared mechanics, failure probability 0.1; 14,000 all-in per arm)

Evaluation on held-out longer compositions (test, goal levels 5-6) and on
shorter ones (val, levels 2-4); training goals were at levels 1-3.

| Arm | Test s0 | Test s1 | Val s0 | Val s1 | Train success s0 (first to last) | Train success s1 | Structural edits after fork |
|---|---|---|---|---|---|---|---|
| F0 fixed library, frozen | 2/20 | 0/20 | 6/20 | 0/20 | 0.37 to 0.58 | 0.07 to 0.28 | 0 |
| F1 fixed library, revised | 1/20 | 0/20 | 6/20 | 1/20 | 0.36 to 0.54 | 0.18 to 0.17 | 2; 3 |
| G0 growing, frozen | 1/20 | 1/20 | 2/20 | 0/20 | 0.41 to 0.51 | 0.08 to 0.07 | 0 |
| G1 growing, revised | 0/20 | 1/20 | 6/20 | 0/20 | 0.41 to 0.50 | 0.05 to 0.17 | 1; 3 |

Seed-block contrasts on test success: `G1 - G0` -0.025 (-0.10 to 0.075);
`F1 - F0` -0.025 (-0.125 to 0.05); `G0 - F0` 0.0 (-0.125 to 0.10);
interaction `(G1 - G0) - (F1 - F0)` 0.0 (-0.10 to 0.125). Not interpretable
with two blocks.

Observations, not claims:

- Success on the training distribution (0.2-0.6 for seed 0) is far above
  success on longer compositions (at most 2/20), so the pilot shows no transfer
  to longer chains under the frozen track.
- The seed blocks differ strongly (seed 0 pretraining reached 0.35 training
  success, seed 1 reached 0.06), which is the variability the full study has
  to absorb.
- Growth after the fork bought only the same-target wrapper (Section 3.3), so
  the G arms did not test growth of meaningful compositions.
- Managers often call a skill whose target already holds (for example 19 of 39
  calls of `achieve[bell]/L2` in one run); each such call costs a `wait`.

## 4.4 Hypergraph-walker study (addendum W of the plan)

Separate ledger `runs/walker-ledger.json`: 22,948 of 32,000 adaptive
interactions (4,377 plumbing before the addendum, 13,416 Stage A, 5,155 Stage
B) and 3,196 of 8,000 reporting. Evidence in `artifacts/walker/`; the figure
`walker_cost_by_episode.png` plots cost relative to the optimum by episode
index within each world. Walkers are planners on hypothesized structures, not
learned policies; the worlds are the only independent units (5 for P2 tasks,
2 for P3 worlds), so everything below is descriptive.

**Stage A, P2 pilot tasks** (30 tasks, 5 unseen worlds x 6 episodes; walkers
over 3 seeds, bounds deterministic):

| Arm | Success, budget 1.0 | Steps / optimal, budget 1.0 | Success, budget 0.6 | Steps / optimal, budget 0.6 |
|---|---|---|---|---|
| `reference` (privileged) | 1.00 | 1.00 | 1.00 | 1.00 |
| `maximal` (full-pool node) | 1.00 | 1.81 | 0.60 | 1.67 |
| `sample` (posterior sampling) | 0.97-1.00 | 1.63-1.66 | 0.60-0.70 | 1.52-1.58 |
| `optimistic` | 0.83 | 1.82 | 0.53 | 1.64 |
| `local_focused` (walk on the graph) | 0.87-1.00 | 1.74-1.82 | 0.53-0.60 | 1.52-1.58 |
| P2 pilot policies (for reference) | 0.00-0.07 | - | - | - |

- **Primary contrast failed.** `optimistic` is not cheaper than `maximal` at
  budget 1.0 (1.82 versus 1.81; cheaper in 2 of 5 worlds) and misses 5 of 30
  tasks: each wrong cheapest guess costs a failed attempt. Design decision W3
  was wrong on these tasks.
- Posterior sampling is cheaper than the brute-force node (1.63-1.66; cheaper
  in 4 of 5 worlds) at equal success, and the cost falls within a world as
  evidence accumulates (from 1.95 in a world's first episode to 1.39 in its
  fifth, pooled), while `maximal` stays near 1.8.
- The local walk on the graph (`local_focused`, episodic evidence) lands close
  to `optimistic`, which uses the exact factorized posterior; in this profile
  the walk neither helps nor hurts compared with exact inference.
- Every walker and the brute-force node solve 83-100% of the tasks on which
  the P2 pilot policies solved 0-7%; the task budget always admits the
  brute-force plan (DECISIONS W2).
- P1 tasks: planning on the supplied structure is optimal (30/30, ratio 1.00).

**Stage A, P3 pilot worlds** (same shared world per seed block; failure
probability 0.1; 1,500 warm-up interactions on training tasks, then the P3
pilot's 20 test tasks at goal levels 5-6):

| Arm | Seed 0 success | Seed 0 steps/opt | Seed 1 success | Seed 1 steps/opt |
|---|---|---|---|---|
| `sample` | 20/20 | 1.52 | 19/20 | 1.38 |
| `maximal` | 19/20 | 1.62 | 18/20 | 1.66 |
| `optimistic` | 2/20 | 2.48 | 7/20 | 1.86 |
| `reference` (privileged, open loop) | 14/20 | 1.55 | 15/20 | 1.41 |
| P3 pilot arms (14,000 interactions each) | 0-2/20 | - | 0-1/20 | - |

With noise no hypothesis is ever eliminated, so "the cheapest supported node"
stays the cheapest node and `optimistic` keeps failing: the rule needs a
support threshold under noise. The privileged reference replays the
deterministic optimal plan and restarts after a failure, which is why it misses
tasks here; ratios are still relative to the deterministic optimum.

**Stage B, goal-only observation** (hypotheses do not factorize; 40 tasks, 5
unseen worlds x 8 episodes; 2 seeds):

| Arm | Success (s0 / s1) | Steps / optimal (s0 / s1) | Search evaluations (s0 / s1) |
|---|---|---|---|
| `reference` (privileged) | 1.00 | 1.00 | - |
| `maximal` | 1.00 | 1.82 | 0 |
| `local_focused` | 0.78 / 0.72 | 1.90 / 2.03 | 370 / 543 |
| `local_uniform` | 0.72 / 0.75 | 2.08 / 2.05 | 3,754 / 2,563 |
| `learned` | 0.60 / 0.68 | 2.24 / 2.14 | 657 / 937 |

Offline search benchmark (primary contrast; held-out evidence of the focused
arm, 35 problems x 5 paired start nodes per seed, at most 400 evaluations):

| Walk | Mean evaluations (s0 / s1) | Median (s0 / s1) | Consistent node found (s0 / s1) |
|---|---|---|---|
| uniform proposals | 119.3 / 104.7 | 69 / 64 | 0.88 / 0.93 |
| focused proposals | 37.6 / 39.2 | 10 / 14 | 0.97 / 0.99 |
| learned proposals | 26.5 / 32.7 | 13 / 23 | 1.00 / 1.00 |

- **Primary criterion met, narrowly.** The learned walk needs fewer evaluations
  on average than the focused heuristic in both seeds and always finds a
  consistent node, but its median is higher: it removes the long, failed
  searches rather than speeding up the typical one. The edit policy was trained
  on 56 problems per seed (1,500 internal searches, 71,507 and 66,681
  evaluations), on evidence collected for 852 and 867 interactions.
- Focus matters: violation-focused proposals need about a third of the
  evaluations of uniform ones.
- **Better search did not give better acting.** Online, the learned walker is
  the worst walker, and every walker is worse than the brute-force node in this
  profile (only goal outcomes are observed, so each wrong hypothesis costs a
  whole failed chain, and 8 episodes per world do not amortize that). The
  policy is rewarded for reaching any consistent node quickly, not for reaching
  one that is cheap or informative to act on.

## 5. What the session shows and does not show

Shows: the mechanisms run end to end under one interaction ledger; public
evidence produces logged, evidence-linked structural revisions; the discovery
pipeline produced controllers that pass held-out validation, are admitted with
recorded provenance and are invoked by a trained manager, while weak candidates
are rejected; the training core learns shallow tasks with the gated encoder
(post hoc).

Does not show: any advantage of the context gate, of adaptive routing, of
library growth or of their combination; transfer to longer compositions by the
learned policies; learned composition of skills; anything about statistical
significance.

Walker study: planning on a node chosen from public evidence solves the tasks
the pilot policies could not, including the P3 pilot's longer test tasks after
1,500 interactions; posterior sampling over nodes beats the brute-force node on
cost when items are observable; the optimistic rule does not (the frozen
primary contrast failed) and breaks under noise. When intermediate items are
hidden, a learned walk finds consistent nodes with fewer evaluations on average
than a focused heuristic, but no walker beats the brute-force node at acting.

## 6. Known issues in the recorded runs

- Provenance (A3): all pilot processes executed the training code of `a79d840`
  (the gated diagnostic: `070252b`), although some manifests name a later HEAD.
- Evaluation rows of the pilot runs report `manager_decisions = 0` because
  unrecorded episodes were not counted (fixed in `6c44ec1`). For P1/P2 the
  decision count equals the primitive length; for P3 it is recovered from the
  logged call records.
- Hyperparameters were tuned only with the gated encoder (A1).
- The P3 miner could not produce genuine compositions (A4, fixed after the pilot).

## 7. Commands not run in this session

```bash
python -m hypergraph_agent.train --config configs/study.yaml --allow-full-study
python -m hypergraph_agent.train --config configs/ablations/p3_macro.yaml --allow-full-study
python -m hypergraph_agent.train --config configs/ablations/p3_unstructured.yaml --allow-full-study
python -m hypergraph_agent.train --config configs/ablations/p2_noisy.yaml --allow-full-study
python -m hypergraph_agent.evaluate --checkpoint runs/<p3 arm run> --config configs/transfer_frozen.yaml
python -m hypergraph_agent.evaluate --checkpoint runs/<p2 run> --config configs/transfer_inference.yaml
```

`configs/study.yaml --dry-run` reports 3,300,000 adaptive interactions for ten
seed blocks; that cost and the seed count should be revisited in light of the
variability above before it is launched.
