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

## Amendments

(none yet)
