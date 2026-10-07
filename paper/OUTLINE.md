# Paper outline (prospective)

A planning document, not a manuscript. There is no abstract, author list or
venue. Every claim is tied to evidence in this repository or marked
unresolved.

## Working title

Separating context weights, prerequisite topology and skill growth in a
controlled crafting benchmark

## 1. Introduction

- Problem: long compositional tasks need both knowledge of what depends on what
  and reusable routines; related work changes weights, structure or
  repertoire, usually together (docs/PRIOR_ART.md).
- Question: with only public interaction evidence, does revising conjunctive
  prerequisite structure help a growing library of closed-loop skills transfer
  to longer compositions, beyond either mechanism alone?
- Contribution candidates and their audit status: a protocol that manipulates
  the four quantities separately around one shared actor-critic (unverified
  methodological contribution); the context gate (potential empirical
  contribution, gate only); belief-driven routing versus the same beliefs in a
  fixed supergraph (potential empirical contribution); the growth by revision
  interaction (potential empirical contribution). None is supported by results
  yet.

## 2. Related work

Options and semi-Markov control; subtask-graph inference (MSGI); planning
networks over hypergraphs (ASNets, STRIPS-HGN, AllSet); action-model learning
(SAM, OHCAM); skill and library growth (DisTop, Deep Skill Graphs, dynamic
option creation, DreamCoder); language-model skill archives (CODE-SHARP,
SkillPyramid, HyperAgent). Source of truth: docs/PRIOR_ART.md, which lists what
must be rechecked before citation.

## 3. Benchmark and information regimes

RecipeQuest rules, profiles, public/private boundary, budgets, exact
reference solver, namespaces and task generation (docs/METHODS.md, Sections 2-3).

## 4. Methods

Encoders and the shared recurrent actor-critic; finite-hypothesis prerequisite
updater, snapshots and deltas; skill specification, executor, library and
discovery; duration-aware PPO; budget accounting (docs/METHODS.md, Sections 4-9).

## 5. Experiments

Protocol and primary contrasts per phase (docs/EXPERIMENT_PLAN.md). The
development pilots size the study; the confirmatory study
(`configs/study.yaml`) has not been run.

## 6. Results

To be written only from recorded runs (docs/RESULTS.md and artifacts/),
including negative, equivalent and failed outcomes.

## 7. Limitations

Symbolic vocabulary supplied; finite hypothesis class with declared pools;
monotone, non-consumptive facts; heuristic proposals and specified
termination; two seeds per pilot; tuning only on the gated encoder.

## Claim-to-evidence table

| # | Claim | Evidence required | Current evidence | Status |
|---|---|---|---|---|
| 1 | The environment, public boundary and reference solver behave as specified | deterministic tests | `tests/test_env.py`, `tests/test_information_boundary.py` | supported (mechanism) |
| 2 | Incidence message passing equals the two-stage hypergraph update | output and gradient equality | `tests/test_representation.py` | supported (mechanism) |
| 3 | The shared PPO core learns shallow tasks | shallow diagnostic versus random agent | pre-registered `set` diagnostic failed (1/26 vs random 2/26); post-hoc `gated` diagnostic 22/30 (RESULTS 3.1) | unresolved (post hoc, one seed) |
| 4 | The context gate improves success over a matched ungated encoder (P1) | 10+ seed blocks, paired intervals | 2-seed pilot at floor (all 0/20) | unresolved |
| 5 | Public evidence produces logged structural revisions (P2) | `structural_incidence_edit` events with evidence ids | 34 and 27 edits in the adaptive runs (RESULTS 3.2) | supported (mechanism) |
| 6 | Belief-driven routing beats the same beliefs in a fixed supergraph (P2) | study-scale contrast plus the mask-as-feature control | 2-seed pilot near floor; control not implemented | unresolved |
| 7 | A non-scripted controller is learned, admitted and used by the manager (P3) | event chain: evidence, candidate, controller, validation, admission, invocation | 4 admissions, 2 rejections, manager invocations (RESULTS 3.3), diagnostic profile | supported (mechanism, provisional) |
| 7b | Learned skills compose (a level-2 controller using a different lower-level skill) | admitted level-2 skill with distinct children, used by the manager | only same-target wrappers were admitted; miner fixed afterwards (DECISIONS A4) | not supported |
| 8 | Growth with revision transfers better than growth without revision (P3) | study-scale G1 - G0 on held-out longer compositions | 2-seed pilot, -0.025 (-0.10 to 0.075) | unresolved |
| 9 | Revision and growth interact | interaction contrast with intervals | 2-seed pilot, 0.0 (-0.10 to 0.125) | unresolved |
| 10 | Any novelty claim | method-level comparison with MSGI, CODE-SHARP, OHCAM and others | audit at abstract or methods depth | unresolved |
| 11 | Choosing a node of the hypothesis graph and planning on it saves interactions over the brute-force node | frozen W-A contrast `optimistic` - `maximal` | 1.82 vs 1.81 (failed); posterior sampling 1.63-1.66 vs 1.81, 4 of 5 worlds (secondary) | primary not supported; sampling variant supported descriptively |
| 12 | A learned walk finds consistent hypotheses with fewer evaluations than a focused heuristic when hypotheses do not factorize | frozen W-B offline benchmark | mean 26.5 / 32.7 vs 37.6 / 39.2, higher median (RESULTS 4.4) | supported narrowly (2 seeds, 5 held-out worlds) |
| 13 | Faster search over hypotheses yields cheaper acting | online cost of the walkers in Stage B | learned walker is the costliest; no walker beats the brute-force node | not supported |
