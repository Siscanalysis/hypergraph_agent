# Design decisions and amendments

Each entry states the decision, the reason and what it rules out. Dated
amendments to the protocol (`docs/EXPERIMENT_PLAN.md`) are at the end.

## Environment and information boundary

**D1. One public type vocabulary, opaque task-local ids.** Every world draws
its facts from 12 resources, 6 facilities and 16 item types. Agents see each
fact's type and kind, never its world-level rule. Fact and recipe keys are
random per task and storage order is permuted, so ids carry no cross-task
identity. This is a declared symbolic inductive bias, not learning from raw
sensors.

**D2. What is hidden in `unknown_prerequisites`.** A recipe's item inputs
(the chain) are public; its base requirement is a hidden nonempty subset, of
size at most 3, of a public pool of 6 base facts (41 hypotheses, uniform
prior). The pool is sampled before and independently of the hidden subset, so
pool membership and order carry no information about it. Recipes are matched
across tasks of one world by their public type-level signature.

**D3. Budgets from public counts.** The task budget is
`ceil(factor x (absent base facts + absent items + 1)) + slack`, a
constructive upper bound computable from the public view in both profiles. It
never uses the hidden truth or the reference planner.

**D4. Exact reference lengths without search blow-up.** For monotone,
deterministic, unit-cost tasks the optimum is the cheapest consistent
derivation (one recipe per crafted item) plus the union of its base facts plus
the submission; `reference_solve` enumerates derivations under a cap and
reports `upper_bound` when the cap is hit. It is checked against breadth-first
search on small tasks.

## Representations

**D5. Information-complete set baseline.** The attention baseline tokenizes
facts, relation nodes and incidences, and gives each incidence token the
identifiers of its two endpoints (random unit vectors bound to opaque keys, in
the style of tokenized graph transformers). It therefore receives the same
public information as the graph encoders but no grouped message passing. A
lossy pairwise projection that drops group identity is not implemented.

**D6. The gate is the only difference between `incidence` and `gated`.** Both
concatenate the recurrent memory into rule and fact updates; `gated` adds a
sigmoid gate per relation node, conditioned on memory and the goal embedding,
that multiplies outgoing messages. This matched-information control follows the
prior-art audit's recommendation. Parameter counts are logged (the gate adds a
small MLP).

**D7. Recurrent core.** A GRU summarizes public history features (budget,
progress, previous choice kind and target type, outcome, option return and
duration). The encoder is conditioned on the current memory. PPO recomputes
whole episodes from the zero initial state; timesteps are never shuffled.

**D8. Exact incidence and hypergraph duplicates are one method.** The dense
two-stage hypergraph form exists only for the equivalence test (outputs and
gradients agree when edge features are constant per role).

## Topology

**D9. World-scoped beliefs, batch-boundary commits.** P2 trains in 8
persistent worlds visited round-robin; beliefs are keyed by the public world
label and reset for unseen worlds. Evidence is collected during a batch and
committed after the PPO update, so the routing structure is fixed within every
batch. Within-episode hard routing is not implemented.

**D10. Active incidences.** For each recipe, the union of the smallest
top-posterior hypothesis set holding 0.9 of the mass, with ties broken by
canonical hypothesis order. A recipe seen for the first time is routed
through its whole pool; its first revision is logged as a delta from that
full-pool structure.

**D11. Same beliefs in the adaptive and fixed arms.** `fixed_supergraph`
uses `frozen_structure` mode: posterior features update from the same
evidence, but no incidence is ever removed or added. `disable_inference` is a
separate diagnostic mode, not a comparator.

## Skills

**D12. Skill identity.** A skill is `achieve[<item type>]/L<level>` with one
argument role (the unique fact of that type), bound by a public binder.
Initiation (`public_target_absent`) and termination (target predicate or
timeout) are specified, not learned; the controller is learned.

**D13. Proposal heuristic and behaviour-cloning start.** Fragments are mined
from the agent's own trajectories and ranked by observed achievements.
Controllers start with behaviour cloning on the agent's own successful
fragments (20 supervised epochs, no interactions, logged as `bc_updates`) and
are then trained with PPO on the public target predicate. The same pipeline
runs in every arm that grows a library.

**D14. Arms do not retrain admitted controllers after the fork.** This holds
equally in all four arms, so the topology switch cannot change the amount of
skill practice.

**D15. Contracts.** At admission a skill stores construction links: the item
inputs and active base facts of the recipes producing its target, weighted by
posterior marginals. With revision on (F1, G1) these links are recomputed from
the current dependency snapshot; with revision off (F0, G0) they stay as
admitted.

**D16. Matched-macro control.** The macro arm admits the most frequent
successful primitive fragment as an open-loop sequence (optionally wrapped in
an explicitly named retry loop), validated under the same criteria.

## Training

**D17. No advantage normalization.** With 0/1 rewards and gamma 1 the raw
advantages already have a fixed scale. Dividing by the batch standard deviation
turned value-function noise into unit-scale gradients in batches without any
success, which made the policy drift and lose entropy without reward. Found
during development (overfit diagnostic); fixed before the plan was frozen.

**D18. Evaluation by sampling.** Every arm is evaluated with its stochastic
policy, so no arm benefits from a greedy decoding that another lacks.

## Hypergraph walkers

**W1. The graph of hypergraphs is implicit.** A node is one complete
hypothesis (a base requirement for every recipe with a hidden one); edges are
single-incidence edits inside the 41-hypothesis class (add, remove, swap).
Neighbours are generated on demand; nothing is materialized.

**W2. The public budget admits a brute-force plan.** The task budget is the
constructive upper bound "gather every base fact, craft every item, submit",
so in deterministic profiles the full-pool node (`maximal`) always succeeds
(a test checks this). Success alone therefore cannot distinguish walkers; the
walker study reports the cost ratio to the optimum and adds a tighter public
budget (factor 0.6) as a variant. The pilot policies' near-zero success means
they did not learn even the brute-force plan.

**W3. Optimism.** Episodic walks start from single-candidate requirements and
`exact` returns the cheapest consistent node; `optimistic` picks the cheapest
node still supported by the factorized posterior. Under deterministic
elimination a wrong optimistic node is refuted by its own failed attempt.

**W4. Episodic evidence.** Walkers that use episode logs replay a hypothesis
over each logged episode and count mismatches with every observed fact (the
goal always; a craft's effect only when items are observable). This is valid
when intermediate items are hidden, where per-recipe updates are not. It
assumes deterministic dynamics and items absent at reset, both declared.

**W5. Goal-only observation profile.** `observe_items: goal_only` hides
intermediate items and the outcome of crafting them (observation, `last_changed`
and `info`), so the posterior over hypotheses no longer factorizes by recipe.
The factorized updater refuses this profile rather than reading masked items
as absent.

## Not implemented (raise or are absent by design)

Within-episode hard routing; relation-local recurrent memory; the continual
cross-task adaptation track; the goal-conditioned primitive-policy control;
the library-transplant diagnostic; the lossy pairwise-projection ablation;
learned termination; learned proposals; automatic curricula.

## Amendments

**A1 (2026-10-07, before any pilot of the session ledger).** Development
before the protocol freeze (plumbing runs and PPO tuning) executed 37,000
adaptive and 3,452 evaluation interactions outside the session ledger: 3,500
plumbing (P1 encoders and one P3 end-to-end run) and 33,500 learning checks and
tuning (depth-1 tasks; learning rate 1e-3 or 3e-3, 4 or 8 epochs, advantage
normalization on or off, one single-task overfit run, one smaller-pool
variant; all with the gated encoder). They are entered in the session ledger as
the run `pre-plan-development`. To stay within the 160,000 session cap the
remaining allocations were reduced (P1 and P2 per-run caps 6,000 to 4,500, P3
per-arm cap 6,000 to 5,000, the shallow diagnostic to one `set` run, smoke to
2,500; reporting caps per run reduced accordingly). The tuning used only the
gated encoder, so any tuning advantage favours the gated arms.

**A2 (2026-10-07, post hoc: decided after the shallow diagnostic and the P1
pilot had been evaluated).** The pre-registered shallow diagnostic (`set`
encoder) failed, and every P1 run scored zero, so the protocol produced no
evidence that the shared training core learns at all. One additional
diagnostic run, identical except for the encoder (`gated`, the encoder the
hyperparameters were tuned on), is run on the same 30 evaluation tasks as the
failed run and the random reference (`configs/diagnostic_shallow_gated.yaml`).
It uses 3,400 previously unallocated interactions moved into the diagnostics
allocation (`ledger amend`), keeping the session total under 160,000. It is
reported as post hoc; it does not replace the recorded failure and cannot
change any P1 contrast.

**A3 (2026-10-07, provenance).** Run manifests created before the commit
"Record run provenance once per process" record the HEAD at run creation. Pilot
processes started at commit `a79d840` and kept executing that code while
later commits changed only playback, documentation and the decision count of
unrecorded evaluation episodes; some of their manifests therefore name
`d4a9865` or `6c44ec1`, or a later commit or working-tree patch (A4). From the
provenance commit on, manifests record the commit and a hash of the source
loaded when the process started. All pilot processes of this session executed
the training code of `a79d840`, except the post-hoc gated diagnostic (A2),
which ran at `070252b`.

**A4 (2026-10-07, post pilot: fragment mining).** Inspecting the P3 pilot
showed that the only admitted level-2 skill (`achieve[bell]/L2`) pinned the
level-1 skill with the same target: a wrapper, not a composition. The cause
was a mining flaw: window boundaries were reset at every item achievement,
including items produced by a called skill, so a window could never contain
"call a skill, then use its product". Mining now (i) keeps skill-produced
items inside the next window and (ii) skips fragments whose achievement is
already produced by a called skill with the same target. A test covers both.
The recorded pilot ran with the old miner and is reported as it ran.
