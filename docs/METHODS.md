# Methods

This describes what the code does, with pointers to the modules. Everything
learned, inferred, specified or supplied is labelled as such.

## 1. Clocks and structures

- `t`: primitive environment transitions. Every operation, including a failed
  or useless one, costs exactly one.
- `k`: decisions of the root agent (the manager in P3). A skill invoked at
  decision `k` lasts `tau_k` primitive steps.
- `n`: version of the persistent dependency model or skill library, changed
  only between rollout/update batches.

Three structures are kept apart: the dependency model (recipes as directed
hyperedges with beliefs about their hidden prerequisites), the call hierarchy
(which skill may call which lower-level skill) and the active routing
structure used by one decision. A hyperedge groups prerequisites; it says
nothing about order, which is left to policies.

## 2. Environment and information boundary (`envs/`)

**Worlds.** Item types are arranged in levels 1..8 (two per level). A level-l
recipe consumes one public item input from level l-1 and a base requirement of
1-3 facts. Each item has a primary recipe and, with probability 0.25, an
alternative recipe (`generator.py`). A task asks for one goal item. Its facts
are the goal's closure (all items reachable through its recipes, including
alternatives), every base fact of every included recipe's pool, one distractor
recipe (whose inputs are already in the task) and, in some configs, extra base
facts. Storage order and fact/recipe keys are random per task.

**Profiles.** In `known_structure` (P1) every recipe's true prerequisites are
public. In `unknown_prerequisites` (P2, P3) item inputs are public and the base
requirement is a hidden nonempty subset (size at most 3) of a public pool of 6
base facts; the pool is drawn before the subset (D2).

**What every agent receives** (`public_schema.PublicTaskSpec`): fact keys,
types, kinds and the goal marker; recipe keys, type-level signatures, effects,
item inputs and either the known base requirement or the candidate pool; the
list of syntactic actions; the task budget; the declared failure probability;
a public world label (used only as a belief scope). Each observation adds the
present flags, the remaining budget, the step index, the last action and
whether it changed anything. Hidden prerequisites, reference plans, eligibility
and the dynamics RNG never cross this boundary; tests check canary pairs of
worlds that differ only in hidden truth, and scan learning modules for imports
of the reference solver.

**Dynamics** (`recipequest.py`). Gather/activate make a base fact true; craft
makes its effect true when all true prerequisites hold; submit succeeds when
the goal holds. With failure probability `epsilon` an eligible state-changing
action fails and leaves the state unchanged. Reward is 1 on success, else 0;
running out of budget is an unsuccessful terminal ("deadline"); the
environment never truncates.

**Budgets** are a constructive public upper bound, `absent base facts + absent
items + 1`, plus a slack of 2 (D3). **Reference lengths** (`reference_solver.py`,
evaluator only) are exact for these monotone tasks (D4) and are reported with
every evaluation episode.

## 3. Shared observation adapter (`representations/features.py`)

A task becomes a bipartite graph of fact nodes and relation nodes (recipes,
plus one node per bound skill in P3). Incidences carry one of five roles:
certain prerequisite, candidate prerequisite, effect, skill target, skill
support; each has a weight (posterior marginal for candidates, contract weight
for skill support, else 1).

- Fact features: type one-hot (34), kind one-hot (3), goal marker, skill-target
  marker, present flag.
- Relation features: effect present, item-input count, known-base count, pool
  size, posterior entropy, supported-hypothesis fraction, MAP mass, skill flag,
  skill level, skill success and duration estimates.
- Graph modes: `known` (P1); `supergraph` (every pool member is an incidence);
  `active` (only pool members in the snapshot's active set). With
  `use_posterior=False` the belief features are replaced by the prior.

Candidates are all syntactic actions plus bound skill calls; each is scored
from its own descriptor and the embedding of its node. There are no
feasibility masks; padding masks exist only for the padding-invariance tests,
and an all-masked row falls back to `wait`.

## 4. Encoders and actor-critic (`representations/`, `agents/policy.py`)

**Recurrent core.** `h_k = GRU(h_{k-1}, MLP(m_k))`, where `m_k` summarizes
public history: remaining budget, progress, fraction of facts present, goal
present, previous choice kind, the type of its target, whether it changed
anything, its return and its duration.

**`set`** (token attention, A1 style): tokens for the memory, each fact, each
relation node and each incidence; incidence tokens carry the identifiers of both
endpoints (random unit vectors bound to opaque keys); two Transformer layers.

**`incidence`** (role-typed message passing, three weight-tied rounds):

```
fact -> rule:  m_e = relu(W_role(e) h_f + U ef_e);  h_r <- LN(h_r + MLP[h_r, sum_e m_e, c])
rule -> fact:  m_e = relu(V_role(e) h_r + U' ef_e); h_f <- LN(h_f + MLP[h_f, sum_e m_e, c])
```

with `c` the projected memory. **`gated`** adds
`alpha_r = sigmoid(g[h_r, c, h_goal])` multiplying every outgoing rule message;
it is the only difference from `incidence` (D6). The dense two-stage
hypergraph form computes the same function and is used only in the
equivalence test.

**Heads.** `score_u = MLP[cand_u, node_u, h_k, h_goal, pooled]`, a softmax over
candidates; `V = MLP[h_k, pooled, h_goal]`. The same module serves P1/P2
agents, the P3 manager and P3 skill controllers.

## 5. Prerequisite inference (`topology/`)

For each recipe the hypothesis class is every nonempty subset of its pool of
size at most 3 (41 hypotheses), with a uniform prior. Evidence is recorded only
for craft attempts whose effect was absent: `x` the public facts present
before, `y` whether the effect appeared.

```
p(y=1 | B, x) = (1 - epsilon) * [item inputs within x and B within x]
log q_new(B) = log q_old(B) + log p(y | B, x) - log Z
```

Updates are in log space. Evidence that every hypothesis rules out is counted
as a contradiction and leaves the posterior unchanged; evidence with missing
item inputs is uninformative. This is an interaction-based updater over a
declared finite class, not a neural topology learner and not open-world causal
discovery.

**Snapshots and edits** (`snapshot.py`). Each commit (between batches) builds an
immutable `TopologySnapshot` per scope: marginals, normalized entropy, support
and MAP mass per recipe, and the active incidence set. In `revise` mode the
active set is the union of the smallest top-posterior hypothesis set holding
0.9 of the mass. Every change of an active set is a `TopologyDelta` (adds,
removes, replacements, evidence ids, entropy before and after). In
`frozen_structure` mode beliefs and features update but incidences never
change; `disabled` drops evidence. Commits are atomic: an invalid edit rolls
back and is logged. Context gates are logged separately as `context_weight`
events.

## 6. Skills (`skills/`)

**Specification** (`spec.py`). `achieve[<item type>]/L<level>`, one argument
role bound to the unique fact of that type by a public binder; initiation
`public_target_absent` and termination `target present or timeout` are
specified; the controller is learned. A spec also stores pinned child versions,
the controller hash, predicted support links (contract), success and duration
estimates, provenance, training cost, validation summary and admission profile
(`diagnostic` admissions are provisional).

**Executor** (`executor.py`). A bounded call stack executes one primitive at a
time. Termination is checked after every primitive (target, task terminal,
timeout capped by the caller's remaining time, dispatch limit). Calls that
cannot act (target already present, invalid binding, depth limit) execute one
`wait`, so no choice is free. Skills cannot submit. Budget exhaustion inside an
option ends the episode administratively; the incomplete option is kept for
cost and excluded from the update.

**Library** (`library.py`). Admission creates an immutable `(key, version)`
entry with a frozen copy of the controller; children must already be admitted
at a strictly lower level (so the call graph is acyclic); retirement cascades
to parents that pin the retired version. Candidates keep their ids and reasons
(proposed, deferred, rejected, admitted).

**Discovery round** (`discovery.py`, `training/phase3.py`).

1. Mine windows of the agent's own recent decisions ending where an item fact
   newly appeared (at least 2 primitive steps, at most 6 decisions); failed
   craft attempts count as attempt evidence. A window containing calls to
   level-l skills proposes a level-(l+1) candidate with those children (at most 3).
2. Canonicalize by (target type, level, children), deduplicate, rank by
   observed achievements, propose at most 8.
3. For at most 3 candidates: behaviour cloning on the agent's own successful
   fragments (20 epochs, logged), then PPO practice on tasks of the `practice`
   namespace whose goal is the target type, reward 1 when the target fact
   holds, 12-step episodes, 1,200 interactions.
4. Validate on 10 episodes of the `skillval` namespace (other inventories,
   bindings and dynamics seeds). Admit if at least 8 episodes ran, success is
   at least 0.5 and successful episodes take at least 2 primitive steps on
   average; at most 2 admissions per round. Otherwise reject with the reason.
   Test tasks are refused by construction.

**Contracts.** At admission a skill stores construction links: item inputs and
active base facts of the recipes producing its target, weighted by marginals.
With revision on, these links are recomputed from the current snapshot at
every batch; with revision off they stay fixed (D15).

**Macro control.** The most frequent successful primitive fragment of a
candidate becomes an open-loop sequence, optionally wrapped in a named retry
loop, validated and admitted under the same criteria (D16).

## 7. Training (`training/`)

**Duration-aware returns** (`returns.py`), with lambda decaying per primitive
step:

```
R_k = sum_{j < tau_k} gamma^j r_{t_k + j}
delta_k = R_k + gamma^tau_k * bootstrap_k * V_{k+1} - V_k
A_k = delta_k + gamma^tau_k * lambda^tau_k * continuation_k * A_{k+1}
```

`gamma = 1` by default, so changing skill durations cannot change the success
objective. A genuine terminal has no bootstrap; an administrative cutoff
bootstraps from the value at the true next decision boundary without chaining.

**Recurrent PPO** (`ppo.py`). Minibatches of whole episodes recomputed from the
zero state; clipped surrogate, value regression, entropy bonus, no advantage
normalization (D17). Before an update every episode's topology and library
snapshot ids are checked against the current ones; a mismatch raises
`StaleBatchError`.

**Batch protocol.** Freeze snapshots, collect complete episodes up to the batch
size, update, then commit evidence (and in P3, run any scheduled discovery
round and admissions) before collecting fresh data.

**P3** (`phase3.py`). Per seed block, pretraining runs 3,000 primitive
interactions of manager exploration, one discovery round of up to 4,500
interactions, then manager training with the new library until 9,000. The
checkpoint (manager, optimizer, beliefs, library, task index) is forked into
F0/F1/G0/G1 with independent post-fork RNG streams; fork hashes are logged and
asserted equal. G arms run one more discovery round after 1,500 post-fork
interactions. Arms do not retrain admitted controllers (D14).

## 8. Evaluation (`evaluation/`)

Tracks: `frozen` (no evidence recorded, nothing updated), `inference`
(world-scoped beliefs from the prior, updated between episodes of the same
world), `continual` (not implemented; raises). Every arm and seed sees the same
evaluation tasks (base seed 10000). Metrics include success, primitive length
for all episodes and for successes, decisions, no-ops, skill calls, failure
statuses and the ratio to the exact reference on successes. Contrasts use a
paired seed-block bootstrap with tasks resampled within blocks.

## 9. Budget accounting (`training/budget.py`, `ledger.py`)

One persistent ledger per session with allocations per phase and separate
adaptive and reporting caps; per-run meters stop a run at its cap. Purposes:
exploration, skill practice, candidate validation, pretraining, evaluation used
for selection, reporting evaluation. The P3 prefix is charged physically once
and logically to every arm. Unit tests use in-memory ledgers.

## 11. Hypergraph walkers (`agents/walker.py`, `agents/edit_policy.py`, `evaluation/walkers.py`)

A second family of agents moves on a graph whose nodes are complete dependency
hypergraphs: one hypothesized base requirement for every recipe with a hidden
one. Two nodes are adjacent when they differ by one single-incidence edit inside
the hypothesis class (add, remove or swap one candidate base fact), the kind of
edit a `TopologyDelta` records. The graph is implicit.

A walker commits to a node, plans on it with the derivation planner
(`envs/derivations.py`, shared with the evaluator's reference solver but given
the walker's own hypothesis), executes the plan, and moves when public
evidence contradicts the node. Moves cost no environment interactions; every
hypothesis evaluation is counted as search work. Node choice:

- `maximal`: the full-pool node (every candidate required); no inference.
- `sample`: a draw from the factorized posterior of Section 5 (Thompson sampling).
- `optimistic`: per recipe, the hypothesis still supported by the posterior that
  needs the fewest base facts not yet held.
- `local_uniform`, `local_focused`, `learned`: a Metropolis walk (temperature
  0.5, restart after 80 evaluations without improvement, at most 400
  evaluations per replan) towards a node with no violated observation, with
  uniformly proposed edits, edits that address a violated observation, or edits
  proposed by a trained policy.
- `exact`: enumeration of the joint space of the attempted recipes (up to 20,000
  nodes), returning the cheapest consistent node.

Episodic evidence: each episode is logged as its sequence of actions with the
observed goal after every step and, when items are observable, each craft's
observed effect. A node's violations are the mismatches when the episode is
replayed under it. With `observe_items: goal_only`, intermediate items and the
outcome of crafting them are hidden, so a failed goal can be explained by any
recipe of the chain and the posterior no longer factorizes.

Learned edit policy: an MLP scores every candidate edit from 12 public
features (edit type; how many violated observations it addresses; the
recipe's share of the blame; current requirement size; how often the recipe
was attempted; how often each base fact was present when the recipe was
crafted in episodes that reached the goal; whether the edit undoes the
previous move). It is trained with REINFORCE on internal search problems built
from evidence collected in training worlds; the return is minus the normalized
number of evaluations to a consistent node, minus 1 when none is found.

## 10. Cost per decision

With `F` facts, `R` relation nodes, `E` incidences, `C` candidates, width `d`
and `L` rounds: the incidence encoders cost `O(L (F + R + E) d^2)`, the token
baseline `O(layers (1 + F + R + E)^2 d)`, scoring `O(C d^2)`. A topology update
costs `O(41 * pool)` per informative attempt. Binding costs `O(skills * F)`.
Parameter counts are recorded in every run manifest.
