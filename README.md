# hypergraph_agent

Toy experiment on training an agent on a hypergraph model of "reasoning":
**RecipeQuest**, a small symbolic crafting benchmark for studying
context-conditioned relational policies, learned prerequisite topology and
growing libraries of reusable closed-loop skills.

> Research question (a hypothesis, not a claim): from public interaction
> evidence, can an agent learn and revise a compact library of closed-loop
> skills and conjunctive prerequisite relations, and does that improve transfer
> to longer compositions beyond topology adaptation or skill discovery alone?

Status: research code at the development-pilot stage; nothing here
establishes a performance or novelty claim. The first development session
([docs/RESULTS.md](docs/RESULTS.md)) ran all three phases under one
160,000-interaction ledger:

- the mechanisms run end to end, and 152 deterministic tests pass;
- public evidence produced logged, evidence-linked revisions of the
  prerequisite structure (P2);
- skills mined from the agent's own trajectories were trained, passed held-out
  validation, were admitted and were then invoked by a trained manager (P3);
  weak candidates were rejected;
- the pre-registered shallow learning check failed for the token-attention
  baseline; a post-hoc repeat with the gated encoder learned;
- no pilot arm transferred to longer compositions, and two seeds per contrast
  support no comparative conclusion.

A follow-up study treats the hypotheses themselves as a graph: each node is a
complete dependency hypergraph, each edge a single-incidence edit, and
"hypergraph walkers" move on it from public evidence and plan on the node they
choose (`python -m hypergraph_agent.walk`). Posterior sampling over nodes
solved the tasks the learned policies could not, at fewer steps than a
brute-force plan. When intermediate items are hidden, walks that sample among
the hypotheses consistent with the evidence pay for early exploration, then
act more cheaply than the brute-force plan once evidence has accumulated: over
episodes 9-16 of fresh worlds they used 9-18% fewer steps in both seeds.
A learned edit policy for the walk did not contribute
([docs/RESULTS.md](docs/RESULTS.md), Section 4.4).

## The game

Facts are Boolean and persistent. A recipe is a directed hyperedge: a
conjunction of prerequisite facts produces one effect fact, and an item can
have alternative recipes (AND within a recipe, OR across recipes). Primitive
operations gather a resource, activate a facility, attempt a recipe, wait or
submit; each costs one step of a public task budget. Reward is 1 on successful
submission and 0 otherwise.

```
ore + fuel + furnace_ready  --smelt-->  ingot
ingot + mould_ready         --shape-->  key
```

## Three phases

| Phase | What is supplied | What is learned or inferred |
|---|---|---|
| P1 | the true recipe groupings | a policy; compares a token-attention baseline, a role-typed incidence (hypergraph) encoder and a context-gated variant |
| P2 | candidate prerequisite pools; the true subsets are hidden | beliefs over a finite hypothesis class (41 per recipe), updated from public attempts; an adaptive incidence structure versus a fixed candidate supergraph with the same beliefs |
| P3 | the same as P2, one shared world per seed block | candidate skills mined from the agent's own trajectories, closed-loop controllers trained on public target predicates, validated and admitted into a versioned library with a call hierarchy; a manager trained with duration-aware PPO |

See [docs/METHODS.md](docs/METHODS.md) for the mechanisms,
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for the frozen protocol,
[docs/DECISIONS.md](docs/DECISIONS.md) for design decisions and amendments and
[docs/PRIOR_ART.md](docs/PRIOR_ART.md) for the related-work audit.

## Installation

Python 3.10 or newer (developed with 3.12), CPU only.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Commands

```bash
python -m hypergraph_agent.play --agent manual --seed 7
python -m hypergraph_agent.play --agent reference --seed 7 --debug   # privileged, labelled
python -m hypergraph_agent.train --config configs/smoke.yaml
python -m hypergraph_agent.train --config configs/phase1_pilot.yaml
python -m hypergraph_agent.train --config configs/phase2_pilot.yaml
python -m hypergraph_agent.train --config configs/phase3_pilot.yaml
python -m hypergraph_agent.train --config configs/study.yaml --dry-run
python -m hypergraph_agent.evaluate --checkpoint runs/<run> --config configs/transfer_frozen.yaml
python -m hypergraph_agent.summarize --runs runs --out artifacts
python -m hypergraph_agent.walk --config configs/walker_stage_a.yaml
python -m hypergraph_agent.walk --config configs/walker_stage_b.yaml
python -m hypergraph_agent.ledger show
```

Every command that touches the environment charges a persistent session
ledger (`runs/ledger.json`), so separate commands and restarts share one
interaction allowance. `configs/study.yaml` refuses to start without
`--allow-full-study`. Raw runs and checkpoints stay in `runs/` (ignored by
Git); `summarize` copies small, attributable records to `artifacts/`.

## Layout

```
src/hypergraph_agent/
  envs/            game, generator, public schema, reference solver (evaluator only)
  representations/ shared observation adapter and encoders
  topology/        hypothesis updater, snapshots, deltas
  skills/          skill specs, library, binder, executor, discovery
  agents/          actor-critic and labelled baselines
  training/        rollouts, returns, PPO, budget ledger, P1/P2 and P3 trainers
  evaluation/      evaluation tracks, metrics, seed-block statistics
configs/           smoke, diagnostics, pilots, study and ablation presets
tests/             deterministic mechanism tests
docs/              methods, plan, decisions, prior art, results, roadmap
paper/OUTLINE.md   prospective outline with a claim-to-evidence table
```

## License

MIT, see [LICENSE](LICENSE).
