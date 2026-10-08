# Roadmap

Prioritized extensions tied to current limitations. None of them is
implemented; each needs a motivating observation, a hypothesis, a strong
control and a cost estimate before it is built.

## Needed to interpret the existing contrasts

1. **Credible-set mask as an input feature for the fixed supergraph (P2).**
   The adaptive arm currently differs from the fixed arm by removing edges; a
   fixed-supergraph policy given the same mask as a feature would show whether
   rewiring adds anything beyond the information it encodes (prior-art audit,
   contribution c).
2. **Library-transplant diagnostic (P3).** Freeze one arm's skill snapshots and
   evidence, then compare selectors and routing on common tasks with equal
   extra training, so G1 versus G0 differences can be separated from the
   different skills each arm discovered.
3. **Goal-conditioned primitive-policy control (P3).** The same goal-practice
   data and budget for a manager without explicit options, to test whether the
   option hierarchy itself matters.
4. **Within-episode hard routing (P2).** Revise the active structure during an
   episode and reconstruct it exactly from stored history for the update; this
   is what a single unseen world would need.
5. **Variance assessment before the full study.** Use development variability
   to size the seed count of `configs/study.yaml` instead of assuming ten.

## Hypergraph walkers

6a. **Train the walk for acting, not only for consistency.** The learned edit
    policy reached consistent nodes faster but acted worse; reward the walk for
    the cost (or expected information) of the plan its node implies.
6b. **Optimism with a support threshold.** Under noise no hypothesis is
    eliminated, so the cheapest supported node never changes; require posterior
    support above a threshold before treating a node as plausible.
6c. **Locate the boundary of the amortization result.** Study R replicated
    the Stage D result at scale in its family, with failure noise, but not in a
    deeper family with more alternatives and larger pools (docs/studies/R_replication.md).
    Vary depth, alternatives, pool size and budget slack one at a time; add an
    exact goal-only reference (a constraint solver over connected groups of
    recipes) and an adaptive group-testing control that uses the evidence
    without posterior weights.
6d. **Start walks from the brute-force node** and remove requirements as evidence
    allows, instead of starting from single-candidate guesses.

## Follow-up studies (docs/studies/)

6e. **Markov graphs (study M).** Pairwise weights ranked candidates about as
    well as the joint posterior offline, yet acting with them cost more and
    PageRank propagation added nothing; no further work on proximity ranking
    is planned. Open: a pairwise learner that keeps disjunctive failure
    evidence (stored clauses), which would make it a hypergraph in all but
    name.
6f. **TechTree baselines (study U).** Add the missing blind baselines (width
    or novelty search, flat macro expansion) and a learned agent, so the
    environment measures something about learning rather than only about
    pooled testing.
6g. **Promotion with real costs (study L).** Separate promotion from
    composing remembered sequences (atoms that are not concatenations), add an
    admission criterion and measure the utility problem when discoveries are
    learned controllers rather than exact sequences, and make the adaptive arm
    reach its margin.

## Later extensions

6. **Learned proposals and termination.** Replace fragment mining and
   predicate termination with learned alternatives, charging every candidate
   validation to the budget; compare against the current heuristic.
7. **Learned task generation.** A teacher proposing achievable public goals
   near current competence, with a fixed external test distribution.
8. **Consumable resources and conflicts.** Quantities and delete effects; the
   reference solver, contracts and hypothesis class need extending, since
   monotone reachability no longer holds.
9. **Ordered tasks and deadlines.** Explicit progress variables or automata
   (reward machines), since a prerequisite hyperedge cannot encode order.
10. **Forgetting and changing worlds.** Library budgets, retirement and
    revalidation cost; distinguish a changing environment from changing beliefs.
11. **Navigation.** A shared scripted navigator first, then a learned one, with
    navigation steps counted in option durations.
12. **Offline tool and workflow simulator.** Typed file/data/tool effects with
    mock tools, compared against tool-schema planning.
13. **Continual adaptation track** (`configs/adaptation_stream.yaml`): parameter
    updates and admissions across a logged task stream.
