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
