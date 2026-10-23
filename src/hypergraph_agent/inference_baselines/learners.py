"""Inference methods that need no third-party code (public data only).

``factorized_marginals``  the repository's per-recipe updater (``topology.inference``):
                          with items observable every informative attempt is a labelled
                          example of one recipe, so the posterior over complete
                          hypergraphs factorizes by recipe and this is its exact form
``sam_preconditions``     a SAM-style safe learner, reimplemented here (not the authors'
                          code): the precondition of an action is the intersection of the
                          states before its successful applications (Juba, Le and Stern,
                          KR 2021, for grounded STRIPS actions), so it never contains a
                          fact that was absent at a success; an action never applied
                          successfully has no model (SAM treats it as inapplicable)
"""

from __future__ import annotations

from ..envs.public_schema import CRAFT
from ..topology.inference import DependencyModel, Evidence


def factorized_marginals(infos: dict, examples, sigs, *, max_size: int = 3) -> tuple[dict, dict]:
    """Marginal P(b in B_r) for the recipes ``sigs`` from the informative attempts
    ``examples`` (``traces.CraftExample``), with the uniform class prior."""
    dm = DependencyModel("x2", 0.0, max_size)
    for s in sigs:
        dm.register_rule(s, infos[s].items, infos[s].pool)
    diag = {"informative": 0, "uninformative": 0, "contradiction": 0}
    for e in examples:
        if e.signature in dm.rules:
            diag[dm.apply(Evidence(f"{e.log_id}:{e.t}", "x2", e.signature, e.present, e.success))] += 1
    scores = {}
    for s in sigs:
        m = dm.rules[s].marginals()
        scores.update({(s, b): m[b] for b in infos[s].pool})
    return scores, diag


def sam_preconditions(steps_by_log) -> dict:
    """{signature: facts present before every successful craft of it} from
    ``traces.successful_steps`` of each log."""
    pre: dict = {}
    for steps in steps_by_log:
        for s in steps:
            if s.kind == CRAFT:
                pre[s.target] = s.before if s.target not in pre else pre[s.target] & s.before
    return pre


def set_scores(infos: dict, sigs, sets: dict) -> tuple[dict, set]:
    """Scores of a set-valued learner: 1 for a pool candidate inside the learned
    set of its recipe, 0 outside; a recipe without a model has every candidate
    at 0 (tied). Returns (scores, recipes with a model)."""
    scores, modelled = {}, set()
    for s in sigs:
        learned = sets.get(s)
        if learned is not None:
            modelled.add(s)
        scores.update({(s, b): float(learned is not None and b in learned) for b in infos[s].pool})
    return scores, modelled
