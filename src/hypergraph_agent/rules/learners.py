"""Every learner of study RL on the scored recipes of one world at one
checkpoint (public data only).

Input, as in study X2: a world's first k logs and the recipes registered by
then (``evaluation.markov_study.infos_at``); the scored recipes are X2's
(``inference_baselines.benchmark.scored_sigs``: the hidden recipes attempted
in those logs). The posteriors read X2's informative attempts
(``inference_baselines.traces.craft_examples``); every rule learner reads
the same attempts restricted to those whose item inputs were all present (an
attempt without them says nothing about the base requirement, which the
posterior's likelihood also encodes), as the pool candidates present before
the attempt and whether the previously absent effect appeared (``attempts``).
The declared slip probability is the evidence's public failure probability
(0 for X2's logs).

Learners (scores of each scored recipe's pool candidates):

``posterior``       X2's factorized posterior (``inference_baselines.learners
                    .factorized_marginals``): uniform prior over the nonempty subsets of
                    at most three candidates, deterministic likelihood; the reference
``posterior_eps``   the same updater (``topology.inference.DependencyModel``) with the
                    declared slip probability; equal to ``posterior`` without slips
``posterior_full``  the same with the declared slip probability and the class of every
                    nonempty subset of the pool, which contains every requirement here
``dnf``             every smallest monotone DNF consistent with all attempts (``dnf``,
                    clingo)
``dnf_rc2``         the same problem solved by MaxSAT (RC2), the cross-check
``dnf_soft``        every optimal DNF of the soft problem with the declared slip
                    probability; identical to ``dnf`` when that probability is 0
``ripper``, ``cart``, ``figs``
                    classifiers decoded into rules (``classical``)
``gplearn``         Boolean symbolic regression (``classical``), exploratory, only in the
                    cells where the analysis enables it
``random``          every candidate tied

A rule learner without a rule for a recipe (no attempt, an unsatisfiable
problem, a classifier without a predicted success) has every candidate at 0
(tied), as X2's set learners; its inferred set is then the full pool.
"""

from __future__ import annotations

import time
from collections import Counter

from ..envs.generator import derive_seed
from ..evaluation.markov_study import infos_at
from ..inference_baselines.benchmark import observable, scored_sigs
from ..inference_baselines.learners import factorized_marginals
from ..inference_baselines.traces import craft_examples
from ..topology.inference import DependencyModel, Evidence
from . import classical, dnf

POSTERIORS = ("posterior", "posterior_eps", "posterior_full")
EXACT = ("dnf", "dnf_rc2", "dnf_soft")
CLASSICAL = ("ripper", "cart", "figs")
EXPLORATORY = ("gplearn",)
LEARNERS = POSTERIORS + EXACT + CLASSICAL + EXPLORATORY + ("random",)
MAX_SIZE = 3  # the posterior's class (X2)


class ItemsUnobservedRecord(ValueError):
    """A record whose logs do not show the outcome of every craft (goal-only)."""


def attempts(infos: dict, examples, sig: str) -> tuple[list, list]:
    """(present sets of the successes, of the failures) of one recipe: its
    informative attempts with every item input present, restricted to its pool."""
    info = infos[sig]
    items, pool = set(info.items), frozenset(info.pool)
    succ, fail = [], []
    for e in examples:
        if e.signature == sig and items <= e.present:
            (succ if e.success else fail).append(e.present & pool)
    return succ, fail


def posterior_scores(infos: dict, examples, sigs, epsilon: float, max_size: int | None) -> tuple[dict, dict]:
    """Marginals of the per-recipe updater with slip probability ``epsilon``
    and hypotheses of at most ``max_size`` candidates (None: the whole pool);
    with (0, 3) this is X2's ``factorized_marginals``."""
    big = max((len(infos[s].pool) for s in sigs), default=1)
    dm = DependencyModel("rl", epsilon, big if max_size is None else max_size)
    for s in sigs:
        dm.register_rule(s, infos[s].items, infos[s].pool)
    diag = {"informative": 0, "uninformative": 0, "contradiction": 0}
    for e in examples:
        if e.signature in dm.rules:
            diag[dm.apply(Evidence(f"{e.log_id}:{e.t}", "rl", e.signature, e.present, e.success))] += 1
    scores = {}
    for s in sigs:
        m = dm.rules[s].marginals()
        scores.update({(s, b): m[b] for b in infos[s].pool})
    return scores, diag


def job_seed(rec: dict, k: int, sig: str, learner: str) -> int:
    return derive_seed("rl_learner", rec.get("set"), rec["strategy"], rec["seed"], rec["world_key"], k, sig,
                       learner) % 2 ** 31


def world_scores(rec: dict, k: int, *, gplearn: bool = False) -> dict:
    """Scores of every learner for the scored recipes of one world record at
    checkpoint ``k``, with the per-recipe rule results and diagnostics."""
    logs = rec["logs"][:k]
    infos = infos_at(rec, k)
    if not observable(rec, logs, infos):
        label = rec.get("world_label", rec["world_key"])
        raise ItemsUnobservedRecord(f"{label}: craft outcomes not observed")
    sigs = scored_sigs(infos, logs)
    eps = float(rec.get("failure_prob") or 0.0)
    ex = craft_examples(logs, infos)
    scores, modelled, diag, seconds = {}, {}, {}, Counter()
    t0 = time.perf_counter()
    scores["posterior"], diag["posterior"] = factorized_marginals(infos, ex, sigs, max_size=MAX_SIZE)
    seconds["posterior"] += time.perf_counter() - t0
    for name, size in (("posterior_eps", MAX_SIZE), ("posterior_full", None)):
        t0 = time.perf_counter()
        scores[name], diag[name] = posterior_scores(infos, ex, sigs, eps, size)
        seconds[name] += time.perf_counter() - t0
    for name in POSTERIORS:
        modelled[name] = set(sigs)
    rule_names = EXACT + CLASSICAL + (EXPLORATORY if gplearn else ())
    for name in rule_names + ("random",):
        scores[name], modelled[name] = {}, set()
    status = {name: Counter() for name in rule_names}
    jobs = []
    for s in sigs:
        pool = infos[s].pool
        succ, fail = attempts(infos, ex, s)
        job = {"sig": s, "successes": len(succ), "failures": len(fail)}
        hard = dnf.make_problem(pool, succ, fail)
        res = {"dnf": dnf.solve_clingo(hard), "dnf_rc2": dnf.solve_rc2(hard)}
        job["hard_agree"] = dnf.agree(res["dnf"], res["dnf_rc2"])
        if eps > 0:
            soft = dnf.make_problem(pool, succ, fail, epsilon=eps)
            res["dnf_soft"], soft_rc2 = dnf.solve_clingo(soft), dnf.solve_rc2(soft)
            job["soft_agree"] = dnf.agree(res["dnf_soft"], soft_rc2)
            seconds["dnf_soft_rc2"] += soft_rc2.seconds
        else:
            res["dnf_soft"] = res["dnf"]  # the soft problem with slip probability 0 is the hard one
        for name in EXACT:
            r = res[name]
            rules = r.rules if r.status != "unsat" else ()
            status[name][r.status] += 1
            seconds[name] += r.seconds if (name != "dnf_soft" or eps > 0) else 0.0
            if rules:
                modelled[name].add(s)
            scores[name].update({(s, b): v for b, v in dnf.rule_scores(rules, pool).items()})
        job["optimal_rules"] = len(res["dnf"].rules)
        job["optimal_rules_soft"] = len(res["dnf_soft"].rules)
        job["cost"], job["cost_soft"] = res["dnf"].cost, res["dnf_soft"].cost
        # diagnostic: some optimal rule of the recipe is a disjunction of two terms
        job["disjunctive"] = any(len(r) > 1 for r in res["dnf"].rules)
        job["disjunctive_soft"] = any(len(r) > 1 for r in res["dnf_soft"].rules)
        for name in CLASSICAL + (EXPLORATORY if gplearn else ()):
            out = classical.classical_rule(name, pool, succ, fail, job_seed(rec, k, s, name))
            status[name][out["status"]] += 1
            seconds[name] += out["seconds"]
            rules = [out["rule"]] if out["rule"] is not None else []
            if rules:
                modelled[name].add(s)
            scores[name].update({(s, b): v for b, v in dnf.rule_scores(rules, pool).items()})
        scores["random"].update({(s, b): 0.0 for b in pool})
        jobs.append(job)
    diag["rule_status"] = {name: dict(c) for name, c in status.items()}
    diag["seconds"] = {name: round(v, 4) for name, v in seconds.items()}
    diag["dnf_jobs"] = jobs
    diag["agreement"] = {"hard_jobs": len(jobs), "hard_agree": sum(j["hard_agree"] for j in jobs),
                         "soft_jobs": sum("soft_agree" in j for j in jobs),
                         "soft_agree": sum(j.get("soft_agree", False) for j in jobs)}
    return {"sigs": sigs, "infos": infos, "scores": scores, "modelled": modelled, "diagnostics": diag,
            "epsilon": eps}
