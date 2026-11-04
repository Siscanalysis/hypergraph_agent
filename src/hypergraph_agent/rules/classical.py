"""Classical rule and tree learners and exploratory Boolean symbolic regression (study RL).

Each learner is a classifier fitted to one recipe's attempts: the features are
the pool candidates (present 1, absent 0, in pool order), the label is
whether the previously absent effect appeared. Libraries (pinned in
docs/studies/RL_rule_learners.md):

``ripper``   RIPPER (Cohen 1995), wittgenstein 0.3.5, library defaults (k 2, prune
             size 0.33)
``cart``     CART, scikit-learn 1.9.1 ``DecisionTreeClassifier``, defaults (gini, grown
             until its leaves are pure)
``figs``     FIGS, imodels 3.0.4 ``FIGSClassifier``, defaults (at most 12 rules)
``gplearn``  Boolean symbolic regression, gplearn 0.4.3 ``SymbolicClassifier`` with the
             settings ``GPLEARN`` (exploratory arm only)

A classifier is fitted only when the recipe's attempts contain both outcomes.
With successes only it predicts success everywhere, which identifies no
requirement (its rule is the empty term); with failures only, or with no
attempt, it has no rule. Every random choice is seeded per job (``seed``).
FIGS (imodels 3.0.4) keeps its ``random_state`` but never passes it on: its
split stumps are scikit-learn trees without a seed, which break ties between
equally good splits with numpy's global generator, seeded from the operating
system in every process. Its fit therefore runs with that generator seeded by
the job and restored afterwards (``_global_numpy_seed``), so its rule depends
on the job alone.

Decoding (the same for every classifier): its own decision (``predict``) on
every subset of the pool; the rule is the monotone DNF whose terms are the
minimal subsets predicted to succeed (each predicted success is covered by a
term, and no term is a strict superset of another). A rule whose only term is
empty predicts success with no candidate present. A classifier that predicts
failure on every subset has no rule. Scores then follow ``dnf.rule_scores``,
as for the exact learners.
"""

from __future__ import annotations

import contextlib
import itertools
import time
import warnings

from .dnf import canonical

LEARNERS = ("ripper", "cart", "figs", "gplearn")
GPLEARN = {"population_size": 200, "generations": 10, "tournament_size": 20,
           "function_set": ("min", "max", "mul", "add", "sub"), "parsimony_coefficient": 0.01,
           "init_depth": (1, 4), "n_jobs": 1}


def subsets(pool) -> list[tuple]:
    """Every subset of the pool, by size, then lexicographically."""
    pool = tuple(sorted(pool))
    return [c for k in range(len(pool) + 1) for c in itertools.combinations(pool, k)]


def minimal_rule(states, predicted) -> tuple | None:
    """The monotone DNF of the minimal predicted-success states, or None."""
    wins = [frozenset(s) for s, p in zip(states, predicted) if p]
    minimal = [s for s in wins if not any(o < s for o in wins)]
    return canonical(minimal) if minimal else None


def _rows(pool, sets) -> list[list[int]]:
    return [[int(c in s) for c in pool] for s in sets]


@contextlib.contextmanager
def _global_numpy_seed(seed: int):
    """numpy's global generator seeded by ``seed`` inside the block, its
    previous state restored afterwards."""
    import numpy as np
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _fit_predict(name: str, X, y, seed: int, grid):
    if name == "ripper":
        import pandas as pd
        import wittgenstein as lw
        cols = [f"c{j}" for j in range(len(X[0]))]
        clf = lw.RIPPER(random_state=seed)
        clf.fit(pd.DataFrame(X, columns=cols), pd.Series(y, name="success"), pos_class=1)
        return [bool(v) for v in clf.predict(pd.DataFrame(grid, columns=cols))]
    if name == "cart":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=seed).fit(X, y)
        return [int(v) == 1 for v in clf.predict(grid)]
    if name == "figs":
        from imodels import FIGSClassifier
        clf = FIGSClassifier(random_state=seed)  # kept for the record; imodels does not use it
        with _global_numpy_seed(seed):
            clf.fit(X, y)
        return [int(v) == 1 for v in clf.predict(grid)]
    if name == "gplearn":
        from gplearn.genetic import SymbolicClassifier
        clf = SymbolicClassifier(random_state=seed, **GPLEARN)
        clf.fit([[float(v) for v in r] for r in X], y)
        return [int(v) == 1 for v in clf.predict([[float(v) for v in r] for r in grid])]
    raise ValueError(f"unknown learner {name!r}")


def classical_rule(name: str, pool, successes, failures, seed: int) -> dict:
    """{"rule": canonical DNF or None, "status", "seconds"} of one learner on
    one recipe's attempts (present sets restricted to the pool)."""
    t0 = time.perf_counter()
    pool = tuple(sorted(pool))
    if not successes and not failures:
        rule, status = None, "no_examples"
    elif not failures:
        rule, status = ((),), "successes_only"
    elif not successes:
        rule, status = None, "failures_only"
    else:
        states = subsets(pool)
        X = _rows(pool, list(successes) + list(failures))
        y = [1] * len(successes) + [0] * len(failures)
        try:
            with warnings.catch_warnings():  # restores showwarning, which wittgenstein calls directly
                warnings.simplefilter("ignore")
                warnings.showwarning = lambda *args, **kwargs: None
                pred = _fit_predict(name, X, y, seed, _rows(pool, [set(s) for s in states]))
        except Exception as exc:  # noqa: BLE001 (recorded per job; the recipe has no rule)
            return {"rule": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200],
                    "seconds": time.perf_counter() - t0}
        rule = minimal_rule(states, pred)
        status = "never" if rule is None else "always" if rule == ((),) else "fitted"
    return {"rule": rule, "status": status, "seconds": time.perf_counter() - t0}
