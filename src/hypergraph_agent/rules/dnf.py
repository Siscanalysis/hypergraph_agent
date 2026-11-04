"""Smallest monotone DNF consistent with a recipe's attempts (study RL).

Problem. One recipe: its public pool of candidate base facts and its
labelled attempts, each the set of candidates present before an informative
attempt whose item inputs were all present, and whether the previously
absent effect appeared. A rule is a monotone DNF over the pool: an OR of at
most ``max_terms`` nonempty AND terms of candidates (positive literals only:
in RecipeQuest a present fact can enable a craft, an absent one never does).
A rule covers an attempt when one of its terms is a subset of the present
candidates.

``hard`` (epsilon 0)     every success covered and no failure covered; the cost
                         is the number of literals (the smallest consistent rule)
``soft`` (epsilon > 0)   every success covered (an ineligible craft never
                         succeeds, so a success is never noise); a covered failure
                         is explained as a slip of probability epsilon. The cost is
                         the negative log posterior up to a constant: each literal
                         costs ln |pool| nats (a description-length prior) and each
                         covered failure ln(1 / epsilon) nats (the slip
                         likelihood), scaled by ``SCALE`` and rounded to integers

Both solvers return every optimal rule (all rules of minimum cost, as a set of
canonical DNFs: terms sorted, candidates sorted), so ties are reported, not
broken:

``solve_clingo``  answer-set programming (clingo, ``--opt-mode=optN``); used term
                  slots come first and their bit values increase strictly, so
                  each rule has exactly one answer set
``solve_rc2``     weighted partial MaxSAT (python-sat's RC2); every optimum found
                  is blocked in each of its slot placements, and the search
                  stops at the first model whose cost exceeds the optimum

A candidate's score under one rule is the share of the rule's terms that
contain it (MSGI's convention in study X2); under several optimal rules, the
mean over them (``rule_scores``). The enumeration cap ``MODEL_CAP`` cannot
bind in study RL: with pools of six there are 1,414 rules of at most two
terms without an absorbed term in all (63 single terms, 1,351 pairs), and with
pools of eight (requirements of 4 or 5 candidates, no slips) the true
requirement is itself a consistent rule, so an optimal rule has at most five
literals, and there are 2,752 rules of at most five literals (1,736 of five).
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass

MAX_TERMS = 2
MODEL_CAP = 5000
SCALE = 1000
STATUSES = ("optimal", "cap", "unsat")


def canonical(terms) -> tuple:
    """A DNF as a sorted tuple of sorted terms (the empty term stays, as ())."""
    return tuple(sorted(tuple(sorted(t)) for t in terms))


@dataclass(frozen=True)
class Problem:
    pool: tuple  # candidate type ids, sorted
    positives: tuple  # distinct present sets of the successes (frozensets)
    negatives: tuple  # (present set, multiplicity) of the failures, distinct states
    epsilon: float = 0.0  # declared slip probability; 0: every failure is a hard constraint
    max_terms: int = MAX_TERMS

    @property
    def soft(self) -> bool:
        return self.epsilon > 0

    def weights(self) -> tuple[int, int | None]:
        """(cost of one literal, cost of one covered failure); hard: (1, None)."""
        if not self.soft:
            return 1, None
        return round(SCALE * math.log(len(self.pool))), round(SCALE * math.log(1.0 / self.epsilon))


def make_problem(pool, successes, failures, epsilon: float = 0.0, max_terms: int = MAX_TERMS) -> Problem:
    """A problem from present sets (restricted to the pool); failures keep
    their multiplicity, which only the soft cost uses."""
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    pool = tuple(sorted(pool))
    inside = set(pool)
    pos = {frozenset(c for c in x if c in inside) for x in successes}
    neg = Counter(frozenset(c for c in x if c in inside) for x in failures)
    return Problem(pool, tuple(sorted(pos, key=sorted)),
                   tuple(sorted(neg.items(), key=lambda kv: sorted(kv[0]))), float(epsilon), max_terms)


@dataclass(frozen=True)
class Result:
    solver: str
    status: str  # "optimal" (every optimal rule), "cap" (enumeration stopped) or "unsat"
    cost: int | None
    rules: tuple  # canonical DNFs, sorted
    seconds: float

    def record(self) -> dict:
        return {"solver": self.solver, "status": self.status, "cost": self.cost,
                "rules": [list(map(list, r)) for r in self.rules], "seconds": round(self.seconds, 6)}


def covers(rule, present) -> bool:
    return any(set(t) <= present for t in rule)


def rule_scores(rules, pool) -> dict:
    """Mean over ``rules`` of the share of a rule's terms containing each
    candidate; every candidate at 0 when there is no rule."""
    out = {b: 0.0 for b in pool}
    for r in rules:
        for t in r:
            for b in t:
                if b in out:
                    out[b] += 1.0 / (len(r) * len(rules))
    return out


# ------------------------------------------------------------------ clingo

def asp_program(p: Problem) -> str:
    """The answer-set program of a problem (candidates by their pool index)."""
    n = len(p.pool)
    wl, wn = p.weights()
    out = ["#defined pos/1. #defined neg/2. #defined has/2.",
           f"slot(1..{p.max_terms}). cand(0..{n - 1}).",
           "w(C, 2**C) :- cand(C).",
           "{ in(T,C) : cand(C) } :- slot(T).",
           "used(T) :- in(T,C).",
           ":- not used(1).",
           ":- used(T), T > 1, not used(T-1).",
           "val(T,V) :- slot(T), V = #sum { W,C : in(T,C), w(C,W) }.",
           ":- used(T+1), slot(T), val(T,V1), val(T+1,V2), V1 >= V2.",
           "ex(I) :- pos(I).",
           "ex(I) :- neg(I,_).",
           "miss(T,I) :- slot(T), ex(I), in(T,C), not has(I,C).",
           "cov(I) :- ex(I), used(T), not miss(T,I).",
           ":- pos(I), not cov(I).",
           f"#minimize {{ {wl},lit,T,C : in(T,C) }}."]
    out.append(f"#minimize {{ {wn}*M,slip,I : neg(I,M), cov(I) }}." if p.soft else ":- neg(I,_), cov(I).")
    out.append("#show in/2.")
    idx = {c: j for j, c in enumerate(p.pool)}
    i = 0
    for x in p.positives:
        out.append(f"pos({i}). " + " ".join(f"has({i},{idx[c]})." for c in sorted(x)))
        i += 1
    for x, m in p.negatives:
        out.append(f"neg({i},{m}). " + " ".join(f"has({i},{idx[c]})." for c in sorted(x)))
        i += 1
    return "\n".join(out) + "\n"


def solve_clingo(p: Problem, cap: int = MODEL_CAP) -> Result:
    import clingo
    t0 = time.perf_counter()
    ctl = clingo.Control(["0", "--opt-mode=optN", "--parallel-mode=1"], logger=lambda code, msg: None)
    ctl.add("base", [], asp_program(p))
    ctl.ground([("base", [])])
    rules, cost, capped = set(), [None], [False]

    def on_model(m) -> bool:
        if not m.optimality_proven:
            return True
        terms: dict = {}
        for s in m.symbols(shown=True):
            t, j = (a.number for a in s.arguments)
            terms.setdefault(t, set()).add(p.pool[j])
        rules.add(canonical(terms.values()))
        cost[0] = sum(m.cost)
        if len(rules) >= cap:
            capped[0] = True
            return False
        return True

    res = ctl.solve(on_model=on_model)
    status = "unsat" if res.unsatisfiable else "cap" if capped[0] else "optimal"
    return Result("clingo", status, cost[0] if status != "unsat" else None, tuple(sorted(rules)),
                  time.perf_counter() - t0)


# --------------------------------------------------------------------- RC2

def maxsat(p: Problem):
    """(WCNF, {(slot, pool index): variable}) of a problem."""
    from pysat.formula import WCNF
    n, k = len(p.pool), p.max_terms
    var = itertools.count(1)
    x = {(t, j): next(var) for t in range(k) for j in range(n)}
    u = [next(var) for _ in range(k)]
    w = WCNF()
    for t in range(k):  # u[t] <-> some literal in term t
        for j in range(n):
            w.append([-x[t, j], u[t]])
        w.append([-u[t]] + [x[t, j] for j in range(n)])
    w.append(list(u))  # at least one term (requirements are publicly nonempty)
    for present in p.positives:  # some used term inside the present set
        s = [next(var) for _ in range(k)]
        w.append(s)
        for t in range(k):
            w.append([-s[t], u[t]])
            w.extend([[-s[t], -x[t, j]] for j in range(n) if p.pool[j] not in present])
    wl, wn = p.weights()
    for present, m in p.negatives:  # every used term has a literal absent from the failure
        absent = [j for j in range(n) if p.pool[j] not in present]
        r = next(var) if p.soft else None
        for t in range(k):
            w.append([-u[t]] + [x[t, j] for j in absent] + ([r] if r else []))
        if r:
            w.append([-r], weight=wn * m)
    for key in sorted(x):
        w.append([-x[key]], weight=wl)
    return w, x


def placements(rule, x: dict, pool, max_terms: int):
    """The variable assignments of every placement of a rule's terms in the slots."""
    n = len(pool)
    for slots in itertools.permutations(range(max_terms), len(rule)):
        member = {t: set() for t in range(max_terms)}
        for term, t in zip(rule, slots):
            member[t] = set(term)
        yield [x[t, j] if pool[j] in member[t] else -x[t, j] for t in range(max_terms) for j in range(n)]


def solve_rc2(p: Problem, cap: int = MODEL_CAP) -> Result:
    from pysat.examples.rc2 import RC2
    t0 = time.perf_counter()
    w, x = maxsat(p)
    rules, best, capped = set(), None, False
    with RC2(w) as rc2:
        while True:
            model = rc2.compute()
            if model is None or (best is not None and rc2.cost > best):
                break
            best = rc2.cost if best is None else best
            true = {lit for lit in model if lit > 0}
            terms = [[p.pool[j] for j in range(len(p.pool)) if x[t, j] in true] for t in range(p.max_terms)]
            rule = canonical(t for t in terms if t)
            rules.add(rule)
            for assignment in placements(rule, x, p.pool, p.max_terms):
                rc2.add_clause([-lit for lit in assignment])
            if len(rules) >= cap:
                capped = True
                break
    status = "unsat" if best is None else "cap" if capped else "optimal"
    return Result("rc2", status, best, tuple(sorted(rules)), time.perf_counter() - t0)


def agree(a: Result, b: Result) -> bool:
    """Same status, optimal cost and set of optimal rules (an enumeration
    stopped at the cap is never compared as agreeing)."""
    return a.status == b.status != "cap" and a.cost == b.cost and a.rules == b.rules
