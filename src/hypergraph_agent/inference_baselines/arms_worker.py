"""Worker process for macq's ARMS (started by ``macq_adapter.run_arms``).

    python -m hypergraph_agent.inference_baselines.arms_worker <jobs.json>

Runs every job of the file in order and writes one JSON line per job to
standard output: ``{"id", "status": "ok" | "error" | "iteration_cap", "precond":
{action: [fact types]}, "seconds", "iterations", "soft_clauses",
"weights_moved", "code"}``. ARMS's own prints go to a discarded buffer. The
parent sets ``PYTHONHASHSEED`` so the result does not depend on the hash seed.

Three changes are applied to the libraries at run time (``install``), which
are unchanged otherwise:

- Weights follow their clauses. In macq 0.3.11 ``ARMS.step3`` builds the
  MAX-SAT problem as ``And(list(constraints_w_weights))``, whose clauses nnf
  keeps in a frozenset, and passes ``list(constraints_w_weights.values())``,
  in insertion order, as the weights; ``to_wcnf`` then pairs the i-th clause
  of the frozenset with the i-th weight, so most soft clauses receive another
  clause's weight. The worker passes each clause the weight ``step3``
  assigned to it (``aligned_weights``) and counts the clauses whose weight
  the released code would have changed (``weights_moved``).
- Empty constraint groups. ``ARMS._calculate_support_rates`` computes
  ``max([])`` when the I3 or the plan constraint group is empty (no relation
  or no action pair reaches the minimum support), which raises; the guard
  returns no weights for an empty group, which is what the algorithm's
  definition gives (no constraints of that group).
- Reproducibility. nnf names the auxiliary variables of its CNF conversion
  with ``os.urandom`` and hashes an AND or OR node with its class object,
  whose hash is its memory address; both change the clause order, the
  variable numbering and therefore the optimum the MAX-SAT solver returns
  among equally weighted ones from process to process. The worker draws the
  names from a generator seeded per job and hashes nodes with the class name
  instead (the problem is unchanged up to variable names and order), so a
  job's result is reproducible.

ARMS repeats its MAX-SAT step until every action counts as fully learned
(its precondition, add or delete list reaches ``upper_bound`` relations);
an action that never does keeps the loop going, solving the same problem
again (seen in development). A job therefore stops after ``ARMS_ITERATIONS``
MAX-SAT steps, with status ``iteration_cap`` and the model as learned so far
(every action's precondition after its last step), a deterministic limit;
the parent's wall-clock limit is only a safety net.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
import time

ARMS_ITERATIONS = 50  # MAX-SAT steps per job (deterministic limit)
_STATS = {"iterations": 0, "soft_clauses": 0, "weights_moved": 0}
_SEEN: dict = {}  # action name -> its LearnedAction, updated in place by ARMS.step5


class IterationCap(RuntimeError):
    """ARMS needed more MAX-SAT steps than ``ARMS_ITERATIONS``."""


def aligned_weights(problem, children: list, weights: list) -> list:
    """The weights in the order ``to_wcnf`` encodes the clauses of ``problem``
    (its iteration order), each the weight ``step3`` gave its clause:
    ``children`` and ``weights`` are step3's clauses and weights in insertion
    order."""
    by_clause = dict(zip(children, weights))
    return [by_clause[c] for c in problem]


def _guard_empty_support(ARMS) -> None:
    orig = ARMS._calculate_support_rates
    if getattr(orig, "_x2_guard", False):
        return

    def guarded(counts, threshold, default):
        return orig(counts, threshold, default) if counts else []

    guarded._x2_guard = True
    ARMS._calculate_support_rates = staticmethod(guarded)


def _pair_weights(arms_mod) -> None:
    """Wrap ``ARMS.step3`` so that every soft clause gets its own weight, and
    count the MAX-SAT steps of a job (``IterationCap`` beyond the limit)."""
    orig = arms_mod.ARMS.step3
    if getattr(orig, "_x2_pairing", False):
        return

    def step3(*args, **kwargs):
        if _STATS["iterations"] >= ARMS_ITERATIONS:  # ``iterations`` counts the steps run
            raise IterationCap(f"more than {ARMS_ITERATIONS} MAX-SAT steps")
        _STATS["iterations"] += 1
        made: dict = {}
        real_and, real_to_wcnf = arms_mod.And, arms_mod.to_wcnf

        def and_(children):
            children = list(children)
            node = real_and(children)
            made[id(node)] = (node, children)
            return node

        def to_wcnf(problem, weights, *a, **kw):
            node, children = made[id(problem)]
            assert node is problem and len(children) == len(weights)
            fixed = aligned_weights(problem, children, list(weights))
            _STATS["soft_clauses"] += len(fixed)
            _STATS["weights_moved"] += sum(w != f for w, f in zip(weights, fixed))
            return real_to_wcnf(problem, fixed, *a, **kw)

        arms_mod.And, arms_mod.to_wcnf = and_, to_wcnf
        try:
            return orig(*args, **kwargs)
        finally:
            arms_mod.And, arms_mod.to_wcnf = real_and, real_to_wcnf

    step3._x2_pairing = True
    arms_mod.ARMS.step3 = staticmethod(step3)


def _track_actions(arms_mod) -> None:
    """Keep every action ``step5`` updates, so a stopped job keeps its model."""
    orig = arms_mod.ARMS.step5
    if getattr(orig, "_x2_track", False):
        return

    def step5(model, actions, *args, **kwargs):
        _SEEN.update({a.name: a for a in actions})
        return orig(model, actions, *args, **kwargs)

    step5._x2_track = True
    arms_mod.ARMS.step5 = staticmethod(step5)


def _seed_aux_names(seed: int) -> None:
    import nnf
    rng = random.Random(seed)
    nnf.Var.aux = staticmethod(lambda: nnf.Var(nnf.Aux(int=rng.getrandbits(128), version=4)))
    nnf.Internal.__hash__ = lambda self: hash((self.__class__.__name__, self.children))


def install() -> None:
    """Apply the run-time changes and the step limit (idempotent)."""
    import macq.extract.arms as arms_mod
    _guard_empty_support(arms_mod.ARMS)
    _pair_weights(arms_mod)
    _track_actions(arms_mod)


def run_job(job: dict) -> dict:
    from macq.extract import Extract, modes
    from macq.observation import PartialObservation

    from .macq_adapter import macq_tracelist, parse_fluents
    from .provenance import code_sha256
    install()
    random.seed(0)  # macq shuffles fluents before hiding a share of them; none is hidden here
    _seed_aux_names(0)
    _STATS.update(iterations=0, soft_clauses=0, weights_moved=0)
    _SEEN.clear()
    t0 = time.time()
    out = {"id": job["id"], "code": code_sha256()}
    try:
        tl = macq_tracelist(job["traces"], job["fluents"], with_object=True)
        with contextlib.redirect_stdout(io.StringIO()):
            obs = tl.tokenize(PartialObservation, percent_missing=0.0)
            model = Extract(obs, modes.ARMS, **job["settings"])
        out.update(status="ok", precond={a.name: parse_fluents(a.precond) for a in model.actions})
    except IterationCap as exc:  # the model as learned so far
        out.update(status="iteration_cap", precond={n: parse_fluents(a.precond) for n, a in _SEEN.items()},
                   error=str(exc))
    except Exception as exc:  # reported per job; the study counts these
        out.update(status="error", precond={}, error=f"{type(exc).__name__}: {exc}")
    return {**out, **_STATS, "seconds": round(time.time() - t0, 3)}


def main(argv=None) -> int:
    from .macq_adapter import _quiet_macq
    _quiet_macq()
    args = sys.argv[1:] if argv is None else argv
    jobs = json.loads(open(args[0], encoding="utf-8").read())
    out = sys.stdout
    for job in jobs:
        out.write(json.dumps(run_job(job)) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
