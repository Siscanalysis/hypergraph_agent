"""Study RL offline benchmark (evaluator side; no interaction).

For every world record, evidence source (policy) and checkpoint k, every
learner (``learners.LEARNERS``) reads the world's first k logs and the
recipes registered by then, and scores each pool candidate of every scored
recipe. The metrics are study X2's, computed by X2's code
(``inference_baselines.benchmark`` and ``evaluation.markov_study``): AUROC
(ties one half), the ranked gathering cost, the precision, recall and F1 of
the inferred set (candidates scoring at least 0.5, else the top-scoring tie
group) and the open-loop acting cost ratio and plan success of the node made
of the inferred sets. A world's value of a per-recipe metric is the mean over
its scored recipes with a hidden requirement.

Sets: ``gen_obs`` (study X2's logs, read through X2's loader; in class,
deterministic), ``slip`` (RL's logs, slip probability 0.1) and ``ooc`` (RL's
logs, requirements of 4 or 5 of 8 candidates, outside the posterior's class).
X2's five benchmark sets (``BENCH_SETS``) are read too and reported
descriptively only, at checkpoint ``BENCH_K`` on ``BENCH_SOURCES`` evidence;
no validity or secondary test reads them (as in X2's S2, the ids of converted
decoys are not exchangeable). The unit is the world; every set's worlds are
used (no selection split). The exploratory ``gplearn`` arm runs only in the
cells ``GPLEARN_CELLS`` (set, source, checkpoint).

Decisions follow study M's four outcomes as X2 does (``decision``); the
primary, the validity tests (V1 to V4), the secondary tests and the
descriptive contrasts are defined in docs/studies/RL_rule_learners.md and
computed here (``aggregate``; V3 and V4 in ``rank_benchmark``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.markov_study import Z_MDD, _ci, auroc, gathering_cost
from ..inference_baselines.benchmark import Cache, acting_metrics, inferred_set, set_metrics
from ..inference_baselines.provenance import code_sha256
from ..inference_baselines.traces import craft_examples
from . import classical, dnf
from .collection import X2_BENCH_SETS
from .learners import CLASSICAL, EXACT, EXPLORATORY, LEARNERS, POSTERIORS, world_scores

CHECKPOINTS = (4, 8, 12, 16)
SETS = ("gen_obs", "slip", "ooc")
SOURCES = ("focused_sample", "random_omit", "maximal")
BENCH_SETS = X2_BENCH_SETS  # X2's converted benchmark worlds: descriptive only
BENCH_K, BENCH_SOURCES = 16, ("focused_sample", "random_omit")
EXPECTED_WORLDS = 60  # per set: seeds 0-4, twelve worlds each
GPLEARN_CELLS = (("ooc", "random_omit", 16),)  # (set, evidence source, checkpoint): the primary cell
MARGINS = {"auroc": 0.03, "f1": 0.05, "act_ratio": 0.05}
LOWER_IS_BETTER = ("act_ratio", "gather")
PRIMARY = {"set": "ooc", "source": "random_omit", "k": 16, "metric": "f1", "a": "dnf", "b": "posterior"}
METRICS = ("auroc", "gather", "precision", "recall", "f1", "act_ratio", "plan_success", "modelled")
PER_WORLD = ("auroc", "f1", "act_ratio", "plan_success")
V2_LEARNERS = POSTERIORS + EXACT + CLASSICAL + ("random",)
V4_LEARNERS, V4_TOLERANCE = ("posterior", "random"), 1e-9
X2_ARTIFACT = "artifacts/inference_baselines/x2_benchmark.json"
LIBRARIES = ("clingo", "python-sat", "wittgenstein", "scikit-learn", "imodels", "gplearn", "numpy", "pandas",
             "scipy")
DIGITS = 9
# secondary tests: (name, set, source, a, b, metric, criterion); criteria: "above" (95% lower end above 0),
# "below" (95% upper end below 0), or a four-way decision outcome; "classical" compares a with each of
# the three classical learners and holds when every interval lies above 0
SECONDARY = (
    ("O1", "ooc", "random_omit", "dnf", "posterior", "act_ratio", "below"),
    ("O3", "ooc", "focused_sample", "dnf", "posterior", "f1", "dnf better"),
    ("O4", "ooc", "random_omit", "dnf", "posterior_full", "f1", "posterior_full better"),
    ("O5", "ooc", "random_omit", "dnf", "random", "f1", "above"),
    ("E2", "gen_obs", "focused_sample", "dnf", "posterior", "f1", "equivalent"),
    ("N2", "slip", "focused_sample", "dnf_soft", "dnf", "f1", "above"),
    ("N3", "slip", "focused_sample", "dnf_soft", "posterior_eps", "f1", "equivalent"),
    ("C1", "gen_obs", "focused_sample", "dnf", "classical", "f1", "above"),
)
# descriptive contrasts: the same interval and four-way decision, no criterion
DESCRIPTIVE = (
    ("E1", "gen_obs", "focused_sample", "dnf", "posterior", "auroc"),
    ("E3", "gen_obs", "focused_sample", "dnf", "posterior", "act_ratio"),
    ("E4", "gen_obs", "random_omit", "dnf", "posterior", "auroc"),
    ("O2", "ooc", "random_omit", "dnf", "posterior", "auroc"),
    ("N1", "slip", "focused_sample", "dnf_soft", "posterior", "f1"),
    ("N4", "slip", "focused_sample", "posterior_eps", "posterior", "auroc"),
    ("C2", "ooc", "random_omit", "dnf", "classical", "f1"),
)


# ================================================================= metrics

def world_metrics(rec: dict, k: int, gplearn: bool = False) -> dict | None:
    """Per-world metrics of every learner at checkpoint ``k`` (None when the
    world has fewer than k logs)."""
    if len(rec["logs"]) < k:
        return None
    res = world_scores(rec, k, gplearn=gplearn)
    truth = {s: set(v) for s, v in rec["truth"]["requirements"].items()}
    sigs = [s for s in res["sigs"] if s in truth]
    acting: dict = {}
    per = {}
    for m, sc in res["scores"].items():
        rows, sets = [], {}
        for s in sigs:
            pool = res["infos"][s].pool
            c = {b: sc[(s, b)] for b in pool}
            S = inferred_set(c, pool)
            sets[s] = S
            rows.append({"auroc": auroc(c, truth[s]), "gather": gathering_cost(c, truth[s]),
                         **set_metrics(S, truth[s]), "modelled": float(s in res["modelled"][m])})
        means = {x: float(np.mean([r[x] for r in rows])) if rows else float("nan")
                 for x in ("auroc", "gather", "precision", "recall", "f1", "modelled")}
        node = tuple(sorted((s, tuple(sorted(v))) for s, v in sets.items()))
        if node not in acting:  # learners with the same inferred node act identically
            acting[node] = acting_metrics(rec, sets)
        per[m] = {**means, **acting[node]}
    diag = res["diagnostics"]
    jobs = diag.pop("dnf_jobs")
    mean = (lambda xs: float(np.mean(xs))) if jobs else (lambda xs: None)
    diag["dnf"] = {"jobs": len(jobs), "unsat": sum(j["cost"] is None for j in jobs),
                   "optimal_rules_mean": mean([j["optimal_rules"] for j in jobs]),
                   "optimal_rules_max": max((j["optimal_rules"] for j in jobs), default=0),
                   "attempts_mean": mean([j["successes"] + j["failures"] for j in jobs]),
                   "disjunctive": sum(j["disjunctive"] for j in jobs),
                   "disjunctive_soft": sum(j["disjunctive_soft"] for j in jobs)}
    return {"world": rec.get("world_label", rec["world_key"]), "set": rec.get("set"), "seed": rec["seed"],
            "source": rec["strategy"], "k": k, "recipes": len(sigs), "epsilon": res["epsilon"],
            "metrics": per, "diagnostics": diag, "complete": rec.get("complete", True)}


def gplearn_here(rec: dict, k: int, cells=GPLEARN_CELLS) -> bool:
    return (rec.get("set"), rec["strategy"], k) in {tuple(c) for c in cells}


def checkpoints_for(rec: dict, checkpoints) -> tuple:
    """Every checkpoint for RL's sets and ``gen_obs``; for X2's benchmark sets
    only ``BENCH_K`` on ``BENCH_SOURCES`` evidence (none on ``maximal``)."""
    if rec.get("set") in BENCH_SETS:
        return (BENCH_K,) if rec["strategy"] in BENCH_SOURCES else ()
    return tuple(checkpoints)


# ============================================================ provenance

def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def options_record(checkpoints, gplearn_cells) -> dict:
    return {"checkpoints": list(checkpoints), "gplearn_cells": [list(c) for c in gplearn_cells],
            "bench": {"sets": list(BENCH_SETS), "k": BENCH_K, "sources": list(BENCH_SOURCES)},
            "max_terms": dnf.MAX_TERMS, "model_cap": dnf.MODEL_CAP, "scale": dnf.SCALE,
            "gplearn": {k: list(v) if isinstance(v, tuple) else v for k, v in classical.GPLEARN.items()},
            "learners": list(LEARNERS)}


def cache_key(options: dict) -> str:
    body = {"study": "rl", "code": code_sha256(), "options": options, "versions": versions()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def provenance_record(key: str, options: dict) -> dict:
    from ..training.run import source_info
    src = source_info(Path(__file__).resolve().parents[3])
    src.pop("_patch", None)
    return {**src, "code": code_sha256(), "versions": versions(), "cache_key": key, "options": options}


class CodeMismatch(RuntimeError):
    """A row was computed with other code than this process runs."""


def _check_code(stamp: str | None, what: str) -> None:
    if stamp is not None and stamp != code_sha256():
        raise CodeMismatch(f"{what} was computed with code {stamp[:12]}, this process runs "
                           f"{code_sha256()[:12]}")


# ================================================================ compute

def row_id(rec: dict, k: int) -> str:
    return f"{rec['run_id']}|{rec['world_key']}|{k}"


def _world_rows(args) -> list:
    rec, checkpoints, cells = args
    code = code_sha256()
    out = []
    for k in checkpoints:
        r = world_metrics(rec, k, gplearn_here(rec, k, cells))
        if r is not None:
            r["code"] = code
        out.append((k, r))
    return out


def _rows_parallel(tasks: list, workers: int, done) -> list:
    """``_world_rows`` of every task in ``workers`` processes; ``done(i, rows)``
    as each finishes; None for a task whose worker failed."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    out: list = [None] * len(tasks)
    try:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(_world_rows, t): i for i, t in enumerate(tasks)}
            for f in as_completed(futs):
                try:
                    out[futs[f]] = f.result()
                except Exception:  # noqa: BLE001 (recomputed in the parent)
                    continue
                done(futs[f], out[futs[f]])
    except BrokenProcessPool:
        pass
    return out


def rank_benchmark(worlds: list[dict], *, checkpoints=CHECKPOINTS, gplearn_cells=GPLEARN_CELLS,
                   workers: int = 1, expected_worlds: int = EXPECTED_WORLDS,
                   cache_dir: str | Path | None = None, resume: bool = False, progress=None,
                   x2_artifact: dict | None = None, x2_chosen=()) -> dict:
    """Every world, source and checkpoint (``checkpoints_for``), then
    ``aggregate``, V3 (``variant_check``) and V4 (``x2_agreement``, against X2's
    summary ``x2_artifact`` when the X2 runs read, ``x2_chosen``, are its
    runs). Rows are cached under ``cache_dir`` (key: code, options, library
    versions) and reused with ``resume``; ``workers`` > 1 computes worlds in
    parallel processes (the same results: every random choice is seeded by its
    job), and a world whose worker fails is recomputed in this process."""
    t0 = time.time()
    options = options_record(checkpoints, gplearn_cells)
    key = cache_key(options)
    cache = Cache(cache_dir, key, resume)
    ks = [checkpoints_for(rec, checkpoints) for rec in worlds]
    results: list = [None] * len(worlds)
    reused = 0
    for i, rec in enumerate(worlds):
        if not ks[i]:
            results[i] = []
            continue
        cached = [(k, cache.get("rows", row_id(rec, k))) for k in ks[i]]
        if all(r is not None for _, r in cached):
            results[i] = [(k, r.get("row")) for k, r in cached]
            for k, r in results[i]:
                if r is not None:
                    _check_code(r.get("code"), f"cached row {row_id(rec, k)}")
                    r["complete"] = rec.get("complete", True)  # the evidence's status now
            reused += 1

    def done(i: int, rows: list) -> None:
        for k, r in rows:
            if r is not None:
                _check_code(r.get("code"), f"row {row_id(worlds[i], k)}")
            cache.put("rows", row_id(worlds[i], k), {"row": r})
            if progress:
                progress(worlds[i], k)

    todo = [i for i, r in enumerate(results) if r is None]
    tasks = {i: (worlds[i], ks[i], tuple(map(tuple, gplearn_cells))) for i in todo}
    fallbacks = 0
    if workers > 1 and len(todo) > 1:
        par = _rows_parallel([tasks[i] for i in todo], workers, lambda j, rows: done(todo[j], rows))
        for j, i in enumerate(todo):
            results[i] = par[j]
    for i in todo:
        if results[i] is None:
            fallbacks += workers > 1 and len(todo) > 1
            results[i] = _world_rows(tasks[i])
            done(i, results[i])
    per = defaultdict(dict)  # (set, source, k) -> world -> row
    for rows in results:
        for k, r in rows:
            if r is not None:
                per[(r["set"], r["source"], k)][r["world"]] = r
    summary = aggregate(per, checkpoints, expected_worlds)
    summary["validity"]["V3"] = variant_check(worlds)
    summary["validity"]["V4"] = x2_agreement(per, x2_artifact, x2_chosen)
    summary["verdict"] = verdict(summary)
    summary.update(rows_reused=reused, parallel_fallbacks=int(fallbacks), seconds=round(time.time() - t0, 1),
                   provenance=provenance_record(key, options))
    return summary


def variant_check(worlds: list[dict]) -> dict:
    """V3 (evaluator side): every hidden requirement of an ``ooc`` world has 4
    or 5 candidates and every other one 1 to 3; ``slip`` logs contain failed
    attempts whose requirement and item inputs were all present (slips), the
    deterministic sets none. X2's benchmark sets are not part of it."""
    sizes, slips = defaultdict(Counter), Counter()
    for rec in worlds:
        st = rec.get("set")
        if st in BENCH_SETS:
            continue
        truth = rec["truth"]["requirements"]
        for s, req in truth.items():
            sizes[st][len(req)] += 1
        for e in craft_examples(rec["logs"], rec["infos"]):
            info = rec["infos"][e.signature]
            if (not e.success and e.signature in truth and set(truth[e.signature]) <= e.present
                    and set(info.items) <= e.present):
                slips[st] += 1
    ok_sizes = all(set(c) <= ({4, 5} if st == "ooc" else {1, 2, 3}) for st, c in sizes.items())
    ok_slips = all((slips[st] > 0) == (st == "slip") for st in sizes)
    return {"sizes": {st: dict(sorted(c.items())) for st, c in sizes.items()}, "slips": dict(slips),
            "test": "ooc requirements of 4-5 candidates, others 1-3; slips only in the slip set",
            "holds": (ok_sizes and ok_slips) if sizes else None}


def _same(a, b) -> float:
    """|a - b|, 0 when both are NaN, infinite when only one is."""
    a_nan, b_nan = a is None or math.isnan(a), b is None or math.isnan(b)
    if a_nan or b_nan:
        return 0.0 if a_nan and b_nan else math.inf
    return abs(a - b)


def x2_agreement(per: dict, artifact: dict | None, x2_chosen=()) -> dict:
    """V4: on ``gen_obs`` RL's per-world values of ``V4_LEARNERS`` equal X2's
    in X2's summary (every source and checkpoint) within ``V4_TOLERANCE``. It
    applies only when the summary covers the X2 runs RL read (otherwise None)."""
    test = (f"gen_obs: per-world {', '.join(PER_WORLD)} of {' and '.join(V4_LEARNERS)} equal X2's "
            f"artifact within {V4_TOLERANCE:g}")
    if artifact is None:
        return {"test": test, "holds": None, "reason": "no X2 summary given"}
    covered = set(x2_chosen) <= set(((artifact.get("runs") or {}).get("chosen")) or ())
    if not x2_chosen or not covered:
        return {"test": test, "holds": None, "reason": "X2's summary covers other runs"}
    xpw = artifact.get("per_world") or {}
    n = missing = 0
    worst = 0.0
    for (st, src, k), ws in per.items():
        if st != "gen_obs":
            continue
        for world, row in ws.items():
            split = "selection" if row["seed"] in (0, 1) else "test" if row["seed"] in (2, 3, 4) else "other"
            entry = xpw.get(f"gen_obs:{split}:{src}@{k}", {}).get(world)
            if entry is None:
                missing += 1
                continue
            for m in V4_LEARNERS:
                for x in PER_WORLD:
                    worst = max(worst, _same(row["metrics"][m][x], entry["metrics"][m][x]))
                    n += 1
    return {"test": test, "values": n, "worlds_missing": missing, "largest_difference": worst,
            "holds": (missing == 0 and worst <= V4_TOLERANCE) if n or missing else None}


# ============================================================== decisions

def decision(diffs, margin: float, lower_is_better: bool = False, names=("a", "b")) -> dict:
    """Study M's four outcomes on per-world differences a - b: "<a> better" if
    the 95% interval lies entirely on a's side of 0, "<b> better" if on b's
    side, otherwise "equivalent" if the 90% interval lies inside (-margin,
    margin), otherwise "inconclusive"; a significant difference takes
    precedence over equivalence."""
    c95, c90 = _ci(diffs), _ci(diffs, 0.90)
    if not c95.get("ci"):
        return {"decision": None, "ci90": None, "margin": margin}
    lo, hi = c95["ci"]
    a_better = hi < 0 if lower_is_better else lo > 0
    b_better = lo > 0 if lower_is_better else hi < 0
    eq = -margin < c90["ci"][0] and c90["ci"][1] < margin
    d = (f"{names[0]} better" if a_better else f"{names[1]} better" if b_better
         else "equivalent" if eq else "inconclusive")
    return {"decision": d, "ci90": c90["ci"], "margin": margin}


def _cell(per: dict, st: str, src: str, k: int) -> list:
    return list(per.get((st, src, k), {}).values())


def _diffs(rows: list, a: str, b: str, metric: str) -> list:
    d = [w["metrics"][a][metric] - w["metrics"][b][metric]
         for w in rows if a in w["metrics"] and b in w["metrics"]]
    return [v for v in d if not math.isnan(v)]


def _complete(rows: list, expected: int) -> bool:
    return len(rows) >= expected and all(w["complete"] for w in rows)


def contrast_test(rows: list, a: str, b: str, metric: str, criterion: str | None, expected: int) -> dict:
    """One contrast on per-world differences a - b; with a criterion, whether
    it holds (None when the cell is incomplete); without one, descriptive."""
    d = _diffs(rows, a, b, metric)
    c = _ci(d)
    dec = decision(d, MARGINS.get(metric, 0.05), metric in LOWER_IS_BETTER, (a, b))
    complete = _complete(rows, expected) and len(d) == len(rows)
    out = {"contrast": f"{a} - {b}", "metric": metric, "n": len(d), **c, "ci90": dec["ci90"],
           "margin": dec["margin"], "decision": dec["decision"], "complete": complete}
    if criterion is None:
        return out
    if criterion == "above":
        holds = c["ci"][0] > 0 if c.get("ci") else None
    elif criterion == "below":
        holds = c["ci"][1] < 0 if c.get("ci") else None
    else:
        holds = None if dec["decision"] is None else dec["decision"] == criterion
    return {**out, "criterion": criterion, "holds": holds if complete else None}


def _contrasts(per: dict, spec, expected: int, k: int) -> dict:
    out = {}
    for name, st, src, a, b, metric, *crit in spec:
        criterion = crit[0] if crit else None
        rows = _cell(per, st, src, k)
        if b == "classical":
            parts = {c: contrast_test(rows, a, c, metric, criterion, expected) for c in CLASSICAL}
            entry = {"cell": f"{st}:{src}@{k}", "contrast": f"{a} - each of {', '.join(CLASSICAL)}",
                     "metric": metric, "parts": parts, "n": min((p["n"] for p in parts.values()), default=0)}
            if criterion is not None:
                hs = [p["holds"] for p in parts.values()]
                entry.update(criterion="every 95% lower end above 0",
                             holds=None if any(h is None for h in hs) else all(hs))
            out[name] = entry
        else:
            out[name] = {"cell": f"{st}:{src}@{k}", **contrast_test(rows, a, b, metric, criterion, expected)}
    return out


def secondary_tests(per: dict, expected: int, k: int = 16) -> dict:
    return _contrasts(per, SECONDARY, expected, k)


def descriptive_contrasts(per: dict, expected: int, k: int = 16) -> dict:
    return _contrasts(per, DESCRIPTIVE, expected, k)


def validity_tests(per: dict) -> dict:
    """V1 (RC2 agrees with clingo on every job; no enumeration stopped at the
    cap) and V2 (maximal evidence on RL's sets and ``gen_obs``: every per-world
    AUROC of the no-information learners is exactly 0.5); V3 and V4 are
    computed in ``rank_benchmark``."""
    jobs = agree = soft_jobs = soft_agree = caps = 0
    for ws in per.values():
        for w in ws.values():
            ag = w["diagnostics"]["agreement"]
            jobs += ag["hard_jobs"]
            agree += ag["hard_agree"]
            soft_jobs += ag["soft_jobs"]
            soft_agree += ag["soft_agree"]
            caps += sum(w["diagnostics"]["rule_status"].get(m, {}).get("cap", 0) for m in EXACT)
    values = [w["metrics"][m]["auroc"] for (st, src, k), ws in per.items()
              if src == "maximal" and st not in BENCH_SETS
              for w in ws.values() for m in V2_LEARNERS if m in w["metrics"]]
    exact = [round(v, DIGITS) == 0.5 for v in values if not math.isnan(v)]
    v1 = (jobs == agree and soft_jobs == soft_agree and caps == 0) if jobs else None
    return {"V1": {"test": "clingo and RC2 give the same optimal cost and the same set of optimal rules in "
                           "every job (hard and soft); no enumeration stopped at the cap",
                   "hard_jobs": jobs, "hard_agree": agree, "soft_jobs": soft_jobs, "soft_agree": soft_agree,
                   "cap_hits": caps, "holds": v1},
            "V2": {"test": "maximal evidence: every per-world AUROC of " + ", ".join(V2_LEARNERS) + " is 0.5",
                   "n": len(exact), "exact_share": float(np.mean(exact)) if exact else None,
                   "holds": all(exact) if exact else None}}


def verdict(summary: dict) -> str | None:
    """The primary decision, None when its data are incomplete, "invalid" when
    a validity test fails (a test that does not apply, None, does not)."""
    if not summary["complete"]:
        return None
    if any(v.get("holds") is False for v in summary["validity"].values()):
        return "invalid"
    return summary["primary_decision"]["decision"]


def aggregate(per: dict, checkpoints, expected_worlds: int = EXPECTED_WORLDS) -> dict:
    table, contrasts, diag = {}, {}, {}
    for key3, ws in sorted(per.items()):
        st, src, kk = key3
        rows = list(ws.values())
        label = f"{st}:{src}@{kk}"
        learners = [m for m in LEARNERS if any(m in w["metrics"] for w in rows)]
        table[label] = {m: {x: _ci([w["metrics"][m][x] for w in rows if m in w["metrics"]]) for x in METRICS}
                        for m in learners}
        table[label].update(n_worlds=len(rows), n_recipes=int(sum(w["recipes"] for w in rows)))
        for a in learners:
            for b in ("posterior", "dnf"):
                if a == b or b not in learners:
                    continue
                for x in ("auroc", "f1", "act_ratio"):
                    d = _diffs(rows, a, b, x)
                    if d:
                        contrasts[f"{label}: {a} - {b} ({x})"] = _ci(d)
        dg = [w["diagnostics"] for w in rows]
        status: dict = defaultdict(Counter)
        seconds: Counter = Counter()
        for d in dg:
            for m, c in d["rule_status"].items():
                status[m].update(c)
            seconds.update(d["seconds"])
        diag[label] = {"rule_status": {m: dict(c) for m, c in status.items()},
                       "seconds": {m: round(v, 2) for m, v in seconds.items()},
                       "posterior_contradictions": {m: int(sum(d[m]["contradiction"] for d in dg))
                                                    for m in POSTERIORS},
                       "dnf_jobs": int(sum(d["dnf"]["jobs"] for d in dg)),
                       "dnf_unsat": int(sum(d["dnf"]["unsat"] for d in dg)),
                       "dnf_disjunctive": int(sum(d["dnf"]["disjunctive"] for d in dg)),
                       "dnf_soft_disjunctive": int(sum(d["dnf"]["disjunctive_soft"] for d in dg)),
                       "dnf_optimal_rules_max": max((d["dnf"]["optimal_rules_max"] for d in dg), default=0)}
    p = PRIMARY
    rows = _cell(per, p["set"], p["source"], p["k"])
    diffs = _diffs(rows, p["a"], p["b"], p["metric"])
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
    dec = decision(diffs, MARGINS[p["metric"]], p["metric"] in LOWER_IS_BETTER, (p["a"], p["b"]))
    complete = _complete(rows, expected_worlds) and len(diffs) == len(rows)

    def entry(r: dict) -> dict:
        return {"seed": r["seed"], "recipes": r["recipes"], "complete": r["complete"],
                "metrics": {m: {x: v[x] for x in PER_WORLD} for m, v in r["metrics"].items()}}

    per_world = {f"{a}:{b}@{c}": {w: entry(r) for w, r in ws.items()}
                 for (a, b, c), ws in sorted(per.items())}
    return {"definition": {
                "learners": list(LEARNERS), "posteriors": list(POSTERIORS), "exact": list(EXACT),
                "classical": list(CLASSICAL), "exploratory": list(EXPLORATORY),
                "checkpoints": list(checkpoints), "descriptive_sets": list(BENCH_SETS),
                "unit": "world (mean over its scored recipes)", "expected_worlds_per_set": expected_worlds,
                "bootstrap": {"resamples": 10_000, "seed": 20261008, "level": 0.95}, "margins": MARGINS,
                "primary": f"{p['metric']} of {p['a']} - {p['b']}, set {p['set']}, {p['source']} evidence, "
                           f"checkpoint {p['k']}, every world",
                "decision": "a better if the 95% interval lies on a's side of 0, b better if on b's side; "
                            "otherwise equivalent if the 90% interval lies inside (-margin, margin); "
                            "otherwise inconclusive"},
            "table": table, "contrasts": contrasts,
            "primary": {**_ci(diffs), "ci90": dec["ci90"], "sd": sd, "n": len(diffs),
                        "mdd": Z_MDD * sd / math.sqrt(len(diffs)) if sd is not None else None},
            "primary_decision": dec, "complete": complete, "validity": validity_tests(per),
            "secondary": secondary_tests(per, expected_worlds, p["k"]),
            "descriptive": descriptive_contrasts(per, expected_worlds, p["k"]), "diagnostics": diag,
            "per_world": per_world}


# ================================================================ reporting

def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _fci(c: dict, nd=3) -> str:
    if not c or c.get("estimate") is None:
        return "-"
    return f"{_f(c['estimate'], nd)} [{_f(c['ci'][0], nd)}, {_f(c['ci'][1], nd)}]"


def _contrast_lines(tests: dict, with_holds: bool) -> list:
    head = "| Test | Cell | Contrast | Metric | Worlds | Estimate [95% CI] | Decision |"
    lines = [head + (" Criterion | Holds |" if with_holds else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|" if with_holds else "")]
    for name, t in tests.items():
        if "parts" in t:
            est, dec_txt = "; ".join(f"{c}: {_fci(v)}" for c, v in t["parts"].items()), "-"
        else:
            est, dec_txt = _fci(t), t["decision"]
        line = f"| {name} | {t['cell']} | {t['contrast']} | {t['metric']} | {t['n']} | {est} | {dec_txt} |"
        lines.append(line + (f" {t['criterion']} | {t['holds']} |" if with_holds else ""))
    return lines


def markdown(s: dict) -> str:
    lines = []
    for metric, nd, title in (("auroc", 3, "AUROC"), ("f1", 3, "F1 of the inferred set"),
                              ("act_ratio", 2, "Acting cost ratio")):
        lines += [f"{title} (mean per world [95% CI]):", "",
                  "| Set:evidence@episodes | Worlds | " + " | ".join(LEARNERS) + " |",
                  "|---|---|" + "---|" * len(LEARNERS)]
        for key, row in s["table"].items():
            cells = " | ".join(_fci(row[m][metric], nd) if m in row else "n/a" for m in LEARNERS)
            lines.append(f"| {key} | {row['n_worlds']} | {cells} |")
        lines.append("")
    p, dec = s["primary"], s["primary_decision"]
    c90 = dec.get("ci90")
    lines += [f"Primary ({s['definition']['primary']}): {_fci(p)}, 90% interval "
              + (f"[{_f(c90[0])}, {_f(c90[1])}]" if c90 else "-")
              + f", margin {dec['margin']:g}, minimum detectable difference {_f(p.get('mdd'))}; decision: "
              f"{dec['decision']}; verdict: {s.get('verdict')}.", "", "Validity tests:", ""]
    for name, v in s["validity"].items():
        lines.append(f"- {name}: {v['test']}; holds: {v['holds']}.")
    lines += ["", "Secondary tests:", ""] + _contrast_lines(s["secondary"], True)
    lines += ["", "Descriptive contrasts:", ""] + _contrast_lines(s.get("descriptive", {}), False)
    return "\n".join(lines) + "\n"
