"""Study X2 offline benchmark (evaluator side; no interaction).

For every world record, evidence source (policy) and checkpoint k, every
method reads the same logs, the world's first k episodes, and the recipes
registered by then, and scores each pool candidate of every hidden recipe
attempted in those logs (the scored recipes):

``posterior``    the repository's joint posterior over complete hypergraphs, marginals. With
                 items observable it factorizes by recipe and is computed exactly in that
                 form (``learners.factorized_marginals``, the per-recipe updater); with
                 goal-only logs by study M's enumeration
                 (``evaluation.markov_study.posterior_marginals``: exact per connected group
                 of recipes within its work budget, Gibbs beyond)
``pair_weight``  study M's online pairwise weights (``agents.markov.PairwiseEvidence``)
``msgi``         MSGI's precondition inference, the authors' code (``msgi_adapter``); items
                 observable
``observer``     macq's Observer (``macq_adapter``); items observable
``arms``         macq's ARMS in a worker process (``macq_adapter``); items observable
``sam``          SAM-style safe learner, reimplemented (``learners``); items observable
``random``       every candidate tied

With goal-only observation the outcome of crafting an intermediate item is
not observed, so the methods marked "items observable" have no input and are
not run (``ITEMS_ONLY``). A set-valued learner scores a candidate 1 inside its
learned precondition and 0 outside; a recipe it has no model for has every
candidate at 0 (tied).

Metrics against the hidden requirement T of each scored recipe: AUROC (ties
one half), the ranked gathering cost (``markov_study.gathering_cost``) and the
precision, recall and F1 of the inferred set S, which is, for every method,
the candidates scoring at least 0.5, or, when there are none, the top-scoring
tie group (requirements are publicly known to be nonempty); precision is 0
for an empty S. Acting cost: on each of the world's recorded tasks (all its
episodes, the same for every method and checkpoint), from the task's initial
facts, the derivation planner's cheapest plan on the inferred node (S for the
scored recipes, the full pool for the others) is executed against the hidden
requirements without interaction; if the goal is not held at its end, the
brute-force plan (full pools) from the state reached completes it; plus one
submission. The world's acting cost ratio is the sum of these lengths over the
sum of the optimal lengths (the same planner on the hidden requirements), and
its plan success rate the share of tasks whose own plan reached the goal.
A world's value of a metric is the mean over its scored recipes with a hidden
requirement (the acting metrics are per task).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..agents.markov import PairwiseEvidence
from ..envs.derivations import StructRecipe, crafting_order, enumerate_derivations, plan_cost
from ..envs.generator import derive_seed
from ..evaluation.markov_study import (
    PRIOR_STRENGTH, Z_MDD, _ci, auroc, gathering_cost, infos_at, posterior_marginals,
)
from .arms_worker import ARMS_ITERATIONS
from .learners import factorized_marginals, sam_preconditions, set_scores
from .macq_adapter import (
    ARMS_SETTINGS, ARMS_STATUSES, ARMS_TIMEOUT, arms_job, arms_preconditions, observer_preconditions,
    run_arms,
)
from .msgi_adapter import msgi_scores
from .provenance import cache_key, code_sha256, msgi_sha256, versions
from .traces import craft_examples, items_observable, successful_steps

METHODS = ("posterior", "pair_weight", "msgi", "observer", "arms", "sam", "random")
OPEN_SOURCE = ("msgi", "observer", "arms")  # the authors' code or the macq library
ITEMS_ONLY = ("msgi", "observer", "arms", "sam")
CHECKPOINTS = (4, 8, 12, 16)
SELECTION_SEEDS, TEST_SEEDS = (0, 1), (2, 3, 4)
PRIMARY = {"set": "gen_obs", "source": "focused_sample", "k": 16}
MARGIN = 0.03
DIGITS = 9
EXPECTED_SELECTION_WORLDS, EXPECTED_TEST_WORLDS = 24, 36
# measurement worlds that are also development worlds (the same converted world keys)
DEV_SEEN = {"bench_msgi_mining": ("bench_msgi_mining:s0:w5", "bench_msgi_mining:s0:w10")}
METRICS = ("auroc", "gather", "precision", "recall", "f1", "act_ratio", "plan_success", "modelled")
PER_WORLD_METRICS = ("auroc", "f1", "act_ratio", "plan_success")
PAIRS = [("posterior", "best"), ("posterior", "msgi"), ("posterior", "observer"), ("posterior", "arms"),
         ("posterior", "sam"), ("posterior", "pair_weight"), ("observer", "sam"),
         ("posterior", "random"), ("best", "random")]


# ================================================================== scores

def scored_sigs(infos: dict, logs) -> list[str]:
    return sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})


def observable(rec: dict, logs, infos: dict) -> bool:
    return rec.get("observe_items") == "all" and all(items_observable(lg, infos) for lg in logs)


def job_id(rec: dict, k: int) -> str:
    return f"{rec['run_id']}|{rec['world_key']}|{k}"


def arms_jobs(worlds: list[dict], checkpoints=CHECKPOINTS) -> list[dict]:
    jobs = []
    for rec in worlds:
        for k in checkpoints:
            logs = rec["logs"][:k]
            infos = infos_at(rec, k)
            if len(logs) == k and observable(rec, logs, infos):
                jobs.append(arms_job(job_id(rec, k), logs, infos))
    return jobs


def method_scores(rec: dict, k: int, arms_result: dict | None = None, posterior_kw: dict | None = None,
                  max_size: int = 3) -> dict:
    """Scores of every applicable method for the scored recipes of one world at
    checkpoint ``k`` (see the module docstring)."""
    logs = rec["logs"][:k]
    infos = infos_at(rec, k)
    sigs = scored_sigs(infos, logs)
    edges = [(s, b) for s in sigs for b in infos[s].pool]
    rng = np.random.default_rng(derive_seed("x2_rank", rec.get("set"), rec["strategy"], rec["seed"],
                                            rec["world_key"], k) % 2 ** 32)
    scores, modelled, diag = {}, {}, {}
    obs = observable(rec, logs, infos)
    t0 = time.time()
    if obs:
        ex = craft_examples(logs, infos)
        post, fdiag = factorized_marginals(infos, ex, sigs, max_size=max_size)
        diag["posterior"] = {"form": "factorized", "exact_recipes": len(sigs), "sampled_recipes": 0,
                             "prior_recipes": 0, **fdiag}
    else:
        post, diag["posterior"] = posterior_marginals(infos, logs, rng, max_size=max_size,
                                                      **(posterior_kw or {}))
        diag["posterior"]["form"] = "enumeration"
    scores["posterior"], modelled["posterior"] = {e: post.get(e, 0.0) for e in edges}, set(sigs)
    diag["seconds_posterior"] = round(time.time() - t0, 3)
    pe = PairwiseEvidence(infos, PRIOR_STRENGTH, max_size).observe_logs(logs)
    scores["pair_weight"], modelled["pair_weight"] = {e: pe.weight(*e) for e in edges}, set(sigs)
    diag["pairwise"] = dict(pe.counts)
    scores["random"], modelled["random"] = {e: 0.0 for e in edges}, set()
    if obs:
        ms, mmod, diag["msgi"] = msgi_scores(infos, ex, sigs)
        scores["msgi"], modelled["msgi"] = {e: ms.get(e, 0.0) for e in edges}, mmod
        steps = [successful_steps(lg, infos) for lg in logs]
        for name, sets in (("sam", sam_preconditions(steps)),
                           ("observer", observer_preconditions(logs, infos)),
                           ("arms", arms_preconditions(arms_result, infos))):
            scores[name], modelled[name] = set_scores(infos, sigs, sets)
        diag["arms"] = {k2: (arms_result or {}).get(k2) for k2 in ("status", "seconds", "error", "iterations",
                                                                    "soft_clauses", "weights_moved")}
    return {"sigs": sigs, "infos": infos, "scores": scores, "modelled": modelled, "observable": obs,
            "diagnostics": diag}


# ================================================================= metrics

def inferred_set(scores: dict, pool) -> frozenset:
    """Candidates scoring at least 0.5, else the top-scoring tie group."""
    r = {b: round(float(scores[b]), DIGITS) for b in pool}
    s = frozenset(b for b, v in r.items() if v >= 0.5)
    return s or frozenset(b for b, v in r.items() if v == max(r.values()))


def set_metrics(S: frozenset, truth: set) -> dict:
    tp = len(S & truth)
    p = tp / len(S) if S else 0.0
    r = tp / len(truth) if truth else float("nan")
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _requirement(sig: str, info, truth: dict):
    return tuple(info.known) if info.known is not None else tuple(truth[sig])


def act(task_sigs, infos: dict, node: dict, truth: dict, initial: frozenset, goal: int,
        plan_cap: int = 4096) -> tuple[int, bool]:
    """Steps to reach ``goal`` acting on ``node`` (see the module docstring)
    and whether the node's own plan reached it."""
    sigs = [s for s in task_sigs if s in infos]
    recipes = [StructRecipe(infos[s].effect, infos[s].items,
                            tuple(infos[s].known) if infos[s].known is not None else tuple(sorted(node[s])))
               for s in sigs]
    req = {s: _requirement(s, infos[s], truth) for s in sigs}
    state, steps = set(initial), 0
    ds, _, _ = enumerate_derivations(recipes, frozenset(state), goal, plan_cap)
    if ds:
        pairs, bases = ds[0]
        for b in sorted(bases):
            state.add(b)
            steps += 1
        for _, ridx in crafting_order(pairs, recipes):
            steps += 1
            s = sigs[ridx]
            if all(x in state for x in req[s]) and all(i in state for i in infos[s].items):
                state.add(infos[s].effect)
    ok = goal in state
    if not ok:
        full = [StructRecipe(infos[s].effect, infos[s].items,
                             tuple(infos[s].known) if infos[s].known is not None else tuple(infos[s].pool))
                for s in sigs]
        ds2, _, _ = enumerate_derivations(full, frozenset(state), goal, plan_cap)
        steps += len(ds2[0][0]) + len(ds2[0][1])
    return steps + 1, ok


def optimal_length(task_sigs, infos: dict, truth: dict, initial: frozenset, goal: int,
                   plan_cap: int = 4096) -> int:
    sigs = [s for s in task_sigs if s in infos]
    recipes = [StructRecipe(infos[s].effect, infos[s].items, _requirement(s, infos[s], truth)) for s in sigs]
    ds, _, _ = enumerate_derivations(recipes, frozenset(initial), goal, plan_cap)
    return plan_cost(ds[0])


def acting_metrics(rec: dict, sets: dict) -> dict:
    """Acting cost ratio and plan success of a node over all recorded tasks of a world."""
    infos, truth = rec["infos"], rec["truth"]["requirements"]
    node = {s: sets.get(s, frozenset(i.pool)) for s, i in infos.items() if i.known is None}
    cost = opt = 0
    ok = []
    for lg, reg in zip(rec["logs"], rec["registered"]):
        c, success = act(reg, infos, node, truth, lg.initial, lg.goal)
        cost += c
        opt += optimal_length(reg, infos, truth, lg.initial, lg.goal)
        ok.append(success)
    return {"act_ratio": cost / opt if opt else float("nan"), "plan_success": float(np.mean(ok)) if ok else
            float("nan")}


def world_metrics(rec: dict, k: int, arms_result: dict | None = None,
                  posterior_kw: dict | None = None) -> dict | None:
    """Per-world metrics of every applicable method at checkpoint ``k``."""
    if len(rec["logs"]) < k:
        return None
    res = method_scores(rec, k, arms_result, posterior_kw)
    truth = {s: set(v) for s, v in rec["truth"]["requirements"].items()}
    sigs = [s for s in res["sigs"] if s in truth]
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
        per[m] = {**means, **acting_metrics(rec, sets)}
    return {"world": rec.get("world_label", rec["world_key"]), "set": rec.get("set"), "seed": rec["seed"],
            "source": rec["strategy"], "k": k, "recipes": len(sigs), "observable": res["observable"],
            "metrics": per, "diagnostics": res["diagnostics"], "complete": rec.get("complete", True)}


# ============================================================== aggregation

def split_of(rec_or_row: dict, set_kind: str | None) -> str:
    if set_kind == "benchmark":
        return "all"
    seed = rec_or_row["seed"]
    return "selection" if seed in SELECTION_SEEDS else "test" if seed in TEST_SEEDS else "other"


def select_best(rows: list[dict], candidates=OPEN_SOURCE) -> dict:
    """The open-source baseline with the largest mean per-world AUROC (ties
    within 1e-6: the order of ``candidates``)."""
    means = {m: float(np.nanmean([w["metrics"][m]["auroc"] for w in rows if m in w["metrics"]] or [np.nan]))
             for m in candidates}
    finite = {m: v for m, v in means.items() if not np.isnan(v)}
    if not finite:
        return {"means": means, "selected": None}
    best = max(finite.values())
    pick = next(m for m in candidates if m in finite and finite[m] >= best - 1e-6)
    return {"means": means, "selected": pick}


def decision(diffs, margin: float = MARGIN) -> dict:
    """Pre-registered decision on per-world differences AUROC posterior minus
    AUROC of the selected baseline: "posterior better" if the 95% lower end is
    above 0, "baseline better" if the 95% upper end is below 0, otherwise
    "equivalent" if the 90% interval lies inside (-margin, margin), otherwise
    "inconclusive"; a significant difference takes precedence."""
    c95, c90 = _ci(diffs), _ci(diffs, 0.90)
    if not c95.get("ci"):
        return {"decision": None, "ci90": None, "margin": margin}
    flags = {"posterior_better": c95["ci"][0] > 0, "baseline_better": c95["ci"][1] < 0,
             "equivalent": -margin < c90["ci"][0] and c90["ci"][1] < margin}
    d = ("posterior better" if flags["posterior_better"] else "baseline better" if flags["baseline_better"]
         else "equivalent" if flags["equivalent"] else "inconclusive")
    return {"decision": d, "ci90": c90["ci"], "margin": margin, **flags}


def _value(w: dict, m: str, metric: str, best: str | None) -> float:
    name = best if m == "best" else m
    if name is None or name not in w["metrics"]:
        return float("nan")
    return w["metrics"][name][metric]


def _diffs(rows: list, a: str, b: str | None, metric: str) -> list:
    d = [_value(w, a, metric, b) - _value(w, "best", metric, b) for w in rows] if b else []
    return [v for v in d if not math.isnan(v)]


# ============================================================ cache, resume

class CodeMismatch(RuntimeError):
    """A worker computed a row or an ARMS result with other code than this process."""


class Cache:
    """Rows (one per run, world and checkpoint) and ARMS results under
    ``<root>/<key>/``; the key covers the code, the vendored MSGI files, the
    options and the library versions (``provenance.cache_key``). Entries are
    always written; they are read only with ``resume``."""

    def __init__(self, root: str | Path | None, key: str, resume: bool):
        self.dir = None if root is None else Path(root) / key
        self.resume = resume
        if self.dir is not None:
            for kind in ("rows", "arms"):
                (self.dir / kind).mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, ident: str) -> Path:
        return self.dir / kind / (hashlib.sha1(ident.encode()).hexdigest() + ".json")

    def get(self, kind: str, ident: str):
        if self.dir is None or not self.resume:
            return None
        p = self._path(kind, ident)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return d["value"] if d.get("ident") == ident else None

    def put(self, kind: str, ident: str, value) -> None:
        if self.dir is None:
            return
        p = self._path(kind, ident)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ident": ident, "value": value}, default=str), encoding="utf-8")
        os.replace(tmp, p)


def row_id(rec: dict, k: int) -> str:
    return f"{rec['run_id']}|{rec['world_key']}|{k}"


def _world_rows(args) -> list:
    rec, checkpoints, arms_by_k, posterior_kw = args
    code = code_sha256()
    out = []
    for k in checkpoints:
        r = world_metrics(rec, k, arms_by_k.get(k), posterior_kw)
        if r is not None:
            r["code"] = code
        out.append((k, r))
    return out


def _check_code(stamp: str | None, what: str) -> None:
    if stamp is not None and stamp != code_sha256():
        raise CodeMismatch(f"{what} was computed with code {stamp[:12]}, this process runs "
                           f"{code_sha256()[:12]}")


def _rows_parallel(tasks: list, workers: int, done) -> list:
    """``_world_rows`` of every task in worker processes (one task per process),
    ``done(i, rows)`` called as each finishes; None for a task whose worker failed."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    out: list = [None] * len(tasks)
    try:
        with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
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


def _arms_results(worlds, checkpoints, cache: Cache, arms_workers: int, arms_timeout: float | None) -> tuple:
    jobs = arms_jobs(worlds, checkpoints)
    results = {j["id"]: v for j in jobs if (v := cache.get("arms", j["id"])) is not None}
    todo = [j for j in jobs if j["id"] not in results]

    def keep(res: dict) -> None:
        _check_code(res.get("code"), f"ARMS job {res['id']}")
        cache.put("arms", res["id"], res)

    if todo:
        kw = {"workers": arms_workers, "on_result": keep}
        if arms_timeout:
            kw["timeout"] = arms_timeout
        results.update(run_arms(todo, **kw))
    for res in results.values():
        _check_code(res.get("code"), f"ARMS job {res['id']}")
    return results, len(jobs), len(jobs) - len(todo)


# ================================================================== analysis

def rank_benchmark(worlds: list[dict], *, checkpoints=CHECKPOINTS, posterior_kw: dict | None = None,
                   arms_workers: int = 1, arms_timeout: float | None = None, workers: int = 1,
                   expected_test_worlds: int = EXPECTED_TEST_WORLDS,
                   expected_selection_worlds: int = EXPECTED_SELECTION_WORLDS, set_kinds: dict | None = None,
                   cache_dir: str | Path | None = None, resume: bool = False, progress=None) -> dict:
    """Every world, source and checkpoint; the selection of the best open-source
    baseline per source on the selection worlds of the primary set; the
    pre-registered primary contrast on its test worlds; the secondary tests.
    ``workers`` > 1 computes worlds in parallel processes (same results: every
    random stream is seeded by its world, source and checkpoint; a world whose
    worker fails is recomputed in this process). Rows and ARMS results are
    cached under ``cache_dir`` and reused with ``resume`` (same key only)."""
    t0 = time.time()
    options = {"checkpoints": list(checkpoints), "posterior_kw": posterior_kw or {},
               "arms_settings": ARMS_SETTINGS, "arms_iterations": ARMS_ITERATIONS,
               "arms_timeout": arms_timeout or ARMS_TIMEOUT}
    key = cache_key(options)
    cache = Cache(cache_dir, key, resume)
    kinds = dict(set_kinds or {})
    for rec in worlds:
        kinds.setdefault(rec.get("set"), rec.get("set_kind"))
    arms, n_jobs, arms_reused = _arms_results(worlds, checkpoints, cache, arms_workers, arms_timeout)
    arms_seconds = round(time.time() - t0, 1)
    per = defaultdict(dict)  # (set, split, source, k) -> world -> row
    results: list = [None] * len(worlds)
    reused = 0
    for i, rec in enumerate(worlds):
        cached = [(k, cache.get("rows", row_id(rec, k))) for k in checkpoints]
        if all(r is not None for _, r in cached):
            results[i] = [(k, r.get("row")) for k, r in cached]
            for k, r in results[i]:
                if r is not None:
                    _check_code(r.get("code"), f"cached row {row_id(rec, k)}")
                    r["complete"] = rec.get("complete", True)  # the evidence's status now, not when cached
            reused += 1

    def done(i: int, rows: list) -> None:
        for k, r in rows:
            if r is not None:
                _check_code(r.get("code"), f"row {row_id(worlds[i], k)}")
            cache.put("rows", row_id(worlds[i], k), {"row": r})
            if progress:
                progress(worlds[i], k)

    todo = [i for i, r in enumerate(results) if r is None]
    tasks = {i: (worlds[i], tuple(checkpoints), {k: arms.get(job_id(worlds[i], k)) for k in checkpoints},
                 posterior_kw) for i in todo}
    fallbacks = 0
    if workers > 1 and len(todo) > 1:
        par = _rows_parallel([tasks[i] for i in todo], workers, lambda j, rows: done(todo[j], rows))
        for j, i in enumerate(todo):
            results[i] = par[j]
    for i in todo:
        if results[i] is None:  # sequential, or its worker failed: the same deterministic computation
            fallbacks += workers > 1 and len(todo) > 1
            results[i] = _world_rows(tasks[i])
            done(i, results[i])
    for rec, rows in zip(worlds, results):
        for k, r in rows:
            if r is not None:
                key4 = (rec.get("set"), split_of(rec, kinds.get(rec.get("set"))), rec["strategy"], k)
                per[key4][r["world"]] = r
    summary = aggregate(per, checkpoints, expected_test_worlds, expected_selection_worlds)
    totals = {s: sum((r.get("status") == s) for r in arms.values()) for s in ARMS_STATUSES}
    summary.update(arms_jobs=n_jobs, arms_reused=arms_reused, arms_seconds=arms_seconds,
                   arms_status_totals=totals, rows_reused=reused, parallel_fallbacks=int(fallbacks),
                   provenance=provenance_record(key, options))
    return summary


def arms_seed_check(worlds: list[dict], summary: dict, hash_seeds, *, arms_workers: int = 1) -> dict:
    """Exploratory: ARMS on the selection worlds of the primary set at the
    primary checkpoint under other string hash seeds (which decide ARMS's ties),
    and whether the selected baseline or "ARMS below Observer" would change."""
    k, out = PRIMARY["k"], {"checkpoint": PRIMARY["k"], "seeds": {}}
    recs = [r for r in worlds if r.get("set") == PRIMARY["set"] and r["seed"] in SELECTION_SEEDS
            and len(r["logs"]) >= k and observable(r, r["logs"][:k], infos_at(r, k))]
    for hs in hash_seeds:
        res = run_arms([arms_job(f"{job_id(r, k)}|hash{hs}", r["logs"][:k], infos_at(r, k)) for r in recs],
                       workers=arms_workers, hash_seed=str(hs))
        by_src = defaultdict(list)
        for r in recs:
            infos, truth = infos_at(r, k), {s: set(v) for s, v in r["truth"]["requirements"].items()}
            sigs = [s for s in scored_sigs(infos, r["logs"][:k]) if s in truth]
            sc, _ = set_scores(infos, sigs, arms_preconditions(res.get(f"{job_id(r, k)}|hash{hs}"), infos))
            if sigs:
                by_src[r["strategy"]].append(float(np.mean([auroc({b: sc[(s, b)] for b in infos[s].pool},
                                                                   truth[s]) for s in sigs])))
        cell = {}
        for src, vals in sorted(by_src.items()):
            base = (summary["best_baseline"].get(src) or {}).get("means") or {}
            mean = float(np.mean(vals))
            others = {m: {"auroc": base[m]} for m in ("msgi", "observer") if base.get(m) is not None}
            pick = select_best([{"metrics": {"arms": {"auroc": mean}, **others}}])["selected"]
            cell[src] = {"arms_mean": mean, "worlds": len(vals),
                         "selected_at_seed_0": summary["best_baseline"].get(src, {}).get("selected"),
                         "selected": pick,
                         "arms_below_observer": mean < base["observer"] if "observer" in others else None}
        status = {s: sum(v.get("status") == s for v in res.values()) for s in ARMS_STATUSES}
        out["seeds"][str(hs)] = {"by_source": cell, "status": status}
    return out


def provenance_record(key: str, options: dict) -> dict:
    from ..training.run import source_info
    src = source_info(Path(__file__).resolve().parents[3])
    src.pop("_patch", None)
    return {**src, "code": code_sha256(), "msgi_files": msgi_sha256(), "versions": versions(),
            "cache_key": key, "options": options}


def _cell(per: dict, st: str, split: str, src: str, k: int) -> list:
    return list(per.get((st, split, src, k), {}).values())


def secondary_tests(per: dict, best: dict, checkpoints) -> dict:
    """The pre-registered secondary tests (docs/studies/X2_inference_baselines.md),
    each decided from the per-world rows."""
    k = PRIMARY["k"]
    out = {}

    def interval_test(name, rows, a, b, metric, side, prediction):
        d = _diffs(rows, a, b, metric) if b else []
        c = _ci(d)
        holds = None
        if c.get("ci"):
            holds = c["ci"][0] > 0 if side == "above" else c["ci"][1] < 0
        out[name] = {"prediction": prediction,
                     "test": f"95% interval of {a} - {b} entirely {side} 0 ({metric})",
                     "baseline": b, "n": len(d), **c, "holds": holds}

    def decision_test(name, rows, a, b, predicted, prediction):
        d = _diffs(rows, a, b, "auroc") if b else []
        dec = decision(d)
        out[name] = {"prediction": prediction,
                     "test": f"decision on {a} - {b} (AUROC), predicted {predicted}",
                     "baseline": b, "n": len(d), **_ci(d), "ci90": dec["ci90"], "decision": dec["decision"],
                     "holds": None if dec["decision"] is None else dec["decision"] == predicted}

    b_fs, b_ro = best.get("focused_sample", {}).get("selected"), best.get("random_omit", {}).get("selected")
    decision_test("S1", _cell(per, "gen_obs", "test", "random_omit", k), "posterior", b_ro,
                  "posterior better", "random_omit evidence: posterior better than its selected baseline")
    exact, ms = [], ("posterior", "pair_weight", "msgi", "observer", "sam", "random")
    for (st, split, src, kk), ws in per.items():
        if st in ("gen_obs", "gen_goal") and src == "maximal":
            exact += [round(w["metrics"][m]["auroc"], DIGITS) == 0.5 for w in ws.values() for m in ms
                      if m in w["metrics"]]
    arms_rows = _cell(per, "gen_obs", "test", "maximal", k)
    arms_c = _ci([w["metrics"]["arms"]["auroc"] - 0.5 for w in arms_rows if "arms" in w["metrics"]])
    out["S2"] = {"prediction": "maximal evidence on generated worlds: exactly 0.5 AUROC for every method "
                               "except arms; arms 0.5 in expectation",
                 "test": "every per-world value exactly 0.5; 95% interval of arms - 0.5 contains 0",
                 "n": len(exact), "exact_share": float(np.mean(exact)) if exact else None, "arms": arms_c,
                 "holds": ((all(exact) and arms_c["ci"][0] <= 0 <= arms_c["ci"][1])
                           if exact and arms_c.get("ci") else None)}
    gaps = [abs(w["metrics"]["observer"][x] - w["metrics"]["sam"][x]) for ws in per.values()
            for w in ws.values() if "sam" in w["metrics"] for x in ("auroc", "f1", "act_ratio")]
    out["S3"] = {"prediction": "observer equals sam in every cell",
                 "test": "largest per-world difference is 0",
                 "n": len(gaps), "largest": max(gaps) if gaps else None,
                 "holds": (max(gaps) == 0) if gaps else None}
    primary_rows = _cell(per, PRIMARY["set"], "test", PRIMARY["source"], k)
    interval_test("S4", primary_rows, "posterior", b_fs, "f1", "above",
                  "F1: posterior above the best baseline")
    interval_test("S5", primary_rows, "posterior", b_fs, "act_ratio", "below",
                  "acting cost ratio: posterior below (cheaper than) the best baseline")
    decision_test("S6", primary_rows, "posterior", "pair_weight", "equivalent",
                  "posterior - pair_weight equivalent within the margin")
    decision_test("S7", _cell(per, "gen_goal", "test", PRIMARY["source"], k), "posterior", "pair_weight",
                  "equivalent", "goal-only: posterior - pair_weight equivalent within the margin")
    for st, seen in DEV_SEEN.items():  # also without the measurement worlds seen in development
        rows = [w for w in _cell(per, st, "all", PRIMARY["source"], k) if w["world"] not in seen]
        if rows:
            interval_test(f"S8 {st} without development worlds", rows, "posterior", b_fs, "auroc", "above",
                          f"{st} without {', '.join(seen)}: posterior above the best baseline")
    for st in sorted({key[0] for key in per if key[1] == "all"}):
        interval_test(f"S8 {st}", _cell(per, st, "all", PRIMARY["source"], k), "posterior", b_fs, "auroc",
                      "above", f"{st}: posterior above the best baseline (descriptive)")
    return out


def _world_entry(r: dict) -> dict:
    return {"seed": r["seed"], "recipes": r["recipes"], "complete": r["complete"],
            "metrics": {m: {x: v[x] for x in PER_WORLD_METRICS} for m, v in r["metrics"].items()}}


def aggregate(per: dict, checkpoints, expected_test_worlds: int = EXPECTED_TEST_WORLDS,
              expected_selection_worlds: int = EXPECTED_SELECTION_WORLDS) -> dict:
    k = PRIMARY["k"]
    best = {}
    for src in sorted({key[2] for key in per}):
        rows = _cell(per, PRIMARY["set"], "selection", src, k)
        ok = len(rows) >= expected_selection_worlds and all(w["complete"] for w in rows)
        sel = select_best(rows) if ok else {"means": None, "selected": None}
        best[src] = {**sel, "n_worlds": len(rows), "complete": ok, "checkpoint": k}
    table, contrasts = {}, {}
    for key4, ws in sorted(per.items()):
        st, split, src, kk = key4
        rows = list(ws.values())
        b = best.get(src, {}).get("selected")
        label = f"{st}:{split}:{src}@{kk}"
        methods = [m for m in METHODS if any(m in w["metrics"] for w in rows)]
        table[label] = {m: {x: _ci([w["metrics"][m][x] for w in rows if m in w["metrics"]]) for x in METRICS}
                        for m in methods}
        table[label].update(n_worlds=len(rows), n_recipes=int(sum(w["recipes"] for w in rows)), best=b)
        for a, c in PAIRS:
            for x in ("auroc", "f1", "act_ratio"):
                d = [_value(w, a, x, b) - _value(w, c, x, b) for w in rows]
                d = [v for v in d if not math.isnan(v)]
                if d:
                    contrasts[f"{label}: {a} - {c} ({x})"] = _ci(d)
    test = {w["world"]: w for w in _cell(per, PRIMARY["set"], "test", PRIMARY["source"], k)}
    selected = best.get(PRIMARY["source"], {}).get("selected")
    diffs = [w["metrics"]["posterior"]["auroc"] - w["metrics"][selected]["auroc"]
             for w in test.values() if selected and selected in w["metrics"]]
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
    dec = decision(diffs)
    complete = (selected is not None and bool(test) and all(w["complete"] for w in test.values())
                and len(test) >= expected_test_worlds and len(diffs) == len(test))
    diag = {}
    for key4, ws in sorted(per.items()):
        dg = [w["diagnostics"] for w in ws.values()]
        diag[f"{key4[0]}:{key4[1]}:{key4[2]}@{key4[3]}"] = {
            "posterior_exact_recipes": int(sum(d["posterior"]["exact_recipes"] for d in dg)),
            "posterior_sampled_recipes": int(sum(d["posterior"]["sampled_recipes"] for d in dg)),
            "posterior_prior_recipes": int(sum(d["posterior"]["prior_recipes"] for d in dg)),
            "posterior_seconds": round(sum(d["seconds_posterior"] for d in dg), 1),
            "msgi": {s: int(sum(d["msgi"][s] for d in dg if "msgi" in d))
                     for s in ("tree", "always", "never", "failed", "multi_term", "negated_literals")},
            "arms_status": {s: int(sum((d.get("arms") or {}).get("status") == s for d in dg))
                            for s in ARMS_STATUSES},
            "arms_weights_moved": int(sum((d.get("arms") or {}).get("weights_moved") or 0 for d in dg)),
            "arms_soft_clauses": int(sum((d.get("arms") or {}).get("soft_clauses") or 0 for d in dg)),
            "factorized_contradictions": int(sum(d["posterior"].get("contradiction", 0) for d in dg)),
        }
    per_world = {f"{a}:{b}:{c}@{d}": {w: _world_entry(r) for w, r in ws.items()}
                 for (a, b, c, d), ws in sorted(per.items())}
    return {"definition": {
                "methods": list(METHODS), "open_source": list(OPEN_SOURCE), "items_only": list(ITEMS_ONLY),
                "checkpoints": list(checkpoints), "selection_seeds": list(SELECTION_SEEDS),
                "test_seeds": list(TEST_SEEDS), "unit": "world (mean over its scored recipes)",
                "expected_selection_worlds": expected_selection_worlds,
                "expected_test_worlds": expected_test_worlds,
                "bootstrap": {"resamples": 10_000, "seed": 20261008, "level": 0.95},
                "primary": f"AUROC posterior - AUROC best open-source baseline (selected on the selection "
                           f"worlds at checkpoint {k}), set {PRIMARY['set']}, {PRIMARY['source']} "
                           f"evidence, checkpoint {k}, test worlds",
                "decision": f"posterior better if the 95% lower end is above 0; baseline better if the 95% "
                            f"upper end is below 0; otherwise equivalent if the 90% interval lies inside "
                            f"(-{MARGIN:g}, {MARGIN:g}); otherwise inconclusive"},
            "best_baseline": best, "table": table, "contrasts": contrasts,
            "primary": {**_ci(diffs), "ci90": dec["ci90"], "sd": sd, "selected": selected,
                        "mdd": Z_MDD * sd / math.sqrt(len(diffs)) if sd is not None else None},
            "primary_decision": dec, "primary_verdict": dec["decision"] if complete else None,
            "complete": complete, "secondary": secondary_tests(per, best, checkpoints),
            "diagnostics": diag, "per_world": per_world}


# ================================================================ reporting

def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _fci(c: dict, nd=3) -> str:
    if not c or c.get("estimate") is None:
        return "-"
    return f"{_f(c['estimate'], nd)} [{_f(c['ci'][0], nd)}, {_f(c['ci'][1], nd)}]"


def markdown(s: dict) -> str:
    lines = ["Best open-source baseline per evidence source (selection worlds, checkpoint "
             f"{PRIMARY['k']}): " + ", ".join(f"{src}: {v['selected']} (worlds {v['n_worlds']})"
                                              for src, v in s["best_baseline"].items()) + ".", "",
             "ARMS jobs by status (string hash seed 0): " + ", ".join(
                 f"{k} {v}" for k, v in (s.get("arms_status_totals") or {}).items()) + ".", ""]
    for metric, nd, title in (("auroc", 3, "AUROC"), ("f1", 3, "F1 of the inferred set"),
                              ("act_ratio", 2, "Acting cost ratio")):
        lines += [f"{title} (mean per world [95% CI]):", "",
                  "| Set:split:evidence@episodes | Worlds | " + " | ".join(METHODS) + " |",
                  "|---|---|" + "---|" * len(METHODS)]
        for key, row in s["table"].items():
            cells = " | ".join(_fci(row[m][metric], nd) if m in row else "n/a" for m in METHODS)
            lines.append(f"| {key} | {row['n_worlds']} | {cells} |")
        lines.append("")
    p = s["primary"]
    c90 = p.get("ci90")
    lines += [f"Primary ({s['definition']['primary']}; selected: {p.get('selected')}): {_fci(p)}, "
              + "90% interval " + (f"[{_f(c90[0])}, {_f(c90[1])}]" if c90 else "-")
              + f", margin {s['primary_decision']['margin']:g}, minimum detectable difference "
              f"{_f(p.get('mdd'))}; decision: {s['primary_verdict']}.", "",
              "Secondary tests:", "", "| Test | Prediction | Worlds | Estimate [95% CI] | Decision | Holds |",
              "|---|---|---|---|---|---|"]
    for name, t in s["secondary"].items():
        est = _fci(t) if "ci" in t else (f"exact share {_f(t.get('exact_share'))}, arms {_fci(t.get('arms'))}"
                                         if name == "S2" else f"largest difference {_f(t.get('largest'))}")
        lines.append(f"| {name} | {t['prediction']} | {t['n']} | {est} | {t.get('decision', '-')} | "
                     f"{t['holds']} |")
    return "\n".join(lines) + "\n"
