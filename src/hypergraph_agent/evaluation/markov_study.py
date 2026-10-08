"""Study M (docs/studies/M_markov_ranking.md): evidence collection, the
offline ranking benchmark (M1), twin constructions and the factorial
analysis (M2).

Evaluator side: rankings are scored against the hidden requirements, which
the collection runner stores in each evidence file under ``truth``. Agents
never import this module.

Rankings of a recipe's pool candidates (M1), all from the same logged
episodes up to a checkpoint and the recipes registered by then (what an agent
would hold at a replan in the latest logged episode):

``posterior``    (a) marginal probability under the joint posterior over complete
                 hypergraphs (uniform prior over the hypothesis class, deterministic
                 replay of every log). Exact for each connected group of recipes:
                 depth-first enumeration pruned by the logs, with the counts of the
                 remaining recipes memoized on the assigned recipes they still share a
                 log with; beyond a work budget, Gibbs sampling among consistent nodes
                 (each move redraws one recipe's hypothesis uniformly among those
                 consistent with the others: a Metropolis-Hastings move with acceptance
                 one) from ``starts`` focused walks, marginals Rao-Blackwellized
``fact_ppr``     (b) PageRank personalized at the recipe's effect on the empirical fact
                 chain: the clique expansion of the fact set of every episode that
                 reached the goal (initial facts, bases gathered and effects crafted up
                 to the goal, the goal), pair weight 1 / (|set| - 1) per episode
``star_ppr``     (c) the agents' score (``agents.markov.star_scores``): PageRank on the
                 star graph with the online pairwise weights, personalized at the recipe
                 node, its effect, or the goal of the latest logged episode
``pair_weight``  the pairwise weight itself (a diagnostic; equal to (c) at restart 1.0)
``andor``        (d) AND-OR factor score: an independent belief per edge with a
                 fair-coin prior and, per recipe, a factor allowing 1 to ``max_size``
                 required candidates (together exactly the uniform class prior); every
                 failure kept as a factor over its monotone formula (OR across the
                 open suspects of an attempt, AND across alternative producers that all
                 failed; a literal that occurs more than once is summed over exactly),
                 successes as eliminations; beliefs by damped loopy sum-product.
                 Successes with several attempted producers of an input give no
                 elimination and are dropped (a disjunction of negations). In -log space
                 AND adds the preconditions' costs as the additive heuristic h_add does,
                 and OR is a soft minimum (h_add's minimum at zero temperature); it
                 shares h_add's assumption that preconditions are independent
``random``       every candidate tied (the expectation of a random order)

PageRank settings have a restart probability in ``RESTARTS``; restart 1.0
means no propagation (``agents.markov.candidate_ppr``).

Run selection (evidence runs and M2 runs alike): one run per arm and seed, the
earliest. A run that did not complete is replaced by the earliest later
completed run with the same config hash, commit and loaded code (runs are
deterministic); replaced and ignored runs are listed. A verdict needs every
run it uses to be complete; otherwise it is None.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np

from ..agents.markov import (
    PPR_SEEDS, EpisodeReplay, PairwiseEvidence, base_sets, candidate_ppr, literals, observations,
    personalized_pagerank, simplify, star_scores,
)
from ..agents.walker import EPISODIC, EpisodeLog, RecipeInfo, Walker, WorldState, local_walk
from ..config import eval_stream_config
from ..envs.derivations import StructRecipe, enumerate_derivations, plan_cost
from ..envs.generator import TaskStream, derive_seed
from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from ..envs.vocabulary import TYPE_NAMES
from ..topology.inference import enumerate_hypotheses
from ..training.budget import BudgetMeter, SessionLedger
from ..training.run import RunContext
from .stats import cluster_bootstrap_mean
from .walkers import NON_ADAPTIVE, eval_order, make_walker, play, stream_budget, variant_config

METHODS = ("posterior", "andor", "star_ppr", "pair_weight", "fact_ppr", "random")
PPR_METHODS = {"star_ppr": PPR_SEEDS, "fact_ppr": ("effect",)}
CHECKPOINTS = (4, 8, 12, 16)
RESTARTS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0)
SELECTION_SEEDS, TEST_SEEDS = (0, 1), (2, 3, 4)
PRIOR_STRENGTH = 2.0
N_BOOT, BOOT_SEED = 10_000, 20261008
Z_MDD = 1.959964 + 0.841621  # two-sided 5% test, 80% power
MARGIN = 0.03  # M1 equivalence margin on AUROC
DIGITS = 9  # scores are compared after rounding, so float noise never breaks a tie
DEFAULT_VARIANT = {"name": "default", "tasks": {}}
RERUN_MATCH = ("config_hash", "commit", "code")


# ======================================================== run selection

def _manifest_info(d: Path) -> dict:
    man = json.loads((d / "manifest.json").read_text())
    src = man.get("source") or {}
    return {"run_id": man.get("run_id", d.name), "dir": d, "seed": man.get("seed"), "arm_id": man.get("arm"),
            "start": man.get("start_time", ""), "config_hash": man.get("config_hash"),
            "commit": src.get("commit"), "code": src.get("loaded_code_sha256"),
            "status_complete": man.get("status") == "completed" and man.get("stop_reason") == "tasks_done",
            "wallclock_s": man.get("wallclock_s")}


def select_runs(runs: list[dict]) -> tuple[list[dict], dict]:
    """One run per (strategy, seed): the earliest, unless it did not complete
    and a later completed run matches it on ``RERUN_MATCH``."""
    groups = defaultdict(list)
    for r in sorted(runs, key=lambda x: (x["start"], x["run_id"])):
        groups[(r["strategy"], r["seed"])].append(r)
    chosen, reruns, ignored = [], [], []
    for g in groups.values():
        pick = g[0]
        if not pick["complete"]:
            same = [x for x in g[1:] if x["complete"] and all(x[k] == pick[k] for k in RERUN_MATCH)]
            if same:
                reruns.append({"incomplete": pick["run_id"], "rerun": same[0]["run_id"]})
                pick = same[0]
        chosen.append(pick)
        ignored += [x["run_id"] for x in g if x is not pick and x is not g[0]]
    return chosen, {"reruns_used": reruns, "ignored": ignored}


# ====================================================== evidence collection

class LoggedMaximal(Walker):
    """``maximal`` (the full-pool node, no inference) with its episode logs kept
    as evidence: the episode loop runs as an episodic walker, the node choice
    is that of ``maximal``, so the actions are the same."""

    def __init__(self, rng, **kw):
        super().__init__("focused_sample", rng, **kw)
        self.label = "walker:maximal"

    def choose_node(self, w, log, held=frozenset()):
        return {sig: frozenset(w.infos[sig].pool) for sig in w.hidden()}, {"evals": 0}


def evidence_agent(strategy: str, wc: dict, seed_key, epsilon: float = 0.0):
    if strategy == "maximal":
        rng = np.random.default_rng(derive_seed("walker", *seed_key) % (2 ** 32))
        return LoggedMaximal(rng, epsilon=epsilon, plan_cap=wc["plan_cap"])
    agent = make_walker(strategy, wc, seed_key, epsilon)
    if agent.strategy not in EPISODIC:
        raise ValueError(f"strategy {strategy!r} keeps no episode logs")
    return agent


def collect_plan(cfg: dict) -> dict:
    """Dry run of a collection config: per-seed sums of task budgets (an upper
    bound on each run's interactions; generator only, no interaction)."""
    seeds, arms = cfg["run"]["seeds"], [(a["id"], a["strategy"]) for a in cfg["arms"]]
    sums = {f"s{s}": stream_budget(cfg, s, DEFAULT_VARIANT) for s in seeds}
    capped = sum(min(cfg["walker"]["per_run_cap"], v) for v in sums.values())
    n_adaptive = sum(s not in NON_ADAPTIVE for _, s in arms)
    es = eval_stream_config(cfg)
    base_seeds = {s: variant_config(cfg, s, DEFAULT_VARIANT)["eval"]["base_seed"] for s in seeds}
    return {"allocation": cfg["run"]["allocation"], "arms": arms, "seeds": seeds,
            "eval": {"namespace": es.namespace, "n_worlds": es.n_worlds,
                     "episodes_per_world": cfg["eval"]["episodes_per_world"],
                     "n_tasks": cfg["eval"]["n_tasks"],
                     "observe_items": es.task.observe_items, "failure_prob": es.task.failure_prob,
                     "base_seed_by_seed": base_seeds},
            "per_run_cap": cfg["walker"]["per_run_cap"], "stream_budget_sums": sums,
            "stream_budget_total": sum(sums.values()),
            "per_run_cap_covers_every_stream": cfg["walker"]["per_run_cap"] >= max(sums.values()),
            "worst_case_adaptive": n_adaptive * capped,
            "worst_case_reporting": (len(arms) - n_adaptive) * capped,
            "ledger": cfg["run"]["ledger"]}


def log_record(lg) -> dict:
    return {"log_id": lg.log_id, "goal": lg.goal, "initial": sorted(lg.initial),
            "steps": [list(s) for s in lg.steps], "goal_obs": list(lg.goal_obs),
            "effect_obs": list(lg.effect_obs), "base_obs": list(getattr(lg, "base_obs", []))}


def log_from_record(d: dict) -> EpisodeLog:
    lg = EpisodeLog(d["log_id"], d["goal"], frozenset(d["initial"]))
    lg.steps = [tuple(s) for s in d["steps"]]
    lg.goal_obs, lg.effect_obs = list(d["goal_obs"]), list(d["effect_obs"])
    if d.get("base_obs") and hasattr(lg, "base_obs"):
        lg.base_obs = list(d["base_obs"])
    return lg


def info_record(i: RecipeInfo) -> dict:
    return {"signature": i.signature, "effect": i.effect, "items": list(i.items), "pool": list(i.pool),
            "known": None if i.known is None else list(i.known)}


def info_from_record(d: dict) -> RecipeInfo:
    return RecipeInfo(d["signature"], d["effect"], tuple(d["items"]), tuple(d["pool"]),
                      None if d["known"] is None else tuple(d["known"]))


def world_records(agent, stream, order, rows, meta: dict) -> list[dict]:
    """One record per world the agent played: public recipe infos, the
    signatures each episode's task registered, episode logs in play order,
    outcome rows and (evaluator only) the hidden requirements."""
    worlds, registered = {}, defaultdict(list)
    for i in order[: len(rows)]:
        t = stream.task(i)
        worlds.setdefault(t.world.world_key, (t.world, t.profile))
        registered[t.world.world_key].append(sorted(r.signature(t.profile) for r in t.rules))
    out = []
    for wk, ws in sorted(agent.worlds.items()):
        world, profile = worlds[wk]
        truth = {r.signature(profile): sorted(r.true_base) for r in world.recipes}
        out.append({**meta, "world_key": wk, "infos": [info_record(i) for i in ws.infos.values()],
                    "registered": registered[wk], "logs": [log_record(lg) for lg in ws.evidence.logs],
                    "rows": [r for r in rows if r["world_key"] == wk],
                    "truth": {"evaluator_only": True,
                              "requirements": {s: truth[s] for s in ws.infos if s in truth}}})
    return out


def collect(cfg: dict, ledger: SessionLedger, runs_dir: str, stamp: str | None = None) -> list[dict]:
    """M1 collection: each evidence arm plays the configured stream of every
    seed (the seed's own worlds with ``eval_world_offset_by_seed``), charged to
    the ledger; every world's evidence is written to ``<run>/evidence/``."""
    wc, alloc = cfg["walker"], cfg["run"]["allocation"]
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    results = []
    for seed in cfg["run"]["seeds"]:
        vcfg = variant_config(cfg, seed, DEFAULT_VARIANT)
        stream_cfg = eval_stream_config(vcfg)
        if stream_cfg.task.failure_prob:
            raise ValueError("study M collects evidence under deterministic dynamics only")
        order = eval_order(stream_cfg, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"])
        for arm in cfg["arms"]:
            strategy = arm["strategy"]
            kind = "reporting" if strategy in NON_ADAPTIVE else "adaptive"
            run_id = f"{cfg['run']['name']}-{arm['id']}-s{seed}-{stamp}"
            ctx = RunContext(runs_dir, run_id, {**vcfg, "arm": arm, "variant": DEFAULT_VARIANT},
                             phase=cfg["run"]["phase"], arm=arm["id"], seed=seed)
            ledger.register_run(run_id, alloc, wc["per_run_cap"],
                                {"strategy": strategy, "variant": "evidence"})
            meter = BudgetMeter(ledger, run_id, alloc, wc["per_run_cap"], kind=kind)
            agent = evidence_agent(strategy, wc, (seed, arm["id"], "evidence"))
            stream = TaskStream(stream_cfg)
            t0 = time.time()
            try:
                rows = play(agent, strategy, stream, order, meter, seed, ctx,
                            "reporting_eval" if kind == "reporting" else "exploration",
                            {"strategy": strategy, "variant": "evidence"})
            finally:
                meter.flush()
            recs = world_records(agent, stream, order, rows,
                                 {"run_id": run_id, "seed": seed, "strategy": strategy, "arm": arm["id"],
                                  "namespace": stream_cfg.namespace, "base_seed": stream_cfg.base_seed})
            (ctx.dir / "evidence").mkdir()
            for rec in recs:
                path = ctx.dir / "evidence" / f"{rec['world_key']}.json"
                path.write_text(json.dumps(rec), encoding="utf-8")
            done = len(rows) == len(order)
            ctx.finish("completed", stop_reason="tasks_done" if done else "interaction_cap",
                       interactions={"physical": meter.used, "kind": kind}, evidence_worlds=len(recs),
                       seconds_per_task=round((time.time() - t0) / max(1, len(rows)), 4),
                       session_budget=ledger.summary())
            results.append({"run_id": run_id, "strategy": strategy, "seed": seed, "tasks": len(rows),
                            "complete": done, "interactions": meter.used, "worlds": len(recs)})
            print(f"{run_id}: {len(rows)} tasks, {meter.used} interactions, {len(recs)} worlds", flush=True)
    return results


def load_evidence(runs_dir: str | Path, prefix: str) -> tuple[list[dict], dict]:
    """World records of the evidence runs under ``<runs_dir>/<prefix>*``, one run
    per (strategy, seed) by ``select_runs``, with infos and logs rebuilt.
    Returns (records, {"reruns_used", "ignored"})."""
    runs = []
    for d in Path(runs_dir).glob(f"{prefix}*"):
        needed = (d / "evidence").is_dir() and (d / "manifest.json").exists() and (d / "config.json").exists()
        if not needed:
            continue
        info = _manifest_info(d)
        cfg = json.loads((d / "config.json").read_text())
        runs.append({**info, "strategy": cfg["arm"]["strategy"], "complete": info["status_complete"]})
    chosen, sel = select_runs(runs)
    out = []
    for run in sorted(chosen, key=lambda x: (x["strategy"], x["seed"])):
        for f in sorted((run["dir"] / "evidence").glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["infos"] = {i["signature"]: info_from_record(i) for i in rec["infos"]}
            rec["logs"] = [log_from_record(x) for x in rec["logs"]]
            rec["complete"] = run["complete"]
            out.append(rec)
    return out, sel


def infos_at(rec: dict, k: int) -> dict:
    """The recipes registered in a world's first ``k`` episodes (all recipes when
    the record predates per-episode registration)."""
    if not rec.get("registered"):
        return rec["infos"]
    sigs = set().union(*map(set, rec["registered"][:k]))
    return {s: i for s, i in rec["infos"].items() if s in sigs}


# ================================================================ rankings

def consistent(node: dict, log, infos: dict) -> bool:
    """``simulate(node, log, infos)`` finds no violation (deterministic replay,
    without traces). A hidden recipe missing from ``node`` is replayed with its
    full pool, which is exact for a recipe whose candidates were all present."""
    state = set(log.initial)
    obs = getattr(log, "base_obs", ())
    for t, ((kind, x), g, seen) in enumerate(zip(log.steps, log.goal_obs, log.effect_obs)):
        if kind in (GATHER, ACTIVATE):
            if t >= len(obs) or obs[t] is not False:
                state.add(x)
        elif kind == CRAFT:
            info = infos[x]
            req = info.known if info.known is not None else node.get(x, info.pool)
            eligible = all(b in state for b in req) and all(i in state for i in info.items)
            if info.effect not in state and eligible:
                state.add(info.effect)
            if seen is not None and (info.effect in state) != seen:
                return False
        if (log.goal in state) != g:
            return False
    return True


def relevant(log, infos: dict) -> tuple[str, ...]:
    """Hidden recipes whose hypothesis can change the replay of ``log``: those
    attempted while some pool candidate was absent."""
    rep = EpisodeReplay(log, infos)
    return tuple(sorted({s for t, s in rep.crafts if infos[s].known is None
                         and any(b not in rep.bases[t] for b in infos[s].pool)}))


def _class(info: RecipeInfo, max_size: int) -> list[frozenset]:
    n = len(info.pool)
    return [frozenset(info.pool[p] for p in h) for h in enumerate_hypotheses(n, min(max_size, n))]


def posterior_marginals(infos: dict, logs: list, rng: np.random.Generator, *, max_size: int = 3,
                        exact_work: int = 2_000_000, starts: int = 4, sweeps: int = 40, burn: int = 10,
                        walk_evals: int = 4000) -> tuple[dict, dict]:
    """Marginal P(b in B_r | logs) for every hidden recipe attempted in ``logs``
    (see the module docstring). Hypotheses that contain a candidate eliminated by
    a success are dropped first (they are inconsistent anyway). Each connected
    group of recipes is computed exactly unless that needs more than
    ``exact_work`` log checks; then it is sampled."""
    sigs = sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})
    elim = {e for lg in logs for kind, _, x in observations(lg, infos) if kind == "success" for e in x}
    hyps = {}
    for s in sigs:
        full = _class(infos[s], max_size)
        hyps[s] = [h for h in full if not any((s, b) in elim for b in h)] or full
    rel = [(lg, r) for lg in logs for r in [relevant(lg, infos)] if r]
    parent = {s: s for s in sigs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, r in rel:
        for s in r[1:]:
            parent[find(s)] = find(r[0])
    comps = defaultdict(list)
    for s in sigs:
        comps[find(s)].append(s)
    cache: dict = {}

    def ok(node, i) -> bool:
        lg, r = rel[i]
        key = (i, tuple(node[s] for s in r))
        v = cache.get(key)
        if v is None:
            v = cache[key] = consistent(node, lg, infos)
        return v

    marg: dict = {}
    diag = {"components": len(comps), "exact_recipes": 0, "sampled_recipes": 0, "prior_recipes": 0,
            "chains": 0, "chains_without_start": 0}
    for comp in comps.values():
        idxs = [i for i, (_, r) in enumerate(rel) if r[0] in comp]
        by_sig = {s: [i for i in idxs if s in rel[i][1]] for s in comp}
        m = _exact(comp, hyps, by_sig, idxs, rel, ok, exact_work)
        if m is not None:
            diag["exact_recipes"] += len(comp)
        else:
            m, ch, miss = _gibbs(comp, hyps, by_sig, ok, rng, infos, [rel[i][0] for i in idxs], sigs,
                                 max_size, starts, sweeps, burn, walk_evals)
            diag["chains"] += ch
            diag["chains_without_start"] += miss
            diag["sampled_recipes"] += len(comp) if m is not None else 0
        if m is None:  # inconsistent evidence or no consistent start: the (filtered) prior
            diag["prior_recipes"] += len(comp)
            m = {(s, b): sum(b in h for h in hyps[s]) / len(hyps[s]) for s in comp for b in infos[s].pool}
        marg.update(m)
    return marg, diag


class _Budget(Exception):
    pass


def _exact(comp, hyps, by_sig, idxs, rel, ok, work: int) -> dict | None:
    """Exact marginals of one group, or None when the evidence is inconsistent
    or the enumeration would need more than ``work`` log checks."""
    # order: next the recipe that completes the most logs (early pruning), then the most constrained
    done, at, order, done_at = set(), set(), [], []
    while len(order) < len(comp):
        def gain(s):
            return sum(1 for i in idxs if i not in done and set(rel[i][1]) <= at | {s})
        s = max(sorted(set(comp) - at), key=lambda s: (gain(s), len(by_sig[s]), -len(hyps[s])))
        order.append(s)
        at.add(s)
        new = [i for i in idxs if i not in done and set(rel[i][1]) <= at]
        done.update(new)
        done_at.append(new)
    # the completions of order[d:] depend on the assigned recipes only through the
    # frontier: assigned recipes that share a log not yet complete before depth d
    frontier, closed = [], set()
    for d in range(len(order)):
        frontier.append(sorted({s for i in idxs if i not in closed for s in rel[i][1] if s in order[:d]}))
        closed.update(done_at[d])
    cands = {s: sorted({b for h in hyps[s] for b in h}) for s in order}
    col = {s: {b: j for j, b in enumerate(cands[s])} for s in order}
    width = max(len(c) for c in cands.values())
    memo, node, spent = {}, {}, [0]

    def rec(d):
        if d == len(order):
            return 1.0, np.zeros((len(order), width))
        key = (d, tuple(node[s] for s in frontier[d]))
        if key in memo:
            return memo[key]
        s = order[d]
        n_all, m_all = 0.0, np.zeros((len(order), width))
        for h in hyps[s]:
            node[s] = h
            spent[0] += max(1, len(done_at[d]))
            if spent[0] > work:
                raise _Budget
            if all(ok(node, i) for i in done_at[d]):
                n, m = rec(d + 1)
                if n:
                    n_all += n
                    m_all += m
                    for b in h:
                        m_all[d, col[s][b]] += n
        del node[s]
        memo[key] = (n_all, m_all)
        return n_all, m_all

    try:
        total, counts = rec(0)
    except _Budget:
        return None
    if not total:
        return None
    return {(s, b): counts[d, col[s][b]] / total for d, s in enumerate(order) for b in cands[s]}


def _gibbs(comp, hyps, by_sig, ok, rng, infos, logs_c, sigs, max_size, starts, sweeps, burn, walk_evals):
    acc, chains, missing = defaultdict(float), 0, 0
    order = sorted(comp)
    for _ in range(starts):
        ws = WorldState("posterior", 0.0, max_size)
        ws.infos.update(infos)
        ws.hypotheses = {s: list(hyps[s]) for s in sigs}
        ws.evidence.logs = list(logs_c)
        start = {s: hyps[s][int(rng.integers(len(hyps[s])))] for s in sigs}
        node, st = local_walk(start, ws, None, "focused", rng, max_size, walk_evals, 0.5, 80)
        if st["violations"]:
            missing += 1
            continue
        chains += 1
        node = dict(node)
        chain, n = defaultdict(float), 0
        for sweep in range(burn + sweeps):
            for s in order:
                cur = node[s]
                good = []
                for h in hyps[s]:
                    node[s] = h
                    if all(ok(node, i) for i in by_sig[s]):
                        good.append(h)
                good = good or [cur]
                if sweep >= burn:
                    for h in good:
                        for b in h:
                            chain[(s, b)] += 1.0 / len(good)
                node[s] = good[int(rng.integers(len(good)))]
            n += sweep >= burn
        for k, v in chain.items():
            acc[k] += v / n
    if not chains:
        return None, 0, missing
    out = {(s, b): acc.get((s, b), 0.0) / chains for s in comp for b in infos[s].pool}
    return out, chains, missing


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _tree_prob(f, q: dict) -> float:
    if f is True or f is False:
        return float(f)
    if f[0] == "lit":
        return q[f[1]]
    ps = [_tree_prob(c, q) for c in f[1]]
    if f[0] == "or":
        return 1.0 - math.prod(1.0 - p for p in ps)
    return math.prod(ps)


def _occurrences(f, out: dict) -> dict:
    if f is not True and f is not False:
        if f[0] == "lit":
            out[f[1]] = out.get(f[1], 0) + 1
        else:
            for c in f[1]:
                _occurrences(c, out)
    return out


def formula_prob(f, q: dict) -> float:
    """P(f) for independent literals with probabilities ``q``. The recursion is
    exact for literals that occur once; literals that occur more than once (an
    item input shared by alternative producers) are summed over exactly."""
    rep = sorted(e for e, n in _occurrences(f, {}).items() if n > 1 and 0 < q[e] < 1)
    if not rep:
        return _tree_prob(f, q)
    total = 0.0
    for vals in product((0.0, 1.0), repeat=len(rep)):
        w = math.prod(q[e] if v else 1.0 - q[e] for e, v in zip(rep, vals))
        total += w * _tree_prob(f, {**q, **dict(zip(rep, vals))})
    return total


def _llr(p1: float, p0: float, clip: float = 30.0) -> float:
    if p0 <= 0:
        return clip if p1 > 0 else 0.0
    if p1 <= 0:
        return -clip
    return max(-clip, min(clip, math.log(p1) - math.log(p0)))


def _card_message(qs: list, max_size: int) -> float:
    """Log-likelihood ratio sent by the "1 <= requirement size <= max_size" factor
    to one candidate, the others independent with probabilities ``qs``."""
    dist = np.zeros(len(qs) + 1)
    dist[0] = 1.0
    for q in qs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    p1 = dist[: max_size].sum()  # 1 + K <= max_size
    p0 = dist[1: max_size + 1].sum()  # 1 <= K <= max_size
    return _llr(p1, p0)


def andor_marginals(infos: dict, logs: list, *, max_size: int = 3, iters: int = 200, damping: float = 0.5,
                    tol: float = 1e-9) -> tuple[dict, dict]:
    """(d) beliefs per edge for the hidden recipes attempted in ``logs``."""
    sigs = sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})
    elim, forms, contradictions = set(), [], 0
    for lg in logs:
        for kind, _, x in observations(lg, infos):
            if kind == "success":
                elim |= x
            else:
                forms.append(x)
    factors = []
    for f in forms:
        g = simplify(f, elim)
        if g is False:
            contradictions += 1
        elif g is not True:
            factors.append(("f", g, sorted(literals(g))))
    for s in sigs:
        scope = [(s, b) for b in infos[s].pool if (s, b) not in elim]
        if scope:
            factors.append(("card", s, scope))
    msgs = {(k, v): 0.0 for k, fac in enumerate(factors) for v in fac[2]}
    it, delta = 0, 0.0
    for it in range(1, iters + 1):
        bel = defaultdict(float)
        for (k, v), m in msgs.items():
            bel[v] += m
        new, delta = {}, 0.0
        for k, (kind, body, scope) in enumerate(factors):
            cav = {u: _sigmoid(bel[u] - msgs[(k, u)]) for u in scope}
            for v in scope:
                if kind == "card":
                    lam = _card_message([cav[u] for u in scope if u != v], max_size)
                else:
                    lam = _llr(formula_prob(body, {**cav, v: 1.0}), formula_prob(body, {**cav, v: 0.0}))
                m = damping * msgs[(k, v)] + (1 - damping) * lam
                delta = max(delta, abs(m - msgs[(k, v)]))
                new[(k, v)] = m
        msgs = new
        if delta < tol:
            break
    bel = defaultdict(float)
    for (k, v), m in msgs.items():
        bel[v] += m
    out = {(s, b): 0.0 if (s, b) in elim else _sigmoid(bel[(s, b)]) for s in sigs for b in infos[s].pool}
    return out, {"factors": len(factors), "iterations": it, "converged": delta < tol,
                 "contradictions": contradictions}


def fact_chain(infos: dict, logs: list) -> tuple[list, dict, np.ndarray]:
    """(b) the empirical fact chain: clique expansion of the fact set of each
    episode that reached the goal."""
    facts = sorted({f for i in infos.values() for f in (i.effect, *i.items, *i.pool, *(i.known or ()))})
    idx = {f: k for k, f in enumerate(facts)}
    W = np.zeros((len(facts), len(facts)))
    for lg in logs:
        hits = [t for t, g in enumerate(lg.goal_obs) if g]
        if not hits:
            continue
        tg = hits[0]
        acquired = set(lg.initial) | set(base_sets(lg)[tg]) | {lg.goal}
        acquired |= {infos[x].effect for k, x in lg.steps[: tg + 1] if k == CRAFT and x in infos}
        members = sorted(f for f in acquired if f in idx)
        if len(members) < 2:
            continue
        x = 1.0 / (len(members) - 1)
        for a, b in combinations(members, 2):
            W[idx[a], idx[b]] += x
            W[idx[b], idx[a]] += x
    return facts, idx, W


def rank_scores(infos: dict, logs: list, rng: np.random.Generator, *, restarts=RESTARTS,
                prior_strength: float = PRIOR_STRENGTH, max_size: int = 3,
                posterior_kw: dict | None = None) -> dict:
    """Scores of every ranking method for the hidden recipes attempted in ``logs``;
    PageRank methods for every seed and restart probability in ``restarts``. The
    star-graph scores are exactly what the agents compute from these logs."""
    sigs = sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})
    edges = [(s, b) for s in sigs for b in infos[s].pool]
    post, pdiag = posterior_marginals(infos, logs, rng, max_size=max_size, **(posterior_kw or {}))
    andor, adiag = andor_marginals(infos, logs, max_size=max_size)
    pe = PairwiseEvidence(infos, prior_strength, max_size).observe_logs(logs)
    _, fidx, FW = fact_chain(infos, logs)
    out = {"posterior": {e: post.get(e, 0.0) for e in edges}, "andor": {e: andor[e] for e in edges},
           "pair_weight": {e: pe.weight(*e) for e in edges}, "random": {e: 0.0 for e in edges}}
    goals = [lg.goal for lg in logs[-1:]]
    for a in restarts:
        for how in PPR_SEEDS:
            out[f"star_ppr:{how}@{a:g}"] = star_scores(infos, pe, a, how, goals, sigs)
        out[f"fact_ppr:effect@{a:g}"] = candidate_ppr(fidx, FW, infos, sigs, a, lambda s: (infos[s].effect,))
    return {"scores": out, "sigs": sigs,
            "diagnostics": {"posterior": pdiag, "andor": adiag, "pairwise": dict(pe.counts)}}


# ================================================================== metrics

def _rounded(scores: dict) -> dict:
    return {k: round(float(v), DIGITS) for k, v in scores.items()}


def auroc(scores: dict, positives: set) -> float:
    """Probability that a required candidate outscores a non-required one (ties count one half)."""
    s = _rounded(scores)
    pos = [v for k, v in s.items() if k in positives]
    neg = [v for k, v in s.items() if k not in positives]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def gathering_cost(scores: dict, positives: set) -> float:
    """Expected number of candidates gathered in descending score order (ties in
    random order) until every required candidate is held."""
    s = _rounded(scores)
    before = 0
    cost = 0.0
    for v in sorted(set(s.values()), reverse=True):
        group = [k for k, x in s.items() if x == v]
        m = sum(k in positives for k in group)
        if m:
            cost = before + m * (len(group) + 1) / (m + 1)
        before += len(group)
    return cost


def recipe_metrics(scores: dict, sig: str, pool, truth: set) -> dict:
    sc = {b: scores[(sig, b)] for b in pool}
    pos = set(truth)
    c = gathering_cost(sc, pos)
    return {"auroc": auroc(sc, pos), "cost": c, "excess": c - len(pos)}


# ======================================================= M1 ranking benchmark

def world_rank_metrics(rec: dict, k: int, *, restarts=RESTARTS, posterior_kw: dict | None = None,
                       prior_strength: float = PRIOR_STRENGTH, max_size: int = 3) -> dict | None:
    """Per-world means over the recipes with evidence after the world's first ``k`` episodes."""
    logs = rec["logs"][:k]
    if len(logs) < k:
        return None
    rng = np.random.default_rng(derive_seed("markov_m1", rec["strategy"], rec["seed"], rec["world_key"], k)
                                % 2 ** 32)
    infos = infos_at(rec, k)
    res = rank_scores(infos, logs, rng, restarts=restarts, prior_strength=prior_strength,
                      max_size=max_size, posterior_kw=posterior_kw)
    truth = {s: set(v) for s, v in rec["truth"]["requirements"].items()}
    sigs = [s for s in res["sigs"] if s in truth]
    per = {}
    for name, scores in res["scores"].items():
        ms = [recipe_metrics(scores, s, infos[s].pool, truth[s]) for s in sigs]
        per[name] = {m: float(np.mean([x[m] for x in ms])) if ms else float("nan")
                     for m in ("auroc", "cost", "excess")}
    return {"world": f"s{rec['seed']}:{rec['world_key']}", "seed": rec["seed"], "recipes": len(sigs),
            "metrics": per, "diagnostics": res["diagnostics"], "complete": rec.get("complete", True)}


def _ci(values, level: float = 0.95) -> dict:
    v = [x for x in values if not np.isnan(x)]
    if not v:
        return {"estimate": None, "ci": None, "n": 0}
    return cluster_bootstrap_mean(v, n_boot=N_BOOT, seed=BOOT_SEED, level=level)


def _select_setting(ws: list, method: str, restarts) -> dict:
    """Largest mean per-world AUROC over seeds x restarts; ties within 1e-6: the
    larger restart, then the earlier seed in ``PPR_METHODS[method]``."""
    seeds = PPR_METHODS[method]
    means = {f"{how}@{a:g}": float(np.nanmean([w["metrics"][f"{method}:{how}@{a:g}"]["auroc"] for w in ws]
                                              or [np.nan])) for how in seeds for a in restarts}
    finite = {k: v for k, v in means.items() if not np.isnan(v)}
    if finite:
        best = max(finite.values())
        pick = min((k for k, v in finite.items() if v >= best - 1e-6),
                   key=lambda k: (-float(k.split("@")[1]), seeds.index(k.split("@")[0])))
    else:
        pick = f"{seeds[0]}@{restarts[0]:g}"
    restart = float(pick.split("@")[1])
    return {"by_setting": means, "selected": {"seed": pick.split("@")[0], "restart": restart},
            "no_propagation": restart >= 1.0,
            "at_grid_boundary": restart in (min(restarts), max(restarts))}


def rank_benchmark(worlds: list[dict], *, checkpoints=CHECKPOINTS, restarts=RESTARTS,
                   selection_seeds=SELECTION_SEEDS, test_seeds=TEST_SEEDS, posterior_kw: dict | None = None,
                   prior_strength: float = PRIOR_STRENGTH, expected_test_worlds: int | None = None,
                   progress=None) -> dict:
    """M1 offline benchmark (no interactions). The PageRank settings are selected
    on the worlds of ``selection_seeds`` (``focused_sample`` evidence, last
    checkpoint); the primary contrast is tested on the worlds of ``test_seeds``
    only: AUROC of the posterior minus AUROC of star-graph PageRank (selected
    setting), last checkpoint, ``focused_sample`` evidence, decided by
    ``m1_decision``."""
    per = defaultdict(dict)  # (split, source, k) -> world -> metrics
    for rec in worlds:
        splits = [n for n, seeds in (("selection", selection_seeds), ("test", test_seeds))
                  if rec["seed"] in seeds]
        if not splits:
            continue
        for k in checkpoints:
            r = world_rank_metrics(rec, k, restarts=restarts, posterior_kw=posterior_kw,
                                   prior_strength=prior_strength)
            for split in splits if r is not None else ():
                per[(split, rec["strategy"], k)][r["world"]] = r
            if progress:
                progress(rec, k)
    last = max(checkpoints)
    sel_worlds = list(per.get(("selection", "focused_sample", last), {}).values())
    sel = {m: _select_setting(sel_worlds, m, restarts) for m in PPR_METHODS}
    names = {m: m for m in METHODS}
    for m, v in sel.items():
        names[m] = f"{m}:{v['selected']['seed']}@{v['selected']['restart']:g}"
    table = {}
    for (split, src, k), ws in sorted(per.items()):
        row = {m: {metric: _ci([w["metrics"][names[m]][metric] for w in ws.values()])
                   for metric in ("auroc", "cost", "excess")} for m in METHODS}
        row["n_worlds"] = len(ws)
        row["n_recipes"] = int(sum(w["recipes"] for w in ws.values()))
        table[f"{split}:{src}@{k}"] = row
    pairs = [("star_ppr", "posterior"), ("star_ppr", "fact_ppr"), ("fact_ppr", "random"),
             ("andor", "star_ppr"),
             ("posterior", "andor"), ("star_ppr", "pair_weight"), ("posterior", "random")]
    contrasts = {}
    for (split, src, k), ws in sorted(per.items()):
        for a, b in pairs:
            d = [w["metrics"][names[a]]["auroc"] - w["metrics"][names[b]]["auroc"] for w in ws.values()]
            contrasts[f"{split}:{src}@{k}: {a} - {b}"] = _ci(d)
    test = per.get(("test", "focused_sample", last), {})
    diffs = [w["metrics"]["posterior"]["auroc"] - w["metrics"][names["star_ppr"]]["auroc"]
             for w in test.values()]
    primary = _ci(diffs)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
    complete = bool(test) and all(w["complete"] for w in test.values()) and \
        (expected_test_worlds is None or len(test) >= expected_test_worlds)
    decision = m1_decision(diffs)
    verdict = decision["decision"] if complete else None
    diag = {f"{split}:{src}@{k}": {
        "posterior_exact_recipes": int(sum(w["diagnostics"]["posterior"]["exact_recipes"]
                                           for w in ws.values())),
        "posterior_sampled_recipes": int(sum(w["diagnostics"]["posterior"]["sampled_recipes"]
                                             for w in ws.values())),
        "posterior_prior_recipes": int(sum(w["diagnostics"]["posterior"]["prior_recipes"]
                                           for w in ws.values())),
        "andor_not_converged": int(sum(not w["diagnostics"]["andor"]["converged"] for w in ws.values())),
        "pairwise": {c: int(sum(w["diagnostics"]["pairwise"][c] for w in ws.values()))
                     for c in ("successes", "failures", "explained", "forced", "blamed", "contradictions")},
    } for (split, src, k), ws in sorted(per.items())}
    return {"definition": {"checkpoints": list(checkpoints), "restarts": list(restarts),
                           "prior_strength": prior_strength, "selection_seeds": list(selection_seeds),
                           "test_seeds": list(test_seeds),
                           "unit": "world (mean over its recipes with evidence)",
                           "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "level": 0.95},
                           "primary": f"AUROC posterior - AUROC star_ppr (selected setting), "
                                      f"focused_sample evidence, checkpoint {last}, test worlds",
                           "decision": f"posterior better if the 95% lower end is above 0; PPR better if the "
                                       f"95% upper end is below 0; otherwise equivalent if the 90% interval "
                                       f"lies inside (-{MARGIN:g}, {MARGIN:g}); otherwise inconclusive",
                           "mdd": "minimum detectable difference (two-sided 5%, power 80%): "
                                  "2.80 x SD of the per-world differences / sqrt(worlds)"},
            "restart_selection": sel, "methods": names, "table": table, "contrasts": contrasts,
            "primary": {**primary, "ci90": decision["ci90"], "sd": sd,
                        "mdd": Z_MDD * sd / math.sqrt(len(diffs)) if sd is not None else None},
            "primary_decision": decision,
            "primary_verdict": verdict, "complete": complete, "diagnostics": diag,
            "per_world": {f"{split}:{src}@{k}": ws for (split, src, k), ws in sorted(per.items())}}


def m1_decision(diffs, margin: float | None = None) -> dict:
    """The pre-registered M1 decision on per-world differences AUROC posterior
    minus AUROC star-graph PageRank: "posterior better" if the lower end of the
    95% world-cluster interval is above 0, "PPR better" if its upper end is
    below 0, otherwise "equivalent" if the 90% interval lies inside
    (-margin, margin), otherwise "inconclusive". A significant difference takes
    precedence over equivalence; every flag is reported."""
    margin = MARGIN if margin is None else margin
    c95, c90 = _ci(diffs), _ci(diffs, 0.90)
    if not c95.get("ci"):
        return {"decision": None, "ci90": None, "margin": margin}
    flags = {"posterior_better": c95["ci"][0] > 0, "ppr_better": c95["ci"][1] < 0,
             "equivalent": -margin < c90["ci"][0] and c90["ci"][1] < margin}
    decision = ("posterior better" if flags["posterior_better"] else "PPR better" if flags["ppr_better"]
                else "equivalent" if flags["equivalent"] else "inconclusive")
    return {"decision": decision, "ci90": c90["ci"], "margin": margin, **flags}


def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _fci(c: dict, nd=3) -> str:
    if not c or c.get("estimate") is None:
        return "-"
    return f"{_f(c['estimate'], nd)} [{_f(c['ci'][0], nd)}, {_f(c['ci'][1], nd)}]"


def markdown_m1(s: dict) -> str:
    sel = s["restart_selection"]
    lines = ["PageRank settings selected on the selection worlds: " + ", ".join(
        f"{m} seed {v['selected']['seed']}, restart {v['selected']['restart']:g}"
        + (" (no propagation)" if v["no_propagation"] else "") for m, v in sel.items()) + ".", "",
        "| Split: evidence @ episodes | Worlds | " + " | ".join(METHODS) + " |",
        "|---|---|" + "---|" * len(METHODS)]
    for key, row in s["table"].items():
        cells = " | ".join(_fci(row[m]["auroc"]) for m in METHODS)
        lines.append(f"| {key} | {row['n_worlds']} | {cells} |")
    lines += ["", "Ranked gathering cost (candidates gathered until the requirement is covered):", "",
              "| Split: evidence @ episodes | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for key, row in s["table"].items():
        lines.append(f"| {key} | " + " | ".join(_fci(row[m]["cost"], 2) for m in METHODS) + " |")
    lines += ["", "Paired AUROC contrasts (mean per-world difference [95% CI]):", "",
              "| Contrast | Estimate |",
              "|---|---|"]
    lines += [f"| {k} | {_fci(v)} |" for k, v in s["contrasts"].items()]
    p = s["primary"]
    c90 = p.get("ci90")
    lines += ["", f"Primary ({s['definition']['primary']}): {_fci(p)}, 90% interval "
              + (f"[{_f(c90[0])}, {_f(c90[1])}]" if c90 else "-")
              + f", margin {s['primary_decision']['margin']:g}, minimum detectable difference "
              f"{_f(p.get('mdd'))}; decision: {s['primary_verdict']}."]
    return "\n".join(lines) + "\n"


# =================================================================== twins

def structural_twins(restart: float = 0.15) -> dict:
    """One recipe {a, b, c} -> g against three recipes {a, b}, {b, c}, {a, c} -> g:
    equal unweighted clique expansions, different hypergraphs and cheapest plans."""
    T = {n: i for i, n in enumerate(TYPE_NAMES)}
    a, b, c, g = T["ore"], T["fuel"], T["sand"], T["ingot"]
    worlds = {"one_triple": [((a, b, c), g)], "three_pairs": [((a, b), g), ((b, c), g), ((a, c), g)]}
    facts = [a, b, c, g]
    out = {}
    for name, H in worlds.items():
        clique = np.zeros((4, 4))
        for bases, eff in H:
            members = [facts.index(x) for x in (*bases, eff)]
            for i, j in combinations(members, 2):
                clique[i, j] = clique[j, i] = 1.0
        star = np.zeros((4 + len(H), 4 + len(H)))
        for r, (bases, eff) in enumerate(H):
            for x in (*bases, eff):
                star[4 + r, facts.index(x)] = star[facts.index(x), 4 + r] = 1.0
        seed_c, seed_s = np.eye(4)[3], np.eye(4 + len(H))[3]
        recipes = [StructRecipe(eff, (), bases) for bases, eff in H]
        ds, _, _ = enumerate_derivations(recipes, frozenset(), g)
        edges = sorted((facts[i], facts[j]) for i, j in zip(*np.nonzero(np.triu(clique))))
        out[name] = {"clique_edges": edges,
                     "clique_ppr": [round(float(x), 9)
                                    for x in personalized_pagerank(clique, seed_c, restart)[:3]],
                     "star_ppr": [round(float(x), 9)
                                  for x in personalized_pagerank(star, seed_s, restart)[:3]],
                     "h_add_goal": min(1 + len(bases) for bases, _ in H),
                     "cheapest_plan": plan_cost(ds[0]), "cheapest_bases": len(ds[0][1])}
    A, B = out["one_triple"], out["three_pairs"]
    out["clique_equal"] = A["clique_edges"] == B["clique_edges"] and A["clique_ppr"] == B["clique_ppr"]
    out["star_differs"] = A["star_ppr"] != B["star_ppr"]
    out["plans_differ"] = A["cheapest_plan"] != B["cheapest_plan"]
    return out


def twin_infos() -> dict:
    T = {n: i for i, n in enumerate(TYPE_NAMES)}
    p1 = tuple(sorted(T[x] for x in ("ore", "fuel", "sand", "wood", "clay", "fiber")))
    p2 = tuple(sorted(T[x] for x in ("stone", "oil", "salt", "wax", "resin", "flint")))
    r1 = RecipeInfo("ingot<=[]+pool{p1}", T["ingot"], (), p1, None)
    r2 = RecipeInfo("key<=[ingot]+pool{p2}", T["key"], (T["ingot"],), p2, None)
    return {r1.signature: r1, r2.signature: r2}


def scripted_log(log_id: int, infos: dict, truth: dict, goal: int, gathers, crafts) -> EpisodeLog:
    """Deterministic episode generated from hidden requirements ``truth``
    (evaluator side), observed in goal-only mode."""
    state, lg = set(), EpisodeLog(log_id, goal, frozenset())
    for b in gathers:
        state.add(b)
        lg.steps.append((GATHER, b))
        lg.goal_obs.append(goal in state)
        lg.effect_obs.append(None)
    for sig in crafts:
        info = infos[sig]
        if all(b in state for b in truth[sig]) and all(i in state for i in info.items):
            state.add(info.effect)
        lg.steps.append((CRAFT, sig))
        lg.goal_obs.append(goal in state)
        lg.effect_obs.append(None)
    return lg


def behavioural_twins(restart: float = 0.15, prior_strength: float = PRIOR_STRENGTH) -> dict:
    """Two worlds that differ only in hidden requirements (A: ingot needs ore,
    key needs oil; B: ingot needs fuel, key needs stone), observed under one
    fixed schedule: two brute-force episodes, an omission of ore and stone
    (fails in both worlds, for different reasons) and a probe omitting stone
    (succeeds in A, fails in B). Reported after each stage for the edge
    (ingot recipe, ore): each method's score and whether its ranking of the
    ingot recipe's candidates differs between the worlds."""
    T = {n: i for i, n in enumerate(TYPE_NAMES)}
    infos = twin_infos()
    r1, r2 = sorted(infos, key=lambda s: infos[s].effect != T["ingot"])
    goal = T["key"]
    truths = {"A": {r1: {T["ore"]}, r2: {T["oil"]}}, "B": {r1: {T["fuel"]}, r2: {T["stone"]}}}
    everything = sorted(infos[r1].pool + infos[r2].pool)
    schedule = [("brute_force", everything), ("brute_force", everything),
                ("omit_ore_stone", [b for b in everything if b not in (T["ore"], T["stone"])]),
                ("omit_stone", [b for b in everything if b != T["stone"]])]
    stages = {"after_brute_force": 2, "after_omission": 3, "after_probe": 4}
    out = {"edge": "(ingot recipe, ore)", "schedule": [s for s, _ in schedule], "stages": {}}
    for stage, n in stages.items():
        res = {}
        for world, truth in truths.items():
            logs = [scripted_log(i, infos, truth, goal, gathers, [r1, r2]) for i, (_, gathers) in
                    enumerate(schedule[:n])]
            sc = rank_scores(infos, logs, np.random.default_rng(0), restarts=(restart,),
                             prior_strength=prior_strength)
            res[world] = {"goal_outcomes": [bool(any(lg.goal_obs)) for lg in logs],
                          "scores": {m: round(sc["scores"][m][(r1, T["ore"])], 6) for m in sc["scores"]},
                          "order": {m: _order(sc["scores"][m], r1, infos[r1].pool) for m in sc["scores"]}}
        out["stages"][stage] = {
            "worlds": res,
            "observations_identical": res["A"]["goal_outcomes"] == res["B"]["goal_outcomes"],
            "ranking_differs": {m: res["A"]["order"][m] != res["B"]["order"][m] for m in res["A"]["order"]},
            "score_differs": {m: res["A"]["scores"][m] != res["B"]["scores"][m] for m in res["A"]["scores"]}}
    return out


def _order(scores: dict, sig: str, pool) -> list:
    """Tie-aware ranking: candidates grouped by equal (rounded) score, best first."""
    s = {b: round(scores[(sig, b)], DIGITS) for b in pool}
    return [sorted(TYPE_NAMES[b] for b in pool if s[b] == v) for v in sorted(set(s.values()), reverse=True)]


# ============================================================ M2 factorial

CELLS = {"focused_sample": ("hyper", "sample"), "hyper_rank": ("hyper", "rank"),
         "pair_sample": ("pair", "sample"), "pair_rank": ("pair", "rank")}
EXTRA = "hyper_sample_ppr"
ANCHORS = ("maximal", "reference")
LATE_FROM, EPISODES = 8, 16
SUCCESS_MARGIN = 0.02
SELECTION_LEVEL = 0.975  # Bonferroni over the two main effects


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_runs(runs_dir: str | Path, prefix: str) -> tuple[list[dict], dict]:
    """Walker runs (not evidence runs) whose directory starts with ``prefix``,
    one per (strategy, seed) by ``select_runs``."""
    runs = []
    for d in Path(runs_dir).glob(f"{prefix}*"):
        if not ((d / "manifest.json").exists() and (d / "config.json").exists()) or (d / "evidence").is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        strategy = (cfg.get("arm") or {}).get("strategy")
        if strategy is None:
            continue
        info = _manifest_info(d)
        rows = _read_jsonl(d / "eval.jsonl")
        runs.append({**info, "strategy": strategy, "rows": rows,
                     "complete": info["status_complete"] and len(rows) == cfg["eval"]["n_tasks"]})
    return select_runs(runs)


def _ratio(rows) -> float:
    opt = sum(r["reference_length"] or 0 for r in rows)
    return sum(r["primitive_length"] for r in rows) / opt if opt else float("nan")


def world_endpoints(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["world_key"]].append(r)
    out = {}
    for wk, rs in by.items():
        late = [r for r in rs if r["world_episode"] >= LATE_FROM]
        replans = max(1, sum(r.get("replans", 0) for r in rs))
        out[wk] = {"late_cost": _ratio(late),
                   "late_success": float(np.mean([r["success"] for r in late])) if late else float("nan"),
                   "n_late": len(late), "evals": float(sum(r.get("evals", 0) for r in rs)),
                   "replans": float(replans),
                   "evals_per_replan": float(sum(r.get("evals", 0) for r in rs)) / replans,
                   "choose_seconds_per_replan": (float(sum(r["choose_seconds"] for r in rs)) / replans
                                                 if all("choose_seconds" in r for r in rs) else float("nan")),
                   "cost_all": _ratio(rs)}
    return out


def factorial(runs: list[dict], expected_worlds: int | None = None) -> dict:
    """M2 endpoints: per-world late-half (episodes 9-16) cost ratio and success,
    the representation and selector main effects and their interaction as
    paired per-world contrasts (primary: the representation effect on late
    cost, 95% interval), the pre-registered selection rule (97.5% intervals,
    Bonferroni over the two main effects), the extra arm against
    focused_sample (secondary) and the anchors."""
    tables = defaultdict(dict)
    for run in runs:
        for wk, v in world_endpoints(run["rows"]).items():
            tables[run["strategy"]][(run["seed"], wk)] = v
    full = EPISODES - LATE_FROM
    worlds = sorted(w for w in set.intersection(*(set(tables.get(s, {})) for s in CELLS))
                    if all(tables[s][w]["n_late"] == full for s in CELLS)) \
                        if all(s in tables for s in CELLS) else []
    t = {s: tables.get(s, {}) for s in (*CELLS, EXTRA, *ANCHORS)}
    effects_fn = {
        "representation (hyper - pair)":
            lambda c: 0.5 * ((c["focused_sample"] - c["pair_sample"]) + (c["hyper_rank"] - c["pair_rank"])),
        "selector (rank - sample)":
            lambda c: 0.5 * ((c["hyper_rank"] - c["focused_sample"]) + (c["pair_rank"] - c["pair_sample"])),
        "interaction":
            lambda c: (c["hyper_rank"] - c["focused_sample"]) - (c["pair_rank"] - c["pair_sample"]),
    }
    effects = {}
    for name, fn in effects_fn.items():
        effects[name] = {}
        for metric in ("late_cost", "late_success"):
            vals = [fn({s: t[s][w][metric] for s in CELLS}) for w in worlds]
            effects[name][metric] = _ci(vals)
            effects[name][f"{metric}_selection"] = _ci(vals, SELECTION_LEVEL)
    incomplete = [r["run_id"] for r in runs if not r["complete"]]
    complete = bool(worlds) and not incomplete and (expected_worlds is None or len(worlds) >= expected_worlds)
    def winner(name, lo_level, hi_level):
        e = effects[name]
        c, s = e["late_cost_selection"], e["late_success"]["estimate"]
        if not c.get("ci") or s is None:
            return None
        if c["ci"][1] < 0 and s >= -SUCCESS_MARGIN:
            return lo_level
        if c["ci"][0] > 0 and -s >= -SUCCESS_MARGIN:
            return hi_level
        return None

    chosen, message = None, None  # selection outcomes need the complete run set
    if complete:
        chosen = {"representation": winner("representation (hyper - pair)", "hyper", "pair"),
                  "selector": winner("selector (rank - sample)", "rank", "sample")}
        message = None if any(v is not None for v in chosen.values()) else "no component brings more"
    mean_cost = {s: float(np.mean([t[s][w]["late_cost"] for w in worlds])) if worlds else None for s in CELLS}
    best, versus = None, None
    if chosen and any(v is not None for v in chosen.values()):
        eligible = [s for s, (r, k) in CELLS.items()
                    if chosen["representation"] in (None, r) and chosen["selector"] in (None, k)]
        best = min(eligible, key=lambda s: mean_cost[s])
        runner = min((s for s in CELLS if s != best), key=lambda s: mean_cost[s])
        d = _ci([t[best][w]["late_cost"] - t[runner][w]["late_cost"] for w in worlds])
        versus = {"runner_up": runner, "difference": d,
                  "status": "confirmed" if d.get("ci") and d["ci"][1] < 0 else "descriptive"}
    rep = effects["representation (hyper - pair)"]["late_cost"]
    primary_verdict = None if not complete or not rep.get("ci") else (
        "hyper lowers late cost" if rep["ci"][1] < 0 else
        "pair lowers late cost" if rep["ci"][0] > 0 else "no difference shown")
    extra_worlds = [w for w in worlds if w in t[EXTRA] and t[EXTRA][w]["n_late"] == full]
    extra = {m: _ci([t[EXTRA][w][m] - t["focused_sample"][w][m] for w in extra_worlds])
             for m in ("evals", "evals_per_replan", "late_cost", "late_success")}
    seconds = {}
    for r in runs:
        reps = sum(x.get("replans", 0) for x in r["rows"])
        if r["wallclock_s"] and reps:
            seconds.setdefault(r["strategy"], []).append(r["wallclock_s"] / reps)
    arms = {}
    for s in (*CELLS, EXTRA, *ANCHORS):
        ws = [w for w in worlds if w in t[s]]
        arms[s] = {"worlds": len(ws),
                   "run_seconds_per_replan": float(np.mean(seconds[s])) if s in seconds else None,
                   **{m: float(np.nanmean([t[s][w][m] for w in ws])) if ws else None
                      for m in ("late_cost", "late_success", "cost_all", "evals", "evals_per_replan",
                                "choose_seconds_per_replan")}}
    return {"definition": {"late_episodes": f"{LATE_FROM + 1}-{EPISODES}", "unit": "world",
                           "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED},
                           "primary": "representation main effect (hyper - pair) on late-half cost, "
                                      "95% interval",
                           "rule": "a level brings more when choosing it lowers late-half cost with a 97.5% "
                                   "interval (Bonferroni over the two main effects) excluding 0 and lowers "
                                   "late-half success by at most 2 percentage points; the winning cell is "
                                   "descriptive unless its paired 95% interval against the runner-up "
                                   "excludes 0"},
            "n_worlds": len(worlds), "expected_worlds": expected_worlds, "complete": complete,
            "incomplete_runs": incomplete, "effects": effects, "primary_verdict": primary_verdict,
            "chosen": chosen, "component_message": message, "winning_cell": best,
            "winning_vs_runner_up": versus,
            "mean_late_cost": mean_cost, "extra_vs_focused_sample": {"n_worlds": len(extra_worlds), **extra},
            "arms": arms}


def markdown_m2(s: dict) -> str:
    lines = ["| Arm | Worlds | Late cost ratio | Late success | All episodes | Evaluations / world "
             "| Evaluations / replan | Seconds / replan (run) | Choice seconds / replan |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a, v in s["arms"].items():
        lines.append(f"| `{a}` | {v['worlds']} | {_f(v['late_cost'], 2)} | {_f(v['late_success'], 2)} | "
                     f"{_f(v['cost_all'], 2)} | {_f(v['evals'], 0)} | {_f(v['evals_per_replan'], 1)} | "
                     f"{_f(v['run_seconds_per_replan'], 4)} | {_f(v['choose_seconds_per_replan'], 4)} |")
    lines += ["", "| Effect | Late cost [95% CI] | Late cost [97.5% CI] | Late success [95% CI] |",
              "|---|---|---|---|"]
    for k, v in s["effects"].items():
        lines.append(f"| {k} | {_fci(v['late_cost'])} | {_fci(v['late_cost_selection'])} | "
                     f"{_fci(v['late_success'])} |")
    e = s["extra_vs_focused_sample"]
    w = s["winning_vs_runner_up"]
    lines += ["", f"Primary (representation main effect on late cost): {s['primary_verdict']}.",
              f"Secondary, `{EXTRA}` - `focused_sample` ({e['n_worlds']} worlds): evaluations per world "
              f"{_fci(e['evals'], 1)}, per replan {_fci(e['evals_per_replan'], 2)}, "
              f"late cost {_fci(e['late_cost'])}, "
              f"late success {_fci(e['late_success'])}.",
              f"Selected levels: {s['chosen']}"
              + (f" ({s['component_message']})" if s["component_message"] else "")
              + f"; winning cell `{s['winning_cell']}`"
              + (f" ({w['status']} against `{w['runner_up']}`: {_fci(w['difference'])})" if w else "")
              + f"; complete: {s['complete']} ({s['n_worlds']} paired worlds)."]
    return "\n".join(lines) + "\n"
