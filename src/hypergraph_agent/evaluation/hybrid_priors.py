"""Studies HP1 and HP2 (docs/studies/HP1_size_prior.md, docs/studies/HP2_schema_priors.md), evaluator side:
the offline evaluation of requirement-size priors (HP1), the analysis of the schema-prior runs (HP2)
and the privileged ``oracle_schema`` agent. Agents never import this module.

    python -m hypergraph_agent.evaluation.hybrid_priors hp1 --config <hp1 config> --runs runs --out <dir>
    python -m hypergraph_agent.evaluation.hybrid_priors hp2 --runs runs --root <checkout> --out <dir>

HP1 (offline, reads logs only, no interaction). For every source (an evidence policy of study M's
M1 collection or of study X2's generated sets), every stream (one run: a seed's worlds in play
order), every world and checkpoint k, the methods read the world's first k logs and the recipes
registered by then; the scored recipes are the hidden recipes attempted in those logs (as in M1 and
X2). Priors: ``uniform`` (the class prior, today's posterior), ``oracle`` (the generator's size
weights; privileged, the bound), ``learned_world`` (sizes learned by empirical Bayes from the recipes
pinned in this world's first k logs), ``learned_stream`` (from those and from every recipe pinned
in the stream's earlier worlds, each read at its last checkpoint), and ``marginal_world`` and
``marginal_stream`` (the same hyperprior updated by the exact likelihood of all the evidence, through
the size polynomials of ``priors.posterior``, instead of the pinned recipes). Posteriors are exact
(``priors.posterior``: weighted enumeration for goal-only logs, the factorized updater with items
observable), with weighted Gibbs sampling for a group beyond the work budget. A recipe is pinned when
the uniform posterior leaves one hypothesis. Metrics per recipe against the hidden requirement:
AUROC (ties one half), the F1 of the inferred set (candidates with marginal at least 0.5, else the
top tie group; X2's rule), the ranked gathering cost and the Brier score of the marginals; a world's
value is the mean over its scored recipes. Tests are paired per-world differences with a percentile
world-cluster bootstrap (10,000 resamples, seed 20261008).

HP2 (online runs of ``configs/hybrid_priors/``): per setting (config), the earliest completed walker
run per arm and seed, one code identity; per unit (a world, or a stream in the stream-scope
settings, whose worlds share one schema) the late cost ratio (episodes 9-16, primitive steps over
optimal steps), late success and search work; paired contrasts with the same bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..agents.hybrid import HybridAgent, PriorSource, hybrid_params, walker_kwargs
from ..envs.generator import family_of
from ..priors.schema import MAX_SIZE, rainbow
from ..topology.inference import size_log_prior
from .stats import cluster_bootstrap_mean, sign_test

N_BOOT, BOOT_SEED = 10_000, 20261008
PRIORS = ("uniform", "oracle", "learned_world", "learned_stream", "marginal_world", "marginal_stream")
METRICS = ("auroc", "f1", "precision", "recall", "gather", "brier")
DIGITS = 9


# ======================================================================= oracle (privileged)

class OracleSchema:
    """PRIVILEGED: the true schema of a world as the prior (HP2's reference
    point), mixed with the uniform prior at ``floor`` so that every hypothesis
    keeps a positive prior. In worlds without a schema the truth is the
    generator's size prior (its arity weights). Evaluator side only."""

    def __init__(self, schema, n_families: int, arity_weights, floor: float = 0.01,
                 max_size: int = MAX_SIZE):
        if not 0.0 < floor < 1.0:
            raise ValueError("the oracle floor must lie in (0, 1)")
        self.schema, self.F, self.floor, self.max_size = schema, n_families, floor, max_size
        self.arity = tuple(float(x) for x in arity_weights)

    def log_prior(self, family: int, pool, hyps) -> np.ndarray:
        n = len(hyps)
        if self.schema.families:
            cls = tuple(self.schema.classes[b] for b in pool)
            pred = rainbow(pool, cls, self.schema.families[family])
            p = np.array([pred.get(h, 0.0) for h in hyps])
        else:
            index = {b: i for i, b in enumerate(pool)}
            p = np.exp(size_log_prior(self.arity, [tuple(index[b] for b in sorted(h)) for h in hyps]))
        return np.log((1.0 - self.floor) * p / p.sum() + self.floor / n)


class OraclePrior(PriorSource):
    """The true schema of the world being played (set by ``OracleSchemaAgent``)."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.truth: dict = {}

    def set_truth(self, world_key: str, oracle: OracleSchema) -> None:
        self.truth[world_key] = oracle

    def version(self, w) -> object:
        return id(self.truth[w.world_key])

    def log_vector(self, w, sig) -> np.ndarray:
        info = w.infos[sig]
        return self.truth[w.world_key].log_prior(family_of(info.effect, int(self.params["n_families"])),
                                                 info.pool, w.hypotheses[sig])

    def counters(self, w) -> dict:
        return {"hp_observed": 0, "hp_pinned": 0, "hp_schema_weight": 1.0}


class OracleSchemaAgent(HybridAgent):
    """PRIVILEGED: ``oracle_schema`` reads the hidden schema of each world before
    its episodes (the evaluator's knowledge, as the reference reads the rules)."""

    def run_episode(self, ex, task, dyn_seed: int) -> dict:
        world = task.world
        if world.schema is None:
            raise ValueError("oracle_schema plays schema worlds only")
        if world.world_key not in self.prior.truth:
            self.prior.set_truth(world.world_key, OracleSchema(world.schema, int(self.params["n_families"]),
                                                               world.config.arity_weights,
                                                               self.params["oracle_floor"]))
        return super().run_episode(ex, task, dyn_seed)


def make_oracle_agent(rng, wc: dict, epsilon: float) -> OracleSchemaAgent:
    params = hybrid_params(wc.get("hybrid"))
    return OracleSchemaAgent("oracle_schema", rng, params, OraclePrior(params), **walker_kwargs(wc, epsilon))


# ============================================================================= HP1: config

HP1_KEYS = {"name", "sources", "checkpoints", "priors", "hyperprior_strength", "oracle_sizes", "primary",
            "secondary", "posterior", "bootstrap", "m1_artifact"}
SOURCE_KEYS = {"name", "study", "prefix", "set", "strategy", "namespace", "expected_worlds"}
TEST_KEYS = {"name", "source", "k", "a", "b", "metric", "holds", "margin"}
HOLDS = ("above", "below", "equivalent", "decision")


def load_hp1(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return validate_hp1(raw, str(path))


def validate_hp1(raw: dict, source: str = "<dict>") -> dict:
    """An HP1 config, validated (unknown keys are errors). A development config
    (name ending in ``-dev``) may read only namespaces ending in ``_dev``, and
    every other config none of them."""
    cfg = json.loads(json.dumps(raw))
    if set(cfg) - HP1_KEYS:
        raise KeyError(f"unknown HP1 keys {sorted(set(cfg) - HP1_KEYS)}")
    for k in ("name", "sources", "checkpoints", "primary"):
        if k not in cfg:
            raise KeyError(f"HP1 config needs {k!r}")
    cfg.setdefault("priors", list(PRIORS))
    cfg.setdefault("hyperprior_strength", 2.0)
    cfg.setdefault("oracle_sizes", [0.2, 0.5, 0.3])
    cfg.setdefault("secondary", [])
    cfg.setdefault("posterior", {})
    cfg.setdefault("bootstrap", {"resamples": N_BOOT, "seed": BOOT_SEED})
    cfg.setdefault("m1_artifact", None)
    if set(cfg["priors"]) - set(PRIORS) or "uniform" not in cfg["priors"]:
        raise ValueError(f"priors must be a subset of {PRIORS} holding 'uniform'")
    dev = cfg["name"].endswith("-dev")
    names = set()
    for s in cfg["sources"]:
        if set(s) - SOURCE_KEYS or {"name", "study", "prefix", "strategy", "namespace"} - set(s):
            raise KeyError(f"source {s.get('name')!r}: keys must be within {sorted(SOURCE_KEYS)}")
        if s["study"] not in ("m1", "x2") or (s["study"] == "x2") != ("set" in s):
            raise ValueError(f"source {s['name']!r}: study m1, or x2 with a set")
        if dev != s["namespace"].endswith("_dev"):
            raise ValueError(f"source {s['name']!r}: development configs read only namespaces ending in "
                             f"'_dev', and only they may")
        if s["name"] in names:
            raise ValueError(f"duplicate source {s['name']!r}")
        names.add(s["name"])
        s.setdefault("expected_worlds", None)
    for t in [dict(cfg["primary"], name="primary", holds="decision")] + cfg["secondary"]:
        if set(t) - TEST_KEYS:
            raise KeyError(f"test {t.get('name')!r}: unknown keys {sorted(set(t) - TEST_KEYS)}")
        bad = (t["source"] not in names or t["a"] not in cfg["priors"] or t["b"] not in cfg["priors"]
               or t["metric"] not in METRICS or int(t["k"]) not in cfg["checkpoints"]
               or t["holds"] not in HOLDS)
        if bad:
            raise ValueError(f"test {t.get('name')!r} is not well defined")
    if "margin" not in cfg["primary"]:
        raise KeyError("the primary needs a margin")
    cfg["_source"] = source
    return cfg


# ============================================================================ HP1: evidence

def load_source(runs_dir: str | Path, s: dict) -> tuple[list[dict], dict]:
    """World records of one source, with the run selection of the study that
    collected them (study M: ``markov_study.load_evidence``; X2:
    ``collection.load_evidence``), and the (commit, code) of the runs used."""
    if s["study"] == "m1":
        from .markov_study import load_evidence
        recs, sel = load_evidence(runs_dir, s["prefix"])
        recs = [r for r in recs if r["strategy"] == s["strategy"]]
    else:
        from ..inference_baselines.collection import load_evidence
        recs, sel = load_evidence(runs_dir, s["prefix"])
        recs = [r for r in recs if r.get("set") == s["set"] and r["strategy"] == s["strategy"]]
    bad = sorted({r["namespace"] for r in recs} - {s["namespace"]})
    if bad:
        raise ValueError(f"source {s['name']!r} found records of namespaces {bad}")
    prov = set()
    for run_id in sorted({r["run_id"] for r in recs}):
        man = json.loads((Path(runs_dir) / run_id / "manifest.json").read_text())
        src = man.get("source") or {}
        prov.add((src.get("commit"), src.get("loaded_code_sha256")))
    return recs, {"reruns_used": sel.get("reruns_used", []), "ignored": sel.get("ignored", []),
                  "provenance": sorted(prov, key=str)}


def play_rank(rec: dict) -> int:
    if rec.get("world_index") is not None:
        return int(rec["world_index"])
    return min(r["order"] for r in rec["rows"])


def streams(recs: list[dict]) -> dict:
    """{run id: its world records in play order}."""
    out = defaultdict(list)
    for r in recs:
        out[r["run_id"]].append(r)
    return {k: sorted(v, key=play_rank) for k, v in sorted(out.items())}


# ============================================================================= HP1: metrics

def _rounded(scores: dict) -> dict:
    return {k: round(float(v), DIGITS) for k, v in scores.items()}


def recipe_metrics(sc: dict, truth: set) -> dict:
    from ..inference_baselines.benchmark import inferred_set, set_metrics
    from .markov_study import auroc, gathering_cost
    pool = sorted(sc)
    S = inferred_set(sc, pool)
    brier = float(np.mean([(float(sc[b]) - (b in truth)) ** 2 for b in pool]))
    return {"auroc": auroc(sc, truth), "gather": gathering_cost(sc, truth), "brier": brier,
            **set_metrics(S, truth)}


def world_checkpoint(rec: dict, k: int, prev: dict, cfg: dict) -> dict | None:
    """Every prior's per-world metrics at checkpoint ``k``, the pinned recipes and
    the learned sizes. ``prev``: the pinned sizes and size polynomials of the
    stream's earlier worlds (at their last checkpoint)."""
    from ..envs.generator import derive_seed
    from ..inference_baselines.benchmark import observable, scored_sigs
    from ..priors.posterior import UniformPrior, factorized_marginals, joint_marginals, pinned
    from ..priors.size import REFERENCE_POOL, SizePrior, learned_sizes, marginal_sizes
    from .markov_study import infos_at
    logs = rec["logs"][:k]
    if len(logs) < k:
        return None
    infos = infos_at(rec, k)
    sigs = scored_sigs(infos, logs)
    if any(len(infos[s].pool) != REFERENCE_POOL for s in sigs):
        raise ValueError("HP1 learns sizes for pools of the reference size only")
    obs = observable(rec, logs, infos) if rec.get("observe_items") == "all" else False
    pk = cfg.get("posterior", {})
    seed = derive_seed("hp1", rec.get("set"), rec["strategy"], rec["seed"], rec["world_key"], k) % 2 ** 32
    cache: dict = {}
    t0 = time.time()

    def marginals(priors, polys=None):
        if obs:
            return factorized_marginals(infos, logs, sigs, priors, polys=polys)
        return joint_marginals(infos, logs, priors, seed, cache=cache, polys=polys, **pk)

    polys: list = []
    (m_uni,), diag = marginals([UniformPrior()], polys)
    # pins only from exactly computed groups: a sampled marginal of 0 or 1 is no proof (reported apart)
    exact = set(diag.get("exact_sigs", ()))
    pins_all = pinned(m_uni, infos, sigs)
    pins = {s: v for s, v in pins_all.items() if s in exact}
    pins_sampled = {s: v for s, v in pins_all.items() if s not in exact}
    sizes_here = [len(v) for v in pins.values()]
    strength = float(cfg["hyperprior_strength"])
    priors = {"uniform": m_uni}
    learned = {"learned_world": learned_sizes(sizes_here, strength),
               "learned_stream": learned_sizes(list(prev["sizes"]) + sizes_here, strength),
               "marginal_world": marginal_sizes(polys, strength),
               "marginal_stream": marginal_sizes(list(prev["polys"]) + polys, strength)}
    rest = [p for p in cfg["priors"] if p != "uniform"]
    objs = []
    for p in rest:
        sizes = tuple(cfg["oracle_sizes"]) if p == "oracle" else learned[p]
        objs.append(SizePrior(sizes, p))
    if objs:
        ms, _ = marginals(objs)
        priors.update(zip(rest, ms))
    truth = {s: set(v) for s, v in rec["truth"]["requirements"].items()}
    scored = [s for s in sigs if s in truth]
    per = {}
    for p, m in priors.items():
        rows = [recipe_metrics({b: m[(s, b)] for b in infos[s].pool}, truth[s]) for s in scored]
        per[p] = {x: float(np.mean([r[x] for r in rows])) if rows else float("nan") for x in METRICS}
    wrong = sum(1 for s, req in pins.items() if s in truth and set(req) != truth[s])
    wrong_sampled = sum(1 for s, req in pins_sampled.items() if s in truth and set(req) != truth[s])
    return {"k": k, "recipes": len(scored), "metrics": per, "pinned": len(pins), "pinned_wrong": wrong,
            "pinned_sampled": len(pins_sampled), "pinned_sampled_wrong": wrong_sampled,
            "pinned_sizes": dict(Counter(sizes_here)),
            "true_sizes_scored": dict(Counter(len(truth[s]) for s in scored)),
            "learned": {p: [round(x, 6) for x in v] for p, v in learned.items()},
            "posterior": {kk: diag.get(kk) for kk in ("form", "exact_recipes", "sampled_recipes",
                                                       "prior_recipes", "poly_groups_left_out")},
            "seconds": round(time.time() - t0, 3), "_pins": {s: sorted(v) for s, v in pins.items()},
            "_polys": polys}


def stream_rows(args) -> list[dict]:
    """Rows of one stream (one run), its worlds in play order: the stream priors
    of a world use every earlier world of the stream at its last checkpoint
    (its pinned recipes; its size polynomials). Module level, so a process pool
    can run streams in parallel."""
    source, recs, cfg = args
    rows, prev = [], {"sizes": [], "polys": []}
    last = max(cfg["checkpoints"])
    for rec in recs:
        end = {"sizes": [], "polys": []}
        for k in cfg["checkpoints"]:
            r = world_checkpoint(rec, k, prev, cfg)
            if r is None:
                continue
            r.update({"source": source, "run_id": rec["run_id"], "seed": rec["seed"],
                      "world": f"s{rec['seed']}:{rec['world_key']}", "world_rank": play_rank(rec),
                      "complete": rec.get("complete", True)})
            if k == last:
                end = {"sizes": [len(v) for v in r["_pins"].values()], "polys": r["_polys"]}
            r.pop("_pins")
            r.pop("_polys")
            rows.append(r)
        prev = {"sizes": prev["sizes"] + end["sizes"], "polys": prev["polys"] + end["polys"]}
    return rows


# ========================================================================== HP1: statistics

def _ci(values, level: float = 0.95, cfg: dict | None = None) -> dict:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"estimate": None, "ci": None, "n": 0}
    b = (cfg or {}).get("bootstrap", {})
    return cluster_bootstrap_mean(v, n_boot=b.get("resamples", N_BOOT), seed=b.get("seed", BOOT_SEED),
                                  level=level)


LOWER_IS_BETTER = ("brier", "gather")


def decision(diffs, margin: float, labels=("learned better", "uniform better"), cfg=None,
             lower_is_better: bool = False) -> dict:
    """Study M's four outcomes on paired per-world differences a - b: "a better"
    if the 95% interval lies entirely on a's side of 0 (above 0, or below 0 for
    a metric where lower is better), "b better" if it lies entirely on the other
    side, otherwise "equivalent" if the 90% interval lies inside (-margin,
    margin), otherwise "inconclusive"; a significant difference takes
    precedence."""
    c95, c90 = _ci(diffs, 0.95, cfg), _ci(diffs, 0.90, cfg)
    if not c95.get("ci"):
        return {"decision": None, "ci95": None, "ci90": None, "margin": margin}
    above, below = c95["ci"][0] > 0, c95["ci"][1] < 0
    flags = {"a_better": below if lower_is_better else above,
             "b_better": above if lower_is_better else below,
             "equivalent": -margin < c90["ci"][0] and c90["ci"][1] < margin}
    d = (labels[0] if flags["a_better"] else labels[1] if flags["b_better"]
         else "equivalent" if flags["equivalent"] else "inconclusive")
    return {"decision": d, "estimate": c95["estimate"], "ci95": c95["ci"], "ci90": c90["ci"],
            "margin": margin, "n": c95["n"], **flags}


def cell(rows: list[dict], source: str, k: int) -> dict:
    return {r["world"]: r for r in rows if r["source"] == source and r["k"] == k}


def run_test(t: dict, rows: list[dict], status: dict, cfg: dict) -> dict:
    """One contrast a - b on per-world differences, in the vocabulary of studies
    X2 and RL: ``criterion`` is the test's rule (``holds`` in the config); a
    "decision" test reports study M's four-way ``decision``, the others report
    ``holds`` (True or False); both are None when the test is incomplete (a
    missing world, a source with problems, or no worlds)."""
    ws = cell(rows, t["source"], int(t["k"]))
    diffs = [w["metrics"][t["a"]][t["metric"]] - w["metrics"][t["b"]][t["metric"]] for w in ws.values()]
    out = {**{k: t[k] for k in ("name", "source", "k", "a", "b", "metric")}, "criterion": t["holds"],
           "n_worlds": len(ws)}
    st = status.get(t["source"], {})
    expected = st.get("expected_worlds")
    problems = list(st.get("problems", []))
    if expected is not None and len(ws) < expected:
        problems.append(f"{len(ws)} of {expected} worlds")
    margin = t.get("margin", cfg["primary"]["margin"])
    dec = decision(diffs, float(margin), (f"{t['a']} better", f"{t['b']} better"), cfg,
                   lower_is_better=t["metric"] in LOWER_IS_BETTER)
    out.update({k: dec.get(k) for k in ("estimate", "ci95", "ci90", "decision")})
    out["margin"] = margin
    if diffs:
        out["sign_test"] = sign_test(diffs)
        out["sd"] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
        out["mdd"] = 2.80 * out["sd"] / math.sqrt(len(diffs)) if out["sd"] is not None else None
    complete = not problems and dec["decision"] is not None
    out.update({"complete": complete, "problems": problems or ([] if complete else ["no worlds"])})
    if not complete:
        out["decision"] = None
        if t["holds"] != "decision":
            out["holds"] = None
        return out
    if t["holds"] == "above":
        out["holds"] = dec["ci95"][0] > 0
    elif t["holds"] == "below":
        out["holds"] = dec["ci95"][1] < 0
    elif t["holds"] == "equivalent":
        out["holds"] = bool(dec["equivalent"])
    return out


def result_of(t: dict):
    """The result of a test: its decision for a "decision" test, else ``holds``."""
    return t.get("decision") if t["criterion"] == "decision" else t.get("holds")


def stream_bootstrap(t: dict, rows: list[dict], cfg: dict) -> dict:
    """Descriptive: the contrast's mean over stream means, with a percentile
    bootstrap over streams (the worlds of a stream share their stream priors)."""
    by = defaultdict(list)
    for w in cell(rows, t["source"], int(t["k"])).values():
        by[w["run_id"]].append(w["metrics"][t["a"]][t["metric"]] - w["metrics"][t["b"]][t["metric"]])
    c = _ci([float(np.mean(v)) for v in by.values()], 0.95, cfg)
    return {"clusters": len(by), "worlds": int(sum(len(v) for v in by.values())),
            "estimate": c.get("estimate"), "ci95": c.get("ci")}


def validity(rows: list[dict], cfg: dict, m1_artifact: dict | None) -> dict:
    """V1: on brute-force evidence every prior is at chance (AUROC exactly 0.5)
    with identical inferred sets; V2: no pin of an exactly computed group
    differs from the truth; V3: the uniform arm reproduces study M's published
    per-world posterior AUROC wherever M and HP1 both computed every recipe
    exactly. ``holds`` is True or False, or None when the test does not apply
    (no brute-force source, no exact pin, no comparable world)."""
    maximal = [r for r in rows if any(s["name"] == r["source"] and s["strategy"] == "maximal"
                                      for s in cfg["sources"])]
    bad_v1 = [f"{r['source']}:{r['world']}@{r['k']}" for r in maximal
              if any(r["metrics"][p]["auroc"] != 0.5 for p in r["metrics"] if r["recipes"])
              or len({round(r["metrics"][p]["f1"], DIGITS) for p in r["metrics"]}) > 1]
    v1 = {"name": "V1 brute-force evidence at chance for every prior", "checked": len(maximal),
          "failures": bad_v1[:20], "holds": not bad_v1 if maximal else None}
    wrong, n_pins = sum(r["pinned_wrong"] for r in rows), sum(r["pinned"] for r in rows)
    v2 = {"name": "V2 every pin of an exactly computed group equals the hidden requirement",
          "pinned": int(n_pins), "wrong": int(wrong), "holds": wrong == 0 if n_pins else None,
          "pinned_sampled": int(sum(r.get("pinned_sampled", 0) for r in rows)),
          "pinned_sampled_wrong": int(sum(r.get("pinned_sampled_wrong", 0) for r in rows))}
    v3 = {"name": "V3 uniform arm reproduces study M's published posterior", "holds": None}
    if m1_artifact is not None:
        checked, gaps = 0, []
        m1_sources = {s["name"]: s for s in cfg["sources"] if s["study"] == "m1"}
        for r in rows:
            s = m1_sources.get(r["source"])
            if s is None:
                continue
            for split in ("selection", "test"):
                w = m1_artifact["per_world"].get(f"{split}:{s['strategy']}@{r['k']}", {}).get(r["world"])
                if w is None:
                    continue
                d = w["diagnostics"]["posterior"]
                if d["sampled_recipes"] or d["prior_recipes"] or r["posterior"].get("sampled_recipes"):
                    continue
                checked += 1
                gap = abs(w["metrics"]["posterior"]["auroc"] - r["metrics"]["uniform"]["auroc"])
                if not gap < 1e-9:
                    gaps.append({"world": r["world"], "k": r["k"], "gap": gap})
        v3 = {**v3, "checked": checked, "failures": gaps[:20], "holds": not gaps if checked else None}
    return {"V1": v1, "V2": v2, "V3": v3}


def _size_counts(ws: dict, field: str) -> dict:
    total: Counter = Counter()
    for w in ws.values():
        total.update({int(a): b for a, b in w[field].items()})
    return dict(sorted(total.items()))


def summarize_hp1(rows: list[dict], status: dict, cfg: dict, m1_artifact: dict | None = None) -> dict:
    table = {}
    for s in cfg["sources"]:
        for k in cfg["checkpoints"]:
            ws = cell(rows, s["name"], k)
            if not ws:
                continue
            table[f"{s['name']}@{k}"] = {
                "n_worlds": len(ws), "n_recipes": int(sum(w["recipes"] for w in ws.values())),
                "pinned": int(sum(w["pinned"] for w in ws.values())),
                "pinned_sampled": int(sum(w.get("pinned_sampled", 0) for w in ws.values())),
                "sampled_recipes": int(sum(w["posterior"].get("sampled_recipes") or 0 for w in ws.values())),
                "metrics": {p: {m: _ci([w["metrics"][p][m] for w in ws.values()], 0.95, cfg)
                                for m in ("auroc", "f1", "gather", "brier")} for p in cfg["priors"]},
                "learned_mean": {p: np.mean([w["learned"][p] for w in ws.values()], axis=0).round(4).tolist()
                                 for p in ("learned_world", "learned_stream", "marginal_world",
                                           "marginal_stream")},
                "pinned_size_counts": _size_counts(ws, "pinned_sizes"),
                "true_size_counts_scored": _size_counts(ws, "true_sizes_scored"),
            }
    primary_test = dict(cfg["primary"], name="primary", holds="decision")
    primary = run_test(primary_test, rows, status, cfg)
    secondary = [run_test(t, rows, status, cfg) for t in cfg["secondary"]]
    val = validity(rows, cfg, m1_artifact)
    invalid = [v["name"] for v in val.values() if v["holds"] is False]
    # studies X2 and RL: None when the primary's data are incomplete, then "invalid" when a validity
    # test fails (one that does not apply, None, does not), otherwise the primary decision
    verdict = None if not primary["complete"] else "invalid" if invalid else primary["decision"]
    return {"definition": {"priors": cfg["priors"], "checkpoints": cfg["checkpoints"],
                           "hyperprior_strength": cfg["hyperprior_strength"],
                           "oracle_sizes": cfg["oracle_sizes"],
                           "unit": "world (mean over its scored recipes)",
                           "bootstrap": cfg["bootstrap"], "primary": cfg["primary"]},
            "sources": status, "table": table, "primary": primary, "primary_verdict": verdict,
            "primary_stream_bootstrap": stream_bootstrap(primary_test, rows, cfg),
            "validity": val, "invalid": invalid, "secondary": secondary,
            "seconds_total": round(sum(r["seconds"] for r in rows), 1)}


def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _fci(c: dict, nd=3) -> str:
    if not c or c.get("estimate") is None:
        return "-"
    return f"{_f(c['estimate'], nd)} [{_f(c['ci'][0], nd)}, {_f(c['ci'][1], nd)}]"


def markdown_hp1(s: dict) -> str:
    priors = s["definition"]["priors"]
    lines = ["# HP1: requirement-size priors (offline)", "",
             "| Source @ episodes | Worlds | Pinned | " + " | ".join(f"AUROC {p}" for p in priors) + " |",
             "|---|---|---|" + "---|" * len(priors)]
    for key, row in s["table"].items():
        lines.append(f"| {key} | {row['n_worlds']} | {row['pinned']} | "
                     + " | ".join(_fci(row["metrics"][p]["auroc"]) for p in priors) + " |")
    lines += ["", "| Source @ episodes | " + " | ".join(f"F1 {p}" for p in priors) + " |",
              "|---|" + "---|" * len(priors)]
    for key, row in s["table"].items():
        lines.append(f"| {key} | " + " | ".join(_fci(row["metrics"][p]["f1"]) for p in priors) + " |")
    p, sb = s["primary"], s.get("primary_stream_bootstrap") or {}
    lines += ["", f"Primary ({p['a']} - {p['b']}, {p['metric']}, {p['source']} @ {p['k']}): "
              f"{_f(p.get('estimate'), 4)} 95% {p.get('ci95')} 90% {p.get('ci90')}, margin {p['margin']}: "
              f"**{s['primary_verdict']}**",
              f"Descriptive, clustered by stream ({sb.get('clusters')} streams): "
              f"{_f(sb.get('estimate'), 4)} 95% {sb.get('ci95')}", "",
              "| Test | Estimate | 95% interval | Result |", "|---|---|---|---|"]
    for t in s["secondary"]:
        ci = t.get("ci95")
        lines.append(f"| {t['name']} | {_f(t.get('estimate'), 4)} | "
                     + (f"[{_f(ci[0], 4)}, {_f(ci[1], 4)}]" if ci else "-") + f" | {result_of(t)} |")
    lines += [""] + [f"- {v['name']}: {v['holds']}" for v in s["validity"].values()]
    return "\n".join(lines) + "\n"


# ============================================================================== HP1: driver

def hp1_status(cfg: dict, runs_dir: str | Path) -> tuple[dict, dict]:
    """(records per source, status per source): completeness, provenance and
    world counts, without computing anything (the dry run)."""
    recs, status = {}, {}
    for s in cfg["sources"]:
        r, sel = load_source(runs_dir, s)
        recs[s["name"]] = r
        problems = []
        if not r:
            problems.append("no records")
        if any(not x.get("complete", True) for x in r):
            problems.append("records from incomplete runs")
        if len(sel.get("provenance", [])) > 1:
            problems.append("runs from more than one (commit, code)")
        status[s["name"]] = {"runs": len(streams(r)), "worlds": len(r),
                             "expected_worlds": s["expected_worlds"],
                             "provenance": [list(x) for x in sel.get("provenance", [])],
                             "reruns_used": sel.get("reruns_used", []), "ignored": sel.get("ignored", []),
                             "problems": problems}
    return recs, status


def run_hp1(cfg: dict, runs_dir: str | Path, workers: int = 1, cache_dir: str | Path | None = None,
            resume: bool = False, progress=print) -> tuple[list[dict], dict]:
    from ..training.run import LOADED_CODE_SHA256
    from ..envs.generator import stable_hash
    recs, status = hp1_status(cfg, runs_dir)
    jobs = [(name, ws, cfg) for name, r in recs.items() for ws in streams(r).values()]
    key = stable_hash({"code": LOADED_CODE_SHA256, "cfg": {k: v for k, v in cfg.items() if k != "_source"}})
    cdir = Path(cache_dir) / key if cache_dir else None
    rows, todo = [], []
    for j in jobs:
        f = cdir / f"{j[0]}__{j[1][0]['run_id']}.json" if cdir else None
        if resume and f is not None and f.exists():
            rows += json.loads(f.read_text(encoding="utf-8"))
        else:
            todo.append((j, f))

    def done(res, f, j):
        rows.extend(res)
        if f is not None:  # every finished stream is cached at once, whatever the others do
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".tmp")  # written whole, then renamed: a kill mid-write
            tmp.write_text(json.dumps(res), encoding="utf-8")  # leaves no partial file for --resume
            os.replace(tmp, f)
        progress(f"{j[0]} {j[1][0]['run_id']}: {len(res)} rows")

    def failed(exc, j):  # the stream's source becomes incomplete; its finished streams are kept
        why = f"stream {j[1][0]['run_id']} failed: {type(exc).__name__}: {exc}"
        status[j[0]].setdefault("problems", []).append(why)
        status[j[0]].setdefault("failed_streams", []).append(j[1][0]["run_id"])
        progress(f"{j[0]} {why}")

    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(stream_rows, j): (j, f) for j, f in todo}
            for fut in as_completed(futs):
                j, f = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 (one stream must not discard the others)
                    failed(exc, j)
                else:
                    done(res, f, j)
    else:
        for j, f in todo:
            try:
                res = stream_rows(j)
            except Exception as exc:  # noqa: BLE001
                failed(exc, j)
            else:
                done(res, f, j)
    rows.sort(key=lambda r: (r["source"], r["run_id"], r["world_rank"], r["k"]))
    return rows, status


# ============================================================================== HP2 analysis

HP2_LATE = 8
HP2_SETTINGS = {  # setting: (config, unit) ; the stream-scope settings are clustered by stream
    "v1_world": ("configs/hybrid_priors/hp2_v1_world.yaml", "world"),
    "v1_stream": ("configs/hybrid_priors/hp2_v1_stream.yaml", "stream"),
    "v2_world": ("configs/hybrid_priors/hp2_v2_world.yaml", "world"),
    "v2_stream": ("configs/hybrid_priors/hp2_v2_stream.yaml", "stream"),
    "none": ("configs/hybrid_priors/hp2_none.yaml", "world"),
}
HP2_DEV_SETTINGS = {k: (c.replace("/hp2_", "/hp2_dev_"), u) for k, (c, u) in HP2_SETTINGS.items()}
EQUIV_SHARE = 0.10  # falsification: equivalence within 10% of posterior's mean
HARM_SHARE = 0.05  # harm check: upper end below 5% of posterior's mean
SUCCESS_MARGIN = 0.02
# (name, setting, metric, a, b, rule): ``below`` holds when the 95% upper end is below 0, ``above``
# when the lower end is above 0, ``harm`` when the upper end is below HARM_SHARE x mean of b,
# ``not_lower`` when the lower end is above -SUCCESS_MARGIN, ``descriptive`` never decides
HP2_TESTS = (
    ("S1 oracle_schema - posterior, late cost ratio, v1 world", "v1_world", "cost_ratio_late",
     "oracle_schema", "focused_sample", "below"),
    ("S2 hybrid - oracle_schema, late cost ratio, v1 world", "v1_world", "cost_ratio_late",
     "hybrid", "oracle_schema", "descriptive"),
    ("S3 rules_only - hybrid, late cost ratio, v1 world", "v1_world", "cost_ratio_late",
     "rules_only", "hybrid", "above"),
    ("S4 hybrid - posterior, late success, v1 world", "v1_world", "success_late",
     "hybrid", "focused_sample", "not_lower"),
    ("S5 hybrid - posterior, late cost ratio, v1 stream", "v1_stream", "cost_ratio_late",
     "hybrid", "focused_sample", "below"),
    ("S6 hybrid - posterior, late cost ratio, v2 world", "v2_world", "cost_ratio_late",
     "hybrid", "focused_sample", "below"),
    ("S7 hybrid - posterior, late cost ratio, v2 stream", "v2_stream", "cost_ratio_late",
     "hybrid", "focused_sample", "below"),
    ("H1 hybrid - posterior, late cost ratio, no schema (harm check)", "none", "cost_ratio_late",
     "hybrid", "focused_sample", "harm"),
    ("S8 focused_sample - maximal, late cost ratio, v1 world", "v1_world", "cost_ratio_late",
     "focused_sample", "maximal", "below"),
)
TABLE_KEYS = ("cost_ratio_late", "cost_ratio_early", "cost_ratio_all", "success_late",
              "first_success_steps", "evals_per_replan", "choose_seconds_per_replan", "observed_end",
              "pinned_end", "schema_weight_end", "models_end")  # descriptive means per arm and setting


def _mean_of(units: list[dict], key: str) -> float | None:
    """Mean of ``key`` over units, skipping units without a number (the hybrid
    counters are absent from the other arms' rows); None when no unit has one."""
    v = [float(u[key]) for u in units if isinstance(u.get(key), (int, float))]
    return float(np.mean(v)) if v else None


def config_identity(runs_dir: Path, run_id: str) -> str:
    """The hash of a run's recorded config without the fields that differ
    between the runs of one arm of one config: the evaluation base seed
    (10,000 + 1,000 per seed), the seed list and the arm list (a rerun of one
    pair, ``walk --seeds s --arms a``, rewrites both; the run's own arm stays
    in its ``arm`` field) and the config's file path. Equal for every seed of
    one arm of one config, first runs and reruns alike."""
    from ..envs.generator import stable_hash
    cfg = json.loads((Path(runs_dir) / run_id / "config.json").read_text(encoding="utf-8"))
    for key in ("_source", "arms"):
        cfg.pop(key, None)
    cfg.get("eval", {}).pop("base_seed", None)
    cfg.get("run", {}).pop("seeds", None)
    return stable_hash(cfg)


def clingo_version() -> str | None:
    try:
        import clingo
    except ImportError:
        return None
    return clingo.__version__


def hp2_units(runs_dir: Path, config: str, unit: str) -> tuple[dict, dict, dict]:
    """({arm: {unit: metrics}}, {arm: problems}, validity counts) of one HP2
    config: per arm and seed the earliest completed run, one code identity, and
    one config identity per arm (``config_identity``). A stream unit's value of
    every metric is the mean over the stream's worlds."""
    from ..config import load_config
    from .policygraph_study import unit_metrics
    from .replication import load_runs
    cfg = load_config(config)
    name = cfg["run"]["name"]
    runs = [r for r in load_runs(runs_dir, f"{name}-") if r["run_id"].startswith(f"{name}-")]
    chosen = {}
    for r in sorted(runs, key=lambda r: (r["start"], r["run_id"])):
        if r["complete"]:
            chosen.setdefault((r["strategy"], r["seed"]), r)
    codes = {(r["commit"], r["code"]) for r in chosen.values()}
    arms, problems, failures, overflow = {}, defaultdict(list), Counter(), Counter()
    for a in cfg["arms"]:
        arm = a["strategy"]
        missing = [s for s in cfg["run"]["seeds"] if (arm, s) not in chosen]
        problems[arm] += [f"seed {s}: no completed run" for s in missing]
        if len(codes) > 1:
            problems[arm].append(f"runs from {len(codes)} code identities")
        idents = {config_identity(runs_dir, chosen[(arm, s)]["run_id"])
                  for s in cfg["run"]["seeds"] if (arm, s) in chosen}
        if len(idents) > 1:
            problems[arm].append(f"runs from {len(idents)} configs")
        per_world = {}
        for s in cfg["run"]["seeds"]:
            r = chosen.get((arm, s))
            if r is None:
                continue
            by = defaultdict(list)
            for row in r["rows"]:
                by[row["world_key"]].append(row)
                failures[arm] += int(not row["success"])
            for wk, rs in by.items():
                m = unit_metrics(rs, HP2_LATE)
                replans = max(1, sum(x.get("replans", 0) for x in rs))
                choose = sum(x.get("choose_seconds", 0.0) for x in rs)
                last = rs[-1]  # rows are in play order: the world's last episode
                m.update({"evals_per_replan": sum(x.get("evals", 0) for x in rs) / replans,
                          "choose_seconds_per_replan": choose / replans,
                          "observed_end": last.get("hp_observed"), "pinned_end": last.get("hp_pinned"),
                          "schema_weight_end": last.get("hp_schema_weight"),
                          "models_end": last.get("hp_models"), "_stream": f"s{s}"})
                overflow[arm] += int((last.get("hp_overflow") or 0) > 0)
                per_world[f"s{s}:{wk}"] = m
        if unit == "stream":
            by_stream = defaultdict(list)
            for m in per_world.values():
                by_stream[m["_stream"]].append(m)
            arms[arm] = {u: {k: _mean_of(ms, k) for k in ms[0] if k != "_stream"}
                         for u, ms in by_stream.items()}
        else:
            arms[arm] = per_world
    # overflow_worlds: worlds whose hidden-class learner had refused a refit (an overflow) by their end
    return arms, dict(problems), {"failed_episodes": dict(failures), "overflow_worlds": dict(overflow)}


def hp2_contrast(name, metric, a: dict, b: dict, expected: int, problems: list, rule: str) -> dict:
    units = sorted(set(a) & set(b))
    d = [a[u][metric] - b[u][metric] for u in units]
    out = {"name": name, "metric": metric, "rule": rule, "n_units": len(units), "expected_units": expected}
    if units:
        c = cluster_bootstrap_mean(d, N_BOOT, BOOT_SEED)
        c90 = cluster_bootstrap_mean(d, N_BOOT, BOOT_SEED, level=0.90)
        mean_b = float(np.mean([b[u][metric] for u in units]))
        out.update({"estimate": c["estimate"], "ci": c["ci"], "ci90": c90["ci"], "sign_test": sign_test(d),
                    "mean_a": float(np.mean([a[u][metric] for u in units])), "mean_b": mean_b})
    if problems or len(units) < expected:
        out.update({"verdict": "incomplete", "problems": problems or [f"{len(units)} of {expected} units"]})
        return out
    lo, hi = out["ci"]
    out["verdict"] = {"below": "holds" if hi < 0 else "does not hold",
                      "above": "holds" if lo > 0 else "does not hold",
                      "harm": "holds" if hi < HARM_SHARE * out["mean_b"] else "does not hold",
                      "not_lower": "holds" if lo > -SUCCESS_MARGIN else "does not hold",
                      "descriptive": "descriptive"}[rule]
    return out


def hp2_analysis(runs_dir: Path, root: Path = Path("."), settings=None,
                 expected: dict | None = None) -> dict:
    from ..config import load_config
    settings = settings or HP2_SETTINGS
    data, out = {}, {"settings": {}, "tests": []}
    for key, (config, unit) in settings.items():
        path = str(Path(root) / config)
        if not Path(path).exists():
            continue
        cfg = load_config(path)
        arms, problems, val = hp2_units(runs_dir, path, unit)
        n_seeds, n_worlds = len(cfg["run"]["seeds"]), cfg["eval"]["n_worlds"]
        exp = (expected or {}).get(key, n_seeds * (n_worlds if unit == "world" else 1))
        data[key] = (arms, problems, exp)
        table = {arm: {k: _mean_of(list(us.values()), k) for k in TABLE_KEYS}
                 for arm, us in sorted(arms.items()) if us}
        out["settings"][key] = {"config": path, "unit": unit, "expected_units": exp, "validity": val,
                                "table": table, "problems": problems}

    def test(name, setting, metric, a, b, rule):
        if setting not in data:
            return {"name": name, "verdict": "incomplete", "problems": [f"no config {setting}"]}
        arms, problems, exp = data[setting]
        probs = list(problems.get(a, [])) + list(problems.get(b, []))
        return hp2_contrast(name, metric, arms.get(a, {}), arms.get(b, {}), exp, probs, rule)

    primary = test("P1 hybrid - posterior, late cost ratio, v1 world", "v1_world", "cost_ratio_late",
                   "hybrid", "focused_sample", "below")
    e1 = None
    if primary.get("ci90") and primary["verdict"] != "incomplete":
        m = EQUIV_SHARE * primary["mean_b"]
        e1 = {"name": "E1 equivalence within 10% of posterior's late cost ratio", "margin": m,
              "ci90": primary["ci90"], "holds": -m < primary["ci90"][0] and primary["ci90"][1] < m}
    # V1 reads the reference's runs of the primary setting; without all of them it is undetermined (None)
    ref = data.get("v1_world")
    ref_runs = ref is not None and "reference" in ref[0] and not ref[1].get("reference")
    v1 = None if not ref_runs else \
        out["settings"]["v1_world"]["validity"]["failed_episodes"].get("reference", 0) == 0
    if primary["verdict"] == "incomplete" or e1 is None or v1 is None:
        verdict = None
    elif not v1:
        verdict = "invalid"
    else:
        p, f = primary["verdict"] == "holds", e1["holds"]
        verdict = ("pass" if p and not f else "negligible" if p and f
                   else "falsified" if f else "inconclusive")
    out.update({"primary": primary, "falsification": e1, "validity_reference_solves_all": v1,
                "verdict": verdict, "tests": [test(*t) for t in HP2_TESTS], "clingo": clingo_version()})
    return out


# ===================================================================================== CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.evaluation.hybrid_priors")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("hp1", help="study HP1: size priors on recorded logs (offline)")
    a.add_argument("--config", required=True)
    a.add_argument("--runs", default="runs")
    a.add_argument("--out", default="artifacts/hybrid_priors")
    a.add_argument("--dry-run", action="store_true", help="list the evidence; compute nothing")
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--cache", default=None, help="per-stream cache, keyed by the code and the config")
    a.add_argument("--resume", action="store_true")
    b = sub.add_parser("hp2", help="study HP2: endpoints of the schema-prior runs")
    b.add_argument("--runs", default="runs")
    b.add_argument("--root", default=".", help="checkout whose configs/ are read (the snapshot)")
    b.add_argument("--out", default="artifacts/hybrid_priors")
    b.add_argument("--dev", action="store_true", help="the development configs (hp2_dev_*.yaml)")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "hp1":
        cfg = load_hp1(args.config)
        if args.dry_run:
            _, status = hp1_status(cfg, args.runs)
            print(json.dumps({"config": cfg["_source"], "sources": status,
                              "checkpoints": cfg["checkpoints"], "priors": cfg["priors"]}, indent=1))
            return 0
        t0 = time.time()
        rows, status = run_hp1(cfg, args.runs, args.workers, args.cache, args.resume)
        art = None
        if cfg.get("m1_artifact") and Path(cfg["m1_artifact"]).exists():
            art = json.loads(Path(cfg["m1_artifact"]).read_text(encoding="utf-8"))
        summary = summarize_hp1(rows, status, cfg, art)
        summary["seconds_wall"] = round(time.time() - t0, 1)
        stem = cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.json").write_text(json.dumps({**summary, "rows": rows}, indent=1, default=str),
                                          encoding="utf-8")
        (out / f"{stem}.md").write_text(markdown_hp1(summary), encoding="utf-8")
        print(markdown_hp1(summary))
        return 0
    res = hp2_analysis(Path(args.runs), Path(args.root), HP2_DEV_SETTINGS if args.dev else HP2_SETTINGS)
    out.mkdir(parents=True, exist_ok=True)
    stem = "hp2_dev_summary" if args.dev else "hp2_summary"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for t in [res["primary"]] + res["tests"]:
        ci = "-" if not t.get("ci") else f"{t['estimate']:.3f} [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}]"
        print(f"{t['name']}: {ci} -> {t['verdict']}")
    print(f"verdict: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
