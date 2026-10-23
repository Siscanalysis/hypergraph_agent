"""Study XS (docs/studies/XS_learner_followup.md): the combined analysis of several member configs.

    python -m hypergraph_agent.baselines analyze --config configs/baselines/xs_rq.yaml

A study file has one top-level key ``xs``: ``name``, ``members`` (label -> member config path; a
relative path is taken from the root of the tree holding the study file, as in ``config_path``) and
``analysis``. Each member is an ordinary config of this package (one budget slack each in study
XS) and runs on its own; the members must share the family, the training namespace, seeds and
budgets, the validation and test streams and the agents, and their learner ids must be unique
across members.

Arms. Learner arms keep their ids; agent arms are relabelled ``<agent>.<label>`` (the agent on the
member's streams, for example ``random.s12``). Derived arms:

- ``<learner>~<agent>``: per unit, the learner's metrics minus those of ``<agent>.<label of the
  learner's member>`` (the agent at the learner's own slack, same split and world);
- ``<arm>#<budget>``: a learner arm (or derived arm) at a training budget other than the test's;
- in tests, ``best_learner`` is the selected learner and ``<agent>.best`` the agent at its slack.

Selection (validation only): the learner with the lowest mean of ``<learner>~<select_against>``
on ``select_metric`` over the validation worlds at the largest budget, each world's value the
mean over the training seeds; ties go to the earlier learner (member order, then config order).
Unless every planned job of every member has a completed run and every learner has a finite
score, there is no best learner and every test is ``incomplete``. Tests are those of
``analysis.run_test`` (paired by test world, world-cluster bootstrap, sign test).
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from . import analysis as xa
from .config import EXPECT, METRICS, TEST_KEYS, config_path, load_config
from ..training.budget import SessionLedger
from .runner import completed, missing_caps, read_jsonl

STUDY_DEFAULTS = {"name": None, "members": {}, "analysis": {
    "late_from": 8, "n_boot": 10_000, "boot_seed": 20261008, "select_metric": "cost_ratio_late",
    "select_against": "random", "tests": []}}
RESERVED = (".", "#", "~", "@")
SHARED_BLOCKS = (("setting", "family"), ("train", "namespace"), ("train", "base_seed"), ("train", "seeds"),
                 ("train", "budgets"), ("test", None), ("validation", None), ("agents", None),
                 ("analysis", "late_from"))


def is_study(path: str | Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return isinstance(raw, dict) and set(raw) == {"xs"}


def load_study(path: str | Path) -> dict:
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("xs") or {}
    extra = set(raw) - set(STUDY_DEFAULTS)
    if extra:
        raise KeyError(f"unknown study keys {sorted(extra)}")
    study = {**copy.deepcopy(STUDY_DEFAULTS), **{k: v for k, v in raw.items() if k != "analysis"}}
    bad = set(raw.get("analysis", {})) - set(STUDY_DEFAULTS["analysis"])
    if bad:
        raise KeyError(f"unknown study analysis keys {sorted(bad)}")
    study["analysis"] = {**STUDY_DEFAULTS["analysis"], **raw.get("analysis", {})}
    if not study["name"] or not study["members"]:
        raise ValueError("a study needs a name and members")
    members = {}
    for label, ref in study["members"].items():
        if any(c in label for c in RESERVED) or label == "best":
            raise ValueError(f"member label {label!r} may not be 'best' or contain {RESERVED}")
        members[label] = load_config(config_path(path, ref))
    first = next(iter(members.values()))
    for label, cfg in members.items():
        for block, key in SHARED_BLOCKS:
            a, b = (cfg[block], first[block]) if key is None else (cfg[block][key], first[block][key])
            if a != b:
                raise ValueError(f"member {label!r} differs from the first member in {block}.{key or '*'}")
        if study["analysis"]["select_against"] not in cfg["validation_agents"]:
            raise ValueError(f"member {label!r} must list {study['analysis']['select_against']!r} in "
                             f"validation_agents")
        if cfg["analysis"]["late_from"] != study["analysis"]["late_from"]:
            raise ValueError(f"member {label!r} and the study differ in analysis.late_from")
    ids = [ln["id"] for cfg in members.values() for ln in cfg["learners"]]
    if len(ids) != len(set(ids)) or any(c in i for i in ids for c in RESERVED):
        raise ValueError(f"learner ids must be unique across members and avoid {RESERVED}: {ids}")
    if any(len(ln["modes"]) != 1 for cfg in members.values() for ln in cfg["learners"]):
        raise ValueError("every learner of a study is evaluated in one action mode")
    tests = study["analysis"]["tests"]
    for t in tests:
        if not {"name", "a", "metric"} <= set(t) or set(t) - TEST_KEYS:
            raise KeyError(f"study test needs name, a, metric and optionally b (got {sorted(t)})")
        if t["metric"] not in METRICS or t.get("expect", "less") not in EXPECT:
            raise ValueError(f"study test {t['name']!r}: unknown metric or expectation")
    if [t.get("role") for t in tests].count("primary") > 1:
        raise ValueError("at most one primary test per study")
    study["_members"], study["_source"] = members, str(path)
    return study


def learner_member(study: dict) -> dict:
    return {ln["id"]: label for label, cfg in study["_members"].items() for ln in cfg["learners"]}


def collect_rows(study: dict) -> tuple[list[dict], dict, dict]:
    """Rows of every member (agents relabelled), bookkeeping and completeness per member."""
    rows, book, comp = [], {}, {}
    for label, cfg in study["_members"].items():
        rs, b = xa.collect(cfg)
        for r in rs:
            rows.append({**r, "arm": f"{r['arm']}.{label}"} if r["kind"] == "agent" else r)
        book[label], comp[label] = b, xa.completeness(cfg)
    return rows, book, comp


def _difference(cell: dict, base: dict) -> dict:
    def diff(a, b):
        return {u: {m: a[u][m] - b[u][m] for m in a[u] if m in b[u] and m != "first_success_censored"}
                for u in a if u in b}
    return {"units": diff(cell["units"], base["units"]),
            "seeds": {s: diff(m, base["units"]) for s, m in cell["seeds"].items()}}


def derived_tables(tabs: dict, study: dict) -> dict:
    """``tabs`` with the derived arms ``<learner>~<agent>`` and the budget aliases ``<arm>#<budget>``."""
    member = learner_member(study)
    agents = study["_members"][next(iter(study["_members"]))]["agents"]
    out = dict(tabs)
    for (arm, b, sp), cell in tabs.items():
        if b is None or arm not in member:
            continue
        for agent in agents:
            base = tabs.get((f"{agent}.{member[arm]}", None, sp))
            if base is not None:
                out[(f"{arm}~{agent}", b, sp)] = _difference(cell, base)
    for (arm, b, sp), cell in list(out.items()):
        if b is not None:
            out[(f"{arm}#{b}", None, sp)] = cell
    return out


def select(tabs: dict, study: dict, top: int, complete: bool) -> dict:
    a = study["analysis"]
    metric, against = a["select_metric"], a["select_against"]
    order = [ln["id"] for cfg in study["_members"].values() for ln in cfg["learners"]]
    scores = {}
    for lid in order:
        cell = tabs.get((f"{lid}~{against}", top, "validation"))
        scores[lid] = xa._mean(cell["units"], metric) if cell else float("nan")
    sign = -1.0 if metric.startswith("success") else 1.0
    out = {"metric": f"{metric} minus {against} at the learner's slack", "split": "validation", "budget": top,
           "scores": scores, "best_learner": None, "best_member": None, "note": None}
    missing = [k for k, v in scores.items() if not np.isfinite(v)]
    if not complete or missing:
        return {**out, "note": f"incomplete: planned jobs without a completed run, or no score for {missing}"}
    best = min(order, key=lambda k: (sign * scores[k], order.index(k)))
    return {**out, "best_learner": best, "best_member": learner_member(study)[best]}


def resolve(name: str | None, sel: dict) -> str | None:
    if name is None:
        return None
    best, label = sel["best_learner"], sel["best_member"]
    if ("best_learner" in name and best is None) or (".best" in name and label is None):
        return None
    out = name.replace("best_learner", best or "best_learner")
    head, dot, tail = out.partition(".best")
    return f"{head}.{label}{tail}" if dot else out


def training_curves(study: dict) -> dict:
    """Per learner and training seed: training-task success, invalid choices and shaping per logged
    window (every 10,000 interactions), from the training runs' events."""
    out = defaultdict(dict)
    for cfg in study["_members"].values():
        for m in completed(cfg["run"]["runs_dir"], cfg["run"]["name"]).values():
            if m["job_kind"] != "train":
                continue
            ev = [e for e in read_jsonl(Path(m["dir"]) / "events.jsonl") if e.get("type") == "training"]
            prev, series = 0, []
            for e in ev:
                span, prev = e["interactions"] - prev, e["interactions"]
                series.append({"interactions": e["interactions"], "tasks": e["tasks"],
                               "success_rate": e["success"] / e["tasks"] if e["tasks"] else None,
                               "invalid_share": e["invalid"] / span if span else None,
                               "shaping": e.get("shaping")})
            out[m["arm"]][str(m["seed"])] = series
    return dict(out)


def training_bins(curves: dict, budgets: list[int]) -> dict:
    """Training-task success rate per learner between consecutive checkpoint budgets (both seeds)."""
    edges = [0, *budgets]
    out = {}
    for arm, by_seed in curves.items():
        rates = {}
        for lo, hi in zip(edges, edges[1:]):
            win = [w for ws in by_seed.values() for w in ws if lo < w["interactions"] <= hi]
            tasks = sum(w["tasks"] for w in win)
            succ = sum((w["success_rate"] or 0) * w["tasks"] for w in win)
            shares = [w["invalid_share"] for w in win if w["invalid_share"] is not None]
            rates[f"{lo}-{hi}"] = {"success_rate": succ / tasks if tasks else None,
                                   "invalid_share": float(np.mean(shares)) if shares else None}
        out[arm] = rates
    return out


def _cell(tabs: dict, arm: str | None, budget: int, split: str) -> dict | None:
    """The cell a test reads for ``arm`` (as ``analysis.run_test`` looks it up)."""
    for key in ((arm, budget, split), (arm, None, split)):
        if arm is not None and key in tabs:
            return tabs[key]
    return None


def seed_contrasts(res: dict, tabs: dict) -> dict | None:
    """Per training seed of ``a``: the mean over the test's worlds of ``a`` (that seed) minus ``b`` (or
    ``a`` alone). Used when ``b`` has no training seeds (an agent) or there is no ``b``, the cases in
    which ``analysis.run_test`` reports no per-seed contrast."""
    if "by_training_seed" in res or not res.get("per_unit"):
        return None
    ca = _cell(tabs, res["a_resolved"], res["budget"], res["split"])
    cb = _cell(tabs, res["b_resolved"], res["budget"], res["split"]) if res.get("b") else None
    if ca is None or len(ca["seeds"]) < 2:
        return None
    metric, out = res["metric"], {}
    for s, units in sorted(ca["seeds"].items(), key=lambda kv: str(kv[0])):
        d = [units[u][metric] - (cb["units"][u][metric] if cb is not None else 0.0)
             for u in res["per_unit"] if u in units and metric in units[u]]
        out[str(s)] = float(np.mean(d)) if d else float("nan")
    return out


def analyse(study: dict) -> dict:
    rows, book, comp = collect_rows(study)
    return analyse_rows(study, rows, comp, book, training_curves(study))


def analyse_rows(study: dict, rows: list[dict], comp: dict, book: dict | None = None,
                 curves: dict | None = None) -> dict:
    """The study's selection, tests and curves from rows (agents already relabelled)."""
    a = study["analysis"]
    first = next(iter(study["_members"].values()))
    top = max(first["train"]["budgets"])
    tabs = derived_tables(xa.tables(rows, a["late_from"], len(first["train"]["seeds"])), study)
    complete = all(c["complete"] for c in comp.values())
    sel = select(tabs, study, top, complete)
    expected = sum(first["test"]["n_worlds"] for _ in first["test"]["seeds"])
    cfg_like = {"analysis": a}
    tests = []
    for t in a["tests"]:
        ra, rb = resolve(t["a"], sel), resolve(t.get("b"), sel)
        if ra is None or (t.get("b") and rb is None) or not complete:
            tests.append({"name": t["name"], "role": t.get("role", "secondary"), "metric": t["metric"],
                          "a": t["a"], "b": t.get("b"), "a_resolved": ra, "b_resolved": rb,
                          "expect": t.get("expect", "less"), "verdict": "incomplete",
                          "reason": "no best learner or incomplete members"})
            continue
        res = xa.run_test({**t, "a": ra, "b": rb} if t.get("b") else {**t, "a": ra}, tabs, None, cfg_like,
                          top, expected)
        per_seed = seed_contrasts(res, tabs)
        if per_seed is not None:  # X's run_test reports b's seeds only; here those of a
            res.update(by_training_seed=per_seed, by_training_seed_of="a")
        tests.append({**res, "a": t["a"], "b": t.get("b")})
    plain = {k: v for k, v in tabs.items() if "~" not in k[0] and "#" not in k[0]}
    curves = curves or {}
    return {"name": study["name"], "members": {k: v["run"]["name"] for k, v in study["_members"].items()},
            "family": first["setting"]["family"], "selection": sel, "tests": tests,
            "completeness": comp, "learning_curves": xa.learning_curves(plain),
            "validation_curves": xa.learning_curves(plain, split="validation"),
            "relative_curves": xa.learning_curves({k: v for k, v in tabs.items()
                                                   if "~" in k[0] and "#" not in k[0]},
                                                  metrics=("cost_ratio_late", "success_late")),
            "per_training_seed": {f"{arm}|{b}": {str(s): xa._mean(m, "cost_ratio_late")
                                                 for s, m in c["seeds"].items()}
                                  for (arm, b, sp), c in plain.items() if b is not None and sp == "test"},
            "training_curves": curves, "training_bins": training_bins(curves, first["train"]["budgets"]),
            "bookkeeping": book or {},
            "definition": {"late_from": a["late_from"], "unit": "test world",
                           "bootstrap": {"resamples": a["n_boot"], "seed": a["boot_seed"]}}}


def markdown(s: dict) -> str:
    sel = s["selection"]
    lines = [f"# Study XS: {s['name']}", "", "Members: " + ", ".join(f"`{k}` ({v})" for k, v in
                                                                    s["members"].items()) + ".", "",
             "## Selection (validation worlds only)", "",
             f"Best learner `{sel['best_learner']}` (member `{sel['best_member']}`) by {sel['metric']} at "
             f"budget {sel['budget']}: " + ", ".join(f"`{k}` {v:+.3f}" for k, v in sel["scores"].items())]
    if sel["note"]:
        lines.append(f"({sel['note']})")
    lines += ["", "## Tests", "",
              "| Test | Role | a - b | Estimate [CI] | Per training seed | Units | Verdict |",
              "|---|---|---|---|---|---|---|"]
    for t in s["tests"]:
        est = "-" if t.get("estimate") is None else \
            f"{t['estimate']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
        pair = f"{t['a_resolved']} - {t['b_resolved']}" if t.get("b") else str(t["a_resolved"])
        seeds = ", ".join(f"s{k} {v:+.3f}" for k, v in (t.get("by_training_seed") or {}).items()) or "-"
        if t.get("by_training_seed") and t.get("by_training_seed_of") != "a":
            seeds += " (seeds of b)"
        lines.append(f"| {t['name']} | {t['role']} | {pair} | {est} | {seeds} | {t.get('n_units', 0)} | "
                     f"{t['verdict']} |")
    lines += ["", "## Learning curves (test worlds)", "",
              "| Arm | Budget | Late cost ratio | Late success | Late steps | Units |",
              "|---|---|---|---|---|---|"]
    for arm, by in sorted(s["learning_curves"].items()):
        for b, m in sorted(by.items(), key=xa._budget_order):
            lines.append(f"| {arm} | {b} | {m['cost_ratio_late']:.3f} | {m['success_late']:.3f} | "
                         f"{m['steps_late']:.1f} | {m['units']} |")
    lines += ["", "## Training-task success between checkpoints (both training seeds)", "",
              "| Learner | Interactions | Success rate | Invalid choices per interaction |",
              "|---|---|---|---|"]
    for arm, bins in sorted(s["training_bins"].items()):
        for span, v in bins.items():
            rate = "-" if v["success_rate"] is None else f"{v['success_rate']:.3f}"
            inv = "-" if v["invalid_share"] is None else f"{v['invalid_share']:.3f}"
            lines.append(f"| {arm} | {span} | {rate} | {inv} |")
    return "\n".join(lines) + "\n"


def figure(s: dict, path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, by in sorted(s["learning_curves"].items()):
        if "agent" in by:
            axes[0].axhline(by["agent"]["cost_ratio_late"], linestyle="--", linewidth=1,
                            label=f"{arm} (agent)", color=f"C{len(axes[0].lines) % 10}")
        else:
            pts = sorted((int(b), m["cost_ratio_late"]) for b, m in by.items())
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=arm)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training interactions")
    axes[0].set_ylabel("late cost ratio (test worlds)")
    axes[0].legend(fontsize=6)
    for arm, by_seed in sorted(s["training_curves"].items()):
        for seed, ws in sorted(by_seed.items()):
            pts = [(w["interactions"], w["success_rate"]) for w in ws if w["success_rate"] is not None]
            if pts:
                axes[1].plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.8, label=f"{arm} s{seed}")
    axes[1].set_xlabel("training interactions")
    axes[1].set_ylabel("training-task success rate")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def budget_check(path: str | Path) -> dict:
    """Whether the ledger can still pay for every planned job without a completed run, for a study file
    (all its members) or a single config: per ledger kind, the sum of those jobs' caps (``need``) against
    what the ledger has left for the allocation (``left``; an existing ledger keeps its own caps, a
    missing one would be created with the declared ones). Reads only; ``ok`` is False when either kind
    falls short (study XS's driver then stops: further budget only through a dated amendment)."""
    if is_study(path):
        study = load_study(path)
        name, members = study["name"], list(study["_members"].values())
    else:
        cfg = load_config(path)
        name, members = cfg["run"]["name"], [cfg]
    if len({(c["run"]["ledger"], c["run"]["allocation"]) for c in members}) != 1:
        raise ValueError("the members of a study must share one ledger and one allocation")
    first = members[0]
    need = {"adaptive": 0, "reporting": 0, "training": 0, "jobs": 0, "training_jobs": 0}
    for c in members:
        for k, v in missing_caps(c).items():
            need[k] += v
    alloc, led = first["run"]["allocation"], first["ledger"]
    ledger = SessionLedger(first["run"]["ledger"], dict(led["allocations"]), led["adaptive_cap"],
                           led["reporting_cap"])
    if alloc not in ledger.state["allocations"]:
        raise KeyError(f"allocation {alloc!r} is not in {first['run']['ledger']}")
    left = {kind: ledger.remaining(alloc, kind) for kind in ("adaptive", "reporting")}
    return {"name": name, "ledger": first["run"]["ledger"], "allocation": alloc,
            "ledger_exists": Path(first["run"]["ledger"]).exists(), "need": need, "left": left,
            "ok": need["adaptive"] <= left["adaptive"] and need["reporting"] <= left["reporting"]}


def write_study_report(path: str | Path, out_dir: str | Path | None = None) -> dict:
    study = load_study(path)
    out = Path(out_dir or Path("artifacts") / "baselines" / study["name"])
    out.mkdir(parents=True, exist_ok=True)
    summary = analyse(study)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(markdown(summary))
    summary["figure"] = figure(summary, out / "learning_curves.png")
    return summary
