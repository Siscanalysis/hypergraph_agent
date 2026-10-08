"""Endpoints of study X (docs/studies/X_open_source_baselines.md) from recorded runs.

    python -m hypergraph_agent.baselines analyze --config configs/baselines/rq_f1_main.yaml

Runs: per job hash the earliest completed run (``runner.completed``);
incomplete runs and later repeats are listed and not used.

Units: a test world (``s<test seed>:<world key>``). Per unit and arm the rows
give (``world_metrics``; episodes ``>= analysis.late_from`` within the world form
the late half; failed episodes count every step they used):

  cost_ratio_late/_early/_all     steps / optimal steps, summed over the episodes
  cost_ratio_late_minus_early     within-world improvement (negative: cheaper late)
  steps_late/_early/_all          mean steps per episode
  success_late/_early/_all        success rate
  first_success_steps             steps in the world up to and including the first
                                  successful episode; censored at the stream total

Learner arms: a learner id (``id@mode`` when a learner is evaluated in more
than one action mode) at a training budget. Its value in a unit is the mean
over its training seeds (units recorded for every seed). The best learner
(``best_learner`` in tests) is the arm with the lowest mean of
``analysis.select_metric`` over the units of ``analysis.select_split`` at the
largest budget; the selection uses no test row unless the config selects on
the test split (development only).

Tests: ``a - b`` per unit (``a`` alone without ``b``), paired over the units both
arms have; the mean gets a percentile world-cluster bootstrap interval
(``evaluation.stats.cluster_bootstrap_mean``, ``analysis.n_boot`` resamples,
seed ``analysis.boot_seed``, 95% unless ``level``) and an exact two-sided sign
test. ``expect``: ``less`` holds when the upper end is below 0, ``greater`` when
the lower end is above 0, ``not_less`` when the upper end is at least 0,
``not_greater`` when the lower end is at most 0. A test whose units are fewer
than the test stream's worlds, or that reads an incomplete cell, is
"incomplete".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.stats import cluster_bootstrap_mean, sign_test
from . import benchmark_sets
from .runner import existing_runs, read_jsonl

COST_METRICS = ("cost_ratio_late", "cost_ratio_early", "cost_ratio_all", "cost_ratio_late_minus_early",
                "steps_late", "steps_early", "steps_all", "first_success_steps")


def _ratio(rs) -> float:
    ref = sum(r["reference_length"] or 0 for r in rs)
    return sum(r["primitive_length"] for r in rs) / ref if ref else float("nan")


def unit_metrics(rs: list[dict], late_from: int) -> dict:
    rs = sorted(rs, key=lambda r: r["world_episode"])
    late = [r for r in rs if r["world_episode"] >= late_from]
    early = [r for r in rs if r["world_episode"] < late_from]
    m = {"episodes": len(rs)}
    for tag, sel in (("late", late), ("early", early), ("all", rs)):
        if sel:
            m[f"cost_ratio_{tag}"] = _ratio(sel)
            m[f"steps_{tag}"] = float(np.mean([r["primitive_length"] for r in sel]))
            m[f"success_{tag}"] = float(np.mean([r["success"] for r in sel]))
    if late and early:
        m["cost_ratio_late_minus_early"] = m["cost_ratio_late"] - m["cost_ratio_early"]
    steps, first = 0, None
    for r in rs:
        steps += r["primitive_length"]
        if r["success"] and first is None:
            first = steps
    m["first_success_steps"] = first if first is not None else steps
    m["first_success_censored"] = first is None
    return m


def world_metrics(rows: list[dict], late_from: int) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["unit"]].append(r)
    return {u: unit_metrics(rs, late_from) for u, rs in by.items()}


def learner_arm(row: dict, modes: dict) -> str:
    return f"{row['arm']}@{row['mode']}" if len(modes.get(row["arm"], ())) > 1 else row["arm"]


def tables(rows: list[dict], late_from: int) -> dict:
    """{(arm, budget, split): {"units": {unit: metrics}, "seeds": {seed: {unit: metrics}}}};
    agents have budget None. A learner's unit metrics are means over its training seeds."""
    modes = defaultdict(set)
    for r in rows:
        if r["kind"] == "learner":
            modes[r["arm"]].add(r["mode"])
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        arm = learner_arm(r, modes) if r["kind"] == "learner" else r["arm"]
        groups[(arm, r.get("budget"), r["split"])][r.get("train_seed")].append(r)
    out = {}
    for key, by_seed in groups.items():
        seeds = {s: world_metrics(rs, late_from) for s, rs in by_seed.items()}
        common = set.intersection(*(set(v) for v in seeds.values()))
        units = {}
        for u in sorted(common):
            names = set.intersection(*(set(seeds[s][u]) for s in seeds))
            units[u] = {k: float(np.mean([seeds[s][u][k] for s in seeds])) for k in names
                        if k not in ("episodes", "first_success_censored")}
            units[u]["first_success_censored"] = float(np.mean([seeds[s][u]["first_success_censored"]
                                                                for s in seeds]))
        out[key] = {"units": units, "seeds": seeds}
    return out


def _mean(units: dict, metric: str) -> float:
    vals = [m[metric] for m in units.values() if metric in m]
    return float(np.mean(vals)) if vals else float("nan")


def select_best(tabs: dict, cfg: dict, top: int) -> dict:
    a = cfg["analysis"]
    metric, split = a["select_metric"], a.get("select_split", "validation")
    configured = split
    learners = sorted({arm for (arm, b, sp) in tabs if b is not None})
    if split == "validation" and not any((arm, top, split) in tabs for arm in learners):
        split = "test"  # no validation rows (development configs): select on the test split, flagged
    scores = {arm: _mean(tabs[(arm, top, split)]["units"], metric)
              for arm in learners if (arm, top, split) in tabs}
    sign = -1.0 if metric.startswith("success") else 1.0
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    order = {ln["id"]: i for i, ln in enumerate(cfg.get("learners", []))}

    def rank(k):  # ties: the earlier learner in the config, then the deterministic mode
        arm, _, mode = k.partition("@")
        return sign * valid[k], order.get(arm, 10 ** 6), mode == "stochastic", k

    best = min(valid, key=rank) if valid else None
    return {"metric": metric, "split": split, "budget": top, "scores": scores, "best_learner": best,
            "note": "selection split differs from the configured one" if split != configured else None}


def select_by_group(tabs: dict, rows: list[dict], cfg: dict, top: int) -> dict:
    """Tuning selection: per learner group, the (candidate, mode) with the best mean selection metric
    on the selection split at the largest budget; ties go to the earlier candidate in the config's
    learner list, then to the deterministic mode."""
    a = cfg["analysis"]
    metric, split = a["select_metric"], a.get("select_split", "validation")
    order = {ln["id"]: i for i, ln in enumerate(cfg.get("learners", []))}
    sign = -1.0 if metric.startswith("success") else 1.0
    cands = {}
    for r in rows:
        if r["kind"] == "learner":
            cands[(r["arm"], r["mode"])] = r["group"]
    modes = defaultdict(set)
    for arm, mode in cands:
        modes[arm].add(mode)
    out = defaultdict(dict)
    for (arm, mode), group in cands.items():
        key = (f"{arm}@{mode}" if len(modes[arm]) > 1 else arm, top, split)
        if key in tabs:
            out[group][(arm, mode)] = _mean(tabs[key]["units"], metric)
    chosen = {}
    for group, scores in out.items():
        valid = {k: v for k, v in scores.items() if np.isfinite(v)}

        def rank(k, valid=valid):
            return sign * valid[k], order.get(k[0], 10 ** 6), k[1] != "deterministic"

        if valid:
            arm, mode = min(valid, key=rank)
            chosen[group] = {"learner": arm, "mode": mode, "score": valid[(arm, mode)],
                             "candidates": {f"{k[0]}@{k[1]}": v for k, v in sorted(scores.items())}}
    return {"metric": metric, "split": split, "budget": top, "groups": chosen}


def _verdict(ci, expect: str) -> str:
    lo, hi = ci
    ok = {"less": hi < 0, "greater": lo > 0, "not_less": hi >= 0, "not_greater": lo <= 0}[expect]
    return "holds" if ok else "does not hold"


def run_test(t: dict, tabs: dict, best: str | None, cfg: dict, top: int, expected: int | None) -> dict:
    a = cfg["analysis"]
    budget, split = t.get("budget", top), t.get("split", "test")

    def resolve(x):
        return best if x == "best_learner" else x

    def cell(arm):
        for key in ((arm, budget, split), (arm, None, split)):
            if key in tabs:
                return key
        return None

    ra, rb = resolve(t["a"]), resolve(t.get("b"))
    out = {"name": t["name"], "role": t.get("role", "secondary"), "metric": t["metric"],
           "a": t["a"], "b": t.get("b"), "a_resolved": ra, "b_resolved": rb, "budget": budget, "split": split,
           "expect": t.get("expect", "less")}
    ka, kb = cell(ra) if ra else None, (cell(rb) if rb else None)
    if ka is None or (t.get("b") and kb is None):
        return {**out, "verdict": "incomplete", "reason": "missing arm"}
    metric = t["metric"]
    ua = tabs[ka]["units"]
    ub = tabs[kb]["units"] if kb else None
    units = sorted(u for u in ua if metric in ua[u] and (ub is None or (u in ub and metric in ub[u])))
    d = np.array([ua[u][metric] - (ub[u][metric] if ub is not None else 0.0) for u in units])
    boot = cluster_bootstrap_mean(d, n_boot=a["n_boot"], seed=a["boot_seed"], level=t.get("level", 0.95))
    out.update({"n_units": len(units), "estimate": boot["estimate"], "ci": boot["ci"],
                "level": boot.get("level"), "sign_test": sign_test(d) if len(d) else None,
                "mean_a": _mean({u: ua[u] for u in units}, metric),
                "mean_b": _mean({u: ub[u] for u in units}, metric) if ub is not None else None,
                "per_unit": dict(zip(units, map(float, d)))})
    if kb is not None and len(tabs[kb]["seeds"]) > 1:  # the contrast against each training seed of b
        out["by_training_seed"] = {
            str(s): float(np.mean([ua[u][metric] - m[u][metric] for u in units if u in m and metric in m[u]]))
            for s, m in tabs[kb]["seeds"].items()}
    if not units or boot["ci"] is None or (expected is not None and len(units) < expected):
        out["verdict"] = "incomplete"
    else:
        out["verdict"] = _verdict(boot["ci"], out["expect"])
    return out


CURVE_METRICS = ("cost_ratio_late", "success_late", "steps_late", "first_success_steps")


def learning_curves(tabs: dict, metrics=CURVE_METRICS, split: str = "test") -> dict:
    out = defaultdict(dict)
    for (arm, budget, sp), cell in sorted(tabs.items(), key=lambda kv: str(kv[0])):
        if sp != split:
            continue
        key = str(budget) if budget is not None else "agent"
        out[arm][key] = {m: _mean(cell["units"], m) for m in metrics}
        out[arm][key]["units"] = len(cell["units"])
    return dict(out)


def analyse_rows(cfg: dict, rows: list[dict], expected: int | None = None,
                 selection: dict | None = None) -> dict:
    """``selection``: a best-learner selection made elsewhere (an evaluation-only config uses that of the
    config whose checkpoints it evaluates)."""
    late_from = cfg["analysis"]["late_from"]
    top = max(cfg["train"]["budgets"])
    tabs = tables(rows, late_from)
    sel = selection or select_best(tabs, cfg, top)
    tests = [run_test(t, tabs, sel["best_learner"], cfg, top, expected) for t in cfg["analysis"]["tests"]]
    return {"selection": sel, "tuning": select_by_group(tabs, rows, cfg, top), "tests": tests,
            "learning_curves": learning_curves(tabs),
            "validation_curves": learning_curves(tabs, split="validation"),
            "cells": {f"{arm}|{b}|{sp}": {"units": len(c["units"]), "training_seeds": sorted(
                str(s) for s in c["seeds"])} for (arm, b, sp), c in tabs.items()}}


def collect(cfg: dict) -> tuple[list[dict], dict]:
    """Rows of the chosen runs and the run bookkeeping (chosen, repeats, incomplete, costs)."""
    runs = existing_runs(cfg["run"]["runs_dir"], cfg["run"]["name"])
    chosen, ignored, incomplete = {}, [], []
    for m in runs:
        if not m["complete"]:
            incomplete.append(m["run_id"])
        elif m["job_hash"] in chosen:
            ignored.append(m["run_id"])
        else:
            chosen[m["job_hash"]] = m
    rows = []
    for m in chosen.values():
        if m["job_kind"] != "train":
            rows += read_jsonl(Path(m["dir"]) / "eval.jsonl")
    costs = {"training": [], "evaluation": defaultdict(lambda: {"interactions": 0, "seconds": 0.0}),
             "agents": defaultdict(lambda: {"interactions": 0, "seconds": 0.0})}
    for m in chosen.values():
        if m["job_kind"] == "train":
            costs["training"].append({"learner": m["arm"], "seed": m["seed"],
                                      "interactions": m["interactions"]["physical"],
                                      "seconds": m.get("training_seconds"), "steps_per_second":
                                      m.get("steps_per_second"), "checkpoints": m.get("checkpoints")})
        else:
            key = "evaluation" if m["job_kind"] == "eval" else "agents"
            c = costs[key][m["arm"]]
            c["interactions"] += m["interactions"]["physical"]
            c["seconds"] += m.get("loop_seconds") or 0.0
    costs = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in costs.items()}
    return rows, {"runs_used": len(chosen), "ignored_repeats": ignored, "incomplete_runs": incomplete,
                  "costs": costs}


def agent_interactions_per_world(tabs_rows: list[dict]) -> dict:
    """Mean primitive steps a repository agent spends in one test world over its whole stream."""
    by = defaultdict(lambda: defaultdict(int))
    for r in tabs_rows:
        if r["kind"] == "agent" and r["split"] == "test":
            by[r["arm"]][r["unit"]] += r["primitive_length"]
    return {arm: float(np.mean(list(u.values()))) for arm, u in by.items()}


def _budget_order(kv) -> tuple:
    return kv[0] == "agent", int(kv[0]) if kv[0].isdigit() else 0


def markdown(summary: dict) -> str:
    book, sel = summary["bookkeeping"], summary["selection"]
    lines = [f"# Study X: {summary['name']}", "", f"Setting `{summary['setting']}`; runs used "
             f"{book['runs_used']}, incomplete {len(book['incomplete_runs'])}.", "", "## Selection", ""]
    lines.append(f"Best learner `{sel['best_learner']}` by {sel['metric']} on the {sel['split']} split at "
                 f"budget {sel['budget']}: "
                 + ", ".join(f"`{k}` {v:.3f}" for k, v in sorted(sel["scores"].items())))
    tun = summary.get("tuning", {}).get("groups", {})
    if any(len(c["candidates"]) > 1 for c in tun.values()):  # a tuning config
        lines += ["", "## Tuning selection (per learner group)", "", "| Group | Selected | Mode | Score | "
                  "Candidates |", "|---|---|---|---|---|"]
        for g, c in sorted(tun.items()):
            cands = ", ".join(f"{k} {v:.3f}" for k, v in c["candidates"].items())
            lines.append(f"| {g} | `{c['learner']}` | {c['mode']} | {c['score']:.3f} | {cands} |")
    lines += ["", "## Tests", "", "| Test | Role | a - b | Estimate [CI] | Units | Verdict |",
              "|---|---|---|---|---|---|"]
    for t in summary["tests"]:
        est = "-" if t.get("estimate") is None else \
            f"{t['estimate']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
        pair = f"{t['a_resolved']} - {t['b_resolved']}" if t.get("b") else str(t["a_resolved"])
        lines.append(f"| {t['name']} | {t['role']} | {pair} | {est} | {t.get('n_units', 0)} | "
                     f"{t['verdict']} |")
    lines += ["", "## Learning curves (test worlds; agents as reference)", "",
              "| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |",
              "|---|---|---|---|---|---|---|"]
    for arm, by in sorted(summary["learning_curves"].items()):
        for b, m in sorted(by.items(), key=_budget_order):
            lines.append(f"| {arm} | {b} | {m['cost_ratio_late']:.3f} | {m['success_late']:.3f} | "
                         f"{m['steps_late']:.1f} | {m['first_success_steps']:.1f} | {m['units']} |")
    conv = summary.get("conversion")
    if conv:
        s = conv["summary"]
        lines += ["", "## Conversion record (benchmark worlds)", "",
                  f"{s['worlds']} worlds; items per world {s['items_mean']:.1f}; promoted elements per "
                  f"world {s['promoted_mean']:.1f}; dropped recipes per world {s['dropped_mean']:.1f}; "
                  f"share of the closure kept as items {s['kept_share_mean']:.2f}.", "",
                  "| World | Goals | Items | Promoted | Dropped recipes | Kept share |",
                  "|---|---|---|---|---|---|"]
        for w in conv["worlds"]:
            goals = ", ".join(w["goals"]) if len(w["goals"]) <= 3 else f"{len(w['goals'])} sink goals"
            promoted = ", ".join(w["promoted"]) or "-"
            lines.append(f"| {w['world_key']} | {goals} | {w['items']} | {promoted} | {len(w['dropped'])} | "
                         f"{w['kept_share']:.2f} |")
    lines += ["", "## Interactions and time", ""]
    for t in summary["bookkeeping"]["costs"]["training"]:
        lines.append(f"- training `{t['learner']}` seed {t['seed']}: {t['interactions']:,} interactions, "
                     f"{t['seconds']} s ({t['steps_per_second']} steps/s)")
    for arm, steps in sorted(summary["agent_interactions_per_world"].items()):
        lines.append(f"- agent `{arm}`: {steps:.0f} interactions per test world over its whole stream")
    return "\n".join(lines) + "\n"


def figure(summary: dict, path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    metric = "cost_ratio_late" if summary["family"] == "recipequest" else "steps_late"
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, by in sorted(summary["learning_curves"].items()):
        if "agent" in by:
            ax.axhline(by["agent"][metric], linestyle="--", linewidth=1, label=f"{arm} (agent)",
                       color=f"C{len(ax.lines) % 10}")
        else:
            pts = sorted((int(b), m[metric]) for b, m in by.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("training interactions")
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def source_selection(cfg: dict) -> dict | None:
    """The best learner of the config named by ``run.checkpoints_from`` (from its own runs)."""
    if not cfg["run"]["checkpoints_from"]:
        return None
    from .config import load_config
    src = load_config(cfg["run"]["checkpoints_from"])
    rows, _ = collect(src)
    top = max(src["train"]["budgets"])
    return {**select_best(tables(rows, src["analysis"]["late_from"]), src, top), "source": src["run"]["name"]}


def write_report(cfg: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, book = collect(cfg)
    expected = sum(cfg["test"]["n_worlds"] for _ in cfg["test"]["seeds"])
    a = cfg["analysis"]
    summary = {"name": cfg["run"]["name"], "setting": cfg["setting"]["name"],
               "family": cfg["setting"]["family"],
               "definition": {"late_from": a["late_from"], "unit": "test world",
                              "bootstrap": {"resamples": a["n_boot"], "seed": a["boot_seed"]}},
               **analyse_rows(cfg, rows, expected, source_selection(cfg)), "bookkeeping": book,
               "agent_interactions_per_world": agent_interactions_per_world(rows)}
    if cfg["setting"].get("loader") == benchmark_sets.LOADER:
        recs = benchmark_sets.records(cfg["setting"])
        summary["conversion"] = {"summary": benchmark_sets.summary(recs), "worlds": recs}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(markdown(summary))
    summary["figure"] = figure(summary, out / "learning_curves.png")
    return summary
