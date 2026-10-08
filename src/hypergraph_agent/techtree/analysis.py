"""Statistics, tables, figures and verdicts for TechTree studies.

    python -m hypergraph_agent.techtree.analysis --config configs/layers/l1_main.yaml --runs runs

Runs. Every (variant, arm, seed) the config expects must have exactly one run.
A run is complete when it finished with every task of its stream (each world
with all its episodes) and was never stopped by the session budget. Two runs of
the same (variant, arm, seed) are an error, unless the earlier ones are
incomplete and share the later run's ``run_hash`` and code identity (the hash
of the package code the process loaded and the commit, from the manifest's
``source``): a deterministic rerun of the same code, which is then used and
listed. A rerun after a code change is refused (it needs a dated amendment). A
(variant, arm) cell with a missing or incomplete run is flagged; every test
that involves it, and a verdict that needs such a test, gets no verdict.

Units: worlds. Per world, arm and variant the episodes give the metrics below;
contrasts pair arms by world (every arm of a seed and variant plays the same
tasks), resample worlds (10,000 percentile-bootstrap resamples, fixed seed) and
add a two-sided exact sign test over worlds. Episodes ``>= late_from`` within a
world form the late half (default: half of ``episodes_per_world``).

Metrics per world:
  steps_late, steps_early, steps_all     mean actions per episode (failures count
                                         their full budget)
  cost_ratio_late/_early/_all            sum of actions / sum of reference lengths
  success_late/_early/_all               success rate
  first_success_steps                    actions in that world up to and including
                                         the first successful episode; censored at
                                         the stream total when none succeeded
  discovered_late                        distinct concepts ever held by the end of
                                         the stream, supplied concepts excluded
  discoveries_new, discoveries_in_replay new discoveries, and those made while
                                         replaying known concepts (incidental
                                         composition); explorer arms only

Table notes: ``discoveries_new`` of ``none`` counts the same concepts again in
every episode (it forgets them), so it is not comparable with the other arms;
``discovered_late`` is not meaningful for the reference and the oracle arms
(they replay supplied or privileged plans).

Test types (``analysis.tests``): ``paired`` (a - b in one variant; ``less`` /
``greater``: the CI excludes 0 on that side; ``not_less`` / ``not_greater``: it
does not lie entirely on the other side), ``across`` (one arm, variant_a -
variant_b, paired by world), ``equivalence`` (90% CI of a - b inside
+-margin, ``margin_abs`` or ``margin_rel`` x the mean of b), ``ratio`` (mean of a
over mean of b across common worlds; passes when the estimate lies in
``band``; the bootstrap CI is reported), ``interaction`` ((a - b) at variant_hi
minus (a - b) at variant_lo, paired by world when both variants hold the same
worlds), ``within_margin`` (a minus the better of ``others`` in every listed
variant, upper CI bound <= margin_rel x that arm's mean), ``all_success`` and
``at_least_reference`` (validity checks over every episode of an arm).

Verdicts (``analysis.verdict``): rule ``all`` passes iff every ``require`` test
passes. Rule ``precedence``: a failed validity test fails the study; if the
primary holds and the falsification (equivalence) does not, the result is a
pass when every ``conditions`` test passes and "confounded" otherwise; if both
hold the benefit is "negligible"; if only the falsification holds the study is
"falsified"; if neither, "inconclusive".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import merged, run_hash, variant_arms, variants

ORACLES = ("oracle", "oracle_library")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ----------------------------------------------------------------- runs
def collect_runs(runs_dir: Path, name: str) -> list[dict]:
    """Every TechTree run of study ``name`` under ``runs_dir``, with its rows and completeness."""
    runs = []
    for man in sorted(Path(runs_dir).glob("*/manifest.json")):
        m = json.loads(man.read_text())
        if not str(m.get("phase", "")).startswith("techtree"):
            continue
        cfg = json.loads((man.parent / "config.json").read_text())
        if cfg["run"]["name"] != name:
            continue
        variant, seed = cfg["variant"], m["seed"]
        _, s = merged(cfg, variant)
        rows = read_jsonl(man.parent / "eval.jsonl")
        per_world = Counter(r["world_key"] for r in rows)
        complete = (m.get("status") == "completed" and len(per_world) == s["n_worlds"]
                    and all(n == s["episodes_per_world"] for n in per_world.values())
                    and all(r["status"] != "budget_exhausted" for r in rows))
        src = m.get("source") or {}
        runs.append({"run_id": m["run_id"], "variant": variant["name"], "arm": cfg["arm"], "seed": seed,
                     "run_hash": run_hash(cfg, variant, cfg["arm"], seed), "start": m.get("start_time", ""),
                     "code": [src.get("loaded_code_sha256"), src.get("commit")],
                     "complete": complete, "rows": rows})
    return runs


def select_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """One run per (variant, arm, seed); deterministic reruns of incomplete runs replace them."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["variant"], r["arm"], r["seed"])].append(r)
    chosen, reruns = [], []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: (r["start"], r["run_id"]))
        *earlier, last = rs
        same = all(not e["complete"] and e["run_hash"] == last["run_hash"] and e["code"] == last["code"]
                   for e in earlier)
        if earlier and not same:
            raise ValueError(f"duplicate runs of {key}: {[r['run_id'] for r in rs]} (only a rerun with the "
                             f"same run hash and the same code may replace an incomplete run; anything else "
                             f"needs a dated amendment)")
        chosen.append(last)
        if earlier:
            reruns.append({"cell": list(key), "used": last["run_id"],
                           "replaced": [e["run_id"] for e in earlier]})
    return chosen, reruns


def completeness(cfg: dict, chosen: list[dict]) -> dict:
    """{(variant, arm): list of problems}; an empty list means the cell is complete."""
    by = {(r["variant"], r["arm"], r["seed"]): r for r in chosen}
    out = {}
    for v in variants(cfg):
        for arm in variant_arms(cfg, v):
            problems = []
            for seed in cfg["run"]["seeds"]:
                r = by.get((v["name"], arm, seed))
                if r is None:
                    problems.append(f"seed {seed}: missing")
                elif not r["complete"]:
                    problems.append(f"seed {seed}: incomplete ({r['run_id']})")
            out[(v["name"], arm)] = problems
    return out


# -------------------------------------------------------------- metrics
def _supplied(row: dict) -> set:
    return set(range(1, 100)) if row["arm"] in ORACLES else set(row.get("reveal_levels") or ())


def _discoveries(row: dict) -> list[tuple[int, int]]:
    """(step, concept) of firings of concepts the arm was not supplied with."""
    sup = _supplied(row)
    return [(f[0], f[1]) for f in row.get("fired", []) if len(f) < 3 or f[2] not in sup]


def world_metrics(rows: list[dict], late_from: int) -> dict:
    """{(variant, arm): {world_key: metrics}}."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["variant"], r["arm"])][r["world_key"]].append(r)
    out = {}
    for key, worlds in groups.items():
        out[key] = {}
        for wk, rs in worlds.items():
            rs = sorted(rs, key=lambda r: r["world_episode"])
            late = [r for r in rs if r["world_episode"] >= late_from]
            early = [r for r in rs if r["world_episode"] < late_from]
            m = {}
            for tag, sel in (("late", late), ("early", early), ("all", rs)):
                if sel:
                    ref = sum(r["reference_length"] for r in sel)
                    m[f"steps_{tag}"] = float(np.mean([r["primitive_length"] for r in sel]))
                    m[f"cost_ratio_{tag}"] = sum(r["primitive_length"] for r in sel) / ref
                    m[f"success_{tag}"] = float(np.mean([r["success"] for r in sel]))
            first = next((r for r in rs if r["success"]), None)
            total = sum(r["primitive_length"] for r in rs)
            m["first_success_steps"] = (first["world_steps_before"] + first["primitive_length"]
                                        if first else total)
            m["first_success_censored"] = first is None
            m["discovered_late"] = len({c for r in rs for _, c in _discoveries(r)})
            for name in ("discoveries_new", "discoveries_in_replay"):  # explorer arms only
                if all(name in r for r in rs):
                    m[name] = sum(r[name] for r in rs)
            m["episodes"] = len(rs)
            out[key][wk] = m
    return out


def bootstrap(d: np.ndarray, n_boot: int, seed: int, level: float) -> list[float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def sign_test(d: np.ndarray) -> float:
    pos, m = int((d > 0).sum()), int((d != 0).sum())
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(pos, m - pos) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def _values(metrics, metric, arm, variant) -> dict:
    return {w: m[metric] for w, m in metrics.get((variant, arm), {}).items() if metric in m}


def paired_values(metrics, metric, a, b, variant) -> tuple[list[str], np.ndarray]:
    ma, mb = _values(metrics, metric, a, variant), _values(metrics, metric, b, variant)
    worlds = sorted(set(ma) & set(mb))
    return worlds, np.array([ma[w] - mb[w] for w in worlds], dtype=float)


def _mean(metrics, metric, arm, variant) -> float:
    vals = list(_values(metrics, metric, arm, variant).values())
    return float(np.mean(vals)) if vals else float("nan")


def _summary(d, worlds, n_boot, seed, level) -> dict:
    if len(d) == 0:
        return {"n_worlds": 0, "estimate": None, "ci": None, "sign_p": None}
    return {"n_worlds": len(d), "estimate": float(d.mean()), "ci": bootstrap(d, n_boot, seed, level),
            "level": level, "sign_p": sign_test(d), "n_positive": int((d > 0).sum()),
            "n_negative": int((d < 0).sum()), "per_world": dict(zip(worlds, map(float, d)))}


def _expect(res: dict, expect: str) -> bool | None:
    if res["ci"] is None:
        return None
    lo, hi = res["ci"]
    return {"less": hi < 0, "greater": lo > 0, "not_less": hi >= 0, "not_greater": lo <= 0}[expect]


def cells_of(t: dict, cfg: dict) -> list[tuple[str, str]]:
    """The (variant, arm) cells a test reads."""
    kind = t["type"]
    if kind in ("paired", "equivalence", "ratio"):
        return [(t["variant"], t["a"]), (t["variant"], t["b"])]
    if kind == "across":
        return [(t["variant_a"], t["arm"]), (t["variant_b"], t["arm"])]
    if kind == "interaction":
        return [(v, x) for v in (t["variant_hi"], t["variant_lo"]) for x in (t["a"], t["b"])]
    if kind == "within_margin":
        return [(v, x) for v in t["variants"] for x in [t["a"], *t["others"]]]
    names = t.get("variants") or [v["name"] for v in variants(cfg) if t["arm"] in variant_arms(cfg, v)]
    return [(v, t["arm"]) for v in names]


def run_test(t: dict, metrics: dict, rows: list[dict], n_boot: int, seed: int) -> dict:
    kind, level = t["type"], t.get("level", 0.90 if t["type"] == "equivalence" else 0.95)
    out = {"name": t.get("name"), "type": kind, "role": t.get("role")}
    if kind == "paired":
        worlds, d = paired_values(metrics, t["metric"], t["a"], t["b"], t["variant"])
        out.update(_summary(d, worlds, n_boot, seed, level))
        out["pass"] = _expect(out, t["expect"])
    elif kind == "across":
        ma, mb = (_values(metrics, t["metric"], t["arm"], v) for v in (t["variant_a"], t["variant_b"]))
        worlds = sorted(set(ma) & set(mb))
        d = np.array([ma[w] - mb[w] for w in worlds], dtype=float)
        out.update(_summary(d, worlds, n_boot, seed, level))
        out["pass"] = _expect(out, t["expect"])
    elif kind == "equivalence":
        worlds, d = paired_values(metrics, t["metric"], t["a"], t["b"], t["variant"])
        out.update(_summary(d, worlds, n_boot, seed, level))
        margin = t["margin_abs"] if t.get("margin_abs") is not None \
            else t["margin_rel"] * abs(_mean(metrics, t["metric"], t["b"], t["variant"]))
        out["margin"] = margin
        out["equivalent"] = None if out["ci"] is None else (out["ci"][0] > -margin and out["ci"][1] < margin)
        out["pass"] = out["equivalent"]
    elif kind == "ratio":
        ma, mb = (_values(metrics, t["metric"], x, t["variant"]) for x in (t["a"], t["b"]))
        worlds = sorted(set(ma) & set(mb))
        if worlds:
            a = np.array([ma[w] for w in worlds], dtype=float)
            b = np.array([mb[w] for w in worlds], dtype=float)
            idx = np.random.default_rng(seed).integers(0, len(worlds), (n_boot, len(worlds)))
            boots = a[idx].mean(axis=1) / b[idx].mean(axis=1)
            q = (1 - level) / 2
            est = float(a.mean() / b.mean())
            out.update({"n_worlds": len(worlds), "estimate": est, "level": level, "band": list(t["band"]),
                        "ci": [float(np.quantile(boots, q)), float(np.quantile(boots, 1 - q))],
                        "pass": bool(t["band"][0] <= est <= t["band"][1])})
        else:
            out.update({"n_worlds": 0, "estimate": None, "ci": None, "pass": None})
    elif kind == "interaction":
        wh, dh = paired_values(metrics, t["metric"], t["a"], t["b"], t["variant_hi"])
        wl, dl = paired_values(metrics, t["metric"], t["a"], t["b"], t["variant_lo"])
        common = sorted(set(wh) & set(wl))
        if common and len(common) == len(wh) == len(wl):
            ih, il = dict(zip(wh, dh)), dict(zip(wl, dl))
            d = np.array([ih[w] - il[w] for w in common])
            out.update(_summary(d, common, n_boot, seed, level))
            out["paired_by_world"] = True
        elif len(dh) and len(dl):
            rng = np.random.default_rng(seed)
            bh = dh[rng.integers(0, len(dh), (n_boot, len(dh)))].mean(axis=1)
            bl = dl[rng.integers(0, len(dl), (n_boot, len(dl)))].mean(axis=1)
            q = (1 - level) / 2
            out.update({"n_worlds": [len(dh), len(dl)], "estimate": float(dh.mean() - dl.mean()),
                        "ci": [float(np.quantile(bh - bl, q)), float(np.quantile(bh - bl, 1 - q))],
                        "level": level, "paired_by_world": False})
        else:
            out.update({"n_worlds": 0, "estimate": None, "ci": None})
        out["pass"] = _expect(out, t["expect"])
    elif kind == "within_margin":
        per = {}
        for v in t["variants"]:
            best = min(t["others"], key=lambda arm: _mean(metrics, t["metric"], arm, v))
            worlds, d = paired_values(metrics, t["metric"], t["a"], best, v)
            res = _summary(d, worlds, n_boot, seed, level)
            margin = t["margin_rel"] * abs(_mean(metrics, t["metric"], best, v))
            res.update({"better_arm": best, "margin": margin,
                        "pass": None if res["ci"] is None else res["ci"][1] <= margin})
            res.pop("per_world", None)
            per[v] = res
        out["by_variant"] = per
        out["pass"] = all(r["pass"] for r in per.values()) if per else None
    elif kind in ("all_success", "at_least_reference"):
        sel = [r for r in rows
               if r["arm"] == t["arm"] and (not t.get("variants") or r["variant"] in t["variants"])]
        if kind == "all_success":
            bad = [r["task_key"] for r in sel if not r["success"]]
        else:
            bad = [r["task_key"] for r in sel if r["primitive_length"] < r["reference_length"]]
        out.update({"n_episodes": len(sel), "violations": len(bad), "examples": bad[:5],
                    "pass": bool(sel) and not bad})
    return out


def verdict(spec: dict | None, tests: dict) -> dict | None:
    if spec is None:
        return None

    def blocked(names):
        return [n for n in names if tests[n].get("incomplete")]

    if spec["rule"] == "all":
        need = spec["require"]
        if blocked(need):
            return {"verdict": "no verdict", "reason": "incomplete cells", "tests": blocked(need)}
        failed = [n for n in need if not tests[n]["pass"]]
        return {"verdict": "fail" if failed else "pass", "failed": failed}
    validity, conditions = spec.get("validity") or [], spec.get("conditions") or []
    core = [spec["primary"], spec["falsification"], *validity]
    if blocked(core):
        return {"verdict": "no verdict", "reason": "incomplete cells", "tests": blocked(core)}
    failed_validity = [n for n in validity if not tests[n]["pass"]]
    if failed_validity:
        return {"verdict": "fail (validity)", "failed": failed_validity}
    primary, equivalent = bool(tests[spec["primary"]]["pass"]), bool(tests[spec["falsification"]]["pass"])
    if primary and equivalent:
        return {"verdict": "negligible", "reason": "primary and equivalence both hold"}
    if primary:
        if blocked(conditions):
            return {"verdict": "no verdict", "reason": "incomplete cells", "tests": blocked(conditions)}
        failed = [n for n in conditions if not tests[n]["pass"]]
        return {"verdict": "confounded", "failed_conditions": failed} if failed else {"verdict": "pass"}
    if equivalent:
        return {"verdict": "falsified", "reason": "equivalence holds, primary does not"}
    return {"verdict": "inconclusive", "reason": "neither the primary nor the equivalence holds"}


def tables(metrics: dict) -> dict:
    out = defaultdict(dict)
    for (variant, arm), worlds in sorted(metrics.items()):
        ms = list(worlds.values())
        keys = sorted({k for m in ms for k in m if k not in ("first_success_censored", "episodes")})
        keys = [k for k in keys if all(k in m for m in ms)]
        row = {k: float(np.mean([m[k] for m in ms if k in m])) for k in keys}
        row["n_worlds"] = len(ms)
        row["censored_first_success"] = int(sum(m["first_success_censored"] for m in ms))
        notes = []
        if arm == "none":
            notes.append("discoveries_new counts rediscoveries in every episode: not comparable")
        if arm in ("reference", *ORACLES):
            notes.append("discovered_late not meaningful (privileged or supplied plans)")
        if notes:
            row["notes"] = notes
        out[variant][arm] = row
    return dict(out)


# one colour per arm in every panel and figure
ARM_COLORS = {"reference": "tab:purple", "oracle": "tab:pink", "oracle_library": "tab:pink",
              "random": "tab:gray", "none": "tab:olive", "pooled": "tab:blue", "isolated": "tab:orange",
              "nomem": "tab:cyan", "blind": "tab:brown", "remember": "tab:orange", "promote": "tab:blue",
              "sham_promote": "tab:red", "adaptive": "tab:green"}


def figures(rows: list[dict], out_dir: Path, name: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    by_var = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    for r in rows:
        s = by_var[r["variant"]][r["arm"]][r["world_episode"]]
        s[0] += r["primitive_length"]
        s[1] += r["reference_length"]
        s[2] += 1
    if not by_var:
        return made
    names = sorted(by_var)
    cols = min(4, len(names))
    nrow = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrow, cols, figsize=(4.0 * cols, 3.3 * nrow), squeeze=False)
    for ax, v in zip(axes.flat, names):
        for arm, eps in sorted(by_var[v].items()):
            xs = sorted(eps)
            ax.plot(xs, [eps[x][0] / max(eps[x][1], 1) for x in xs], marker="o", markersize=3,
                    color=ARM_COLORS.get(arm), label=f"{arm} (episodes={sum(eps[x][2] for x in xs)})")
        ax.set_yscale("log")
        ax.set_title(v, fontsize=8)
        ax.set_xlabel("episode index within a world")
        ax.set_ylabel("actions / reference actions")
        ax.legend(fontsize=5)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_cost_by_episode.png", dpi=120)
    plt.close(fig)
    made.append(f"{name}_cost_by_episode.png")

    # discovery curves: distinct concepts discovered (supplied ones excluded) against actions in a world
    curves = defaultdict(lambda: defaultdict(list))
    per_world = defaultdict(list)
    order = ("variant", "arm", "seed", "world_key", "world_episode")
    for r in sorted(rows, key=lambda r: tuple(r[k] for k in order)):
        per_world[(r["variant"], r["arm"], r["seed"], r["world_key"])].append(r)
    for (v, arm, _, _), rs in per_world.items():
        seen, pts = set(), []
        for r in rs:
            for step, c in _discoveries(r):
                if c not in seen:
                    seen.add(c)
                    pts.append((r["world_steps_before"] + step, len(seen)))
        curves[v][arm].append(pts)
    fig, axes = plt.subplots(nrow, cols, figsize=(4.0 * cols, 3.3 * nrow), squeeze=False)
    for ax, v in zip(axes.flat, names):
        for arm, worlds in sorted(curves[v].items()):
            horizon = max((p[0] for pts in worlds for p in pts), default=1)
            grid = np.linspace(0, horizon, 60)
            vals = [[max([n for s, n in pts if s <= g], default=0) for g in grid] for pts in worlds]
            ax.plot(grid, np.mean(vals, axis=0), color=ARM_COLORS.get(arm),
                    label=f"{arm} (worlds={len(worlds)})")
        ax.set_title(v, fontsize=8)
        ax.set_xlabel("cumulative actions in a world")
        ax.set_ylabel("distinct concepts discovered")
        ax.legend(fontsize=5)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_discovery_curves.png", dpi=120)
    plt.close(fig)
    made.append(f"{name}_discovery_curves.png")
    return made


def analyse(cfg: dict, runs: list[dict]) -> dict:
    a = cfg["analysis"]
    late_from = a["late_from"] or cfg["stream"]["episodes_per_world"] // 2
    chosen, reruns = select_runs(runs)
    status = completeness(cfg, chosen)
    rows = [row for r in chosen if (r["variant"], r["arm"]) in status for row in r["rows"]]
    metrics = world_metrics(rows, late_from)
    tests = {}
    for t in a["tests"]:
        res = run_test(t, metrics, rows, a["n_boot"], a["boot_seed"])
        bad = [f"{v}/{arm}" for v, arm in cells_of(t, cfg) if status.get((v, arm), ["not configured"])]
        if bad:
            res.update({"incomplete": bad, "pass": None})
            if "equivalent" in res:
                res["equivalent"] = None
        tests[t.get("name")] = res
    return {"name": cfg["run"]["name"], "study": cfg["run"]["study"], "late_from": late_from,
            "n_rows": len(rows), "incomplete_cells": {f"{v}/{arm}": p for (v, arm), p in status.items() if p},
            "reruns_used": reruns, "tables": tables(metrics), "tests": list(tests.values()),
            "verdict": verdict(a["verdict"], tests),
            "runtime": {"episodes": len(rows),
                        "episode_s_mean": float(np.mean([r["episode_s"] for r in rows])) if rows else None}}


def _fmt(t: dict) -> str:
    if t.get("incomplete"):
        return f"{t['name']}: NO VERDICT, incomplete cells {t['incomplete']}"
    if t["type"] in ("all_success", "at_least_reference"):
        return f"{t['name']}: {t['violations']} violations in {t['n_episodes']} episodes -> pass={t['pass']}"
    if t["type"] == "within_margin":
        parts = [f"{v}: vs {r['better_arm']} {r['estimate']:.3g} CI {r['ci']} margin {r['margin']:.3g}"
                 for v, r in t["by_variant"].items() if r["ci"] is not None]
        return f"{t['name']}: " + "; ".join(parts) + f" -> pass={t['pass']}"
    if t.get("ci") is None:
        return f"{t['name']}: no data"
    if t["type"] == "equivalence":
        tail = f"equivalent={t['equivalent']} (margin {t['margin']:.3g})"
    elif t["type"] == "ratio":
        tail = f"band {t['band']} -> pass={t['pass']}"
    else:
        tail = f"pass={t['pass']}"
    sp = f" sign p={t['sign_p']:.3g}" if t.get("sign_p") is not None else ""
    return (f"{t['name']}: {t['estimate']:.3g} CI [{t['ci'][0]:.3g}, {t['ci'][1]:.3g}] "
            f"worlds={t['n_worlds']}{sp} -> {tail}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.techtree.analysis")
    p.add_argument("--config", required=True)
    p.add_argument("--runs", default=None, help="runs directory (default: run.runs_dir)")
    p.add_argument("--out", default=None, help="output directory (default: artifacts/techtree/<name>)")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)
    runs = collect_runs(Path(args.runs or cfg["run"]["runs_dir"]), cfg["run"]["name"])
    if not runs:
        print("no runs found for", cfg["run"]["name"])
        return 1
    out_dir = Path(args.out or Path("artifacts") / "techtree" / cfg["run"]["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(cfg, runs)
    if not args.no_figures:
        rows = [row for r in select_runs(runs)[0] for row in r["rows"]]
        res["figures"] = figures(rows, out_dir, cfg["run"]["name"])
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    notes = set()
    for v, arms in res["tables"].items():
        print(f"== {v}")
        for arm, m in arms.items():
            mark = " *" if m.get("notes") else ""
            notes.update(f"{arm}: {n}" for n in m.get("notes", []))
            print(f"  {arm:15s} worlds={m['n_worlds']:3d} first_success={m['first_success_steps']:8.1f} "
                  f"steps late={m.get('steps_late', float('nan')):6.1f} "
                  f"success late={m.get('success_late', float('nan')):.2f} "
                  f"cost/ref late={m.get('cost_ratio_late', float('nan')):.2f}{mark}")
    for n in sorted(notes):
        print(f"  * {n}")
    for t in res["tests"]:
        print(_fmt(t))
    if res["incomplete_cells"]:
        print("incomplete cells:", json.dumps(res["incomplete_cells"]))
    if res["reruns_used"]:
        print("deterministic reruns used:", json.dumps(res["reruns_used"]))
    print("verdict:", json.dumps(res["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
