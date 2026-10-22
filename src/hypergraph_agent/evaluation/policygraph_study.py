"""Study PG (docs/studies/PG_policy_graph.md): RecipeQuest endpoints and the paired comparison with
study X's learners, from recorded runs (evaluator side; agents never import this module).

    python -m hypergraph_agent.evaluation.policygraph_study rq --runs runs --root <checkout>
    python -m hypergraph_agent.evaluation.policygraph_study x --runs runs --x-runs <X runs> --root <checkout>

``--root`` is the checkout whose ``configs/`` and ``artifacts/baselines/`` are read (the frozen
snapshot for the measurement; default the current directory); outputs go to ``--out``
(default ``artifacts/policygraph``). The TechTree tests of the study are computed by
``techtree.analysis`` from the configs of ``configs/policygraph_tt/``; this module adds what that
analysis cannot read.

Units: a world, labelled ``s<seed>:<world key>`` as in study X. Per unit and arm (``unit_metrics``,
the definitions of study X): late, early and overall cost ratios (actions over reference actions,
summed over the episodes; a failed episode counts every action it used), mean actions per episode,
success rates, and actions to the first success in the world (censored at the stream total).
Episodes ``>= late_from`` (RecipeQuest 8, TechTree 6) form the late half.

Tests: ``a - b`` per unit, paired over the units both arms have; the mean gets a percentile
world-cluster bootstrap interval (``evaluation.stats.cluster_bootstrap_mean``, 10,000 resamples,
seed 20261008, 95%) and an exact two-sided sign test. ``less`` holds when the upper end is below 0.
A test is "incomplete" (it never holds) when either arm lacks a complete run for some seed, or
when its paired units are fewer than the stream's worlds.

``rq``: the PG arms and the repository's agents on RecipeQuest (config ``pg_rq_main.yaml``; per arm
and seed the earliest completed walker run, all from one code identity; see ``walker_arm_units``).

``x``: ``pg`` (and, descriptively, ``pg_greedy``) against study X's best learner in each setting,
on the same test worlds. The best learner and its action mode are read from X's committed summary
(``artifacts/baselines/<X name>/summary.json``), its rows from X's evaluation runs at 1,000,000
training interactions (local run directories, ``--x-runs``; the earliest completed run per training
seed), and its value in a unit is the mean over its training seeds, as in X. Before any contrast
the module checks that (i) X's rows reproduce the committed per-unit values of X's own test
between one of X's agents and the best learner, and (ii) every task of X's rows has the same
reference length in PG's rows of the same unit and episode (the same tasks); a failed check aborts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import cluster_bootstrap_mean, sign_test

N_BOOT, BOOT_SEED = 10_000, 20261008
RQ_CONFIG, RQ_LATE, RQ_UNITS = "configs/policygraph_rq/pg_rq_main.yaml", 8, 50
RQ_TESTS = (
    ("PR1 pg - pg_uniform, late cost ratio, F1", "cost_ratio_late", "pg", "pg_uniform"),
    ("PR2 pg_greedy - pg_uniform, late cost ratio, F1", "cost_ratio_late", "pg_greedy", "pg_uniform"),
    ("PR3 focused_sample - pg, late cost ratio, F1", "cost_ratio_late", "focused_sample", "pg"),
    ("PR4 maximal - pg, late cost ratio, F1", "cost_ratio_late", "maximal", "pg"),
    ("PR5 focused_sample - pg, actions to the first success, F1", "first_success_steps", "focused_sample",
     "pg"),
)
# setting: (PG config, X run name, late_from, units, metric, the X test whose rows are checked)
X_SETTINGS = {
    "tt_u_k3": ("configs/policygraph_tt/pg_u_main.yaml", "x-ttu-main", 6, 30, "steps_late", "XU2"),
    "tt_l2_d2": ("configs/policygraph_tt/pg_l_main.yaml", "x-ttl-main", 6, 32, "steps_late", "XL1"),
    "rq_f1": (RQ_CONFIG, "x-rq-main", 8, 50, "cost_ratio_late", "X1 "),
}
X_LABELS = {"tt_u_k3": "XT-U", "tt_l2_d2": "XT-L", "rq_f1": "XT-R"}  # pg_greedy's tests are descriptive
X_BUDGET = 1_000_000
X_ARMS = ("pg", "pg_greedy")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def unit_metrics(rs: list[dict], late_from: int) -> dict:
    rs = sorted(rs, key=lambda r: r["world_episode"])
    m = {"episodes": len(rs)}
    for tag, sel in (("late", [r for r in rs if r["world_episode"] >= late_from]),
                     ("early", [r for r in rs if r["world_episode"] < late_from]), ("all", rs)):
        if sel:
            ref = sum(r["reference_length"] or 0 for r in sel)
            m[f"cost_ratio_{tag}"] = sum(r["primitive_length"] for r in sel) / ref if ref else float("nan")
            m[f"steps_{tag}"] = float(np.mean([r["primitive_length"] for r in sel]))
            m[f"success_{tag}"] = float(np.mean([r["success"] for r in sel]))
    steps, first = 0, None
    for r in rs:
        steps += r["primitive_length"]
        if r["success"] and first is None:
            first = steps
    m["first_success_steps"] = first if first is not None else steps
    m["first_success_censored"] = first is None
    return m


def units_of(rows: list[dict], late_from: int) -> dict:
    """{unit: metrics} from rows that carry ``_unit``."""
    by = defaultdict(list)
    for r in rows:
        by[r["_unit"]].append(r)
    return {u: unit_metrics(rs, late_from) for u, rs in by.items()}


def contrast(name: str, metric: str, a: dict, b: dict, expected: int, incomplete: list) -> dict:
    """``a - b`` per unit (expectation ``less``)."""
    units = sorted(set(a) & set(b))
    d = np.array([a[u][metric] - b[u][metric] for u in units], dtype=float)
    out = {"name": name, "metric": metric, "n_units": len(units), "expected_units": expected}
    if len(units):
        boot = cluster_bootstrap_mean(d, N_BOOT, BOOT_SEED)
        out.update({"estimate": boot["estimate"], "ci": boot["ci"], "sign_test": sign_test(d),
                    "mean_a": float(np.mean([a[u][metric] for u in units])),
                    "mean_b": float(np.mean([b[u][metric] for u in units])),
                    "per_unit": dict(zip(units, map(float, d)))})
    if incomplete or len(units) < expected:
        why = incomplete or [f"{len(units)} of {expected} units"]
        out.update({"verdict": "incomplete", "incomplete": why})
    else:
        out["verdict"] = "holds" if out["ci"][1] < 0 else "does not hold"
    return out


def problems_of(arms: dict, problems: dict, *names) -> list:
    out = []
    for x in names:
        out += problems.get(x, []) if x in arms else [f"no runs of {x}"]
    return out


# --------------------------------------------------------------------- PG runs
def walker_runs(runs_dir: Path, config: str) -> tuple[dict, list]:
    """(config, every walker run of a PG RecipeQuest config, complete or not)."""
    from ..config import load_config
    from .replication import load_runs
    cfg = load_config(config)
    name = cfg["run"]["name"]
    return cfg, [r for r in load_runs(runs_dir, f"{name}-") if r["run_id"].startswith(f"{name}-")]


def walker_arm_units(runs_dir: Path, config: str, late_from: int) -> tuple[dict, dict, dict]:
    """({arm: {unit: metrics}}, {arm: problems}, {unit: {episode: reference length}}) of the walker
    runs of a PG RecipeQuest config: per arm and seed the earliest completed run (runs are
    deterministic given the code, so a rerun of a missing pair with ``--seeds``/``--arms`` replays
    its stream); incomplete runs are not used (``rq_analysis`` lists them). Runs made by more than
    one code identity (commit and loaded code) make every arm a problem."""
    cfg, runs = walker_runs(runs_dir, config)
    chosen = {}
    for r in sorted(runs, key=lambda r: (r["start"], r["run_id"])):
        if r["complete"]:
            chosen.setdefault((r["strategy"], r["seed"]), r)
    rows, problems, refs = defaultdict(list), defaultdict(list), defaultdict(dict)
    for r in chosen.values():
        for row in r["rows"]:
            rows[r["strategy"]].append(row)
            refs[row["_unit"]][row["world_episode"]] = row["reference_length"]
    codes = {(r["commit"], r["code"]) for r in chosen.values()}
    for arm in [a["strategy"] for a in cfg["arms"]]:
        missing = [s for s in cfg["run"]["seeds"] if (arm, s) not in chosen]
        problems[arm] += [f"seed {s}: no completed run" for s in missing]
        if len(codes) > 1:
            problems[arm].append(f"runs from {len(codes)} code identities")
    return {arm: units_of(rs, late_from) for arm, rs in rows.items()}, dict(problems), dict(refs)


def techtree_arm_units(runs_dir: Path, config: str, late_from: int) -> tuple[dict, dict, dict]:
    """The same for a PG TechTree config (runs selected by ``techtree.analysis``)."""
    from ..techtree.analysis import collect_runs, completeness, select_runs
    from ..techtree.config import load_config
    cfg = load_config(config)
    chosen, _ = select_runs(collect_runs(Path(runs_dir), cfg["run"]["name"]))
    status = completeness(cfg, chosen)
    rows, refs = defaultdict(list), defaultdict(dict)
    for r in chosen:
        for row in r["rows"]:
            row = {**row, "_unit": f"s{r['seed']}:{row['world_key']}"}
            rows[r["arm"]].append(row)
            refs[row["_unit"]][row["world_episode"]] = row["reference_length"]
    problems = {arm: p for (_, arm), p in status.items()}
    return {arm: units_of(rs, late_from) for arm, rs in rows.items()}, problems, dict(refs)


def rq_analysis(runs_dir: Path, root: Path = Path("."), config: str = RQ_CONFIG) -> dict:
    """``root``: the checkout whose ``configs/`` (and, for ``x``, ``artifacts/``) are read."""
    config = str(Path(root) / config)
    arms, problems, _ = walker_arm_units(runs_dir, config, RQ_LATE)
    tests = [contrast(n, m, arms.get(a, {}), arms.get(b, {}), RQ_UNITS, problems_of(arms, problems, a, b))
             for n, m, a, b in RQ_TESTS]
    keys = ("cost_ratio_late", "cost_ratio_early", "success_late", "steps_late", "first_success_steps")
    table = {arm: {k: float(np.mean([m[k] for m in us.values()])) for k in keys}
             for arm, us in sorted(arms.items())}
    incomplete = sorted(r["run_id"] for r in walker_runs(runs_dir, config)[1] if not r["complete"])
    return {"config": config, "late_from": RQ_LATE, "unit": "test world", "tests": tests, "table": table,
            "problems": problems, "incomplete_runs": incomplete}


# ----------------------------------------------------------------------- study X
def x_rows(x_runs: Path, x_name: str, kind: str, arm: str, mode: str | None = None) -> dict:
    """{run seed: rows} of X's runs of one arm: the earliest completed run per training seed (learner
    evaluations at the largest budget on the test split) or per test seed (agents)."""
    pattern = (f"{x_name}-eval-{arm}-s*-b{X_BUDGET}-test-{mode[:3]}-*" if kind == "learner"
               else f"{x_name}-agent-{arm}-test-s*")
    out: dict = {}
    for d in sorted(Path(x_runs).glob(pattern), key=lambda p: p.name):
        man = json.loads((d / "manifest.json").read_text())
        if man.get("status") != "completed" or man["seed"] in out:
            continue
        out[man["seed"]] = [{**r, "_unit": r["unit"]} for r in read_jsonl(d / "eval.jsonl")]
    return out


def x_best_units(x_runs: Path, summary: dict, late_from: int) -> tuple[str, str, dict, list]:
    """(best learner, mode, {unit: metrics averaged over training seeds}, its rows)."""
    best = summary["selection"]["best_learner"]
    mode = next(g["mode"] for g in summary["tuning"]["groups"].values() if g["learner"] == best)
    by_seed = x_rows(x_runs, summary["name"], "learner", best, mode)
    if len(by_seed) < 2:
        raise RuntimeError(f"X's evaluation runs of {best} at {X_BUDGET} were not found under {x_runs}")
    per_seed = [units_of(rs, late_from) for rs in by_seed.values()]
    units = sorted(set.intersection(*(set(p) for p in per_seed)))
    avg = {u: {k: float(np.mean([p[u][k] for p in per_seed])) for k in per_seed[0][u]
               if k != "first_success_censored"} for u in units}
    return best, mode, avg, [r for rs in by_seed.values() for r in rs]


def check_x_rows(x_runs: Path, summary: dict, prefix: str, best_units: dict, late_from: int) -> dict:
    """X's rows must reproduce the committed per-unit values of X's test ``prefix``."""
    t = next(t for t in summary["tests"] if t["name"].startswith(prefix))
    agent = {u: m for rs in x_rows(x_runs, summary["name"], "agent", t["a_resolved"]).values()
             for u, m in units_of(rs, late_from).items()}
    worst = max(abs(agent[u][t["metric"]] - best_units[u][t["metric"]] - v) for u, v in t["per_unit"].items())
    if worst > 1e-9 or len(t["per_unit"]) != len(best_units):
        raise RuntimeError(f"X's rows do not reproduce the committed test {t['name']!r} "
                           f"(largest gap {worst})")
    return {"test": t["name"], "units": len(t["per_unit"]), "largest_gap": worst}


def check_same_tasks(rows: list, pg_refs: dict) -> int:
    """Every (unit, episode) of X's rows has the same reference length in PG's rows."""
    for r in rows:
        if pg_refs.get(r["_unit"], {}).get(r["world_episode"]) != r["reference_length"]:
            raise RuntimeError(f"task {r['_unit']} episode {r['world_episode']} differs between X and PG")
    return len(rows)


def x_analysis(runs_dir: Path, x_runs: Path, root: Path = Path(".")) -> dict:
    out = {"budget": X_BUDGET, "settings": {}}
    for setting, (config, x_name, late, expected, metric, prefix) in X_SETTINGS.items():
        config = str(Path(root) / config)
        summary = json.loads((Path(root) / "artifacts" / "baselines" / x_name / "summary.json")
                             .read_text(encoding="utf-8"))
        best, mode, best_units, rows = x_best_units(x_runs, summary, late)
        check = check_x_rows(x_runs, summary, prefix, best_units, late)
        read = walker_arm_units if "policygraph_rq" in config else techtree_arm_units
        arms, problems, refs = read(runs_dir, config, late)
        tasks = check_same_tasks(rows, refs) if refs else 0
        tests = []
        for arm in X_ARMS:
            label = X_LABELS[setting] + ("" if arm == "pg" else " (descriptive)")
            tests.append(contrast(f"{label} {arm} - X best learner ({best}), {metric}, {setting}", metric,
                                  arms.get(arm, {}), best_units, expected, problems_of(arms, problems, arm)))
        out["settings"][setting] = {"pg_config": config, "x_runs": x_name, "best_learner": best, "mode": mode,
                                    "row_check": check, "tasks_compared": tasks, "tests": tests}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.evaluation.policygraph_study")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rq", help="RecipeQuest endpoints of study PG")
    r.add_argument("--runs", default="runs")
    r.add_argument("--root", default=".", help="checkout whose configs/ are read (e.g. the snapshot)")
    r.add_argument("--out", default="artifacts/policygraph")
    x = sub.add_parser("x", help="pg against study X's best learners, paired by test world")
    x.add_argument("--runs", default="runs")
    x.add_argument("--x-runs", required=True, help="directory holding study X's run directories")
    x.add_argument("--root", default=".", help="checkout whose configs/ and artifacts/ are read")
    x.add_argument("--out", default="artifacts/policygraph")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "rq":
        res, stem = rq_analysis(Path(args.runs), Path(args.root)), "pg_rq_summary"
        tests = res["tests"]
    else:
        res, stem = x_analysis(Path(args.runs), Path(args.x_runs), Path(args.root)), "pg_x_summary"
        tests = [t for s in res["settings"].values() for t in s["tests"]]
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for t in tests:
        ci = "-" if not t.get("ci") else f"{t['estimate']:.3f} [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}]"
        print(f"{t['name']}: {ci} units={t['n_units']} -> {t['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
