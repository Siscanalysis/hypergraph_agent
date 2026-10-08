"""Endpoints of study R (docs/studies/R_replication.md) from recorded runs.

A cell is a task family (the run name after its first hyphen, without
``-bounds``: ``rep-f1-bounds`` is F1) crossed with the evaluated failure
probability. The unit of analysis is a world: with ``eval_world_offset_by_seed``
every seed plays its own worlds, so worlds are independent and a seed only
labels the walker's random stream. In a family whose tasks each get a fresh
world (``world_mode: per_task``, the no-recurrence control F0) the unit is a
block of ``episodes_per_world`` consecutive tasks of a run and the position in
the block replaces the episode index.

Per unit and arm, the late (early) cost ratio is the sum of primitive steps
over the sum of optimal steps in episodes 9-16 (1-8); a failed episode counts
with every step it used, its whole budget. For a contrast ``a - b`` and each
unit paired across the two arms (all 16 episodes recorded in both):

* the late difference ``a_late - b_late``;
* the difference-in-differences ``(a_late - b_late) - (a_early - b_early)``;
* the late success difference ``success_late(a) - success_late(b)``.

Each mean gets a world-cluster percentile bootstrap interval (the units are
resampled, the same resamples for every statistic); the late difference also
gets an exact sign test. A statistic passes when the upper end of its interval
is below 0 and every run it needs completed its stream; with an incomplete run
or too few units it is "incomplete" and never passes. The cell verdict is the
late difference of ``focused_sample - maximal``; in the control family it is
descriptive. The amortization explanation is supported at a noise level when
the difference-in-differences passes in F1 and its interval in F0 does not lie
below 0.

The break-even episode, descriptive only, is the first episode index e such
that the pooled per-episode cost ratio of ``focused_sample`` is at or below
that of ``maximal`` for every episode from e to 16, and "never" if there is no
such e; "never" is later than any episode.

One run counts per cell, arm and seed: the earliest. A run that did not
complete may be replaced by its earliest completed rerun with the same config
hash, code, seed, arm and variant (runs are deterministic); a completed run is
never replaced. Replaced and ignored runs are listed. Rows are read as
recorded; nothing is interpolated.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import cluster_bootstrap_mean, sign_test

ARM, BOUND, CONTROL, REFERENCE = "focused_sample", "maximal", "random_omit", "reference"
CONTRASTS = ((ARM, BOUND), (ARM, CONTROL))
LATE_FROM, EPISODES = 8, 16
PRIMARY = ("F1", 0.0)
REPLICATION = ("F1", "F2")  # the replication is general when all their cells pass
NULL_CONTROL = "F0"  # no recurrence: evidence is never reused
N_BOOT, BOOT_SEED = 10_000, 20261008
RERUN_MATCH = ("config_hash", "commit", "code", "seed", "arm_id", "variant")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _ratio(rows) -> float:
    opt = sum(r["reference_length"] or 0 for r in rows)
    return sum(r["primitive_length"] for r in rows) / opt if opt else float("nan")


def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def load_runs(runs_dir: str | Path, prefix: str) -> list[dict]:
    """Every run directory whose name starts with ``prefix``, with its cell,
    arm, seed, provenance, completeness and rows tagged with unit and position."""
    out = []
    for d in sorted(Path(runs_dir).glob(f"{prefix}*")):
        if not (d / "manifest.json").exists() or not (d / "config.json").exists():
            continue
        man = json.loads((d / "manifest.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        if "arm" not in cfg or cfg["arm"].get("strategy") is None:
            continue
        family = cfg["run"]["name"].split("-", 1)[-1].removesuffix("-bounds").upper()  # rep-f1-bounds: F1
        eps = float(cfg["eval"]["tasks"].get("failure_prob", cfg["train"]["tasks"]["failure_prob"]))
        blocks = (cfg["eval"]["world_mode"] or cfg["train"]["world_mode"]) == "per_task"
        size = cfg["eval"]["episodes_per_world"]
        rows = _read_jsonl(d / "eval.jsonl")
        for r in rows:
            r["_unit"] = f"s{man['seed']}:b{r['order'] // size}" if blocks else f"s{man['seed']}:{r['world_key']}"
            r["_pos"] = r["order"] % size if blocks else r["world_episode"]
        src = man.get("source") or {}
        out.append({"run_id": man["run_id"], "family": family, "eps": eps, "strategy": cfg["arm"]["strategy"],
                    "arm_id": cfg["arm"].get("id"), "seed": man["seed"], "variant": cfg["variant"]["name"],
                    "start": man.get("start_time", ""), "config_hash": man.get("config_hash"),
                    "commit": src.get("commit"), "code": src.get("loaded_code_sha256"),
                    "rows": rows, "n_expected": cfg["eval"]["n_tasks"], "units": "blocks" if blocks else "worlds",
                    "complete": man.get("status") == "completed" and man.get("stop_reason") == "tasks_done"
                    and len(rows) == cfg["eval"]["n_tasks"]})
    return out


def select_runs(runs: list[dict]) -> tuple[list[dict], dict]:
    """One run per cell, arm and seed (see the module docstring)."""
    groups = defaultdict(list)
    for r in sorted(runs, key=lambda x: (x["start"], x["run_id"])):
        groups[(r["family"], r["eps"], r["strategy"], r["seed"])].append(r)
    chosen, reruns, ignored = [], [], []
    for g in groups.values():
        pick = g[0]
        if not pick["complete"]:
            same = [x for x in g[1:] if x["complete"] and all(x[k] == pick[k] for k in RERUN_MATCH)]
            if same:
                reruns.append({"incomplete": pick["run_id"], "rerun": same[0]["run_id"]})
                pick = same[0]
        chosen.append(pick)
        replaced = g[0] if pick is not g[0] else None
        ignored += [x["run_id"] for x in g if x is not pick and x is not replaced]
    return chosen, {"reruns_used": reruns, "ignored": ignored}


def unit_table(rows: list[dict]) -> dict:
    """Per unit: early, late and overall cost ratios, late success and episode counts."""
    by = defaultdict(list)
    for r in rows:
        by[r["_unit"]].append(r)
    out = {}
    for u, rs in by.items():
        late = [r for r in rs if r["_pos"] >= LATE_FROM]
        early = [r for r in rs if r["_pos"] < LATE_FROM]
        out[u] = {"late": _ratio(late), "early": _ratio(early), "all": _ratio(rs), "n_late": len(late),
                  "n_early": len(early), "success_late": _mean([r["success"] for r in late])}
    return out


def arm_summary(runs: list[dict]) -> dict:
    rows = [r for run in runs for r in run["rows"]]
    late = [r for r in rows if r["_pos"] >= LATE_FROM]
    by_pos = defaultdict(list)
    for r in rows:
        by_pos[r["_pos"]].append(r)
    return {
        "runs": len(runs), "incomplete_runs": [x["run_id"] for x in runs if not x["complete"]],
        "episodes": len(rows), "units": len({r["_unit"] for r in rows}),
        "cost_ratio_early": _ratio([r for r in rows if r["_pos"] < LATE_FROM]),
        "cost_ratio_late": _ratio(late), "cost_ratio_all": _ratio(rows),
        "success_all": _mean([r["success"] for r in rows]), "success_late": _mean([r["success"] for r in late]),
        "omitted_fraction_late": (sum(r.get("omitted_candidates", 0) for r in late)
                                  / max(1, sum(r.get("pool_candidates", 0) for r in late))),
        "cost_ratio_late_by_seed": {str(x["seed"]): _ratio([r for r in x["rows"] if r["_pos"] >= LATE_FROM])
                                    for x in runs},
        "by_episode": {str(k + 1): {"n": len(v), "cost_ratio": _ratio(v), "success": _mean([r["success"] for r in v])}
                       for k, v in sorted(by_pos.items())},
    }


def break_even(arms: dict) -> int | str | None:
    """First episode index (1-based) from which the pooled cost ratio of
    ``focused_sample`` stays at or below that of ``maximal`` through the last
    episode; "never" if there is none; None without both arms."""
    if ARM not in arms or BOUND not in arms:
        return None
    a, b = arms[ARM]["by_episode"], arms[BOUND]["by_episode"]
    first = "never"
    for e in sorted(int(x) for x in set(a) & set(b)):
        if a[str(e)]["cost_ratio"] <= b[str(e)]["cost_ratio"]:
            first = e if first == "never" else first
        else:
            first = "never"
    return first


def episode_rank(e) -> int:
    """Order of break-even episodes: "never" is later than any episode."""
    return EPISODES + 1 if e in ("never", None) else int(e)


def _statistic(values, complete: bool, n_boot: int, boot_seed: int) -> dict:
    boot = cluster_bootstrap_mean(values, n_boot=n_boot, seed=boot_seed)
    verdict = "incomplete" if not complete else ("pass" if boot["ci"][1] < 0 else "fail")
    return {**boot, "verdict": verdict}


def contrast(runs_a: list[dict], runs_b: list[dict], expected_units: int | None, n_boot: int,
             boot_seed: int) -> dict:
    ta = {u: v for x in runs_a for u, v in unit_table(x["rows"]).items()}
    tb = {u: v for x in runs_b for u, v in unit_table(x["rows"]).items()}
    half = EPISODES - LATE_FROM
    paired = sorted(u for u in set(ta) & set(tb) if ta[u]["n_late"] == tb[u]["n_late"] == half
                    and ta[u]["n_early"] == tb[u]["n_early"] == LATE_FROM)
    late = [ta[u]["late"] - tb[u]["late"] for u in paired]
    did = [(ta[u]["late"] - tb[u]["late"]) - (ta[u]["early"] - tb[u]["early"]) for u in paired]
    success = [ta[u]["success_late"] - tb[u]["success_late"] for u in paired]
    incomplete = [x["run_id"] for x in runs_a + runs_b if not x["complete"]]
    complete = bool(paired) and not incomplete and (expected_units is None or len(paired) >= expected_units)
    return {**_statistic(late, complete, n_boot, boot_seed), "n_units": len(paired),
            "expected_units": expected_units, "incomplete_runs": incomplete,
            "sign_test": sign_test(late), "units_lower": int(sum(d < 0 for d in late)),
            "mean_unit_late": [_mean([ta[u]["late"] for u in paired]), _mean([tb[u]["late"] for u in paired])],
            "did": _statistic(did, complete, n_boot, boot_seed),
            "success_late_difference": cluster_bootstrap_mean(success, n_boot=n_boot, seed=boot_seed),
            "by_unit": {u: {"late": [ta[u]["late"], tb[u]["late"]], "early": [ta[u]["early"], tb[u]["early"]],
                            "success_late": [ta[u]["success_late"], tb[u]["success_late"]]} for u in paired}}


def amortization(cells: dict) -> dict:
    """Pre-registered check per noise level: the difference-in-differences of
    ``focused_sample - maximal`` passes in F1 and does not lie below 0 in F0."""
    out = {}
    name = f"{ARM} - {BOUND}"
    for eps in sorted({c["failure_prob"] for c in cells.values()}):
        f1 = cells.get(f"{REPLICATION[0]} eps={eps:g}", {}).get("contrasts", {}).get(name)
        f0 = cells.get(f"{NULL_CONTROL} eps={eps:g}", {}).get("contrasts", {}).get(name)
        if f1 is None or f0 is None:
            continue
        d1, d0 = f1["did"], f0["did"]
        if "incomplete" in (d1["verdict"], d0["verdict"]):
            supported = "incomplete"
        else:
            supported = "yes" if d1["ci"][1] < 0 and d0["ci"][1] >= 0 else "no"
        out[f"eps={eps:g}"] = {"F1_did": {k: d1[k] for k in ("estimate", "ci", "n")},
                               "F0_did": {k: d0[k] for k in ("estimate", "ci", "n")}, "supported": supported,
                               "role": "pre-registered" if eps == PRIMARY[1] else "reported"}
    return out


def analyze(runs: list[dict], expected_units: int | None = None, n_boot: int = N_BOOT,
            boot_seed: int = BOOT_SEED) -> dict:
    chosen, selection = select_runs(runs)
    cells: dict = defaultdict(lambda: defaultdict(list))
    for run in chosen:
        cells[(run["family"], run["eps"])][run["strategy"]].append(run)
    out = {"definition": {"late_episodes": f"{LATE_FROM + 1}-{EPISODES}", "early_episodes": f"1-{LATE_FROM}",
                          "unit": "world (block in F0)", "contrasts": [f"{a} - {b}" for a, b in CONTRASTS],
                          "bootstrap": {"resamples": n_boot, "seed": boot_seed, "level": 0.95},
                          "pass": "upper end of the 95% world-cluster bootstrap interval below 0, runs complete",
                          "failed_episodes": "count with every step used (their whole budget)"},
           "primary_cell": f"{PRIMARY[0]} eps={PRIMARY[1]:g}", "cells": {}, **selection}
    order = lambda kv: (kv[0] != PRIMARY, kv[0][0] == NULL_CONTROL, kv[0])
    for (family, eps), arms in sorted(cells.items(), key=order):
        role = ("primary" if (family, eps) == PRIMARY else "control" if family == NULL_CONTROL
                else "secondary")
        summaries = {s: arm_summary(rs) for s, rs in sorted(arms.items())}
        con = {f"{a} - {b}": contrast(arms[a], arms[b], expected_units, n_boot, boot_seed)
               for a, b in CONTRASTS if a in arms and b in arms}
        main = con.get(f"{ARM} - {BOUND}")
        out["cells"][f"{family} eps={eps:g}"] = {
            "family": family, "failure_prob": eps, "role": role, "units": arms[next(iter(arms))][0]["units"],
            "verdict": "missing" if main is None else "descriptive" if role == "control" else main["verdict"],
            "break_even_episode": break_even(summaries), "contrasts": con, "arms": summaries,
        }
    rep = [c for c in out["cells"].values() if c["family"] in REPLICATION]
    out["primary_verdict"] = out["cells"].get(out["primary_cell"], {}).get("verdict", "missing")
    out["general"] = len(rep) == 2 * len(REPLICATION) and all(c["verdict"] == "pass" for c in rep)
    out["amortization"] = amortization(out["cells"])
    out["break_even_later_under_noise"] = {}
    for fam in sorted({c["family"] for c in out["cells"].values()}):
        be = {c["failure_prob"]: c["break_even_episode"] for c in out["cells"].values() if c["family"] == fam}
        if len(be) == 2:
            lo, hi = sorted(be)
            out["break_even_later_under_noise"][fam] = episode_rank(be[hi]) > episode_rank(be[lo])
    return out


def _f(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(k: dict, nd=3) -> str:
    return f"{_f(k.get('estimate'), nd)} [{_f(k['ci'][0], nd)}, {_f(k['ci'][1], nd)}]" if k.get("ci") else "-"


def markdown(summary: dict) -> str:
    lines = ["Late half (episodes 9-16), paired per unit:", "",
             "| Cell | Role | Contrast | Units | Late ratios (a / b) | Mean difference [95% CI] | a lower in "
             "| Sign test p | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for label, c in summary["cells"].items():
        for name, k in c["contrasts"].items():
            m = k["mean_unit_late"]
            verdict = "descriptive" if c["role"] == "control" else k["verdict"]
            lines.append(f"| {label} | {c['role']} | `{name}` | {k['n_units']} | {_f(m[0])} / {_f(m[1])} "
                         f"| {_ci(k)} | {k['units_lower']} of {k['n_units']} "
                         f"| {_f(k['sign_test']['p_two_sided'], 4)} | {verdict} |")
    lines += ["", "Difference-in-differences (late minus early paired difference) and late success difference:", "",
              "| Cell | Contrast | D [95% CI] | D below 0 | Late success difference [95% CI] |",
              "|---|---|---|---|---|"]
    for label, c in summary["cells"].items():
        for name, k in c["contrasts"].items():
            lines.append(f"| {label} | `{name}` | {_ci(k['did'])} | {k['did']['verdict']} | "
                         f"{_ci(k['success_late_difference'])} |")
    lines += ["", "Pooled cost ratios (steps / optimal steps), success and omitted candidates:", "",
              "| Cell | Arm | Runs | Episodes 1-8 | Episodes 9-16 | All | Success (late) | Omitted (late) |",
              "|---|---|---|---|---|---|---|---|"]
    for label, c in summary["cells"].items():
        for s, a in c["arms"].items():
            inc = f" ({len(a['incomplete_runs'])} incomplete)" if a["incomplete_runs"] else ""
            lines.append(f"| {label} | `{s}` | {a['runs']}{inc} | {_f(a['cost_ratio_early'])} | "
                         f"{_f(a['cost_ratio_late'])} | {_f(a['cost_ratio_all'])} | "
                         f"{_f(a['success_all'])} ({_f(a['success_late'])}) | {_f(a['omitted_fraction_late'])} |")
    lines += ["", "Break-even episode (descriptive): " + "; ".join(
        f"{label} {c['break_even_episode'] if c['break_even_episode'] is not None else '-'}"
        for label, c in summary["cells"].items()) + "."]
    for eps, a in summary["amortization"].items():
        lines.append(f"Amortization check at {eps} ({a['role']}): F1 D {_ci(a['F1_did'])}, F0 D {_ci(a['F0_did'])}; "
                     f"supported: {a['supported']}.")
    lines.append(f"Primary cell {summary['primary_cell']}: {summary['primary_verdict']}. "
                 f"General replication (every F1 and F2 cell passes): {'yes' if summary['general'] else 'no'}.")
    if summary["reruns_used"]:
        lines.append("Incomplete runs replaced by their reruns: " + ", ".join(
            f"{x['incomplete']} by {x['rerun']}" for x in summary["reruns_used"]) + ".")
    if summary["ignored"]:
        lines.append(f"Runs ignored (a run for the same cell, arm and seed counts): {', '.join(summary['ignored'])}.")
    return "\n".join(lines) + "\n"


def figure(summary: dict, path: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = list(summary["cells"].items())
    if not cells:
        return None
    colors = {REFERENCE: "tab:purple", BOUND: "tab:red", ARM: "tab:olive", CONTROL: "tab:gray"}
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.6), squeeze=False)
    for ax, (label, c) in zip(axes[0], cells):
        for s, a in c["arms"].items():
            eps = sorted(a["by_episode"], key=int)
            ax.plot([int(e) for e in eps], [a["by_episode"][e]["cost_ratio"] for e in eps], marker="o",
                    markersize=3, color=colors.get(s), label=f"{s} (episodes={a['episodes']})")
        ax.axvline(LATE_FROM + 0.5, color="gray", lw=0.8, ls="--")
        k = c["contrasts"].get(f"{ARM} - {BOUND}", {})
        ax.set_title(f"{label} ({k.get('n_units', 0)} paired {c['units']})", fontsize=8)
        ax.set_xlabel("position in a block" if c["units"] == "blocks" else "episode index within a world")
        ax.set_ylabel("primitive steps / optimal steps")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def write_report(runs_dir: str | Path, prefix: str, out_dir: str | Path, expected_units: int | None = None,
                 n_boot: int = N_BOOT, make_figure: bool = True) -> dict:
    runs = load_runs(runs_dir, prefix)
    summary = analyze(runs, expected_units, n_boot)
    summary["source"] = {"runs_dir": str(runs_dir), "prefix": prefix, "runs": sorted(r["run_id"] for r in runs)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if make_figure:
        summary["figure"] = figure(summary, out / "replication_cost_by_episode.png")
    (out / "replication_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "replication_summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary
