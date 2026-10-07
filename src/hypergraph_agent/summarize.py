"""Summarize recorded runs into tables, contrasts, figures and small evidence copies.

    python -m hypergraph_agent.summarize --runs runs --out artifacts

Everything is computed from run manifests and logs. Figures are drawn only
from recorded episodes; nothing is interpolated or synthesized.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation.stats import difference, interaction

PRIMARY = {
    "p1": [("gated", "set"), ("gated", "incidence")],
    "p2": [("adaptive_incidence", "fixed_supergraph"), ("fixed_supergraph", "evidence_set")],
    "p3": [("G1", "G0"), ("F1", "F0"), ("G0", "F0")],
}
EVIDENCE_EVENTS = {"structural_incidence_edit", "topology_commit", "topology_rollback",
                   "freeze_dependency_revision", "skill_proposed", "skill_reproposed",
                   "skill_deferred", "skill_rejected", "skill_admitted", "skill_retired",
                   "skill_validation", "skill_practice_batch", "discovery_round",
                   "discovery_summary", "context_weight", "batch", "search_benchmark",
                   "edit_policy_training"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(roots: list[Path]) -> list[dict]:
    runs = []
    for root in roots:
        for man in sorted(root.rglob("manifest.json")):
            m = json.loads(man.read_text())
            m["_dir"] = str(man.parent)
            runs.append(m)
    return runs


def eval_rows(run: dict) -> dict[str, list[dict]]:
    d = Path(run["_dir"])
    return {f.stem: read_jsonl(f) for f in sorted(d.glob("eval*.jsonl"))}


def success_rate(rows):
    return float(np.mean([r["success"] for r in rows])) if rows else None


def summarize(runs: list[dict]) -> dict:
    out = {"runs": [], "contrasts": {}}
    by_phase = defaultdict(lambda: defaultdict(dict))  # phase -> arm -> seed -> outcomes
    for r in runs:
        if r.get("arm") == "pretrain" or r.get("status") != "completed":
            out["runs"].append({k: r.get(k) for k in ("run_id", "phase", "arm", "seed", "status",
                                                       "stop_reason", "interactions", "library")})
            continue
        ev = eval_rows(r)
        main_key = "eval_test" if "eval_test" in ev else "eval"
        rows = ev.get(main_key, [])
        by_phase[r["phase"]][r["arm"]][r["seed"]] = [float(x["success"]) for x in rows]
        out["runs"].append({
            "run_id": r["run_id"], "phase": r["phase"], "arm": r["arm"], "seed": r["seed"],
            "status": r["status"], "stop_reason": r.get("stop_reason"),
            "interactions": r.get("interactions"), "topology_deltas": r.get("topology_deltas"),
            "eval_set": main_key, "eval_n": len(rows), "eval_success": success_rate(rows),
            "other_eval": {k: {"n": len(v), "success": success_rate(v)} for k, v in ev.items() if k != main_key},
            "library": r.get("library"),
        })
    for phase, arms in by_phase.items():
        res = {}
        for a, b in PRIMARY.get(phase, []):
            if a in arms and b in arms:
                res[f"{a} - {b}"] = difference(arms, a, b, n_boot=2000)
        if phase == "p3" and all(k in arms for k in ("G1", "G0", "F1", "F0")):
            res["interaction (G1-G0)-(F1-F0)"] = interaction(arms, n_boot=2000)
        out["contrasts"][phase] = res
    return out


def figures(runs: list[dict], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    done = [r for r in runs if r.get("status") == "completed" and r.get("arm") != "pretrain"]
    if not done:
        return made
    # 1: success versus all-in adaptive interactions (one point per trained run)
    fig, ax = plt.subplots(figsize=(6, 4))
    for phase in sorted({r["phase"] for r in done}):
        pts = []
        for r in done:
            if r["phase"] != phase:
                continue
            ev = eval_rows(r)
            rows = ev.get("eval_test") or ev.get("eval") or []
            inter = r.get("interactions") or {}
            cost = inter.get("all_in_logical", inter.get("adaptive_physical"))
            if rows and cost is not None:
                pts.append((cost, success_rate(rows), r["arm"]))
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=phase, alpha=0.8)
            for x, y, arm in pts:
                ax.annotate(arm, (x, y), fontsize=6, alpha=0.7)
    ax.set_xlabel("all-in adaptive primitive interactions per run")
    ax.set_ylabel("evaluation success rate")
    ax.set_title(f"Success vs learning cost (n={len(done)} runs, one point per run)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_cost.png", dpi=120)
    plt.close(fig)
    made.append("success_vs_cost.png")

    # 2: success versus reference primitive horizon, pooled per phase/arm
    fig, ax = plt.subplots(figsize=(6, 4))
    any_line = False
    for phase in sorted({r["phase"] for r in done}):
        arms = defaultdict(list)
        for r in done:
            if r["phase"] == phase:
                ev = eval_rows(r)
                arms[r["arm"]] += ev.get("eval_test") or ev.get("eval") or []
        for arm, rows in sorted(arms.items()):
            rows = [x for x in rows if x.get("reference_length")]
            if not rows:
                continue
            edges = np.unique(np.quantile([x["reference_length"] for x in rows], [0, .25, .5, .75, 1]))
            if len(edges) < 2:
                continue
            idx = np.clip(np.digitize([x["reference_length"] for x in rows], edges[1:-1]), 0, len(edges) - 2)
            xs, ys, ns = [], [], []
            for b in range(len(edges) - 1):
                sel = [x for x, i in zip(rows, idx) if i == b]
                if sel:
                    xs.append(np.mean([x["reference_length"] for x in sel]))
                    ys.append(success_rate(sel))
                    ns.append(len(sel))
            ax.plot(xs, ys, marker="o", label=f"{phase}:{arm} (episodes={sum(ns)})")
            any_line = True
    if any_line:
        ax.set_xlabel("optimal primitive plan length (evaluator reference)")
        ax.set_ylabel("success rate (pooled over seeds)")
        ax.set_title("Success vs required primitive horizon")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / "success_vs_horizon.png", dpi=120)
        made.append("success_vs_horizon.png")
    plt.close(fig)

    # 3: mechanism diagnostics: cumulative structural edits (P2) and skill calls per batch (P3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for r in runs:
        ev = read_jsonl(Path(r["_dir"]) / "events.jsonl")
        if r["phase"] == "p2" and r.get("arm") == "adaptive_incidence":
            batches = [e["batch"] for e in ev if e["type"] == "batch"]
            edits, n, cum = [e for e in ev if e["type"] == "structural_incidence_edit"], 0, []
            per_batch = defaultdict(int)
            last = 0
            for e in ev:
                if e["type"] == "batch":
                    last = e["batch"]
                elif e["type"] == "structural_incidence_edit":
                    per_batch[last] += len(e["removes"]) + len(e["adds"])
            for b in batches:
                n += per_batch[b]
                cum.append(n)
            if batches:
                axes[0].plot(batches, cum, label=f"seed {r['seed']}")
        if r["phase"] == "p3":
            b = [e for e in ev if e["type"] == "batch"]
            if b:
                axes[1].plot(range(len(b)), [e.get("skill_calls", 0) for e in b], alpha=0.7,
                             label=f"{r['arm']} s{r['seed']}")
    axes[0].set_title("P2 adaptive: cumulative incidence edits")
    axes[0].set_xlabel("batch")
    axes[1].set_title("P3: root skill calls per manager batch")
    axes[1].set_xlabel("manager batch (run order)")
    for a in axes:
        if a.lines:
            a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_diagnostics.png", dpi=120)
    plt.close(fig)
    made.append("mechanism_diagnostics.png")

    # 4: walkers: cost relative to the optimum by episode index within a world
    colors = {"reference": "tab:purple", "maximal": "tab:red", "sample": "tab:blue",
              "optimistic": "tab:green", "local_uniform": "tab:gray", "local_focused": "tab:orange",
              "learned": "tab:brown", "exact": "tab:pink"}
    groups = defaultdict(dict)  # panel -> strategy -> ep -> sums
    for r in done:
        if not str(r["phase"]).startswith("walker"):
            continue
        cfg_path = Path(r["_dir"]) / "config.json"
        study = json.loads(cfg_path.read_text())["run"]["name"] if cfg_path.exists() else r["phase"]
        study = study.replace("-bounds", "")
        for row in eval_rows(r).get("eval", []):
            variant = row.get("variant", "")
            panel = f"{study}: training worlds" if variant == "collection" else f"{study} {variant}".strip()
            g = groups[panel].setdefault(row["strategy"], {})
            s = g.setdefault(row["world_episode"], [0, 0, 0])
            s[0] += row["primitive_length"]
            s[1] += row["reference_length"] or 0
            s[2] += 1
    # panels need several episodes per world (fresh-world-per-task studies have one)
    groups = {p: g for p, g in groups.items() if any(len(eps) > 1 for eps in g.values())}
    if groups:
        panels = sorted(groups)
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
        for ax, panel in zip(axes[0], panels):
            for strat, eps in sorted(groups[panel].items()):
                xs = sorted(eps)
                ax.plot(xs, [eps[x][0] / max(eps[x][1], 1) for x in xs], marker="o", markersize=3,
                        color=colors.get(strat), label=f"{strat} (episodes={sum(eps[x][2] for x in xs)})")
            ax.set_title(panel, fontsize=8)
            ax.set_xlabel("episode index within an unseen world")
            ax.set_ylabel("primitive steps / optimal steps")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / "walker_cost_by_episode.png", dpi=120)
        plt.close(fig)
        made.append("walker_cost_by_episode.png")
    return made


def copy_evidence(runs: list[dict], out_dir: Path) -> None:
    for r in runs:
        src = Path(r["_dir"])
        dst = out_dir / "runs" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "config.json"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        events = [e for e in read_jsonl(src / "events.jsonl") if e.get("type") in EVIDENCE_EVENTS]
        with open(dst / "events.jsonl", "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        for f in src.glob("eval*.jsonl"):
            shutil.copy2(f, dst / f.name)
        if (src / "library" / "library.json").exists():
            (dst / "library").mkdir(exist_ok=True)
            shutil.copy2(src / "library" / "library.json", dst / "library" / "library.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.summarize")
    p.add_argument("--runs", nargs="+", required=True, help="run directories or parents of runs")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--no-copy", action="store_true")
    args = p.parse_args(argv)
    runs = collect([Path(x) for x in args.runs])
    if not runs:
        print("no runs found")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    summary["figures"] = figures(runs, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if not args.no_copy:
        copy_evidence(runs, out_dir)
    print(json.dumps(summary["contrasts"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
