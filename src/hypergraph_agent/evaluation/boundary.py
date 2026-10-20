"""Study R2 (docs/studies/R2_boundary.md): where the study R result stops replicating.

Two parts, both on the evaluator side.

Family statistics (generator, reference solver and the brute-force plan; no
environment interaction). For every task a run plays: the optimal length;
the hidden recipes of the task (every rule, including the distractor); the
alternative recipes (second recipes of the items the goal needs); the
hypothesis entropy, the sum over hidden recipes of log2 of the class size (41
for pools of 6, 92 for pools of 8, requirements of 1-3); the crafts of the
optimal plan; the brute-force length, the cheapest plan on the full-pool node,
which is what ``maximal`` executes at failure probability 0 (every craft
succeeds); and the budget. Per world over its 16 episodes: the distinct
recipes and the recurrence, the share of a task's recipes already seen in an
earlier task of the world. ``guarantee`` is DECISIONS W2: every reference
exact, the brute-force enumeration not truncated, no budget at the cap (so
every budget is the public bound plus the slack) and the brute-force plan
within every budget, so ``maximal`` cannot fail without noise.

Endpoints (recorded runs). A family is the run name after its first hyphen,
without ``-bounds`` (``r2-f2s6`` is F2S6, ``r2-f2-d`` is F2-D); every family is
played at failure probability 0. Within a family the endpoints are those of
study R (``evaluation.replication``): per world, the late (episodes 9-16)
paired difference ``focused_sample - maximal`` of the cost ratio, its
world-cluster bootstrap interval, the difference-in-differences, the late
success difference and the break-even episode. Added here:

* a three-way reading of the late difference: ``advantage`` (interval below
  0, the study R pass), ``reversed`` (interval above 0, as F2 in study R),
  ``inconclusive``, or ``incomplete`` (an incomplete run, too few worlds, or no
  usable bounds);
* the decomposition of the pooled late excess, ``sum(steps_a - steps_b) /
  sum(optimal)`` over late episodes, into the episodes ``focused_sample``
  failed and the ones it solved (the two parts add up to the excess), both
  arms' cost ratios on the solved ones, and the attribution of an excess to
  the failed episodes (at least half of it) or the solved ones;
* failure by goal depth, and how many failed late episodes ended at the
  deadline after at least one contradicted guess;
* the pre-registered combinations: depth against headroom (D and DH, with F2-A
  as the near-matched deep check) and the caveats of the two controls (I, the
  F1-side positive control; F2, the replica of study R's F2);
* slack contrasts between a looser-budget family and its base, which play the
  same worlds and tasks: per world, the late difference in the looser family
  minus that in the base, with the same bootstrap.

A slack family runs only ``focused_sample``: at failure probability 0
``maximal`` and the reference never reach the budget and do not depend on it,
so the base family's bounds are used. A borrowed run is used only if, seed by
seed, its recorded generator settings equal those of the slack family apart
from the slack, and every episode of the slack family has a counterpart in it
with the same task, goal depth and optimal length and a recorded budget that
differs by exactly the slack difference (``borrow_check``). Run selection and
the rerun rule are those of study R.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import eval_stream_config
from ..envs.derivations import StructRecipe, enumerate_derivations, plan_cost
from ..envs.generator import TaskStream
from ..envs.reference_solver import reference_solve
from ..topology.inference import enumerate_hypotheses
from . import replication as R
from .stats import cluster_bootstrap_mean, sign_test
from .walkers import eval_order, variant_config

MAX_SIZE = 3  # the walkers' requirement class: nonempty subsets of at most 3 candidates
LATE_FROM = R.LATE_FROM


# ------------------------------------------------------------------ family statistics

def brute_force_plan(task) -> tuple[int, bool]:
    """Length of the cheapest plan on the full-pool node (every candidate
    required), what ``maximal`` executes at failure probability 0, and whether
    the derivation enumeration was truncated (then the length is only an upper
    bound and need not be what ``maximal`` plays)."""
    recipes = [StructRecipe(r.effect, r.item_inputs, r.pool) for r in task.rules]
    ds, truncated, _ = enumerate_derivations(recipes, frozenset(task.initial_true), task.goal)
    return plan_cost(ds[0]), truncated


def brute_force_length(task) -> int:
    return brute_force_plan(task)[0]


def task_statistics(task) -> dict:
    ref = reference_solve(task)
    closure = task.closure_items()
    brute, truncated = brute_force_plan(task)
    return {
        "optimal": ref.length, "exact": ref.exact, "depth": task.depth, "budget": task.budget,
        "hidden": len(task.rules), "alternatives": sum(r.variant == 1 and r.effect in closure for r in task.rules),
        "entropy": sum(math.log2(len(enumerate_hypotheses(len(r.pool), min(MAX_SIZE, len(r.pool)))))
                       for r in task.rules),
        "plan_crafts": sum(k.startswith("craft:") for k in ref.plan),
        "brute_force": brute, "brute_force_truncated": truncated,
        "signatures": sorted({r.signature(task.profile) for r in task.rules}),
    }


def stream_tasks(cfg: dict, seeds=None) -> list[dict]:
    """Every task the runs of a config play, in play order, with its unit
    (seed and world) and episode index, under the config's first evaluation
    variant (every R2 config has one)."""
    variant = (cfg["walker"]["eval_variants"] or [{"name": "default", "tasks": {}}])[0]
    out = []
    for seed in cfg["run"]["seeds"] if seeds is None else seeds:
        scfg = eval_stream_config(variant_config(cfg, seed, variant))
        stream = TaskStream(scfg)
        per_world: dict = {}
        for i in eval_order(scfg, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"]):
            t = stream.task(i)
            wk = t.world.world_key
            per_world[wk] = per_world.get(wk, -1) + 1
            out.append({"seed": seed, "unit": f"s{seed}:{wk}", "episode": per_world[wk],
                        "task_key": t.task_key, **task_statistics(t)})
    return out


def _mean_range(xs) -> dict:
    return {"mean": float(np.mean(xs)), "min": int(min(xs)), "max": int(max(xs))}


def family_statistics(cfg: dict, seeds=None) -> dict:
    """The study R family table for one config, plus the W2 guarantee."""
    rows = stream_tasks(cfg, seeds)
    cap = eval_stream_config(cfg).task.max_budget
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit"]].append(r)
    recur, recur_late, distinct = [], [], []
    for rs in by_unit.values():
        seen: set = set()
        for r in sorted(rs, key=lambda x: x["episode"]):
            if r["episode"] > 0:
                share = sum(s in seen for s in r["signatures"]) / len(r["signatures"])
                recur.append(share)
                if r["episode"] >= LATE_FROM:
                    recur_late.append(share)
            seen |= set(r["signatures"])
        distinct.append(len(seen))
    opt = [r["optimal"] for r in rows]
    brute = [r["brute_force"] for r in rows]
    guarantee = {"tasks": len(rows), "all_exact": all(r["exact"] for r in rows),
                 "brute_force_exact": not any(r["brute_force_truncated"] for r in rows),
                 "max_budget": max(r["budget"] for r in rows), "budget_cap": cap,
                 "below_cap": all(r["budget"] < cap for r in rows),
                 "brute_force_within_budget": all(b <= r["budget"] for b, r in zip(brute, rows)),
                 "min_budget_minus_brute_force": int(min(r["budget"] - b for b, r in zip(brute, rows)))}
    guarantee["holds"] = all(guarantee[k] for k in ("all_exact", "brute_force_exact", "below_cap",
                                                     "brute_force_within_budget"))
    return {
        "tasks": len(rows), "units": len(by_unit),
        "depth": {str(d): sum(r["depth"] == d for r in rows) for d in sorted({r["depth"] for r in rows})},
        "optimal_length": _mean_range(opt),
        "hidden_recipes": _mean_range([r["hidden"] for r in rows]),
        "alternative_recipes": float(np.mean([r["alternatives"] for r in rows])),
        "entropy_bits": float(np.mean([r["entropy"] for r in rows])),
        "plan_crafts": float(np.mean([r["plan_crafts"] for r in rows])),
        "distinct_recipes_per_world": float(np.mean(distinct)),
        "recurrence": float(np.mean(recur)) if recur else 0.0,
        "recurrence_late": float(np.mean(recur_late)) if recur_late else 0.0,
        "headroom": float(np.mean([b / o for b, o in zip(brute, opt)])),
        "headroom_pooled": sum(brute) / sum(opt),
        "budget": _mean_range([r["budget"] for r in rows]),
        "budget_over_optimal_pooled": sum(r["budget"] for r in rows) / sum(opt),
        "guarantee": guarantee,
    }


def family_label(run_name: str) -> str:
    """The family of a run name, as ``replication.load_runs`` reads it."""
    return run_name.split("-", 1)[-1].removesuffix("-bounds").upper()


# ------------------------------------------------------------------ endpoints

ARM, BOUND, REFERENCE = R.ARM, R.BOUND, R.REFERENCE
PRIMARY = "D"
# kind of each family; this order is the order of every table
KINDS = {
    "D": "one factor from F1 (primary)", "DH": "D with pools of 8 (headroom-matched depth)",
    "A": "one factor from F1", "P": "one factor from F1", "I": "one factor from F1 (positive control)",
    "F2-D": "one factor removed from F2", "F2-A": "one factor removed from F2 (near-matched deep check)",
    "F2-P": "one factor removed from F2", "F2": "F2 replica, slack 2 (control)", "F2S6": "F2, slack 6",
    "F2S12": "F2, slack 12", "DS6": "D, slack 6", "DS12": "D, slack 12",
}
# slack families: their slack-2 base, whose bounds they use (identical episodes at failure probability 0)
BASES = {"DS6": "D", "DS12": "D", "F2S6": "F2", "F2S12": "F2"}
SLACK_CONTRASTS = (("F2S6", "F2"), ("F2S12", "F2"), ("DS6", "D"), ("DS12", "D"))
# controls: family, the reading it is expected to show, and what a different reading qualifies
CONTROLS = (("I", "advantage", "D's reading (the F1-side positive control at R2's code)"),
            ("F2", "reversed", "every comparison with F2 (the replica of study R's F2)"))
TASK_FIELDS = ("task_key", "depth", "reference_length")


def generator_settings(cfg: dict) -> tuple[dict, int | None]:
    """What defines the tasks of a recorded run (its ``config.json``): the
    world, the task settings other than the budget slack and the failure
    probability, and the evaluation stream (namespace, the seed's base seed,
    worlds, tasks); and the budget slack."""
    train, ev = cfg.get("train") or {}, cfg.get("eval") or {}
    tasks = {**(train.get("tasks") or {}), **(ev.get("tasks") or {})}
    slack = tasks.pop("budget_slack", None)
    tasks.pop("failure_prob", None)
    stream = {"namespace": ev.get("namespace"), "base_seed": ev.get("base_seed"),
              "world_mode": ev.get("world_mode") or train.get("world_mode"),
              "n_worlds": ev.get("n_worlds") or train.get("n_worlds"), "n_tasks": ev.get("n_tasks"),
              "episodes_per_world": ev.get("episodes_per_world"), "goal_levels": ev.get("goal_levels")}
    return {"world": cfg.get("world"), "tasks": tasks, "stream": stream}, slack


def load_runs(runs_dir: str | Path, prefix: str) -> list[dict]:
    """``replication.load_runs`` plus each run's generator settings and budget
    slack from its recorded config (``generator`` and ``slack``)."""
    runs = R.load_runs(runs_dir, prefix)
    settings = {}
    for d in Path(runs_dir).glob(f"{prefix}*"):
        if (d / "manifest.json").exists() and (d / "config.json").exists():
            run_id = json.loads((d / "manifest.json").read_text())["run_id"]
            settings[run_id] = generator_settings(json.loads((d / "config.json").read_text()))
    for r in runs:
        r["generator"], r["slack"] = settings.get(r["run_id"], (None, None))
    return runs


def _episodes(runs: list[dict]) -> dict:
    return {(r["_unit"], r["_pos"]): r for x in runs for r in x["rows"]}


def borrow_check(looser: list[dict], base: list[dict]) -> dict:
    """Whether runs of a looser-budget family (``looser``) and runs of its
    base family played the same tasks: seed by seed, the same recorded
    generator settings apart from the slack and a known slack on both sides;
    every episode of ``looser`` with a counterpart in ``base`` that has the
    same task, goal depth and optimal length and a recorded budget larger by
    exactly the slack difference."""
    by_seed = {x["seed"]: x for x in base}
    problems, shared, missing, mismatched = [], 0, 0, 0
    for xa in looser:
        xb = by_seed.get(xa["seed"])
        if xb is None:
            problems.append(f"seed {xa['seed']}: no base run")
            continue
        if xa.get("generator") is None or xa.get("generator") != xb.get("generator"):
            problems.append(f"seed {xa['seed']}: generator settings differ or are unknown")
        if xa.get("slack") is None or xb.get("slack") is None:
            problems.append(f"seed {xa['seed']}: slack unknown")
            continue
        diff = xa["slack"] - xb["slack"]
        eb = {(r["_unit"], r["_pos"]): r for r in xb["rows"]}
        for r in xa["rows"]:
            s = eb.get((r["_unit"], r["_pos"]))
            if s is None:
                missing += 1
                continue
            shared += 1
            mismatched += any(r.get(f) != s.get(f) for f in TASK_FIELDS) or \
                r.get("budget") is None or s.get("budget") is None or r["budget"] - s["budget"] != diff
    return {"shared_episodes": shared, "missing_episodes": missing, "mismatched": int(mismatched),
            "problems": problems, "ok": shared > 0 and not missing and not mismatched and not problems}


def reading(k: dict | None) -> str:
    """Three-way reading of a paired difference's interval."""
    if not k or k.get("verdict") == "incomplete" or not k.get("ci"):
        return "incomplete"
    lo, hi = k["ci"]
    return "advantage" if hi < 0 else "reversed" if lo > 0 else "inconclusive"


def excess_decomposition(runs_a: list[dict], runs_b: list[dict]) -> dict:
    """Pooled late excess of arm a over arm b, ``sum(steps_a - steps_b) /
    sum(optimal)`` over the late episodes both played, split by whether arm a
    failed the episode; both arms' cost ratios on the episodes a solved; and
    the attribution of a positive excess: to the failed episodes when they
    carry at least half of it (``from_failed >= from_solved``), else to the
    solved ones."""
    ea, eb = _episodes(runs_a), _episodes(runs_b)
    keys = sorted(k for k in set(ea) & set(eb) if k[1] >= LATE_FROM)
    opt = sum(ea[k]["reference_length"] or 0 for k in keys)
    if not opt:
        return {"late_episodes": len(keys)}
    failed = [k for k in keys if not ea[k]["success"]]
    solved = [k for k in keys if ea[k]["success"]]

    def part(ks):
        return sum(ea[k]["primitive_length"] - eb[k]["primitive_length"] for k in ks) / opt

    def ratio(e, ks):
        o = sum(e[k]["reference_length"] or 0 for k in ks)
        return sum(e[k]["primitive_length"] for k in ks) / o if o else float("nan")

    out = {"late_episodes": len(keys), "failed_episodes": len(failed), "excess": part(keys),
           "from_failed": part(failed), "from_solved": part(solved),
           "solved_ratio": [ratio(ea, solved), ratio(eb, solved)],
           "failed_at_deadline_after_contradiction": int(sum(
               ea[k].get("status") == "deadline" and ea[k].get("contradictions", 0) > 0 for k in failed))}
    out["attribution"] = (None if out["excess"] <= 0 else
                          "failed episodes" if out["from_failed"] >= out["from_solved"] else "solved episodes")
    return out


def failure_by_depth(runs: list[dict]) -> dict:
    """Failure rate of an arm by goal depth, over all and over late episodes."""
    rows = [r for x in runs for r in x["rows"]]
    out = {}
    for d in sorted({r["depth"] for r in rows}):
        rs = [r for r in rows if r["depth"] == d]
        late = [r for r in rs if r["_pos"] >= LATE_FROM]
        out[str(d)] = {"n": len(rs), "failure": 1 - float(np.mean([r["success"] for r in rs])),
                       "n_late": len(late),
                       "failure_late": 1 - float(np.mean([r["success"] for r in late])) if late else float("nan")}
    return out


def slack_contrast(looser: dict, base: dict, expected_units: int | None, n_boot: int = R.N_BOOT,
                   boot_seed: int = R.BOOT_SEED) -> dict:
    """Per world, the paired late difference ``focused_sample - maximal`` in
    the looser-budget family minus that in its base family, with the change of
    the early difference and of ``focused_sample``'s late success. Refused
    (``paired: false``, reading ``incomplete``) unless the two samplers pass
    ``borrow_check`` and both families have bounds. Reading: ``lower``
    (interval below 0: the looser budget lowers the sampler's late excess),
    ``higher``, ``inconclusive`` or ``incomplete``."""
    check = borrow_check(looser.get(ARM, []), base.get(ARM, []))
    if not check["ok"] or BOUND not in looser or BOUND not in base:
        return {"tasks": check, "paired": False, "verdict": "incomplete", "reading": "incomplete"}
    tabs = [{u: v for x in runs for u, v in R.unit_table(x["rows"]).items()}
            for runs in (looser[ARM], looser[BOUND], base[ARM], base[BOUND])]
    half = R.EPISODES - LATE_FROM
    units = sorted(u for u in set.intersection(*(set(t) for t in tabs))
                   if all(t[u]["n_late"] == half and t[u]["n_early"] == LATE_FROM for t in tabs))
    fa, ma, fb, mb = tabs
    late = [(fa[u]["late"] - ma[u]["late"]) - (fb[u]["late"] - mb[u]["late"]) for u in units]
    early = [(fa[u]["early"] - ma[u]["early"]) - (fb[u]["early"] - mb[u]["early"]) for u in units]
    success = [fa[u]["success_late"] - fb[u]["success_late"] for u in units]
    incomplete = sorted({x["run_id"] for s in (looser, base) for a in (ARM, BOUND) for x in s[a]
                         if not x["complete"]})
    complete = bool(units) and not incomplete and (expected_units is None or len(units) >= expected_units)
    k = {"tasks": check, "paired": True, **R._statistic(late, complete, n_boot, boot_seed),
         "n_units": len(units), "expected_units": expected_units, "incomplete_runs": incomplete,
         "units_lower": int(sum(d < 0 for d in late)), "sign_test": sign_test(late),
         "early_change": cluster_bootstrap_mean(early, n_boot=n_boot, seed=boot_seed),
         "success_late_change": cluster_bootstrap_mean(success, n_boot=n_boot, seed=boot_seed)}
    k["reading"] = {"advantage": "lower", "reversed": "higher"}.get(reading(k), reading(k))
    return k


def depth_against_headroom(d: str, dh: str, f2a: str | None = None) -> str:
    """Pre-registered reading of D and DH (D with F1's brute-force headroom) together;
    F2-A, the near-matched deep check, is read like DH and a disagreement is unresolved."""
    if "incomplete" in (d, dh) or "missing" in (d, dh):
        return "incomplete"
    if d != "reversed":
        return "D not reversed: no reversal to attribute"
    if f2a is not None and f2a not in ("incomplete", "missing") and f2a != dh:
        return f"unresolved: DH is {dh} but F2-A is {f2a}"
    return {"reversed": "depth matters beyond headroom", "advantage": "headroom, not depth",
            "inconclusive": "headroom accounts for part of D's reversal; depth beyond headroom not shown"}[dh]


def analyze(runs: list[dict], expected_units: int | None = None, n_boot: int = R.N_BOOT,
            boot_seed: int = R.BOOT_SEED) -> dict:
    chosen, selection = R.select_runs(runs)
    fams: dict = defaultdict(lambda: defaultdict(list))
    for run in chosen:
        label = run["family"] if run["eps"] == 0.0 else f"{run['family']} eps={run['eps']:g}"
        fams[label][run["strategy"]].append(run)
    borrowed = {}
    for fam, base in BASES.items():
        if ARM not in fams.get(fam, {}):
            continue
        borrowed[fam] = {"base": base, "used": [], "checks": {}}
        for s in (BOUND, REFERENCE):
            if s in fams[fam] or s not in fams.get(base, {}):
                continue
            check = borrow_check(fams[fam][ARM], fams[base][s])
            borrowed[fam]["checks"][s] = check
            if check["ok"]:
                fams[fam][s] = fams[base][s]
                borrowed[fam]["used"].append(s)
    out = {"definition": {"late_episodes": f"{LATE_FROM + 1}-{R.EPISODES}", "early_episodes": f"1-{LATE_FROM}",
                          "unit": "world", "contrast": f"{ARM} - {BOUND}",
                          "bootstrap": {"resamples": n_boot, "seed": boot_seed, "level": 0.95},
                          "reading": "advantage: interval below 0 (the study R pass); reversed: interval above 0; "
                                     "otherwise inconclusive; incomplete runs, too few worlds or no usable bounds: "
                                     "incomplete",
                          "failed_episodes": "count with every step used (their whole budget)"},
           "primary_family": PRIMARY, "families": {}, "bounds_borrowed": borrowed, **selection}
    order = {f: i for i, f in enumerate(KINDS)}
    for fam in sorted(fams, key=lambda f: (order.get(f, len(order)), f)):
        arms = fams[fam]
        summaries = {s: R.arm_summary(rs) for s, rs in sorted(arms.items())}
        main = (R.contrast(arms[ARM], arms[BOUND], expected_units, n_boot, boot_seed)
                if ARM in arms and BOUND in arms else None)
        used = borrowed.get(fam, {}).get("used")
        out["families"][fam] = {
            "role": "primary" if fam == PRIMARY else "secondary", "kind": KINDS.get(fam, "other"),
            "bounds_from": (borrowed[fam]["base"] if used else fam) if BOUND in arms else None,
            "verdict": "incomplete" if main is None else main["verdict"],
            "reading": "incomplete" if main is None else reading(main),
            "break_even_episode": R.break_even(summaries),
            "contrasts": {} if main is None else {f"{ARM} - {BOUND}": main},
            "decomposition": {} if main is None else excess_decomposition(arms[ARM], arms[BOUND]),
            "failure_by_depth": failure_by_depth(arms[ARM]) if ARM in arms else {},
            "arms": summaries, "units": "worlds",
        }
    get = lambda f: out["families"].get(f, {}).get("reading", "missing")
    out["primary_verdict"] = out["families"].get(PRIMARY, {}).get("verdict", "missing")
    out["primary_reading"] = get(PRIMARY)
    out["depth_against_headroom"] = {"D": get("D"), "DH": get("DH"), "F2-A": get("F2-A"),
                                     "reading": depth_against_headroom(get("D"), get("DH"), get("F2-A"))}
    out["caveats"] = [f"{fam} is {get(fam)} (expected {want}): this qualifies {what}"
                      for fam, want, what in CONTROLS if get(fam) != want]
    out["slack_contrasts"] = {f"{a} vs {b}": slack_contrast(fams[a], fams[b], expected_units, n_boot, boot_seed)
                              for a, b in SLACK_CONTRASTS if ARM in fams.get(a, {}) and ARM in fams.get(b, {})}
    return out


# ------------------------------------------------------------------ reports

_f, _ci = R._f, R._ci


def markdown(summary: dict) -> str:
    name = f"{ARM} - {BOUND}"
    lines = ["Late half (episodes 9-16), `focused_sample - maximal` paired per world:", "",
             "| Family | Kind | Worlds | Late ratios (sampler / maximal) | Mean difference [95% CI] "
             "| Sampler lower in | Sign test p | Reading | Break-even |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, c in summary["families"].items():
        k = c["contrasts"].get(name)
        if k is None:
            lines.append(f"| {fam} | {c['kind']} | - | - | - | - | - | {c['reading']} | - |")
            continue
        m = k["mean_unit_late"]
        lines.append(f"| {fam} | {c['kind']} | {k['n_units']} | {_f(m[0])} / {_f(m[1])} | {_ci(k)} | "
                     f"{k['units_lower']} of {k['n_units']} | {_f(k['sign_test']['p_two_sided'], 4)} | "
                     f"{c['reading']} | {c['break_even_episode']} |")
    lines += ["", "Difference-in-differences, late success difference and the late excess by outcome "
              "(pooled; failed episodes count their whole budget):", "",
              "| Family | DiD [95% CI] | Late success difference [95% CI] | Late excess | from failed (n) "
              "| from solved | Attributed to | Solved: sampler / maximal | Failed at deadline after a contradiction |",
              "|---|---|---|---|---|---|---|---|---|"]
    for fam, c in summary["families"].items():
        k, d = c["contrasts"].get(name), c["decomposition"]
        if k is None or "excess" not in d:
            continue
        lines.append(f"| {fam} | {_ci(k['did'])} | {_ci(k['success_late_difference'])} | {_f(d['excess'], 3)} | "
                     f"{_f(d['from_failed'], 3)} ({d['failed_episodes']} of {d['late_episodes']}) | "
                     f"{_f(d['from_solved'], 3)} | {d['attribution'] or '-'} | "
                     f"{_f(d['solved_ratio'][0])} / {_f(d['solved_ratio'][1])} | "
                     f"{d['failed_at_deadline_after_contradiction']} of {d['failed_episodes']} |")
    lines += ["", "Pooled cost ratios (steps / optimal steps), success and omitted candidates:", "",
              "| Family | Arm | Runs | Episodes 1-8 | Episodes 9-16 | All | Late by seed | Success (late) "
              "| Omitted (late) | Bounds from |", "|---|---|---|---|---|---|---|---|---|---|"]
    for fam, c in summary["families"].items():
        for s, a in c["arms"].items():
            inc = f" ({len(a['incomplete_runs'])} incomplete)" if a["incomplete_runs"] else ""
            src = c["bounds_from"] if s != ARM and c["bounds_from"] else fam
            seeds = " / ".join(_f(v) for _, v in sorted(a["cost_ratio_late_by_seed"].items()))
            lines.append(f"| {fam} | `{s}` | {a['runs']}{inc} | {_f(a['cost_ratio_early'])} | "
                         f"{_f(a['cost_ratio_late'])} | {_f(a['cost_ratio_all'])} | {seeds} | "
                         f"{_f(a['success_all'])} ({_f(a['success_late'])}) | {_f(a['omitted_fraction_late'])} | "
                         f"{src} |")
    lines += ["", "Late failure of `focused_sample` by goal depth:", "",
              "| Family | " + " | ".join(f"depth {d}" for d in range(1, 6)) + " |", "|---" * 6 + "|"]
    for fam, c in summary["families"].items():
        fd = c["failure_by_depth"]
        cells = [f"{_f(fd[str(d)]['failure_late'])} (n={fd[str(d)]['n_late']})" if str(d) in fd else "-"
                 for d in range(1, 6)]
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    if summary["slack_contrasts"]:
        lines += ["", "Slack contrasts, paired per world (looser minus base; same worlds and tasks):", "",
                  "| Contrast | Worlds | Change of the late difference [95% CI] | Lower in | Reading "
                  "| Change of the early difference | Change of late success |", "|---|---|---|---|---|---|---|"]
        for label, k in summary["slack_contrasts"].items():
            if not k["paired"]:
                t = k["tasks"]
                lines.append(f"| {label} | - | not paired ({t['mismatched']} mismatched, {t['missing_episodes']} "
                             f"missing episodes) | - | {k['reading']} | - | - |")
                continue
            lines.append(f"| {label} | {k['n_units']} | {_ci(k)} | {k['units_lower']} of {k['n_units']} | "
                         f"{k['reading']} | {_ci(k['early_change'])} | {_ci(k['success_late_change'])} |")
    dh = summary["depth_against_headroom"]
    lines += ["", f"Primary family {summary['primary_family']}: {summary['primary_reading']} "
                  f"(study R criterion: {summary['primary_verdict']}).",
              f"Depth against headroom: D {dh['D']}, DH {dh['DH']} (F2-A {dh['F2-A']}): {dh['reading']}."]
    lines += [f"Caveat: {c}." for c in summary["caveats"]]
    for fam, b in summary["bounds_borrowed"].items():
        checks = "; ".join(f"{s}: {c['shared_episodes']} shared, {c['mismatched']} mismatched, "
                           f"{c['missing_episodes']} missing episodes" + "".join(f", {p}" for p in c["problems"])
                           for s, c in b["checks"].items())
        lines.append(f"{fam} uses the bounds of {b['base']}: {', '.join(b['used']) or 'none'} "
                     f"({checks or 'no base runs'}).")
    if summary["reruns_used"]:
        lines.append("Incomplete runs replaced by their reruns: " + ", ".join(
            f"{x['incomplete']} by {x['rerun']}" for x in summary["reruns_used"]) + ".")
    if summary["ignored"]:
        lines.append(f"Runs ignored (a run for the same family, arm and seed counts): {', '.join(summary['ignored'])}.")
    return "\n".join(lines) + "\n"


def families_markdown(stats: dict) -> str:
    cols = [("optimal length, mean (range)", lambda s: _mr(s["optimal_length"])),
            ("hidden recipes per task, mean (range)", lambda s: _mr(s["hidden_recipes"])),
            ("alternative recipes per task", lambda s: _f(s["alternative_recipes"])),
            ("hypothesis entropy per task, bits", lambda s: _f(s["entropy_bits"], 1)),
            ("crafts in the optimal plan", lambda s: _f(s["plan_crafts"])),
            ("distinct recipes per world over 16 episodes", lambda s: _f(s["distinct_recipes_per_world"], 1)),
            ("recurrence, episodes 2-16 (9-16)", lambda s: f"{_f(s['recurrence'])} ({_f(s['recurrence_late'])})"),
            ("headroom: brute-force plan / optimum, per-task mean (pooled)",
             lambda s: f"{_f(s['headroom'])} ({_f(s['headroom_pooled'])})"),
            ("budget, mean (range)", lambda s: _mr(s["budget"])),
            ("budget / optimum, pooled", lambda s: _f(s["budget_over_optimal_pooled"])),
            ("W2 guarantee (smallest budget minus brute-force length)",
             lambda s: f"{'holds' if s['guarantee']['holds'] else 'FAILS'} "
                       f"({s['guarantee']['min_budget_minus_brute_force']})")]
    fams = list(stats)
    lines = ["| | " + " | ".join(fams) + " |", "|---" * (len(fams) + 1) + "|"]
    for label, fn in cols:
        lines.append(f"| {label} | " + " | ".join(fn(stats[f]) for f in fams) + " |")
    return "\n".join(lines) + "\n"


def _mr(v: dict) -> str:
    return f"{v['mean']:.1f} ({v['min']}-{v['max']})"


def figure(summary: dict, path: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = list(summary["families"].items())
    if not cells:
        return None
    cols = min(4, len(cells))
    rows = math.ceil(len(cells) / cols)
    colors = {REFERENCE: "tab:purple", BOUND: "tab:red", ARM: "tab:olive"}
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    for ax, (fam, c) in zip(axes.flat, cells):
        for s, a in c["arms"].items():
            eps = sorted(a["by_episode"], key=int)
            ax.plot([int(e) for e in eps], [a["by_episode"][e]["cost_ratio"] for e in eps], marker="o",
                    markersize=3, color=colors.get(s), label=f"{s} (episodes={a['episodes']})")
        ax.axvline(LATE_FROM + 0.5, color="gray", lw=0.8, ls="--")
        k = c["contrasts"].get(f"{ARM} - {BOUND}", {})
        ax.set_title(f"{fam}: {c['kind']} ({k.get('n_units', 0)} paired worlds)", fontsize=8)
        ax.set_xlabel("episode index within a world")
        ax.set_ylabel("primitive steps / optimal steps")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def write_report(runs_dir: str | Path, prefix: str, out_dir: str | Path, expected_units: int | None = None,
                 n_boot: int = R.N_BOOT, make_figure: bool = True) -> dict:
    runs = load_runs(runs_dir, prefix)
    summary = analyze(runs, expected_units, n_boot)
    summary["source"] = {"runs_dir": str(runs_dir), "prefix": prefix, "runs": sorted(r["run_id"] for r in runs)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if make_figure:
        summary["figure"] = figure(summary, out / "boundary_cost_by_episode.png")
    (out / "boundary_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (out / "boundary_summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def write_families(config_paths, out_dir: str | Path | None = None) -> dict:
    """Family statistics of the given configs (generator only), keyed by
    family; two configs of the same family are refused."""
    from ..config import load_config
    stats = {}
    for p in config_paths:
        cfg = load_config(p)
        fam = family_label(cfg["run"]["name"])
        if fam in stats:
            raise ValueError(f"family {fam} given twice: {stats[fam]['config']} and {p}")
        stats[fam] = {"config": str(p), **family_statistics(cfg)}
    order = {f: i for i, f in enumerate(KINDS)}
    stats = dict(sorted(stats.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "boundary_families.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        (out / "boundary_families.md").write_text(families_markdown(stats), encoding="utf-8")
    return stats
