"""Study R2 plumbing: configs (one ledger declaration, fresh namespaces, caps,
one changed factor per family, slack levels sharing worlds and tasks), the
brute-force length and the slack invariance of the bounds, the family
statistics against study R's table, and the endpoint analysis. Run records in
the analysis tests are invented fixtures and never appear as measured results."""

import json
from pathlib import Path

import pytest
import yaml

from hypergraph_agent.config import eval_stream_config, load_config
from hypergraph_agent.envs.generator import TaskStream
from hypergraph_agent.evaluation import boundary as B
from hypergraph_agent.evaluation import walkers as W
from hypergraph_agent.evaluation.boundary import (
    BASES, KINDS, analyze, brute_force_length, depth_against_headroom, excess_decomposition, family_label,
    family_statistics, load_runs, reading, write_families,
)
from hypergraph_agent.training.budget import SessionLedger

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "configs" / "boundary"
MEASURE = sorted(p for p in BOUNDARY.glob("*.yaml") if not p.name.startswith("dev_"))
DEV = sorted(BOUNDARY.glob("dev_*.yaml"))
SLACK = {"d_s6": ("d", 6), "d_s12": ("d", 12), "f2_s6": ("f2", 6), "f2_s12": ("f2", 12)}
DECLARATION = ("runs/R2-ledger.json", {"dev": 6000, "r2": 278000}, 284000, 352000)


def cfg(stem: str) -> dict:
    return load_config(BOUNDARY / f"{stem}.yaml")


def settings(c: dict) -> dict:
    """Everything that defines a family's tasks: world, task and stream settings."""
    s = eval_stream_config(c)
    task = {k: v for k, v in vars(s.task).items() if k != "failure_prob"}
    return {**{f"world.{k}": v for k, v in vars(s.world).items()}, **{f"task.{k}": v for k, v in task.items()},
            "world_mode": s.world_mode, "n_worlds": s.n_worlds, "goal_levels": s.goal_levels}


def changed(a: dict, b: dict) -> set:
    ga, gb = settings(a), settings(b)
    return {k for k in ga if ga[k] != gb[k]}


# ------------------------------------------------------------------ configs

def test_one_ledger_declaration_and_allocations():
    assert len(MEASURE) == 13 and len(DEV) == 13
    ledgers = set()
    for p in MEASURE + DEV:
        c = load_config(p)
        b = c["budget"]
        ledgers.add((c["run"]["ledger"], json.dumps(b["ledger_allocations"], sort_keys=True),
                     b["ledger_adaptive_cap"], b["ledger_reporting_cap"]))
        dev = p.name.startswith("dev_")
        assert c["run"]["allocation"] == ("dev" if dev else "r2")
        assert c["run"]["name"].startswith("r2dev-" if dev else "r2-")
        assert c["run"]["seeds"] == ([0] if dev else [0, 1, 2, 3, 4])
        assert c["walker"]["eval_world_offset_by_seed"] and c["walker"]["consistent_moves"] == 50
        assert c["walker"]["eval_variants"] == [{"name": "eps0", "tasks": {"failure_prob": 0.0}}]
        arms = [a["strategy"] for a in c["arms"]]
        slack = p.stem.removeprefix("dev_") in SLACK
        assert arms == (["focused_sample"] if slack else ["focused_sample", "maximal", "reference"])
    path, alloc, adaptive, reporting = DECLARATION
    assert ledgers == {(path, json.dumps(alloc, sort_keys=True), adaptive, reporting)}


def test_caps_are_the_largest_stream_budget_sums_and_the_ledger_covers_every_run():
    adaptive = {"dev": 0, "r2": 0}
    reporting = {"dev": 0, "r2": 0}
    for p in MEASURE + DEV:
        c = load_config(p)
        v = c["walker"]["eval_variants"][0]
        sums = [W.stream_budget(c, s, v) for s in c["run"]["seeds"]]
        assert c["walker"]["per_run_cap"] == max(sums)
        adaptive[c["run"]["allocation"]] += sum(sums)
        reporting[c["run"]["allocation"]] += (len(c["arms"]) - 1) * sum(sums)
    b = cfg("d")["budget"]
    assert all(b["ledger_allocations"][k] >= n for k, n in adaptive.items())
    assert b["ledger_adaptive_cap"] == sum(b["ledger_allocations"].values())
    assert b["ledger_reporting_cap"] >= sum(reporting.values())
    assert adaptive["dev"] + reporting["dev"] < 15_000  # development stays modest even in the worst case


def test_namespaces_are_fresh_and_shared_only_by_slack_levels():
    others = set()
    for p in (ROOT / "configs").rglob("*.yaml"):
        if p.parent != BOUNDARY:
            raw = yaml.safe_load(p.read_text()) or {}
            ns = (raw.get("eval") or {}).get("namespace") if isinstance(raw, dict) else None
            if ns:
                others.add(ns)
    spaces = {p.stem: cfg(p.stem)["eval"]["namespace"] for p in MEASURE}
    assert not set(spaces.values()) & others and all(n.startswith("r2_") for n in spaces.values())
    for stem, (base, _) in SLACK.items():
        assert spaces[stem] == spaces[base]
    assert len(set(spaces.values())) == len(MEASURE) - len(SLACK)
    assert {cfg(p.stem)["eval"]["namespace"] for p in DEV} == {"r2_dev"}
    assert {family_label(cfg(p.stem)["run"]["name"]) for p in MEASURE} == set(KINDS)


def test_each_family_changes_one_factor():
    f1, f2 = (load_config(ROOT / "configs" / "replication" / f"{n}.yaml") for n in ("f1", "f2"))
    one_from_f1 = {"d": {"task.depth_min", "task.depth_max"}, "a": {"world.p_alternative"},
                   "p": {"world.pool_size"}, "i": {"world.p_extra_item_input"}}
    one_from_f2 = {"f2_minus_d": {"task.depth_min", "task.depth_max"}, "f2_minus_a": {"world.p_alternative"},
                   "f2_minus_p": {"world.pool_size"}, "f2": set()}
    for stem, diff in one_from_f1.items():
        assert changed(cfg(stem), f1) == diff
    for stem, diff in one_from_f2.items():
        assert changed(cfg(stem), f2) == diff
    assert changed(cfg("dh"), cfg("d")) == {"world.pool_size"} and settings(cfg("dh"))["world.pool_size"] == 8
    for stem, (base, slack) in SLACK.items():
        assert changed(cfg(stem), cfg(base)) == {"task.budget_slack"}
        assert eval_stream_config(cfg(stem)).task.budget_slack == slack


def test_dev_configs_mirror_the_measurement_configs():
    for p in DEV:
        d, m = cfg(p.stem), cfg(p.stem.removeprefix("dev_"))
        assert changed(d, m) == {"n_worlds"} and d["arms"] == m["arms"]
        assert d["run"]["name"] == m["run"]["name"].replace("r2-", "r2dev-", 1)
        assert (d["eval"]["n_worlds"], d["eval"]["n_tasks"], d["eval"]["episodes_per_world"]) == (1, 16, 16)


def test_slack_levels_pair_with_their_base_by_world_and_task():
    for stem, (base, slack) in SLACK.items():
        a, b = cfg(stem), cfg(base)
        assert [x["id"] for x in a["arms"]] == ["focused_sample"]
        assert a["walker"]["eval_variants"] == b["walker"]["eval_variants"]
        for seed in (0, 3):
            v = a["walker"]["eval_variants"][0]
            sa, sb = (eval_stream_config(W.variant_config(c, seed, v)) for c in (a, b))
            ta, tb = TaskStream(sa), TaskStream(sb)
            for i in W.eval_order(sa, 160, 16)[:24]:
                x, y = ta.task(i), tb.task(i)
                assert (x.world.world_key, x.task_key, x.goal, x.rules) == (y.world.world_key, y.task_key, y.goal,
                                                                           y.rules)
                assert x.budget - y.budget == slack - 2


# ------------------------------------------------------------------ bounds and statistics

def small(stem: str, slack: int | None = None, arms=("maximal", "reference")) -> dict:
    c = cfg(f"dev_{stem}")
    c["eval"]["n_tasks"], c["eval"]["episodes_per_world"] = 6, 6
    if slack is not None:
        c["train"]["tasks"]["budget_slack"] = slack
    c["arms"] = [{**c["arms"][0], "id": a, "strategy": a} for a in arms]
    return c


def rows_of(runs: Path) -> dict:
    out = {}
    for d in runs.iterdir():
        arm = json.loads((d / "config.json").read_text())["arm"]["id"]
        out[arm] = [json.loads(x) for x in (d / "eval.jsonl").read_text().splitlines()]
    return out


@pytest.mark.parametrize("stem", ["d", "f2"])
def test_bounds_do_not_depend_on_the_slack_and_maximal_runs_the_brute_force_plan(stem, tmp_path):
    played = {}
    for slack in (2, 12):
        c = small(stem, slack)
        W.run_walker_study(c, SessionLedger(None, allocations={"dev": 10_000}), str(tmp_path / f"s{slack}"))
        played[slack] = rows_of(tmp_path / f"s{slack}")
    drop = ("t", "budget")
    for arm in ("maximal", "reference"):
        a, b = played[2][arm], played[12][arm]
        assert [{k: v for k, v in r.items() if k not in drop} for r in a] == \
            [{k: v for k, v in r.items() if k not in drop} for r in b]
        assert all(r["success"] for r in a) and [r["budget"] + 10 for r in a] == [r["budget"] for r in b]
    c = small(stem)
    s = TaskStream(eval_stream_config(W.variant_config(c, 0, c["walker"]["eval_variants"][0])))
    assert [brute_force_length(s.task(r["stream_index"])) for r in played[2]["maximal"]] == \
        [r["primitive_length"] for r in played[2]["maximal"]]


@pytest.mark.parametrize("fam, table", [
    ("f1", {"hidden": (3.7, 2, 9), "alternatives": 0.57, "entropy": 19.7, "distinct": 8.9,
            "recurrence": (0.91, 0.98), "headroom": (1.91, 1.83), "budget": (19.4, 12, 27)}),
    ("f2", {"hidden": (10.1, 4, 18), "alternatives": 3.45, "entropy": 65.6, "distinct": 17.5,
            "recurrence": (0.95, 0.99), "headroom": (1.88, 1.84), "budget": (27.4, 22, 31)}),
])
def test_family_statistics_reproduce_study_r_table(fam, table):
    s = family_statistics(load_config(ROOT / "configs" / "replication" / f"{fam}.yaml"))
    assert (s["tasks"], s["units"]) == (800, 50)
    h, b = s["hidden_recipes"], s["budget"]
    assert (round(h["mean"], 1), h["min"], h["max"]) == table["hidden"]
    assert round(s["alternative_recipes"], 2) == table["alternatives"]
    assert round(s["entropy_bits"], 1) == table["entropy"]
    assert round(s["distinct_recipes_per_world"], 1) == table["distinct"]
    assert (round(s["recurrence"], 2), round(s["recurrence_late"], 2)) == table["recurrence"]
    assert (round(s["headroom"], 2), round(s["headroom_pooled"], 2)) == table["headroom"]
    assert (round(b["mean"], 1), b["min"], b["max"]) == table["budget"]
    assert s["guarantee"]["holds"] and s["guarantee"]["brute_force_exact"]


def test_every_family_keeps_the_brute_force_guarantee(tmp_path):
    stats = write_families(MEASURE, tmp_path)
    assert set(stats) == set(KINDS)
    for fam, s in stats.items():
        g = s["guarantee"]
        assert g["holds"] and g["brute_force_exact"] and g["tasks"] == 800 and g["min_budget_minus_brute_force"] >= 3
    for fam, base in BASES.items():  # slack levels: the base family's structure, a larger budget
        a, b = stats[fam], stats[base]
        assert a["entropy_bits"] == b["entropy_bits"] and a["headroom"] == b["headroom"]
        assert a["budget"]["mean"] > b["budget"]["mean"]
    # DH restores F1's brute-force headroom at D's depth
    assert abs(stats["DH"]["headroom_pooled"] - 1.83) < 0.05 < 1.83 - stats["D"]["headroom_pooled"]
    assert abs(stats["DH"]["plan_crafts"] - stats["D"]["plan_crafts"]) < 0.1
    assert "W2 guarantee" in (tmp_path / "boundary_families.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="given twice"):
        write_families([BOUNDARY / "dev_i.yaml", BOUNDARY / "dev_i.yaml"])


def test_a_truncated_brute_force_enumeration_breaks_the_guarantee(monkeypatch):
    c = cfg("dev_f2")
    assert family_statistics(c)["guarantee"]["holds"]
    full = B.enumerate_derivations
    monkeypatch.setattr(B, "enumerate_derivations", lambda r, h, g, cap=4096: full(r, h, g, 1))
    g = family_statistics(c)["guarantee"]
    assert not g["brute_force_exact"] and not g["holds"]


# ------------------------------------------------------------------ analysis fixtures

WORLD = {"pool_size": 6}


def fake_run(root, name, strategy, seed, steps, n_units=3, complete=True, start="2026-10-09T10", success=None,
             task=lambda seed, u, e: f"T{seed}{u}{e}", run_id=None, depth=lambda u, e: 3 + e % 3, slack=2,
             world=None, ref=lambda u, e: 10, budget=None):
    """``steps(unit, episode)`` gives the primitive steps of an episode whose optimum is ``ref(unit,
    episode)`` (10); ``success(unit, episode)`` defaults to every episode succeeding; the recorded
    budget is 25 plus the slack unless ``budget(unit, episode)`` is given."""
    run_id = run_id or f"{name}-{strategy}-eps0-s{seed}-x"
    d = root / run_id
    d.mkdir()
    rows = []
    for u in range(n_units):
        for e in range(16):
            ok = True if success is None else bool(success(u, e))
            rows.append({"order": u * 16 + e, "world_key": f"W{seed}{u}", "world_episode": e,
                         "primitive_length": steps(u, e), "reference_length": ref(u, e), "success": ok,
                         "status": "success" if ok else "deadline", "contradictions": 0 if ok else 1,
                         "task_key": task(seed, u, e), "depth": depth(u, e),
                         "budget": 25 + slack if budget is None else budget(u, e),
                         "omitted_candidates": 2, "pool_candidates": 6})
    if not complete:
        rows = rows[:-2]
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "config.json").write_text(json.dumps({
        "run": {"name": name}, "arm": {"id": strategy, "strategy": strategy}, "variant": {"name": "eps0"},
        "world": world or WORLD,
        "eval": {"tasks": {"failure_prob": 0.0}, "n_tasks": 16 * n_units, "episodes_per_world": 16,
                 "world_mode": None, "namespace": "unit", "base_seed": 10000 + 1000 * seed, "n_worlds": n_units},
        "train": {"tasks": {"failure_prob": 0.0, "depth_min": 3, "depth_max": 5, "budget_slack": slack},
                  "world_mode": "pool", "n_worlds": 8}}))
    (d / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "seed": seed, "status": "completed" if complete else "running",
        "stop_reason": "tasks_done" if complete else None, "start_time": start, "config_hash": "h",
        "source": {"commit": "c0", "loaded_code_sha256": "s0"}}))


def family(root, name, fs_steps, fs_success=None, seeds=(0, 1), bounds=True, **kw):
    for seed in seeds:
        fake_run(root, name, "focused_sample", seed, fs_steps, success=fs_success, **kw)
        if bounds:
            kw2 = {k: v for k, v in kw.items() if k != "slack"}
            fake_run(root, name, "maximal", seed, lambda u, e: 17, **kw2)
            fake_run(root, name, "reference", seed, lambda u, e: 10, **kw2)


def test_family_readings_and_late_excess_decomposition(tmp_path):
    family(tmp_path, "r2-a", lambda u, e: 20 if e < 8 else 14 + u)  # cheaper late: advantage
    # D: late episode 15 of every world fails at a budget of 30, other late episodes cost 16
    family(tmp_path, "r2-d", lambda u, e: 20 if e < 8 else (30 if e == 15 else 16 + 2 * u),
           fs_success=lambda u, e: e != 15)
    family(tmp_path, "r2-p", lambda u, e: 17 + (u - 1) * 3)  # mean 0, spread: inconclusive
    out = analyze(load_runs(tmp_path, "r2-"), expected_units=6, n_boot=500)
    fams = out["families"]
    assert list(fams) == ["D", "A", "P"] and fams["D"]["role"] == "primary"
    assert [fams[f]["reading"] for f in ("D", "A", "P")] == ["reversed", "advantage", "inconclusive"]
    assert out["primary_reading"] == "reversed" and fams["A"]["verdict"] == "pass"
    d = fams["D"]["decomposition"]
    assert d["late_episodes"] == 48 and d["failed_episodes"] == 6
    assert d["from_failed"] == pytest.approx(6 * 13 / 480)
    assert d["from_solved"] == pytest.approx(2 * 7 * (-1 + 1 + 3) / 480)
    assert d["excess"] == pytest.approx(d["from_failed"] + d["from_solved"])
    assert d["attribution"] == "failed episodes"  # 0.1625 from failed against 0.0875 from solved
    assert d["solved_ratio"][1] == pytest.approx(1.7) and d["failed_at_deadline_after_contradiction"] == 6
    assert fams["A"]["decomposition"]["attribution"] is None  # no excess
    assert fams["D"]["failure_by_depth"]["3"]["failure_late"] == pytest.approx(1 / 3)  # episode 15: depth 3
    assert fams["D"]["failure_by_depth"]["5"]["failure_late"] == pytest.approx(0.0)
    assert fams["D"]["break_even_episode"] == "never" and fams["A"]["break_even_episode"] == 9
    assert analyze(load_runs(tmp_path, "r2-"), expected_units=7, n_boot=100)["primary_reading"] == "incomplete"
    # the controls: I is missing and F2 is missing, so both caveats are raised
    assert [c.split(" ")[0] for c in out["caveats"]] == ["I", "F2"]


def test_attribution_of_an_excess():
    rows = lambda steps, ok: [{"rows": [{"_unit": "u", "_pos": e, "primitive_length": steps(e), "success": ok(e),
                                         "reference_length": 10} for e in range(16)]}]
    bound = rows(lambda e: 17, lambda e: True)
    d = excess_decomposition(rows(lambda e: 30 if e == 15 else 17, lambda e: e != 15), bound)
    assert d["from_solved"] == 0 and d["from_failed"] > 0 and d["attribution"] == "failed episodes"
    d = excess_decomposition(rows(lambda e: 30 if e == 15 else 19, lambda e: e != 15), bound)
    assert d["from_solved"] > d["from_failed"] > 0 and d["attribution"] == "solved episodes"


def test_depth_against_headroom_and_controls(tmp_path):
    assert depth_against_headroom("reversed", "reversed") == "depth matters beyond headroom"
    assert depth_against_headroom("reversed", "advantage") == "headroom, not depth"
    assert depth_against_headroom("reversed", "incomplete") == "incomplete"
    assert depth_against_headroom("reversed", "inconclusive").startswith("headroom accounts for part")
    assert depth_against_headroom("inconclusive", "reversed").startswith("D not reversed")
    family(tmp_path, "r2-d", lambda u, e: 20 + u)
    family(tmp_path, "r2-dh", lambda u, e: 14 + u)
    family(tmp_path, "r2-i", lambda u, e: 14 + u)
    family(tmp_path, "r2-f2", lambda u, e: 21 + u)
    out = analyze(load_runs(tmp_path, "r2-"), expected_units=6, n_boot=300)
    assert out["depth_against_headroom"]["reading"] == "headroom, not depth" and out["caveats"] == []
    assert list(out["families"]) == ["D", "DH", "I", "F2"]


def test_slack_families_use_their_base_bounds_only_after_the_task_check(tmp_path):
    family(tmp_path, "r2-f2", lambda u, e: 20, fs_success=lambda u, e: e % 4)
    family(tmp_path, "r2-f2s12", lambda u, e: 20 if e < 8 else 16 + u, bounds=False, slack=12)
    out = analyze(load_runs(tmp_path, "r2-"), expected_units=6, n_boot=500)
    f = out["families"]["F2S12"]
    assert f["bounds_from"] == "F2" and set(f["arms"]) == {"focused_sample", "maximal", "reference"}
    b = out["bounds_borrowed"]["F2S12"]
    assert b["used"] == ["maximal", "reference"] and b["checks"]["maximal"]["shared_episodes"] == 96
    k = out["slack_contrasts"]["F2S12 vs F2"]
    assert k["paired"] and k["n_units"] == 6 and k["reading"] == "lower"
    assert k["estimate"] == pytest.approx(sum(1.6 + u / 10 - 2.0 for u in (0, 1, 2)) / 3)
    assert k["early_change"]["estimate"] == pytest.approx(0.0)
    assert k["success_late_change"]["estimate"] == pytest.approx(0.25)
    md = B.markdown(out)
    assert "F2S12 uses the bounds of F2: maximal, reference" in md and "Late by seed" in md


@pytest.mark.parametrize("change, problem", [
    ({"task": lambda s, u, e: f"S{s}{u}{e}"}, "mismatched"),
    ({"depth": lambda u, e: 4}, "mismatched"),
    ({"ref": lambda u, e: 11}, "mismatched"),
    ({"budget": lambda u, e: 31 + (e == 3 and u == 0)}, "mismatched"),  # one budget off by one step
    ({"world": {"pool_size": 8}}, "problems"),
])
def test_borrowing_is_refused_unless_tasks_depth_optimum_budget_and_settings_agree(tmp_path, change, problem):
    family(tmp_path, "r2-f2", lambda u, e: 20)
    family(tmp_path, "r2-f2s6", lambda u, e: 18, bounds=False, slack=6, **change)
    out = analyze(load_runs(tmp_path, "r2-"), expected_units=6, n_boot=100)
    f = out["families"]["F2S6"]
    assert f["reading"] == f["verdict"] == "incomplete" and f["bounds_from"] is None
    check = out["bounds_borrowed"]["F2S6"]["checks"]["maximal"]
    assert not check["ok"] and (check[problem] if problem == "problems" else check[problem] > 0)
    assert out["slack_contrasts"]["F2S6 vs F2"]["reading"] == "incomplete"
    assert "| F2S6 |" in B.markdown(out)


def test_borrowing_is_refused_when_a_base_bounds_run_misses_episodes(tmp_path):
    family(tmp_path, "r2-d", lambda u, e: 20, bounds=False)
    for seed in (0, 1):
        fake_run(tmp_path, "r2-d", "maximal", seed, lambda u, e: 17, complete=seed == 0)
    family(tmp_path, "r2-ds6", lambda u, e: 18, bounds=False, slack=6)
    out = analyze(load_runs(tmp_path, "r2-"), n_boot=100)
    check = out["bounds_borrowed"]["DS6"]["checks"]["maximal"]
    assert check["missing_episodes"] == 2 and not check["ok"]
    assert out["families"]["DS6"]["reading"] == "incomplete" and out["families"]["D"]["reading"] == "incomplete"


def test_reading_and_decomposition_edge_cases():
    assert reading(None) == "incomplete" and reading({"verdict": "incomplete", "ci": [-1, -0.5]}) == "incomplete"
    assert reading({"verdict": "fail", "ci": [-0.1, 0.2]}) == "inconclusive"
    assert excess_decomposition([], []) == {"late_episodes": 0}


def test_report_files_and_cli(tmp_path):
    from hypergraph_agent.analyze_boundary import main
    runs = tmp_path / "runs"
    runs.mkdir()
    family(runs, "r2-d", lambda u, e: 20 if e < 8 else 14)
    family(runs, "r2dev-d", lambda u, e: 30)  # development prefix: not read with r2-
    out = tmp_path / "out"
    assert main(["endpoints", "--runs", str(runs), "--prefix", "r2-", "--out", str(out), "--n-boot", "200"]) == 0
    summary = json.loads((out / "boundary_summary.json").read_text())
    assert list(summary["families"]) == ["D"] and summary["primary_reading"] == "incomplete"  # 6 of 50 worlds
    assert "| D |" in (out / "boundary_summary.md").read_text(encoding="utf-8")
    assert (out / "boundary_cost_by_episode.png").exists()
    assert main(["endpoints", "--runs", str(runs), "--prefix", "none-", "--out", str(out), "--no-figure"]) == 1
    assert main(["families", "--configs", str(BOUNDARY / "dev_i.yaml"), "--out", str(out)]) == 0
    assert json.loads((out / "boundary_families.json").read_text())["I"]["tasks"] == 16


def test_depth_against_headroom_is_unresolved_when_dh_and_f2a_disagree():
    from hypergraph_agent.evaluation.boundary import depth_against_headroom
    assert depth_against_headroom("reversed", "reversed", "reversed") == "depth matters beyond headroom"
    assert depth_against_headroom("reversed", "reversed", "advantage").startswith("unresolved")
    assert depth_against_headroom("advantage", "reversed", "advantage").startswith("D not reversed")
