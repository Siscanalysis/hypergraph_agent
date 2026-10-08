"""Study R plumbing: noise per evaluation variant, worlds per seed, the
privileged reference under noise, independence from study M's agents, and the
endpoint analysis. Run records in the analysis tests are invented fixtures and
never appear as measured results."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hypergraph_agent.config import eval_stream_config, load_config
from hypergraph_agent.envs.generator import TaskStream
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import reference_solve
from hypergraph_agent.evaluation import walkers as W
from hypergraph_agent.evaluation.replication import (
    analyze, break_even, episode_rank, load_runs, write_report,
)
from hypergraph_agent.evaluation.stats import cluster_bootstrap_mean, sign_test
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.budget import SessionLedger

from helpers import meter

ROOT = Path(__file__).resolve().parents[1]


def small_cfg(tmp_path, **walker):
    raw = {
        "run": {"name": "t", "phase": "unit", "allocation": "dev", "seeds": [0, 1]},
        "train": {"world_mode": "pool", "n_worlds": 2,
                  "tasks": {"profile": "unknown_prerequisites", "depth_min": 1, "depth_max": 2,
                            "n_distractor_base": 0, "observe_items": "goal_only"}},
        "eval": {"namespace": "unit", "n_tasks": 4, "track": "inference", "n_worlds": 2, "episodes_per_world": 2},
        "walker": {"per_run_cap": 400, "eval_world_offset_by_seed": True, "consistent_moves": 5,
                   "eval_variants": [{"name": "eps0", "tasks": {"failure_prob": 0.0}},
                                     {"name": "eps10", "tasks": {"failure_prob": 0.1}}], **walker},
        "arms": [{"id": "focused_sample", "strategy": "focused_sample"},
                 {"id": "random_omit", "strategy": "random_omit"},
                 {"id": "maximal", "strategy": "maximal"}, {"id": "reference", "strategy": "reference"}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_walkers_get_the_failure_probability_of_the_stream_they_play(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path, omit_prob=0.5)
    seen = []
    make = W.make_walker

    def spy(strategy, wc, seed_key, epsilon, policy=None):
        seen.append((strategy, seed_key[2], epsilon))
        return make(strategy, wc, seed_key, epsilon, policy)

    monkeypatch.setattr(W, "make_walker", spy)
    runs = tmp_path / "runs"
    results = W.run_walker_study(cfg, SessionLedger(None, allocations={"dev": 50_000}), str(runs))
    assert len(results) == 2 * 2 * 4
    assert {(s, v): e for s, v, e in seen} == {(s, v): {"eps0": 0.0, "eps10": 0.1}[v]
                                               for s in ("focused_sample", "random_omit", "maximal")
                                               for v in ("eps0", "eps10")}
    for d in runs.iterdir():
        rows = [json.loads(x) for x in (d / "eval.jsonl").read_text().splitlines()]
        assert {r["failure_prob"] for r in rows} == {json.loads((d / "config.json").read_text())
                                                     ["eval"]["tasks"]["failure_prob"]}


def test_each_seed_plays_its_own_worlds_and_noise_levels_share_them(tmp_path):
    cfg = small_cfg(tmp_path)
    v0, v10 = cfg["walker"]["eval_variants"]
    keys = {}
    for seed in (0, 1):
        for v in (v0, v10):
            vcfg = W.variant_config(cfg, seed, v)
            assert vcfg["eval"]["base_seed"] == cfg["eval"]["base_seed"] + 1000 * seed
            s = TaskStream(eval_stream_config(vcfg))
            keys[seed, v["name"]] = [(s.task(i).world.world_key, s.task(i).task_key) for i in range(4)]
    assert keys[0, "eps0"] == keys[0, "eps10"] and keys[1, "eps0"] == keys[1, "eps10"]
    assert not {w for w, _ in keys[0, "eps0"]} & {w for w, _ in keys[1, "eps0"]}
    cfg["walker"]["eval_world_offset_by_seed"] = False
    assert W.variant_config(cfg, 1, v0)["eval"]["base_seed"] == cfg["eval"]["base_seed"]


def test_stream_budget_bounds_every_run(tmp_path):
    cfg = small_cfg(tmp_path)
    v = cfg["walker"]["eval_variants"][1]
    stream_cfg = eval_stream_config(W.variant_config(cfg, 0, v))
    s = TaskStream(stream_cfg)
    assert W.stream_budget(cfg, 0, v) == sum(s.task(i).budget for i in range(4))


def test_replication_configs_share_one_ledger_and_cover_their_streams():
    paths = sorted((ROOT / "configs" / "replication").glob("*.yaml"))
    assert len(paths) >= 9
    ledgers = set()
    for p in paths:
        cfg = load_config(p)
        b = cfg["budget"]
        ledgers.add((cfg["run"]["ledger"], json.dumps(b["ledger_allocations"], sort_keys=True),
                     b["ledger_adaptive_cap"], b["ledger_reporting_cap"]))
        assert cfg["run"]["allocation"] == ("dev" if p.stem.startswith("dev") else "rep")
        assert (cfg["eval"]["namespace"] == "replication_dev") == p.stem.startswith("dev_f")
        if not p.stem.startswith("dev"):
            assert cfg["eval"]["namespace"] == f"rep_{p.stem[:2]}"
    assert len(ledgers) == 1
    for fam in ("f0", "f1", "f2"):  # noise variants share budgets; bounds play the walkers' streams
        cfg, bounds = (load_config(ROOT / "configs" / "replication" / f"{n}.yaml") for n in (fam, f"{fam}_bounds"))
        for key in ("world", "train", "eval"):
            assert cfg[key] == bounds[key]
        assert cfg["walker"]["per_run_cap"] == bounds["walker"]["per_run_cap"]
        assert cfg["walker"]["eval_variants"] == bounds["walker"]["eval_variants"]
        for seed in cfg["run"]["seeds"]:
            assert W.stream_budget(cfg, seed, cfg["walker"]["eval_variants"][0]) <= cfg["walker"]["per_run_cap"]


@pytest.mark.parametrize("eps", [0.0, 0.3])
def test_reference_replans_from_the_true_state(eps):
    from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStreamConfig, WorldConfig
    s = TaskStream(TaskStreamConfig("val", 3, "pool", 2, 0, WorldConfig(),
                                    TaskConfig(profile=PROFILE_UNKNOWN, depth_min=2, depth_max=3, n_distractor_base=0,
                                               failure_prob=eps, budget_factor=3.0)))
    env, m, failed = RecipeQuestEnv(), meter(), 0
    for i in range(12):
        t = s.task(i)
        r = W.reference_episode(Executor(env, m, "reporting_eval", torch.Generator()), t, i)
        assert r["success"]
        if eps == 0:
            assert r["replans"] == 0 and r["primitive_length"] == reference_solve(t).length
        else:  # every failed step is retried at once, so each replan adds exactly one step
            assert r["primitive_length"] == reference_solve(t).length + r["replans"]
            failed += r["replans"]
    assert eps == 0 or failed > 0


def test_reference_solve_from_a_partial_state():
    from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStreamConfig, WorldConfig
    s = TaskStream(TaskStreamConfig("val", 3, "pool", 2, 0, WorldConfig(),
                                    TaskConfig(profile=PROFILE_UNKNOWN, depth_min=2, depth_max=3, n_distractor_base=0)))
    t = s.task(0)
    full = reference_solve(t)
    assert reference_solve(t, held=t.initial_true) == full
    env = RecipeQuestEnv()
    env.reset(seed=0, options={"task": t})
    idx = env.public_spec.action_index()
    for k, key in enumerate(full.plan[:-1]):
        env.step(idx[key])
        held = {f for f, p in env._present.items() if p}
        assert reference_solve(t, held=held).length == full.length - k - 1


def test_sign_test_and_cluster_bootstrap():
    assert sign_test([-1, -2, -3, -4, -5])["p_two_sided"] == pytest.approx(2 / 32)
    assert sign_test([-1, 1, 0]) == {"negative": 1, "positive": 1, "zero": 1, "p_two_sided": 1.0}
    a = cluster_bootstrap_mean([-0.3, -0.1, -0.2, 0.05], n_boot=2000, seed=1)
    assert a == cluster_bootstrap_mean([-0.3, -0.1, -0.2, 0.05], n_boot=2000, seed=1)
    assert a["ci"][0] <= a["estimate"] <= a["ci"][1] and a["n"] == 4
    assert cluster_bootstrap_mean([])["estimate"] is None


def test_run_summaries_of_fresh_world_streams_use_block_positions():
    rows = [{"order": k, "world_key": f"T{k}", "world_episode": 0, "primitive_length": 20 if k % 16 < 8 else 12,
             "reference_length": 10, "success": True, "status": "success", "manager_decisions": 1,
             "noop_or_invalid": 0} for k in range(32)]
    s = W._summarize_rows(rows, 16)
    assert s["half_split_episode"] == 8 and s["cost_ratio_early"] == pytest.approx(2.0)
    assert s["cost_ratio_late"] == pytest.approx(1.2) and set(s["cost_ratio_late_by_world"]) == {"block0", "block1"}
    assert len(s["by_world_episode"]) == 16
    assert W._summarize_rows(rows)["half_split_episode"] == 0  # without blocks: one episode per world


# ------------------------------------------------------------ independence from study M

BLOCK_MARKOV = r'''
import importlib.abc, importlib.util, json, sys, tempfile
mode = sys.argv[1]


class Block(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != "hypergraph_agent.agents.markov":
            return None
        if mode == "absent":
            raise ModuleNotFoundError(name)
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("broken module")


sys.meta_path.insert(0, Block())
from hypergraph_agent.config import load_config
from hypergraph_agent.evaluation import walkers as W
from hypergraph_agent.evaluation.replication import analyze
from hypergraph_agent.training.budget import SessionLedger
from hypergraph_agent.walk import plan

cfg = load_config(sys.argv[2])
cfg["eval"]["n_tasks"], cfg["eval"]["n_worlds"], cfg["eval"]["episodes_per_world"] = 2, 1, 2
cfg["run"]["seeds"] = [0]
plan(cfg)
res = W.run_walker_study(cfg, SessionLedger(None, allocations={"dev": 10_000}), tempfile.mkdtemp())
try:
    W.make_walker("not_a_walker", cfg["walker"], (0, "x", "y"), 0.0)
    other = None
except Exception as exc:
    other = type(exc).__name__
print(json.dumps({"runs": len(res), "markov_loaded": sys.modules.get("hypergraph_agent.agents.markov") is not None,
                  "other": other}))
'''


@pytest.mark.parametrize("mode", ["absent", "broken"])
def test_study_r_runs_without_the_markov_agents(mode, tmp_path):
    script = tmp_path / "block.py"
    script.write_text(BLOCK_MARKOV)
    out = subprocess.run([sys.executable, str(script), mode, str(ROOT / "configs" / "replication" / "dev_f1.yaml")],
                         capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"runs": 8, "markov_loaded": False,
                   "other": "ModuleNotFoundError" if mode == "absent" else "RuntimeError"}


def test_markov_strategy_names_never_shadow_walker_strategies():
    try:
        from hypergraph_agent.agents.markov import MARKOV_STRATEGIES
    except Exception as exc:  # study M's module is optional for study R
        pytest.skip(f"agents.markov not importable: {exc!r}")
    from hypergraph_agent.agents.walker import STRATEGIES
    assert not set(MARKOV_STRATEGIES) & set(STRATEGIES)


# ------------------------------------------------------------ analysis fixtures

def fake_run(root, name, strategy, seed, eps, steps, n_units=3, per_task=False, complete=True,
             start="2026-10-08T10", config_hash="h", run_id=None, success=lambda u, e: True):
    """``steps(unit, position)`` gives the primitive steps of an episode whose optimum is 10."""
    run_id = run_id or f"{name}-{strategy}-eps{eps}-s{seed}-x"
    d = root / run_id
    d.mkdir()
    rows = []
    for u in range(n_units):
        for e in range(16):
            k = u * 16 + e
            rows.append({"order": k, "world_key": f"T{k}" if per_task else f"W{seed}{u}",
                         "world_episode": 0 if per_task else e, "primitive_length": steps(u, e),
                         "reference_length": 10, "success": bool(success(u, e))})
    if not complete:
        rows = rows[:-3]
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = {"run": {"name": name}, "arm": {"id": strategy, "strategy": strategy}, "variant": {"name": f"eps{eps}"},
           "eval": {"tasks": {"failure_prob": eps}, "n_tasks": 16 * n_units, "episodes_per_world": 16,
                    "world_mode": "per_task" if per_task else None},
           "train": {"tasks": {"failure_prob": 0.0}, "world_mode": "pool"}}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "seed": seed, "status": "completed" if complete else "running",
        "stop_reason": "tasks_done" if complete else None, "start_time": start, "config_hash": config_hash,
        "source": {"commit": "c0", "loaded_code_sha256": "s0"}}))


def amortizing(u, e):  # 2.0 early, then 1.4 (plus 0.1 x unit) late
    return 20 if e < 8 else 14 + u


def test_paired_world_analysis(tmp_path):
    for seed in (0, 1):
        fake_run(tmp_path, "rep-f1", "focused_sample", seed, 0.0, amortizing, success=lambda u, e: e >= 8 or u)
        fake_run(tmp_path, "rep-f1-bounds", "maximal", seed, 0.0, lambda u, e: 17)
        fake_run(tmp_path, "rep-f1", "random_omit", seed, 0.0, lambda u, e: 19, success=lambda u, e: e % 2)
        fake_run(tmp_path, "rep-f1-bounds", "reference", seed, 0.0, lambda u, e: 10)
    fake_run(tmp_path, "rdev-f1", "focused_sample", 0, 0.0, amortizing)  # another prefix: ignored
    out = analyze(load_runs(tmp_path, "rep-"), expected_units=6, n_boot=500)
    c = out["cells"]["F1 eps=0"]
    k = c["contrasts"]["focused_sample - maximal"]
    assert c["role"] == "primary" and c["units"] == "worlds" and k["n_units"] == 6
    assert k["estimate"] == pytest.approx(np.mean([1.4 + u / 10 - 1.7 for u in (0, 1, 2)]))
    assert k["ci"][1] < 0 and k["verdict"] == "pass" and k["units_lower"] == 6
    # difference-in-differences: (1.4 + u/10 - 1.7) - (2.0 - 1.7)
    assert k["did"]["estimate"] == pytest.approx(np.mean([u / 10 - 0.6 for u in (0, 1, 2)]))
    assert k["did"]["verdict"] == "pass" and k["success_late_difference"]["estimate"] == 0.0
    r = c["contrasts"]["focused_sample - random_omit"]
    assert r["verdict"] == "pass" and r["success_late_difference"]["estimate"] == pytest.approx(0.5)
    assert c["break_even_episode"] == 9 and c["verdict"] == "pass"
    assert c["arms"]["focused_sample"]["cost_ratio_early"] == pytest.approx(2.0)
    assert out["primary_verdict"] == "pass" and not out["general"]  # F2 cells are missing
    assert analyze(load_runs(tmp_path, "rep-"), expected_units=7, n_boot=100)["primary_verdict"] == "incomplete"


def test_only_incomplete_runs_are_replaced_and_only_by_identical_reruns(tmp_path):
    fake_run(tmp_path, "rep-f2-bounds", "maximal", 0, 0.1, lambda u, e: 17)
    fake_run(tmp_path, "rep-f2", "focused_sample", 0, 0.1, amortizing, complete=False, run_id="rep-fs-a")
    out = analyze(load_runs(tmp_path, "rep-"), n_boot=100)
    assert out["cells"]["F2 eps=0.1"]["verdict"] == "incomplete" and not out["reruns_used"]
    # a rerun with another configuration does not count: still incomplete
    fake_run(tmp_path, "rep-f2", "focused_sample", 0, 0.1, amortizing, start="2026-10-08T11", config_hash="other",
             run_id="rep-fs-b")
    out = analyze(load_runs(tmp_path, "rep-"), n_boot=100)
    assert out["cells"]["F2 eps=0.1"]["verdict"] == "incomplete" and out["ignored"] == ["rep-fs-b"]
    # an identical rerun replaces the incomplete run; later repeats are ignored
    fake_run(tmp_path, "rep-f2", "focused_sample", 0, 0.1, amortizing, start="2026-10-08T12", run_id="rep-fs-c")
    fake_run(tmp_path, "rep-f2", "focused_sample", 0, 0.1, lambda u, e: 30, start="2026-10-08T13", run_id="rep-fs-d")
    out = analyze(load_runs(tmp_path, "rep-"), n_boot=100)
    assert out["reruns_used"] == [{"incomplete": "rep-fs-a", "rerun": "rep-fs-c"}]
    assert sorted(out["ignored"]) == ["rep-fs-b", "rep-fs-d"]
    assert out["cells"]["F2 eps=0.1"]["verdict"] == "pass"
    # a completed run is never replaced
    fake_run(tmp_path, "rep-f2-bounds", "maximal", 0, 0.1, lambda u, e: 1, start="2026-10-09T10", run_id="rep-mx-e")
    out = analyze(load_runs(tmp_path, "rep-"), n_boot=100)
    assert "rep-mx-e" in out["ignored"] and out["cells"]["F2 eps=0.1"]["verdict"] == "pass"


def test_no_recurrence_control_and_amortization_check(tmp_path):
    for eps in (0.0, 0.1):
        fake_run(tmp_path, "rep-f0", "focused_sample", 0, eps, lambda u, e: 19 - (e < 8) * (u % 2), per_task=True)
        fake_run(tmp_path, "rep-f0-bounds", "maximal", 0, eps, lambda u, e: 17, per_task=True)
        fake_run(tmp_path, "rep-f1", "focused_sample", 0, eps, amortizing)
        fake_run(tmp_path, "rep-f1-bounds", "maximal", 0, eps, lambda u, e: 17)
    out = analyze(load_runs(tmp_path, "rep-"), n_boot=200)
    c = out["cells"]["F0 eps=0"]
    k = c["contrasts"]["focused_sample - maximal"]
    assert c["role"] == "control" and c["units"] == "blocks" and k["n_units"] == 3
    assert c["verdict"] == "descriptive" and k["estimate"] == pytest.approx(0.2)
    assert k["did"]["estimate"] == pytest.approx(0.1 / 3) and k["did"]["ci"][1] >= 0
    assert c["break_even_episode"] == "never"
    assert out["amortization"]["eps=0"] == {**out["amortization"]["eps=0"], "supported": "yes",
                                            "role": "pre-registered"}
    assert out["amortization"]["eps=0.1"]["role"] == "reported"
    assert out["break_even_later_under_noise"] == {"F0": False, "F1": False}


def test_break_even_is_the_start_of_the_final_run_of_cheaper_episodes():
    def arms(fs):
        return {"focused_sample": {"by_episode": {str(e + 1): {"cost_ratio": x} for e, x in enumerate(fs)}},
                "maximal": {"by_episode": {str(e + 1): {"cost_ratio": 1.5} for e in range(len(fs))}}}

    assert break_even(arms([1.0, 2.0, 1.0, 1.5])) == 3
    assert break_even(arms([1.0, 1.0, 1.0])) == 1
    assert break_even(arms([1.0, 1.0, 2.0])) == "never"
    assert episode_rank("never") > episode_rank(16) > episode_rank(3)


def test_report_files_and_cli(tmp_path):
    from hypergraph_agent.analyze_replication import main
    runs = tmp_path / "runs"
    runs.mkdir()
    fake_run(runs, "rep-f1", "focused_sample", 0, 0.1, amortizing)
    fake_run(runs, "rep-f1-bounds", "maximal", 0, 0.1, lambda u, e: 17)
    out = tmp_path / "out"
    assert main(["--runs", str(runs), "--prefix", "rep-", "--out", str(out), "--n-boot", "200"]) == 0
    summary = json.loads((out / "replication_summary.json").read_text())
    cell = summary["cells"]["F1 eps=0.1"]
    assert cell["role"] == "secondary" and cell["verdict"] == "incomplete"  # 3 worlds; the default needs 50
    assert "F1 eps=0.1" in (out / "replication_summary.md").read_text(encoding="utf-8")
    assert (out / "replication_cost_by_episode.png").exists()
    assert write_report(runs, "none-", out, make_figure=False)["cells"] == {}
