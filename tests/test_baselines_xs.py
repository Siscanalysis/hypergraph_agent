"""Study XS: potential-based shaping (public information, telescoping), inherited tuning, configs,
the combined analysis of the slack members, and that study X's hashes and paths are unchanged."""

import ast
import inspect
import json
import pathlib
import re
from dataclasses import replace

import numpy as np
import pytest
import yaml

from hypergraph_agent.baselines import config as xconfig
from hypergraph_agent.baselines import shaping as xshaping
from hypergraph_agent.baselines import xs
from hypergraph_agent.baselines.encoders import RQLayout, TTLayout
from hypergraph_agent.baselines.sources import TaskItem, evaluation_source, training_source
from hypergraph_agent.baselines.wrappers import RecipeQuestGym, TechTreeGym
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, stable_hash
from hypergraph_agent.envs.public_schema import public_view
from hypergraph_agent.envs.vocabulary import is_base
from hypergraph_agent.techtree.generator import TechStream, TechStreamConfig, TechWorldConfig
from hypergraph_agent.training.budget import BudgetMeter, SessionLedger

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "hypergraph_agent" / "baselines"
XS_CONFIGS = sorted((ROOT / "configs" / "baselines").glob("xs_*.yaml"))
STUDIES = [p for p in XS_CONFIGS if xs.is_study(p)]
MEMBERS = [p for p in XS_CONFIGS if not xs.is_study(p)]
F1_TASKS = {"profile": "unknown_prerequisites", "depth_min": 1, "depth_max": 3, "n_distractor_base": 0,
            "observe_items": "goal_only"}
RQ_SETTING = {"name": "rq_unit", "family": "recipequest", "world": {}, "tasks": F1_TASKS, "stream": {},
              "agent_study": None, "loader": None, "max_variants": 2}
U_WORLD = {"n_primitives": 4, "n_slots": 3, "n_unlocks": 1, "unlock_level": 1, "n_levels": 1,
           "concepts_per_level": [2], "combo_length": 3}
U_STREAM = {"goal_levels": [1], "goal_rule": "unlock", "episode_budget": 100,
            "unlock_visibility": "announced", "signal": False}
L_WORLD = {"n_primitives": 3, "n_levels": 3, "concepts_per_level": [3, 3, 3], "combo_length": 2,
           "length_rule": "linear", "reuse_depth": 2}
TT_L = {"name": "tt_unit", "family": "techtree", "world": L_WORLD, "tasks": {},
        "stream": {"goal_levels": [3], "episode_budget": 100}, "agent_study": "L", "loader": None,
        "max_variants": 2}
TT_U = {**TT_L, "world": U_WORLD, "stream": U_STREAM, "agent_study": "U"}
RQ_SHAPING = {"potential": "rq_goal_closure"}
TT_SHAPING = {"potential": "tt_levels"}


def meter():
    return BudgetMeter(SessionLedger(None, {"dev": 10 ** 7}, 10 ** 7, 10 ** 7), "t", "dev", 10 ** 6)


def rq_items(n_worlds=2, episodes=3, seed=0, tasks=F1_TASKS):
    src = evaluation_source("recipequest", {**RQ_SETTING, "tasks": tasks}, "xs_unit", 10_000, seed, n_worlds,
                            episodes)
    out = []
    while (b := src.next_block()) is not None:
        out += b
    return out


class ListSource:
    def __init__(self, blocks):
        self.blocks_, self.E = list(blocks), max(len(b) for b in blocks)

    def next_block(self):
        return self.blocks_.pop(0) if self.blocks_ else None


def hidden_twin(task, rng):
    """The same task with every recipe's hidden requirement redrawn inside its public pool."""
    rules = []
    for r in task.rules:
        k = int(rng.integers(1, min(3, len(r.pool)) + 1))
        drawn = rng.choice(r.pool, k, replace=False)
        rules.append(replace(r, true_base=tuple(sorted(int(b) for b in drawn))))
    world = replace(task.world, recipes=tuple(
        next((x for x in rules if x.effect == r.effect and x.signature(PROFILE_UNKNOWN)
              == r.signature(PROFILE_UNKNOWN)), r) for r in task.world.recipes))
    return replace(task, rules=tuple(rules), world=world)


# ------------------------------------------------------------- potentials
def test_closure_bases_are_the_public_pools_of_the_goal_closure():
    checked = 0
    for item in rq_items(4, 3):
        task, spec = item.task, public_view(item.task)
        closure = task.closure_items()
        want = {b for r in task.rules if r.effect in closure for b in r.pool}  # evaluator side
        got = {spec.facts[i].type_id for i in xshaping.rq_closure_bases(spec)}
        assert got == want and all(is_base(t) for t in got)
        distractors = [r for r in task.rules if r.effect not in closure]
        checked += bool(distractors)
    assert checked > 0  # tasks with a distractor recipe outside the closure were seen


def test_potentials_are_bounded_and_track_public_progress():
    items = rq_items(1, 1)
    env = RecipeQuestGym(ListSource([items]), shaping=xshaping.make_shaper(RQ_SHAPING, 0.99), meter=meter())
    env.reset()
    assert env.shaping.phi == 0.0  # nothing held at the start of an F1 task
    spec = env.env.public_spec
    bases = xshaping.rq_closure_bases(spec)
    amap = env.layout.action_map(spec)
    for b in bases:
        canon = int(np.flatnonzero(amap == next(i for i, d in enumerate(spec.actions) if d.fact == b))[0])
        env.step(canon)
    assert env.shaping.phi == pytest.approx(0.5)  # every candidate base held, goal not yet
    with pytest.raises(ValueError):
        xshaping.normalize({"potential": "rq_goal_closure", "base_weight": 0.8, "goal_weight": 0.5})
    with pytest.raises(KeyError):
        xshaping.normalize({"potential": "tt_levels", "base_weight": 0.5})
    assert xshaping.normalize(None) is None


def _rq_shaping(task, actions):
    env = RecipeQuestGym(ListSource([[TaskItem(task, task.world.world_key, 0, 0, 0)]]),
                         shaping=xshaping.make_shaper(RQ_SHAPING, 0.99), meter=meter())
    env.reset()
    out = [env.shaping.phi]
    for a in actions:
        _, r, term, _, info = env.step(a)
        out.append((r, info["shaping"]))
        if term:
            break
    return out


def test_recipequest_twins_differing_in_hidden_structure_give_identical_shaping():
    layout, rng = RQLayout(), np.random.default_rng(11)
    checked = 0
    for item in rq_items(4, 4):
        twin = hidden_twin(item.task, rng)
        if all(a.true_base == b.true_base for a, b in zip(item.task.rules, twin.rules)):
            continue
        spec = public_view(item.task)
        goal_type = spec.facts[spec.goal].type_id
        amap = layout.action_map(spec)
        crafts_goal = {layout.craft_index(goal_type, v) for v in range(2)}
        allowed = [c for c in np.flatnonzero(amap >= 0) if c not in crafts_goal and c != layout.submit]
        actions = [int(rng.choice(allowed)) for _ in range(item.task.budget)]
        a, b = _rq_shaping(item.task, actions), _rq_shaping(twin, actions)
        assert a == b and any(x[1] != 0 for x in a[1:])
        checked += 1
    assert checked >= 8


def _tt_twins():
    cfg = TechWorldConfig(**{**L_WORLD, "concepts_per_level": (3, 3, 3)})
    s0 = TechStream(TechStreamConfig("xs_unit", 0, world=replace(cfg, reuse_depth=0), goal_levels=(3,)))
    s2 = TechStream(TechStreamConfig("xs_unit", 0, world=replace(cfg, reuse_depth=2), goal_levels=(3,)))
    return s0.task(0, 0), s2.task(0, 0)


def test_techtree_twins_differing_in_hidden_structure_give_identical_shaping():
    a, b = _tt_twins()
    assert a.world.sequences != b.world.sequences
    actions = [0, 3, 1, 3, 2, 3, 2, 4, 1, 0, 3]  # single presses never complete a sequence of length >= 2
    out = []
    for t in (a, b):
        env = TechTreeGym(ListSource([[TaskItem(t, t.world.world_key, 0, 0, 0)]]), TTLayout(9, 3, 3, 6),
                          shaping=xshaping.make_shaper(TT_SHAPING, 0.99), meter=meter())
        env.reset()
        out.append([env.shaping.phi] + [env.step(x)[4]["shaping"] for x in actions])
    assert out[0] == out[1]


def test_tt_potential_weights_held_concepts_by_public_level():
    from hypergraph_agent.techtree.env import TechObservation
    from hypergraph_agent.techtree.env import public_view as tt_public_view
    a, _ = _tt_twins()
    spec = tt_public_view(a)
    pot = xshaping.TTPotential(0.5, 0.5)
    n = len(spec.concept_levels)
    lv = np.asarray(spec.concept_levels, dtype=float)
    held = tuple(i == 0 for i in range(n))
    obs = TechObservation(held, (True,) * 5, 100, 0, None, None)
    assert pot(spec, obs) == pytest.approx(0.5 * lv[0] / lv.sum() + 0.5 * float(spec.goal == 0))
    every = TechObservation((True,) * n, (True,) * 5, 100, 0, None, None)
    assert pot(spec, every) == pytest.approx(1.0)


HIDDEN = re.compile(r"true_base|\.required\(\)|reference_solve|reference_solver|debug_view|\._task\b|"
                    r"\.sequences\b|\.parents\b|\.unlocks\b|\.entries\b|\.compositional\b|revealed_links|"
                    r"needs_unlock|_present\b|_held\b|closure_items|\.world\b")


def test_shaping_module_names_no_hidden_field_and_imports_only_public_modules():
    text = (PKG / "shaping.py").read_text()
    offenders = [line for line in text.splitlines()
                 if HIDDEN.search(line) and not line.lstrip().startswith("#")]
    assert not offenders, offenders
    allowed = {"__future__", "numpy", "envs.public_schema", "techtree.env"}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            assert {a.name for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "") in allowed, node.module


# ------------------------------------------------------------- telescoping
def _episode_returns(env, gamma, policy_rng, max_steps=10 ** 4):
    """Plays learner episodes until the source is exhausted; per episode (phi0, discounted shaping sum,
    environment rewards)."""
    from hypergraph_agent.baselines.wrappers import SourceExhausted
    out = []
    while True:
        try:
            env.reset()
        except SourceExhausted:
            return out
        phi0, total, disc, rewards = env.shaping.phi, 0.0, 1.0, []
        for _ in range(max_steps):
            mask = env.action_masks()
            _, r, term, _, info = env.step(int(policy_rng.choice(np.flatnonzero(mask))))
            total += disc * info["shaping"]
            rewards.append(r - info["shaping"])
            disc *= gamma
            if term:
                break
        out.append((phi0, total, rewards))


@pytest.mark.parametrize("trial", [False, True])
def test_shaped_returns_telescope_in_recipequest(trial):
    gamma = 0.9
    blocks = [rq_items(1, 3, seed=s) for s in range(4)]
    env = RecipeQuestGym(ListSource(blocks), trial=trial, shaping=xshaping.make_shaper(RQ_SHAPING, gamma),
                         meter=meter())
    eps = _episode_returns(env, gamma, np.random.default_rng(3))
    assert len(eps) == (4 if trial else 12)
    for phi0, total, rewards in eps:
        assert total == pytest.approx(-phi0, abs=1e-9)  # Phi = 0 at the end of every learner episode
        assert set(np.round(rewards, 9)) <= {0.0, 1.0}  # the environment's reward is unchanged


def test_shaped_returns_telescope_in_techtree():
    gamma = 0.95
    src = training_source("techtree", TT_U, "xs_unit", 0, 0, 0, 1, 3)
    blocks = [src.next_block() for _ in range(2)]
    env = TechTreeGym(ListSource(blocks), TTLayout(2, 1, 7, 3),
                      shaping=xshaping.make_shaper(TT_SHAPING, gamma), meter=meter())
    eps = _episode_returns(env, gamma, np.random.default_rng(5))
    assert len(eps) == 6
    for phi0, total, _ in eps:
        assert total == pytest.approx(-phi0, abs=1e-9)


def test_unshaped_wrappers_are_unchanged():
    items = rq_items(1, 1)
    env = RecipeQuestGym(ListSource([items]), meter=meter())
    env.reset()
    _, r, _, _, info = env.step(env.layout.wait)
    assert r == 0.0 and "shaping" not in info and env.shaping is None


def test_shaping_gamma_is_the_libraries_default():
    from sb3_contrib import MaskablePPO, RecurrentPPO
    from stable_baselines3 import PPO
    for cls in (MaskablePPO, PPO, RecurrentPPO):
        assert inspect.signature(cls.__init__).parameters["gamma"].default == xshaping.DEFAULT_GAMMA
    assert xshaping.learner_gamma({"gamma": 0.999}) == 0.999


# ----------------------------------------------------------- hashes, configs
def test_hashes_of_unshaped_learners_are_those_of_study_x(tmp_path):
    ln = xconfig._learner({"id": "m", "algo": "maskable_ppo", "params": {"n_steps": 256}})
    assert ln["shaping"] is None
    old = {k: ln[k] for k in xconfig.TRAINING_KEYS}
    assert xconfig.training_keys(ln) == old
    cfg = {"setting": {"name": "s", "family": "recipequest"}, "train": {"namespace": "n", "seeds": [0]}}
    assert xconfig.train_hash(cfg, ln, 0) == stable_hash({"study": "X", "setting": xconfig.setting_hash(cfg),
                                                          "seed": 0, "learner": old})
    shaped = xconfig._learner({"id": "m", "algo": "maskable_ppo", "params": {"n_steps": 256},
                               "shaping": RQ_SHAPING})
    assert xconfig.train_hash(cfg, shaped, 0) != xconfig.train_hash(cfg, ln, 0)


def test_xs_configs_load_plan_and_share_one_ledger():
    from hypergraph_agent.baselines.runner import plan
    assert len(STUDIES) == 2 and len(MEMBERS) == 4
    ledgers, totals = set(), {}
    for path in MEMBERS:
        cfg = xconfig.load_config(path)
        assert cfg["run"]["ledger"] == "runs/XS-ledger.json"
        ledgers.add(json.dumps(cfg["ledger"], sort_keys=True))
        out = plan(cfg)
        alloc = cfg["run"]["allocation"]
        t = totals.setdefault(alloc, {"adaptive": 0, "reporting": 0, "training": 0})
        t["adaptive"] += out["caps"]["adaptive_total"]
        t["reporting"] += out["caps"]["reporting_total"]
        t["training"] += out["caps"]["training_total"]
        if alloc == "dev":
            assert all(n.endswith("_dev") for n in xconfig.namespaces(cfg))
        else:
            assert cfg["run"]["requires_flag"]
            assert not any(n.endswith("_dev") for n in xconfig.namespaces(cfg))
        assert out["jobs"]["agent"] == len(cfg["agents"]) * len(cfg["test"]["seeds"]) + len(
            cfg["validation"]["seeds"])  # random also plays the validation stream
    assert len(ledgers) == 1
    led = json.loads(ledgers.pop())
    for alloc, t in totals.items():  # every planned job and a full rerun of every training job
        assert t["adaptive"] + t["training"] <= led["allocations"][alloc]
    assert sum(led["allocations"].values()) <= led["adaptive_cap"]
    assert sum(t["reporting"] for t in totals.values()) <= led["reporting_cap"]
    assert led["allocations"]["xs_main"] == 40_100_000 and led["adaptive_cap"] == 40_700_000
    for path in STUDIES:
        study = xs.load_study(path)
        assert set(study["_members"]) == {"s2", "s12"}
        slacks = {k: c["setting"]["tasks"]["budget_slack"] for k, c in study["_members"].items()}
        assert slacks == {"s2": 2, "s12": 12}


def test_xs_namespaces_are_fresh():
    names = {n for p in MEMBERS for n in xconfig.namespaces(xconfig.load_config(p))}
    others = [p for p in (ROOT / "configs").rglob("*.yaml") if not p.name.startswith("xs_")]
    used = " ".join(p.read_text() for p in others)
    assert names and not any(n in used for n in names)
    assert not any(n.startswith("x_") for n in names)


def test_xs_learners_carry_study_x_tuned_settings(monkeypatch, tmp_path):
    from hypergraph_agent.baselines.tuning import TuningError, apply_tuning, verify_tuning
    monkeypatch.chdir(ROOT)
    for name in ("xs_rq_s2", "xs_rq_s12"):
        path = ROOT / "configs" / "baselines" / f"{name}.yaml"
        choice = verify_tuning(xconfig.load_config(path))
        assert set(choice.values()) == {"mppo.wide@deterministic"}
        with pytest.raises(TuningError):
            apply_tuning(path)
        text = path.read_text().replace("ent_coef: 0.01,", "ent_coef: 0.02,", 1)
        other = tmp_path / "configs" / "baselines" / f"{name}.yaml"
        other.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "configs" / "baselines" / "tune_rq_f1.yaml").write_text(
            (ROOT / "configs" / "baselines" / "tune_rq_f1.yaml").read_text())
        other.write_text(text)
        with pytest.raises(TuningError):
            verify_tuning(xconfig.load_config(other))


def test_slack_members_share_tasks_and_differ_only_in_budget():
    a = xconfig.load_config(ROOT / "configs" / "baselines" / "xs_rq_s2.yaml")
    b = xconfig.load_config(ROOT / "configs" / "baselines" / "xs_rq_s12.yaml")
    for split in ("test", "validation"):
        sp = a[split]
        sa = evaluation_source("recipequest", a["setting"], sp["namespace"], sp["base_seed"], 0, 3, 4)
        sb = evaluation_source("recipequest", b["setting"], sp["namespace"], sp["base_seed"], 0, 3, 4)
        for _ in range(3):
            for x, y in zip(sa.next_block(), sb.next_block()):
                assert (x.world_key, x.world_episode, x.dyn_seed, x.task.task_key, x.task.goal) == \
                    (y.world_key, y.world_episode, y.dyn_seed, y.task.task_key, y.task.goal)
                assert y.task.budget - x.task.budget == 10
    sa = training_source("recipequest", a["setting"], "xs_rq_train", 30_000, 1, 3, 8, 16).next_block()
    sb = training_source("recipequest", b["setting"], "xs_rq_train", 30_000, 1, 3, 8, 16).next_block()
    assert [x.task.task_key for x in sa] == [y.task.task_key for y in sb]


def test_study_x_configs_plan_no_validation_agent():
    from hypergraph_agent.baselines.runner import jobs
    for path in (ROOT / "configs" / "baselines").glob("*.yaml"):
        if path.name.startswith("xs_"):
            continue
        cfg = xconfig.load_config(path)
        if cfg["setting"].get("loader"):
            continue  # benchmark sets: data may be absent; they declare no validation stream
        assert cfg["validation_agents"] == [] and all(j["split"] == "test" for j in jobs(cfg)["agent"])


# ------------------------------------------------------------ the analysis
def _member(tmp, label, slack, ids, late_from=1, seeds=(0,), allocation=200_000):
    body = {"run": {"name": f"xs-unit-{label}", "allocation": "dev", "ledger": str(tmp / "XS-ledger.json"),
                    "runs_dir": str(tmp / "runs")},
            "ledger": {"allocations": {"dev": allocation}, "adaptive_cap": 200_000, "reporting_cap": 200_000},
            "setting": {"name": "rq_unit", "family": "recipequest",
                        "tasks": {**F1_TASKS, "budget_slack": slack}},
            "train": {"namespace": "xs_unit_train_dev", "base_seed": 30_000, "seeds": list(seeds),
                      "budgets": [32, 64], "n_envs": 2, "episodes_per_world": 2},
            "test": {"namespace": "xs_unit_eval_dev", "base_seed": 10_000, "seeds": [0], "n_worlds": 2,
                     "episodes_per_world": 2},
            "validation": {"namespace": "xs_unit_val_dev", "base_seed": 10_000, "seeds": [0], "n_worlds": 2,
                           "episodes_per_world": 2},
            "learners": [{"id": ids[0], "group": ids[0], "algo": "maskable_ppo",
                          "params": {"n_steps": 16, "batch_size": 16}},
                         {"id": ids[1], "group": ids[1], "algo": "maskable_ppo",
                          "params": {"n_steps": 16, "batch_size": 16}, "shaping": RQ_SHAPING}],
            "agents": ["maximal", "random"], "validation_agents": ["random"],
            "analysis": {"late_from": late_from, "tests": []}}
    p = tmp / f"m_{label}.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def _study(tmp, tests, late_from=1, **member):
    members = {"s2": str(_member(tmp, "s2", 2, ["s2_a", "s2_b"], late_from, **member)),
               "s12": str(_member(tmp, "s12", 12, ["s12_a", "s12_b"], late_from, **member))}
    p = tmp / "study.yaml"
    p.write_text(yaml.safe_dump({"xs": {"name": "xs-unit", "members": members, "analysis": {
        "late_from": late_from, "n_boot": 500, "boot_seed": 0, "tests": tests}}}))
    return p


def _rows(arm, units, steps, kind, split, budget=None, seed=0):
    out = []
    for u in units:
        for e in range(2):
            out.append({"arm": arm, "kind": kind, "split": split, "test_seed": 0, "unit": f"s0:{u}",
                        "world_key": u, "world_episode": e, "order": e, "success": True, "status": "success",
                        "primitive_length": steps, "reference_length": 10, "reference_exact": True,
                        "train_seed": seed if kind == "learner" else None, "budget": budget,
                        "group": arm if kind == "learner" else None,
                        "mode": "deterministic" if kind == "learner" else None})
    return out


def test_study_selects_on_validation_against_random_at_the_learners_slack(tmp_path):
    tests = [{"name": "p", "role": "primary", "a": "best_learner", "b": "random.best",
              "metric": "cost_ratio_late"},
             {"name": "f", "role": "factor", "a": "s12_a~random", "b": "s2_a~random",
              "metric": "cost_ratio_late"},
             {"name": "t", "role": "factor", "a": "best_learner", "b": "best_learner#32",
              "metric": "cost_ratio_late"}]
    study = xs.load_study(_study(tmp_path, tests))
    val, test = ["V0", "V1"], ["W0", "W1", "W2"]
    rows = (_rows("random.s2", val, 30, "agent", "validation")
            + _rows("random.s12", val, 40, "agent", "validation")
            + _rows("random.s2", test, 30, "agent", "test") + _rows("random.s12", test, 40, "agent", "test"))
    for arm, v, t in (("s2_a", 29, 29), ("s2_b", 30, 28), ("s12_a", 35, 35), ("s12_b", 41, 20)):
        rows += _rows(arm, val, v, "learner", "validation", 64) + _rows(arm, test, t, "learner", "test", 64)
        rows += _rows(arm, test, 45, "learner", "test", 32)
    complete = {"s2": {"complete": True}, "s12": {"complete": True}}
    out = xs.analyse_rows(study, rows, complete)
    sel = out["selection"]
    # s12_a is costlier than s2_a in raw terms but furthest below random at its own slack
    assert sel["best_learner"] == "s12_a" and sel["best_member"] == "s12"
    assert sel["scores"]["s12_a"] == pytest.approx(-0.5) and sel["scores"]["s2_a"] == pytest.approx(-0.1)
    p, f, t = out["tests"]
    assert p["b_resolved"] == "random.s12" and p["estimate"] == pytest.approx(-0.5) and p["n_units"] == 3
    assert f["estimate"] == pytest.approx(-0.5 - (-0.1)) and t["estimate"] == pytest.approx(3.5 - 4.5)
    assert "s12_a" in out["learning_curves"] and "random.s12" in out["learning_curves"]
    assert not any("~" in k or "#" in k for k in out["learning_curves"])
    partial = xs.analyse_rows(study, rows, {"s2": {"complete": True}, "s12": {"complete": False}})
    assert partial["selection"]["best_learner"] is None
    assert {x["verdict"] for x in partial["tests"]} == {"incomplete"}


def test_study_rejects_inconsistent_members(tmp_path):
    p = _study(tmp_path, [])
    m = tmp_path / "m_s12.yaml"
    body = yaml.safe_load(m.read_text())
    body["test"]["n_worlds"] = 3
    m.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError):
        xs.load_study(p)


def test_study_end_to_end_with_shaped_training(tmp_path):
    from hypergraph_agent.baselines.runner import existing_runs, run_study
    tests = [{"name": "p", "role": "primary", "a": "best_learner", "b": "random.best",
              "metric": "cost_ratio_late"}]
    path = _study(tmp_path, tests)
    study = xs.load_study(path)
    for cfg in study["_members"].values():
        run_study(cfg)
        runs = existing_runs(cfg["run"]["runs_dir"], cfg["run"]["name"])
        assert all(m["complete"] for m in runs)
        trains = [m for m in runs if m["job_kind"] == "train"]
        assert sorted(m["interactions"]["physical"] for m in trains) == [64, 64]
        vals = [m for m in runs if m["job_kind"] == "agent" and "-validation-" in m["run_id"]]
        assert len(vals) == 1 and vals[0]["arm"] == "random"
    summary = xs.write_study_report(path, tmp_path / "out")
    assert summary["selection"]["best_learner"] in {"s2_a", "s2_b", "s12_a", "s12_b"}
    assert summary["tests"][0]["verdict"] != "incomplete" and summary["tests"][0]["n_units"] == 2
    assert (tmp_path / "out" / "summary.md").exists() and set(summary["training_curves"]) == \
        {"s2_a", "s2_b", "s12_a", "s12_b"}
    ledger = json.loads((tmp_path / "XS-ledger.json").read_text())
    assert ledger["adaptive_total"] + ledger["reporting_total"] == sum(
        r["used"] for r in ledger["runs"].values())


def test_per_training_seed_contrasts_against_an_agent(tmp_path):
    tests = [{"name": "p", "role": "primary", "a": "best_learner", "b": "random.best",
              "metric": "cost_ratio_late"},
             {"name": "w", "a": "best_learner", "metric": "cost_ratio_late"},
             {"name": "f", "role": "factor", "a": "s2_b", "b": "s2_a", "metric": "cost_ratio_late"}]
    study = xs.load_study(_study(tmp_path, tests, seeds=(0, 1)))
    val, test = ["V0", "V1"], ["W0", "W1"]
    rows = []
    for split, units in (("validation", val), ("test", test)):
        rows += _rows("random.s2", units, 30, "agent", split) + _rows("random.s12", units, 40, "agent", split)
    for arm, by_seed in (("s2_a", (29, 31)), ("s2_b", (30, 30)), ("s12_a", (34, 36)), ("s12_b", (41, 41))):
        for seed, steps in enumerate(by_seed):
            for split, units in (("validation", val), ("test", test)):
                rows += _rows(arm, units, steps, "learner", split, 64, seed)
            rows += _rows(arm, test, 45, "learner", "test", 32, seed)
    out = xs.analyse_rows(study, rows, {"s2": {"complete": True}, "s12": {"complete": True}})
    p, w, f = out["tests"]
    assert out["selection"]["best_learner"] == "s12_a" and p["estimate"] == pytest.approx(-0.5)
    assert p["by_training_seed_of"] == "a"
    assert p["by_training_seed"] == {"0": pytest.approx(-0.6), "1": pytest.approx(-0.4)}
    assert w["by_training_seed"] == {"0": pytest.approx(3.4), "1": pytest.approx(3.6)}
    assert "by_training_seed_of" not in f  # a learner b: X's per-seed contrast (seeds of b) is kept
    assert f["by_training_seed"] == {"0": pytest.approx(0.1), "1": pytest.approx(-0.1)}
    assert "s0 -0.600, s1 -0.400" in xs.markdown(out)


def test_budget_check_covers_the_missing_jobs(tmp_path):
    from hypergraph_agent.baselines.__main__ import main
    from hypergraph_agent.baselines.runner import plan
    path = _study(tmp_path, [])
    res = xs.budget_check(path)
    planned = [plan(c)["caps"] for c in xs.load_study(path)["_members"].values()]
    assert res["ok"] and not res["ledger_exists"]
    assert res["need"]["adaptive"] == sum(c["adaptive_total"] for c in planned)
    assert res["need"]["reporting"] == sum(c["reporting_total"] for c in planned)
    assert res["need"]["training"] == sum(c["training_total"] for c in planned) == 4 * 64
    assert main(["check-budget", "--config", str(path)]) == 0
    small = tmp_path / "small"
    small.mkdir()
    path = _study(small, [], allocation=200)  # less than the four training jobs' caps
    res = xs.budget_check(path)
    assert not res["ok"] and res["left"]["adaptive"] == 200
    assert main(["check-budget", "--config", str(path)]) == 1
    assert not (small / "XS-ledger.json").exists()  # the check only reads


def test_budget_check_counts_only_jobs_without_a_completed_run(tmp_path):
    from hypergraph_agent.baselines.runner import run_study
    path = _study(tmp_path, [])
    study = xs.load_study(path)
    before = xs.budget_check(path)
    run_study(study["_members"]["s2"])
    after = xs.budget_check(path)
    assert after["need"]["training"] == before["need"]["training"] - 2 * 64
    assert after["need"]["jobs"] < before["need"]["jobs"] and after["ok"]
    used = json.loads((tmp_path / "XS-ledger.json").read_text())["by_allocation"]["dev"]
    assert after["left"]["adaptive"] == 200_000 - used


DRIVER = ROOT / "scripts" / "run_xs_overnight.sh"


def test_driver_requires_a_clean_snapshot_and_checks_the_budget():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'git -C "$XS_SNAPSHOT" status --porcelain' in text and "rev-parse HEAD" in text
    assert "check-budget" in text and text.index("check-budget") < text.index("--allow-measurement --resume")
    measurement = text.split('if [ "$MODE" = "measurement" ]; then', 1)[1].split("else", 1)[0]
    assert 'SRC="$XS_SNAPSHOT/src"' in measurement and 'SRC="src"' not in measurement


def _native_bash():
    import shutil
    import sys
    bash = shutil.which("bash")
    if bash is None or (sys.platform == "win32" and "system32" in bash.lower()):
        return None  # no bash, or the launcher of the Windows Subsystem for Linux
    return bash


@pytest.mark.skipif(_native_bash() is None, reason="needs a native bash")
def test_driver_refuses_without_a_clean_snapshot(tmp_path):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "XS_SNAPSHOT"}
    for extra, args in (({}, []), ({}, ["--bogus"]), ({"XS_SNAPSHOT": str(tmp_path)}, [])):
        proc = subprocess.run([_native_bash(), str(DRIVER), *args], cwd=ROOT, env={**env, **extra},
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2, (extra, args, proc.stderr)
        if not extra and not args:
            assert "XS_SNAPSHOT is not set" in proc.stderr


# ---------------------------------------------------------------- hygiene
def test_xs_files_are_short_and_plain():
    files = [PKG / "shaping.py", PKG / "xs.py", pathlib.Path(__file__), *XS_CONFIGS,
             ROOT / "scripts" / "run_xs_overnight.sh", ROOT / "docs" / "studies" / "XS_learner_followup.md"]
    long = [f"{f.name}:{n}" for f in files
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if len(line) > 110]
    assert not long, long
    dash = "[" + chr(0x2013) + chr(0x2014) + "]"  # en and em dashes
    dashes = [f.name for f in files if re.search(dash, f.read_text(encoding="utf-8"))]
    assert not dashes, dashes
