"""Study X: fixed-space wrappers, information boundary, seeding, ledger accounting, analysis."""

import ast
import json
import pathlib
import re
from dataclasses import replace

import numpy as np
import pytest

from hypergraph_agent.baselines import config as xconfig
from hypergraph_agent.baselines.encoders import EpisodeContext, RQLayout, TTLayout
from hypergraph_agent.baselines.sources import BlockSource, TaskItem, evaluation_source, training_source
from hypergraph_agent.baselines.wrappers import RecipeQuestGym, SourceExhausted, TechTreeGym
from hypergraph_agent.envs.generator import (
    PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig,
)
from hypergraph_agent.envs.public_schema import CRAFT, public_view
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.vocabulary import ITEM_TYPES
from hypergraph_agent.techtree.generator import TechStream, TechStreamConfig, TechWorldConfig
from hypergraph_agent.training.budget import BudgetMeter, SessionLedger

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "hypergraph_agent" / "baselines"
CONFIGS = sorted((ROOT / "configs" / "baselines").glob("*.yaml"))
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
TT_U = {"name": "tt_unit", "family": "techtree", "world": U_WORLD, "tasks": {}, "stream": U_STREAM,
        "agent_study": "U", "loader": None, "max_variants": 2}
TT_L = {**TT_U, "world": L_WORLD, "stream": {"goal_levels": [3], "episode_budget": 100}, "agent_study": "L"}


def meter(cap=10 ** 6, kind="adaptive"):
    return BudgetMeter(SessionLedger(None, {"dev": 10 ** 7}, 10 ** 7, 10 ** 7), "t", "dev", cap, kind=kind)


def rq_items(n_worlds=2, episodes=3, namespace="x_unit", seed=0, tasks=F1_TASKS):
    setting = {**RQ_SETTING, "tasks": tasks}
    src = evaluation_source("recipequest", setting, namespace, 10_000, seed, n_worlds, episodes)
    out = []
    while (b := src.next_block()) is not None:
        out += b
    return out


class ListSource:
    """A finite source over given blocks (test fixture)."""

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


# ------------------------------------------------------------- layouts
def test_recipequest_layout_maps_every_task_action_once():
    layout = RQLayout()
    assert layout.n_actions == 12 + 6 + 16 * 2 + 2
    for item in rq_items(3, 4):
        spec = public_view(item.task)
        amap = layout.action_map(spec)
        valid = amap[amap >= 0]
        assert sorted(valid.tolist()) == list(range(len(spec.actions)))
        assert amap[layout.wait] >= 0 and amap[layout.submit] >= 0
        for canon, a in enumerate(amap):
            if a < 0:
                continue
            d = spec.actions[a]
            if d.fact is not None:
                assert canon == spec.facts[d.fact].type_id
            elif d.kind == CRAFT:
                eff = spec.facts[spec.rules[d.rule].effect].type_id
                first = layout.craft0 + (eff - ITEM_TYPES[0]) * 2
                assert first <= canon < first + 2


def test_recipe_slots_follow_public_signatures():
    layout = RQLayout()
    found = 0
    for item in rq_items(6, 6):
        spec = public_view(item.task)
        slots = layout.slots(spec)
        by_effect = {}
        for j, r in enumerate(spec.rules):
            by_effect.setdefault(r.effect, []).append(j)
        for js in by_effect.values():
            if len(js) == 2:
                found += 1
                first, second = sorted(js, key=lambda j: slots[j])
                assert spec.rules[first].signature < spec.rules[second].signature
    assert found > 0


def test_recipequest_observation_is_bounded_and_deterministic():
    items = rq_items(2, 4)
    obs = []
    for _ in range(2):
        env = RecipeQuestGym(ListSource([items[:4], items[4:]]), meter=meter())
        x, _ = env.reset()
        seq = [x]
        rng = np.random.default_rng(1)
        for _ in range(60):
            x, r, term, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            seq.append(x)
            assert x.dtype == np.float32 and x.shape == (env.layout.dim,) and 0 <= x.min() and x.max() <= 1
            if term:
                try:
                    x, _ = env.reset()
                except SourceExhausted:
                    break
                seq.append(x)
        obs.append(np.stack(seq))
    assert np.array_equal(obs[0], obs[1])


def test_goal_only_observation_never_shows_intermediate_items():
    layout = RQLayout()
    for item in rq_items(4, 4):
        task = item.task
        env = RecipeQuestGym(ListSource([[item]]), meter=meter())
        env.reset()
        spec = env.spec
        goal_type = spec.facts[spec.goal].type_id
        hidden_items = {spec.facts[r.effect].type_id for r in spec.rules} - {goal_type}
        # gather every base fact, then craft every recipe once in storage order
        amap = env.action_masks()
        for canon in np.flatnonzero(amap):
            if canon < layout.craft0:
                x, *_ = env.step(int(canon))
        for canon in np.flatnonzero(amap):
            if layout.craft0 <= canon < layout.wait and not env.env._done:
                x, *_ = env.step(int(canon))
                present = x[layout.static_dim:layout.static_dim + 34]
                assert all(present[t] == 0 for t in hidden_items)
        assert task.observe_items == "goal_only"


def test_techtree_layouts_and_masks():
    for setting, dims in ((TT_U, (2, 1, 7, 3)), (TT_L, (9, 3, 3, 6))):
        layout = TTLayout(*dims)
        src = training_source("techtree", setting, "x_unit", 0, 0, 0, 1, 12)
        env = TechTreeGym(src, layout, meter=meter())
        x, _ = env.reset()
        assert x.shape == (layout.dim,) and env.action_space.n == dims[2] + 2
        mask = env.action_masks()
        assert mask.dtype == bool and mask.shape == (dims[2] + 2,)
        if setting is TT_U:
            assert mask.tolist() == [True] * 4 + [False] * 3 + [True, True]  # announced slots start inactive


# ------------------------------------------------- information boundary
def _rq_play(task, actions):
    env = RecipeQuestGym(ListSource([[TaskItem(task, task.world.world_key, 0, 0, 0)]]), meter=meter())
    xs = [env.reset()[0]]
    masks = [env.action_masks()]
    for a in actions:
        x, _, term, _, _ = env.step(a)
        xs.append(x)
        masks.append(env.action_masks())
        if term:
            break
    return np.stack(xs), np.stack(masks)


def test_recipequest_twins_differing_in_hidden_structure_give_identical_observations():
    """Goal-only observation: any history that never crafts the goal has the same public
    outcomes whatever the hidden requirements, so the observations must coincide."""
    layout, rng = RQLayout(), np.random.default_rng(7)
    checked = 0
    for item in rq_items(4, 4):
        twin = hidden_twin(item.task, rng)
        if all(a.true_base == b.true_base for a, b in zip(item.task.rules, twin.rules)):
            continue
        assert public_view(item.task) == public_view(twin)
        spec = public_view(item.task)
        goal_type = spec.facts[spec.goal].type_id
        amap = layout.action_map(spec)
        crafts_goal = {layout.craft_index(goal_type, v) for v in range(2)}
        allowed = [c for c in np.flatnonzero(amap >= 0) if c not in crafts_goal and c != layout.submit]
        actions = [int(rng.choice(allowed)) for _ in range(item.task.budget - 1)]
        xa, ma = _rq_play(item.task, actions)
        xb, mb = _rq_play(twin, actions)
        assert np.array_equal(xa, xb) and np.array_equal(ma, mb)
        checked += 1
    assert checked >= 8


def _tt_canary_pair():
    from hypergraph_agent.techtree.generator import TechStream, TechStreamConfig, TechWorldConfig
    cfg = TechWorldConfig(**{**L_WORLD, "concepts_per_level": (3, 3, 3)})
    s0 = TechStream(TechStreamConfig("x_unit", 0, world=replace(cfg, reuse_depth=0), goal_levels=(3,)))
    s2 = TechStream(TechStreamConfig("x_unit", 0, world=replace(cfg, reuse_depth=2), goal_levels=(3,)))
    return s0.task(0, 0), s2.task(0, 0)


def test_techtree_twins_differing_in_hidden_structure_give_identical_observations():
    a, b = _tt_canary_pair()
    assert a.world.sequences != b.world.sequences
    layout = TTLayout(9, 3, 3, 6)
    # single presses separated by wait never complete a sequence of length >= 2: identical public outcomes
    actions = [0, 3, 1, 3, 2, 3, 2, 4, 1]
    out = []
    for t in (a, b):
        env = TechTreeGym(ListSource([[TaskItem(t, t.world.world_key, 0, 0, 0)]]), layout, meter=meter())
        xs = [env.reset()[0]]
        for act in actions:
            xs.append(env.step(act)[0])
        out.append(np.stack(xs))
    assert np.array_equal(out[0], out[1])


def test_techtree_unlock_twins_give_identical_first_observations():
    world = TechWorldConfig(**{**U_WORLD, "concepts_per_level": (2,)})
    stream = TechStream(TechStreamConfig("x_unit", 0, world=world, goal_levels=(1,), goal_rule="unlock"))
    t = stream.task(0, 0)
    w = t.world
    other = 4 + (w.unlocks[0][1] - 4 + 1) % 3
    twin = replace(t, world=replace(w, unlocks=((w.unlocks[0][0], other),)))
    layout = TTLayout(2, 1, 7, 3)
    xs = []
    for task in (t, twin):
        env = TechTreeGym(ListSource([[TaskItem(task, w.world_key, 0, 0, 0)]]), layout, meter=meter())
        xs.append(env.reset()[0])
    assert np.array_equal(xs[0], xs[1])


HIDDEN = re.compile(r"true_base|\.required\(\)|reference_solve|reference_solver|debug_view|\._task\b|"
                    r"\.sequences\b|\.parents\b|\.unlocks\b|\.entries\b|\.compositional\b|revealed_links|"
                    r"needs_unlock|_present\b|_held\b")
PUBLIC_MODULES = ("encoders.py", "wrappers.py", "learners.py")


def test_learner_side_modules_never_name_hidden_fields():
    offenders = []
    for name in PUBLIC_MODULES:
        for n, line in enumerate((PKG / name).read_text().splitlines(), 1):
            if HIDDEN.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{name}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_encoders_import_only_public_modules():
    allowed = {"__future__", "dataclasses", "numpy", "envs.public_schema", "envs.vocabulary", "techtree.env"}
    tree = ast.parse((PKG / "encoders.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "") in allowed, node.module


def test_wrapper_info_is_public_only():
    items = rq_items(1, 1)
    env = RecipeQuestGym(ListSource([items]), meter=meter())
    _, info0 = env.reset()
    _, _, _, _, info = env.step(env.layout.wait)
    assert info0 == {} and set(info) == {"termination", "changed", "invalid", "task_end"}
    src = training_source("techtree", TT_U, "x_unit", 0, 0, 0, 1, 2)
    env = TechTreeGym(src, TTLayout(2, 1, 7, 3), meter=meter())
    _, info0 = env.reset()
    _, _, _, _, info = env.step(0)
    assert info0 == {} and set(info) == {"termination", "fired", "invalid", "task_end"}


# --------------------------------------------------------------- dynamics
def test_invalid_actions_execute_wait_and_cost_a_step():
    items = rq_items(1, 1)
    m = meter()
    env = RecipeQuestGym(ListSource([items]), meter=m)
    env.reset()
    absent = int(np.flatnonzero(~env.action_masks())[0])
    budget0 = env.env.public_spec.budget
    x, r, term, _, info = env.step(absent)
    assert info["invalid"] and m.used == 1 and env.interactions == 1
    assert env.env._budget_left == budget0 - 1
    src = training_source("techtree", TT_U, "x_unit", 0, 0, 0, 1, 2)
    env = TechTreeGym(src, TTLayout(2, 1, 7, 3), meter=meter())
    env.reset()
    _, _, _, _, info = env.step(5)  # an announced slot that is not active
    assert info["invalid"] and env._trace == []


def test_trial_mode_spans_a_world_block():
    items = rq_items(2, 3)
    blocks = [items[:3], items[3:]]
    env = RecipeQuestGym(ListSource(blocks), trial=True, meter=meter())
    ends, resets = [], 0
    while True:
        try:
            env.reset()
        except SourceExhausted:
            break
        resets += 1
        term = False
        while not term:
            _, _, term, _, info = env.step(env.layout.wait)
            if info.get("task_end"):
                ends.append(term)
    assert resets == 2 and ends == [False, False, True, False, False, True]
    # task mode: one learner episode per task, still in world-major order
    env = RecipeQuestGym(ListSource([items[:3], items[3:]]), meter=meter())
    keys = []
    for _ in range(6):
        env.reset()
        keys.append(env.spec.task_key)
        term = False
        while not term:
            _, _, term, _, _ = env.step(env.layout.wait)
    assert keys == [i.task.task_key for i in items]


def test_episode_context_is_public_and_bounded():
    v = EpisodeContext(15, 16, False, True).vector()
    assert v.tolist() == pytest.approx([15 / 16, 0.0, 1.0])


# ---------------------------------------------------------------- seeding
def test_sources_are_deterministic_and_disjoint():
    def keys(src, n):
        out = []
        for _ in range(n):
            out += [(i.task.task_key, i.world_key) for i in src.next_block()]
        return out
    a = keys(training_source("recipequest", RQ_SETTING, "x_unit_train", 30_000, 0, 0, 2, 4), 3)
    b = keys(training_source("recipequest", RQ_SETTING, "x_unit_train", 30_000, 0, 0, 2, 4), 3)
    c = keys(training_source("recipequest", RQ_SETTING, "x_unit_train", 30_000, 0, 1, 2, 4), 3)
    d = keys(training_source("recipequest", RQ_SETTING, "x_unit_train", 30_000, 1, 0, 2, 4), 3)
    test = [(i.task.task_key, i.world_key) for i in rq_items(3, 4, namespace="x_unit_test")]
    assert a == b
    worlds = [{w for _, w in x} for x in (a, c, d, test)]
    assert all(not (worlds[i] & worlds[j]) for i in range(4) for j in range(i + 1, 4))


def test_recipequest_test_stream_matches_the_walker_stream_order():
    from hypergraph_agent.evaluation.walkers import eval_order
    items = rq_items(3, 4, namespace="x_unit_test", seed=2)
    cfg = TaskStreamConfig("x_unit_test", 12_000, "pool", 3, 0, WorldConfig(), TaskConfig(**F1_TASKS))
    stream = TaskStream(cfg)
    assert [i.order for i in items] == eval_order(cfg, 12, 4)
    assert [i.task.task_key for i in items] == [stream.task(j).task_key for j in eval_order(cfg, 12, 4)]
    assert [i.world_episode for i in items] == [0, 1, 2, 3] * 3


def test_techtree_test_stream_matches_the_study_stream():
    from hypergraph_agent.baselines.sources import tt_stream_config
    src = evaluation_source("techtree", TT_U, "x_unit_test", 0, 3, 2, 5)
    items = []
    while (b := src.next_block()) is not None:
        items += b
    stream = TechStream(tt_stream_config(TT_U, "x_unit_test", 3, 2, 5))
    assert [i.task.task_key for i in items] == [stream.task(w, e).task_key for w, e in stream.order()]


def test_external_loader_extension_point(monkeypatch):
    import types
    import sys
    mod = types.ModuleType("x_fake_benchmark")
    stream = TaskStream(TaskStreamConfig("x_unit_ext", 1, "pool", 2, 0, WorldConfig(),
                                         TaskConfig(**F1_TASKS)))

    def load(setting, namespace, seed, world_index, episodes):
        if world_index >= 2:
            return None
        return [stream.task(e * 2 + world_index) for e in range(episodes)]

    mod.load = load
    monkeypatch.setitem(sys.modules, "x_fake_benchmark", mod)
    setting = {**RQ_SETTING, "loader": "x_fake_benchmark:load"}
    src = evaluation_source("recipequest", setting, "x_unit_ext", 0, 0, 5, 3)
    blocks = []
    while (b := src.next_block()) is not None:
        blocks.append(b)
    assert len(blocks) == 2 and all(len(b) == 3 for b in blocks)
    env = RecipeQuestGym(ListSource(blocks), meter=meter())
    env.reset()
    assert env.action_masks().sum() > 0


# -------------------------------------------------------------- configs
def _write(tmp_path, body: dict, name="x.yaml"):
    import yaml
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body))
    return p


def unit_config(tmp_path, **over):
    body = {
        "run": {"name": "x-unit", "allocation": "dev", "ledger": str(tmp_path / "X-ledger.json"),
                "runs_dir": str(tmp_path / "runs")},
        "ledger": {"allocations": {"dev": 100_000}, "adaptive_cap": 100_000, "reporting_cap": 100_000},
        "setting": {"name": "rq_unit", "family": "recipequest", "tasks": F1_TASKS},
        "train": {"namespace": "x_unit_train_dev", "base_seed": 30_000, "seeds": [0], "budgets": [32, 64],
                  "n_envs": 2, "episodes_per_world": 2},
        "test": {"namespace": "x_unit_eval_dev", "base_seed": 10_000, "seeds": [0], "n_worlds": 1,
                 "episodes_per_world": 2},
        "learners": [{"id": "mppo", "algo": "maskable_ppo", "params": {"n_steps": 16, "batch_size": 16}}],
        "agents": ["maximal", "random"],
        "analysis": {"late_from": 1, "tests": [
            {"name": "u1", "role": "primary", "a": "maximal", "b": "best_learner",
             "metric": "cost_ratio_late"}]},
    }
    for k, v in over.items():
        body[k] = v if not isinstance(v, dict) or k not in body else {**body[k], **v}
    return xconfig.load_config(_write(tmp_path, body))


def benchmark_data_present(cfg: dict) -> bool:
    graph = (cfg["setting"].get("benchmark") or {}).get("graph")
    if graph is None:
        return True
    from hypergraph_agent.benchmarks import loaders
    return loaders.available(loaders.GRAPHS[graph][0])


def test_every_study_config_loads_and_plans():
    from hypergraph_agent.baselines.runner import plan
    assert CONFIGS
    ledgers = set()
    for path in CONFIGS:
        cfg = xconfig.load_config(path)
        if not benchmark_data_present(cfg):
            continue  # a fetch-only benchmark whose data was not downloaded
        out = plan(cfg)
        assert (out["jobs"]["train"] > 0) != bool(cfg["run"]["checkpoints_from"])
        assert out["jobs"]["eval"] > 0 and out["caps"]["adaptive_total"] > 0
        ledgers.add(json.dumps(cfg["ledger"], sort_keys=True))
        assert cfg["run"]["ledger"] == "runs/X-ledger.json"
        if cfg["run"]["allocation"] in xconfig.DEV_ALLOCATIONS:
            assert all(n.endswith("_dev") for n in xconfig.namespaces(cfg))
        else:
            assert cfg["run"]["requires_flag"]
            assert not any(n.endswith("_dev") for n in xconfig.namespaces(cfg))
    assert len(ledgers) == 1  # one ledger declaration shared by every config


def test_config_rules(tmp_path):
    with pytest.raises(ValueError):
        unit_config(tmp_path, test={"namespace": "x_unit_eval"})  # dev allocation on a measurement namespace
    with pytest.raises(ValueError):
        unit_config(tmp_path, train={"budgets": [33]})  # not a multiple of n_envs
    with pytest.raises(KeyError):
        unit_config(tmp_path, train={"bogus": 1})
    with pytest.raises(KeyError):
        unit_config(tmp_path, learners=[{"id": "a", "algo": "ppo", "colour": 1}])
    with pytest.raises(ValueError):
        unit_config(tmp_path, learners=[{"id": "a", "algo": "a2c"}])
    with pytest.raises(ValueError):
        unit_config(tmp_path, agents=["pooled"])  # a TechTree arm in a RecipeQuest setting
    cfg = unit_config(tmp_path)
    assert cfg["learners"][0]["group"] == "maskable_ppo" and cfg["learners"][0]["modes"] == ["deterministic"]


def test_measurement_configs_need_the_flag_and_a_tuning_record(tmp_path):
    from hypergraph_agent.baselines.__main__ import main
    path = ROOT / "configs" / "baselines" / "rq_f1_main.yaml"
    with pytest.raises(SystemExit):
        main(["run", "--config", str(path)])
    with pytest.raises(SystemExit):
        main(["run", "--config", str(path), "--allow-measurement"])  # tuning_record is unset
    assert not (tmp_path / "runs").exists()


# ----------------------------------------------------- runner accounting
@pytest.fixture(scope="module")
def unit_study(tmp_path_factory):
    from hypergraph_agent.baselines.runner import existing_runs, run_study
    tmp = tmp_path_factory.mktemp("xstudy")
    learners = [
        {"id": "mppo", "algo": "maskable_ppo", "params": {"n_steps": 16, "batch_size": 16},
         "modes": ["deterministic", "stochastic"]},
        {"id": "ppo", "algo": "ppo", "params": {"n_steps": 16, "batch_size": 16}},
        {"id": "rppo", "algo": "recurrent_ppo", "trial": True,
         "params": {"n_steps": 16, "batch_size": 16, "features_dim": 8,
                    "policy_kwargs": {"lstm_hidden_size": 8, "shared_lstm": True,
                                      "enable_critic_lstm": False}}},
        {"id": "rnd", "group": "rnd", "algo": "maskable_ppo", "params": {"n_steps": 16, "batch_size": 16},
         "rnd": {"beta": 0.1, "update_every": 4, "batch": 8}},
        {"id": "dqn", "algo": "dqn", "params": {"buffer_size": 500, "learning_starts": 8}},
    ]
    cfg = unit_config(tmp, learners=learners, agents=["focused_sample", "maximal", "reference", "random"])
    results = run_study(cfg)
    return cfg, results, existing_runs(cfg["run"]["runs_dir"], cfg["run"]["name"])


def test_runner_charges_every_interaction(unit_study):
    cfg, results, runs = unit_study
    ledger = json.loads(pathlib.Path(cfg["run"]["ledger"]).read_text())
    assert all(m["complete"] for m in runs)
    trains = [m for m in runs if m["job_kind"] == "train"]
    assert len(trains) == 5 and all(m["interactions"]["physical"] == 64 for m in trains)
    for m in runs:
        rows = [json.loads(x) for x in (pathlib.Path(m["dir"]) / "eval.jsonl").read_text().splitlines()] \
            if m["job_kind"] != "train" else []
        if m["job_kind"] != "train":
            assert m["interactions"]["physical"] == sum(r["primitive_length"] for r in rows) > 0
        assert ledger["runs"][m["run_id"]]["used"] == m["interactions"]["physical"]
    adaptive = sum(m["interactions"]["physical"] for m in runs if m["interactions"]["kind"] == "adaptive")
    reporting = sum(m["interactions"]["physical"] for m in runs if m["interactions"]["kind"] == "reporting")
    assert ledger["adaptive_total"] == adaptive and ledger["reporting_total"] == reporting
    kinds = {(m["job_kind"], m["arm"]): m["interactions"]["kind"] for m in runs}
    assert kinds[("eval", "rppo")] == "adaptive" and kinds[("eval", "mppo")] == "reporting"
    assert kinds[("agent", "focused_sample")] == "adaptive" and kinds[("agent", "maximal")] == "reporting"


def test_runner_rows_cover_the_test_stream_with_references(unit_study):
    cfg, _, runs = unit_study
    for m in runs:
        if m["job_kind"] == "train":
            assert sorted(c["budget"] for c in m["checkpoints"]) == [32, 64]
            ordered = sorted(m["checkpoints"], key=lambda c: c["budget"])
            assert [c["interactions"] for c in ordered] == [32, 64]
            continue
        rows = [json.loads(x) for x in (pathlib.Path(m["dir"]) / "eval.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and [r["world_episode"] for r in rows] == [0, 1]
        assert all(r["reference_exact"] and r["reference_length"] > 0 for r in rows)
        if m["arm"] == "mppo":
            assert all(r["invalid_actions"] == 0 for r in rows)  # masks are respected
        if m["arm"] == "reference":
            assert all(r["success"] and r["primitive_length"] == r["reference_length"] for r in rows)
    agent_tasks = {m["arm"]: [json.loads(x)["order"] for x in (pathlib.Path(m["dir"]) / "eval.jsonl")
                              .read_text().splitlines()] for m in runs if m["job_kind"] == "agent"}
    assert len({tuple(v) for v in agent_tasks.values()}) == 1


def test_completed_jobs_are_not_repeated(unit_study, tmp_path):
    from hypergraph_agent.baselines.runner import existing_runs, run_study
    cfg, _, runs = unit_study
    assert run_study(cfg) == []  # every job already has a completed run
    measure = {**cfg, "run": {**cfg["run"], "requires_flag": True}}
    with pytest.raises(RuntimeError):
        run_study(measure)
    assert run_study(measure, resume=True) == []
    assert len(existing_runs(cfg["run"]["runs_dir"], cfg["run"]["name"])) == len(runs)


BENCH_LOADER = "hypergraph_agent.baselines.benchmark_sets:load"
BENCH_TASKS = {"profile": "unknown_prerequisites", "n_distractor_base": 0, "observe_items": "goal_only"}


def test_evaluation_only_config_reuses_the_source_checkpoints(unit_study, tmp_path):
    from hypergraph_agent.baselines.analysis import write_report
    from hypergraph_agent.baselines.runner import existing_runs, run_study
    cfg, _, runs = unit_study
    body = {"run": {"name": "x-unit-bench", "allocation": "dev", "ledger": cfg["run"]["ledger"],
                    "runs_dir": cfg["run"]["runs_dir"], "checkpoints_from": cfg["_source"]},
            "ledger": cfg["ledger"],
            "setting": {"name": "bench_unit", "family": "recipequest", "loader": BENCH_LOADER,
                        "benchmark": {"graph": "psketch_craft", "worlds": "whole", "n_worlds": 1},
                        "tasks": BENCH_TASKS},
            "test": {"namespace": "x_unit_bench_dev", "base_seed": 0, "seeds": [0], "n_worlds": 1,
                     "episodes_per_world": 2, "budgets": [64]},
            "agents": ["focused_sample", "maximal"],
            "analysis": {"late_from": 1, "tests": [{"name": "b", "a": "focused_sample", "b": "best_learner",
                                                    "metric": "cost_ratio_late"}]}}
    bench = xconfig.load_config(_write(tmp_path, body, "bench.yaml"))
    assert [ln["id"] for ln in bench["learners"]] == [ln["id"] for ln in cfg["learners"]]
    run_study(bench)
    made = existing_runs(bench["run"]["runs_dir"], "x-unit-bench")
    trains = {m["arm"]: m["run_id"] for m in runs if m["job_kind"] == "train"}
    assert made and all(m["complete"] for m in made) and not any(m["job_kind"] == "train" for m in made)
    evals = [m for m in made if m["job_kind"] == "eval"]
    assert len(evals) == 6 and all(m["parent_run"] == trains[m["arm"]] for m in evals)
    summary = write_report(bench, tmp_path / "report")
    assert summary["selection"]["source"] == "x-unit" and summary["tests"][0]["n_units"] == 1
    assert summary["conversion"]["summary"]["worlds"] == 1


def test_benchmark_sets_are_deterministic_and_record_conversions():
    from hypergraph_agent.baselines import benchmark_sets
    setting = {"name": "b", "family": "recipequest", "loader": BENCH_LOADER,
               "benchmark": {"graph": "crafter", "worlds": "whole", "n_worlds": 3}, "tasks": BENCH_TASKS}
    recs = benchmark_sets.records(setting)
    assert len(recs) == 3 and len({r["world_key"] for r in recs}) == 3
    assert all(r["promoted"] == ["wood"] and r["items"] == 16 and 0 < r["kept_share"] <= 1 for r in recs)
    a = benchmark_sets.load(setting, "x_unit_bench", 0, 1, 4)
    b = benchmark_sets.load(setting, "x_unit_bench", 0, 1, 4)
    assert [t.task_key for t in a] == [t.task_key for t in b]
    assert benchmark_sets.load(setting, "x_unit_bench", 0, 3, 4) is None
    items = {ty for ty, _ in a[0].world.levels}
    assert all(t.goal in items and t.observe_items == "goal_only" for t in a)
    src = evaluation_source("recipequest", setting, "x_unit_bench", 0, 0, 3, 4)
    blocks = []
    while (blk := src.next_block()) is not None:
        blocks.append(blk)
    assert len(blocks) == 3
    env = RecipeQuestGym(ListSource(blocks), meter=meter())
    env.reset()
    assert env.action_masks().sum() > 2
    per_goal = {**setting, "benchmark": {"graph": "psketch_craft", "worlds": "per_goal", "n_worlds": 4}}
    recs = benchmark_sets.records(per_goal)
    assert len(recs) == 4 and all(len(r["goals"]) == 1 for r in recs)


def test_benchmark_configs_play_their_worlds_once(tmp_path):
    with pytest.raises(ValueError):
        unit_config(tmp_path, setting={"name": "b", "family": "recipequest", "loader": BENCH_LOADER,
                                       "benchmark": {"graph": "crafter", "n_worlds": 3},
                                       "tasks": BENCH_TASKS},
                    test={"n_worlds": 2})


def test_training_is_deterministic(tmp_path):
    from hypergraph_agent.baselines.runner import existing_runs, run_study
    out = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        cfg = unit_config(d, agents=[])
        run_study(cfg, stamp="fixed")
        runs = existing_runs(cfg["run"]["runs_dir"], cfg["run"]["name"])
        evals = [m for m in runs if m["job_kind"] == "eval"]
        rows = [[(r["primitive_length"], r["success"]) for r in map(json.loads, (pathlib.Path(m["dir"]) /
                 "eval.jsonl").read_text().splitlines())] for m in sorted(evals, key=lambda m: m["run_id"])]
        out.append(rows)
    assert out[0] == out[1]


def test_rnd_bonus_is_positive_and_learned():
    from stable_baselines3.common.vec_env import DummyVecEnv
    from hypergraph_agent.baselines.learners import rnd_vec_env
    items = rq_items(1, 1)
    venv = DummyVecEnv([lambda: RecipeQuestGym(ListSource([items] * 400), meter=meter())])
    env = rnd_vec_env(venv, seed=0, update_every=4, batch=4, lr=1e-2)
    env.reset()
    first, last = [], []
    wait = RQLayout().wait
    for t in range(200):
        env.step_async(np.array([wait]))
        _, _, _, infos = env.step_wait()
        (first if t < 10 else last).append(infos[0]["intrinsic_reward"])
    assert min(first) > 0 and env.stats["updates"] > 10


# --------------------------------------------------------------- analysis
def _rows(arm, worlds, late, early, kind="agent", seed=None, budget=None, split="test", success=True):
    out = []
    for w in worlds:
        for e in range(4):
            steps = late[w] if e >= 2 else early[w]
            out.append({"arm": arm, "kind": kind, "split": split, "test_seed": 0, "unit": f"s0:{w}",
                        "world_key": w, "world_episode": e, "order": e, "success": success,
                        "status": "success",
                        "primitive_length": steps, "reference_length": 10, "reference_exact": True,
                        "train_seed": seed, "budget": budget, "group": arm if kind == "learner" else None,
                        "mode": "deterministic" if kind == "learner" else None})
    return out


def test_analysis_pairs_worlds_selects_on_validation_and_averages_training_seeds():
    from hypergraph_agent.baselines.analysis import analyse_rows
    worlds = [f"W{i}" for i in range(12)]
    rows = _rows("focused_sample", worlds, {w: 15 for w in worlds}, {w: 20 for w in worlds})
    for seed, cost in ((0, 18), (1, 22)):
        flat = {w: cost for w in worlds}
        rows += _rows("good", worlds, flat, flat, "learner", seed, 64)
        rows += _rows("good", worlds, {w: 30 for w in worlds}, {w: 30 for w in worlds}, "learner", seed, 32)
        rows += _rows("good", worlds[:4], {w: 20 for w in worlds[:4]}, {w: 20 for w in worlds[:4]}, "learner",
                      seed, 64, split="validation")
        rows += _rows("bad", worlds, {w: 40 for w in worlds}, {w: 40 for w in worlds}, "learner", seed, 64)
        rows += _rows("bad", worlds[:4], {w: 40 for w in worlds[:4]}, {w: 40 for w in worlds[:4]}, "learner",
                      seed, 64, split="validation")
    cfg = {"analysis": {"late_from": 2, "n_boot": 2000, "boot_seed": 0, "select_metric": "cost_ratio_late",
                        "select_split": "validation",
                        "tests": [{"name": "p", "role": "primary", "a": "focused_sample", "b": "best_learner",
                                   "metric": "cost_ratio_late", "expect": "less"},
                                  {"name": "d", "a": "focused_sample",
                                   "metric": "cost_ratio_late_minus_early"}]},
           "train": {"budgets": [32, 64]}, "setting": {"family": "recipequest"}}
    out = analyse_rows(cfg, rows)
    assert out["selection"]["best_learner"] == "good"
    p = out["tests"][0]
    assert p["b_resolved"] == "good" and p["n_units"] == 12
    assert p["estimate"] == pytest.approx(1.5 - 2.0) and p["verdict"] == "holds"
    assert out["tests"][1]["estimate"] == pytest.approx(-0.5)
    curve = out["learning_curves"]["good"]
    assert curve["32"]["cost_ratio_late"] == pytest.approx(3.0)
    assert curve["64"]["cost_ratio_late"] == pytest.approx(2.0)


def test_selection_ties_follow_the_config_order_then_the_deterministic_mode():
    from hypergraph_agent.baselines.analysis import analyse_rows
    worlds = ["W0", "W1"]
    same = {w: 20 for w in worlds}
    rows = []
    for arm, group in (("zeta", "g"), ("alpha", "g")):
        for mode in ("stochastic", "deterministic"):
            for r in _rows(arm, worlds, same, same, "learner", 0, 64):
                rows.append({**r, "group": group, "mode": mode})
    cfg = {"analysis": {"late_from": 2, "n_boot": 100, "boot_seed": 0, "select_metric": "cost_ratio_late",
                        "select_split": "test", "tests": []},
           "train": {"budgets": [64]}, "learners": [{"id": "zeta"}, {"id": "alpha"}]}
    out = analyse_rows(cfg, rows)
    assert out["selection"]["best_learner"] == "zeta@deterministic"
    assert out["tuning"]["groups"]["g"]["learner"] == "zeta" and out["tuning"]["groups"]["g"]["mode"] == \
        "deterministic"


def test_shared_ledger_retries_a_replace_blocked_by_a_reader(tmp_path, monkeypatch):
    import os
    from hypergraph_agent.baselines import runner
    led = runner.SharedLedger(tmp_path / "L.json", {"dev": 100}, 100, 100)
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("held open by a reader")
        return real(src, dst)

    monkeypatch.setattr(runner.os, "replace", flaky)
    monkeypatch.setattr(runner.SharedLedger, "PAUSE", 0.0)
    led.register_run("r", "dev", 10)
    led.record("r", "dev", 3, {"exploration": 3}, "adaptive")
    assert calls["n"] == 4 and json.loads((tmp_path / "L.json").read_text())["adaptive_total"] == 3


def test_baselines_lines_are_short():
    files = [*PKG.glob("*.py"), pathlib.Path(__file__), *CONFIGS]
    long = [f"{f.name}:{n}" for f in files for n, line in enumerate(f.read_text().splitlines(), 1)
            if len(line) > 110]
    assert not long, long
