"""Study PG plumbing (docs/studies/PG_policy_graph.md): the policy-graph agents of TechTree and
RecipeQuest, their information boundary, credit and common random numbers, the configs (one
ledger declaration, caps, X's test streams reproduced task for task) and the evaluator-side
analysis."""

import ast
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hypergraph_agent.techtree.config import load_config, variant_arms, variants
from hypergraph_agent.techtree.env import TechTreeEnv
from hypergraph_agent.techtree.generator import TechStream, TechStreamConfig, TechWorldConfig
from hypergraph_agent.techtree.policygraph import (
    MODES, LoopErasedPath, PolicyGraph, WorldGraph, check_params, make_policygraph, pick,
)

from helpers import meter
from test_techtree import ALLOWED_IMPORTS, boundary_violations, run

ROOT = pathlib.Path(__file__).resolve().parents[1]
TT = ROOT / "src" / "hypergraph_agent" / "techtree"
TT_CONFIGS = sorted((ROOT / "configs" / "policygraph_tt").glob("*.yaml"))
RQ_CONFIGS = sorted((ROOT / "configs" / "policygraph_rq").glob("*.yaml"))
DOC = ROOT / "docs" / "studies" / "PG_policy_graph.md"
FILES = [TT / "policygraph.py", ROOT / "src" / "hypergraph_agent" / "agents" / "policygraph.py",
         ROOT / "src" / "hypergraph_agent" / "evaluation" / "policygraph_study.py", pathlib.Path(__file__),
         *TT_CONFIGS, *RQ_CONFIGS]
K3 = TechWorldConfig(n_primitives=4, n_slots=3, n_unlocks=1, unlock_level=1, n_levels=1,
                     concepts_per_level=(2,), combo_length=3)
L2 = TechWorldConfig(n_primitives=3, n_levels=3, concepts_per_level=(3, 3, 3), combo_length=2, reuse_depth=2)


def rng(seed=0):
    return np.random.default_rng(seed)


def k3_tasks(seed=0, epw=12, world=K3, goal_levels=(1,), rule="unlock", ns="pg_unit_dev"):
    st = TechStream(TechStreamConfig(ns, seed, 1, epw, world, goal_levels, rule, 100, "announced", False))
    return [st.task(w, e) for w, e in st.order()]


def play(agent, tasks):
    env = TechTreeEnv()
    return [run(agent, env, t, meter(100_000)) for t in tasks]


# ------------------------------------------------------------ core
def test_loop_erasure_keeps_the_last_action_of_each_node():
    p = LoopErasedPath()
    for node, a in [("A", 1), ("B", 2), ("A", 3), ("C", 4), ("D", 5), ("C", 6), ("E", 7)]:
        p.visit(node, a)
    assert p.steps == [("A", 3), ("C", 6), ("E", 7)]
    assert p.pos == {"A": 0, "C": 1, "E": 2}


def test_pick_maps_one_variate_like_a_uniform_choice():
    for k in (1, 3, 7):
        for u in np.linspace(0, 0.999, 37):
            assert pick([1.0] * k, u) == int(u * k)
    assert pick([0, 1, 0, 1], 0.2) == 1 and pick([0, 1, 0, 1], 0.7) == 3
    assert pick([1.0, 3.0], 0.3) == 1


def test_weights_by_mode():
    g = WorldGraph(alpha0=1.0)
    acts = (0, 1, 2)
    assert g.weights("n", acts, "thompson") == [1.0, 1.0, 1.0]  # no counts: uniform in every mode
    p = LoopErasedPath()
    p.visit("n", 1)
    g.credit(p)
    g.credit(p)
    assert g.weights("n", acts, "thompson") == [1.0, 3.0, 1.0]
    assert g.weights("n", acts, "greedy") == [0.0, 1.0, 0.0]
    assert g.weights("n", acts, "uniform") == [1.0, 1.0, 1.0]
    assert g.successes == 2 and g.counters()["count_nodes"] == 1


def test_params_are_checked():
    assert check_params(None) == {"order": "max_length_minus_1", "alpha0": 1.0}
    for bad in ({"order": -1}, {"order": "two"}, {"alpha0": 0}, {"alpha0": True}):
        with pytest.raises(ValueError):
            check_params(bad)
    with pytest.raises(KeyError):
        check_params({"temperature": 1})


# ------------------------------------------------------------ TechTree agent
def test_three_arms_act_identically_until_the_first_success():
    tasks = k3_tasks(seed=3)
    rows = {arm: play(make_policygraph(arm, rng(7)), tasks) for arm in MODES}
    first = next(i for i, r in enumerate(rows["pg"]) if r["success"])
    for arm in ("pg_uniform", "pg_greedy"):
        for a, b in zip(rows["pg"][:first + 1], rows[arm][:first + 1]):
            assert (a["primitive_length"], a["fired"], a["success"]) == \
                (b["primitive_length"], b["fired"], b["success"])
    assert first + 1 < len(tasks)


def test_greedy_replays_the_loop_erased_path_after_one_success():
    tasks = k3_tasks(seed=1)
    agent = make_policygraph("pg_greedy", rng(2))
    rows = play(agent, tasks)
    first = next(i for i, r in enumerate(rows) if r["success"])
    assert rows[first]["pg_credited"] == rows[first]["pg_path"] > 0
    nxt = rows[first + 1]
    assert nxt["success"] and nxt["primitive_length"] == rows[first]["pg_path"] + 1  # the path, then submit


def test_counts_come_only_from_successful_episodes():
    tasks = k3_tasks(seed=0)
    agent = make_policygraph("pg", rng(5))
    rows = play(agent, tasks)
    g = next(iter(agent.worlds.values()))
    total = sum(sum(c.values()) for c in g.counts.values())
    assert total == sum(r["pg_credited"] for r in rows) == sum(r["pg_path"] for r in rows if r["success"])
    assert all(r["pg_credited"] == 0 for r in rows if not r["success"])
    assert g.successes == sum(r["success"] for r in rows)


def test_node_order_and_contents_are_public():
    agent = PolicyGraph(rng(), "thompson")
    spec_k3, spec_l2 = (TechTreeEnv() for _ in range(2))
    spec_k3.reset(options={"task": k3_tasks()[0]})
    spec_l2.reset(options={"task": k3_tasks(world=L2, goal_levels=(3,), rule="uniform")[0]})
    assert agent.order(spec_k3.public_spec) == 2 and agent.order(spec_l2.public_spec) == 5
    for presses in range(7):  # the window is the last min(presses, m) presses; m = 5 as in L2
        ep = SimpleNamespace(spec=SimpleNamespace(goal=0), held=frozenset(), trace=list(range(presses)))
        assert agent.node(ep, 5)[2] == tuple(range(max(0, presses - 5), presses))
        assert agent.node(ep, 0)[2] == ()
    agent = make_policygraph("pg", rng(), {"order": 1})
    rows = play(agent, k3_tasks(epw=2))
    nodes = next(iter(agent.worlds.values())).nodes
    assert all(len(n[2]) <= 1 and isinstance(n[1], frozenset) for n in nodes) and rows


def test_factory_builds_every_arm_with_shared_seeds():
    for arm in [*MODES, "pooled", "remember", "promote", "random", "oracle", "oracle_library"]:
        make_policygraph(arm, rng())
    with pytest.raises(KeyError):
        make_policygraph("dqn", rng())
    seeds = {make_policygraph(a, rng(9)).stream_seed for a in MODES}
    assert len(seeds) == 1
    from hypergraph_agent.techtree.study import reporting_arms
    assert reporting_arms("PG") == ("random", "oracle", "oracle_library", "reference", "pg_uniform")


def test_policygraph_module_passes_the_source_check():
    src = (TT / "policygraph.py").read_text()
    assert not boundary_violations(src)
    allowed = ALLOWED_IMPORTS | {"layered"}  # layered is itself an agent module under the same rule
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        assert set(mods) <= allowed, mods


def test_recipequest_agent_never_reads_the_task():
    src = (ROOT / "src" / "hypergraph_agent" / "agents" / "policygraph.py").read_text()
    assert not re.search(r"\btask\.\w", src)  # the task object only goes to the executor's reset
    assert not re.search(r"reference_solver|true_base|\.required\(\)|debug_view|\._task\b", src)


# ------------------------------------------------------------ RecipeQuest agent
def test_recipequest_dispatch_and_names():
    from hypergraph_agent.agents.markov import MARKOV_STRATEGIES
    from hypergraph_agent.agents.policygraph import PG_STRATEGIES, RecipePolicyGraph
    from hypergraph_agent.agents.walker import STRATEGIES
    from hypergraph_agent.config import load_config as load_rq
    from hypergraph_agent.evaluation.walkers import make_walker
    assert not set(PG_STRATEGIES) & (set(STRATEGIES) | set(MARKOV_STRATEGIES))
    wc = load_rq(RQ_CONFIGS[0])["walker"]
    for s in PG_STRATEGIES:
        a = make_walker(s, wc, (0, s, "eps0"), 0.0)
        assert isinstance(a, RecipePolicyGraph) and a.strategy == s
    with pytest.raises(ValueError):
        make_walker("not_a_walker", wc, (0, "x", "y"), 0.0)


def test_recipequest_arms_share_random_numbers_until_a_success(tmp_path):
    from hypergraph_agent.config import load_config as load_rq
    from hypergraph_agent.evaluation import walkers as W
    from hypergraph_agent.training.budget import SessionLedger
    cfg = load_rq(ROOT / "configs" / "policygraph_rq" / "pg_rq_dev.yaml")
    cfg["run"]["seeds"], cfg["eval"]["n_worlds"], cfg["eval"]["n_tasks"] = [0], 1, 16
    cfg["arms"] = [a for a in cfg["arms"] if a["strategy"] in MODES]
    ledger = SessionLedger(None, {"dev": 100_000}, 100_000, 100_000)
    W.run_walker_study(cfg, ledger, str(tmp_path))
    import json
    rows = {}
    for d in tmp_path.iterdir():
        if (d / "eval.jsonl").exists():
            arm = json.loads((d / "config.json").read_text())["arm"]["strategy"]
            rows[arm] = [json.loads(x) for x in (d / "eval.jsonl").read_text().splitlines()]
    assert set(rows) == set(MODES) and all(len(r) == 16 for r in rows.values())
    first = next((i for i, r in enumerate(rows["pg"]) if r["success"]), 15)
    keys = ("primitive_length", "success", "pg_decisions")
    for arm in ("pg_uniform", "pg_greedy"):
        assert [[r[k] for k in keys] for r in rows[arm][:first + 1]] == \
            [[r[k] for k in keys] for r in rows["pg"][:first + 1]]
    assert all(r["pg_credited"] == (r["pg_path"] if r["success"] else 0) for rs in rows.values() for r in rs)


# ------------------------------------------------------------ configs
def ledger_block(path):
    raw = yaml.safe_load(path.read_text())
    if "ledger" in raw and isinstance(raw["ledger"], dict):
        led = raw["ledger"]
        return led["allocations"], led["adaptive_cap"], led["reporting_cap"]
    b = raw["budget"]
    return b["ledger_allocations"], b["ledger_adaptive_cap"], b["ledger_reporting_cap"]


def test_every_pg_config_declares_one_ledger():
    blocks = {repr(ledger_block(p)) for p in TT_CONFIGS + RQ_CONFIGS}
    assert len(blocks) == 1
    alloc, adaptive_cap, _ = ledger_block(TT_CONFIGS[0])
    assert adaptive_cap == sum(alloc.values()) and set(alloc) == {"dev", "pg_tt", "pg_rq"}
    for p in TT_CONFIGS:
        assert load_config(p)["run"]["ledger"] == "runs/PG-ledger.json"
    for p in RQ_CONFIGS:
        assert yaml.safe_load(p.read_text())["run"]["ledger"] == "runs/PG-ledger.json"


def test_techtree_configs_plan_and_cover_their_allocation():
    from hypergraph_agent.techtree.study import plan
    alloc, _, reporting_cap = ledger_block(TT_CONFIGS[0])
    mains = [p for p in TT_CONFIGS if "main" in p.name]
    totals = {p.name: plan(load_config(p))["caps"] for p in TT_CONFIGS}
    assert sum(totals[p.name]["adaptive_total"] for p in mains) == alloc["pg_tt"]
    assert totals["pg_dev.yaml"]["adaptive_total"] <= alloc["dev"] / 2
    for p in TT_CONFIGS:
        cfg = load_config(p)
        assert cfg["run"]["study"] == "PG"
        for v in variants(cfg):
            for arm in variant_arms(cfg, v):
                if arm != "reference":
                    make_policygraph(arm, rng(), cfg["agent"])
        namespaces = {s["namespace"] for s in plan(cfg)["variants"]}
        if "dev" in p.name:
            assert cfg["run"]["allocation"] == "dev" and all(n.endswith("_dev") for n in namespaces)
        else:
            assert cfg["run"]["requires_flag"] and cfg["run"]["allocation"] == "pg_tt"
            assert namespaces <= {"x_tt_u_test", "x_tt_l_test"}
    from hypergraph_agent.config import load_config as load_rq
    from hypergraph_agent.walk import plan as walk_plan
    rq = sum(walk_plan(load_rq(p))["caps"]["worst_case_reporting"] for p in RQ_CONFIGS)
    assert sum(t["reporting_total"] for t in totals.values()) + rq <= reporting_cap


def test_recipequest_configs_cover_every_stream():
    from hypergraph_agent.config import load_config as load_rq
    from hypergraph_agent.walk import plan as walk_plan
    alloc, _, _ = ledger_block(RQ_CONFIGS[0])
    for p in RQ_CONFIGS:
        cfg = load_rq(p)
        out = walk_plan(cfg)
        assert out["caps"]["per_run_cap_covers_every_stream"]
        assert cfg["walker"]["per_run_cap"] == max(out["caps"]["stream_budget_sums"].values())
        if "main" in p.name:
            assert cfg["run"]["allocation"] == "pg_rq" and cfg["eval"]["namespace"] == "x_rq_test"
            assert out["caps"]["worst_case_adaptive"] == alloc["pg_rq"]
        else:
            assert cfg["run"]["allocation"] == "dev" and cfg["eval"]["namespace"].endswith("_dev")
            assert out["caps"]["worst_case_adaptive"] <= alloc["dev"] / 10


def test_measurement_configs_need_the_flag():
    from hypergraph_agent.techtree.__main__ import main
    for p in TT_CONFIGS:
        if "main" in p.name:
            assert main(["--config", str(p), "--dry-run"]) == 0
            with pytest.raises(SystemExit):
                main(["--config", str(p)])


def test_dev_namespaces_are_fresh():
    others = [p for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name not in ("policygraph_tt",
                                                                                       "policygraph_rq")]
    for ns in ("pg_tt_u_dev", "pg_tt_l_dev", "pg_rq_dev"):
        assert not any(ns in p.read_text() for p in others)


# ------------------------------------------------------------ the same tasks as study X
def test_techtree_streams_are_study_x_test_streams():
    from hypergraph_agent.baselines.sources import evaluation_source
    from hypergraph_agent.techtree.config import stream_config
    pairs = {"pg_u_main.yaml": "tt_u_k3_main.yaml", "pg_l_main.yaml": "tt_l2_d2_main.yaml"}
    for mine, theirs in pairs.items():
        cfg = load_config(ROOT / "configs" / "policygraph_tt" / mine)
        x = yaml.safe_load((ROOT / "configs" / "baselines" / theirs).read_text())
        assert cfg["run"]["seeds"] == x["test"]["seeds"]
        for seed in cfg["run"]["seeds"]:
            st = TechStream(stream_config(cfg, variants(cfg)[0], seed)[0])
            a = [(t.task_key, t.world.world_key, t.goal) for t in (st.task(w, e) for w, e in st.order())]
            sp = x["test"]
            src = evaluation_source("techtree", x["setting"], sp["namespace"], sp["base_seed"], seed,
                                    sp["n_worlds"], sp["episodes_per_world"])
            b = []
            while (block := src.next_block()) is not None:
                b += [(it.task.task_key, it.world_key, it.task.goal) for it in block]
            assert a == b


def test_recipequest_stream_is_study_x_test_stream():
    from hypergraph_agent.baselines.sources import evaluation_source
    from hypergraph_agent.config import eval_stream_config
    from hypergraph_agent.config import load_config as load_rq
    from hypergraph_agent.envs.generator import TaskStream
    from hypergraph_agent.evaluation.walkers import eval_order, variant_config
    cfg = load_rq(ROOT / "configs" / "policygraph_rq" / "pg_rq_main.yaml")
    x = yaml.safe_load((ROOT / "configs" / "baselines" / "rq_f1_main.yaml").read_text())
    assert cfg["run"]["seeds"] == x["test"]["seeds"]
    for seed in cfg["run"]["seeds"]:
        sc = eval_stream_config(variant_config(cfg, seed, cfg["walker"]["eval_variants"][0]))
        st = TaskStream(sc)
        order = eval_order(sc, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"])
        a = [(st.task(i).task_key, i) for i in order]
        sp = x["test"]
        src = evaluation_source("recipequest", x["setting"], sp["namespace"], sp["base_seed"], seed,
                                sp["n_worlds"], sp["episodes_per_world"])
        b = []
        while (block := src.next_block()) is not None:
            b += [(it.task.task_key, it.order) for it in block]
        assert a == b  # same tasks, order and stream positions (hence dynamics seeds)


# ------------------------------------------------------------ runner and analysis
def test_runner_and_analysis_end_to_end(tmp_path):
    from hypergraph_agent.techtree.analysis import analyse, collect_runs
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "pg.yaml"
    p.write_text(f"""
run: {{name: pg-e2e, study: PG, seeds: [0], runs_dir: {(tmp_path / 'runs').as_posix()}}}
agent: {{order: max_length_minus_1, alpha0: 1.0}}
world: {{n_primitives: 3, n_slots: 2, n_unlocks: 1, n_levels: 1, concepts_per_level: [2], combo_length: 2}}
stream: {{namespace: unit_dev, n_worlds: 2, episodes_per_world: 4, goal_levels: [1], goal_rule: unlock,
          episode_budget: 60}}
arms: [pg, pg_uniform, pg_greedy, pooled, random, reference]
analysis:
  late_from: 2
  tests:
    - {{type: paired, name: p, metric: steps_late, a: pg, b: pg_uniform, variant: default, expect: less}}
    - {{type: all_success, name: ref_ok, arm: reference}}
  verdict: {{rule: all, require: [ref_ok]}}
""")
    cfg = load_config(p)
    ledger = SessionLedger(None, {"dev": 100_000}, 100_000, 100_000)
    assert len(run_study(cfg, ledger, str(tmp_path / "runs"))) == 6
    runs = collect_runs(tmp_path / "runs", "pg-e2e")
    rows = [row for r in runs for row in r["rows"]]
    reporting = sum(r["primitive_length"] for r in rows if r["arm"] in ("pg_uniform", "random", "reference"))
    assert ledger.state["reporting_total"] == reporting
    assert ledger.state["adaptive_total"] == sum(r["primitive_length"] for r in rows) - reporting
    res = analyse(cfg, runs)
    assert not res["incomplete_cells"] and res["verdict"]["verdict"] == "pass"
    summary = res["row_summaries"]["policy_graph"]["default"]
    assert set(summary) == set(MODES) and summary["pg"]["episodes"] == 8


def test_analysis_helpers():
    from hypergraph_agent.evaluation import policygraph_study as P
    from hypergraph_agent.techtree.analysis import world_metrics
    rows = [{"world_episode": e, "primitive_length": n, "reference_length": 5, "success": n < 100,
             "world_key": "W", "variant": "v", "arm": "a", "world_steps_before": 0, "fired": []}
            for e, n in enumerate([100, 40, 100, 7, 6, 9])]
    acc = 0
    for r in rows:
        r["world_steps_before"], acc = acc, acc + r["primitive_length"]
    mine = P.unit_metrics(rows, 3)
    theirs = world_metrics(rows, 3)[("v", "a")]["W"]
    for k in ("steps_late", "cost_ratio_late", "success_late", "first_success_steps", "cost_ratio_early"):
        assert mine[k] == pytest.approx(theirs[k])
    a = {f"u{i}": {"m": float(i)} for i in range(5)}
    b = {f"u{i}": {"m": float(i) + 1} for i in range(5)}
    assert P.contrast("t", "m", a, b, 5, [])["verdict"] == "holds"
    assert P.contrast("t", "m", a, b, 6, [])["verdict"] == "incomplete"
    assert P.contrast("t", "m", a, b, 5, ["seed 0: missing"])["verdict"] == "incomplete"
    assert P.contrast("t", "m", b, a, 5, [])["verdict"] == "does not hold"
    with pytest.raises(RuntimeError):
        P.check_same_tasks([{"_unit": "u", "world_episode": 0, "reference_length": 5}],
                           {"u": {0: 6}})
    assert P.check_same_tasks([{"_unit": "u", "world_episode": 0, "reference_length": 5}], {"u": {0: 5}}) == 1


# ------------------------------------------------------------ hygiene
def test_pg_files_have_short_lines_and_no_dashes():
    texts = {f: f.read_text(encoding="utf-8") for f in FILES + ([DOC] if DOC.exists() else [])}
    long = [f"{f.name}:{n}" for f, text in texts.items()
            for n, line in enumerate(text.splitlines(), 1) if len(line) > 110]
    assert not long, long
    dashes = (chr(0x2014), " -" + "- ")  # an em dash or a spaced double hyphen, built so this file passes
    assert not [f.name for f, text in texts.items() if any(d in text for d in dashes)]
