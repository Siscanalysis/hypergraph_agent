"""Study U2 baselines: IW(1) and IW(2), flat macro expansion, the archive explorer; their charging,
soundness and information boundary; the U2 configs and the runner."""

import ast
import itertools
import pathlib
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from hypergraph_agent.techtree.baselines import (
    BASELINES, DEFAULT_PARAMS, IW1, IW2, ROW_SUMMARIES, Archive, IteratedWidth, MacroSystematic, check_params,
    counter_summary, decode, make_baseline, subsets,
)
from hypergraph_agent.techtree.config import load_config, variant_arms, variants
from hypergraph_agent.techtree.env import TechTreeEnv, agent_interface
from hypergraph_agent.techtree.explorers import Episode
from hypergraph_agent.techtree.generator import TechStream, TechStreamConfig, TechWorldConfig

from helpers import meter
from test_techtree import ALLOWED_IMPORTS, _soundness, boundary_violations, run, stream, u_world

ROOT = pathlib.Path(__file__).resolve().parents[1]
TT = ROOT / "src" / "hypergraph_agent" / "techtree"
U2_CONFIGS = sorted((ROOT / "configs" / "unlock2").glob("*.yaml"))
U2_DOC = ROOT / "docs" / "studies" / "U2_techtree_baselines.md"
L_WORLD = TechWorldConfig()  # 3 primitives, levels 1-3 with three concepts each, combos of 2


def rng(seed=0):
    return np.random.default_rng(seed)


def blocked_task(k=3, budget=6000):
    """A silent U task (7 actions) whose key is typed only with its own, still inactive, slot:
    nothing can ever fire, and conditional silent eliminations keep the level open for the engine,
    so search passes run to their end."""
    t = stream(u_world(k=k, P=4, slots=3), rule="unlock", vis="silent", epw=1, budget=budget).task(0, 0)
    w = t.world
    (key, slot), = w.unlocks
    seqs = list(w.sequences)
    seqs[key] = (slot,) * k
    return replace(t, world=replace(w, sequences=tuple(seqs)))


def episode_on(task):
    penv, token_for = agent_interface(TechTreeEnv())
    return Episode(penv, token_for(task), meter())


class LogEvaluations:
    def evaluate(self, tree, node, ep):
        before = tree.novel
        super().evaluate(tree, node, ep)
        self.log.append((tree.passes, tree.width, node, tree.novel > before))


class LoggedIW1(LogEvaluations, IW1):
    pass


class LoggedIW2(LogEvaluations, IW2):
    pass


class LoggedIW(LogEvaluations, IteratedWidth):
    pass


class LoggedMacro(MacroSystematic):
    def _search(self, k, ep, level):
        chunk = super()._search(k, ep, level)
        self.log.append(chunk.symbols)
        return chunk


class LoggedArchive(Archive):
    def _search(self, k, ep, level):
        chunk = super()._search(k, ep, level)
        if chunk.kind == "search_primitive":
            self.log.append((ep.held, self.world_state(ep)["target"]))
        return chunk


def logged(cls, seed=0):
    agent = cls(rng(seed)).configure()
    agent.log = []
    return agent


# ----------------------------------------------------------------------- IW(k)
@pytest.mark.parametrize("width", [1, 2])
def test_one_iw_pass_expands_only_novel_nodes(width):
    t = blocked_task(k=3)
    agent = logged(LoggedIW1 if width == 1 else LoggedIW2)
    run(agent, TechTreeEnv(), t, meter(10_000))
    tree = agent.per_world[t.world.world_key]["trees"][frozenset()]
    assert tree.passes >= 2  # the first pass ended without any firing
    first = [(node, novel) for p, _, node, novel in agent.log if p == 1]
    novel = {node for node, nov in first if nov}
    assert all(len(node) == 1 or node[:-1] in novel for node, _ in first)  # children of novel nodes only
    assert len({node for node, _ in first}) == len(first)  # each node generated once per pass

    def depth(d):
        return [node for node, _ in first if len(node) == d]

    order = [node[0] for node in depth(1)]
    assert sorted(order) == list(range(7)) and len(depth(2)) == 49
    f = order[0]
    if width == 1:
        # novelty at width 1 keeps one child per depth-1 node: (a, f), f the first successor
        assert {node for node in depth(2) if node in novel} == {(a, f) for a in order}
        assert set(depth(3)) == {(a, f, c) for a in order for c in order}  # a seventh of the 343
    else:
        assert set(depth(2)) <= novel
        assert set(depth(3)) == set(itertools.product(order, repeat=3))  # every 3-press sequence


def test_iw_route_reuses_the_trace_only_where_the_state_allows():
    agent = make_baseline("iw2", rng())
    ep = episode_on(stream(u_world(k=3, P=4, slots=3), rule="unlock").task(0, 0))
    k = agent.knowledge(ep.spec)
    wait = ep.spec.wait_action
    assert agent.route(k, ep, (1, 2), 3) == (1, 2)  # empty trace: no clear needed
    ep.step(1)
    assert agent.route(k, ep, (1, 2), 3) == (2,)  # the trace is a prefix of the node
    assert agent.route(k, ep, (0, 2), 3) == (wait, 0, 2)  # a short node needs the exact trace
    ep.step(2)
    ep.step(3)
    assert agent.route(k, ep, (2, 3, 0), 3) == (0,)  # a full node: shortest completing suffix
    assert agent.route(k, ep, (3, 1, 1), 3) == (1, 1)
    assert agent.route(k, ep, (0, 0, 0), 3) == (0, 0, 0)


def test_stale_slot_presses_are_not_reused_in_silent_mode():
    agent = make_baseline("iw2", rng())
    ep = episode_on(stream(u_world(k=3, P=4, slots=3), rule="unlock", vis="silent").task(0, 0))
    k = agent.knowledge(ep.spec)
    ep.trace, ep.trace_ctx, ep.held = [4, 0], [frozenset(), frozenset()], frozenset({0})
    assert agent.reusable(k, ep, 1) and not agent.reusable(k, ep, 2)
    ep.trace_ctx = [frozenset({0}), frozenset({0})]
    assert agent.reusable(k, ep, 2)


@pytest.mark.parametrize("vis", ["announced", "silent"])
def test_macro_and_archive_arms_are_signal_blind(vis):
    """The macro arms and the archive never read the signal, so they act identically with and
    without it (the IW arms read it as an atom and are run in the signal cells)."""
    for k in (2, 3):
        for arm in ("macro_uniform", "macro_sys", "archive"):
            rows = []
            for sig in (False, True):
                s = stream(u_world(k=k, P=4, slots=3), rule="unlock", vis=vis, signal=sig, n_worlds=2, epw=4,
                           seed=k)
                agent, env, m = make_baseline(arm, rng(3)), TechTreeEnv(), meter(100_000)
                rows.append([(r["primitive_length"], r["fired"])
                             for r in (run(agent, env, s.task(w, e), m) for w, e in s.order())])
            assert rows[0] == rows[1], (arm, k)


def test_iw_trees_are_rooted_where_the_context_was_entered():
    """The first tree of an episode starts at the empty trace; the tree of the key's context starts
    at the trace that fired the key, so with a full root no node of that context needs a clear."""
    t = stream(u_world(k=3, P=4, slots=3), rule="unlock", epw=1, budget=400).task(0, 0)
    (key, _), = t.world.unlocks
    agent, env = make_baseline("iw2", rng()), TechTreeEnv()
    pressed, step = [], env.step
    env.step = lambda a: (pressed.append(a), step(a))[1]
    row = run(agent, env, t, meter())
    trees = agent.per_world[t.world.world_key]["trees"]
    fired_at = next(s for s, c, _ in row["fired"] if c == key)
    assert trees[frozenset()].root == ()
    assert trees[frozenset({key})].root == t.world.sequences[key]  # L = k: the key's own presses
    assert row["success"] and env.public_spec.wait_action not in pressed[fired_at:]


def test_iterated_width_raises_the_width_after_each_exhausted_pass():
    t = blocked_task(k=2, budget=2500)
    agent = logged(LoggedIW)
    row = run(agent, TechTreeEnv(), t, meter())
    widths = {p: w for p, w, _, _ in agent.log}
    assert widths[1] == 1 and widths[2] == 2 and max(widths) >= 3  # 2 = L, the varying atoms here
    assert all(w == 2 for p, w in widths.items() if p >= 2) and row["iw_width"] == 2
    assert len(list(subsets([1, 2, 3, 4], 2))) == 6 and list(subsets([1, 2], 0)) == [()]


def test_counter_summary():
    base = {"variant": "v", "macro_steps": 0}
    rows = [{**base, "arm": "archive", "archive_selected": 1, "archive_cell_size": 3, "archive_cells": 2,
             "archive_explore": 10},
            {**base, "arm": "archive", "archive_selected": 0, "archive_cell_size": 0, "archive_cells": 4,
             "archive_explore": 0},
            {**base, "arm": "iw", "iw_passes": 1, "iw_generated": 5, "iw_novel": 3, "iw_pruned": 2,
             "iw_width": 2},
            {**base, "arm": "pooled"}]
    s = counter_summary(rows)
    assert set(s) == {"v/archive", "v/iw"} and ROW_SUMMARIES == {"baseline_counters": counter_summary}
    assert s["v/archive"]["archive_cell_size_mean_selected"] == 3 and s["v/archive"]["archive_cells_max"] == 4
    assert s["v/archive"]["archive_explore"] == 10 and s["v/archive"]["episodes"] == 2
    assert s["v/iw"]["iw_width_max"] == 2 and s["v/iw"]["iw_generated"] == 5
    assert "macro_steps" not in s["v/iw"]


# ----------------------------------------------------------------- flat macros
def test_decode_enumerates_sequences_in_lexicographic_order():
    acts = (0, 2, 5)
    assert [decode(acts, 2, i) for i in range(9)] == list(itertools.product(acts, repeat=2))
    assert decode(acts, 1, 2) == (5,)


def test_macro_sys_runs_every_macro_once_per_pass_shortest_first():
    t = blocked_task(k=2, budget=200)
    agent = logged(LoggedMacro)
    run(agent, TechTreeEnv(), t, meter())
    macros, acts = agent.log, range(7)
    assert sorted(macros[:7]) == [(a,) for a in acts]
    assert sorted(macros[7:56]) == sorted(itertools.product(acts, repeat=2))
    assert sorted(macros[56:63]) == [(a,) for a in acts]  # a new pass in new random orders
    assert macros[7:56] != sorted(macros[7:56])


def test_macro_uniform_draws_uniformly_from_the_expanded_action_set():
    agent = make_baseline("macro_uniform", rng())
    ep = episode_on(stream(u_world(k=3, P=4, slots=3), rule="unlock").task(0, 0))
    k = agent.knowledge(ep.spec)
    draws = [agent._search(k, ep, 1).symbols for _ in range(8400)]  # no action is executed
    counts = Counter(len(d) for d in draws)
    assert abs(counts[1] - 400) < 80 and abs(counts[2] - 1600) < 160 and abs(counts[3] - 6400) < 240
    assert all(s < 4 for d in draws for s in d)  # slots are listed only once available (announced)
    assert len(set(d for d in draws if len(d) == 3)) == 64
    # the concatenated presses are an i.i.d. uniform stream, as uniformly random primitives are
    stream_ = [s for d in draws for s in d]
    pairs = Counter(zip(stream_, stream_[1:]))
    expected = (len(stream_) - 1) / 16
    assert len(pairs) == 16 and all(abs(n - expected) < 0.1 * expected for n in pairs.values())


def test_macro_bound_parameter():
    agent = logged(LoggedMacro)
    agent.configure({"macro_bound": 2})
    agent.log = []
    run(agent, TechTreeEnv(), blocked_task(k=3, budget=300), meter())
    assert agent.log and max(len(m) for m in agent.log) == 2 and len(agent.log) > 7 + 49


# --------------------------------------------------------------------- archive
def test_archive_selects_the_largest_cell_then_the_least_selected():
    agent = make_baseline("archive", rng())
    t = TechStream(TechStreamConfig("unit", 0, 1, 1, L_WORLD, (3,), "uniform", 100)).task(0, 0)
    ep = episode_on(t)
    k = agent.knowledge(ep.spec)
    cells = {frozenset(): 0, frozenset({0}): 3, frozenset({1}): 1, frozenset({0, 1}): 5, frozenset({1, 2}): 2}
    agent.per_world[ep.spec.world_key] = {"cells": cells}
    chunk = agent._search(k, ep, 1)
    st = agent.world_state(ep)
    assert st["target"] == frozenset({1, 2}) and cells[frozenset({1, 2})] == 3
    assert chunk.kind == "search_primitive" and len(chunk.symbols) == 1  # nothing known to replay yet
    assert agent._search(k, ep, 1).kind == "search_primitive" and cells[frozenset({1, 2})] == 3


def test_archive_explores_only_from_its_selected_cell():
    s = TechStream(TechStreamConfig("unit", 1, 2, 6, replace(L_WORLD, reuse_depth=2), (3,), "uniform", 100))
    agent, env, m = logged(LoggedArchive), TechTreeEnv(), meter(100_000)
    tasks = [s.task(w, e) for w, e in s.order()]
    for t in tasks:
        run(agent, env, t, m)
    assert agent.log and all(target <= held for held, target in agent.log)
    assert any(len(target) > 3 for _, target in agent.log)  # cells beyond the engine's gating
    _soundness(agent, tasks)


# ------------------------------------------------------- shared engine, charging
@pytest.mark.parametrize("arm", sorted(BASELINES))
def test_every_press_is_charged(arm):
    s = stream(u_world(k=3, P=4, slots=3), rule="unlock", n_worlds=1, epw=3)
    agent, env, m = make_baseline(arm, rng()), TechTreeEnv(), meter(100_000)
    pressed, step = [], env.step
    env.step = lambda a: (pressed.append(a), step(a))[1]
    rows = [run(agent, env, s.task(0, e), m) for e in range(3)]
    assert m.used == len(pressed) == sum(r["primitive_length"] for r in rows)
    for r in rows:
        assert r["search_steps"] + r["replay_steps"] + r["idle_steps"] + r["success"] == r["primitive_length"]
        if arm.startswith("macro"):
            assert r["macro_steps"] == r["search_steps"]


@pytest.mark.parametrize("arm", sorted(BASELINES))
def test_baselines_keep_the_shared_evidence_sound_and_replay_after_success(arm):
    for k, vis, sig in itertools.product((2, 3), ("announced", "silent"), (False, True)):
        s = stream(u_world(k=k, P=3, slots=2), rule="unlock", vis=vis, signal=sig, n_worlds=2, epw=5,
                   budget=150, seed=k)
        tasks = [s.task(w, e) for w, e in s.order()]
        agent, env, m = make_baseline(arm, rng()), TechTreeEnv(), meter(100_000)
        rows = [run(agent, env, t, m) for t in tasks]
        _soundness(agent, tasks)
        for i in range(0, len(rows), 5):
            first = next((j for j in range(5) if rows[i + j]["success"]), None)
            if first is not None:
                later = rows[i + first + 1:i + 5]
                assert all(r["success"] and r["primitive_length"] <= 2 * k + 3 for r in later)


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("arm", sorted(BASELINES))
def test_baselines_are_sound_in_the_layered_world(arm, depth):
    world = replace(L_WORLD, reuse_depth=depth)
    s = TechStream(TechStreamConfig("unit", 2, 2, 4, world, (3,), "uniform", 100))
    tasks = [s.task(w, e) for w, e in s.order()]
    agent, env, m = make_baseline(arm, rng()), TechTreeEnv(), meter(100_000)
    rows = [run(agent, env, t, m) for t in tasks]
    _soundness(agent, tasks)
    assert m.used == sum(r["primitive_length"] for r in rows)


def test_factory_parameters_and_common_random_numbers():
    assert check_params(None) == DEFAULT_PARAMS
    with pytest.raises(KeyError):
        check_params({"width": 3})
    with pytest.raises(ValueError):
        check_params({"macro_bound": 0})
    with pytest.raises(ValueError):
        check_params({"trace_atoms": True})
    with pytest.raises(KeyError):
        make_baseline("dqn", rng())
    seeds = {make_baseline(a, rng(7)).pref_seed for a in [*BASELINES, "pooled", "remember", "promote"]}
    assert len(seeds) == 1  # every arm of a seed starts from the same per-world preference seed
    assert type(make_baseline("pooled", rng())).__name__ == "Explorer"
    assert type(make_baseline("promote", rng())).__name__ == "Promote"


# ------------------------------------------------------------ information boundary
def test_baselines_module_passes_the_source_check():
    src = (TT / "baselines.py").read_text()
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


# -------------------------------------------------------------- configs, runner
@pytest.mark.parametrize("path", U2_CONFIGS, ids=lambda p: p.name)
def test_u2_configs_load_plan_and_build_every_arm(path):
    from hypergraph_agent.techtree.study import make_agent, plan, reporting_arms
    cfg = load_config(path)
    out = plan(cfg)
    assert cfg["run"]["study"] == "U2" and out["runs"] > 0
    assert reporting_arms("U2") == ("random", "oracle", "oracle_library", "reference")
    for v in variants(cfg):
        for arm in variant_arms(cfg, v):
            if arm != "reference":
                make_agent("U2", arm, rng(), cfg["agent"])
    namespaces = {s["namespace"] for s in out["variants"]}
    alloc = cfg["ledger"]["allocations"]
    if "dev" in path.name:
        assert cfg["run"]["allocation"] == "dev" and all(n.endswith("_dev") for n in namespaces)
        assert out["caps"]["adaptive_total"] <= alloc["dev"]
    else:
        assert cfg["run"]["requires_flag"] and namespaces == {"techtree_u2", "techtree_u2l"}
        assert out["caps"]["adaptive_total"] == alloc["u2_main"]
        assert cfg["ledger"]["adaptive_cap"] == sum(alloc.values())
        assert out["caps"]["reporting_total"] <= cfg["ledger"]["reporting_cap"] - 20_000
    others = [p for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name != "unlock2"]
    assert not any("techtree_u2" in p.read_text() for p in others)  # fresh namespaces


def test_u2_configs_share_one_ledger_block_and_the_main_needs_the_flag():
    from hypergraph_agent.techtree.__main__ import main
    cfgs = [load_config(p) for p in U2_CONFIGS]
    assert len({repr(c["ledger"]) for c in cfgs}) == 1 and len({repr(c["agent"]) for c in cfgs}) == 1
    main_cfg = next(p for p in U2_CONFIGS if "main" in p.name)
    assert main(["--config", str(main_cfg), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        main(["--config", str(main_cfg)])


def test_runner_and_analysis_end_to_end(tmp_path):
    from hypergraph_agent.techtree.analysis import analyse, collect_runs
    from hypergraph_agent.techtree.config import run_hash
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "u2.yaml"
    p.write_text(f"""
run: {{name: u2-e2e, study: U2, seeds: [0], runs_dir: {(tmp_path / 'runs').as_posix()}}}
agent: {{trace_atoms: max_length, macro_bound: max_length}}
world: {{n_primitives: 3, n_slots: 2, n_unlocks: 1, n_levels: 1, concepts_per_level: [2], combo_length: 2}}
stream: {{namespace: unit_dev, n_worlds: 1, episodes_per_world: 3, goal_levels: [1], goal_rule: unlock,
          episode_budget: 80}}
variants:
  - {{name: k2, arms: [pooled, iw, iw1, iw2, macro_uniform, macro_sys, archive, reference]}}
  - {{name: l-d2, world: {{n_slots: 0, n_unlocks: 0, n_levels: 3, concepts_per_level: [3, 3, 3],
                          reuse_depth: 2}},
     stream: {{goal_levels: [3], goal_rule: uniform}}, arms: [remember, archive, reference]}}
arms: [pooled, remember, iw, iw1, iw2, macro_uniform, macro_sys, archive, reference]
analysis:
  tests:
    - {{type: paired, name: p, metric: first_success_steps, a: iw2, b: pooled, variant: k2, expect: greater}}
    - {{type: within_margin, name: wm, metric: steps_late, a: remember, others: [archive], variants: [l-d2],
       margin_rel: 0.0}}
    - {{type: all_success, name: ref_ok, arm: reference}}
  verdict: {{rule: all, require: [ref_ok]}}
""")
    cfg = load_config(p)
    ledger = SessionLedger(None, {"dev": 100_000}, 100_000, 100_000)
    results = run_study(cfg, ledger, str(tmp_path / "runs"))
    assert len(results) == 11
    runs = collect_runs(tmp_path / "runs", "u2-e2e")
    rows = [row for r in runs for row in r["rows"]]
    assert all(r["complete"] for r in runs) and len(rows) == 11 * 3
    ref = sum(r["primitive_length"] for r in rows if r["arm"] == "reference")
    assert ledger.state["reporting_total"] == ref
    assert ledger.state["adaptive_total"] == sum(r["primitive_length"] for r in rows) - ref
    assert all("iw_passes" in r for r in rows if r["arm"].startswith("iw"))
    assert all("archive_cell_size" in r for r in rows if r["arm"] == "archive")
    res = analyse(cfg, runs)
    assert not res["incomplete_cells"] and res["verdict"]["verdict"] == "pass"
    counters = res["row_summaries"]["baseline_counters"]
    assert {"k2/iw", "k2/iw2", "k2/macro_sys", "l-d2/archive"} <= set(counters)
    assert "k2/pooled" not in counters
    bare = {**cfg, "agent": {}}
    v = variants(cfg)[0]
    assert run_hash(cfg, v, "iw2", 0) != run_hash(bare, v, "iw2", 0)  # the declared parameters are hashed


def test_u2_files_have_short_lines():
    files = [TT / "baselines.py", pathlib.Path(__file__), *U2_CONFIGS] + ([U2_DOC] if U2_DOC.exists() else [])
    long = [f"{f.name}:{n}" for f in files
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if len(line) > 110]
    assert not long, long
