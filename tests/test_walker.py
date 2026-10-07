import numpy as np
import pytest
import torch

from hypergraph_agent.agents.edit_policy import (
    EditPolicy, benchmark_search, search_problems, train_edit_policy,
)
from hypergraph_agent.agents.walker import (
    Walker, apply_edit, focused_edits, neighbours, simulate,
)
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import reference_solve
from hypergraph_agent.skills.executor import Executor

from helpers import CHAIN_FACTS, T, chain_world, meter, task


def stream(observe="all", seed=4, n_worlds=2, depth=(1, 3), factor=1.0):
    return TaskStream(TaskStreamConfig(
        "val", seed, "pool", n_worlds, 0, WorldConfig(),
        TaskConfig(profile=PROFILE_UNKNOWN, depth_min=depth[0], depth_max=depth[1], n_distractor_base=0,
                   observe_items=observe, budget_factor=factor)))


def play(walker, tasks, cap=10_000):
    m, env, rows = meter(cap), RecipeQuestEnv(), []
    for i, t in enumerate(tasks):
        rows.append(walker.run_episode(Executor(env, m, "exploration", torch.Generator().manual_seed(i)), t, i))
    return rows


def true_node(t):
    sig = {r: r.signature(t.profile) for r in t.rules}
    return {sig[r]: frozenset(r.true_base) for r in t.rules}


def test_planner_on_the_true_structure_is_optimal():
    t = task(chain_world(), "key", CHAIN_FACTS)  # known structure
    w = Walker("maximal", np.random.default_rng(0))
    ex = Executor(RecipeQuestEnv(), meter(), "exploration", torch.Generator())
    ex.reset(t, 0)
    ws = w.world(ex.spec.world_key)
    ws.register(ex.spec, w.rng)
    plan = w.plan(ex.spec, ws, {}, frozenset())
    assert len(plan) == reference_solve(t).length


def test_neighbours_stay_inside_the_hypothesis_class():
    s = stream()
    w = Walker("local_focused", np.random.default_rng(0))
    play(w, [s.task(0)])
    ws = next(iter(w.worlds.values()))
    sigs = ws.hidden()
    for e in neighbours(ws.node, ws.infos, sigs, 3):
        new = apply_edit(ws.node, e)[e.signature]
        old = ws.node[e.signature]
        assert 1 <= len(new) <= 3 and new <= set(ws.infos[e.signature].pool)
        assert len(new ^ old) in (1, 2) and new != old


@pytest.mark.parametrize("observe", ["all", "goal_only"])
def test_replay_semantics_match_the_environment(observe):
    s = stream(observe)
    w = Walker("local_focused", np.random.default_rng(1))
    tasks = [s.task(i) for i in range(6)]
    play(w, tasks)
    for ws in w.worlds.values():
        truth = {}
        for t in tasks:
            if t.world.world_key == ws.world_key:
                truth.update(true_node(t))
        node = {sig: truth.get(sig, ws.node[sig]) for sig in ws.node}
        for log in ws.evidence.logs:
            assert simulate(node, log, ws.infos)[0] == []  # the hidden truth explains every observation


def test_goal_only_observations_hide_intermediate_items():
    s = stream("goal_only", depth=(2, 3))
    t = s.task(0)
    env = RecipeQuestEnv()
    obs, _ = env.reset(seed=0, options={"task": t})
    spec = env.public_spec
    assert not spec.items_observable
    plan = reference_solve(t).plan
    idx = spec.action_index()
    goal = spec.goal
    for key in plan:
        obs, r, term, trunc, info = env.step(idx[key])
        a = spec.actions[idx[key]]
        if a.rule is not None and spec.rules[a.rule].effect != goal:
            assert info["changed"] is None and obs.last_changed is None
        for i, f in enumerate(spec.facts):
            if f.kind == 2 and i != goal:
                assert obs.present[i] is False
    assert r == 1.0


def test_local_walks_and_enumeration_find_consistent_nodes():
    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(6)])
    ws = max(w.worlds.values(), key=lambda x: len(x.evidence.logs))
    for strategy in ("local_uniform", "local_focused", "exact"):
        probe = Walker(strategy, np.random.default_rng(3), exact_cap=200_000)
        probe.worlds[ws.world_key] = ws
        node, st = probe.choose_node(ws, None)
        assert ws.evidence.violations(node) == 0, (strategy, st)


def test_focused_edits_target_violations_only():
    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(4)])
    ws = max(w.worlds.values(), key=lambda x: len(x.evidence.logs))
    full = {sig: frozenset(ws.infos[sig].pool) for sig in ws.node}
    # with every candidate required, all observed successes become violations of kind "U"
    viol = sum(len(simulate(full, lg, ws.infos)[0]) for lg in ws.evidence.logs)
    edits = focused_edits(full, ws.evidence.logs, ws.infos, 6)
    assert (viol == 0) == (not edits)
    assert all(e.remove is not None for e in edits)


def test_maximal_node_always_fits_the_public_budget():
    # the public budget is the brute-force plan, so the full-pool node always succeeds deterministically
    s = stream()
    rows = play(Walker("maximal", np.random.default_rng(0)), [s.task(i) for i in range(8)])
    assert all(r["success"] for r in rows)


@pytest.mark.parametrize("strategy", ["sample", "optimistic"])
def test_factorized_walkers_solve_and_learn(strategy):
    s = stream(seed=11, n_worlds=1)
    w = Walker(strategy, np.random.default_rng(0))
    rows = play(w, [s.task(i) for i in range(6)])
    assert sum(r["success"] for r in rows) >= 4
    ws = next(iter(w.worlds.values()))
    assert any(b.n_informative > 0 for b in ws.dep.rules.values())


def test_episodic_walkers_refuse_noisy_dynamics():
    t = task(chain_world(), "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN)
    w = Walker("local_uniform", np.random.default_rng(0), epsilon=0.1)
    with pytest.raises(ValueError):
        w.run_episode(Executor(RecipeQuestEnv(), meter(), "exploration", torch.Generator()), t, 0)


def test_consistent_mixing_never_leaves_the_consistent_set():
    from hypergraph_agent.agents.walker import mix_consistent
    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(6)])
    ws = max(w.worlds.values(), key=lambda x: len(x.evidence.logs))
    node, _ = Walker("local_focused", np.random.default_rng(3)).choose_node(ws, None)
    assert ws.evidence.violations(node) == 0
    before = ws.evidence.evaluations
    mixed, st = mix_consistent(node, ws, None, np.random.default_rng(4), 3, 60)
    assert ws.evidence.violations(mixed) == 0
    assert st["mix_evals"] == ws.evidence.evaluations - before - 1 and st["mix_accepted"] > 0


@pytest.mark.parametrize("strategy", ["focused_sample", "learned_sample"])
def test_sampling_walkers_plan_on_consistent_nodes(strategy):
    s = stream("goal_only", seed=7)
    torch.manual_seed(0)
    w = Walker(strategy, np.random.default_rng(5), policy=EditPolicy(8), consistent_moves=20)
    rows = play(w, [s.task(i) for i in range(6)])
    assert any(r["success"] for r in rows)
    for ws in w.worlds.values():
        node, st = w.choose_node(ws, None)  # a fresh replan on all logged evidence
        assert ws.evidence.violations(node) == 0 and st.get("mix_evals", 0) > 0


def test_saved_edit_policy_is_reused_with_its_cost(tmp_path):
    import json
    from hypergraph_agent.evaluation.walkers import load_policy
    run = tmp_path / "walk-b-collector-s3-x"
    run.mkdir()
    torch.manual_seed(0)
    pol = EditPolicy(8)
    torch.save(pol.state_dict(), run / "edit_policy.pt")
    (run / "manifest.json").write_text(json.dumps(
        {"run_id": run.name, "interactions": {"physical": 777}, "policy_training": {"iters": 5}}))
    wc = {"policy_from": str(tmp_path / "walk-b-collector-s{seed}-*" / "edit_policy.pt"), "policy": {"hidden": 8}}
    loaded, stats, cost, rid = load_policy(wc, 3)
    assert cost == 777 and rid == run.name and stats == {"iters": 5}
    for a, b in zip(pol.parameters(), loaded.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(FileNotFoundError):
        load_policy(wc, 4)


def test_edit_policy_trains_and_benchmarks():
    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(6)])
    probs = search_problems(w.worlds.values())
    assert probs
    torch.manual_seed(0)
    pol = EditPolicy(8)
    stats = train_edit_policy(pol, probs, iters=20, max_evals=60, temperature=0.5, restart_after=30,
                              lr=1e-2, rng=np.random.default_rng(0))
    assert stats["iters"] == 20 and stats["search_evaluations"] > 0
    bench = benchmark_search(probs[:2], ("uniform", "focused", "learned"), policy=pol, starts=2,
                             max_evals=60, temperature=0.5, restart_after=30)
    assert {m: v["n"] for m, v in bench.items()} == {"uniform": 4, "focused": 4, "learned": 4}
