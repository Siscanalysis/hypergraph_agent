from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hypergraph_agent.agents.edit_policy import (
    EditPolicy, benchmark_search, search_problems, train_edit_policy,
)
from hypergraph_agent.agents.walker import (
    EpisodeLog, RecipeInfo, Walker, WorldState, apply_edit, filter_states, focused_edits, mix_consistent,
    neighbours, random_node, simulate,
)
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig
from hypergraph_agent.envs.public_schema import ACTIVATE, CRAFT, GATHER
from hypergraph_agent.topology.inference import enumerate_hypotheses
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import reference_solve
from hypergraph_agent.skills.executor import Executor

from helpers import CHAIN_FACTS, T, chain_world, meter, task


def stream(observe="all", seed=4, n_worlds=2, depth=(1, 3), factor=1.0, eps=0.0):
    return TaskStream(TaskStreamConfig(
        "val", seed, "pool", n_worlds, 0, WorldConfig(),
        TaskConfig(profile=PROFILE_UNKNOWN, depth_min=depth[0], depth_max=depth[1], n_distractor_base=0,
                   observe_items=observe, budget_factor=factor, failure_prob=eps)))


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


# ------------------------------------------------------------ failure noise

def noisy_worlds(observe, eps, seed=7, n=8, strategy="focused_sample"):
    """Worlds of a walker that played noisy tasks (logs with hidden failures)."""
    s = stream(observe, seed=seed, eps=eps)
    w = Walker(strategy, np.random.default_rng(seed), epsilon=eps, consistent_moves=10)
    rows = play(w, [s.task(i) for i in range(n)])
    return w, rows


def brute_likelihood(node, log, infos, eps):
    """Independent check of filter_states: the summed probability of every
    failure pattern that reproduces all observations (no merging or normalizing)."""

    def rec(t, state):
        if t == len(log.steps):
            return 1.0
        kind, x = log.steps[t]
        branches = [(state, 1.0)]
        if kind in (GATHER, ACTIVATE) and log.base_present(t):
            branches = [(state | {x}, 1.0)]
        elif kind == CRAFT:
            info = infos[x]
            req = info.known if info.known is not None else node[x]
            if info.effect not in state and all(b in state for b in req) and all(i in state for i in info.items):
                branches = [(state | {info.effect}, 1 - eps), (state, eps)]
        total = 0.0
        for s, p in branches:
            seen = log.effect_obs[t]
            if p == 0 or (seen is not None and (infos[x].effect in s) != seen) or (log.goal in s) != log.goal_obs[t]:
                continue
            total += p * rec(t + 1, s)
        return total

    return rec(0, frozenset(log.initial))


def nodes_to_check(ws, rng, k=30):
    return [dict(ws.node)] + [random_node(ws, ws.hidden(), rng, ws.node) for _ in range(k)]


@pytest.mark.parametrize("observe", ["all", "goal_only"])
def test_noise_free_likelihood_is_positive_exactly_without_violations(observe):
    s = stream(observe, seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(8)])
    rng, checked, consistent = np.random.default_rng(0), 0, 0
    for ws in w.worlds.values():
        for node in nodes_to_check(ws, rng):
            for log in ws.evidence.logs:
                ll, dist = filter_states(node, log, ws.infos, 0.0)
                clean = not simulate(node, log, ws.infos)[0]
                assert (ll > -np.inf) == clean and ll in (0.0, -np.inf)
                assert len(dist) == (1 if clean else 0)
                checked, consistent = checked + 1, consistent + clean
    assert checked > 100 and 0 < consistent < checked


@pytest.mark.parametrize("observe", ["all", "goal_only"])
def test_filtered_likelihood_matches_enumeration_of_failure_patterns(observe):
    eps = 0.25
    w, _ = noisy_worlds(observe, eps)
    rng, checked, positive = np.random.default_rng(1), 0, 0
    for ws in w.worlds.values():
        for node in nodes_to_check(ws, rng, 10):
            for log in ws.evidence.logs:
                if sum(k == CRAFT for k, _ in log.steps) > 12:
                    continue
                ll = filter_states(node, log, ws.infos, eps)[0]
                ref = brute_likelihood(node, log, ws.infos, eps)
                assert (ll == -np.inf and ref == 0.0) or np.isclose(np.exp(ll), ref, rtol=1e-9)
                checked, positive = checked + 1, positive + (ref > 0)
    assert checked > 50 and positive > 10


def test_under_noise_only_impossible_observations_are_violations():
    eps = 0.1
    w, _ = noisy_worlds("goal_only", eps)
    rng = np.random.default_rng(3)
    for ws in w.worlds.values():
        assert ws.evidence.epsilon == eps
        for node in nodes_to_check(ws, rng):
            assert (ws.evidence.violations(node) == 0) == (ws.evidence.loglik(node) > -np.inf)


def test_noise_free_beliefs_equal_the_replay():
    def replay(ws, node, log):  # the planning beliefs before failure noise was supported
        state = set(log.initial)
        for (kind, x), g in zip(log.steps, log.goal_obs):
            if kind in (GATHER, ACTIVATE):
                state.add(x)
            elif kind == CRAFT:
                info = ws.infos[x]
                req = info.known if info.known is not None else node[x]
                if all(b in state for b in req) and all(i in state for i in info.items):
                    state.add(info.effect)
            (state.add if g else state.discard)(log.goal)
        return frozenset(state)

    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(8)])
    spec, obs = SimpleNamespace(items_observable=False, facts=()), SimpleNamespace(present=())
    rng, n = np.random.default_rng(4), 0
    for ws in w.worlds.values():
        for _ in range(20):
            node = random_node(ws, ws.hidden(), rng, ws.node)
            for log in ws.evidence.logs:
                for k in range(len(log.steps) + 1):  # every prefix, as during an episode
                    part = EpisodeLog(log.log_id, log.goal, log.initial, log.steps[:k], log.goal_obs[:k],
                                      log.effect_obs[:k], log.base_obs[:k])
                    assert w.believed(spec, ws, node, part, obs) == replay(ws, node, part)
                    n += 1
    assert n > 500


def test_noise_free_mixing_makes_the_same_moves_and_draws():
    def old_mix(node, w, log, rng, max_size, moves):  # the consistent moves before noise was supported
        ev = w.evidence
        start = ev.evaluations
        attempted = sorted({s for lg in ev.all_logs(log) for s in lg.attempted() if w.infos[s].known is None})
        cur = dict(node)
        nb = neighbours(cur, w.infos, attempted, max_size)
        accepted = 0
        for _ in range(moves):
            if not nb:
                break
            nxt = apply_edit(cur, nb[int(rng.integers(len(nb)))])
            if ev.violations(nxt, log) != 0:
                continue
            nb2 = neighbours(nxt, w.infos, attempted, max_size)
            if rng.random() < min(1.0, len(nb) / max(len(nb2), 1)):
                cur, nb = nxt, nb2
                accepted += 1
        return cur, {"mix_evals": ev.evaluations - start, "mix_accepted": accepted}

    s = stream("goal_only", seed=7)
    w = Walker("local_focused", np.random.default_rng(2))
    play(w, [s.task(i) for i in range(8)])
    for k, ws in enumerate(w.worlds.values()):
        node, _ = Walker("local_focused", np.random.default_rng(3)).choose_node(ws, None)
        r1, r2 = np.random.default_rng(k), np.random.default_rng(k)
        assert mix_consistent(node, ws, None, r1, 3, 80) == old_mix(node, ws, None, r2, 3, 80)
        assert r1.random() == r2.random()


def test_noisy_consistent_moves_sample_the_posterior():
    """One hidden recipe (41 hypotheses) and noisy goal outcomes: the consistent
    moves visit nodes in proportion to prior x likelihood."""
    eps, pool, goal = 0.2, tuple(range(6)), T["key"]
    ws = WorldState("w", eps, 3)
    ws.infos["r"] = RecipeInfo("r", goal, (), pool, None)
    ws.hypotheses["r"] = [frozenset(pool[p] for p in h) for h in enumerate_hypotheses(6, 3)]
    ws.node["r"] = frozenset({0})
    episodes = [((0, 1), False), ((0, 1, 2), True), ((0, 3), False), ((1, 2), False), ((0, 2), True)]
    for i, (gathered, success) in enumerate(episodes):
        n = len(gathered)
        ws.evidence.logs.append(EpisodeLog(i, goal, frozenset(), [(GATHER, b) for b in gathered] + [(CRAFT, "r")],
                                           [False] * n + [success], [None] * (n + 1), [True] * n + [None]))
    post = np.array([np.exp(sum(filter_states({"r": h}, lg, ws.infos, eps)[0] for lg in ws.evidence.logs))
                     for h in ws.hypotheses["r"]])
    post /= post.sum()
    rng, visits = np.random.default_rng(0), np.zeros(len(post))
    index = {h: i for i, h in enumerate(ws.hypotheses["r"])}
    node, _ = mix_consistent(dict(ws.node), ws, None, rng, 3, 200)  # burn-in
    for _ in range(30_000):
        node, _ = mix_consistent(node, ws, None, rng, 3, 1)
        visits[index[node["r"]]] += 1
    assert 0.5 * np.abs(visits / visits.sum() - post).sum() < 0.05


def test_maximal_beliefs_recover_from_hidden_failures():
    """Goal-only chain under heavy noise: after the goal fails to appear, the
    filtered beliefs stop assuming that the hidden ingot was made, so the
    brute-force node re-crafts it instead of repeating the last craft."""
    t = replace(task(chain_world(), "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN, failure_prob=0.5, budget=60),
                observe_items="goal_only")
    rows = play(Walker("maximal", np.random.default_rng(0), epsilon=0.5), [t] * 20)
    assert all(r["success"] for r in rows)


@pytest.mark.parametrize("strategy", ["focused_sample", "maximal"])
def test_walkers_play_noisy_goal_only_tasks(strategy):
    s = stream("goal_only", seed=7, eps=0.1)
    tasks = [s.task(i) for i in range(10)]
    w = Walker(strategy, np.random.default_rng(7), epsilon=0.1, consistent_moves=10)
    rows = play(w, tasks)
    assert any(r["success"] for r in rows)
    assert all(r["primitive_length"] <= t.budget and r["status"] in ("success", "deadline")
               for r, t in zip(rows, tasks))
    for ws in w.worlds.values():
        for log in ws.evidence.logs:
            assert len(log.base_obs) == len(log.steps)


def test_visible_base_failures_trigger_a_replan():
    s = stream("goal_only", seed=7, eps=0.4)
    rows = play(Walker("maximal", np.random.default_rng(0), epsilon=0.4), [s.task(i) for i in range(6)])
    assert sum(r["base_failures"] for r in rows) > 0
    assert all(r["replans"] > r["base_failures"] for r in rows if r["base_failures"])


def test_walker_epsilon_must_match_the_declared_noise():
    s = stream("goal_only", seed=7, eps=0.1)
    with pytest.raises(ValueError):
        play(Walker("focused_sample", np.random.default_rng(0)), [s.task(0)])
    with pytest.raises(ValueError):
        play(Walker("local_focused", np.random.default_rng(0), epsilon=0.1), [s.task(0)])


def test_random_omission_matches_its_rate_and_ignores_evidence():
    s = stream("goal_only", seed=7)
    with pytest.raises(ValueError):
        Walker("random_omit", np.random.default_rng(0))
    w = Walker("random_omit", np.random.default_rng(0), omit_prob=0.6)
    rows = play(w, [s.task(i) for i in range(8)])
    omitted = sum(r["omitted_candidates"] for r in rows) / sum(r["pool_candidates"] for r in rows)
    assert 0.5 < omitted < 0.7
    ws = next(iter(w.worlds.values()))
    assert not ws.evidence.logs  # nothing is kept: the node never depends on evidence
    nodes = [w.choose_node(ws, None)[0] for _ in range(200)]
    assert all(1 <= len(n[s]) <= len(ws.infos[s].pool) for n in nodes for s in n)
    keep = Walker("random_omit", np.random.default_rng(0), omit_prob=0.0)
    assert keep.choose_node(ws, None)[0] == Walker("maximal", np.random.default_rng(0)).choose_node(ws, None)[0]


def test_walkers_report_their_omission_rate():
    s = stream("goal_only", seed=7)
    rows = play(Walker("focused_sample", np.random.default_rng(5), consistent_moves=10),
                [s.task(i) for i in range(6)])
    assert all(r["pool_candidates"] > 0 and 0 < r["omitted_candidates"] < r["pool_candidates"] for r in rows)
    rows = play(Walker("maximal", np.random.default_rng(5)), [s.task(i) for i in range(6)])
    assert all(r["omitted_candidates"] == 0 for r in rows)


@pytest.mark.parametrize("q, n", [(0.663, 6), (0.699, 8)])
def test_realized_omission_rate_follows_the_keep_one_rule(q, n):
    """Omitting each of n candidates with probability q and keeping one when all
    are omitted removes q - q**n / n of a pool on average (the protocol's q)."""
    ws = WorldState("w", 0.0, 3)
    ws.infos["r"] = RecipeInfo("r", T["key"], (), tuple(range(n)), None)
    w = Walker("random_omit", np.random.default_rng(0), omit_prob=q)
    omitted = np.mean([1 - len(w.choose_node(ws, None)[0]["r"]) / n for _ in range(40_000)])
    assert omitted == pytest.approx(q - q ** n / n, abs=0.005)
