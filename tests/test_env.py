import numpy as np
import pytest

from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, make_task, make_world, WorldConfig, TaskConfig
from hypergraph_agent.envs.public_schema import public_view
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import bfs_shortest, reference_solve

from helpers import CHAIN_FACTS, T, chain_world, small_stream, task


def act(env, key):
    return env.step(env.public_spec.action_index()[key])


def keyfor(env, kind, name):
    spec = env.public_spec
    for a in spec.actions:
        if a.fact is not None and spec.facts[a.fact].type_id == T[name] and a.key.startswith(kind):
            return a.key
        if a.rule is not None and kind == "craft" and spec.facts[spec.rules[a.rule].effect].type_id == T[name]:
            return a.key
    raise KeyError(name)


def craft_keys(env, name):
    spec = env.public_spec
    return [a.key for a in spec.actions
            if a.rule is not None and spec.facts[spec.rules[a.rule].effect].type_id == T[name]]


def test_conjunction_requires_every_prerequisite():
    env = RecipeQuestEnv()
    env.reset(seed=0, options={"task": task(chain_world(), "key", CHAIN_FACTS)})
    act(env, keyfor(env, "gather", "ore"))
    act(env, keyfor(env, "gather", "fuel"))
    obs, r, term, trunc, info = act(env, craft_keys(env, "ingot")[0])
    assert not info["changed"]  # furnace not ready: one tail fact is not enough
    act(env, keyfor(env, "activate", "furnace_ready"))
    obs, r, term, trunc, info = act(env, craft_keys(env, "ingot")[0])
    assert info["changed"] and obs.present[env.public_spec.fact_by_type()[T["ingot"]]]


def test_alternative_recipes_are_disjunctive():
    env = RecipeQuestEnv()
    t = task(chain_world(), "key", CHAIN_FACTS, initial=("ingot",))
    env.reset(seed=0, options={"task": t})
    act(env, keyfor(env, "gather", "wax"))
    changed = [act(env, k)[4]["changed"] for k in craft_keys(env, "key")]
    assert sum(changed) == 1  # only the wax recipe applies; the other stays ineligible


def test_noop_costs_budget_and_terminal_reward():
    env = RecipeQuestEnv()
    t = task(chain_world(), "key", CHAIN_FACTS, initial=("ingot", "wax"), budget=3)
    obs, _ = env.reset(seed=0, options={"task": t})
    obs, r, term, trunc, info = act(env, "submit")  # goal absent: no-op
    assert (r, term, obs.budget_left, obs.t) == (0.0, False, 2, 1)
    obs, r, term, trunc, info = act(env, craft_keys(env, "ingot")[0])  # ineligible: no-op
    assert not info["changed"] and obs.budget_left == 1
    obs, r, term, trunc, info = act(env, "submit")
    assert r == 0.0 and term and info["termination"] == "deadline" and not trunc


def test_success_reward_is_exactly_one():
    env = RecipeQuestEnv()
    t = task(chain_world(), "key", CHAIN_FACTS, initial=("ingot", "wax"))
    env.reset(seed=0, options={"task": t})
    for k in craft_keys(env, "key"):
        act(env, k)
    obs, r, term, trunc, info = act(env, "submit")
    assert r == 1.0 and term and not trunc and info["termination"] == "success"
    with pytest.raises(RuntimeError):
        act(env, "wait")


def test_deterministic_seed_replay_with_failures():
    stream = small_stream(failure_prob=0.3)
    t = stream.task(0)
    plan = reference_solve(t).plan
    traces = []
    for _ in range(2):
        env = RecipeQuestEnv()
        env.reset(seed=123, options={"task": t})
        idx = env.public_spec.action_index()
        traces.append([act(env, k)[0].present for k in plan if not env._done])
    assert traces[0] == traces[1]


def test_stochastic_failure_preserves_state():
    t = task(chain_world(), "ingot", CHAIN_FACTS, initial=("ore", "fuel", "furnace_ready"),
             failure_prob=0.5, budget=50)
    outcomes = set()
    for seed in range(20):
        env = RecipeQuestEnv()
        obs0, _ = env.reset(seed=seed, options={"task": t})
        obs, r, term, trunc, info = act(env, craft_keys(env, "ingot")[0])
        if not info["changed"]:
            assert obs.present == obs0.present
        outcomes.add(info["changed"])
    assert outcomes == {True, False}


@pytest.mark.parametrize("seed", range(12))
def test_reference_matches_bfs_and_plan_executes(seed):
    t = small_stream(seed=seed, depth=(1, 3)).task(seed)
    ref = reference_solve(t)
    bfs_len, status, _ = bfs_shortest(t, 400_000)
    assert status == "optimal" and ref.exact and ref.length == bfs_len
    env = RecipeQuestEnv()
    env.reset(seed=0, options={"task": t})
    idx = env.public_spec.action_index()
    for k in ref.plan:
        obs, r, term, trunc, info = env.step(idx[k])
    assert r == 1.0 and term


def test_planner_bounds_report_status():
    stream = small_stream(seed=5, depth=(3, 3))
    t = stream.task(0)
    _, status, _ = bfs_shortest(t, max_expansions=3)
    assert status == "limit"
    ref = reference_solve(t, max_derivations=1)
    assert ref.length is not None and ref.lower_bound <= ref.length


def test_unique_producer_chain_length_formula():
    # chain without alternatives: distinct base gathers + crafts + submit
    w = make_world(11, WorldConfig(n_levels=4, items_per_level=2, p_alternative=0.0))
    t = make_task(w, 7, TaskConfig(depth_min=3, depth_max=3, n_distractor_rules=0), "train")
    closure = t.closure_items()
    bases = {b for r in t.rules if r.effect in closure for b in r.true_base}
    assert reference_solve(t).length == len(bases) + len(closure) + 1


def test_budget_is_public_upper_bound():
    for i in range(10):
        t = small_stream(seed=i, depth=(1, 3)).task(i)
        assert reference_solve(t).length <= t.budget


def test_manifest_hash_is_reproducible_and_namespaced():
    a = small_stream("val", seed=1).manifest(5)
    b = small_stream("val", seed=1).manifest(5)
    c = small_stream("test", seed=1).manifest(5)
    assert a["manifest_hash"] == b["manifest_hash"] != c["manifest_hash"]
    assert not {e["task_key"] for e in a["entries"]} & {e["task_key"] for e in c["entries"]}
