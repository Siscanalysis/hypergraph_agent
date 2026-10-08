"""Study L arms: identical engine and evidence, different representation of discoveries."""

from dataclasses import replace

import numpy as np
import pytest

from hypergraph_agent.techtree.env import TechTreeEnv
from hypergraph_agent.techtree.generator import (
    TechStream, TechStreamConfig, TechWorldConfig, reference_solve, revealed_links,
)
from hypergraph_agent.techtree.layered import make_layered

from helpers import meter
from test_techtree import run

WORLD = TechWorldConfig()  # 3 primitives, levels 1-3 with three concepts each, combos of 2


def stream(depth, seed=0, n_worlds=2, epw=4, budget=150):
    return TechStream(TechStreamConfig("unit", seed, n_worlds, epw, replace(WORLD, reuse_depth=depth), (3,),
                                       "uniform", budget))


def play(arm, s, reveal=(1,), seed=0):
    agent, env, m = make_layered(arm, np.random.default_rng(seed)), TechTreeEnv(), meter(200_000)
    rows, tasks = [], []
    for w, e in s.order():
        t = s.task(w, e)
        if arm == "oracle_library":
            agent.reveal(t.world.world_key, revealed_links(t.world))
        elif reveal:
            agent.reveal(t.world.world_key, revealed_links(t.world, reveal))
        rows.append(run(agent, env, t, m))
        tasks.append(t)
    return agent, rows, tasks


def test_promote_beats_remember_when_discoveries_compose():
    # replays of adjacent known concepts also test their concatenation (pooled trace), so
    # remember can find a composition incidentally in a single world; compare totals
    s = stream(2, n_worlds=4, epw=3)
    _, rem, _ = play("remember", s)
    _, pro, _ = play("promote", s)
    assert 2 * sum(r["search_steps"] for r in pro) < sum(r["search_steps"] for r in rem)
    assert all(r["macro_steps"] == 0 for r in rem) and sum(r["macro_steps"] for r in pro) > 0


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_composition_check_matches_the_world(depth):
    s = stream(depth, n_worlds=2, epw=4)
    agent, _, tasks = play("promote", s)
    for t in tasks[::4]:
        k = agent.worlds[t.world.world_key]
        w = t.world
        for c, seq in k.known.items():
            assert seq == w.sequences[c]
            if w.levels[c] >= 2:
                assert k.is_composition(c, seq) == w.compositional[c]
        if depth == 0:
            assert k.comp == 0
        if depth == 2:
            assert k.flat == 0


def test_macro_tests_never_find_flat_concepts():
    s = stream(0, n_worlds=2, epw=4)
    _, rows, _ = play("promote", s)
    assert sum(r["macro_steps"] for r in rows) > 0
    assert sum(r["compositional_new"] for r in rows) == 0


def test_sham_nodes_match_counts_and_lengths_and_are_useless():
    s = stream(2, n_worlds=1, epw=3)
    agent, rows, tasks = play("sham_promote", s)
    k = agent.worlds[tasks[0].world.world_key]
    assert set(k.sham) <= set(k.known)  # nodes are added when a search needs them
    k.ensure_sham(agent.rng)
    assert set(k.sham) == set(k.known)
    assert all(len(k.sham[c]) == len(k.known[c]) for c in k.known)
    assert not set(k.sham.values()) & set(k.known.values())


def test_adaptive_belief_follows_the_world_kind():
    beliefs = {}
    for depth in (0, 2):
        _, rows, _ = play("adaptive", stream(depth, n_worlds=2, epw=4))
        beliefs[depth] = np.mean([rows[i]["belief_compositional"] for i in (3, 7)])
    assert beliefs[0] < 0.3 < 0.7 < beliefs[2]


def test_revealed_links_reach_every_arm_and_none_forgets_the_rest():
    s = stream(1, n_worlds=1, epw=3)
    for arm in ("none", "remember", "promote"):
        agent, rows, tasks = play(arm, s)
        assert all(r["revealed_total"] == 3 for r in rows)
        if arm == "none":
            assert not agent.worlds and all(r["known_total"] - r["revealed_total"] == r["discoveries_new"]
                                            for r in rows)


def test_oracle_library_replays_at_reference_cost():
    s = stream(1, n_worlds=2, epw=3)
    _, rows, tasks = play("oracle_library", s)
    for r, t in zip(rows, tasks):
        ref = reference_solve(t).length
        assert r["success"] and r["search_steps"] == 0 and ref <= r["primitive_length"] <= ref + 6


def test_runner_supplies_links_to_agents_but_not_to_the_reference(tmp_path):
    from hypergraph_agent.techtree.config import load_config
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "l.yaml"
    p.write_text(f"""
run: {{name: l-e2e, study: L, runs_dir: {(tmp_path / 'runs').as_posix()}}}
stream: {{namespace: unit_dev, n_worlds: 1, episodes_per_world: 2, reveal_levels: [1]}}
variants: [{{name: d2, world: {{reuse_depth: 2}}}}]
arms: [none, promote, reference]
""")
    ledger = SessionLedger(None, {"dev": 10_000}, 10_000, 10_000)
    res = run_study(load_config(p), ledger, str(tmp_path / "runs"))
    assert [r["n"] for r in res] == [2, 2, 2]


@pytest.mark.parametrize("arm", ["remember", "promote", "sham_promote", "adaptive", "none"])
def test_every_arm_keeps_sound_evidence(arm):
    from test_techtree import _soundness
    for depth in (0, 2):
        agent, rows, tasks = play(arm, stream(depth, n_worlds=1, epw=3, seed=depth), reveal=())
        _soundness(agent, tasks)
        assert all(r["primitive_length"] <= 150 for r in rows)
