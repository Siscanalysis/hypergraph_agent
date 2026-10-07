from dataclasses import replace

import pytest

from hypergraph_agent.envs.public_schema import SUBMIT, WAIT
from hypergraph_agent.representations.features import SkillCandidate

from helpers import CHAIN_FACTS, T, chain_world, task
from skill_fixtures import admit_scripted, cand_for, executor, ingot_key_library


def test_skill_expands_into_counted_primitives():
    lib, ingot, _ = ingot_key_library()
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
    assert out.status == "target_success" and out.tau == 4
    assert ex.n_primitive == 4 and m.used == 4 and ex.env.total_steps == 4


def test_nested_durations_sum_without_double_counting():
    lib, ingot, key = ingot_key_library()
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    out = ex.run_skill(cand_for(ex, "key"), depth=1)
    assert out.status == "target_success" and out.tau == 6 and m.used == 6
    inner = [c for c in ex.call_log if c["depth"] == 2]
    outer = [c for c in ex.call_log if c["depth"] == 1]
    assert inner[0]["tau"] == 4 and inner[0]["ref"] == list(ingot.ref)
    assert outer[0]["tau"] == 6 and outer[0]["ref"] == list(key.ref)


def test_already_satisfied_calls_cost_a_wait_each():
    lib, _, _ = ingot_key_library()
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS, initial=("ingot",)), 0)
    for i in range(3):  # repeated calls cannot loop for free
        out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
        assert out.status == "already_satisfied" and out.tau == 1
    assert m.used == 3


def test_invalid_binding_costs_one_step():
    lib, ingot, _ = ingot_key_library()
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    bad = replace(cand_for(ex, "ingot"), target_fact=ex.spec.fact_by_type()[T["ore"]])
    out = ex.run_skill(bad, depth=1)
    assert out.status == "invalid_binding" and out.tau == 1 and m.used == 1


def test_timeout_and_task_cancellation():
    lib, _, _ = ingot_key_library(timeout_ingot=3)
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
    assert out.status == "timeout" and out.tau == 3 and not out.task_terminal

    lib, _, _ = ingot_key_library()
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS, budget=2), 0)
    out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
    assert out.task_terminal and out.status == "deadline" and out.tau == 2


def test_parent_limit_caps_child_duration():
    lib, ingot, key = ingot_key_library(timeout_key=2)
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    out = ex.run_skill(cand_for(ex, "key"), depth=1)
    assert out.tau == 2 and out.status == "timeout"


def test_maximum_depth_is_enforced():
    lib, _, _ = ingot_key_library()
    ex, m = executor(lib, max_depth=1)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    ex.run_skill(cand_for(ex, "key"), depth=1)
    assert any(c["status"] == "depth_exceeded" and c["tau"] == 1 for c in ex.call_log)


def test_closed_loop_retry_under_stochastic_failure():
    taus = set()
    for seed in range(8):
        lib, _, _ = ingot_key_library(timeout_ingot=20)
        ex, m = executor(lib, seed=seed)
        ex.reset(task(chain_world(), "key", CHAIN_FACTS, failure_prob=0.5, budget=50), seed)
        out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
        assert out.status == "target_success"
        taus.add(out.tau)
    assert max(taus) > 4  # failed attempts were retried, each costing a primitive


def test_skill_controllers_cannot_submit():
    lib, _, _ = ingot_key_library()
    ex, _ = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS), 0)
    struct = ex.structure(target_fact=ex.spec.fact_by_type()[T["ingot"]], include_submit=False)
    assert "submit" not in struct.cand_keys


def test_open_loop_macro_and_retry_wrapper():
    from hypergraph_agent.envs.public_schema import ACTIVATE, CRAFT, GATHER
    from hypergraph_agent.skills.library import SkillLibrary
    seq = ((GATHER, T["ore"]), (GATHER, T["fuel"]), (ACTIVATE, T["furnace_ready"]), (CRAFT, T["ingot"]))
    lib = SkillLibrary()
    admit_scripted(lib, "ingot", 1, timeout=20, macro=seq)
    ex, m = executor(lib)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS, initial=("ore",)), 0)
    out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
    assert out.status == "target_success" and out.tau == 4  # open loop: re-gathers held ore

    lib = SkillLibrary()
    admit_scripted(lib, "ingot", 1, timeout=40, macro=seq, retries=5)
    ex, m = executor(lib, seed=1)
    ex.reset(task(chain_world(), "key", CHAIN_FACTS, failure_prob=0.6, budget=60), 3)
    out = ex.run_skill(cand_for(ex, "ingot"), depth=1)
    assert out.tau % 4 == 0 and out.tau <= 24
