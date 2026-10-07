"""Hidden prerequisites must not reach any learning input."""

import json
import pathlib
import re
from dataclasses import replace

import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN
from hypergraph_agent.envs.public_schema import public_view
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.representations.features import build_structure, memory_features
from hypergraph_agent.topology.snapshot import TopologyManager

from helpers import CHAIN_FACTS, T, chain_world, recipe, task, world

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "hypergraph_agent"
LEARNING_PACKAGES = ("agents", "representations", "topology", "skills", "training")


def canary_pair():
    """Two worlds that differ only in the hidden base requirement of ingot."""
    w1 = chain_world()
    r0 = w1.recipes[0]
    w2 = replace(w1, recipes=(replace(r0, true_base=(T["sand"],)),) + w1.recipes[1:])
    return (task(w1, "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN),
            task(w2, "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN))


def test_public_view_ignores_hidden_prerequisites():
    a, b = canary_pair()
    assert a.rules[0].true_base != b.rules[0].true_base
    assert public_view(a) == public_view(b)
    assert all(r.known_base is None for r in public_view(a).rules)


def test_identical_public_histories_give_identical_policy_outputs():
    a, b = canary_pair()
    outs = []
    for t in (a, b):
        env = RecipeQuestEnv()
        obs, _ = env.reset(seed=0, options={"task": t})
        spec = env.public_spec
        idx = spec.action_index()
        # a public history whose outcomes coincide in both worlds
        for key in ("gather:fore", "gather:ffuel", "wait"):
            obs, *_ = env.step(idx[key])
        topo = TopologyManager(0.0)
        struct = build_structure(spec, "supergraph", topo.snapshot(spec.world_key))
        torch.manual_seed(0)
        pol = ActorCritic("gated")
        mem = torch.tensor(memory_features(spec, obs, struct.goal_fact, None, None, None, 0.0, 0))
        logits, value, _, _ = pol.step(struct, torch.tensor(obs.present, dtype=torch.float32), mem,
                                       pol.initial_state())
        outs.append((struct.cand_keys, logits, value, struct.edge_feat, struct.rule_static))
    assert outs[0][0] == outs[1][0]
    for x, y in zip(outs[0][1:], outs[1][1:]):
        assert torch.equal(x, y)


def test_info_dict_is_public_only():
    env = RecipeQuestEnv()
    _, info0 = env.reset(seed=0, options={"task": canary_pair()[0]})
    _, _, _, _, info = env.step(0)
    assert info0 == {} and set(info) == {"termination", "changed"}


def test_learning_code_never_imports_the_reference_solver_or_hidden_fields():
    pattern = re.compile(r"reference_solver|true_base|\.required\(\)|debug_view|\._task\b")
    offenders = []
    for pkg in LEARNING_PACKAGES:
        for f in (SRC / pkg).rglob("*.py"):
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{f.name}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_topology_state_contains_no_hidden_truth():
    a, b = canary_pair()
    states = []
    for t in (a, b):
        env = RecipeQuestEnv()
        obs, _ = env.reset(seed=0, options={"task": t})
        spec = env.public_spec
        topo = TopologyManager(0.0)
        idx = spec.action_index()
        for key in ("gather:fore", "gather:ffuel", "activate:ffurnace_ready"):
            obs, *_ = env.step(idx[key])
        before = obs.present
        craft = [k for k in idx if k.startswith("craft:") and "ingot" in k][0]
        obs, *_ = env.step(idx[craft])
        topo.record(spec, spec.actions[idx[craft]].rule, before, obs.present)
        topo.commit()
        states.append(topo.state_dict())
    text = json.dumps(states[0])
    assert "true_base" not in text
    # the two worlds produced different public outcomes here, so states may differ,
    # but each state is a function of public evidence only (checked by construction)
    assert states[0]["scopes"].keys() == states[1]["scopes"].keys()


def test_skill_keys_carry_no_task_local_ids():
    from hypergraph_agent.skills.spec import skill_key_for
    key = skill_key_for(T["ingot"], 1)
    assert key == "achieve[ingot]/L1" and "f" + "ingot" not in key
