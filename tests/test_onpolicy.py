import numpy as np
import pytest
import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.skills.spec import controller_hash
from hypergraph_agent.topology.snapshot import TopologyManager
from hypergraph_agent.training.ppo import PPOConfig, StaleBatchError, ppo_update
from hypergraph_agent.training.rollout import run_episode

from helpers import CHAIN_FACTS, chain_world, meter, small_stream, task
from skill_fixtures import admit_scripted, executor, ingot_key_library


def _episode(enc, profile="known_structure", mode="known", topo=None, seed=0):
    torch.manual_seed(seed)
    pol = ActorCritic(enc, d=32, hidden=32)
    t = small_stream(profile=profile, seed=seed).task(0)
    snap = topo.snapshot(t.world.world_key) if topo is not None else None
    ex = Executor(RecipeQuestEnv(), meter(), "exploration", torch.Generator().manual_seed(seed),
                  topo=topo, topo_snapshot=snap, graph_mode=mode)
    return pol, run_episode(pol, ex, t, seed)


@pytest.mark.parametrize("enc,profile,mode", [
    ("set", "known_structure", "known"), ("incidence", "known_structure", "known"),
    ("gated", "known_structure", "known"), ("gated", PROFILE_UNKNOWN, "supergraph"),
    ("gated", PROFILE_UNKNOWN, "active"),
])
def test_likelihood_reconstruction(enc, profile, mode):
    topo = TopologyManager(0.0) if profile == PROFILE_UNKNOWN else None
    pol, rec = _episode(enc, profile, mode, topo)
    with torch.no_grad():
        logits, values, _, _ = pol.forward_sequence(rec.struct, rec.present, rec.mem)
    logp = torch.log_softmax(logits, -1).gather(1, rec.actions.view(-1, 1)).squeeze(1)
    assert torch.allclose(logp, rec.logp, atol=1e-5)
    assert torch.allclose(values, rec.values, atol=1e-5)
    assert list(rec.struct.cand_keys[a] for a in rec.actions.tolist()) == rec.choice_keys


def test_topology_mutation_between_collection_and_update_fails():
    topo = TopologyManager(0.0)
    pol, rec = _episode("gated", PROFILE_UNKNOWN, "active", topo)
    assert topo.pending, "the episode should have produced public evidence"
    topo.commit()  # structural edit between collection and the update
    expected = lambda r: (topo.snapshot(r.world_key).snapshot_id, None)
    opt = torch.optim.Adam(pol.parameters())
    with pytest.raises(StaleBatchError):
        ppo_update(pol, opt, [rec], PPOConfig(), np.random.default_rng(0), expected)


def test_library_mutation_between_collection_and_update_fails():
    lib, _, _ = ingot_key_library()
    ex, _ = executor(lib)
    torch.manual_seed(0)
    pol = ActorCritic("gated", d=32, hidden=32)
    rec = run_episode(pol, ex, task(chain_world(), "key", CHAIN_FACTS), 0)
    admit_scripted(lib, "key", 1, prefs=["wait"])  # new admission after collection
    expected = lambda r: (None, lib.snapshot().library_id)
    with pytest.raises(StaleBatchError):
        ppo_update(pol, torch.optim.Adam(pol.parameters()), [rec], PPOConfig(),
                   np.random.default_rng(0), expected)


def test_admitted_controllers_are_frozen_copies():
    lib, ingot, _ = ingot_key_library()
    before = controller_hash(lib.controllers[ingot.ref])
    admit_scripted(lib, "ingot", 1, prefs=["wait"])
    lib._cache.clear()
    net = lib.controller(ingot.ref)
    assert controller_hash(net.state_dict()) == before == ingot.controller_ref
    assert not any(p.requires_grad for p in net.parameters())


def test_ppo_update_changes_parameters_and_is_finite():
    pol, rec = _episode("incidence")
    rec.R[-1] = 1.0  # synthetic reward fixture to force a non-zero gradient
    before = [p.detach().clone() for p in pol.parameters()]
    stats = ppo_update(pol, torch.optim.Adam(pol.parameters(), lr=1e-3), [rec], PPOConfig(epochs=1),
                       np.random.default_rng(0))
    assert stats["n_updates"] == 1 and np.isfinite(stats["pg_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters()))
