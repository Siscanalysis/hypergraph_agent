import numpy as np
import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStreamConfig, WorldConfig
from hypergraph_agent.evaluation.protocol import evaluate
from hypergraph_agent.topology.snapshot import TopologyManager
from hypergraph_agent.training.rollout import run_episode

from helpers import CHAIN_FACTS, T, chain_world, meter, task
from skill_fixtures import cand_for, executor, ingot_key_library


def test_role_bound_skill_runs_in_a_permuted_task_with_new_ids():
    lib, ingot, _ = ingot_key_library()
    ex, m = executor(lib)
    t = task(chain_world(), "key", CHAIN_FACTS, order=[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
             rule_order=[2, 1, 0], prefix="new_")
    ex.reset(t, 0)
    c = cand_for(ex, "ingot")
    assert ex.spec.facts[c.target_fact].key.startswith("new_")
    lib._cache[ingot.ref].prefs = ["gather:new_fore", "gather:new_ffuel",
                                   "activate:new_ffurnace_ready", "craft:new_r0_"]
    assert ex.run_skill(c, depth=1).status == "target_success"


def _p2_stream(ns="val"):
    return TaskStreamConfig(ns, 99, "pool", 2, 0, WorldConfig(n_levels=3, items_per_level=2),
                            TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=2))


def test_frozen_evaluation_leaves_training_state_untouched():
    topo = TopologyManager(0.0)
    before = topo.state_dict()
    torch.manual_seed(0)
    pol = ActorCritic("gated", d=16, hidden=16)
    arm = {"graph_mode": "active", "topology": "revise", "use_posterior": True}
    rows = evaluate(pol, arm, _p2_stream(), n_tasks=3, track="frozen", meter=meter(500, "reporting"),
                    action_gen=torch.Generator().manual_seed(0), dyn_rng=np.random.default_rng(0),
                    topo=topo)
    assert len(rows) == 3 and topo.state_dict() == before and not topo.pending


def test_inference_track_resets_at_new_worlds():
    torch.manual_seed(0)
    pol = ActorCritic("gated", d=16, hidden=16)
    arm = {"graph_mode": "active", "topology": "revise", "use_posterior": True}
    rows = evaluate(pol, arm, _p2_stream(), n_tasks=4, track="inference", meter=meter(800, "reporting"),
                    action_gen=torch.Generator().manual_seed(0), dyn_rng=np.random.default_rng(0),
                    episodes_per_world=2)
    first_per_world = {}
    for r in rows:
        first_per_world.setdefault(r["world_key"], r)
    # the first episode of every world is collected under the empty prior snapshot
    assert all(r["snapshot_id"] == first_per_world[r["world_key"]]["snapshot_id"]
               for r in first_per_world.values())
    assert len({r["world_key"] for r in rows}) == 2


def test_no_recurrent_state_leaks_between_episodes():
    t = task(chain_world(), "key", CHAIN_FACTS)
    torch.manual_seed(0)
    pol = ActorCritic("gated", d=16, hidden=16)
    firsts = []
    for _ in range(2):
        from hypergraph_agent.envs.recipequest import RecipeQuestEnv
        from hypergraph_agent.skills.executor import Executor
        ex = Executor(RecipeQuestEnv(), meter(), "exploration", torch.Generator().manual_seed(1))
        rec = run_episode(pol, ex, t, 0)
        firsts.append(rec.logp[0])
    assert torch.equal(firsts[0], firsts[1])
