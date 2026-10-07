"""Overfit sanity check of the actor-critic/PPO loop on one tiny fixed task
where only ``submit`` earns reward. It checks gradient direction and
likelihood-ratio plumbing; it is not an experiment result."""

import numpy as np
import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.ppo import PPOConfig, ppo_update
from hypergraph_agent.training.rollout import run_episode

from helpers import CHAIN_FACTS, chain_world, meter, task


def test_ppo_learns_a_one_step_task():
    t = task(chain_world(), "key", CHAIN_FACTS, initial=("key",), budget=4)
    torch.manual_seed(0)
    pol = ActorCritic("gated", d=32, hidden=32)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    gen, m, rng = torch.Generator().manual_seed(0), meter(50_000), np.random.default_rng(0)
    env = RecipeQuestEnv()

    def batch(n):
        return [run_episode(pol, Executor(env, m, "exploration", gen), t, i) for i in range(n)]

    start = np.mean([r.success for r in batch(40)])
    for _ in range(15):
        ppo_update(pol, opt, batch(16), PPOConfig(epochs=4), rng)
    end = np.mean([r.success for r in batch(40)])
    assert start < 0.5 and end > 0.9
