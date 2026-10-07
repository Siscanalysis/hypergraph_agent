"""Text playback of one RecipeQuest episode.

    python -m hypergraph_agent.play --agent manual --seed 7
    python -m hypergraph_agent.play --agent random --seed 7 --profile unknown_prerequisites
    python -m hypergraph_agent.play --agent reference --seed 7 --debug
    python -m hypergraph_agent.play --agent learned --checkpoint runs/<run> --seed 7

Playback interactions are demonstrations for a human and are not used for
learning. The reference agent is privileged and labelled as such; the
``--debug`` evaluator view shows hidden prerequisites and is never an agent input.
"""

from __future__ import annotations

import argparse
import sys

import torch

from .agents.baselines import ManualPolicy, RandomPolicy, ReferencePolicy
from .envs.generator import PROFILES, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig
from .envs.recipequest import RecipeQuestEnv, render_public
from .envs.reference_solver import reference_solve
from .skills.executor import Executor
from .topology.snapshot import TopologyManager
from .training.budget import BudgetMeter, SessionLedger
from .training.rollout import run_episode


class _Narrator:
    """Wraps a policy and prints the public view before every root decision."""

    def __init__(self, policy, ex, depth_only_root=True):
        self.policy, self.ex = policy, ex
        self.label = getattr(policy, "label", "learned")

    def initial_state(self):
        return self.policy.initial_state()

    def step(self, struct, present, mem, h):
        print("\n" + render_public(self.ex.spec, self.ex.obs))
        for i, k in enumerate(struct.cand_keys):
            print(f"  [{i:2d}] {k}")
        logits, value, h2, alpha = self.policy.step(struct, present, mem, h)
        print(f"{self.label} chooses: {struct.cand_keys[int(torch.argmax(logits))]}"
              if not isinstance(self.policy, ManualPolicy) else "")
        return logits, value, h2, alpha


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hypergraph_agent.play")
    p.add_argument("--agent", choices=["manual", "random", "reference", "learned"], default="manual")
    p.add_argument("--checkpoint")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--profile", choices=PROFILES, default="known_structure")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--failure-prob", type=float, default=0.0)
    p.add_argument("--debug", action="store_true", help="also print the evaluator-only truth")
    args = p.parse_args(argv)

    stream = TaskStream(TaskStreamConfig(
        "play", args.seed, "per_task", world=WorldConfig(),
        task=TaskConfig(profile=args.profile, depth_min=args.depth, depth_max=args.depth,
                        failure_prob=args.failure_prob)))
    task = stream.task(0)
    env = RecipeQuestEnv()
    graph_mode = "known" if args.profile == "known_structure" else "active"
    topo = TopologyManager(args.failure_prob) if graph_mode != "known" else None
    library = None
    if args.agent == "manual":
        policy = ManualPolicy()
    elif args.agent == "random":
        policy = RandomPolicy()
    elif args.agent == "reference":
        policy = ReferencePolicy(reference_solve(task).plan)
    else:
        if not args.checkpoint:
            p.error("--checkpoint is required for --agent learned")
        from .checkpoints import load_agent
        policy, arm, topo_saved, library, _ = load_agent(args.checkpoint)
        graph_mode = arm["graph_mode"] if args.profile != "known_structure" else "known"
        topo = topo_saved or topo
    meter = BudgetMeter(SessionLedger(None, allocations={"playback": 10_000}), "playback", "playback", 10_000)
    snap = topo.snapshot(task.world.world_key) if topo is not None else None
    ex = Executor(env, meter, "reporting_eval", torch.Generator().manual_seed(args.seed),
                  topo=topo, topo_snapshot=snap, graph_mode=graph_mode, library=library,
                  lib_snapshot=library.snapshot() if library is not None else None)
    label = getattr(policy, "label", "learned")
    print(f"agent: {label}")
    if args.debug:
        env.reset(seed=args.seed, options={"task": task})
        print(env.debug_view())
    rec = run_episode(_Narrator(policy, ex), ex, task, args.seed, record=False,
                      greedy=args.agent == "learned")
    print("\n" + render_public(ex.spec, ex.obs))
    print(f"\nresult: {rec.status}, success={rec.success}, primitive steps={rec.n_primitive}, "
          f"skill calls={rec.n_skill_calls}")
    for c in rec.call_log:
        print(f"  {'  ' * (c['depth'] - 1)}skill {c['ref'][0]}@{c['ref'][1]} -> {c['status']} ({c['tau']} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
