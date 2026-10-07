"""Loading saved agents (flat P1/P2 runs and P3 arm runs)."""

from __future__ import annotations

from pathlib import Path

import torch

from .agents.policy import ActorCritic
from .skills.library import SkillLibrary
from .topology.snapshot import TopologyManager


def load_agent(path: str | Path):
    """Return (policy, arm, topology manager or None, library or None, saved config)."""
    p = Path(path)
    run_dir = p if p.is_dir() else p.parent
    ckpt_path = p if p.is_file() else run_dir / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    if "policy" in ckpt:
        policy = ActorCritic(**ckpt["policy_config"])
        policy.load_state_dict(ckpt["policy"])
        arm = ckpt["arm"]
        config = ckpt.get("config", {})
    else:
        policy = ActorCritic(**ckpt["manager_config"])
        policy.load_state_dict(ckpt["manager"])
        a = ckpt.get("arm", {})
        arm = {"id": a.get("id", "pretrain"), "graph_mode": "active", "topology": "revise",
               "use_posterior": True, "revision": a.get("revision", True)}
        config = ckpt.get("config", {})
    topo = None
    if ckpt.get("topology"):
        st = ckpt["topology"]
        c = st["config"]
        topo = TopologyManager(c["epsilon"], c["max_size"], c["credible_mass"], st["mode"])
        topo.load_state_dict(st)
    library = SkillLibrary.load(run_dir / "library") if (run_dir / "library").exists() else None
    return policy, arm, topo, library, config
