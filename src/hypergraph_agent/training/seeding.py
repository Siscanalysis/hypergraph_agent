"""Separate, named random streams for one run.

Task sampling, action sampling, dynamics, discovery, validation and minibatch
order use independent generators derived from the run seed, so changing how
often one component draws does not shift another. States are serializable for
restart. Bitwise reproducibility across hardware or library versions is not
promised.
"""

from __future__ import annotations

import base64

import numpy as np
import torch

from ..envs.generator import derive_seed

STREAMS = ("tasks", "dynamics", "discovery", "validation", "minibatch")


class RNGStreams:
    def __init__(self, run_seed: int, label: str = ""):
        self.run_seed = run_seed
        self.label = label
        self.np = {name: np.random.default_rng(derive_seed("stream", run_seed, label, name))
                   for name in STREAMS}
        self.actions = torch.Generator().manual_seed(derive_seed("stream", run_seed, label, "actions"))
        self.init_seed = derive_seed("stream", run_seed, label, "init") % (2 ** 31)

    def seed_torch_init(self):
        torch.manual_seed(self.init_seed)

    def next_seed(self, name: str) -> int:
        return int(self.np[name].integers(0, 2 ** 31 - 1))

    def fork(self, label: str) -> "RNGStreams":
        """Independent post-fork streams (e.g. one per P3 arm)."""
        return RNGStreams(self.run_seed, f"{self.label}/{label}")

    def state_dict(self) -> dict:
        return {
            "run_seed": self.run_seed, "label": self.label,
            "np": {k: g.bit_generator.state for k, g in self.np.items()},
            "actions": base64.b64encode(self.actions.get_state().numpy().tobytes()).decode(),
        }

    def load_state_dict(self, state: dict):
        for k, st in state["np"].items():
            self.np[k].bit_generator.state = st
        raw = np.frombuffer(base64.b64decode(state["actions"]), dtype=np.uint8).copy()
        self.actions.set_state(torch.from_numpy(raw))
