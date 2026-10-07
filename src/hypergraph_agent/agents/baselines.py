"""Non-learned agents with unmistakable labels.

``RandomPolicy``     uniform over syntactic candidates.
``ReferencePolicy``  PRIVILEGED: replays the evaluator's reference plan, which
                     reads hidden prerequisites. Evaluation/playback only;
                     never a training signal.
``ManualPolicy``     reads choices from stdin (text playback).
"""

from __future__ import annotations

import torch


class RandomPolicy:
    label = "random"
    hidden = 1

    def initial_state(self):
        return torch.zeros(self.hidden)

    def step(self, struct, present, mem, h):
        return torch.zeros(len(struct.cand_keys)), torch.tensor(0.0), h, None


class ReferencePolicy:
    label = "reference (PRIVILEGED: uses hidden prerequisites; evaluator only)"
    hidden = 1

    def __init__(self, plan: tuple[str, ...]):
        self.plan = list(plan)
        self.i = 0

    def initial_state(self):
        return torch.zeros(self.hidden)

    def step(self, struct, present, mem, h):
        key = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        logits = torch.full((len(struct.cand_keys),), -1e9)
        logits[struct.cand_keys.index(key)] = 0.0
        return logits, torch.tensor(0.0), h, None


class ManualPolicy:
    label = "manual"
    hidden = 1

    def __init__(self, show=print, ask=input):
        self.show, self.ask = show, ask

    def initial_state(self):
        return torch.zeros(self.hidden)

    def step(self, struct, present, mem, h):
        while True:
            raw = self.ask("choice number (or key): ").strip()
            if raw in struct.cand_keys:
                k = struct.cand_keys.index(raw)
                break
            if raw.isdigit() and int(raw) < len(struct.cand_keys):
                k = int(raw)
                break
            self.show("not a valid choice")
        logits = torch.full((len(struct.cand_keys),), -1e9)
        logits[k] = 0.0
        return logits, torch.tensor(0.0), h, None
