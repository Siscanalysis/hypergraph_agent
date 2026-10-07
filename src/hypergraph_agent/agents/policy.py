"""Shared recurrent actor-critic over variable-size candidate sets.

Per decision k:
    h_k = GRU(h_{k-1}, embed(public history summary m_k))
    node embeddings, pooled = encoder(graph, present flags, h_k)
    score_u = MLP([candidate kind/skill descriptor, embedding of its node,
                   h_k, goal embedding, pooled])
    mu(u | .) = softmax over syntactic candidates
    V = MLP([h_k, pooled, goal embedding])

Candidates are identified by stable keys stored with every rollout; there is
no output slot whose meaning changes when the skill library changes. The same
module serves the P1/P2 agents, the P3 manager and P3 skill controllers.
"""

from __future__ import annotations

import torch
from torch import nn

from ..representations.features import CAND_FEATS, MEM_FEATS, GraphStructure
from ..representations.gated_relations import make_encoder
from ..representations.incidence import mlp


def masked_logits(logits: torch.Tensor, mask: torch.Tensor | None, fallback: int) -> torch.Tensor:
    """Apply a candidate mask; if every candidate of a row is masked, only the
    declared public fallback candidate (normally ``wait``) remains."""
    if mask is None:
        return logits
    mask = mask.clone()
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask[empty, fallback] = True
    return logits.masked_fill(~mask, float("-inf"))


class ActorCritic(nn.Module):
    def __init__(self, encoder: str = "gated", d: int = 64, hidden: int = 64, rounds: int = 3,
                 layers: int = 2, heads: int = 4):
        super().__init__()
        self.config = {"encoder": encoder, "d": d, "hidden": hidden, "rounds": rounds,
                       "layers": layers, "heads": heads}
        self.encoder_kind = encoder
        self.hidden = hidden
        self.mem_in = nn.Linear(MEM_FEATS, hidden)
        self.gru = nn.GRU(hidden, hidden)
        self.encoder = make_encoder(encoder, d, hidden, rounds, layers, heads)
        self.cand_in = nn.Linear(CAND_FEATS, d)
        self.score = mlp(3 * d + hidden + d, d, 1)
        self.value = mlp(hidden + 2 * d, d, 1)

    def initial_state(self) -> torch.Tensor:
        return torch.zeros(self.hidden)

    def core(self, mem: torch.Tensor, h0: torch.Tensor):
        x = torch.relu(self.mem_in(mem)).unsqueeze(1)  # [T, 1, H]
        out, hT = self.gru(x, h0.view(1, 1, -1))
        return out[:, 0], hT.view(-1)

    def heads(self, struct: GraphStructure, present: torch.Tensor, h: torch.Tensor):
        hf, hr, pooled, alpha = self.encoder(struct, present, h)
        t = present.shape[0]
        kind, idx = struct.cand_node_kind, struct.cand_node_idx
        is_f = (kind == 1).view(1, -1, 1).to(hf.dtype)
        is_r = (kind == 2).view(1, -1, 1).to(hf.dtype)
        node = hf[:, torch.where(kind == 1, idx, 0)] * is_f
        if hr.shape[1] > 0:
            node = node + hr[:, torch.where(kind == 2, idx, 0)] * is_r
        goal = hf[:, struct.goal_fact]
        n_c = struct.cand_feat.shape[0]
        cand = torch.relu(self.cand_in(struct.cand_feat)).unsqueeze(0).expand(t, -1, -1)
        ctx = torch.cat([h, goal, pooled], -1).unsqueeze(1).expand(-1, n_c, -1)
        logits = self.score(torch.cat([cand, node, ctx], -1)).squeeze(-1)
        values = self.value(torch.cat([h, pooled, goal], -1)).squeeze(-1)
        return logits, values, alpha

    def forward_sequence(self, struct: GraphStructure, present: torch.Tensor, mem: torch.Tensor,
                         h0: torch.Tensor | None = None):
        """Recompute a whole decision sequence (used by PPO)."""
        h0 = self.initial_state() if h0 is None else h0
        h, hT = self.core(mem, h0)
        logits, values, alpha = self.heads(struct, present, h)
        return logits, values, hT, alpha

    def step(self, struct: GraphStructure, present: torch.Tensor, mem: torch.Tensor,
             h_prev: torch.Tensor):
        """One decision during a rollout; numerically the same path as forward_sequence."""
        h, hT = self.core(mem.unsqueeze(0), h_prev)
        logits, values, alpha = self.heads(struct, present.unsqueeze(0), h)
        return logits[0], values[0], hT, (alpha[0] if alpha is not None else None)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
