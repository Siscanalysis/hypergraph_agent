"""Role-typed bipartite fact-rule message passing, optionally context-gated.

Each recipe (and each bound skill) is an explicit relation node; prerequisite,
candidate-prerequisite, effect and skill incidences are typed edges, so group
identity and roles are preserved. One round:

    fact -> rule : m_e = relu(W_role(e) h_fact + U ef_e),  agg_r = sum_e m_e
                   h_r <- LN(h_r + MLP([h_r, agg_r, c]))
    gate (gated variant only): alpha_r = sigmoid(g([h_r, c, h_goal]))
    rule -> fact : m_e = alpha_r * relu(V_role(e) h_rule + U' ef_e)
                   h_f <- LN(h_f + MLP([h_f, agg_f, c]))

``c`` is the recurrent memory. The ungated encoder concatenates the same
memory into both updates, so the two variants differ only by the
multiplicative gate. Rounds are weight-tied.

``hypergraph_forward`` computes the same function with dense incidence
matrices (a two-stage vertex->hyperedge->vertex update); it exists for the
equivalence test and is exact when edge features are constant per role.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import EDGE_FEATS, FACT_FEATS, NUM_ROLES, RULE_FEATS, GraphStructure


def mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))


def node_inputs(struct: GraphStructure, present: torch.Tensor):
    """Per-step fact and rule features: [B, N_f, FACT_FEATS], [B, N_r, RULE_FEATS]."""
    b = present.shape[0]
    xf = torch.cat([struct.fact_static.expand(b, -1, -1), present.unsqueeze(-1)], dim=-1)
    eff = present[:, struct.rule_effect].unsqueeze(-1)
    xr = torch.cat([eff, struct.rule_static.expand(b, -1, -1)], dim=-1)
    return xf, xr


def masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if h.shape[1] == 0:
        return h.new_zeros(h.shape[0], h.shape[2])
    if mask is None:
        return h.mean(dim=1)
    w = mask.to(h.dtype).view(1, -1, 1)
    return (h * w).sum(dim=1) / w.sum().clamp_min(1.0)


class IncidenceEncoder(nn.Module):
    def __init__(self, d: int = 64, ctx_dim: int = 64, rounds: int = 3, gated: bool = False):
        super().__init__()
        self.d, self.rounds, self.gated = d, rounds, gated
        self.fact_in = nn.Linear(FACT_FEATS, d)
        self.rule_in = nn.Linear(RULE_FEATS, d)
        self.ctx_in = nn.Linear(ctx_dim, d)
        self.msg_f2r = nn.Linear(d, d * NUM_ROLES)
        self.msg_r2f = nn.Linear(d, d * NUM_ROLES)
        self.edge_f2r = nn.Linear(EDGE_FEATS, d, bias=False)
        self.edge_r2f = nn.Linear(EDGE_FEATS, d, bias=False)
        self.rule_up = mlp(3 * d, d, d)
        self.fact_up = mlp(3 * d, d, d)
        self.ln_r = nn.LayerNorm(d)
        self.ln_f = nn.LayerNorm(d)
        self.gate = mlp(3 * d, d, 1) if gated else None

    def _start(self, struct, present, ctx):
        xf, xr = node_inputs(struct, present)
        return torch.relu(self.fact_in(xf)), torch.relu(self.rule_in(xr)), torch.relu(self.ctx_in(ctx))

    def _rule_update(self, hr, agg_r, c):
        return self.ln_r(hr + self.rule_up(torch.cat([hr, agg_r, c.unsqueeze(1).expand_as(hr)], -1)))

    def _fact_update(self, hf, agg_f, c):
        return self.ln_f(hf + self.fact_up(torch.cat([hf, agg_f, c.unsqueeze(1).expand_as(hf)], -1)))

    def _alpha(self, hr, hf, c, goal):
        if self.gate is None:
            return None
        g = hf[:, goal].unsqueeze(1).expand_as(hr)
        return torch.sigmoid(self.gate(torch.cat([hr, c.unsqueeze(1).expand_as(hr), g], -1)))

    def forward(self, struct: GraphStructure, present: torch.Tensor, ctx: torch.Tensor,
                fact_mask: torch.Tensor | None = None, rule_mask: torch.Tensor | None = None):
        hf, hr, c = self._start(struct, present, ctx)
        b, n_f, n_r, d = hf.shape[0], hf.shape[1], hr.shape[1], self.d
        ef, er, role = struct.edge_fact, struct.edge_rule, struct.edge_role
        e_f2r = self.edge_f2r(struct.edge_feat)
        e_r2f = self.edge_r2f(struct.edge_feat)
        alpha = None
        for _ in range(self.rounds):
            m = self.msg_f2r(hf).view(b, n_f, NUM_ROLES, d)[:, ef, role]
            agg_r = hr.new_zeros(b, n_r, d).index_add(1, er, torch.relu(m + e_f2r))
            hr = self._rule_update(hr, agg_r, c)
            alpha = self._alpha(hr, hf, c, struct.goal_fact)
            m = torch.relu(self.msg_r2f(hr).view(b, n_r, NUM_ROLES, d)[:, er, role] + e_r2f)
            if alpha is not None:
                m = m * alpha[:, er]
            agg_f = hf.new_zeros(b, n_f, d).index_add(1, ef, m)
            hf = self._fact_update(hf, agg_f, c)
        pooled = masked_mean(hf, fact_mask) + masked_mean(hr, rule_mask)
        return hf, hr, pooled, alpha


def incidence_matrices(struct: GraphStructure) -> torch.Tensor:
    """Dense role-typed incidence tensor H[role, fact, rule] (edge multiplicities)."""
    h = torch.zeros(NUM_ROLES, struct.n_facts, struct.n_rules)
    h.index_put_((struct.edge_role, struct.edge_fact, struct.edge_rule),
                 torch.ones(len(struct.edge_role)), accumulate=True)
    return h


def hypergraph_forward(enc: IncidenceEncoder, struct: GraphStructure, present: torch.Tensor,
                       ctx: torch.Tensor):
    """Same computation as ``IncidenceEncoder.forward`` via dense incidence
    matrices. Requires edge features that are constant within each role."""
    hf, hr, c = enc._start(struct, present, ctx)
    b, n_f, n_r, d = hf.shape[0], hf.shape[1], hr.shape[1], enc.d
    H = incidence_matrices(struct)
    role_feat = torch.zeros(NUM_ROLES, EDGE_FEATS)
    for rho in range(NUM_ROLES):
        sel = struct.edge_role == rho
        if sel.any():
            feats = struct.edge_feat[sel]
            if not torch.allclose(feats, feats[:1].expand_as(feats)):
                raise ValueError("hypergraph form needs role-constant edge features")
            role_feat[rho] = feats[0]
    u_f2r = enc.edge_f2r(role_feat)  # [R, d]
    u_r2f = enc.edge_r2f(role_feat)
    alpha = None
    for _ in range(enc.rounds):
        m = torch.relu(enc.msg_f2r(hf).view(b, n_f, NUM_ROLES, d) + u_f2r)  # [b, f, R, d]
        agg_r = torch.einsum("pfr,bfpd->brd", H, m)
        hr = enc._rule_update(hr, agg_r, c)
        alpha = enc._alpha(hr, hf, c, struct.goal_fact)
        m = torch.relu(enc.msg_r2f(hr).view(b, n_r, NUM_ROLES, d) + u_r2f)  # [b, r, R, d]
        if alpha is not None:
            m = m * alpha.unsqueeze(-1)
        agg_f = torch.einsum("pfr,brpd->bfd", H, m)
        hf = enc._fact_update(hf, agg_f, c)
    pooled = hf.mean(dim=1) + hr.mean(dim=1)
    return hf, hr, pooled, alpha
