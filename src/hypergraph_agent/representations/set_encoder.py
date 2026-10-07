"""Attention baseline over a set of entity and incidence tokens.

Facts, relation nodes and incidences are separate tokens; each incidence token
carries the identifiers of its two endpoints, in the style of tokenized graph
transformers (random node identifiers). The baseline therefore receives the
same public information as the graph encoders, but computes with generic
self-attention instead of grouped incidence message passing. Identifiers are
bound to opaque task-local keys, so permuting storage order permutes outputs
exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import EDGE_FEATS, FACT_FEATS, ID_DIM, RULE_FEATS, GraphStructure
from .incidence import masked_mean, node_inputs


class SetEncoder(nn.Module):
    def __init__(self, d: int = 64, ctx_dim: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.d = d
        self.fact_in = nn.Linear(FACT_FEATS, d)
        self.rule_in = nn.Linear(RULE_FEATS, d)
        self.edge_in = nn.Linear(EDGE_FEATS, d)
        self.id_in = nn.Linear(2 * ID_DIM, d, bias=False)
        self.ctx_in = nn.Linear(ctx_dim, d)
        self.token_type = nn.Embedding(4, d)  # ctx, fact, rule, incidence
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0,
                                           batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, struct: GraphStructure, present: torch.Tensor, ctx: torch.Tensor,
                fact_mask: torch.Tensor | None = None, rule_mask: torch.Tensor | None = None):
        b = present.shape[0]
        xf, xr = node_inputs(struct, present)
        n_f, n_r, n_e = struct.n_facts, struct.n_rules, len(struct.edge_fact)
        tt = self.token_type.weight
        ids_f = torch.cat([struct.fact_ids, struct.fact_ids], -1)
        ids_r = torch.cat([struct.rule_ids, struct.rule_ids], -1)
        ids_e = torch.cat([struct.fact_ids[struct.edge_fact], struct.rule_ids[struct.edge_rule]], -1)
        t_ctx = (self.ctx_in(ctx) + tt[0]).unsqueeze(1)
        t_f = self.fact_in(xf) + self.id_in(ids_f) + tt[1]
        t_r = self.rule_in(xr) + self.id_in(ids_r) + tt[2]
        t_e = (self.edge_in(struct.edge_feat) + self.id_in(ids_e) + tt[3]).expand(b, -1, -1)
        seq = torch.cat([t_ctx, t_f, t_r, t_e], dim=1)
        pad = None
        if fact_mask is not None or rule_mask is not None:
            fm = fact_mask if fact_mask is not None else torch.ones(n_f, dtype=torch.bool)
            rm = rule_mask if rule_mask is not None else torch.ones(n_r, dtype=torch.bool)
            keep = torch.cat([torch.ones(1, dtype=torch.bool), fm, rm, torch.ones(n_e, dtype=torch.bool)])
            pad = (~keep).unsqueeze(0).expand(b, -1)
        out = self.tf(seq, src_key_padding_mask=pad)
        hf = out[:, 1:1 + n_f]
        hr = out[:, 1 + n_f:1 + n_f + n_r]
        pooled = out[:, 0] + masked_mean(hf, fact_mask)
        return hf, hr, pooled, None
