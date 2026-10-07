"""Encoder factory.

``set``        attention over entity and incidence tokens (common-information baseline)
``incidence``  role-typed incidence message passing with memory concatenated
``gated``      the same plus a memory- and goal-conditioned sigmoid gate per
               relation node on its outgoing messages

A graph encoder applied to the ``active`` versus the ``supergraph`` incidence
set is the adaptive-incidence versus fixed-supergraph contrast of P2; it is the
same encoder class on a different structure.
"""

from __future__ import annotations

from torch import nn

from .incidence import IncidenceEncoder
from .set_encoder import SetEncoder

ENCODERS = ("set", "incidence", "gated")


def make_encoder(kind: str, d: int, ctx_dim: int, rounds: int = 3, layers: int = 2,
                 heads: int = 4) -> nn.Module:
    if kind == "set":
        return SetEncoder(d, ctx_dim, layers, heads)
    if kind == "incidence":
        return IncidenceEncoder(d, ctx_dim, rounds, gated=False)
    if kind == "gated":
        return IncidenceEncoder(d, ctx_dim, rounds, gated=True)
    raise ValueError(f"unknown encoder {kind!r}; choose from {ENCODERS}")
