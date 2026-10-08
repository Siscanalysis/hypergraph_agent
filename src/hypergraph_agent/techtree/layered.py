"""Study L arms: how a discovery is represented (public information only).

Every arm shares the elimination engine, memory, evidence and replay planner
of ``explorers.Explorer``; they differ only in the hypothesis sets offered to a
new search:

- ``remember`` (U's ``pooled``): a discovery is a literal link, replayed
  whenever it is needed (gating, the goal), never used inside a new hypothesis;
  searches range over primitive windows.
- ``promote``: every known concept is also a node of a higher layer (a macro
  action). A search at level l first tests the expansions of ordered pairs of
  distinct known nodes whose lengths add up to the level's length (nodes of
  every level included), then falls back to primitive windows when those are
  exhausted.
- ``sham_promote``: as many promoted nodes as ``promote`` has, each a random
  base sequence of the same length as a real known concept (useless as atoms):
  a control for the added branching and the cost of testing promoted nodes.
- ``adaptive``: as ``promote``, but before every test it draws q from
  Beta(1 + c, 1 + f), where c and f count this world's discoveries that were,
  or were not, the concatenation of two concepts known at the time (an exact
  check: a concept's parents are held, hence known, when it fires). It tests a
  macro hypothesis when q / (M kM) >= (1 - q) / (W kW), where M is the size of
  the full macro set at the level and W the number of windows of the current
  tier outside it (both counted over the usable alphabet, before any
  elimination), and kM, kW the presses needed to reach the next open window of
  each.
  Under the model "each concept is a composition with probability q" this
  compares expected discoveries per press; failed tests shrink each layer's
  posterior and open set in the same proportion, so the rule depends on q and
  the set sizes only.
- ``none``: no memory between episodes. ``oracle_library``: every concept
  supplied from the start (privileged upper bound, supplied by the runner).

Promotion has no admission gate: discoveries are exact (a firing reveals the
sequence), so there is nothing to validate.
"""

from __future__ import annotations

import numpy as np

from .explorers import Explorer, RandomAgent


class Promote(Explorer):
    def nodes(self, k) -> dict:
        return k.known

    def choose(self, k, ep, level, open_):
        macro = k.macro_mask(level, self.nodes(k)) & open_
        if macro.any():
            return "search_macro", macro
        return super().choose(k, ep, level, open_)


class ShamPromote(Promote):
    def nodes(self, k) -> dict:
        k.ensure_sham(self.rng)
        return k.sham


class Adaptive(Promote):
    def choose(self, k, ep, level, open_):
        alpha = k.alpha_mask(level, self.usable(k, ep))
        macro_all = k.macro_mask(level, k.known) & alpha
        macro = macro_all & open_
        prim = self.primitive_mask(k, level, open_ & ~macro_all)
        if not macro.any() or not prim.any():
            return ("search_macro", macro) if macro.any() else ("search_primitive", prim)
        tiers = k.extra_count[level]
        n_macro = int(macro_all.sum())
        n_prim = int((alpha & ~macro_all & (tiers == tiers[prim].min())).sum())
        k_macro = self.extension(k, ep, level, macro, pick=False)
        k_prim = self.extension(k, ep, level, prim, pick=False)
        q = self.rng.beta(1 + k.comp, 1 + k.flat)
        if q * n_prim * k_prim >= (1 - q) * n_macro * k_macro:
            return "search_macro", macro
        return "search_primitive", prim


L_ARMS = ("none", "remember", "promote", "sham_promote", "adaptive", "oracle_library", "random")


def make_layered(name: str, rng: np.random.Generator):
    if name == "none":
        return Explorer(rng, memory=False)
    if name in ("remember", "oracle_library"):
        return Explorer(rng)
    if name == "promote":
        return Promote(rng)
    if name == "sham_promote":
        return ShamPromote(rng)
    if name == "adaptive":
        return Adaptive(rng)
    if name == "random":
        return RandomAgent(rng)
    raise KeyError(f"unknown study L arm {name!r}")
