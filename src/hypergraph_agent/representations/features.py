"""Shared public observation adapter.

Every agent receives the same tensors built here; encoders differ only in how
they compute with them. Inputs are the public task spec, the public present
flags, an optional immutable topology snapshot and optional skill candidates.

Graph modes
-----------
``known``       known_structure profile: certain prerequisite incidences.
``supergraph``  unknown profile: every public candidate pool member is an
                incidence (role PRE_CANDIDATE) carrying its posterior marginal.
``active``      unknown profile: only pool members in the snapshot's active
                set are incidences; the same marginals are attached.
``use_posterior=False`` replaces the posterior features by the prior (used by
the evidence-only baseline and the ``disable_inference`` diagnostic).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import torch

from ..envs.generator import stable_hash
from ..envs.public_schema import (
    NUM_CHOICE_KINDS, SKILL, SUBMIT, PublicObservation, PublicTaskSpec,
)
from ..envs.vocabulary import NUM_KINDS, NUM_TYPES
from ..topology.snapshot import TopologySnapshot, prior_view

PRE_CERTAIN, PRE_CANDIDATE, EFFECT, SKILL_TARGET, SKILL_SUPPORT = range(5)
NUM_ROLES = 5

FACT_STATIC = NUM_TYPES + NUM_KINDS + 2  # type, kind, is_goal, is_target
FACT_FEATS = FACT_STATIC + 1  # + present
# items/3, known base/3, pool/6, entropy, support, MAP mass, is_skill, level/3,
# skill success estimate, skill duration estimate
RULE_STATIC = 10
RULE_FEATS = RULE_STATIC + 1  # + effect present
EDGE_FEATS = NUM_ROLES + 1  # role one-hot + weight (posterior marginal or 1)
CAND_FEATS = NUM_CHOICE_KINDS + 3  # kind one-hot + skill level, success, duration
MEM_FEATS = 4 + NUM_CHOICE_KINDS + NUM_TYPES + 3
ID_DIM = 16

GRAPH_MODES = ("known", "supergraph", "active")


@functools.lru_cache(maxsize=65536)
def identifier(key: str, dim: int = ID_DIM) -> tuple[float, ...]:
    """Deterministic pseudo-random unit vector bound to an entity key.

    Keys are opaque and task-local, so identifiers carry no cross-task identity;
    they only let attention models recover which tokens belong together.
    """
    rng = np.random.default_rng(int(stable_hash(("id", key)), 16))
    v = rng.standard_normal(dim)
    return tuple(float(x) for x in v / np.linalg.norm(v))


@dataclass(frozen=True)
class SkillCandidate:
    """A bound skill call offered as a selectable choice."""

    skill_key: str  # canonical skill id
    version: int
    level: int
    target_fact: int  # bound fact index in this task
    support: tuple[tuple[int, float], ...]  # (fact index, weight) from the contract
    success_est: float
    duration_est: float  # primitive steps

    @property
    def choice_key(self) -> str:
        return f"skill:{self.skill_key}@{self.version}->{self.target_fact}"


@dataclass
class GraphStructure:
    n_facts: int
    n_rules: int  # recipe nodes followed by skill nodes
    fact_static: torch.Tensor  # [N_f, FACT_STATIC]
    rule_static: torch.Tensor  # [N_r, RULE_STATIC]
    rule_effect: torch.Tensor  # [N_r] long
    edge_fact: torch.Tensor  # [E] long
    edge_rule: torch.Tensor  # [E] long
    edge_role: torch.Tensor  # [E] long
    edge_feat: torch.Tensor  # [E, EDGE_FEATS]
    cand_feat: torch.Tensor  # [C, CAND_FEATS]
    cand_node_kind: torch.Tensor  # [C] long: 0 none, 1 fact, 2 rule node
    cand_node_idx: torch.Tensor  # [C] long
    cand_keys: tuple[str, ...]
    cand_choices: tuple  # ("prim", action index) or ("skill", SkillCandidate)
    goal_fact: int
    fact_ids: torch.Tensor  # [N_f, ID_DIM]
    rule_ids: torch.Tensor  # [N_r, ID_DIM]
    meta: dict = field(default_factory=dict)


def _onehot(n: int, i: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def build_structure(
    spec: PublicTaskSpec,
    graph_mode: str,
    snapshot: TopologySnapshot | None = None,
    use_posterior: bool = True,
    skills: tuple[SkillCandidate, ...] = (),
    target_fact: int | None = None,
    include_submit: bool = True,
    include_primitives: bool = True,
) -> GraphStructure:
    if graph_mode not in GRAPH_MODES:
        raise ValueError(f"unknown graph_mode {graph_mode!r}")
    known = spec.profile == "known_structure"
    if known != (graph_mode == "known"):
        raise ValueError(f"graph_mode {graph_mode!r} does not match profile {spec.profile!r}")

    fact_static = []
    for i, f in enumerate(spec.facts):
        fact_static.append(_onehot(NUM_TYPES, f.type_id) + _onehot(NUM_KINDS, f.kind)
                           + [float(f.is_goal), float(i == target_fact)])

    rule_static, rule_effect, rule_keys = [], [], []
    ef, er, erole, efeat = [], [], [], []

    def edge(fact, rule, role, w):
        ef.append(fact)
        er.append(rule)
        erole.append(role)
        efeat.append(_onehot(NUM_ROLES, role) + [float(w)])

    for j, r in enumerate(spec.rules):
        rule_keys.append(r.key)
        rule_effect.append(r.effect)
        for i in r.item_inputs:
            edge(i, j, PRE_CERTAIN, 1.0)
        edge(r.effect, j, EFFECT, 1.0)
        if known:
            for i in r.known_base:
                edge(i, j, PRE_CERTAIN, 1.0)
            rule_static.append([len(r.item_inputs) / 3, len(r.known_base) / 3, 0.0,
                                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            continue
        types = [spec.facts[i].type_id for i in r.item_inputs]
        pool_types = [spec.facts[i].type_id for i in r.pool]
        view = snapshot.rule(r.signature) if (snapshot is not None and use_posterior) else None
        if view is None:
            view = prior_view(r.signature, types, pool_types)
        fallback = view.active if (snapshot is not None and snapshot.rule(r.signature)) else frozenset(pool_types)
        active = fallback if graph_mode == "active" else frozenset(pool_types)
        for i in r.pool:
            t = spec.facts[i].type_id
            if t in active:
                edge(i, j, PRE_CANDIDATE, view.marginal(t))
        rule_static.append([len(r.item_inputs) / 3, 0.0, len(r.pool) / 6,
                            view.entropy_norm, view.support_frac, view.map_mass, 0.0, 0.0, 0.0, 0.0])

    n_recipes = len(spec.rules)
    for s_idx, sk in enumerate(skills):
        j = n_recipes + s_idx
        rule_keys.append(sk.choice_key)
        rule_effect.append(sk.target_fact)
        edge(sk.target_fact, j, SKILL_TARGET, 1.0)
        for fi, w in sk.support:
            edge(fi, j, SKILL_SUPPORT, w)
        rule_static.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                            sk.level / 3, sk.success_est, min(sk.duration_est / 20.0, 2.0)])

    cand_feat, nk, ni, keys, choices = [], [], [], [], []
    if include_primitives:
        for a, d in enumerate(spec.actions):
            if d.kind == SUBMIT and not include_submit:
                continue
            cand_feat.append(_onehot(NUM_CHOICE_KINDS, d.kind) + [0.0, 0.0, 0.0])
            if d.fact is not None:
                nk.append(1)
                ni.append(d.fact)
            elif d.rule is not None:
                nk.append(2)
                ni.append(d.rule)
            else:
                nk.append(0)
                ni.append(0)
            keys.append(d.key)
            choices.append(("prim", a))
    for s_idx, sk in enumerate(skills):
        cand_feat.append(_onehot(NUM_CHOICE_KINDS, SKILL)
                         + [sk.level / 3, sk.success_est, min(sk.duration_est / 20.0, 2.0)])
        nk.append(2)
        ni.append(n_recipes + s_idx)
        keys.append(sk.choice_key)
        choices.append(("skill", sk))
    if not keys:
        raise ValueError("no selectable candidates")

    fact_keys = [f.key for f in spec.facts]
    return GraphStructure(
        n_facts=len(spec.facts),
        n_rules=len(rule_static),
        fact_static=torch.tensor(fact_static, dtype=torch.float32),
        rule_static=torch.tensor(rule_static, dtype=torch.float32).reshape(-1, RULE_STATIC),
        rule_effect=torch.tensor(rule_effect, dtype=torch.long),
        edge_fact=torch.tensor(ef, dtype=torch.long),
        edge_rule=torch.tensor(er, dtype=torch.long),
        edge_role=torch.tensor(erole, dtype=torch.long),
        edge_feat=torch.tensor(efeat, dtype=torch.float32).reshape(-1, EDGE_FEATS),
        cand_feat=torch.tensor(cand_feat, dtype=torch.float32),
        cand_node_kind=torch.tensor(nk, dtype=torch.long),
        cand_node_idx=torch.tensor(ni, dtype=torch.long),
        cand_keys=tuple(keys),
        cand_choices=tuple(choices),
        goal_fact=spec.goal if target_fact is None else target_fact,
        fact_ids=torch.tensor([identifier(k) for k in fact_keys], dtype=torch.float32),
        rule_ids=torch.tensor([identifier(k) for k in rule_keys], dtype=torch.float32).reshape(-1, ID_DIM),
        meta={"graph_mode": graph_mode, "use_posterior": use_posterior,
              "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
              "n_edges": len(ef), "n_candidates": len(keys)},
    )


def present_tensor(obs: PublicObservation) -> torch.Tensor:
    return torch.tensor(obs.present, dtype=torch.float32)


def memory_features(spec: PublicTaskSpec, obs: PublicObservation, goal_fact: int,
                    prev_kind: int | None, prev_target_type: int | None,
                    prev_changed: bool | None, prev_reward: float, prev_duration: int) -> list[float]:
    """Public history summary fed to the recurrent core at each decision."""
    present = obs.present
    v = [obs.budget_left / max(spec.budget, 1), obs.t / max(spec.budget, 1),
         sum(present) / max(len(present), 1), float(present[goal_fact])]
    v += _onehot(NUM_CHOICE_KINDS, prev_kind) if prev_kind is not None else [0.0] * NUM_CHOICE_KINDS
    v += _onehot(NUM_TYPES, prev_target_type) if prev_target_type is not None else [0.0] * NUM_TYPES
    v += [float(bool(prev_changed)), float(prev_reward), min(prev_duration / 20.0, 2.0)]
    return v
