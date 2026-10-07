"""Versioned topology snapshots and structural deltas.

Three different quantities are kept apart:

* posterior features (marginals, entropy, support): change whenever evidence
  is committed, in every mode except ``disabled``;
* active incidences (the discrete routing structure used by the
  ``active`` graph mode): change only in ``revise`` mode, and every change is
  a logged ``TopologyDelta`` with the supporting evidence ids;
* context gates inside an encoder: per-decision weights, logged separately as
  ``context_weight`` events by the trainer, never as topology edits.

Snapshots are immutable and only change at ``commit()``, which the trainer
calls between rollout/update batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..envs.generator import stable_hash
from .inference import DependencyModel, Evidence, RuleBelief, evidence_from_transition

MODES = ("revise", "frozen_structure", "disabled")


@dataclass(frozen=True)
class RuleView:
    signature: str
    item_inputs: tuple[int, ...]
    pool: tuple[int, ...]
    active: frozenset  # active candidate base types
    marginals: tuple[float, ...]  # aligned with pool
    entropy_norm: float
    support_frac: float
    map_mass: float

    def marginal(self, type_id: int) -> float:
        return self.marginals[self.pool.index(type_id)]


def prior_view(signature: str, item_inputs, pool, max_size: int = 3) -> RuleView:
    from .inference import enumerate_hypotheses
    pool = tuple(sorted(pool))
    hyps = enumerate_hypotheses(len(pool), min(max_size, len(pool)))
    m = tuple(sum(1 for h in hyps if p in h) / len(hyps) for p in range(len(pool)))
    return RuleView(signature, tuple(sorted(item_inputs)), pool, frozenset(pool), m,
                    1.0, 1.0, 1.0 / len(hyps))


def _view(belief: RuleBelief, active: frozenset) -> RuleView:
    n = len(belief.hypotheses)
    m = belief.marginals()
    return RuleView(
        signature=belief.signature,
        item_inputs=belief.item_inputs,
        pool=belief.pool,
        active=active,
        marginals=tuple(m[t] for t in belief.pool),
        entropy_norm=belief.entropy() / math.log(n) if n > 1 else 0.0,
        support_frac=belief.support() / n,
        map_mass=float(belief.posterior().max()),
    )


@dataclass(frozen=True)
class TopologySnapshot:
    version: int
    scope_key: str
    rules: tuple[RuleView, ...]  # sorted by signature
    inference_config_hash: str
    mode: str
    snapshot_id: str = ""
    _index: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_index", {r.signature: r for r in self.rules})
        if not self.snapshot_id:
            body = [self.version, self.scope_key, self.mode, self.inference_config_hash,
                    [(r.signature, sorted(r.active), [round(x, 9) for x in r.marginals])
                     for r in self.rules]]
            object.__setattr__(self, "snapshot_id", "S" + stable_hash(body))

    def rule(self, signature: str) -> RuleView | None:
        return self._index.get(signature)


@dataclass(frozen=True)
class TopologyDelta:
    scope_key: str
    parent_version: int
    new_version: int
    adds: tuple  # (signature, type id)
    removes: tuple
    replacements: tuple  # signatures with both adds and removes
    evidence_ids: tuple
    confidence_changes: tuple  # (signature, entropy before, entropy after)
    reason: str
    edit_kind: str = "structural_incidence_edit"

    def to_dict(self) -> dict:
        return {
            "type": self.edit_kind, "scope": self.scope_key,
            "parent_version": self.parent_version, "new_version": self.new_version,
            "adds": [list(a) for a in self.adds], "removes": [list(r) for r in self.removes],
            "replacements": list(self.replacements), "evidence_ids": list(self.evidence_ids),
            "confidence_changes": [list(c) for c in self.confidence_changes],
            "reason": self.reason,
        }


class TopologyManager:
    def __init__(self, epsilon: float, max_size: int = 3, credible_mass: float = 0.9,
                 mode: str = "revise"):
        if mode not in MODES:
            raise ValueError(f"unknown topology mode {mode!r}")
        self.epsilon = epsilon
        self.max_size = max_size
        self.credible_mass = credible_mass
        self.mode = mode
        self.models: dict[str, DependencyModel] = {}
        self.active: dict[str, dict[str, frozenset]] = {}
        self.versions: dict[str, int] = {}
        self._snapshots: dict[str, TopologySnapshot] = {}
        self.pending: list[tuple[Evidence, tuple, tuple]] = []
        self.deltas: list[TopologyDelta] = []
        self.events: list[dict] = []
        self._counter = 0
        self.n_dropped = 0

    # ------------------------------------------------------------ queries
    def config(self) -> dict:
        return {"epsilon": self.epsilon, "max_size": self.max_size,
                "credible_mass": self.credible_mass, "prior": "uniform"}

    def config_hash(self) -> str:
        return stable_hash(self.config())

    def snapshot(self, scope_key: str) -> TopologySnapshot:
        if scope_key not in self._snapshots:
            self._snapshots[scope_key] = TopologySnapshot(0, scope_key, (), self.config_hash(), self.mode)
        return self._snapshots[scope_key]

    # ----------------------------------------------------------- evidence
    def record(self, spec, rule_index: int, before, after) -> Evidence | None:
        """Queue public evidence from a craft attempt (applied at commit)."""
        self._counter += 1
        ev = evidence_from_transition(spec, rule_index, before, after,
                                      f"{spec.world_key}:e{self._counter}", spec.world_key)
        if ev is not None:
            r = spec.rules[rule_index]
            items = tuple(spec.facts[i].type_id for i in r.item_inputs)
            pool = tuple(spec.facts[i].type_id for i in r.pool)
            self.pending.append((ev, items, pool))
        return ev

    def freeze_structure(self) -> None:
        """Freeze active incidences; posterior features keep updating."""
        if self.mode == "revise":
            self.mode = "frozen_structure"
            self.events.append({"type": "freeze_dependency_revision", "versions": dict(self.versions)})

    # ------------------------------------------------------------- commit
    def commit(self, reason: str = "batch_boundary") -> list[TopologyDelta]:
        pending, self.pending = self.pending, []
        if self.mode == "disabled":
            self.n_dropped += len(pending)
            return []
        by_scope: dict[str, list] = {}
        for item in pending:
            by_scope.setdefault(item[0].scope_key, []).append(item)
        deltas = []
        for scope, items in sorted(by_scope.items()):
            try:
                delta = self._commit_scope(scope, items, reason)
            except (KeyError, ValueError) as exc:  # atomic: nothing was swapped in
                self.events.append({"type": "topology_rollback", "scope": scope, "error": str(exc)})
                continue
            if delta is not None:
                deltas.append(delta)
        self.deltas.extend(deltas)
        return deltas

    def _commit_scope(self, scope: str, items: list, reason: str) -> TopologyDelta | None:
        model = self.models.get(scope) or DependencyModel(scope, self.epsilon, self.max_size)
        beliefs = {k: v.copy() for k, v in model.rules.items()}
        old_active = dict(self.active.get(scope, {}))
        new_rules = []
        for ev, item_types, pool in items:
            if ev.signature not in beliefs:
                tmp = DependencyModel(scope, self.epsilon, self.max_size)
                beliefs[ev.signature] = tmp.register_rule(ev.signature, item_types, pool)
                new_rules.append(ev.signature)
        outcomes = {"informative": 0, "uninformative": 0, "contradiction": 0}
        used_ids = []
        for ev, _, _ in items:
            outcomes[model.apply(ev, beliefs)] += 1
            used_ids.append(ev.evidence_id)

        new_active = {}
        for sig, b in beliefs.items():
            if self.mode == "revise":
                new_active[sig] = b.credible_union(self.credible_mass)
            else:
                new_active[sig] = old_active.get(sig, frozenset(b.pool))
        for sig, act in new_active.items():  # validation before the swap
            if not act or not act <= set(beliefs[sig].pool):
                raise ValueError(f"invalid active set for {sig}")

        parent = self.versions.get(scope, 0)
        version = parent + 1
        adds, removes, repl, conf = [], [], [], []
        for sig in sorted(new_active):
            # a rule seen for the first time was routed through its full public pool
            old = old_active.get(sig, frozenset(beliefs[sig].pool))
            a = sorted(new_active[sig] - old)
            r = sorted(old - new_active[sig])
            adds += [(sig, t) for t in a]
            removes += [(sig, t) for t in r]
            if a and r:
                repl.append(sig)
            if a or r:
                n = len(beliefs[sig].hypotheses)
                before = model.rules[sig].entropy() if sig in model.rules else math.log(n)
                conf.append((sig, round(before, 6), round(beliefs[sig].entropy(), 6)))

        # swap in atomically
        model.rules = beliefs
        self.models[scope] = model
        self.active[scope] = new_active
        self.versions[scope] = version
        views = tuple(_view(beliefs[s], new_active[s]) for s in sorted(beliefs))
        self._snapshots[scope] = TopologySnapshot(version, scope, views, self.config_hash(), self.mode)
        self.events.append({"type": "topology_commit", "scope": scope, "version": version,
                            "outcomes": outcomes, "registered": new_rules, "reason": reason})
        if adds or removes:
            return TopologyDelta(scope, parent, version, tuple(adds), tuple(removes), tuple(repl),
                                 tuple(used_ids), tuple(conf), reason)
        return None

    # -------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "config": self.config(), "mode": self.mode, "counter": self._counter,
            "versions": dict(self.versions),
            "scopes": {
                scope: {
                    sig: {"item_inputs": list(b.item_inputs), "pool": list(b.pool),
                          "log_post": [float(x) for x in b.log_post],
                          "active": sorted(self.active[scope][sig]),
                          "n_informative": b.n_informative, "n_contradictions": b.n_contradictions,
                          "n_evidence": len(b.evidence_ids)}
                    for sig, b in m.rules.items()
                } for scope, m in self.models.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        cfg = state["config"]
        self.epsilon, self.max_size, self.credible_mass = cfg["epsilon"], cfg["max_size"], cfg["credible_mass"]
        self.mode = state["mode"]
        self._counter = state["counter"]
        self.versions = dict(state["versions"])
        self.models, self.active, self._snapshots = {}, {}, {}
        for scope, rules in state["scopes"].items():
            model = DependencyModel(scope, self.epsilon, self.max_size)
            self.active[scope] = {}
            for sig, d in rules.items():
                b = model.register_rule(sig, d["item_inputs"], d["pool"])
                b.log_post = np.asarray(d["log_post"], dtype=float)
                b.n_informative, b.n_contradictions = d["n_informative"], d["n_contradictions"]
                self.active[scope][sig] = frozenset(d["active"])
            self.models[scope] = model
            views = tuple(_view(model.rules[s], self.active[scope][s]) for s in sorted(model.rules))
            self._snapshots[scope] = TopologySnapshot(self.versions.get(scope, 0), scope, views,
                                                      self.config_hash(), self.mode)
