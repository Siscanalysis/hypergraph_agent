import math

import numpy as np
import pytest

from hypergraph_agent.envs.generator import PROFILE_UNKNOWN
from hypergraph_agent.envs.public_schema import public_view
from hypergraph_agent.representations.features import PRE_CANDIDATE, build_structure
from hypergraph_agent.topology.inference import DependencyModel, Evidence, enumerate_hypotheses
from hypergraph_agent.topology.snapshot import TopologyManager

from helpers import CHAIN_FACTS, T, chain_world, task

POOL = tuple(T[x] for x in ("ore", "fuel", "furnace_ready", "sand", "wood", "clay"))
SIG = "ingot<=[]+pool{...}"


def model(eps=0.0):
    m = DependencyModel("W", eps)
    m.register_rule(SIG, (), POOL)
    return m


def ev(present, y, i=0, sig=SIG):
    return Evidence(f"e{i}", "W", sig, frozenset(T[p] for p in present), y)


def support(m):
    b = m.rules[SIG]
    return {tuple(sorted(b.pool[p] for p in h)) for h, q in zip(b.hypotheses, b.posterior()) if q > 1e-12}


def test_hypothesis_class_size():
    assert len(enumerate_hypotheses(6, 3)) == 41
    b = model().rules[SIG]
    assert np.isclose(b.posterior().sum(), 1.0)


def test_deterministic_success_and_failure_elimination():
    m = model()
    assert m.apply(ev(("ore", "fuel", "furnace_ready"), True)) == "informative"
    allowed = {T["ore"], T["fuel"], T["furnace_ready"]}
    assert all(set(h) <= allowed for h in support(m))  # success: absent facts not required
    m.apply(ev(("ore", "fuel"), False, 1))
    assert all(not set(h) <= {T["ore"], T["fuel"]} for h in support(m))  # failure: present sets ruled out
    assert support(m) == {tuple(sorted(s)) for s in (
        (T["furnace_ready"],), (T["ore"], T["furnace_ready"]), (T["fuel"], T["furnace_ready"]),
        (T["ore"], T["fuel"], T["furnace_ready"]))}


def test_noisy_failure_is_not_proof():
    m = model(eps=0.1)
    m.apply(ev(("ore",), False))
    b = m.rules[SIG]
    post = b.posterior()
    i_ore = b.hypotheses.index((POOL.index(T["ore"]),))
    i_fuel = b.hypotheses.index((POOL.index(T["fuel"]),))
    assert post[i_ore] > 0
    assert post[i_ore] / post[i_fuel] == pytest.approx(0.1)


def test_indistinguishable_hypotheses_keep_mass():
    m = model()
    for i in range(3):  # ore and fuel always appear together
        m.apply(ev(("ore", "fuel"), True, i))
    s = support(m)
    assert (T["ore"],) in s and tuple(sorted((T["ore"], T["fuel"]))) in s and (T["fuel"],) in s


def test_contradictory_and_out_of_class_evidence():
    m = model()
    m.apply(ev(("ore",), False))
    before = m.rules[SIG].log_post.copy()
    assert m.apply(ev(("ore",), True, 1)) == "contradiction"  # only {ore} present, but ore alone was ruled out
    assert np.array_equal(before, m.rules[SIG].log_post)
    assert m.rules[SIG].n_contradictions == 1


def test_missing_item_input_makes_failure_uninformative():
    m = DependencyModel("W", 0.0)
    m.register_rule("key", (T["ingot"],), POOL)
    assert m.apply(Evidence("e", "W", "key", frozenset({T["ore"]}), False)) == "uninformative"
    assert m.apply(Evidence("e2", "W", "key", frozenset({T["ore"]}), True)) == "contradiction"


def _spec(world_key="Wtest"):
    t = task(chain_world(), "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN)
    return public_view(t)


def _record(mgr, spec, present_names, y):
    idx = {f.type_id: i for i, f in enumerate(spec.facts)}
    rule = next(j for j, r in enumerate(spec.rules) if spec.facts[r.effect].type_id == T["ingot"])
    before = [False] * len(spec.facts)
    for p in present_names:
        before[idx[T[p]]] = True
    after = list(before)
    after[idx[T["ingot"]]] = y
    return mgr.record(spec, rule, tuple(before), tuple(after))


def test_active_incidence_edit_is_logged_with_evidence():
    spec = _spec()
    mgr = TopologyManager(0.0, mode="revise")
    s0 = mgr.snapshot(spec.world_key)
    e = _record(mgr, spec, ("ore", "fuel", "furnace_ready"), True)
    deltas = mgr.commit()
    assert s0.version == 0 and s0.rules == ()  # older snapshot unchanged
    assert len(deltas) == 1 and deltas[0].edit_kind == "structural_incidence_edit"
    removed = {t for _, t in deltas[0].removes}
    assert {T["sand"], T["wood"], T["clay"]} <= removed and e.evidence_id in deltas[0].evidence_ids
    _record(mgr, spec, ("ore", "fuel", "furnace_ready", "sand"), True)  # uninformative refinement
    assert mgr.commit() == []
    s2 = mgr.snapshot(spec.world_key)
    sig = next(r.signature for r in spec.rules if spec.facts[r.effect].type_id == T["ingot"])
    assert s2.rule(sig).active <= {T["ore"], T["fuel"], T["furnace_ready"]}
    assert s2.snapshot_id != s0.snapshot_id and e.evidence_id
    struct = build_structure(spec, "active", s2)
    sup = build_structure(spec, "supergraph", s2)
    n_act = int((struct.edge_role == PRE_CANDIDATE).sum())
    n_sup = int((sup.edge_role == PRE_CANDIDATE).sum())
    assert n_act < n_sup  # the routing structure genuinely changed


def test_adaptive_and_frozen_share_posterior_features():
    spec = _spec()
    a = TopologyManager(0.0, mode="revise")
    f = TopologyManager(0.0, mode="frozen_structure")
    for mgr in (a, f):
        _record(mgr, spec, ("ore", "fuel", "furnace_ready"), True)
        _record(mgr, spec, ("ore", "fuel"), False)
        mgr.commit()
    sa, sf = a.snapshot(spec.world_key), f.snapshot(spec.world_key)
    for ra, rf in zip(sa.rules, sf.rules):
        assert ra.marginals == rf.marginals and ra.entropy_norm == rf.entropy_norm
    assert any(ra.active != rf.active for ra, rf in zip(sa.rules, sf.rules))
    assert not f.deltas and a.deltas


def test_disabled_inference_changes_nothing():
    spec = _spec()
    mgr = TopologyManager(0.0, mode="disabled")
    _record(mgr, spec, ("ore",), True)
    assert mgr.commit() == [] and mgr.snapshot(spec.world_key).rules == () and mgr.n_dropped == 1


def test_scopes_reset_per_world():
    spec = _spec()
    mgr = TopologyManager(0.0)
    _record(mgr, spec, ("ore", "fuel", "furnace_ready"), True)
    mgr.commit()
    assert mgr.snapshot("another_world").rules == ()


def test_invalid_edit_rolls_back(monkeypatch):
    spec = _spec()
    mgr = TopologyManager(0.0)
    _record(mgr, spec, ("ore", "fuel", "furnace_ready"), True)
    mgr.commit()
    snap, version = mgr.snapshot(spec.world_key), dict(mgr.versions)
    from hypergraph_agent.topology import inference
    monkeypatch.setattr(inference.RuleBelief, "credible_union", lambda self, mass: frozenset({999}))
    _record(mgr, spec, ("ore", "fuel"), False)
    mgr.commit()
    assert mgr.snapshot(spec.world_key) is snap and mgr.versions == version
    assert mgr.events[-1]["type"] == "topology_rollback"


def test_state_round_trip():
    spec = _spec()
    mgr = TopologyManager(0.1)
    _record(mgr, spec, ("ore", "fuel", "furnace_ready"), True)
    mgr.commit()
    other = TopologyManager(0.1)
    other.load_state_dict(mgr.state_dict())
    assert other.snapshot(spec.world_key).snapshot_id == mgr.snapshot(spec.world_key).snapshot_id
