import pytest
import torch

from hypergraph_agent.agents.policy import ActorCritic, masked_logits
from hypergraph_agent.envs.generator import PROFILE_UNKNOWN
from hypergraph_agent.envs.public_schema import public_view, initial_observation
from hypergraph_agent.representations.features import (
    EFFECT, PRE_CERTAIN, build_structure, memory_features,
)
from hypergraph_agent.representations.incidence import IncidenceEncoder, hypergraph_forward
from hypergraph_agent.representations.set_encoder import SetEncoder

from helpers import CHAIN_FACTS, chain_world, task

ENCODERS = ("set", "incidence", "gated")


def inputs(t, mode="known"):
    spec = public_view(t)
    obs = initial_observation(t)
    struct = build_structure(spec, mode)
    present = torch.tensor([obs.present], dtype=torch.float32)
    mem = torch.tensor([memory_features(spec, obs, struct.goal_fact, None, None, None, 0.0, 0)])
    return spec, struct, present, mem


@pytest.mark.parametrize("enc", ENCODERS)
def test_permutation_equivariance(enc):
    w = chain_world()
    base = task(w, "key", CHAIN_FACTS, initial=("ore",))
    perm = task(w, "key", CHAIN_FACTS, initial=("ore",), order=[5, 3, 11, 0, 9, 1, 7, 2, 10, 4, 8, 6],
                rule_order=[2, 0, 1])
    torch.manual_seed(0)
    pol = ActorCritic(enc)
    outs = []
    for t in (base, perm):
        spec, struct, present, mem = inputs(t)
        logits, values, _, _ = pol.forward_sequence(struct, present, mem)
        outs.append((dict(zip(struct.cand_keys, logits[0].tolist())), float(values[0])))
    assert outs[0][0].keys() == outs[1][0].keys()
    for k in outs[0][0]:
        assert outs[0][0][k] == pytest.approx(outs[1][0][k], abs=1e-5)
    assert outs[0][1] == pytest.approx(outs[1][1], abs=1e-5)


def _pad(struct, n_extra):
    import dataclasses
    fs = torch.cat([struct.fact_static, torch.zeros(n_extra, struct.fact_static.shape[1])])
    ids = torch.cat([struct.fact_ids, torch.randn(n_extra, struct.fact_ids.shape[1])])
    return dataclasses.replace(struct, n_facts=struct.n_facts + n_extra, fact_static=fs, fact_ids=ids)


@pytest.mark.parametrize("enc_cls", [IncidenceEncoder, SetEncoder])
def test_padding_does_not_affect_real_nodes(enc_cls):
    _, struct, present, mem = inputs(task(chain_world(), "key", CHAIN_FACTS))
    torch.manual_seed(0)
    enc = enc_cls(64, 64)
    ctx = torch.randn(1, 64)
    hf, hr, pooled, _ = enc(struct, present, ctx)
    padded = _pad(struct, 3)
    pp = torch.cat([present, torch.ones(1, 3)], dim=1)
    mask = torch.tensor([True] * struct.n_facts + [False] * 3)
    hf2, hr2, pooled2, _ = enc(padded, pp, ctx, fact_mask=mask)
    assert torch.allclose(hf, hf2[:, :struct.n_facts], atol=1e-5)
    assert torch.allclose(hr, hr2, atol=1e-5)
    assert torch.allclose(pooled, pooled2, atol=1e-5)


def test_candidate_indexing_follows_public_actions():
    spec, struct, _, _ = inputs(task(chain_world(), "key", CHAIN_FACTS))
    assert struct.cand_keys == tuple(a.key for a in spec.actions)
    for i, (kind, idx) in enumerate(zip(struct.cand_node_kind.tolist(), struct.cand_node_idx.tolist())):
        a = spec.actions[i]
        assert (kind, idx) == ((1, a.fact) if a.fact is not None else (2, a.rule) if a.rule is not None else (0, 0))


def test_roles_are_directional():
    _, struct, present, mem = inputs(task(chain_world(), "key", CHAIN_FACTS))
    torch.manual_seed(0)
    enc = IncidenceEncoder(64, 64)
    ctx = torch.zeros(1, 64)
    hf, hr, _, _ = enc(struct, present, ctx)
    flipped = struct.edge_role.clone()
    pre, eff = (flipped == PRE_CERTAIN).nonzero()[0, 0], (flipped == EFFECT).nonzero()[0, 0]
    flipped[pre], flipped[eff] = EFFECT, PRE_CERTAIN
    import dataclasses
    hf2, hr2, _, _ = enc(dataclasses.replace(struct, edge_role=flipped), present, ctx)
    assert not torch.allclose(hr, hr2)


@pytest.mark.parametrize("gated", [False, True])
def test_incidence_and_hypergraph_forms_are_equivalent(gated):
    _, struct, present, mem = inputs(task(chain_world(), "key", CHAIN_FACTS, initial=("ore", "wax")))
    torch.manual_seed(1)
    enc = IncidenceEncoder(32, 16, rounds=3, gated=gated)
    ctx = torch.randn(2, 16)
    present = present.expand(2, -1)
    a = enc(struct, present, ctx)
    b = hypergraph_forward(enc, struct, present, ctx)
    for x, y in zip(a[:3], b[:3]):
        assert torch.allclose(x, y, atol=1e-5)
    ga = torch.autograd.grad(a[2].sum() + a[0].pow(2).sum(), list(enc.parameters()), allow_unused=True)
    gb = torch.autograd.grad(b[2].sum() + b[0].pow(2).sum(), list(enc.parameters()), allow_unused=True)
    for x, y in zip(ga, gb):
        if x is None:
            assert y is None
        else:
            assert torch.allclose(x, y, atol=1e-5)


def test_hypergraph_form_refuses_incompatible_edge_features():
    t = task(chain_world(), "key", CHAIN_FACTS, profile=PROFILE_UNKNOWN)
    spec = public_view(t)
    from hypergraph_agent.topology.snapshot import TopologyManager
    struct = build_structure(spec, "supergraph", TopologyManager(0.0).snapshot(spec.world_key))
    struct.edge_feat[0, -1] = 0.123  # heterogeneous weights within a role
    struct.edge_feat[1, -1] = 0.456
    enc = IncidenceEncoder(16, 16)
    obs = initial_observation(t)
    with pytest.raises(ValueError):
        hypergraph_forward(enc, struct, torch.tensor([obs.present], dtype=torch.float32), torch.zeros(1, 16))


def test_all_masked_rows_fall_back_without_nans():
    logits = torch.randn(2, 4)
    mask = torch.tensor([[False] * 4, [True, False, True, False]])
    out = masked_logits(logits, mask, fallback=3)
    probs = torch.softmax(out, -1)
    assert torch.isfinite(probs).all()
    assert probs[0, 3] == 1.0 and probs[1, 1] == 0.0


@pytest.mark.parametrize("enc", ENCODERS)
def test_step_matches_sequence(enc):
    _, struct, present, mem = inputs(task(chain_world(), "key", CHAIN_FACTS))
    torch.manual_seed(0)
    pol = ActorCritic(enc)
    seq_p = present.expand(3, -1).clone()
    seq_p[1, 0] = 1.0
    seq_m = mem.expand(3, -1).clone()
    seq_m[2, 0] = 0.5
    logits, values, _, _ = pol.forward_sequence(struct, seq_p, seq_m)
    h = pol.initial_state()
    for k in range(3):
        l, v, h, _ = pol.step(struct, seq_p[k], seq_m[k], h)
        assert torch.allclose(l, logits[k], atol=1e-5) and abs(float(v - values[k])) < 1e-5


def test_parameter_counts_are_reported():
    counts = {e: ActorCritic(e).parameter_count() for e in ENCODERS}
    assert all(c > 0 for c in counts.values())
