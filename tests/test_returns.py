"""Arithmetic fixtures for duration-aware returns (not experiment results)."""

import numpy as np
import pytest

from hypergraph_agent.training.returns import option_return, segment_gae, smdp_gae


def test_hand_calculated_option():
    R = option_return([1.0, 2.0], gamma=0.5)
    assert R == 2.0
    _, target = segment_gae([R], [2], [0.0], final_value=4.0, terminal=False, gamma=0.5, lam=1.0)
    assert target[0] == pytest.approx(3.0)
    _, target = segment_gae([R], [2], [0.0], final_value=4.0, terminal=True, gamma=0.5, lam=1.0)
    assert target[0] == pytest.approx(2.0)


def test_gamma_one_counts_every_primitive_reward():
    assert option_return([0, 0, 1, 0], 1.0) == 1.0
    adv, tgt = segment_gae([0.0, 1.0], [5, 3], [0.2, 0.4], 0.0, True, 1.0, 1.0)
    assert tgt.tolist() == pytest.approx([1.0, 1.0])


def test_primitive_tau_one_reduces_to_standard_gae():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=6), rng.normal(size=6)
    g, lam, vlast = 0.9, 0.8, 0.3
    adv, _ = segment_gae(r, [1] * 6, v, vlast, False, g, lam)
    ref, nxt = np.zeros(6), 0.0
    vals = list(v) + [vlast]
    for k in reversed(range(6)):
        delta = r[k] + g * vals[k + 1] - v[k]
        nxt = delta + g * lam * nxt * (k < 5)
        ref[k] = nxt
    assert adv == pytest.approx(ref)


def test_lambda_decays_per_primitive_step():
    adv, _ = smdp_gae([0.0, 1.0], [3, 1], [0.0, 0.0], [0.0, 0.0], [1, 0], [1, 0], 1.0, 0.5)
    assert adv[0] == pytest.approx(0.5 ** 3 * 1.0)


def test_cutoff_bootstraps_without_chaining():
    adv_cut, tgt_cut = segment_gae([0.0], [2], [0.1], final_value=0.7, terminal=False, gamma=1.0, lam=0.9)
    adv_term, tgt_term = segment_gae([0.0], [2], [0.1], final_value=0.7, terminal=True, gamma=1.0, lam=0.9)
    assert tgt_cut[0] == pytest.approx(0.7) and tgt_term[0] == pytest.approx(0.0)


def test_nested_durations_are_counted_once():
    # a parent option of 5 steps containing a 3-step child: the parent's tau is 5, not 8
    rewards = [0, 0, 0, 0, 1]
    assert option_return(rewards, 0.9) == pytest.approx(0.9 ** 4)
    with pytest.raises(ValueError):
        smdp_gae([0.0], [0], [0.0], [0.0], [1], [0], 1.0, 1.0)
