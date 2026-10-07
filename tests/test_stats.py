"""Synthetic fixtures for the statistics code. These numbers are invented test
inputs and never appear as measured results."""

import pytest

from hypergraph_agent.evaluation.metrics import aggregate
from hypergraph_agent.evaluation.stats import contrast, difference, interaction

SYNTHETIC = {  # method -> seed block -> per-task outcomes (fixture)
    "A": {0: [1, 1, 0, 1], 1: [1, 0, 1, 1], 2: [1, 1, 1, 1]},
    "B": {0: [0, 1, 0, 0], 1: [0, 0, 1, 0], 2: [0, 1, 0, 1]},
}


def test_paired_difference_and_interval():
    out = difference(SYNTHETIC, "A", "B", n_boot=500)
    assert out["estimate"] == pytest.approx((0.5 + 0.5 + 0.5) / 3)
    lo, hi = out["ci"]
    assert lo <= out["estimate"] <= hi and out["n_blocks"] == 3


def test_missing_blocks_are_reported_not_imputed():
    scores = {"A": {0: 0.5, 1: 0.7}, "B": {0: 0.2}}
    out = difference(scores, "A", "B", n_boot=100)
    assert out["n_blocks"] == 1 and out["missing_blocks"] == [1]
    assert out["estimate"] == pytest.approx(0.3)
    assert difference({"A": {0: 1.0}, "B": {}}, "A", "B")["estimate"] is None


def test_interaction_contrast():
    scores = {"G1": {0: 0.9, 1: 0.8}, "G0": {0: 0.5, 1: 0.4},
              "F1": {0: 0.4, 1: 0.5}, "F0": {0: 0.3, 1: 0.4}}
    out = interaction(scores, n_boot=200)
    assert out["estimate"] == pytest.approx(((0.9 - 0.5) - (0.4 - 0.3) + (0.8 - 0.4) - (0.5 - 0.4)) / 2)


def test_aggregate_counts_failures():
    rows = [
        {"success": True, "status": "success", "primitive_length": 4, "manager_decisions": 4,
         "noop_or_invalid": 0, "reference_length": 4, "reference_exact": True, "depth": 1},
        {"success": False, "status": "deadline", "primitive_length": 10, "manager_decisions": 6,
         "noop_or_invalid": 3, "reference_length": 5, "reference_exact": True, "depth": 2},
    ]
    agg = aggregate(rows)
    assert agg["success_rate"] == 0.5 and agg["n_failures"] == 1
    assert agg["mean_primitive_length_all"] == 7 and agg["mean_length_over_optimal_success"] == 1.0
    assert set(aggregate(rows, by="depth")) == {"1", "2"}
    assert contrast({"A": {}}, {"A": 1.0})["n_blocks"] == 0
