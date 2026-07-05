import numpy as np

from kelly_risk_engine import kelly_binary, kelly_multi_outcome


def test_kelly_binary_returns_zero_for_negative_edge():
    assert kelly_binary(0.4, 1.5) == 0.0


def test_kelly_multi_outcome_respects_total_stake_limit():
    probs = np.array([0.4, 0.3, 0.3])
    odds = np.array([2.0, 3.0, 4.0])

    result = kelly_multi_outcome(probs, odds, max_total_stake=0.25)

    assert result["converged"] is True
    assert result["total_stake"] <= 0.25 + 1e-9
    assert result["stake_1"] >= 0.0
    assert result["stake_X"] >= 0.0
    assert result["stake_2"] >= 0.0
