import numpy as np
import pytest

from hawkes_goals_process import HawkesGoalsProcess
from pass_network_topology import PassNetworkAnalyzer
from quantum_match_state import MatchOutcome, QuantumAmplitudes, QuantumMatchState


def test_hawkes_fit_returns_reasonable_params():
    process = HawkesGoalsProcess(match_duration_min=90.0)
    params = process.fit([10.0, 20.0, 45.0, 60.0])

    assert params.mu > 0.0
    assert params.alpha >= 0.0
    assert params.beta > 0.0

    distribution = process.goal_distribution(params, max_goals=3, n_simulations=200)
    assert distribution.sum() == pytest.approx(1.0, abs=1e-3)


def test_quantum_state_collapse_is_normalized():
    state = QuantumMatchState(
        state=QuantumAmplitudes.from_probabilities_and_phases(
            MatchOutcome(0.5, 0.3, 0.2)
        )
    )

    probs = state.collapse()
    assert np.isclose(sum(probs), 1.0, atol=1e-9)
    assert state.diagnosis()["probabilities"]["p_home"] >= 0.0


def test_pass_network_analyzer_returns_feature_vector():
    analyzer = PassNetworkAnalyzer(n_players=3)
    matrix = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    metrics = analyzer.analyze(matrix)
    vector = metrics.as_feature_vector()

    assert vector.shape == (9,)
    assert np.isfinite(vector).all()
