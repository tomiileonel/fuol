"""
Tests numéricos de las propiedades matemáticas del motor de riesgo.

Cubren las dos correcciones críticas de `kelly_risk_engine.py`:

1. `ruin_probability`: la interpretación del parámetro `outcome_variance`
   debe ser Var[X] (varianza del resultado por unidad apostada), y la
   función debe producir valores razonables (no siempre 1.0).

2. `kelly_robust_binary` (alias retrocompatible `kelly_bayesian_binary`):
   para cuantil <= 0.5 se garantiza shrinkage_factor <= 1.
"""
import math
import numpy as np
import pytest

from kelly_risk_engine import (
    kelly_binary,
    kelly_robust_binary,
    kelly_bayesian_binary,
    ruin_probability,
    optimal_fractional_kelly,
    compute_binary_outcome_variance,
    compute_binary_edge,
)


# ----------------------------------------------------------------------
# Helpers: varianza y edge
# ----------------------------------------------------------------------
def test_compute_binary_outcome_variance_typical_even_money():
    # o=2.0, p=0.55: edge=0.10, Var[X] = 0.55*1 + 0.45 - 0.01 = 0.99
    var = compute_binary_outcome_variance(0.55, 2.0)
    assert var == pytest.approx(0.99, abs=1e-9)


def test_compute_binary_outcome_variance_fair_bet_is_one_minus_p_squared():
    # o=2.0, p=0.5: edge=0, Var[X] = 0.5 + 0.5 - 0 = 1.0
    var = compute_binary_outcome_variance(0.5, 2.0)
    assert var == pytest.approx(1.0, abs=1e-9)


def test_compute_binary_outcome_variance_invalid_inputs_return_zero():
    assert compute_binary_outcome_variance(0.5, 1.0) == 0.0  # cuota inválida
    assert compute_binary_outcome_variance(1.5, 2.0) == 0.0  # prob inválida
    assert compute_binary_outcome_variance(-0.1, 2.0) == 0.0


def test_compute_binary_edge():
    assert compute_binary_edge(0.55, 2.0) == pytest.approx(0.10, abs=1e-9)
    assert compute_binary_edge(0.5, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert compute_binary_edge(0.4, 2.0) == pytest.approx(-0.20, abs=1e-9)


# ----------------------------------------------------------------------
# ruin_probability: propiedades matemáticas
# ----------------------------------------------------------------------
class TestRuinProbabilityProperties:
    def test_zero_kelly_means_zero_ruin(self):
        # No apostar => no hay riesgo de ruina
        assert ruin_probability(0.0, edge=0.10, outcome_variance=1.0) == 0.0

    def test_negative_edge_means_certain_ruin(self):
        # edge <= 0 => mu <= 0 => P(ruina) = 1
        assert ruin_probability(0.25, edge=-0.05, outcome_variance=1.0) == 1.0
        assert ruin_probability(0.25, edge=0.0, outcome_variance=1.0) == 1.0

    def test_p_ruin_decreases_when_edge_increases(self):
        # Para f y sigma² fijos, mayor edge => menor P(ruina)
        p_low_edge = ruin_probability(0.10, edge=0.05, outcome_variance=1.0)
        p_high_edge = ruin_probability(0.10, edge=0.15, outcome_variance=1.0)
        assert p_low_edge > p_high_edge

    def test_p_ruin_increases_when_kelly_fraction_increases(self):
        # Para edge y sigma² fijos (en la región donde mu > 0),
        # mayor f => mayor P(ruina)
        f_low = 0.05
        f_high = 0.20
        edge, var = 0.10, 1.0
        p_low = ruin_probability(f_low, edge, var)
        p_high = ruin_probability(f_high, edge, var)
        assert p_low < p_high, f"P(ruina) con f={f_low} ({p_low}) debe ser menor que con f={f_high} ({p_high})"

    def test_p_ruin_is_a_probability_bounded_in_0_1(self):
        # Nunca debe retornar fuera de [0, 1]
        for f in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
            for edge in [-0.10, 0.0, 0.05, 0.10, 0.30]:
                for var in [0.5, 1.0, 2.0, 5.0]:
                    p = ruin_probability(f, edge, var)
                    assert 0.0 <= p <= 1.0, f"P(ruina)={p} fuera de [0,1] para f={f}, edge={edge}, var={var}"

    def test_typical_even_money_bet_gives_reasonable_ruin_prob(self):
        # Caso realista: o=2.0, p=0.55, edge=0.10, Var[X]=0.99
        # Kelly completo: f* = (0.55*2 - 1)/1 = 0.10
        # Con f=0.10 (Kelly completo), P(ruina) debe ser > 0 pero << 1
        p_ruin_full_kelly = ruin_probability(0.10, edge=0.10, outcome_variance=0.99)
        assert 0.0 < p_ruin_full_kelly < 1.0
        # Y debe ser razonablemente bajo (no > 50%)
        assert p_ruin_full_kelly < 0.5

    def test_invalid_bankroll_ratio_raises(self):
        with pytest.raises(ValueError):
            ruin_probability(0.10, edge=0.10, outcome_variance=1.0, bankroll_ratio_target=1.5)
        with pytest.raises(ValueError):
            ruin_probability(0.10, edge=0.10, outcome_variance=1.0, bankroll_ratio_target=0.0)


# ----------------------------------------------------------------------
# optimal_fractional_kelly: propiedades
# ----------------------------------------------------------------------
class TestOptimalFractionalKelly:
    def test_zero_edge_returns_zero(self):
        result = optimal_fractional_kelly(edge=0.0, outcome_variance=1.0)
        assert result["optimal_fraction_of_kelly"] == 0.0

    def test_negative_edge_returns_zero(self):
        result = optimal_fractional_kelly(edge=-0.05, outcome_variance=1.0)
        assert result["optimal_fraction_of_kelly"] == 0.0

    def test_typical_even_money_finds_reasonable_fraction(self):
        # o=2.0, p=0.55, edge=0.10, Var[X]=0.99
        result = optimal_fractional_kelly(
            edge=0.10,
            outcome_variance=0.99,
            max_ruin_prob=0.01,
        )
        frac = result["optimal_fraction_of_kelly"]
        assert 0.0 < frac <= 1.0, f"Fracción óptima {frac} fuera de (0, 1]"
        assert result["ruin_prob_at_optimum"] <= 0.01 + 1e-9

    def test_stricter_ruin_threshold_yields_lower_fraction(self):
        # Umbral más exigente (1%) => fracción menor que umbral holgado (10%)
        strict = optimal_fractional_kelly(edge=0.10, outcome_variance=0.99, max_ruin_prob=0.01)
        loose = optimal_fractional_kelly(edge=0.10, outcome_variance=0.99, max_ruin_prob=0.10)
        assert strict["optimal_fraction_of_kelly"] <= loose["optimal_fraction_of_kelly"]


# ----------------------------------------------------------------------
# kelly_robust_binary (alias kelly_bayesian_binary): shrinkage property
# ----------------------------------------------------------------------
class TestKellyRobustBinary:
    def test_shrinkage_le_one_for_default_quantile(self):
        """Con quantile=0.25 (default), shrinkage_factor debe ser <= 1."""
        np.random.seed(42)
        # Posterior Beta(80,65): media ~0.55, std ~0.04
        samples = np.random.beta(80, 65, size=5000)
        result = kelly_robust_binary(samples, decimal_odds=1.95)
        assert result["shrinkage_factor"] <= 1.0 + 1e-9, (
            f"shrinkage_factor={result['shrinkage_factor']} > 1 con quantile=0.25 "
            "(debe ser <= 1 por monoticidad de kelly_binary en p)"
        )
        assert result["f_robust"] <= result["f_plugin_naive"] + 1e-12

    def test_alias_kelly_bayesian_binary_returns_same_result(self):
        np.random.seed(1)
        s = np.random.beta(60, 50, size=1000)
        r1 = kelly_robust_binary(s, 2.0, quantile=0.25)
        r2 = kelly_bayesian_binary(s, 2.0)  # alias con default
        assert r1 == r2

    def test_higher_quantile_gives_higher_or_equal_stake(self):
        # Cuantil mayor => p_effective mayor => f_robust mayor o igual
        np.random.seed(0)
        s = np.random.beta(80, 65, size=5000)
        r_low = kelly_robust_binary(s, 1.95, quantile=0.10)
        r_high = kelly_robust_binary(s, 1.95, quantile=0.40)
        assert r_high["f_robust"] >= r_low["f_robust"] - 1e-9

    def test_empty_samples_returns_zero_recommendation(self):
        result = kelly_robust_binary(np.array([]), 2.0)
        assert result["f_robust"] == 0.0
        assert result["f_plugin_naive"] == 0.0

    def test_invalid_quantile_raises(self):
        s = np.array([0.5, 0.55, 0.6])
        with pytest.raises(ValueError):
            kelly_robust_binary(s, 2.0, quantile=0.0)
        with pytest.raises(ValueError):
            kelly_robust_binary(s, 2.0, quantile=1.5)

    def test_high_quantile_warns_about_overbetting(self):
        # Cuantil > 0.5 puede producir shrinkage > 1; se debe advertir
        np.random.seed(0)
        s = np.random.beta(80, 65, size=5000)  # media 0.55
        result = kelly_robust_binary(s, 1.95, quantile=0.75)
        if result["shrinkage_factor"] > 1.0:
            assert "sobre-apuesta" in result["recommendation"].lower() or "conservador" in result["recommendation"].lower()
