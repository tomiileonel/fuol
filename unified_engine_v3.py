from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

from unified_engine import (
    EloRating,
    TimeWeighter,
    BayesianGoalRates,
    DixonColes,
    ELO_INITIAL,
    AVG_GOALS_WC_HISTORICAL,
)
from hawkes_goals_process import HawkesGoalsProcess, HawkesParameters
from quantum_match_state import QuantumMatchState, QuantumAmplitudes, MatchOutcome
from pass_network_topology import PassNetworkAnalyzer
from hierarchical_rates import HierarchicalBayesianRates
from markov_state_chain import ContinuousTimeMarkovChain


@dataclass(frozen=True, slots=True)
class PredictionV3:
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float
    mu_away: float
    quantum_coherence: float
    interpretation: str
    p_home_dc: float
    p_draw_dc: float
    p_away_dc: float
    lambda_hawkes: float
    mu_hawkes: float
    confidence_score: float
    recommendation: str
    top_5_scores: list[dict]

    def to_dict(self) -> dict:
        return {
            'p1': self.p_home,
            'px': self.p_draw,
            'p2': self.p_away,
            'lam': self.lambda_home,
            'mu': self.mu_away,
            'quantum_coherence': self.quantum_coherence,
            'interpretation': self.interpretation,
            'p_home_dc': self.p_home_dc,
            'p_draw_dc': self.p_draw_dc,
            'p_away_dc': self.p_away_dc,
            'lambda_hawkes': self.lambda_hawkes,
            'mu_hawkes': self.mu_hawkes,
            'confidence_score': self.confidence_score,
            'recommendation': self.recommendation,
            'top_5_scores': self.top_5_scores,
        }

    def __getitem__(self, key: str):
        return self.to_dict()[key]

    def get(self, key: str, default=None):
        return self.to_dict().get(key, default)

    @property
    def p1(self) -> float:
        return self.p_home

    @property
    def px(self) -> float:
        return self.p_draw

    @property
    def p2(self) -> float:
        return self.p_away

    @property
    def lam(self) -> float:
        return self.lambda_home

    @property
    def mu(self) -> float:
        return self.mu_away


class UnifiedEngineV3:
    def __init__(
        self,
        team_a: str,
        team_b: str,
        matches_a: list[dict],
        matches_b: list[dict],
        venue: str = 'N',
        modifiers_a: Optional[dict] = None,
        modifiers_b: Optional[dict] = None,
        team_confederations: Optional[dict[str, str]] = None,
        pass_matrices: Optional[tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.team_a = team_a
        self.team_b = team_b
        self.matches_a = sorted(matches_a, key=lambda m: m.get('date', '1970-01-01'))
        self.matches_b = sorted(matches_b, key=lambda m: m.get('date', '1970-01-01'))
        self.venue = venue
        self.modifiers_a = modifiers_a or {}
        self.modifiers_b = modifiers_b or {}
        self.team_confs = team_confederations or {}
        self.pass_matrices = pass_matrices
        self.config = self._load_config(config=config, config_path=config_path)
        self.half_life_days = self.config.get('model', {}).get('unified_engine', {}).get('half_life_days', 365.0)
        self.ensemble_weights = self.config.get('model', {}).get('unified_engine', {}).get('ensemble_weights', {'dc': 0.7, 'hawkes': 0.2, 'hier': 0.1})

        self.elo_a = EloRating(team_a)
        self.elo_a.elo_from_matches(self.matches_a)
        self.elo_b = EloRating(team_b)
        self.elo_b.elo_from_matches(self.matches_b)

        self.hawkes = HawkesGoalsProcess()
        self.hierarchical = HierarchicalBayesianRates()
        self.ctmc = ContinuousTimeMarkovChain()

    def predict(self) -> PredictionV3:
        elo_ratio_a = self.elo_a.expected_goal_ratio(self.elo_b.rating, self.venue)
        elo_ratio_b = 1.0 / elo_ratio_a

        weighter = TimeWeighter(half_life=self.half_life_days)
        bayes = BayesianGoalRates()

        lam_dc, _, _, _, _, _ = bayes.compute_rates(self.matches_a, elo_ratio_a, weighter, self.modifiers_a)
        mu_dc, _, _, _, _, _ = bayes.compute_rates(self.matches_b, elo_ratio_b, weighter, self.modifiers_b)

        dc = DixonColes()
        matrix_dc = dc.score_matrix(lam_dc, mu_dc, rho=-0.13)
        p1_dc, px_dc, p2_dc = dc.extract_1x2(matrix_dc)
        top_5_dc = dc.top_k_scores(matrix_dc, k=5)

        goal_minutes_a = [m.get('minute', 45) for m in self.matches_a if 'minute' in m]
        goal_minutes_b = [m.get('minute', 45) for m in self.matches_b if 'minute' in m]
        hawkes_params_a = self.hawkes.fit(goal_minutes_a) if goal_minutes_a else HawkesParameters(0.015, 0.005, 0.3)
        hawkes_params_b = self.hawkes.fit(goal_minutes_b) if goal_minutes_b else HawkesParameters(0.015, 0.005, 0.3)

        lam_hawkes = hawkes_params_a.expected_goals_90min
        mu_hawkes = hawkes_params_b.expected_goals_90min

        n_a = len(self.matches_a)
        n_b = len(self.matches_b)
        dc_weight = float(self.ensemble_weights.get('dc', 0.7))
        hawkes_weight = float(self.ensemble_weights.get('hawkes', 0.2))
        hier_weight = float(self.ensemble_weights.get('hier', 0.1))
        total_weight = dc_weight + hawkes_weight + hier_weight
        if total_weight > 0:
            dc_weight /= total_weight
            hawkes_weight /= total_weight
            hier_weight /= total_weight

        lam_final = dc_weight * lam_dc + hawkes_weight * lam_hawkes
        mu_final = dc_weight * mu_dc + hawkes_weight * mu_hawkes

        all_matches = self.matches_a + self.matches_b
        if len(all_matches) >= 10 and self.team_confs:
            try:
                posterior = self.hierarchical.fit(all_matches, self.team_confs)
                lam_hier, mu_hier = self.hierarchical.predict_team_rates(self.team_a)
                lam_final = (1.0 - hier_weight) * lam_final + hier_weight * lam_hier
                mu_final = (1.0 - hier_weight) * mu_final + hier_weight * mu_hier
            except Exception:
                pass

        matrix_final = dc.score_matrix(lam_final, mu_final, rho=-0.13)
        p1_final, px_final, p2_final = dc.extract_1x2(matrix_final)
        top_5_final = dc.top_k_scores(matrix_final, k=5)

        q_state = QuantumMatchState(state=QuantumAmplitudes.from_probabilities_and_phases(MatchOutcome(p1_final, px_final, p2_final)))
        if self.venue == 'H':
            q_state = q_state.apply_home_advantage(strength=0.15)
        elif self.venue == 'A':
            q_state = q_state.apply_home_advantage(strength=-0.15)

        inj_a = self.modifiers_a.get('injury_modifier', 1.0)
        if inj_a < 0.9:
            q_state = q_state.apply_injury_impact('home', severity=1.0 - inj_a)
        inj_b = self.modifiers_b.get('injury_modifier', 1.0)
        if inj_b < 0.9:
            q_state = q_state.apply_injury_impact('away', severity=1.0 - inj_b)

        final_probs = q_state.collapse()
        coherence = q_state.state.coherence_measure()
        diagnosis = q_state.diagnosis()

        network_boost = 1.0
        if self.pass_matrices:
            try:
                analyzer = PassNetworkAnalyzer()
                metrics_a = analyzer.analyze(self.pass_matrices[0])
                if metrics_a.algebraic_connectivity > 0.5:
                    network_boost = 1.05
                elif metrics_a.algebraic_connectivity < 0.2:
                    network_boost = 0.95
            except Exception:
                pass
        lam_final *= network_boost

        confidence = self._compute_confidence(coherence, n_a, n_b)
        recommendation = self._generate_recommendation(final_probs, confidence)

        return PredictionV3(
            p_home=final_probs.home,
            p_draw=final_probs.draw,
            p_away=final_probs.away,
            lambda_home=lam_final,
            mu_away=mu_final,
            quantum_coherence=coherence,
            interpretation=diagnosis['interpretation'],
            p_home_dc=p1_dc,
            p_draw_dc=px_dc,
            p_away_dc=p2_dc,
            lambda_hawkes=lam_hawkes,
            mu_hawkes=mu_hawkes,
            confidence_score=confidence,
            recommendation=recommendation,
            top_5_scores=top_5_final,
        )

    def predict_live(self, home_goals: int, away_goals: int, minute_elapsed: float) -> dict:
        self.ctmc.calibrate_from_matches(self.matches_a + self.matches_b)
        live_pred = self.ctmc.predict_live(home_goals, away_goals, minute_elapsed)
        return {
            'current_state': live_pred.current_state,
            'minutes_elapsed': live_pred.minutes_elapsed,
            'p_home_win': live_pred.p_home_win,
            'p_draw': live_pred.p_draw,
            'p_away_win': live_pred.p_away_win,
            'expected_home_remaining': live_pred.expected_home_goals_remaining,
            'expected_away_remaining': live_pred.expected_away_goals_remaining,
        }

    @staticmethod
    def _load_config(config: Optional[dict] = None, config_path: Optional[str] = None) -> dict:
        if config is not None:
            return config
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as handle:
                    return json.load(handle)
        return {}

    @staticmethod
    def _compute_confidence(coherence: float, n_a: int, n_b: int) -> float:
        coherence_score = min(1.0, coherence / 0.8)
        data_score = min(1.0, (n_a + n_b) / 40.0)
        return 0.6 * coherence_score + 0.4 * data_score

    @staticmethod
    def _generate_recommendation(probs: MatchOutcome, confidence: float) -> str:
        p_max = max(probs.home, probs.draw, probs.away)
        if confidence < 0.4:
            return 'NO APOSTAR: Señales contradictorias, modelo incierto.'
        if p_max < 0.45:
            return 'PARTIDO ABIERTO: Sin favorito claro. Considerar mercados alternativos.'
        if p_max >= 0.65 and confidence >= 0.7:
            dominant = 'LOCAL' if probs.home == p_max else ('EMPATE' if probs.draw == p_max else 'VISITANTE')
            return f'APUESTA FUERTE: {dominant} favorito con alta confianza ({p_max:.1%}).'
        if p_max >= 0.55:
            dominant = 'LOCAL' if probs.home == p_max else ('EMPATE' if probs.draw == p_max else 'VISITANTE')
            return f'APUESTA MODERADA: {dominant} con ventaja moderada ({p_max:.1%}).'
        return 'APUESTA CONSERVADORA: Ventaja marginal. Stake fraccional recomendado.'
