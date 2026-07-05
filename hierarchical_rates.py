from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class HierarchicalPosterior:
    team_attack_rates: NDArray[np.floating]
    team_defense_rates: NDArray[np.floating]
    confederation_means: dict[str, float]
    global_mean: float
    shrinkage_factors: NDArray[np.floating]
    effective_n: NDArray[np.floating]


class HierarchicalBayesianRates:
    def __init__(self, prior_strength: float = 6.0, confederation_priors: Optional[dict[str, float]] = None) -> None:
        self.prior_strength = prior_strength
        self.conf_priors = confederation_priors or {
            'UEFA': 1.35,
            'CONMEBOL': 1.32,
            'AFC': 1.28,
            'CAF': 1.25,
            'CONCACAF': 1.30,
            'OFC': 1.20,
            'DEFAULT': 1.30,
        }
        self._fitted = False
        self._team_index: dict[str, int] = {}
        self._posterior: Optional[HierarchicalPosterior] = None

    def fit(self, matches: list[dict], team_confederations: dict[str, str]) -> HierarchicalPosterior:
        teams = sorted({m['home'] for m in matches} | {m['away'] for m in matches})
        self._team_index = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        goals_for = np.zeros(n_teams, dtype=float)
        goals_against = np.zeros(n_teams, dtype=float)
        n_matches = np.zeros(n_teams, dtype=float)

        for m in matches:
            i_home = self._team_index[m['home']]
            i_away = self._team_index[m['away']]
            goals_for[i_home] += float(m['gh'])
            goals_against[i_home] += float(m['ga'])
            n_matches[i_home] += 1.0

            goals_for[i_away] += float(m['ga'])
            goals_against[i_away] += float(m['gh'])
            n_matches[i_away] += 1.0

        lambda_mle = np.where(n_matches > 0, goals_for / np.maximum(n_matches, 1.0), 1.3)
        mu_mle = np.where(n_matches > 0, goals_against / np.maximum(n_matches, 1.0), 1.3)

        confs = [team_confederations.get(team, 'DEFAULT') for team in teams]
        conf_means = np.array([self.conf_priors.get(conf, 1.30) for conf in confs], dtype=float)
        shrinkage = n_matches / (n_matches + self.prior_strength)
        lambda_post = shrinkage * lambda_mle + (1.0 - shrinkage) * conf_means
        mu_post = shrinkage * mu_mle + (1.0 - shrinkage) * conf_means

        effective_n = n_matches + self.prior_strength
        confederation_means = {}
        for conf in sorted(set(confs)):
            idx = [i for i, c in enumerate(confs) if c == conf]
            confederation_means[conf] = float(np.mean(lambda_post[idx]))

        self._posterior = HierarchicalPosterior(
            team_attack_rates=lambda_post,
            team_defense_rates=mu_post,
            confederation_means=confederation_means,
            global_mean=float(np.mean(lambda_post)),
            shrinkage_factors=shrinkage,
            effective_n=effective_n,
        )
        self._fitted = True
        return self._posterior

    def predict_team_rates(self, team: str) -> tuple[float, float]:
        if not self._fitted or self._posterior is None:
            raise RuntimeError('Debe llamar fit() primero')
        if team not in self._team_index:
            return (self._posterior.global_mean, self._posterior.global_mean)
        idx = self._team_index[team]
        return (
            float(self._posterior.team_attack_rates[idx]),
            float(self._posterior.team_defense_rates[idx]),
        )

    def compare_teams(self, team_a: str, team_b: str, n_simulations: int = 1000) -> dict:
        if not self._fitted or self._posterior is None:
            raise RuntimeError('Debe llamar fit() primero')
        if team_a not in self._team_index or team_b not in self._team_index:
            return {'p_a_better': 0.5, 'p_equal': 0.0, 'p_b_better': 0.5}

        rng = np.random.default_rng(42)
        i_a = self._team_index[team_a]
        i_b = self._team_index[team_b]
        n_a = self._posterior.effective_n[i_a]
        n_b = self._posterior.effective_n[i_b]
        lambda_a_mean = self._posterior.team_attack_rates[i_a]
        lambda_b_mean = self._posterior.team_attack_rates[i_b]

        samples_a = rng.gamma(lambda_a_mean * n_a, 1.0 / n_a, size=n_simulations)
        samples_b = rng.gamma(lambda_b_mean * n_b, 1.0 / n_b, size=n_simulations)

        p_a_better = float(np.mean(samples_a > samples_b))
        p_equal = float(np.mean(np.abs(samples_a - samples_b) < 0.05))
        p_b_better = float(np.mean(samples_b > samples_a))
        return {
            'p_a_better': p_a_better,
            'p_equal': p_equal,
            'p_b_better': p_b_better,
            'lambda_a_mean': lambda_a_mean,
            'lambda_b_mean': lambda_b_mean,
            'lambda_a_std': float(samples_a.std()),
            'lambda_b_std': float(samples_b.std()),
        }
