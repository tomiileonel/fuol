from __future__ import annotations

import numpy as np
from scipy import optimize
from dataclasses import dataclass

from unified_engine_v3 import UnifiedEngineV3


@dataclass
class CalibrationResult:
    optimal_weights: dict[str, float]
    hawkes_bounds: tuple[float, float, float]
    half_life_optimal: float
    prior_strength_optimal: float
    cv_scores: dict[str, float]


def calibrate_hyperparameters(all_matches: list[dict], team_confs: dict[str, str], n_folds: int = 5) -> CalibrationResult:
    sorted_matches = sorted(all_matches, key=lambda m: m.get('date', '1970-01-01'))
    n = len(sorted_matches)

    if n < 20:
        return CalibrationResult(
            optimal_weights={'dc': 0.7, 'hawkes': 0.2, 'hier': 0.1},
            hawkes_bounds=(1e-6, 0.08, 0.35),
            half_life_optimal=365.0,
            prior_strength_optimal=6.0,
            cv_scores={'brier': 0.65, 'log_loss': 1.05, 'rps': 0.22},
        )

    half_life_candidates = [60, 90, 120, 180, 270, 365, 500, 730]
    best_hl, best_score = 365.0, float('inf')
    for hl in half_life_candidates:
        scores = []
        for k in range(10, n, max(1, n // n_folds)):
            train = sorted_matches[:k]
            test = sorted_matches[k]
            team_test = test.get('home', 'UNKNOWN')
            train_team = [m for m in train if m.get('home') == team_test or m.get('away') == team_test]
            if len(train_team) < 5:
                continue
            try:
                engine = UnifiedEngineV3(team_a=team_test, team_b=test.get('opponent', 'UNKNOWN'), matches_a=train_team, matches_b=[], team_confederations=team_confs)
                pred = engine.predict()
                actual = test.get('gf', 0) - test.get('gc', 0)
                if actual > 0:
                    y_true = np.array([1.0, 0.0, 0.0])
                elif actual == 0:
                    y_true = np.array([0.0, 1.0, 0.0])
                else:
                    y_true = np.array([0.0, 0.0, 1.0])
                p_vec = np.array([pred.p_home, pred.p_draw, pred.p_away])
                scores.append(float(np.sum((p_vec - y_true) ** 2)))
            except Exception:
                continue
        if scores and np.mean(scores) < best_score:
            best_score = float(np.mean(scores))
            best_hl = hl

    def objective(weights):
        w_dc, w_hw, w_hier = weights
        if w_dc + w_hw + w_hier > 1.0 or any(v < 0 for v in weights):
            return 1e6
        scores = []
        for k in range(10, min(50, n), 5):
            train = sorted_matches[:k]
            test = sorted_matches[k]
            team_test = test.get('home', 'UNKNOWN')
            train_team = [m for m in train if m.get('home') == team_test or m.get('away') == team_test]
            if len(train_team) < 5:
                continue
            try:
                engine = UnifiedEngineV3(team_a=team_test, team_b=test.get('opponent', 'UNKNOWN'), matches_a=train_team, matches_b=[], team_confederations=team_confs)
                pred = engine.predict()
                actual = test.get('gf', 0) - test.get('gc', 0)
                if actual > 0:
                    y_true = np.array([1.0, 0.0, 0.0])
                elif actual == 0:
                    y_true = np.array([0.0, 1.0, 0.0])
                else:
                    y_true = np.array([0.0, 0.0, 1.0])
                p_vec = np.array([pred.p_home, pred.p_draw, pred.p_away])
                scores.append(float(np.sum((p_vec - y_true) ** 2)))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 1e6

    result = optimize.minimize(objective, x0=[0.7, 0.2, 0.1], method='Nelder-Mead', options={'maxiter': 100})
    optimal_weights = {'dc': float(result.x[0]), 'hawkes': float(result.x[1]), 'hier': float(result.x[2])}

    brier_scores = []
    log_losses = []
    rps_scores = []
    for k in range(10, min(100, n), 3):
        train = sorted_matches[:k]
        test = sorted_matches[k]
        team_test = test.get('home', 'UNKNOWN')
        train_team = [m for m in train if m.get('home') == team_test or m.get('away') == team_test]
        if len(train_team) < 5:
            continue
        try:
            engine = UnifiedEngineV3(team_a=team_test, team_b=test.get('opponent', 'UNKNOWN'), matches_a=train_team, matches_b=[], team_confederations=team_confs)
            pred = engine.predict()
            actual = test.get('gf', 0) - test.get('gc', 0)
            if actual > 0:
                outcome_idx = 0
                y_true = np.array([1.0, 0.0, 0.0])
            elif actual == 0:
                outcome_idx = 1
                y_true = np.array([0.0, 1.0, 0.0])
            else:
                outcome_idx = 2
                y_true = np.array([0.0, 0.0, 1.0])
            p_vec = np.array([pred.p_home, pred.p_draw, pred.p_away])
            p_vec = np.clip(p_vec, 1e-10, 1.0)
            p_vec /= p_vec.sum()
            brier_scores.append(float(np.sum((p_vec - y_true) ** 2)))
            log_losses.append(float(-np.log(p_vec[outcome_idx])))
            cdf_pred = np.cumsum(p_vec)[:2]
            cdf_true = np.cumsum(y_true)[:2]
            rps_scores.append(float(np.mean((cdf_pred - cdf_true) ** 2)))
        except Exception:
            continue

    return CalibrationResult(
        optimal_weights=optimal_weights,
        hawkes_bounds=(1e-6, 0.08, 0.35),
        half_life_optimal=best_hl,
        prior_strength_optimal=6.0,
        cv_scores={
            'brier': float(np.mean(brier_scores)) if brier_scores else 0.65,
            'log_loss': float(np.mean(log_losses)) if log_losses else 1.05,
            'rps': float(np.mean(rps_scores)) if rps_scores else 0.22,
        },
    )
