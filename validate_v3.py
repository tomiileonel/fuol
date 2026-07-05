from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from unified_engine_v3 import UnifiedEngineV3
from kelly_risk_engine import kelly_multi_outcome


@dataclass
class ValidationReport:
    n_matches_evaluated: int
    brier_score: float
    log_loss: float
    rps: float
    hit_rate_pct: float
    sharpe_ratio: float
    roi_pct: float
    calibration_bins: dict[str, dict]
    recommendations: list[str]


def validate_model(train_matches: list[dict], test_matches: list[dict], team_confs: dict[str, str], initial_bankroll: float = 10000.0) -> ValidationReport:
    if not test_matches:
        return ValidationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, ['No hay datos de prueba disponibles.'])

    brier_scores = []
    log_losses = []
    rps_scores = []
    hits = []
    bankroll = initial_bankroll
    trades = []

    for i, test_match in enumerate(test_matches):
        team_home = test_match.get('home', 'UNKNOWN')
        team_away = test_match.get('away', 'UNKNOWN')
        test_date = test_match.get('date', '9999-99-99')
        train_home = [m for m in train_matches if m.get('date', '9999-99-99') < test_date and (m.get('home') == team_home or m.get('away') == team_home)]
        train_away = [m for m in train_matches if m.get('date', '9999-99-99') < test_date and (m.get('home') == team_away or m.get('away') == team_away)]
        if len(train_home) < 2 or len(train_away) < 2:
            train_home = train_home + train_matches[:2]
            train_away = train_away + train_matches[:2]

        try:
            engine = UnifiedEngineV3(team_a=team_home, team_b=team_away, matches_a=train_home, matches_b=train_away, venue='H', team_confederations=team_confs)
            pred = engine.predict()
            actual_gf = test_match.get('gh', 0)
            actual_gc = test_match.get('ga', 0)
            if actual_gf > actual_gc:
                y_true = np.array([1.0, 0.0, 0.0])
                outcome_idx = 0
            elif actual_gf == actual_gc:
                y_true = np.array([0.0, 1.0, 0.0])
                outcome_idx = 1
            else:
                y_true = np.array([0.0, 0.0, 1.0])
                outcome_idx = 2

            p_vec = np.array([pred.p_home, pred.p_draw, pred.p_away])
            p_vec = np.clip(p_vec, 1e-10, 1.0)
            p_vec /= p_vec.sum()

            brier = float(np.sum((p_vec - y_true) ** 2))
            brier_scores.append(brier)
            log_losses.append(float(-np.log(p_vec[outcome_idx])))
            cdf_pred = np.cumsum(p_vec)[:2]
            cdf_true = np.cumsum(y_true)[:2]
            rps_scores.append(float(np.mean((cdf_pred - cdf_true) ** 2)))
            hits.append(int(np.argmax(p_vec) == outcome_idx))

            market_odds = 1.0 / p_vec * 1.05
            kelly_result = kelly_multi_outcome(probs=p_vec, decimal_odds=market_odds, max_total_stake=0.25)
            for selection, stake_frac in [('1', kelly_result['stake_1']), ('X', kelly_result['stake_X']), ('2', kelly_result['stake_2'])]:
                if stake_frac < 1e-6:
                    continue
                stake_amount = bankroll * stake_frac
                won = False
                if selection == '1' and outcome_idx == 0:
                    won = True
                elif selection == 'X' and outcome_idx == 1:
                    won = True
                elif selection == '2' and outcome_idx == 2:
                    won = True
                if won:
                    odds_idx = 0 if selection == '1' else (1 if selection == 'X' else 2)
                    payout = stake_amount * market_odds[odds_idx]
                    pnl = payout - stake_amount
                else:
                    pnl = -stake_amount
                bankroll += pnl
                trades.append({'match': f'{team_home} vs {team_away}', 'selection': selection, 'stake': stake_amount, 'won': won, 'pnl': pnl})
        except Exception:
            continue

    n_evaluated = len(brier_scores)
    if n_evaluated == 0:
        return ValidationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, ['No se pudieron evaluar partidos.'])

    brier_avg = float(np.mean(brier_scores))
    log_loss_avg = float(np.mean(log_losses))
    rps_avg = float(np.mean(rps_scores))
    hit_rate = float(np.mean(hits) * 100)
    if trades:
        pnls = [t['pnl'] for t in trades]
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0.0
    else:
        sharpe = 0.0
    roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100

    recommendations = []
    if brier_avg < 0.60:
        recommendations.append('✓ Excelente calibración (Brier < 0.60)')
    elif brier_avg < 0.65:
        recommendations.append('○ Buena calibración (Brier 0.60-0.65)')
    else:
        recommendations.append('✗ Calibración mejorable (Brier > 0.65). Considerar recalibrar.')
    if hit_rate > 55:
        recommendations.append(f'✓ Hit rate alto ({hit_rate:.1f}%)')
    elif hit_rate > 45:
        recommendations.append(f'○ Hit rate moderado ({hit_rate:.1f}%)')
    else:
        recommendations.append(f'✗ Hit rate bajo ({hit_rate:.1f}%). Revisar features.')
    if roi > 0:
        recommendations.append(f'✓ ROI positivo ({roi:.1f}%)')
    else:
        recommendations.append(f'✗ ROI negativo ({roi:.1f}%). Ajustar estrategia de Kelly.')
    if sharpe > 1.0:
        recommendations.append(f'✓ Sharpe Ratio excelente ({sharpe:.2f})')
    elif sharpe > 0.5:
        recommendations.append(f'○ Sharpe Ratio aceptable ({sharpe:.2f})')
    else:
        recommendations.append(f'✗ Sharpe Ratio bajo ({sharpe:.2f}). Alto riesgo relativo.')

    return ValidationReport(n_evaluated, brier_avg, log_loss_avg, rps_avg, hit_rate, sharpe, roi, {}, recommendations)
