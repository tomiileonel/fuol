from __future__ import annotations

from run_calibration import load_sample_matches
from run_calibration import calibrate_hyperparameters, DirichletProcessMixture
from run_validation import validate_model
from production_monitor import ProductionLogger


if __name__ == '__main__':
    matches, team_confs = load_sample_matches()
    dp = DirichletProcessMixture(alpha=1.0, n_iterations=50)
    dp_result = dp.fit([m.get('gh', m.get('gf', 0)) for m in matches])
    calib = calibrate_hyperparameters(matches, team_confs)

    train_matches = matches[:3]
    test_matches = matches[3:]
    report = validate_model(train_matches, test_matches, team_confs)

    logger = ProductionLogger()
    logger.log_signal('pipeline_v3', calib.cv_scores['brier'], report.brier_score, report.brier_score - calib.cv_scores['brier'], 110, 'OK')

    print('=== FUOL v3.0 pipeline completo ===')
    print(f'Clusters DP: {dp_result.clusters}')
    print(f'Pesos óptimos: {calib.optimal_weights}')
    print(f'Brier CV: {calib.cv_scores["brier"]:.4f}')
    print(f'Brier out-of-sample: {report.brier_score:.4f}')
    print(f'Hit Rate: {report.hit_rate_pct:.1f}%')
