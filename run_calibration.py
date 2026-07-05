from __future__ import annotations

import json
from pathlib import Path

from calibrate_v3 import calibrate_hyperparameters
from dirichlet_process_mixture import DirichletProcessMixture
from unified_engine_v3 import UnifiedEngineV3


def load_sample_matches(path: str | None = None) -> tuple[list[dict], dict[str, str]]:
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return data.get('matches', []), data.get('team_confs', {})

    return [
        {'date': '2024-01-01', 'home': 'ARGENTINA', 'away': 'CHILE', 'gh': 2, 'ga': 1, 'minute': 10},
        {'date': '2024-02-01', 'home': 'ARGENTINA', 'away': 'PERU', 'gh': 1, 'ga': 0, 'minute': 25},
        {'date': '2024-01-15', 'home': 'BRASIL', 'away': 'URUGUAY', 'gh': 1, 'ga': 1, 'minute': 45},
        {'date': '2024-02-15', 'home': 'BRASIL', 'away': 'COL', 'gh': 2, 'ga': 0, 'minute': 60},
        {'date': '2024-03-01', 'home': 'FRANCIA', 'away': 'BELGICA', 'gh': 2, 'ga': 2, 'minute': 50},
        {'date': '2024-03-10', 'home': 'FRANCIA', 'away': 'ESPANA', 'gh': 3, 'ga': 1, 'minute': 40},
    ], {
        'ARGENTINA': 'CONMEBOL',
        'BRASIL': 'CONMEBOL',
        'FRANCIA': 'UEFA',
        'BELGICA': 'UEFA',
        'ESPANA': 'UEFA',
        'CHILE': 'CONMEBOL',
        'PERU': 'CONMEBOL',
        'URUGUAY': 'CONMEBOL',
        'COL': 'CONMEBOL',
    }


if __name__ == '__main__':
    matches, team_confs = load_sample_matches()
    dp = DirichletProcessMixture(alpha=1.0, n_iterations=50)
    rates = [m.get('gh', m.get('gf', 0)) for m in matches]
    result = dp.fit(rates)
    calib = calibrate_hyperparameters(matches, team_confs)
    print('Calibración v3.0 completada')
    print(f'Clusters DP: {result.clusters}')
    print(f'Pesos óptimos: {calib.optimal_weights}')
    print(f'Brier CV: {calib.cv_scores["brier"]:.4f}')
