from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from calibrate_v3 import calibrate_hyperparameters
from dirichlet_process_mixture import DirichletProcessMixture
from production_logger import ProductionLogger


def load_historical_data(data_path: str) -> tuple[list[dict], dict[str, str], dict[str, tuple[float, float]]]:
    data_dir = Path(data_path)
    matches: list[dict] = []
    team_confs: dict[str, str] = {}
    team_stats: dict[str, list[tuple[float, float]]] = {}

    results_csv = data_dir / 'results.csv'
    if results_csv.exists():
        import csv
        with open(results_csv, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    home = row.get('home_team', row.get('team_a', ''))
                    away = row.get('away_team', row.get('team_b', ''))
                    match = {
                        'home': home,
                        'away': away,
                        'gh': int(row.get('goals_home', row.get('gf', 0))),
                        'ga': int(row.get('goals_away', row.get('gc', 0))),
                        'date': row.get('date', '1970-01-01')[:10],
                        'competition': row.get('competition', 'N'),
                    }
                    matches.append(match)
                    for team, goals_for, goals_against in [(home, match['gh'], match['ga']), (away, match['ga'], match['gh'])]:
                        team_stats.setdefault(team, []).append((goals_for, goals_against))
                except (ValueError, KeyError):
                    continue

    team_stats_mle = {}
    for team, stats in team_stats.items():
        if stats:
            avg_for = sum(s[0] for s in stats) / len(stats)
            avg_against = sum(s[1] for s in stats) / len(stats)
            team_stats_mle[team] = (max(avg_for, 0.1), max(avg_against, 0.1))
        else:
            team_stats_mle[team] = (1.3, 1.3)

    for team in set(m['home'] for m in matches) | set(m['away'] for m in matches):
        team_confs[team] = {
            'ARGENTINA': 'CONMEBOL', 'BRASIL': 'CONMEBOL', 'URUGUAY': 'CONMEBOL', 'CHILE': 'CONMEBOL',
            'ALEMANIA': 'UEFA', 'ESPAÑA': 'UEFA', 'FRANCIA': 'UEFA', 'INGLATERRA': 'UEFA',
            'SENEGAL': 'CAF', 'MARRUECOS': 'CAF', 'JAPÓN': 'AFC', 'COREA DEL SUR': 'AFC',
            'EEUU': 'CONCACAF', 'MÉXICO': 'CONCACAF',
        }.get(team, 'DEFAULT')

    return matches, team_confs, team_stats_mle


def run_integration(data_path: str, output_path: str, dp_alpha: float = 1.0, cv_folds: int = 5) -> dict:
    matches, team_confs, team_stats_mle = load_historical_data(data_path)
    if len(matches) < 20:
        config = {
            'version': '3.0.0-default',
            'generated_at': datetime.utcnow().isoformat(),
            'model': {'unified_engine': {'half_life_days': 365.0, 'prior_strength': 6.0, 'ensemble_weights': {'dc': 0.7, 'hawkes': 0.2, 'hier': 0.1}}},
        }
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config_v3.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return config

    dp = DirichletProcessMixture(alpha=dp_alpha, min_cluster_size=2)
    dp_result = dp.fit(team_stats_mle, n_iterations=200, burn_in=50)
    calib = calibrate_hyperparameters(matches, team_confs, n_folds=cv_folds)

    config = {
        'version': '3.0.0',
        'generated_at': datetime.utcnow().isoformat(),
        'model': {
            'unified_engine': {
                'half_life_days': calib.half_life_optimal,
                'prior_strength': calib.prior_strength_optimal,
                'ensemble_weights': calib.optimal_weights,
                'hawkes_bounds': calib.hawkes_bounds,
            },
            'dirichlet_process': {
                'alpha': dp_alpha,
                'clusters': {
                    str(k): {
                        'attack_mean': params.get('attack_mean', 0.0),
                        'defense_mean': params.get('defense_mean', 0.0),
                        'n_members': params.get('n_members', 0),
                        'members': list(params.get('member_teams', [])),
                    }
                    for k, params in dp_result.clusters.items()
                },
                'team_assignments': dp_result.team_assignments,
            },
        },
        'validation': {'cv_scores': calib.cv_scores},
    }

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config_v3.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger = ProductionLogger(db_path=str(output_dir / 'production_log.db'))
    logger.log_event('integration_complete', metadata={'version': '3.0.0', 'clusters': dp_result.n_clusters}, metrics={'brier': calib.cv_scores['brier']})

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FUOL v3.0 integration')
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--output', default='./config_v3')
    parser.add_argument('--dp-alpha', type=float, default=1.0)
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()

    try:
        result = run_integration(args.data_path, args.output, dp_alpha=args.dp_alpha, cv_folds=args.cv_folds)
        print('Integración completada')
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as exc:
        print(f'Error en integración: {exc}', file=sys.stderr)
        sys.exit(1)
