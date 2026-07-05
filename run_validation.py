from __future__ import annotations

from validate_v3 import validate_model


if __name__ == '__main__':
    train_matches = [
        {'date': '2024-01-01', 'home': 'ARGENTINA', 'away': 'CHILE', 'gh': 2, 'ga': 1},
        {'date': '2024-02-01', 'home': 'ARGENTINA', 'away': 'PERU', 'gh': 1, 'ga': 0},
        {'date': '2024-01-15', 'home': 'BRASIL', 'away': 'URUGUAY', 'gh': 1, 'ga': 1},
        {'date': '2024-02-15', 'home': 'BRASIL', 'away': 'COL', 'gh': 2, 'ga': 0},
    ]
    test_matches = [
        {'date': '2024-03-01', 'home': 'FRANCIA', 'away': 'BELGICA', 'gh': 2, 'ga': 2},
        {'date': '2024-03-10', 'home': 'FRANCIA', 'away': 'ESPANA', 'gh': 3, 'ga': 1},
        {'date': '2024-03-20', 'home': 'ARGENTINA', 'away': 'BRASIL', 'gh': 1, 'ga': 1},
    ]
    team_confs = {
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

    report = validate_model(train_matches, test_matches, team_confs)
    print('Validación v3.0 completada')
    print(f'Partidos evaluados: {report.n_matches_evaluated}')
    print(f'Brier: {report.brier_score:.4f}')
    print(f'Hit Rate: {report.hit_rate_pct:.1f}%')
