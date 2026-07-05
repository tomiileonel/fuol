from unified_engine_v3 import UnifiedEngineV3


def test_unified_engine_v3_predicts_reasonably():
    matches_a = [
        {'date': '2024-01-01', 'home': 'ARGENTINA', 'away': 'CHILE', 'gh': 2, 'ga': 1, 'minute': 10},
        {'date': '2024-02-01', 'home': 'ARGENTINA', 'away': 'PERU', 'gh': 1, 'ga': 0, 'minute': 23},
    ]
    matches_b = [
        {'date': '2024-01-15', 'home': 'BRASIL', 'away': 'URUGUAY', 'gh': 1, 'ga': 1, 'minute': 45},
        {'date': '2024-02-15', 'home': 'BRASIL', 'away': 'COL', 'gh': 2, 'ga': 0, 'minute': 60},
    ]

    engine = UnifiedEngineV3(
        team_a='ARGENTINA',
        team_b='BRASIL',
        matches_a=matches_a,
        matches_b=matches_b,
        venue='H',
        team_confederations={'ARGENTINA': 'CONMEBOL', 'BRASIL': 'CONMEBOL'},
    )

    pred = engine.predict()
    assert 0.0 <= pred.p_home <= 1.0
    assert 0.0 <= pred.p_draw <= 1.0
    assert 0.0 <= pred.p_away <= 1.0
    assert pred.confidence_score >= 0.0
