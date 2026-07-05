from fastapi.testclient import TestClient

from api_server import app


def test_live_prediction_endpoint_returns_payload():
    client = TestClient(app)
    response = client.get('/api/live_prediction/FRA_SEN_WC26')
    assert response.status_code == 200
    payload = response.json()
    assert payload['match_id'] == 'FRA_SEN_WC26'
    assert 'engine_prediction' in payload


def test_monitoring_endpoint_returns_metrics():
    client = TestClient(app)
    response = client.get('/api/monitoring')
    assert response.status_code == 200
    payload = response.json()
    assert 'backtest' in payload
