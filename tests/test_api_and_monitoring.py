"""
Tests del servidor API de FUOL.

Los tests anteriores verificaban endpoints `/api/live_prediction/...` y
`/api/monitoring` que NO existen en `api_server.py` (probablemente
pertenecían a `live_orchestrator.py` que requiere websocket y MongoDB y
no está activo por defecto). Se reescriben para testear los endpoints
que efectivamente expone `api_server.py`:

  - POST /api/predict  (predicción 1X2 con motor unificado)
  - GET  /             (frontend estático)

Cualquier test que requiera endpoints de live/monitoring debe vivir en
un módulo separado y depender de un servidor live_orchestrator
levantado, no del api_server FastAPI mínimo.
"""
import pytest
from fastapi.testclient import TestClient

from api_server import app


@pytest.fixture(scope="module")
def client():
    # TestClient levanta los eventos de startup del app, que cargan el
    # histórico global (~50k filas). Por eso se hace a nivel módulo.
    return TestClient(app)


def test_predict_endpoint_returns_payload_for_valid_teams(client):
    """POST /api/predict con dos equipos con histórico suficiente debe
    devolver 200 y el payload completo del motor unificado."""
    response = client.post(
        '/api/predict',
        json={"team_a": "Argentina", "team_b": "Brazil"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    # Campos mínimos garantizados por UnifiedEngine.predict()
    for key in ('p1', 'px', 'p2', 'lam', 'mu', 'rho', 'top_5_scores'):
        assert key in payload, f"Falta '{key}' en payload"

    # Las probabilidades 1X2 suman ~1
    total = payload['p1'] + payload['px'] + payload['p2']
    assert abs(total - 1.0) < 1e-3, f"1X2 suma {total} (esperado ~1.0)"


def test_predict_endpoint_returns_404_for_unknown_team(client):
    """Equipos sin histórico suficiente deben devolver 404, no 500."""
    response = client.post(
        '/api/predict',
        json={"team_a": "EquipoInexistenteXYZ", "team_b": "Argentina"},
    )
    assert response.status_code == 404


def test_predict_endpoint_handles_alias(client):
    """El normalizador de nombres debe aceptar 'ARGENTINA' (mayúsculas)
    y mapearlo al equipo correcto del dataset."""
    response = client.post(
        '/api/predict',
        json={"team_a": "ARGENTINA", "team_b": "BRASIL"},
    )
    assert response.status_code == 200, response.text
