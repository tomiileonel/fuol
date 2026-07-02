import time
import json
from data_pipeline import AdvancedDataPipeline
from api_client import APIFootballClient
from supreme_engine import SupremePredictionEngine
from performance_tracker import ModelTelemetry
import numpy as np

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]]),
    "4-4-2": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,14], [40,34], [40,48], [45,54], [75,25], [75,43]])
}

def run_shadow_mode_test():
    print("[Orchestrator] Iniciando Shadow Mode Test (Liga Argentina)...")
    client = APIFootballClient(use_mock=True)
    pipeline = AdvancedDataPipeline()
    telemetry = ModelTelemetry()
    
    # 1. Fase Pre-Match: Extraer Fixture y Formaciones de la API
    fixture_id = 9999
    raw_data = client.fetch_match_data(fixture_id)
    
    # Extraemos la telemetría para River (Home) y Boca (Away)
    river_telemetry = pipeline.parse_raw_match_data(raw_data, target_team_id=100)
    boca_telemetry = pipeline.parse_raw_match_data(raw_data, target_team_id=200)
    
    form_a = river_telemetry["formation_str"]
    form_b = boca_telemetry["formation_str"]
    print(f"[Orchestrator] River Plate ({form_a}) vs Boca Juniors ({form_b})")
    
    # Mocks de historial para Bayes (Normalmente vendrían de la DB)
    river_history = [{"xg_for": 1.8, "xg_against": 0.5}, {"xg_for": 2.0, "xg_against": 1.0}]
    boca_history = [{"xg_for": 1.2, "xg_against": 1.1}, {"xg_for": 0.8, "xg_against": 0.9}]
    
    # Inyectamos en el Supreme Engine
    engine = SupremePredictionEngine(
        "River Plate", "Boca Juniors",
        [{"gf": m["xg_for"], "gc": m["xg_against"]} for m in river_history], 
        [{"gf": m["xg_for"], "gc": m["xg_against"]} for m in boca_history],
        FORMACIONES
    )
    
    res, _ = engine.run_pipeline(form_str_a=form_a, form_str_b=form_b)
    
    print("\n[Orchestrator] Vector Resultante de Predicción:")
    print(json.dumps({
        "1X2": res["1X2"],
        "Marcador_Mas_Probable": [int(x) for x in res["Marcador Exacto"]]
    }, indent=2))
    
    # Log en SQLite
    telemetry.log_prediction("River Plate", "Boca Juniors", res)
    
    # 2. Extracción de Odds (Cuotas del mercado)
    print("\n[Orchestrator] Extrayendo cuotas del mercado (Las Vegas / Bet365)...")
    odds_data = client.fetch_odds_data(fixture_id)
    bets = odds_data["response"][0]["bookmakers"][0]["bets"][0]["values"]
    print("Cuotas:", bets)
    
    # 3. Fase Post-Match: Simulación de resultado real y cálculo de Brier
    print("\n[Orchestrator] Inyectando resultado post-partido (River 2-0 Boca)...")
    telemetry.log_actual_result("River Plate", "Boca Juniors", 2, 0)
    
    # Audit
    telemetry.calculate_metrics()

if __name__ == "__main__":
    run_shadow_mode_test()
