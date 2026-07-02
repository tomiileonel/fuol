import time
import numpy as np
from data_pipeline import AdvancedDataPipeline
from performance_tracker import ModelTelemetry
from api_client import APIFootballClient
from supreme_engine import SupremePredictionEngine

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]]),
    "4-4-2": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,14], [40,34], [40,48], [45,54], [75,25], [75,43]])
}

class ShadowModePoC:
    def __init__(self, api_key=None, use_mock=True):
        # Inyectamos nuestro wrapper de API
        self.client = APIFootballClient(api_key=api_key, use_mock=use_mock)
        self.telemetry = ModelTelemetry()
        self.pipeline = AdvancedDataPipeline()

    def ejecutar_viernes_predicciones(self, league_id=128, season=2026):
        """Extrae el fixture del fin de semana y corre el Supreme Engine."""
        print("[PoC] Iniciando extraccion de fixture y predicciones pre-partido...")
        
        # En la vida real, iteraríamos sobre todos los partidos del fin de semana.
        # Aquí usamos nuestro fixture mock de la Liga Argentina.
        fixture_id = 9999
        raw_data = self.client.fetch_match_data(fixture_id)
        
        # Parseamos usando nuestro AdvancedDataPipeline adaptado a API-Football
        home_team = self.pipeline.parse_raw_match_data(raw_data, target_team_id=100)
        away_team = self.pipeline.parse_raw_match_data(raw_data, target_team_id=200)
        
        home_name = "River Plate"
        away_name = "Boca Juniors"
        
        print(f"[PoC] Procesando: {home_name} ({home_team['formation_str']}) vs {away_name} ({away_team['formation_str']})")
        
        # Simulamos historial para calcular Priors (normalmente vendría de telemetría SQLite)
        mock_history_home = [{"xg_for": 1.8, "xg_against": 0.5}]
        mock_history_away = [{"xg_for": 1.2, "xg_against": 1.1}]
        
        engine = SupremePredictionEngine(
            home_name, away_name,
            [{"gf": m["xg_for"], "gc": m["xg_against"]} for m in mock_history_home], 
            [{"gf": m["xg_for"], "gc": m["xg_against"]} for m in mock_history_away],
            FORMACIONES
        )
        
        # Corremos el pipeline predictivo
        res, _ = engine.run_pipeline(form_str_a=home_team["formation_str"], form_str_b=away_team["formation_str"])
        
        # Guardamos en la base de datos
        self.telemetry.log_prediction(home_name, away_name, res)
        print("[PoC] Prediccion guardada exitosamente en supreme_predictions.db")
        
        # Simulación de Snapshot manual
        print("[PoC] Extrayendo cuotas del mercado (Mock Bet365)...")
        self.telemetry.log_market_odds(home_name, away_name, 2.10, 3.10, 3.60)
        
        return [fixture_id]

    def ejecutar_lunes_auditoria(self, fixture_ids):
        """Busca los resultados reales y cierra el ciclo de feedback."""
        print("[PoC] Inyectando realidad empirica para auditoria de Brier Score...")
        
        # En un flujo real, consultariamos la API para verificar si el status es FT (Full Time)
        # y extraeriamos los goles. Aqui lo simulamos.
        # Resultado simulado: River 1 - 1 Boca
        self.telemetry.log_actual_result("River Plate", "Boca Juniors", 1, 1)
        
        print("[PoC] Evaluando desempeño del algoritmo...")
        metrics = self.telemetry.calculate_metrics()
        return metrics

if __name__ == "__main__":
    poc = ShadowModePoC(use_mock=True)
    
    print("\n" + "="*50)
    print(" FASE 1: INGESTA Y PREDICCION (VIERNES)")
    print("="*50)
    fixtures_pendientes = poc.ejecutar_viernes_predicciones()
    
    print("\n" + "="*50)
    print(" FASE 2: BUCLE DE ESPERA (SAB/DOM)")
    print("="*50)
    print("[PoC] Sistema a la espera de status 'FT' en los partidos...")
    time.sleep(2) # Simulación de paso del tiempo
    
    print("\n" + "="*50)
    print(" FASE 3: AUDITORIA Y CIERRE (LUNES)")
    print("="*50)
    poc.ejecutar_lunes_auditoria(fixtures_pendientes)
