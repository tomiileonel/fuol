import time
import numpy as np
from data_pipeline import AdvancedDataPipeline
from performance_tracker import ModelTelemetry
from api_client import APIFootballClient
from unified_engine import UnifiedEngine, run_prediction

# NOTA: FORMACIONES (grillas de coordenadas x,y para renderizado táctico) se elimina
# de este archivo. UnifiedEngine no acepta formaciones -- ese input pertenecía al
# viejo SupremePredictionEngine (basado en Voronoi/geometría), que ya no existe.
# Mantenerlo aquí era código muerto que nunca se pasaba a ningún lado.

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
        
        # Simulamos historial para calcular Priors (normalmente vendría de telemetría SQLite).
        # Incluimos 'date' y 'opponent' explícitos: sin ellos, TimeWeighter y EloRating
        # caen en sus fallbacks (peso plano, Elo 1600 por defecto) y el motor pierde
        # toda la ventaja de la ponderación temporal y el prior histórico.
        mock_history_home = [
            {"date": "2026-06-05", "opponent": "Colombia", "gf": 2, "gc": 0, "res": "W"},
            {"date": "2026-06-10", "opponent": "Uruguay",  "gf": 1, "gc": 1, "res": "D"},
            {"date": "2026-06-18", "opponent": "Paraguay", "gf": 3, "gc": 1, "res": "W"},
        ]
        mock_history_away = [
            {"date": "2026-06-04", "opponent": "Chile",    "gf": 1, "gc": 1, "res": "D"},
            {"date": "2026-06-11", "opponent": "Peru",     "gf": 0, "gc": 2, "res": "L"},
            {"date": "2026-06-19", "opponent": "Ecuador",  "gf": 2, "gc": 2, "res": "D"},
        ]

        # Corremos la predicción con el motor real (UnifiedEngine vía run_prediction)
        res = run_prediction(
            home_name, away_name,
            mock_history_home, mock_history_away,
            venue="H", verbose=False
        )

        # Guardamos en la base de datos (schema plano: p1/px/p2/lam/mu)
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
