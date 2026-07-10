"""
fuol_scout_cli.py
Módulo Scouter Automatizado para obtener contexto cualitativo (Fondos 0).
Uso: python fuol_scout_cli.py "Equipo A" "Equipo B"
"""
import sys
import json
import requests
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

from unified_engine import UnifiedEngine
from data_pipeline import DataPipeline

class ContextScouter:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (FUOL_AI_Agent)'}
        self.fbref_team_ids = {
            "ARGENTINA": "f9fddd6e",
            "SUIZA": "81021a70",
            "BRASIL": "304635c3",
            "INGLATERRA": "18bb7c10",
            "NORUEGA": "6c4961d1"
        }

    def get_fbref_stats(self, team_name: str) -> dict:
        # Mock de FBref (pendiente de implementación completa con pandas.read_html)
        return {
            "ppda": 10.5,
            "xg_buildup": 1.2,
            "packing": 15
        }

    def get_weather(self, venue: str) -> dict:
        venues = {
            "BUENOS AIRES": (-34.6037, -58.3816),
            "OSLO": (59.9139, 10.7522),
            "BERNA": (46.9480, 7.4474),
            "LONDRES": (51.5074, -0.1278)
        }
        coords = venues.get(venue.upper(), (0, 0))
        if coords == (0, 0):
            return {"error": "Sede no encontrada"}
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&current=temperature_2m,wind_speed_10m,rain"
        try:
            resp = requests.get(url, headers=self.headers, timeout=5).json()
            return {
                "temp_c": resp.get('current', {}).get('temperature_2m', 'N/A'),
                "wind_kmh": resp.get('current', {}).get('wind_speed_10m', 'N/A'),
                "rain": "Sí" if resp.get('current', {}).get('rain', 0) > 0 else "No"
            }
        except:
            return {"error": "Fallo API Clima"}

    def scout(self, team_a: str, team_b: str) -> dict:
        print(f"[Scouter] Recopilando datos para {team_a} vs {team_b}...", file=sys.stderr)
        
        # 1. Contexto Externo
        stats_a = self.get_fbref_stats(team_a)
        stats_b = self.get_fbref_stats(team_b)
        weather = self.get_weather("BUENOS AIRES") # Sede hardcodeada para el ejemplo
        
        # 2. Base Matemática (Motor FUOL)
        print("[Scouter] Ejecutando motor Dixon-Coles...", file=sys.stderr)
        pipeline = DataPipeline()
        df, _ = pipeline.prepare_data()
        df['date'] = pd.to_datetime(df['date'])
        hist_df = df[df['date'] < pd.Timestamp.now().normalize()]
        
        matches_a = pipeline.get_team_history(hist_df, team_a)
        matches_b = pipeline.get_team_history(hist_df, team_b)
        
        engine = UnifiedEngine(team_a, team_b, matches_a, matches_b, venue='N', half_life=365.0)
        pred = engine.predict()
        
        if 'score_matrix' in pred:
            pred['score_matrix'] = pred['score_matrix'].tolist()

        # 3. Ensamblar JSON Enriquecido para la IA
        master_doc = {
            "fixture": {
                "home": team_a,
                "away": team_b,
                "date": datetime.now().strftime('%Y-%m-%d')
            },
            "context": {
                "weather": weather,
                "referees": {"central": "TBD", "var": "TBD"} # Pendiente de scrapear 24h antes
            },
            "advanced_metrics": {
                "home": stats_a,
                "away": stats_b
            },
            "fuol_math_engine": pred
        }
        return master_doc

def main():
    if len(sys.argv) < 3:
        print("Error: Uso -> python fuol_scout_cli.py 'Equipo A' 'Equipo B'")
        sys.exit(1)

    team_a = sys.argv[1].upper()
    team_b = sys.argv[2].upper()

    scouter = ContextScouter()
    master_doc = scouter.scout(team_a, team_b)

    # Imprimir SOLO el JSON en stdout para que la IA (Antigravity) lo consuma
    print(json.dumps(master_doc, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
