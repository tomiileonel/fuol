import numpy as np
import json
from datetime import datetime

class AdvancedDataPipeline:
    def __init__(self, api_provider="Sportmonks_Simulated"):
        self.provider = api_provider
        
    # ==============================================================================
    # 1. EXTRACCIÓN (Simulación de Ingesta de API)
    # ==============================================================================
    def parse_raw_match_data(self, raw_json, target_team_id=None):
        """
        Extrae la telemetría profunda de un payload JSON de API-Football.
        Parsea 'formation' estática si no hay posiciones medias.
        """
        data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        
        try:
            resp = data.get("response", [])[0]
            date = resp.get("fixture", {}).get("date", "1970-01-01").split("T")[0]
            
            # Identificamos qué lado del JSON es el equipo objetivo
            home_team = resp.get("teams", {}).get("home", {})
            away_team = resp.get("teams", {}).get("away", {})
            is_home = (target_team_id is None) or (home_team.get("id") == target_team_id)
            
            opp_name = away_team.get("name", "Unknown") if is_home else home_team.get("name", "Unknown")
            
            # Buscar alineación
            lineups = resp.get("lineups", [])
            formation_str = "4-3-3" # Fallback
            for l in lineups:
                team_id = l.get("team", {}).get("id")
                if (is_home and team_id == home_team.get("id")) or (not is_home and team_id == away_team.get("id")):
                    formation_str = l.get("formation", "4-3-3")
                    
            # Buscar estadísticas
            stats_list = resp.get("statistics", [])
            team_stats = []
            opp_stats = []
            for s in stats_list:
                team_id = s.get("team", {}).get("id")
                if (is_home and team_id == home_team.get("id")) or (not is_home and team_id == away_team.get("id")):
                    team_stats = s.get("statistics", [])
                else:
                    opp_stats = s.get("statistics", [])
                    
            def get_stat(stats_array, stat_name, default=0.0):
                for s in stats_array:
                    if s.get("type", "").lower() == stat_name.lower():
                        val = s.get("value")
                        if val is None: return default
                        if isinstance(val, str) and "%" in val: return float(val.replace("%", ""))
                        return float(val)
                return default

            parsed_record = {
                "date": date,
                "opponent": opp_name,
                "venue": "H" if is_home else "A",
                "formation_str": formation_str,
                "xg_for": get_stat(team_stats, "expected_goals", 1.0),
                "xg_against": get_stat(opp_stats, "expected_goals", 1.0),
                "gf": get_stat(team_stats, "expected_goals", 1.0),  # Inyección directa de xG para el UnifiedEngine
                "gc": get_stat(opp_stats, "expected_goals", 1.0),   # Inyección directa de xG para el UnifiedEngine
                "possession_pct": get_stat(team_stats, "ball possession", 50.0),
                "field_tilt": get_stat(team_stats, "ball possession", 50.0), # Fallback si no hay field tilt
                "avg_positions": [], # Vacío provocará el fallback a FORMACIONES
                "actual_gf": resp.get("goals", {}).get("home", 0) if is_home else resp.get("goals", {}).get("away", 0),
                "actual_gc": resp.get("goals", {}).get("away", 0) if is_home else resp.get("goals", {}).get("home", 0)
            }
            return parsed_record
        except Exception as e:
            print(f"Error parseando JSON: {e}")
            return None

    # ==============================================================================
    # 2. TRANSFORMACIÓN TENSORIAL (Álgebra y Topología)
    # ==============================================================================
    def transform_telemetry_to_prior(self, historical_matches, qualitative_modifier=1.0):
        """
        Convierte una serie temporal de partidos en los Priors Bayesianos 
        exactos que necesita el motor, usando xG en lugar de goles empíricos.
        Permite inyectar un factor cualitativo (fatiga, lesiones) para asimetría.
        """
        if not historical_matches:
            return 1.0, 1.0 # Fallback de seguridad
            
        xg_f = np.array([m["xg_for"] for m in historical_matches])
        xg_a = np.array([m["xg_against"] for m in historical_matches])
        
        # Ponderación exponencial temporal (Concept Drift)
        # Los partidos de hace 1 año pesan menos que los de ayer.
        n = len(historical_matches)
        weights = np.exp(np.linspace(-1.5, 0, n))
        weights /= weights.sum()
        
        # Inyectamos el modificador cualitativo directamente al prior (ej. 0.8 = merma ofensiva)
        lambda_prior = np.average(xg_f, weights=weights) * qualitative_modifier
        # Si el equipo está mermado (qualitative_modifier < 1.0), su defensa se asume más vulnerable
        mu_prior = np.average(xg_a, weights=weights) / qualitative_modifier
        
        return lambda_prior, mu_prior

    def transform_positions_to_voronoi_matrix(self, match_record):
        """
        Toma las posiciones (x, y) de la API y las formatea al tensor.
        Si la API no las tiene, usa el fallback basado en 'formation_str'.
        """
        raw_pos = match_record.get("avg_positions", [])
        if len(raw_pos) != 11:
            # Fallback a la formación entregada por la API o 4-3-3
            form_str = match_record.get("formation_str", "4-3-3")
            # Para simplificar la inyección de FORMACIONES, la importaremos si es necesario
            # Pero dado que data_pipeline no tiene FORMACIONES importada directamente,
            # usamos una versión quemada aquí o lo resolvemos asumiendo que el Engine
            # usa este tensor como fallback.
            # Mejor aún: devolvemos None y el Engine lo resuelve, o devolvemos form_str
            return form_str
            
        # Normalización estricta al plano cartesiano del campo (105x68m)
        # Normalización estricta al plano cartesiano del campo (105x68m)
        tensor = np.zeros((11, 2))
        for idx, player in enumerate(raw_pos):
            tensor[idx, 0] = max(0.0, min(105.0, float(player["x"])))
            tensor[idx, 1] = max(0.0, min(68.0, float(player["y"])))
            
        return tensor

    # ==============================================================================
    # 3. CARGA (Ensamblaje del Payload para el Engine)
    # ==============================================================================
    def build_engine_payload(self, team_history, current_match_telemetry):
        """
        Orquesta la transformación final. Ahora devuelve las listas compatibles 
        con el UnifiedEngine, exponiendo la telemetría enriquecida.
        """
        voronoi_tensor = self.transform_positions_to_voronoi_matrix(current_match_telemetry)
        field_tilt = current_match_telemetry.get("field_tilt", 50.0) / 100.0
        
        return {
            "enriched_matches": team_history,
            "voronoi_formation": voronoi_tensor,
            "momentum_modifier": field_tilt 
        }
