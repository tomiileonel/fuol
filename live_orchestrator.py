from live_scouting_provider import LiveScoutingOrchestrator, TeamDossier, ScoutingDataError
from unified_engine import UnifiedEngine
from typing import Dict, Any
import asyncio

class LiveMatchPredictor:
    """Orquesta la recolección de datos en vivo y corre el UnifiedEngine."""
    
    def __init__(self, scouter: LiveScoutingOrchestrator):
        self.scouter = scouter

    async def predict_match(self, team_a: str, team_b: str, venue: str = 'N') -> Dict[str, Any]:
        print(f"[LiveMatchPredictor] Iniciando predicción en vivo para {team_a} vs {team_b}")
        
        try:
            dossier_a, dossier_b = await asyncio.gather(
                self.scouter.scout_team(team_a),
                self.scouter.scout_team(team_b)
            )
        except ScoutingDataError as e:
            print(f"[LiveMatchPredictor ERROR] Falló la recolección de datos: {e}")
            return {"error": str(e)}

        print("[LiveMatchPredictor] Datos recolectados exitosamente. Corriendo UnifiedEngine...")
        
        engine = UnifiedEngine(
            team_a=team_a,
            team_b=team_b,
            matches_a=dossier_a.to_engine_matches(),
            matches_b=dossier_b.to_engine_matches(),
            venue=venue,
            modifiers_a=dossier_a.to_engine_modifiers(),
            modifiers_b=dossier_b.to_engine_modifiers(),
            base_elo_a=dossier_a.elo_estimate,
            base_elo_b=dossier_b.elo_estimate
        )
        
        try:
            prediction = engine.predict()
            if 'score_matrix' in prediction:
                prediction['score_matrix'] = prediction['score_matrix'].tolist()
        except Exception as e:
            print(f"[LiveMatchPredictor ERROR] Falló UnifiedEngine: {e}")
            return {"error": str(e)}

        master_document = {
            "match_id": f"{team_a}_vs_{team_b}",
            "metadata": {
                "team_a_info": {
                    "coach": dossier_a.coach_name,
                    "value_eur": dossier_a.financials.total_value_eur,
                    "fifa_rank": dossier_a.fifa_rank,
                    "elo_estimate": dossier_a.elo_estimate
                },
                "team_b_info": {
                    "coach": dossier_b.coach_name,
                    "value_eur": dossier_b.financials.total_value_eur,
                    "fifa_rank": dossier_b.fifa_rank,
                    "elo_estimate": dossier_b.elo_estimate
                }
            },
            "engine_prediction": prediction
        }
        
        return master_document
