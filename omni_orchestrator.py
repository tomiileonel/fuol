import asyncio
from api_client import EnhancedAPIClient
from unified_engine import UnifiedEngine
from ai_web_agent import QualititaveWebAnalyst
from financial_scraper import TransfermarktScraper
from database_manager import MongoDBLake

TEAM_ID_MAP = {
    "Argentina": 26,
    "France": 67,
    "Brazil": 6,
    "England": 10,
    "Spain": 9,
    "Australia": 20,
    "Egypt": 32,
    "Senegal": 24,
    "Belgium": 1
}

class OmniContextOrchestrator:
    """
    El Orquestador Maestro del sistema Quant 360°.
    Recolecta datos estadísticos, financieros y cualitativos asincronamente
    y los fusiona en un documento maestro en MongoDB.
    """
    def __init__(self):
        self.api = EnhancedAPIClient(use_mock=True) # Set to True for testing without burning API quota
        self.ai_analyst = QualititaveWebAnalyst()
        self.finance = TransfermarktScraper()
        self.db = MongoDBLake()

    async def gather_360_context(self, team_a: str, team_b: str):
        print(f"[{team_a} vs {team_b}] Desplegando redes de recolección 360°...")
        
        # Obtenemos IDs para la API
        id_a = TEAM_ID_MAP.get(team_a, 100)
        id_b = TEAM_ID_MAP.get(team_b, 200)
        
        # 1. Ejecutar recolección masiva en paralelo (Asincronismo puro)
        print("[Orchestrator] Lanzando llamadas concurrentes (API + Web + Scraper)...")
        data_a, data_b, h2h, news, finances = await asyncio.gather(
            self.api.get_full_profile(id_a),
            self.api.get_full_profile(id_b),
            self.api.get_head_to_head(id_a, id_b),
            self.ai_analyst.search_web_news(team_a, team_b),
            self.finance.get_squad_values(team_a, team_b)
        )

        # 2. El Analista de IA convierte las noticias de internet en números
        print("[Orchestrator] Extrayendo modificadores cualitativos con LLM...")
        qualitative_modifiers = self.ai_analyst.extract_modifiers(news)

        # 3. Correr tu UnifiedEngine con toda la carga
        print("[Orchestrator] Corriendo UnifiedEngine...")
        try:
            engine = UnifiedEngine(
                team_a=team_a, 
                team_b=team_b,
                matches_a=data_a.get('last_100_matches', []), 
                matches_b=data_b.get('last_100_matches', []),
                modifiers_a=qualitative_modifiers.get('team_a', {}), 
                modifiers_b=qualitative_modifiers.get('team_b', {})
            )
            prediction = engine.predict()
            
            # Serialize Numpy objects for MongoDB
            # Convert prediction matrix to standard python types if present
            if 'score_matrix' in prediction:
                prediction['score_matrix'] = prediction['score_matrix'].tolist()
        except Exception as e:
            print(f"[Orchestrator ERROR] Fallo al ejecutar UnifiedEngine: {e}")
            prediction = {"error": str(e)}

        # 4. Ensamblar el Payload Maestro
        print("[Orchestrator] Ensamblando Master Document...")
        master_document = {
            "match_id": f"{team_a}_vs_{team_b}",
            "metadata": {
                "team_a_info": {
                    "coach": data_a.get('coach'), 
                    "squad": data_a.get('roster'), 
                    "value_eur": finances.get('team_a'), 
                    "group_points": data_a.get('group_pts')
                },
                "team_b_info": {
                    "coach": data_b.get('coach'), 
                    "squad": data_b.get('roster'), 
                    "value_eur": finances.get('team_b'), 
                    "group_points": data_b.get('group_pts')
                },
            },
            "web_context": {
                "raw_news": news,
                "extracted_modifiers": qualitative_modifiers
            },
            "engine_prediction": prediction
        }

        # 5. Guardar en el Data Lake (No más archivos .py sueltos)
        print("[Orchestrator] Guardando perfil en MongoDB...")
        success = self.db.save_match_profile(master_document)
        
        if success:
            print(f"[{team_a} vs {team_b}] Operación 360° completada exitosamente.")
        else:
            print(f"[{team_a} vs {team_b}] Completado, pero falló guardado en BD.")
            
        return master_document

if __name__ == "__main__":
    async def run_demo():
        orchestrator = OmniContextOrchestrator()
        result = await orchestrator.gather_360_context("Argentina", "France")
        # Mostrar resumen
        print("\n--- RESUMEN DEL MASTER DOCUMENT ---")
        print(f"Match ID: {result['match_id']}")
        print(f"Modificadores A: {result['web_context']['extracted_modifiers'].get('team_a')}")
        print(f"Predicción P(1): {result['engine_prediction'].get('p1', 'N/A')}")
        print(f"Valor A (EUR): {result['metadata']['team_a_info']['value_eur']}")
        
    # Run the event loop
    asyncio.run(run_demo())
