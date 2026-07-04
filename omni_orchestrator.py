import asyncio
from database_manager import MongoDBLake
from live_orchestrator import LiveMatchPredictor
from live_scouting_provider import LiveScoutingOrchestrator
from search_backends import DDGSearchBackend

class OmniContextOrchestrator:
    """
    El Orquestador Maestro del sistema Quant 360°.
    Recolecta datos estadísticos, financieros y cualitativos asincronamente
    desde la web en vivo, y los fusiona en un documento maestro en MongoDB.
    """
    def __init__(self):
        # Inyectar las nuevas dependencias (Live Scouting Layer)
        self.backend = DDGSearchBackend()
        self.scouter = LiveScoutingOrchestrator(self.backend)
        self.live_predictor = LiveMatchPredictor(self.scouter)
        self.db = MongoDBLake()

    async def gather_360_context(self, team_a: str, team_b: str):
        print(f"[{team_a} vs {team_b}] Desplegando redes de recolección 360° en vivo...")
        
        # 1. Correr el LiveMatchPredictor completo (Web Scrape + UnifiedEngine)
        master_document = await self.live_predictor.predict_match(team_a, team_b)
        
        if "error" in master_document:
            print(f"[{team_a} vs {team_b}] Abortando guardado por error: {master_document['error']}")
            return master_document
            
        # 2. Guardar en el Data Lake
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
        result = await orchestrator.gather_360_context("Francia", "Paraguay")
        if "error" not in result:
            print("\n--- RESUMEN DEL MASTER DOCUMENT ---")
            print(f"Match ID: {result['match_id']}")
            print(f"Predicción P(1): {result['engine_prediction'].get('p1', 'N/A')}")
            print(f"Valor A (EUR): {result['metadata']['team_a_info']['value_eur']}")
            print(f"Valor B (EUR): {result['metadata']['team_b_info']['value_eur']}")
        else:
            print("Hubo un error en la recolección o predicción.")
            
    # Run the event loop
    asyncio.run(run_demo())
