import asyncio
from omni_orchestrator import OmniContextOrchestrator

async def run_demo():
    orchestrator = OmniContextOrchestrator()
    # Mocking search_web_news to avoid duckduckgo hanging
    orchestrator.ai_analyst.search_web_news = lambda a, b: asyncio.sleep(0, result={"team_a": [], "team_b": []})
    orchestrator.finance.get_squad_values = lambda a, b: asyncio.sleep(0, result={"team_a": 150_000_000, "team_b": 150_000_000})
    
    result = await orchestrator.gather_360_context("Australia", "Egypt")
    print("\n--- RESUMEN DEL MASTER DOCUMENT ---")
    print(f"Match ID: {result['match_id']}")
    print(f"Predicción P(1): {result['engine_prediction'].get('p1', 'N/A')}")
    print(f"Predicción P(X): {result['engine_prediction'].get('px', 'N/A')}")
    print(f"Predicción P(2): {result['engine_prediction'].get('p2', 'N/A')}")
    if 'top_5_scores' in result['engine_prediction']:
        print("Top 5 Scores:")
        for s in result['engine_prediction']['top_5_scores']:
            print(f"  {s['score']} -> {s['prob']:.2%}")
    print(f"Goles Esperados (lam, mu): {result['engine_prediction'].get('lam', 'N/A')}, {result['engine_prediction'].get('mu', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(run_demo())
