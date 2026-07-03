import os
import json
import asyncio
from duckduckgo_search import DDGS
from google import genai
from google.genai import types

class QualititaveWebAnalyst:
    """
    Agente de IA NLP que busca noticias en internet y las convierte en 
    modificadores numéricos para el motor predictivo unificado.
    """
    def __init__(self):
        # We assume the user has a GEMINI_API_KEY environment variable.
        self.api_key = os.environ.get("GEMINI_API_KEY")
        # In a real environment, we'd initialize the client if we have the key.
        # Otherwise, we use a fallback mock so the orchestrator doesn't crash in local dev.
        self.has_llm = bool(self.api_key)
        if self.has_llm:
            self.client = genai.Client(api_key=self.api_key)

    async def search_web_news(self, team_a: str, team_b: str) -> dict:
        """
        Busca las últimas noticias para ambos equipos utilizando DuckDuckGo Search.
        """
        news_data = {"team_a": [], "team_b": []}
        
        try:
            # We use asyncio.to_thread to run the sync DDGS in a non-blocking way
            def get_news(query):
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=5))
                    
            query_a = f"{team_a} national football team injuries news today"
            results_a = await asyncio.to_thread(get_news, query_a)
            news_data["team_a"] = [res.get("body", res.get("title", "")) for res in results_a]
            
            query_b = f"{team_b} national football team injuries news today"
            results_b = await asyncio.to_thread(get_news, query_b)
            news_data["team_b"] = [res.get("body", res.get("title", "")) for res in results_b]

                
        except Exception as e:
            print(f"[Web Search ERROR] DuckDuckGo search failed: {e}")
            
        return news_data

    def extract_modifiers(self, news_data: dict) -> dict:
        """
        Utiliza el LLM (Gemini) para interpretar las noticias y extraer modificadores:
        - injury_modifier: 0.85 (lesiones graves) a 1.0 (equipo completo)
        - travel_fatigue: 0.90 (mucha fatiga) a 1.0 (sin fatiga)
        """
        default_modifiers = {
            "team_a": {"injury_modifier": 1.0, "travel_fatigue": 1.0},
            "team_b": {"injury_modifier": 1.0, "travel_fatigue": 1.0}
        }
        
        if not self.has_llm:
            print("[NLP Agent] GEMINI_API_KEY no encontrada. Devolviendo modificadores por defecto.")
            return default_modifiers
            
        prompt = f"""
        Eres un analista de datos deportivos Quant. Te proporcionaré las últimas noticias
        buscadas sobre dos equipos nacionales de fútbol.
        Tu tarea es determinar los multiplicadores de fuerza para cada equipo basados 
        en problemas de lesiones (injury_modifier) o fatiga/contexto negativo (travel_fatigue).
        
        Reglas para los modificadores:
        - 1.0 = Situación normal o perfecta. Ninguna lesión clave, sin fatiga inusual.
        - 0.85 = Estrella principal lesionada o crisis grave en el plantel.
        - 0.90 a 0.95 = Lesiones menores o fatiga reportada.
        
        Noticias Equipo A: {news_data.get('team_a')}
        Noticias Equipo B: {news_data.get('team_b')}
        
        DEBES responder estrictamente en formato JSON:
        {{
            "team_a": {{"injury_modifier": float, "travel_fatigue": float}},
            "team_b": {{"injury_modifier": float, "travel_fatigue": float}}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            parsed_json = json.loads(response.text)
            
            # Guardrails (NLP Clamp)
            # Aseguramos que los modificadores no excedan los límites matemáticos (0.80 - 1.20)
            for team in ['team_a', 'team_b']:
                if team in parsed_json:
                    parsed_json[team]['injury_modifier'] = max(0.80, min(1.20, float(parsed_json[team].get('injury_modifier', 1.0))))
                    parsed_json[team]['travel_fatigue'] = max(0.80, min(1.20, float(parsed_json[team].get('travel_fatigue', 1.0))))
            
            return parsed_json
        except Exception as e:
            print(f"[NLP Agent ERROR] Fallo al generar modificadores con Gemini: {e}")
            return default_modifiers

if __name__ == "__main__":
    # Test script
    async def test():
        agent = QualititaveWebAnalyst()
        news = await agent.search_web_news("Argentina", "France")
        print("News:", news)
        mods = agent.extract_modifiers(news)
        print("Modifiers:", mods)
        
    asyncio.run(test())
