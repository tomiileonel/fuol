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
 
        NOTA: el rango [0.85, 1.0] es intencional y coincide con el rango que
        el propio prompt le pide al LLM. Estos modificadores solo capturan
        factores negativos (lesiones, fatiga) — no existe hoy un mecanismo
        para que una noticia positiva (DT nuevo, buen momento anímico, regreso
        de una figura) empuje la tasa de gol esperada por ENCIMA del valor base.
        Si en el futuro se quiere modelar eso, hace falta un modificador
        separado y explícito, no ampliar este rango de vuelta a 1.20 sin que
        el prompt lo contemple.
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
            # Aseguramos que los modificadores no excedan los límites que el
            # propio prompt define: 0.85 (peor caso) a 1.0 (situación normal).
            # Antes este rango era [0.80, 1.20], lo cual permitía que el LLM
            # devolviera valores por encima de 1.0 (equipo "mejorado") que el
            # prompt nunca pide ni explica cómo interpretar. Se corrige acá
            # para que el guardrail refleje lo que el prompt realmente pide,
            # en vez de ser más permisivo que las instrucciones que el propio
            # LLM recibió.
            for team in ['team_a', 'team_b']:
                if team in parsed_json:
                    parsed_json[team]['injury_modifier'] = max(0.85, min(1.0, float(parsed_json[team].get('injury_modifier', 1.0))))
                    parsed_json[team]['travel_fatigue'] = max(0.85, min(1.0, float(parsed_json[team].get('travel_fatigue', 1.0))))
            
            return parsed_json
        except Exception as e:
            print(f"[NLP Agent ERROR] Fallo al generar modificadores con Gemini: {e}")
            return default_modifiers
 
    def explain_prediction(self, prediction_json: dict, team_a: str, team_b: str) -> str:
        """
        Toma el JSON de salida del motor (predict()) y usa el LLM para explicar
        los resultados probabilísticos en lenguaje natural futbolístico.
 
        Antes de construir el prompt, valida qué campos de los que el prompt
        promete explicar realmente vinieron en prediction_json. Si UnifiedEngine
        cambia de forma en el futuro, o si se le pasa un diccionario incompleto
        por error, este chequeo lo detecta ANTES de llamar al LLM — en vez de
        depender de que el propio LLM note la ausencia y decida no inventar,
        que era la única defensa que existía antes de este cambio.
        """
        if not self.has_llm:
            return "[NLP Agent] GEMINI_API_KEY no encontrada. No se puede generar la explicación."
 
        # Campos que el prompt de traducción promete reportar en su
        # "Estructura de salida". Si UnifiedEngine.predict() cambia de forma,
        # esta lista es lo primero que hay que revisar y actualizar.
        expected_fields = [
            "p1", "px", "p2", "top_5_scores",
            "lam", "mu", "lam_std", "mu_std", "ci_lam_90", "ci_mu_90",
            "elo_a", "elo_b", "rho", "half_life_days",
        ]
        missing = [f for f in expected_fields if f not in prediction_json]
 
        missing_note = ""
        if missing:
            missing_note = (
                "\nADVERTENCIA: los siguientes campos NO vinieron en el JSON: "
                f"{', '.join(missing)}. Para cada uno, decí explícitamente "
                "'este dato no está disponible' en la sección que corresponda. "
                "NO los completes con estimaciones propias ni con conocimiento "
                "futbolístico general, bajo ninguna circunstancia.\n"
            )
 
        prompt = f"""
Actuá como analista cuantitativo de fútbol. Te voy a dar el output JSON crudo de un modelo estadístico (Dixon-Coles + Elo + inferencia bayesiana) para un partido específico. Tu única tarea es traducir ese JSON a un análisis en español, profesional y legible, sin inventar ni recalcular ningún número.
{missing_note}
Reglas estrictas
1. No inventes ni sobrescribas ningún valor numérico. Todos los números de tu respuesta —probabilidades, marcadores, intervalos— tienen que salir literalmente del JSON. Si un dato que te pido no está en el JSON, decilo explícitamente en vez de estimarlo o completarlo de memoria.
2. No redondees para que "suene mejor". Si el JSON dice p1: 0.42, se reporta 42%, no "cerca de la mitad" ni "favorito claro".
3. Nunca conviertas un intervalo de credibilidad en un número único. Si el JSON trae ci_lam_90 o ci_mu_90, esos rangos van completos en la respuesta. No elijas el punto medio y lo presentes como si fuera la única cifra.
4. No uses la palabra "seguro", "garantizado", "sin duda" ni equivalentes. El modelo trabaja con probabilidades porque el resultado tiene varianza irreducible (Poisson). Tu lenguaje tiene que reflejar eso, no ocultarlo.
5. Si p1, px y p2 están dentro de ~10 puntos porcentuales entre sí, decilo explícitamente como partido parejo. No fuerces un favorito claro donde el modelo no lo ve.
6. Distinguí siempre "lo que dice el modelo" de "contexto cualitativo adicional que vos agregues." Si querés mencionar una lesión reciente, un cambio de DT, o algo que no está en el JSON, marcalo aparte como nota cualitativa — nunca lo mezcles con los números del modelo como si el modelo ya lo hubiera considerado.
7. No completes con "sabiduría futbolística general" ningún campo que falte. Si el JSON no trae top_5_scores, no propongas vos un marcador probable a ojo.
 
Estructura de salida
1. Resumen ejecutivo (2-3 líneas): Selección A vs. Selección B. Un párrafo corto con el resultado más probable según p1/px/p2 y el marcador de mayor probabilidad en top_5_scores, citando los porcentajes exactos del JSON.
2. Probabilidades de resultado: Tabla o lista con p1, px, p2 convertidos a porcentaje. Mencioná explícitamente si el modelo ve el partido como parejo o con favorito claro, según la regla 5.
3. Marcadores más probables: Los top_5_scores del JSON, cada uno con su probabilidad. Aclarar que son los marcadores de mayor probabilidad individual, no que alguno de ellos sea "el resultado esperado".
4. Tasas de gol esperadas e incertidumbre: lam, mu (goles esperados por selección) junto con lam_std, mu_std y los intervalos ci_lam_90, ci_mu_90 completos. Explicar en una frase qué significa el intervalo.
5. Parámetros del modelo (transparencia): elo_a, elo_b, rho, half_life_days. Una línea explicando qué es cada uno en términos simples.
6. Notas cualitativas (opcional, solo si las agrego yo aparte del JSON): Cualquier factor que el modelo no capture explícitamente, marcado con claridad como aporte externo al modelo.
7. Límite del análisis (una frase de cierre, siempre presente): Recordar brevemente que el modelo da probabilidades calibradas, no certezas, porque el resultado de un partido tiene varianza irreducible.
 
Datos del partido
Selección A: {team_a}
Selección B: {team_b}
 
JSON de predict():
{json.dumps(prediction_json, indent=2)}
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            text = response.text
            # Nota automática visible en el output final: si faltaron campos,
            # esto queda registrado en el propio texto que el usuario lee,
            # no solo en el prompt interno. Esto NO garantiza que el LLM haya
            # obedecido la instrucción de no inventar — eso depende de que el
            # modelo de lenguaje respete la regla 7 — pero al menos deja
            # constancia visible de que faltaba algo, para que se pueda
            # revisar manualmente si el texto de arriba igual menciona esos
            # datos como si estuvieran presentes.
            if missing:
                text += (
                    f"\n\n[Nota automática del sistema: los campos {', '.join(missing)} "
                    "no estaban presentes en el JSON de predict(). Si el texto de arriba "
                    "menciona esos datos igual, revisar manualmente — puede ser una "
                    "estimación no autorizada del modelo de lenguaje.]"
                )
            return text
        except Exception as e:
            return f"[NLP Agent ERROR] Fallo al generar explicación: {e}"
 
 
if __name__ == "__main__":
    # Test script
    async def test():
        agent = QualititaveWebAnalyst()
        news = await agent.search_web_news("Argentina", "France")
        print("News:", news)
        mods = agent.extract_modifiers(news)
        print("Modifiers:", mods)
        
    asyncio.run(test())
