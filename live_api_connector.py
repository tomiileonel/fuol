import asyncio
import websockets
import json
import time
import logging
from production_logger import ProductionLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("LiveAPI")

class LiveAPIConnector:
    def __init__(self, engine, ws_url="ws://localhost:5001", match_id="UNKNOWN", max_messages=None):
        self.engine = engine
        self.ws_url = ws_url
        self.match_id = match_id
        self.max_messages = max_messages
        self.logger_db = ProductionLogger()
        self.is_active = True
        
        # Parámetros de Riesgo
        self.LATENCY_LIMIT_MS = 2000.0
        self.ANOMALY_LIMIT = 0.20 # 20% absolute difference (3-sigma)
        self.processed_messages = 0

    def normalize_data(self, raw_data):
        odds = raw_data.get("odds", {})
        o_a, o_d, o_b = odds.get("team_a", 0), odds.get("draw", 0), odds.get("team_b", 0)
        
        # Circuit Breaker 1: Validación de Integridad
        if o_a <= 0 or o_b <= 0 or o_d <= 0:
            raise ValueError(f"Payload corrupto: Cuotas negativas o cero. {odds}")
            
        p_a, p_b, p_d = 1/o_a, 1/o_b, 1/o_d
        overround = p_a + p_b + p_d
        
        return {
            "team_a": p_a / overround,
            "team_b": p_b / overround,
            "draw": p_d / overround
        }

    async def connect_and_listen(self):
        logger.info(f"Conectando a stream WebSocket en {self.ws_url} ...")
        try:
            # Obtener probabilidades base del motor estadístico
            engine_probs = self.engine.predict()
            engine_prob_a = engine_probs['p1']
            
            # Entropía basada en Brier Score histórico (placeholder, se enlazará luego)
            brier_score = 0.15 
            
        except Exception as e:
            logger.error(f"Error obteniendo prior del motor: {e}")
            engine_prob_a = 0.50
            brier_score = 0.20
            
        async with websockets.connect(self.ws_url) as ws:
            logger.info("Conexión WebSocket establecida. Escuchando ticks...")
            while self.is_active:
                try:
                    message = await ws.recv()
                    raw_data = json.loads(message)
                    
                    receive_time = time.time()
                    sent_time = raw_data.get("timestamp", receive_time)
                    latency = (receive_time - sent_time) * 1000
                    
                    match_id = raw_data.get("match_id", self.match_id)
                    tick = raw_data.get("tick", 0)
                    
                    if latency > self.LATENCY_LIMIT_MS:
                        logger.warning(f"[Tick {tick}] STALE DATA DESCARTADO. Latencia {latency:.0f}ms > {self.LATENCY_LIMIT_MS}ms")
                        self.logger_db.log_signal(match_id, engine_prob_a, 0, 0, latency, "STALE_DATA")
                        continue
                        
                    market_probs = self.normalize_data(raw_data)
                    market_prob_a = market_probs["team_a"]
                    
                    alpha = engine_prob_a - market_prob_a
                    
                    # Criterio de Kelly Fraccional
                    b = (1.0 / market_prob_a) - 1.0 if market_prob_a > 0 else 0
                    p = engine_prob_a
                    q = 1.0 - p
                    kelly_f = ((b * p) - q) / b if b > 0 else 0
                    
                    # Ajuste dinámico de fracción según el Brier Score (Entropía)
                    # A menor Brier Score (mejor predicción histórica), mayor confianza
                    # Escala base: 10% Kelly. Si Brier es alto (ej > 0.25), baja a 5%.
                    confidence_multiplier = max(0.5, 1.0 - (brier_score * 2))
                    fractional_kelly = max(0, kelly_f * 0.10 * confidence_multiplier)
                    
                    if abs(alpha) > self.ANOMALY_LIMIT:
                        logger.error(f"[Tick {tick}] CISNE NEGRO DETECTADO. Alpha |{alpha:.2f}| > {self.ANOMALY_LIMIT}. Pausando motor.")
                        self.logger_db.log_signal(match_id, engine_prob_a, market_prob_a, alpha, latency, "PAUSED_ANOMALY")
                        self.is_active = False
                        break
                        
                    if alpha > 0.03:
                        logger.info(f"[Tick {tick}] VALUE BET: Alpha +{alpha*100:.2f}% | Kelly Rec: {fractional_kelly*100:.2f}% | Lat: {latency:.0f}ms")
                    else:
                        logger.info(f"[Tick {tick}] NO ACTION: Alpha {alpha*100:.2f}% | Lat: {latency:.0f}ms")
                        
                    self.logger_db.log_signal(match_id, engine_prob_a, market_prob_a, alpha, latency, "OK")
                    self.processed_messages += 1
                    if self.max_messages is not None and self.processed_messages >= self.max_messages:
                        logger.info(f"Se alcanzó el límite de mensajes ({self.max_messages}). Finalizando conexión.")
                        self.is_active = False
                        break
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.error("Desconectado del servidor WebSocket.")
                    break
                except Exception as e:
                    logger.error(f"Error procesando mensaje: {e}")

    def run(self):
        asyncio.run(self.connect_and_listen())
