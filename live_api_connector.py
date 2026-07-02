import time
import requests
import logging
from production_logger import ProductionLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("LiveAPI")

class LiveAPIConnector:
    def __init__(self, engine, match_id, base_url="http://localhost:5000/v1/odds"):
        self.engine = engine
        self.match_id = match_id
        self.base_url = base_url
        self.logger_db = ProductionLogger()
        self.is_active = True
        
        # Parámetros de Riesgo
        self.LATENCY_LIMIT_MS = 2000.0
        self.ANOMALY_LIMIT = 0.20 # 20% absolute difference (3-sigma)

    def fetch_market_odds(self):
        start_time = time.time()
        try:
            # Conexión al servidor (o al Mock durante la fase de test)
            response = requests.get(f"{self.base_url}/{self.match_id}", timeout=2.5)
            latency = (time.time() - start_time) * 1000 # ms
            
            if response.status_code == 200:
                return response.json(), latency
            else:
                return None, latency
        except Exception as e:
            logger.error(f"Error de conexión HTTP: {e}")
            return None, (time.time() - start_time) * 1000

    def normalize_data(self, raw_data):
        odds = raw_data.get("odds", {})
        o_a, o_b, o_d = odds.get("team_a", 0), odds.get("team_b", 0), odds.get("draw", 0)
        
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

    def run_live_loop(self, iterations=10, interval=2):
        logger.info(f"Iniciando Live Fire Mode para {self.match_id} (Umbral Latencia: {self.LATENCY_LIMIT_MS}ms, Anomalía: {self.ANOMALY_LIMIT*100}%)")
        
        try:
            engine_probs, _ = self.engine.run_pipeline()
            engine_prob_a = engine_probs["1X2"][0] 
        except Exception as e:
            engine_prob_a = 0.52 
            
        for i in range(iterations):
            if not self.is_active:
                logger.error("Sistema PAUSADO por Stop-Loss Algorítmico.")
                break
                
            try:
                # 1. Fetch
                raw_data, latency = self.fetch_market_odds()
                
                if raw_data is None:
                     logger.error(f"[Tick {i}] Error de fetch o timeout. Latencia {latency:.0f}ms")
                     time.sleep(interval)
                     continue
                
                # 2. Circuit Breaker: Latency
                if latency > self.LATENCY_LIMIT_MS:
                    logger.warning(f"[Tick {i}] STALE DATA DESCARTADO. Latencia {latency:.0f}ms > {self.LATENCY_LIMIT_MS}ms")
                    self.logger_db.log_signal(self.match_id, engine_prob_a, 0, 0, latency, "STALE_DATA")
                    time.sleep(interval)
                    continue
                    
                # 3. Normalize & Circuit Breaker: Integrity
                market_probs = self.normalize_data(raw_data)
                market_prob_a = market_probs["team_a"]
                
                # 4. Calculate Alpha
                alpha = engine_prob_a - market_prob_a
                
                # 5. Circuit Breaker: Anomaly Stop-Loss
                if abs(alpha) > self.ANOMALY_LIMIT:
                    logger.error(f"[Tick {i}] CISNE NEGRO DETECTADO. Alpha |{alpha:.2f}| > {self.ANOMALY_LIMIT}. Pausando motor.")
                    self.logger_db.log_signal(self.match_id, engine_prob_a, market_prob_a, alpha, latency, "PAUSED_ANOMALY")
                    self.is_active = False
                    continue
                
                # 6. Ejecución Normal
                if alpha > 0.03: 
                    logger.info(f"[Tick {i}] VALUE BET: Alpha +{alpha*100:.2f}%. (Engine: {engine_prob_a*100:.1f}%, Market: {market_prob_a*100:.1f}%) [Lat: {latency:.0f}ms]")
                else:
                    logger.info(f"[Tick {i}] NO ACTION: Alpha {alpha*100:.2f}% [Lat: {latency:.0f}ms]")
                    
                self.logger_db.log_signal(self.match_id, engine_prob_a, market_prob_a, alpha, latency, "OK")
                
            except ValueError as e:
                logger.error(f"[Tick {i}] INTEGRIDAD FALLIDA: {e}")
                self.logger_db.log_signal(self.match_id, engine_prob_a, 0, 0, latency, "CORRUPT_PAYLOAD")
            except Exception as e:
                logger.error(f"[Tick {i}] ERROR INESPERADO: {e}")
                
            time.sleep(interval)
