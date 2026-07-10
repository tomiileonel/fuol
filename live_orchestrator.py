import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from unified_engine import UnifiedEngine
from paper_trader import PaperTrader
from data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LiveOrchestrator:
    def __init__(self, data_dir="live_data"):
        self.data_dir = data_dir
        self.fixtures_path = os.path.join(data_dir, "fixtures.csv")
        self.results_path = os.path.join(data_dir, "results.csv")
        self.processed_path = os.path.join(data_dir, "processed")
        
        os.makedirs(self.processed_path, exist_ok=True)
        self.trader = PaperTrader()
        self.pipeline = DataPipeline()
        self.history_df = None # Loaded lazily to reconstruct team history

    def _load_and_clean_csv(self, path):
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return pd.DataFrame()

    async def _process_fixtures(self):
        df = self._load_and_clean_csv(self.fixtures_path)
        if df.empty:
            return

        now = datetime.now()
        # Escanea partidos desde hace 2 horas (por si se retrasó el script) hasta dentro de 15 mins
        window_start = now - timedelta(hours=2)
        window_end = now + timedelta(minutes=15)
        
        mask = (df['date'] >= window_start) & (df['date'] <= window_end)
        matches_to_process = df[mask]
        
        if matches_to_process.empty:
            return

        if self.history_df is None:
            try:
                self.history_df, _ = self.pipeline.prepare_data()
            except Exception:
                self.history_df = pd.DataFrame()

        for _, match in matches_to_process.iterrows():
            match_id = f"{match['date'].strftime('%Y%m%d')}_{match['home_team']}_{match['away_team']}"
            
            # Check si ya se apostó en este partido
            existing = await self.trader.ledger.find_one({"match_id": match_id})
            if existing:
                continue
                
            try:
                h = match['home_team']
                a = match['away_team']
                
                h_hist = self.pipeline.get_team_history(self.history_df, h) if not self.history_df.empty else []
                a_hist = self.pipeline.get_team_history(self.history_df, a) if not self.history_df.empty else []
                
                engine = UnifiedEngine(h, a, h_hist, a_hist)
                pred = engine.predict()
                
                probs = [pred.get('p1', 0.33), pred.get('px', 0.33), pred.get('p2', 0.33)]
                
                # Des-vigging naive para el orquestador (o extraer directamente pin_H)
                odds_h = match.get('pin_H', match.get('odds_H', 0.0))
                odds_d = match.get('pin_D', match.get('odds_D', 0.0))
                odds_a = match.get('pin_A', match.get('odds_A', 0.0))
                
                odds = [odds_h, odds_d, odds_a]
                selections = ['1', 'X', '2']
                
                if any(pd.isna(o) or o <= 1.0 for o in odds):
                    continue

                best_idx = -1
                max_ev = 0.0
                for i in range(3):
                    ev = (probs[i] * odds[i]) - 1.0
                    if ev > max_ev:
                        max_ev = ev
                        best_idx = i
                
                if best_idx != -1 and max_ev > 0:
                    logger.info(f"[ALPHA DETECTADO] {match['home_team']} vs {match['away_team']} -> Bet {selections[best_idx]} (EV: {max_ev:.2%})")
                    res = await self.trader.place_bet(
                        match_id=match_id,
                        selection=selections[best_idx],
                        engine_prob=probs[best_idx],
                        market_odds=odds[best_idx]
                    )
                    if res.get('success', False):
                        logger.info(f"Paper Trade ejecutado. Stake: ${res['trade']['stake']:.2f}")
                    else:
                        logger.warning(f"Trade rechazado: {res.get('reason')}")
                else:
                    logger.info(f"No Edge positivo para {match_id}. No se apuesta.")

            except Exception as e:
                logger.error(f"Error inferencia {match_id}: {e}")

    async def _process_results(self):
        """Escanea results.csv y liquida las apuestas pendientes."""
        df = self._load_and_clean_csv(self.results_path)
        if df.empty:
            return

        # Asumimos columnas: date, home_team, away_team, home_score, away_score
        if 'home_score' not in df.columns or 'away_score' not in df.columns:
            return
            
        for _, match in df.iterrows():
            match_id = f"{match['date'].strftime('%Y%m%d')}_{match['home_team']}_{match['away_team']}"
            
            hs, as_ = int(match['home_score']), int(match['away_score'])
            if hs > as_:
                selection_won = '1'
            elif hs == as_:
                selection_won = 'X'
            else:
                selection_won = '2'
                
            # Ejecutar settlement atómico
            result = await self.trader.settle_bet(match_id, selection_won)
            if result.get('success', False):
                logger.info(f"Settlement completado para {match_id}. Trades liquidados: {result.get('settled_count', 0)}")
            else:
                logger.debug(f"Error liquidando {match_id}: {result.get('error')}")

        # Mover results.csv a procesados para no re-procesar
        try:
            new_path = os.path.join(self.processed_path, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.rename(self.results_path, new_path)
        except Exception as e:
            pass # Si falla el rename, no es crítico, el loop de DB previene doble liquidación.

    async def run(self):
        """Bucle principal asíncrono infinito."""
        logger.info("Iniciando bucle de monitoreo...")
        while True:
            try:
                await self._process_fixtures()
                await self._process_results()
            except Exception as e:
                logger.critical(f"Error fatal en el bucle principal: {e}")
            
            # Dormir 60 segundos entre ciclos para no consumir CPU
            await asyncio.sleep(60)

if __name__ == "__main__":
    orchestrator = LiveOrchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Orquestador detenido manualmente.")
