import os
import datetime
from motor.motor_asyncio import AsyncIOMotorClient

class PaperTrader:
    def __init__(self, db_uri="mongodb://localhost:27017", initial_bankroll=10000.0):
        self.client = AsyncIOMotorClient(db_uri)
        self.db = self.client.fuol_lake
        self.ledger = self.db.paper_trading_ledger
        self.initial_bankroll = initial_bankroll

    def calculate_fractional_kelly(self, market_prob, engine_prob, brier_score=0.15, fractional_scale=0.10):
        """
        Calcula el tamaño óptimo de la posición usando Criterio de Kelly (modulado por Brier).
        Retorna la fracción del bankroll a apostar (0 a 1).
        """
        if market_prob <= 0 or engine_prob <= 0:
            return 0.0
            
        b = (1.0 / market_prob) - 1.0
        p = engine_prob
        q = 1.0 - p
        
        kelly_f = ((b * p) - q) / b if b > 0 else 0.0
        
        # Penalizamos la confianza si el Brier Score es alto (históricamente fallamos)
        confidence_multiplier = max(0.5, 1.0 - (brier_score * 2))
        
        # Fracción defensiva (Fractional Kelly)
        fractional_kelly = max(0.0, kelly_f * fractional_scale * confidence_multiplier)
        
        return fractional_kelly

    async def get_current_bankroll(self):
        """Calcula el bankroll disponible restando el PnL y stakes pendientes."""
        cursor = self.ledger.find({})
        current = self.initial_bankroll
        async for trade in cursor:
            if trade.get("status") == "PENDING":
                current -= trade.get("stake", 0)
            elif trade.get("status") == "WON":
                # Recuperas el stake + ganancia neta
                profit = trade.get("stake", 0) * (trade.get("market_odds", 1.0) - 1.0)
                current += profit
            elif trade.get("status") == "LOST":
                current -= trade.get("stake", 0)
        return current

    async def place_bet(self, match_id: str, selection: str, engine_prob: float, market_odds: float, brier_score: float = 0.15):
        """
        Calcula si hay Alpha y ejecuta la orden si el Kelly > 0.
        """
        market_prob = 1.0 / market_odds
        alpha = engine_prob - market_prob
        
        if alpha <= 0:
            return {"success": False, "reason": "Alpha Negativo o Cero", "alpha": alpha}
            
        fraction = self.calculate_fractional_kelly(market_prob, engine_prob, brier_score)
        if fraction <= 0:
            return {"success": False, "reason": "Kelly <= 0", "alpha": alpha}
            
        current_bankroll = await self.get_current_bankroll()
        stake = current_bankroll * fraction
        
        # Guardrail: No apostar menos de $1 ni más de $1000 en un solo trade
        stake = max(1.0, min(1000.0, stake))
        
        trade_doc = {
            "timestamp": datetime.datetime.now().isoformat(),
            "match_id": match_id,
            "action": "OPEN_POSITION",
            "selection": selection,
            "model_alpha": alpha,
            "engine_prob": engine_prob,
            "market_prob": market_prob,
            "market_odds": market_odds,
            "kelly_fraction_used": fraction,
            "stake": stake,
            "status": "PENDING"
        }
        
        await self.ledger.insert_one(trade_doc)
        return {"success": True, "trade": {k: v for k, v in trade_doc.items() if k != "_id"}}

    async def get_portfolio_summary(self):
        """Genera el resumen PnL para la UI."""
        cursor = self.ledger.find({}).sort("timestamp", -1)
        history = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            history.append(doc)
            
        bankroll = await self.get_current_bankroll()
        roi = ((bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        
        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": bankroll,
            "roi_percent": roi,
            "history": history
        }

if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("[PaperTrader] Ejecutando Tests Automatizados...")
        trader = PaperTrader(initial_bankroll=10000.0)
        
        # Test 1: Alpha Negativo
        res1 = await trader.place_bet("Test_Match_1", "1", engine_prob=0.40, market_odds=2.0)
        assert res1["success"] == False
        print("[OK] Test 1 Passed: Bloqueado por Alpha Negativo")
        
        # Test 2: Alpha Positivo
        res2 = await trader.place_bet("Test_Match_2", "1", engine_prob=0.60, market_odds=2.0)
        assert res2["success"] == True
        print(f"[OK] Test 2 Passed: Orden creada con Stake ${res2['trade']['stake']:.2f}")
        
        # Limpieza de la DB de tests
        await trader.ledger.delete_many({"match_id": {"$regex": "^Test_Match"}})
        
    asyncio.run(run_tests())
