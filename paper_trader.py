import os
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument

class PaperTrader:
    def __init__(self, db_uri="mongodb://localhost:27017", initial_bankroll=10000.0):
        self.client = AsyncIOMotorClient(db_uri)
        self.db = self.client.fuol_lake
        self.ledger = self.db.paper_trading_ledger
        self.initial_bankroll = initial_bankroll

    def calculate_kelly_multi_outcome(self, model_probs: list[float], market_odds: list[float], brier_score=0.15, fractional_scale=0.10):
        """
        Calcula el tamaño óptimo de la posición para mercados mutuamente excluyentes (1X2).
        Retorna las fracciones a apostar para [Home, Draw, Away].
        """
        fractions = [0.0, 0.0, 0.0]
        
        # Encontrar la selección con mayor alpha/EV
        best_idx = -1
        max_ev = 0.0
        
        for i in range(3):
            if market_odds[i] <= 0:
                continue
            implied = 1.0 / market_odds[i]
            ev = (model_probs[i] * market_odds[i]) - 1.0
            
            if ev > max_ev:
                max_ev = ev
                best_idx = i
                
        if best_idx == -1:
            return fractions # No edge found
            
        # Si apostamos solo a una selección (la mejor):
        # f = EV / (Odds - 1)
        # Esto es Kelly para apuestas independientes, pero al ser mutuamente excluyentes
        # y si solo elegimos una, la fórmula converge a la misma.
        b = market_odds[best_idx] - 1.0
        p = model_probs[best_idx]
        kelly_f = ((b * p) - (1 - p)) / b if b > 0 else 0.0
        
        confidence_multiplier = max(0.5, 1.0 - (brier_score * 2))
        fractional_kelly = max(0.0, kelly_f * fractional_scale * confidence_multiplier)
        
        fractions[best_idx] = fractional_kelly
        return fractions

    async def get_current_bankroll(self):
        """Calcula el bankroll disponible en O(1) desde el estado global."""
        wallet = await self.db.wallet_state.find_one({"_id": "main_wallet"})
        if not wallet:
            return self.initial_bankroll
        return wallet.get("balance", self.initial_bankroll)

    async def place_bet(self, match_id: str, selection: str, engine_prob: float, market_odds: float, brier_score: float = 0.15):
        """
        Calcula si hay Alpha y ejecuta la orden si el Kelly > 0.
        """
        market_prob = 1.0 / market_odds
        alpha = engine_prob - market_prob
        
        if alpha <= 0:
            return {"success": False, "reason": "Alpha Negativo o Cero", "alpha": alpha}
            
        # Backward compatibility with single-outcome API calls in tests
        fractions = self.calculate_kelly_multi_outcome([engine_prob, 0, 0], [market_odds, 0, 0], brier_score)
        fraction = fractions[0]
        if fraction <= 0:
            return {"success": False, "reason": "Kelly <= 0", "alpha": alpha}
            
        current_bankroll = await self.get_current_bankroll()
        stake = current_bankroll * fraction
        
        # Guardrail: No apostar menos de $1 ni más de $1000 en un solo trade
        stake = max(1.0, min(1000.0, stake))
        
        # 1. Asegurar que la billetera exista
        await self.db.wallet_state.update_one(
            {"_id": "main_wallet"},
            {"$setOnInsert": {"balance": self.initial_bankroll}},
            upsert=True
        )
        
        # 2. Descontar atómicamente si hay fondos (Evita Race Condition)
        updated_wallet = await self.db.wallet_state.find_one_and_update(
            {"_id": "main_wallet", "balance": {"$gte": stake}},
            {"$inc": {"balance": -stake}},
            return_document=ReturnDocument.AFTER
        )
        
        if not updated_wallet:
            return {"success": False, "reason": "Fondos insuficientes o Race Condition bloqueada", "alpha": alpha}
        
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
            "clv": None, # Para ser llenado post-partido por clv_tracker
            "status": "PENDING"
        }
        
        await self.ledger.insert_one(trade_doc)
        return {"success": True, "trade": {k: v for k, v in trade_doc.items() if k != "_id"}}

    async def settle_bet(self, match_id: str, selection_won: str):
        """
        Liquida todas las apuestas (PENDING) asociadas a un match_id dado el resultado real usando transacciones ACID.
        """
        async with await self.client.start_session() as session:
            async with session.start_transaction():
                try:
                    cursor = self.ledger.find({"match_id": match_id, "status": "PENDING"}, session=session)
                    settled_count = 0
                    async for trade in cursor:
                        stake = trade["stake"]
                        odds = trade["market_odds"]
                        selection = trade["selection"]
                        
                        if str(selection) == str(selection_won):
                            payout = stake * odds
                            status = "WON"
                            await self.db.wallet_state.update_one(
                                {"_id": "main_wallet"}, {"$inc": {"balance": payout}}, session=session
                            )
                        else:
                            payout = 0.0
                            status = "LOST"
                            
                        await self.ledger.update_one(
                            {"_id": trade["_id"]},
                            {"$set": {"status": status, "payout": payout, 
                                      "settled_at": datetime.datetime.now().isoformat()}},
                            session=session
                        )
                        settled_count += 1
                    
                    return {"success": True, "match_id": match_id, "settled_count": settled_count}
                except Exception as e:
                    return {"success": False, "error": str(e)}

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
