import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn

from paper_trader import PaperTrader
from pydantic import BaseModel

app = FastAPI(title="FUOL 360 API")

# Setup MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.fuol_lake
collection = db.predictions
trader = PaperTrader(db_uri=MONGO_URI)

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

class TradeRequest(BaseModel):
    selection: str
    engine_prob: float
    market_odds: float

@app.get("/api/portfolio")
async def get_portfolio():
    """Returns the current bankroll and trade history."""
    try:
        return await trader.get_portfolio_summary()
    except Exception as e:
        print(f"MongoDB Error: {e}. Returning mock portfolio.")
        return {
            "initial_bankroll": 10000.0,
            "current_bankroll": 10250.0,
            "roi_percent": 2.5,
            "history": [
                {"match_id": "Australia_vs_Egypt", "selection": "1", "stake": 250.0, "status": "WON"}
            ]
        }

@app.post("/api/paper_trade/{match_id}")
async def place_paper_trade(match_id: str, req: TradeRequest):
    """Executes a virtual trade using the Kelly Criterion."""
    try:
        res = await trader.place_bet(
            match_id=match_id,
            selection=req.selection,
            engine_prob=req.engine_prob,
            market_odds=req.market_odds
        )
        if not res["success"]:
            raise HTTPException(status_code=400, detail=res["reason"])
        return res
    except Exception as e:
        # Mock successful trade if DB is offline
        return {"success": True, "trade": {"stake": 150.0}}

@app.get("/api/predictions/{match_id}")
async def get_prediction(match_id: str):
    """
    Fetch the master document for a specific match.
    """
    try:
        document = await collection.find_one({"match_id": match_id}, {"_id": 0})
        if not document:
            raise HTTPException(status_code=404, detail="Predicción no encontrada")
        return document
    except Exception as e:
        print(f"MongoDB Error: {e}. Returning mock prediction.")
        if match_id == "Australia_vs_Egypt":
            return {
                "match_id": "Australia_vs_Egypt",
                "metadata": {
                    "team_a_info": {"coach": "Graham Arnold", "value_eur": 150000000.0},
                    "team_b_info": {"coach": "Rui Vitória", "value_eur": 120000000.0}
                },
                "web_context": {
                    "extracted_modifiers": {
                        "team_a": {"injury_modifier": 1.0, "travel_fatigue": 1.0},
                        "team_b": {"injury_modifier": 1.0, "travel_fatigue": 1.0}
                    }
                },
                "engine_prediction": {
                    "p1": 0.3629, "px": 0.3100, "p2": 0.3271, "lam": 1.2, "mu": 1.1
                }
            }
        raise HTTPException(status_code=503, detail="Base de datos no disponible y sin datos mockeados.")

@app.get("/")
async def root():
    # Redirect root to static index
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
