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
    return await trader.get_portfolio_summary()

@app.post("/api/paper_trade/{match_id}")
async def place_paper_trade(match_id: str, req: TradeRequest):
    """Executes a virtual trade using the Kelly Criterion."""
    res = await trader.place_bet(
        match_id=match_id,
        selection=req.selection,
        engine_prob=req.engine_prob,
        market_odds=req.market_odds
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["reason"])
    return res

@app.get("/api/predictions/{match_id}")
async def get_prediction(match_id: str):
    """
    Fetch the master document for a specific match.
    """
    document = await collection.find_one({"match_id": match_id}, {"_id": 0})
    if not document:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return document

@app.get("/")
async def root():
    # Redirect root to static index
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
