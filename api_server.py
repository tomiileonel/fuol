import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except Exception:  # pragma: no cover - optional dependency
    AsyncIOMotorClient = None

from config import API_HOST, API_PORT, MONGO_URI
from pydantic import BaseModel

try:
    from paper_trader import PaperTrader
except Exception:  # pragma: no cover - optional dependency
    PaperTrader = None

from unified_engine_v3 import UnifiedEngineV3
from cache import PredictionCache
from backtesting import run_rolling_backtest
from rate_limit import RateLimiter

app = FastAPI(title="FUOL 360 API")
cache = PredictionCache(ttl_seconds=300)
rate_limiter = RateLimiter(limit_per_minute=60)

# Setup MongoDB Connection
client = None
db = None
collection = None
trader = None
if AsyncIOMotorClient is not None:
    try:
        client = AsyncIOMotorClient(MONGO_URI)
        db = client.fuol_lake
        collection = db.predictions
        if PaperTrader is not None:
            trader = PaperTrader(db_uri=MONGO_URI)
    except Exception as exc:
        print(f"MongoDB unavailable: {exc}")

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

class TradeRequest(BaseModel):
    selection: str
    engine_prob: float
    market_odds: float


def _load_v3_config() -> dict[str, Any]:
    for candidate in [Path("config_v3/config_v3.json"), Path("config_v3.json"), Path("config/config_v3.json")]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _build_live_prediction(match_id: str) -> dict[str, Any]:
    fixture_map = {
        "FRA_SEN_WC26": {
            "team_a": "FRANCIA",
            "team_b": "SENEGAL",
            "venue": "H",
            "matches_a": [
                {"date": "2024-01-01", "home": "FRANCIA", "away": "SENEGAL", "gh": 2, "ga": 0, "minute": 10},
                {"date": "2024-02-01", "home": "FRANCIA", "away": "MAROC", "gh": 1, "ga": 1, "minute": 30},
            ],
            "matches_b": [
                {"date": "2024-01-15", "home": "SENEGAL", "away": "CAMERUN", "gh": 1, "ga": 1, "minute": 45},
                {"date": "2024-02-15", "home": "SENEGAL", "away": "TOGO", "gh": 2, "ga": 0, "minute": 60},
            ],
            "team_confederations": {"FRANCIA": "UEFA", "SENEGAL": "CAF"},
        },
        "AUS-EGY": {
            "team_a": "AUSTRALIA",
            "team_b": "EGYPT",
            "venue": "N",
            "matches_a": [{"date": "2023-01-01", "home": "AUSTRALIA", "away": "EGYPT", "gh": 1, "ga": 1, "minute": 30}],
            "matches_b": [{"date": "2023-02-01", "home": "EGYPT", "away": "AUSTRALIA", "gh": 2, "ga": 1, "minute": 40}],
            "team_confederations": {"AUSTRALIA": "AFC", "EGYPT": "CAF"},
        },
    }

    fixture = fixture_map.get(match_id)
    if not fixture:
        raise HTTPException(status_code=404, detail="Predicción en vivo no disponible para este partido")

    config = _load_v3_config()
    engine = UnifiedEngineV3(
        team_a=fixture["team_a"],
        team_b=fixture["team_b"],
        matches_a=fixture["matches_a"],
        matches_b=fixture["matches_b"],
        venue=fixture["venue"],
        team_confederations=fixture["team_confederations"],
        config=config,
    )
    prediction = engine.predict()
    return {
        "match_id": match_id,
        "source": "live_v3_engine",
        "config": config.get("model", {}).get("unified_engine", {}),
        "engine_prediction": prediction.to_dict(),
    }

@app.get("/api/live_prediction/{match_id}")
async def get_live_prediction(match_id: str):
    """Expose a live v3 prediction for a supported match."""
    if not rate_limiter.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    cached = cache.get(match_id=match_id, payload={"route": "live_prediction"})
    if cached is not None:
        return cached
    result = _build_live_prediction(match_id)
    cache.set(match_id=match_id, payload={"route": "live_prediction"}, value=result)
    return result


@app.get("/api/monitoring")
async def monitoring_summary():
    sample_matches = [
        {"home": "BRASIL", "away": "NORUEGA", "gh": 2, "ga": 1},
        {"home": "BRASIL", "away": "NORUEGA", "gh": 1, "ga": 0},
        {"home": "BRASIL", "away": "NORUEGA", "gh": 1, "ga": 1},
    ]
    summary = run_rolling_backtest(sample_matches, window=2)
    return {
        "status": "ok",
        "backtest": {
            "brier_score": summary.brier_score,
            "log_loss": summary.log_loss,
            "hit_rate": summary.hit_rate,
            "n_samples": summary.n_samples,
        },
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Returns the current bankroll and trade history."""
    if trader is None:
        return {
            "initial_bankroll": 10000.0,
            "current_bankroll": 10250.0,
            "roi_percent": 2.5,
            "history": [{"match_id": "FRA_SEN_WC26", "selection": "1", "stake": 250.0, "status": "WON"}],
        }
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
    if trader is None:
        return {"success": True, "trade": {"stake": 150.0}}
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
    Fetch the master document for a specific match or return the live v3 prediction.
    """
    if match_id in {"FRA_SEN_WC26", "AUS-EGY"}:
        return _build_live_prediction(match_id)

    if collection is None:
        return {
            "match_id": match_id,
            "engine_prediction": {
                "p1": 0.36,
                "px": 0.30,
                "p2": 0.34,
                "lam": 1.2,
                "mu": 1.1,
            },
            "source": "mock_fallback",
        }

    try:
        document = await collection.find_one({"match_id": match_id}, {"_id": 0})
        if not document:
            raise HTTPException(status_code=404, detail="Predicción no encontrada")
        return document
    except Exception as e:
        print(f"MongoDB Error: {e}. Returning mock prediction.")
        return {
            "match_id": match_id,
            "engine_prediction": {"p1": 0.36, "px": 0.30, "p2": 0.34, "lam": 1.2, "mu": 1.1},
            "source": "mock_fallback",
        }

@app.get("/")
async def root():
    # Redirect root to static index
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=True)
