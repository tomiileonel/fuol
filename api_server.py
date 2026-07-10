"""
api_server.py — Servidor API ligero para la Interfaz Visual FUOL.
Expone el motor unificado a través de un endpoint REST.
"""
import warnings
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os

warnings.filterwarnings('ignore')

from unified_engine import UnifiedEngine
from data_pipeline import DataPipeline

app = FastAPI(title="FUOL Quant Engine API")

# Permitir CORS para que el frontend (HTML/JS) pueda comunicarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar datos históricos en memoria UNA vez al arrancar el servidor
print("Cargando histórico global en memoria...")
pipeline = DataPipeline()
HIST_DF, _ = pipeline.prepare_data()
HIST_DF['date'] = pd.to_datetime(HIST_DF['date'])
HIST_DF = HIST_DF[HIST_DF['date'] < pd.Timestamp.now().normalize()]
print("Histórico cargado. Servidor listo.")

class PredictionRequest(BaseModel):
    team_a: str
    team_b: str

@app.post("/api/predict")
async def get_prediction(req: PredictionRequest):
    team_a = req.team_a.upper()
    team_b = req.team_b.upper()
    
    matches_a = pipeline.get_team_history(HIST_DF, team_a)
    matches_b = pipeline.get_team_history(HIST_DF, team_b)
    
    if not matches_a or len(matches_a) < 5:
        raise HTTPException(status_code=404, detail=f"Sin historial suficiente para {team_a}")
    if not matches_b or len(matches_b) < 5:
        raise HTTPException(status_code=404, detail=f"Sin historial suficiente para {team_b}")

    try:
        engine = UnifiedEngine(
            team_a=team_a, team_b=team_b,
            matches_a=matches_a, matches_b=matches_b,
            venue='N', half_life=365.0
        )
        pred = engine.predict()
        
        # Serializar matriz para JSON
        if 'score_matrix' in pred:
            pred['score_matrix'] = pred['score_matrix'].tolist()
            
        return pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir archivos estáticos (Frontend)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
