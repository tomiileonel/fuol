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
import difflib

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

ALL_TEAMS = set(HIST_DF['home_team'].unique()) | set(HIST_DF['away_team'].unique())
TEAM_ALIASES = {
    "SUIZA": "Switzerland", 
    "ARGENTINA": "Argentina", 
    "BRASIL": "Brazil", 
    "INGLATERRA": "England", 
    "NORUEGA": "Norway",
    "ALEMANIA": "Germany",
    "ESPAÑA": "Spain",
    "FRANCIA": "France",
    "ITALIA": "Italy"
}

def normalize_team_name(team_input: str) -> str:
    """Traduce y hace fuzzy matching para encontrar el equipo en la DB."""
    team_upper = team_input.upper().strip()
    
    # 1. Si es un alias conocido
    if team_upper in TEAM_ALIASES:
        mapped = TEAM_ALIASES[team_upper]
        if mapped in ALL_TEAMS:
            return mapped
    
    # 2. Búsqueda exacta (case-insensitive)
    for team in ALL_TEAMS:
        if team.upper() == team_upper:
            return team
            
    # 3. Búsqueda difusa
    matches = difflib.get_close_matches(team_input, list(ALL_TEAMS), n=1, cutoff=0.85)
    if matches:
        return matches[0]
        
    return team_input

print("Histórico cargado. Servidor listo.")

class PredictionRequest(BaseModel):
    team_a: str
    team_b: str

@app.post("/api/predict")
async def get_prediction(req: PredictionRequest):
    # Normalizar nombres antes de buscar en el motor
    team_a_norm = normalize_team_name(req.team_a)
    team_b_norm = normalize_team_name(req.team_b)
    
    matches_a = pipeline.get_team_history(HIST_DF, team_a_norm)
    matches_b = pipeline.get_team_history(HIST_DF, team_b_norm)
    
    if not matches_a or len(matches_a) < 5:
        raise HTTPException(status_code=404, detail=f"Equipo '{req.team_a}' no encontrado en el histórico (Buscado como: '{team_a_norm}')")
    if not matches_b or len(matches_b) < 5:
        raise HTTPException(status_code=404, detail=f"Equipo '{req.team_b}' no encontrado en el histórico (Buscado como: '{team_b_norm}')")

    try:
        engine = UnifiedEngine(
            team_a=team_a_norm, team_b=team_b_norm,
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
