import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from unified_engine import UnifiedEngine
from performance_tracker import ModelTelemetry
import websockets

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def stream_to_frontend():
    telemetry = ModelTelemetry()
    aus = [{"gf": 1.2, "gc": 0.8}, {"gf": 1.5, "gc": 1.0}]
    egy = [{"gf": 1.0, "gc": 0.5}, {"gf": 0.8, "gc": 1.2}]
    aus_dyn = telemetry.synchronize_knowledge_base("AUSTRALIA", aus)
    egy_dyn = telemetry.synchronize_knowledge_base("EGIPTO", egy)
    
    engine = UnifiedEngine(
        team_a="AUSTRALIA", team_b="EGIPTO",
        matches_a=aus_dyn, matches_b=egy_dyn,
        venue="N"
    )
    
    engine_probs = engine.predict()
    engine_prob_a = engine_probs['p1']
    brier_score = 0.15 
    
    while True:
        try:
            async with websockets.connect("ws://localhost:5001") as ws:
                print("Connected to Mock Server from Backend")
                while True:
                    msg = await ws.recv()
                    raw = json.loads(msg)
                    tick = raw.get("tick", 0)
                    market_a = raw.get("odds", {}).get("team_a", 0)
                    
                    if market_a == 0:
                        continue
                        
                    p_a = 1/market_a
                    p_b = 1/raw["odds"]["team_b"]
                    p_d = 1/raw["odds"]["draw"]
                    overround = p_a + p_b + p_d
                    market_prob_a = p_a / overround
                    
                    alpha = engine_prob_a - market_prob_a
                    
                    b = (1.0 / market_prob_a) - 1.0 if market_prob_a > 0 else 0
                    p = engine_prob_a
                    q = 1.0 - p
                    kelly_f = ((b * p) - q) / b if b > 0 else 0
                    confidence_multiplier = max(0.5, 1.0 - (brier_score * 2))
                    fractional_kelly = max(0, kelly_f * 0.10 * confidence_multiplier)
                    
                    payload = {
                        "tick": tick,
                        "market_prob": market_prob_a,
                        "engine_prob": engine_prob_a,
                        "alpha": alpha,
                        "kelly": fractional_kelly,
                        "status": "DANGER" if abs(alpha) > 0.20 else "OK"
                    }
                    await manager.broadcast(json.dumps(payload))
        except Exception as e:
            print("Esperando conexión a servidor mock...", e)
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(stream_to_frontend())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
