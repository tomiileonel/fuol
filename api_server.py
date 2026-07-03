import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn

app = FastAPI(title="FUOL 360 API")

# Setup MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.fuol_lake
collection = db.predictions

# Serve static files for frontend
# Ensure the 'static' directory exists where api_server.py is running
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

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
