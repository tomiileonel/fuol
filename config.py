import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_KELLY_STAKE = float(os.getenv("MAX_KELLY_STAKE", "0.25"))
