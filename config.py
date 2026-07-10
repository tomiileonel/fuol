"""
config.py — Centralized Configuration and Hyperparameters for FUOL
"""

import os

# ---------------------------------------------------------------------------
# API & Infrastructure
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
MAX_KELLY_STAKE = float(os.getenv("MAX_KELLY_STAKE", 0.25))

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

# Decay & Rates
DEFAULT_HALF_LIFE = 365.0       # Exponential decay half-life in days
PRIOR_STRENGTH = 6.0            # "Imaginary games" of prior confidence

# Elo System
ELO_K_WC = 40.0                 # K-factor for World Cup
ELO_K_QUALIF = 30.0             # K-factor for Qualifiers/Continental
ELO_K_FRIENDLY = 20.0           # K-factor for Friendlies
ELO_HOME_ADV = 100.0            # Elo points home advantage
ELO_SCALE = 400.0               # Logistic scale for Elo

# Goals & Lambda
AVG_GOALS_HISTORICAL = 1.32     # Historical avg goals per team (2.64 total)
LAMBDA_SCALE = 0.23             # Elo-to-lambda multiplier (calibratable)

# Dixon-Coles
DEFAULT_RHO = -0.13             # Default dependence parameter

# Ensemble Weights
ENSEMBLE_WEIGHT_DC = 0.70
ENSEMBLE_WEIGHT_HAWKES = 0.20
ENSEMBLE_WEIGHT_HIER = 0.10
