import sys
import numpy as np

sys.path.append(r"c:\Users\Usuario\Desktop\fuol")
from unified_engine import run_prediction

# Mocking 100 recent matches for France (very strong)
fra_matches = [
    {"date": "2026-06-28", "gf": 3, "gc": 0, "res": "W", "opponent": "SUECIA", "xg_for": 2.8, "xg_against": 0.4},
    {"date": "2026-06-24", "gf": 2, "gc": 0, "res": "W", "opponent": "DEFAULT", "xg_for": 2.1, "xg_against": 0.5},
    {"date": "2026-06-19", "gf": 4, "gc": 1, "res": "W", "opponent": "DEFAULT", "xg_for": 3.2, "xg_against": 0.8},
    {"date": "2026-06-14", "gf": 1, "gc": 0, "res": "W", "opponent": "DEFAULT", "xg_for": 1.5, "xg_against": 0.2},
] * 25

# Mocking 100 recent matches for Paraguay (resilient but less dominant)
par_matches = [
    {"date": "2026-06-29", "gf": 1, "gc": 1, "res": "D", "opponent": "ALEMANIA", "xg_for": 0.7, "xg_against": 2.1}, # Won on pens
    {"date": "2026-06-25", "gf": 0, "gc": 0, "res": "D", "opponent": "DEFAULT", "xg_for": 0.4, "xg_against": 1.0},
    {"date": "2026-06-20", "gf": 1, "gc": 2, "res": "L", "opponent": "DEFAULT", "xg_for": 0.9, "xg_against": 1.8},
    {"date": "2026-06-15", "gf": 2, "gc": 1, "res": "W", "opponent": "DEFAULT", "xg_for": 1.5, "xg_against": 1.2},
] * 25

result = run_prediction(
    team_a="FRANCIA", 
    team_b="PARAGUAY", 
    matches_a=fra_matches, 
    matches_b=par_matches, 
    venue='N',
    run_backtest=False,
    verbose=False
)

print("=== PREDICCION ===")
matrix = result['score_matrix']
flat = matrix.flatten()
indices = np.argsort(flat)[::-1][:15] # top 15

total_prob = 0
for idx in indices:
    i, j = np.unravel_index(idx, matrix.shape)
    prob = flat[idx] * 100
    total_prob += prob
    print(f"FRA {i} - {j} PAR: {prob:.2f}%")
print(f"Probabilidades sumadas del top 15: {total_prob:.2f}%")
