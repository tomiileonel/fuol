import sys
import os

sys.path.append(r"c:\Users\Usuario\Desktop\fuol")

from unified_engine import run_prediction

can_matches = [
    {"date": "2026-06-25", "gf": 1, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2026-06-20", "gf": 2, "gc": 1, "res": "W", "opponent": "DEFAULT"},
    {"date": "2026-06-15", "gf": 1, "gc": 1, "res": "D", "opponent": "DEFAULT"},
    {"date": "2026-06-05", "gf": 0, "gc": 2, "res": "L", "opponent": "FRANCIA"},
    {"date": "2026-05-30", "gf": 2, "gc": 0, "res": "W", "opponent": "EEUU"},
    {"date": "2025-11-20", "gf": 3, "gc": 1, "res": "W", "opponent": "DEFAULT"},
    {"date": "2025-10-15", "gf": 1, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2025-09-10", "gf": 0, "gc": 1, "res": "L", "opponent": "DEFAULT"},
    {"date": "2025-06-20", "gf": 2, "gc": 2, "res": "D", "opponent": "DEFAULT"},
    {"date": "2025-03-25", "gf": 4, "gc": 0, "res": "W", "opponent": "DEFAULT"},
] * 10

mar_matches = [
    {"date": "2026-06-26", "gf": 2, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2026-06-21", "gf": 3, "gc": 1, "res": "W", "opponent": "DEFAULT"},
    {"date": "2026-06-16", "gf": 1, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2026-06-06", "gf": 2, "gc": 2, "res": "D", "opponent": "ESPAÑA"},
    {"date": "2026-05-25", "gf": 1, "gc": 1, "res": "D", "opponent": "DEFAULT"},
    {"date": "2025-11-22", "gf": 2, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2025-10-16", "gf": 0, "gc": 0, "res": "D", "opponent": "DEFAULT"},
    {"date": "2025-09-12", "gf": 1, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2025-06-22", "gf": 3, "gc": 0, "res": "W", "opponent": "DEFAULT"},
    {"date": "2025-03-27", "gf": 1, "gc": 2, "res": "L", "opponent": "DEFAULT"},
] * 10

result = run_prediction(
    team_a="MARRUECOS", 
    team_b="CANADÁ", 
    matches_a=mar_matches, 
    matches_b=can_matches, 
    venue='N',
    run_backtest=False,
    verbose=False
)

print("TOP 10 SCORES:")
# We'll calculate top 10 from the matrix directly just to be thorough
import numpy as np
matrix = result['score_matrix']
flat = matrix.flatten()
indices = np.argsort(flat)[::-1][:15] # top 15

total_prob = 0
for idx in indices:
    i, j = np.unravel_index(idx, matrix.shape)
    prob = flat[idx] * 100
    total_prob += prob
    print(f"MAR {i} - {j} CAN: {prob:.2f}%")
print(f"Sum of top 15 probs: {total_prob:.2f}%")
