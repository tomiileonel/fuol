import json
import numpy as np
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

senegal = [{"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1}, {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0}, {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0}]
belgium = [{"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0}, {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3}, {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1}]

telemetry = ModelTelemetry()
senegal_dynamic = telemetry.synchronize_knowledge_base("SENEGAL", senegal)
belgium_dynamic = telemetry.synchronize_knowledge_base("BÉLGICA", belgium)

res = run_prediction("SENEGAL", "BÉLGICA", senegal_dynamic, belgium_dynamic, venue="N", verbose=False)

matrix = res["score_matrix"]
max_g = matrix.shape[0]

# Add the 1-0 score to the probabilities
# Since Senegal has 1 goal and Belgium has 0 goals, the final score is (i+1, j) where i,j are remainder goals
p1 = 0
px = 0
p2 = 0
best_prob = 0
best_score = (0, 0)

for i in range(max_g):
    for j in range(max_g):
        final_i = i + 1  # Senegal
        final_j = j + 0  # Belgium
        
        prob = matrix[i][j]
        if prob > best_prob:
            best_prob = prob
            best_score = (final_i, final_j)
            
        if final_i > final_j:
            p1 += prob
        elif final_i == final_j:
            px += prob
        else:
            p2 += prob

print("Lambda (Senegal):", res["lam"])
print("Mu (Belgica):", res["mu"])
print(f"Probabilidad de que gane Senegal (con 1-0): {p1*100:.2f}%")
print(f"Probabilidad de empate (con 1-0): {px*100:.2f}%")
print(f"Probabilidad de que gane Belgica (remontada): {p2*100:.2f}%")
print(f"Marcador Exacto mas probable al final del partido: {best_score[0]} - {best_score[1]}")
