import json
import numpy as np
from supreme_engine import SupremePredictionEngine
from performance_tracker import ModelTelemetry

senegal = [{"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1}, {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0}, {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0}]
belgium = [{"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0}, {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3}, {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1}]

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]])
}

telemetry = ModelTelemetry()
senegal_dynamic = telemetry.synchronize_knowledge_base("SENEGAL", senegal)
belgium_dynamic = telemetry.synchronize_knowledge_base("BÉLGICA", belgium)

engine = SupremePredictionEngine("SENEGAL", "BÉLGICA", senegal_dynamic, belgium_dynamic, FORMACIONES)
res, _ = engine.run_pipeline("4-3-3", "4-2-3-1")

matrix = res["Matriz Cruda"]
max_g = matrix.shape[0]

# Add the 1-0 score to the probabilities
# Since Senegal has 1 goal and Belgium has 0 goals, the final score is (i+1, j) where i,j are remainder goals
# Let's find the most likely final score and the updated 1X2 probabilities.
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

print("Lambda (Senegal):", res["Tasas Finales (Interferencia)"][0])
print("Mu (Belgica):", res["Tasas Finales (Interferencia)"][1])
print(f"Probabilidad de que gane Senegal (con 1-0): {p1*100:.2f}%")
print(f"Probabilidad de empate (con 1-0): {px*100:.2f}%")
print(f"Probabilidad de que gane Belgica (remontada): {p2*100:.2f}%")
print(f"Marcador Exacto mas probable al final del partido: {best_score[0]} - {best_score[1]}")
