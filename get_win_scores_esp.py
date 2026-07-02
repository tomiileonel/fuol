import numpy as np
from supreme_engine import SupremePredictionEngine
from performance_tracker import ModelTelemetry

espana = [{"gf": 2.1, "gc": 0.7, "res": "W"}, {"gf": 1.8, "gc": 0.6, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}, {"gf": 1.5, "gc": 1.0, "res": "D"}]
austria = [{"gf": 1.2, "gc": 1.1, "res": "W"}, {"gf": 0.8, "gc": 1.5, "res": "L"}, {"gf": 1.0, "gc": 1.0, "res": "D"}, {"gf": 1.5, "gc": 1.2, "res": "W"}]

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]])
}

telemetry = ModelTelemetry()
esp_dynamic = telemetry.synchronize_knowledge_base("ESPAÑA", espana)
aus_dynamic = telemetry.synchronize_knowledge_base("AUSTRIA", austria)

engine = SupremePredictionEngine("ESPAÑA", "AUSTRIA", esp_dynamic, aus_dynamic, FORMACIONES)
res, _ = engine.run_pipeline("4-3-3", "4-2-3-1")

matrix = res["Matriz Cruda"]
max_g = matrix.shape[0]

win_probs = []
for i in range(max_g):
    for j in range(max_g):
        if i > j:
            win_probs.append((matrix[i][j], i, j))

win_probs.sort(key=lambda x: x[0], reverse=True)

print("Principales marcadores de victoria para España:")
for prob, esp_g, aus_g in win_probs[:5]:
    print(f"{esp_g} - {aus_g} (Probabilidad: {prob*100:.2f}%)")
