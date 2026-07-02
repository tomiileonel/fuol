import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supreme_engine import SupremePredictionEngine, render_tactical_dashboard
from performance_tracker import ModelTelemetry

espana = [{"gf": 2.1, "gc": 0.7, "res": "W"}, {"gf": 1.8, "gc": 0.6, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}, {"gf": 1.5, "gc": 1.0, "res": "D"}]
austria = [{"gf": 1.2, "gc": 1.1, "res": "W"}, {"gf": 0.8, "gc": 1.5, "res": "L"}, {"gf": 1.0, "gc": 1.0, "res": "D"}, {"gf": 1.5, "gc": 1.2, "res": "W"}]

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]])
}

telemetry = ModelTelemetry()
esp_dynamic = telemetry.synchronize_knowledge_base("ESPAÑA", espana)
aus_dynamic = telemetry.synchronize_knowledge_base("AUSTRIA", austria)

engine = SupremePredictionEngine("ESPAÑA", "AUSTRIA", esp_dynamic, aus_dynamic, FORMACIONES)
res, momentum = engine.run_pipeline("4-3-3", "4-2-3-1")

p1, px, p2 = res["1X2"]
# Ajuste por eliminación directa (Knockout)
prob_esp_avanza = p1 + (px * 0.5)
prob_aus_avanza = p2 + (px * 0.5)

print(f"Probabilidad de que avance ESPAÑA: {prob_esp_avanza*100:.2f}%")
print(f"Probabilidad de que avance AUSTRIA: {prob_aus_avanza*100:.2f}%")

render_tactical_dashboard("ESPAÑA", "AUSTRIA", res, momentum)
plt.savefig('dashboard_esp_aus.png', dpi=150, bbox_inches='tight', facecolor='#08080C')
print("Dashboard guardado como dashboard_esp_aus.png")
