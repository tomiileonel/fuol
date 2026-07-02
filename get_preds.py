import json
import numpy as np
from supreme_engine import SupremePredictionEngine
from performance_tracker import ModelTelemetry

usa = [{"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5}, {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0}, {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3}]
bosnia = [{"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5}, {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4}, {"gf":3,"gc":1}]

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]])
}

telemetry = ModelTelemetry()
usa_dynamic = telemetry.synchronize_knowledge_base("EEUU", usa)
bosnia_dynamic = telemetry.synchronize_knowledge_base("BOSNIA-HERZ.", bosnia)

engine2 = SupremePredictionEngine("EEUU", "BOSNIA-HERZ.", usa_dynamic, bosnia_dynamic, FORMACIONES)
res2, _ = engine2.run_pipeline("4-3-3", "3-4-2-1")

print("Lambda Final:", res2["Tasas Finales (Interferencia)"][0])
print("Mu Final:", res2["Tasas Finales (Interferencia)"][1])
print("Matrix shape:", res2["Matriz Cruda"].shape)
print("Max exact score:", res2["Marcador Exacto"])
