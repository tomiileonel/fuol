import json
import numpy as np
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

usa = [{"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5}, {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0}, {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3}]
bosnia = [{"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5}, {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4}, {"gf":3,"gc":1}]

telemetry = ModelTelemetry()
usa_dynamic = telemetry.synchronize_knowledge_base("EEUU", usa)
bosnia_dynamic = telemetry.synchronize_knowledge_base("BOSNIA-HERZ.", bosnia)

res2 = run_prediction("EEUU", "BOSNIA-HERZ.", usa_dynamic, bosnia_dynamic, venue="H", verbose=False)

print("Lambda Final:", res2["lam"])
print("Mu Final:", res2["mu"])
print("Matrix shape:", res2["score_matrix"].shape)
print("Max exact score:", res2["top_5_scores"][0]["score"])
