import json
import numpy as np
import datetime
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

australia_raw = [
    {"gf": 1.2, "gc": 0.8, "xg_for": 1.3, "xg_against": 0.9}, 
    {"gf": 1.5, "gc": 1.0, "xg_for": 1.6, "xg_against": 0.8}, 
    {"gf": 0.8, "gc": 1.1, "xg_for": 1.0, "xg_against": 1.2}, 
    {"gf": 1.0, "gc": 0.5, "xg_for": 1.1, "xg_against": 0.6}
]
egipto_raw = [
    {"gf": 1.0, "gc": 0.5, "xg_for": 1.1, "xg_against": 0.6}, 
    {"gf": 0.8, "gc": 1.2, "xg_for": 0.9, "xg_against": 1.3}, 
    {"gf": 1.1, "gc": 0.8, "xg_for": 1.2, "xg_against": 0.9}, 
    {"gf": 0.5, "gc": 1.0, "xg_for": 0.7, "xg_against": 1.1}
]

def hydratar_metadata(matches_raw, ref_date="2026-07-03", days_between=14):
    ref = datetime.date.fromisoformat(ref_date)
    n = len(matches_raw)
    out = []
    for idx, m in enumerate(matches_raw):
        d = ref - datetime.timedelta(days=days_between * (n - 1 - idx))
        rec = dict(m)
        rec["date"] = d.isoformat()
        rec["opponent"] = f"Rival {idx + 1}"
        out.append(rec)
    return out

australia = hydratar_metadata(australia_raw)
egipto = hydratar_metadata(egipto_raw)

telemetry = ModelTelemetry()
aus_dynamic = telemetry.synchronize_knowledge_base("AUSTRALIA", australia)
egy_dynamic = telemetry.synchronize_knowledge_base("EGIPTO", egipto)

res = run_prediction("AUSTRALIA", "EGIPTO", aus_dynamic, egy_dynamic, venue="N", verbose=False)

p1, px, p2 = res["p1"], res["px"], res["p2"]
prob_aus_avanza = p1 + (px * 0.5)
prob_egy_avanza = p2 + (px * 0.5)

print(f"Probabilidad de victoria AUSTRALIA (90m): {p1*100:.2f}%")
print(f"Probabilidad de EMPATE (90m): {px*100:.2f}%")
print(f"Probabilidad de victoria EGIPTO (90m): {p2*100:.2f}%")
print(f"Probabilidad de que avance AUSTRALIA: {prob_aus_avanza*100:.2f}%")
print(f"Probabilidad de que avance EGIPTO: {prob_egy_avanza*100:.2f}%")
print("Top 3 marcadores exactos:")
for score in res["top_5_scores"][:3]:
    print(f"  {score['score']}: {score['probability']*100:.2f}%")
