import json
import numpy as np
import datetime
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

espana_raw = [{"gf": 2.1, "gc": 0.7, "res": "W"}, {"gf": 1.8, "gc": 0.6, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}, {"gf": 1.5, "gc": 1.0, "res": "D"}]
austria_raw = [{"gf": 1.2, "gc": 1.1, "res": "W"}, {"gf": 0.8, "gc": 1.5, "res": "L"}, {"gf": 1.0, "gc": 1.0, "res": "D"}, {"gf": 1.5, "gc": 1.2, "res": "W"}]

def hydratar_metadata(matches_raw, ref_date="2026-07-03", days_between=14):
    """Inyecta 'date'/'opponent' sintéticos (ver get_preds.py para el detalle del porqué)."""
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

espana = hydratar_metadata(espana_raw)
austria = hydratar_metadata(austria_raw)

telemetry = ModelTelemetry()
esp_dynamic = telemetry.synchronize_knowledge_base("ESPAÑA", espana)
aus_dynamic = telemetry.synchronize_knowledge_base("AUSTRIA", austria)

res = run_prediction("ESPAÑA", "AUSTRIA", esp_dynamic, aus_dynamic, venue="N", verbose=False)

p1, px, p2 = res["p1"], res["px"], res["p2"]
# Ajuste por eliminación directa (Knockout)
prob_esp_avanza = p1 + (px * 0.5)
prob_aus_avanza = p2 + (px * 0.5)

print(f"Probabilidad de que avance ESPAÑA: {prob_esp_avanza*100:.2f}%")
print(f"Probabilidad de que avance AUSTRIA: {prob_aus_avanza*100:.2f}%")

# NOTA: render_tactical_dashboard removido debido a la deprecación de los 
# modelos geométricos (Voronoi) y cinéticos (EDOs) en favor de métodos
# puramente estadísticos (Dixon-Coles + Gamma-Poisson).
print("Dashboard gráfico deshabilitado. Motor estadístico unificado activo.")
