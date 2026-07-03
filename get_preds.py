import json
import numpy as np
import datetime
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

usa_raw = [{"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5}, {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0}, {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3}]
bosnia_raw = [{"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5}, {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4}, {"gf":3,"gc":1}]

def hydratar_metadata(matches_raw, ref_date="2026-07-03", days_between=14):
    """
    Inyecta 'date' y 'opponent' sintéticos y cronológicamente ordenados.
    Sin 'date', TimeWeighter.compute_weights() y EloRating.elo_from_matches()
    caen en su fallback ('1970-01-01' / Elo 1600 por defecto) y el motor
    ignora por completo la ponderación temporal y el prior histórico de Elo.
    Esto es un placeholder razonable para demos; en producción 'date' y
    'opponent' deben venir del pipeline real de datos (ver AdvancedDataPipeline).
    """
    ref = datetime.date.fromisoformat(ref_date)
    n = len(matches_raw)
    out = []
    for idx, m in enumerate(matches_raw):
        # El partido más reciente queda al final de la lista (idx = n-1 -> más cerca de ref_date)
        d = ref - datetime.timedelta(days=days_between * (n - 1 - idx))
        rec = dict(m)
        rec["date"] = d.isoformat()
        rec["opponent"] = f"Rival {idx + 1}"
        out.append(rec)
    return out

usa = hydratar_metadata(usa_raw)
bosnia = hydratar_metadata(bosnia_raw)

telemetry = ModelTelemetry()
usa_dynamic = telemetry.synchronize_knowledge_base("EEUU", usa)
bosnia_dynamic = telemetry.synchronize_knowledge_base("BOSNIA-HERZ.", bosnia)

res2 = run_prediction("EEUU", "BOSNIA-HERZ.", usa_dynamic, bosnia_dynamic, venue="H", verbose=False)

print("Lambda Final:", res2["lam"])
print("Mu Final:", res2["mu"])
print("Matrix shape:", res2["score_matrix"].shape)
print("Max exact score:", res2["top_5_scores"][0]["score"])
