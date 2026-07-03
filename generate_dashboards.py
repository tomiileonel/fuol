import numpy as np
import datetime
from unified_engine import run_prediction
from performance_tracker import ModelTelemetry

# ==============================================================================
# NOTA DE MIGRACIÓN
# ==============================================================================
# Este script originalmente instanciaba SupremePredictionEngine y llamaba a
# render_tactical_dashboard(team_a, team_b, res, momentum) -- ninguna de las
# dos existe en el código actual:
#   - SupremePredictionEngine fue reemplazada por UnifiedEngine (Dixon-Coles +
#     Gamma-Poisson + Elo).
#   - La única función de dashboard real en tournament_dashboard.py es
#     render_tournament_center(champions_distribution, n_simulations), que
#     grafica la distribución de campeones de una simulación Monte Carlo de
#     TORNEO completo (ver tournament_simulator.py / MonteCarloTournament),
#     no un dashboard táctico de un partido individual. Su firma es
#     incompatible con un output de partido único como el de run_prediction().
#
# Por eso este script ya no intenta dibujar un dashboard: imprime el resumen
# estadístico de cada enfrentamiento usando el motor real. Si querés un
# dashboard visual por partido, hay que escribir una función nueva (p. ej.
# render_matchup_summary) -- no reutilizar render_tournament_center.
# ==============================================================================

def hydratar_metadata(matches_raw, ref_date="2026-07-03", days_between=14):
    """Inyecta 'date'/'opponent' sintéticos para activar TimeWeighter/Elo."""
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

belgium_raw = [{"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0}, {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3}, {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1}]
senegal_raw = [{"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1}, {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0}, {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0}]
usa_raw = [{"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5}, {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0}, {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3}]
bosnia_raw = [{"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5}, {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4}, {"gf":3,"gc":1}]

belgium = hydratar_metadata(belgium_raw)
senegal = hydratar_metadata(senegal_raw)
usa = hydratar_metadata(usa_raw)
bosnia = hydratar_metadata(bosnia_raw)

telemetry = ModelTelemetry()
belgium_dynamic = telemetry.synchronize_knowledge_base("BÉLGICA", belgium)
senegal_dynamic = telemetry.synchronize_knowledge_base("SENEGAL", senegal)
usa_dynamic = telemetry.synchronize_knowledge_base("EEUU", usa)
bosnia_dynamic = telemetry.synchronize_knowledge_base("BOSNIA-HERZ.", bosnia)

def resumen(team_a, team_b, dyn_a, dyn_b, venue="N"):
    res = run_prediction(team_a, team_b, dyn_a, dyn_b, venue=venue, verbose=False)
    telemetry.log_prediction(team_a, team_b, res)
    print(f"\n{team_a} vs {team_b}  (Elo: {res['elo_a']} - {res['elo_b']})")
    print(f"  P(1)={res['p1']*100:.1f}%  P(X)={res['px']*100:.1f}%  P(2)={res['p2']*100:.1f}%")
    print(f"  lam={res['lam']} mu={res['mu']}  Top marcador: {res['top_5_scores'][0]['score']} "
          f"({res['top_5_scores'][0]['prob']*100:.1f}%)")
    return res

# 1. BÉLGICA vs SENEGAL
resumen("BÉLGICA", "SENEGAL", belgium_dynamic, senegal_dynamic, venue="N")

# 2. EEUU vs BOSNIA-HERZ.
resumen("EEUU", "BOSNIA-HERZ.", usa_dynamic, bosnia_dynamic, venue="H")

print("\nResumenes guardados en supreme_predictions.db vía ModelTelemetry.log_prediction().")
