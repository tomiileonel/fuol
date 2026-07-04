import json
import numpy as np
from unified_engine import UnifiedEngine, WalkForwardBacktester
from fase1_baseline import cargar_historial
team_a, team_b = "ARGENTINA", "FRANCE"
all_matches_a = cargar_historial("results.csv", team_a)
all_matches_b = cargar_historial("results.csv", team_b)
venue = "N"
from test_significancia import bootstrap_paired_rps

def evaluate_lambda(lambda_scale: float):
    UnifiedEngine.LAMBDA_SCALE = lambda_scale
    backtester = WalkForwardBacktester()
    metrics = backtester.run_walkforward(team_a, team_b, all_matches_a, all_matches_b, venue=venue, half_life=365, eval_start_idx=0)
    return metrics['rps_by_match']

def main():
    print("Evaluando con LAMBDA_SCALE = 0.23 (Viejo)...")
    rps_old = evaluate_lambda(0.23)
    
    print("Evaluando con LAMBDA_SCALE = 0.4195 (Nuevo)...")
    rps_new = evaluate_lambda(0.4195)
    
    # Emparejar por fecha
    fechas_comunes = sorted(list(set(rps_old.keys()) & set(rps_new.keys())))
    
    if not fechas_comunes:
        print("No hay fechas comunes para comparar.")
        return
        
    vec_old = np.array([rps_old[f] for f in fechas_comunes])
    vec_new = np.array([rps_new[f] for f in fechas_comunes])
    
    rps_mean_old = vec_old.mean()
    rps_mean_new = vec_new.mean()
    
    print(f"\nResultados (N = {len(fechas_comunes)} partidos pareados):")
    print(f"RPS Medio (Viejo 0.23): {rps_mean_old:.4f}")
    print(f"RPS Medio (Nuevo 0.4195): {rps_mean_new:.4f}")
    
    diferencia = rps_mean_old - rps_mean_new # Positivo significa que el nuevo es mejor
    print(f"Mejora (Viejo - Nuevo): {diferencia:.4f}")
    
    print("\nCalculando bootstrap pareado (B=1000)...")
    res = bootstrap_paired_rps(rps_old, rps_new)
    
    ci_lower, ci_upper = res['ci_95']
    print(f"Intervalo de confianza 95% para la mejora de RPS: [{ci_lower:.5f}, {ci_upper:.5f}]")
    print(res['veredicto'])

if __name__ == "__main__":
    main()
