"""
FASE 3 DEL PROTOCOLO — Validación fuera de muestra (la fase que evita
autoengañarte con el barrido de la Fase 2).
 
Idea: partimos el historial en dos mitades CRONOLÓGICAS (no al azar).
- Mitad 1 (más vieja): se usa para elegir el mejor half_life (repite
  la lógica de la Fase 2, pero solo con esta mitad).
- Mitad 2 (más nueva): se usa SOLO para confirmar. El half_life ganador
  de la Mitad 1 se evalúa acá, sin volver a optimizar nada.
 
Si el half_life ganador en la Mitad 1 también da buen RPS en la Mitad 2
(mejor o similar al baseline evaluado en esa misma Mitad 2), hay bases
reales para adoptarlo. Si en la Mitad 2 el resultado es notoriamente
peor, es la señal de que la Fase 2 sobreajustó a las particularidades
de la Mitad 1, y el valor default (o algo más conservador) es más
confiable que el "ganador" del barrido.
"""
import json
from pathlib import Path
from datetime import datetime
 
from fase1_baseline import cargar_historial
 
 
def partir_cronologicamente(matches: list[dict], fraccion_primera_mitad: float = 0.5):
    """
    matches YA debe venir ordenado cronológicamente (cargar_historial
    ya lo hace). Esto simplemente corta la lista en dos, sin mezclar.
    """
    corte = int(len(matches) * fraccion_primera_mitad)
    return matches[:corte], matches[corte:]
 
 
def main():
    csv_path = "results.csv"
    team_a = "ARGENTINA"
    team_b = "FRANCE"
    venue = "N"
 
    if not Path("fase2_barrido_resultado.json").exists():
        print("❌ No encontré fase2_barrido_resultado.json. Corré fase2_barrido.py primero.")
        return
 
    with open("fase2_barrido_resultado.json", encoding="utf-8") as f:
        fase2 = json.load(f)
 
    half_life_ganador = fase2["mejor_candidato"]["half_life"]
    print(f"Half_life ganador de la Fase 2 (a confirmar acá): {half_life_ganador}")
 
    matches_a = cargar_historial(csv_path, team_a)
    matches_b = cargar_historial(csv_path, team_b)
 
    corte_a = int(len(matches_a) * 0.5)
    
    print(f"Total de partidos: {team_a}={len(matches_a)}, {team_b}={len(matches_b)}")
    print(f"Evaluando Mitad 2 desde el partido {corte_a} para {team_a}")
 
    from unified_engine import WalkForwardBacktester
 
    def correr(half_life, optimize_rho):
        backtester = WalkForwardBacktester(min_train_size=10)
        return backtester.run_walkforward(
            team_a=team_a, team_b=team_b,
            all_matches_a=matches_a, all_matches_b=matches_b,
            venue=venue, half_life=half_life, optimize_rho=optimize_rho,
            eval_start_idx=corte_a
        )
 
    # Baseline evaluado SOLO en la Mitad 2, para comparar manzanas con manzanas.
    resultado_baseline_mitad2 = correr(half_life=365, optimize_rho=True)
    rps_baseline_mitad2 = resultado_baseline_mitad2.get("rps_mean") or resultado_baseline_mitad2.get("rps")
 
    # Candidato ganador evaluado en la Mitad 2 (nunca vista durante la elección).
    resultado_candidato_mitad2 = correr(half_life=half_life_ganador, optimize_rho=True)
    rps_candidato_mitad2 = resultado_candidato_mitad2.get("rps_mean") or resultado_candidato_mitad2.get("rps")
 
    print(f"\n=== CONFIRMACIÓN EN MITAD 2 (datos nunca vistos por la Fase 2) ===")
    print(f"Baseline (half_life=365) en Mitad 2:        RPS={rps_baseline_mitad2:.4f}")
    print(f"Candidato (half_life={half_life_ganador}) en Mitad 2: RPS={rps_candidato_mitad2:.4f}")
 
    diferencia = rps_baseline_mitad2 - rps_candidato_mitad2
 
    print(f"\nDiferencia: {diferencia:.4f}")
    if diferencia > 0.01:
        print("✅ El candidato sigue siendo mejor incluso en datos que no vio durante")
        print("   la elección. Hay base razonable para adoptarlo en producción.")
    elif diferencia > -0.01:
        print("➖ La diferencia es marginal (dentro de lo que podría ser ruido de")
        print("   muestra). No hay evidencia fuerte de que el candidato sea")
        print("   realmente mejor que el default — considerá quedarte con el")
        print("   default por simplicidad, salvo que tengas más datos para confirmar.")
    else:
        print("❌ El candidato empeora en datos nuevos. Esto es la señal clásica de")
        print("   que la Fase 2 sobreajustó al ruido de su propia muestra. NO adoptar")
        print("   este half_life en producción — quedarse con el default (365) o")
        print("   repetir el barrido con más datos históricos.")
 
    with open("fase3_confirmacion_resultado.json", "w", encoding="utf-8") as f:
        json.dump({
            "half_life_candidato": half_life_ganador,
            "rps_baseline_mitad2": rps_baseline_mitad2,
            "rps_candidato_mitad2": rps_candidato_mitad2,
            "diferencia": diferencia,
            "decision_sugerida": (
                "adoptar" if diferencia > 0.01 else
                "marginal_revisar" if diferencia > -0.01 else
                "descartar_usar_default"
            ),
            # RPS crudo por partido -- necesario para test_significancia.py.
            # Sin esto, "diferencia" es un punto sin intervalo de confianza.
            "rps_by_match_baseline": resultado_baseline_mitad2.get("rps_by_match", {}),
            "rps_by_match_candidato": resultado_candidato_mitad2.get("rps_by_match", {}),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
 
    print("\n✅ Guardado en fase3_confirmacion_resultado.json")
    print("   Corré test_significancia.py para saber si esa diferencia es real")
    print("   o ruido de muestra -- el numero solo, sin CI, no alcanza para decidir.")
 
 
if __name__ == "__main__":
    main()
