"""
FASE 2 DEL PROTOCOLO — Barrido de half_life.
 
Corre el mismo walk-forward backtest de la Fase 1, pero repitiéndolo
con distintos valores de half_life, y deja que fit_rho optimice rho
automáticamente en cada corrida (no hace falta elegir rho a mano).
 
Requiere: haber corrido fase1_baseline.py primero, y tener
baseline_resultado.json en la misma carpeta para comparar.
 
IMPORTANTE sobre qué significa "mejor" acá: el candidato con menor
RPS agregado en ESTE barrido es el mejor DE ESTA MUESTRA, no
necesariamente el mejor de verdad. Con pocos partidos, dos valores
de half_life pueden dar RPS muy parecidos por pura casualidad de
muestra. La Fase 3 (abajo) es la que decide si la diferencia
observada acá es lo bastante grande como para confiar en ella.
"""
import json
from pathlib import Path
from datetime import datetime
 
# Reusa la misma función de carga que la Fase 1.
# Si ya la tenés en un módulo compartido, importala de ahí en vez
# de copiarla — la dejo acá inline para que este script sea autocontenido.
from fase1_baseline import cargar_historial
 
 
def main():
    csv_path = "results.csv"
    team_a = "ARGENTINA"
    team_b = "FRANCE"
    venue = "N"
 
    if not Path("baseline_resultado.json").exists():
        print("❌ No encontré baseline_resultado.json.")
        print("   Corré primero fase1_baseline.py — sin ese número de referencia,")
        print("   no hay con qué comparar los resultados de este barrido.")
        return
 
    with open("baseline_resultado.json", encoding="utf-8") as f:
        baseline = json.load(f)
 
    matches_a = cargar_historial(csv_path, team_a)
    matches_b = cargar_historial(csv_path, team_b)
 
    from unified_engine import WalkForwardBacktester
 
    # Candidatos de half_life, en días. 180 = medio año, 365 = un año
    # (el default), 730 = dos años. Rango razonable para fútbol de
    # selecciones, donde los ciclos relevantes son mundial (4 años) y
    # forma reciente (últimos 12-18 meses aprox).
    candidatos_half_life = [90, 180, 270, 365, 540, 730]
 
    resultados = []
    for hl in candidatos_half_life:
        backtester = WalkForwardBacktester(min_train_size=10)
        resultado = backtester.run_walkforward(
            team_a=team_a,
            team_b=team_b,
            all_matches_a=matches_a,
            all_matches_b=matches_b,
            venue=venue,
            half_life=hl,
            optimize_rho=True,  # deja que fit_rho ajuste rho por MLE en cada corrida
        )
        rps_agregado = resultado.get("rps_mean") or resultado.get("rps")
        resultados.append({
            "half_life": hl,
            "rps": rps_agregado,
            "rho_ajustado": resultado.get("rho"),
            "resultado_completo": resultado,
        })
        print(f"half_life={hl:>4} días -> RPS={rps_agregado} (rho ajustado: {resultado.get('rho')})")
 
    resultados.sort(key=lambda r: r["rps"])
 
    print("\n=== RANKING (menor RPS primero = mejor) ===")
    for r in resultados:
        marca = " <- mejor de este barrido" if r == resultados[0] else ""
        print(f"  half_life={r['half_life']:>4}  RPS={r['rps']:.4f}{marca}")
 
    baseline_rps = baseline["resultado"].get("rps_mean") or baseline["resultado"].get("rps")
    mejor = resultados[0]
    diferencia = baseline_rps - mejor["rps"]
 
    print(f"\nBaseline (half_life default): RPS={baseline_rps:.4f}")
    print(f"Mejor candidato de este barrido: half_life={mejor['half_life']}, RPS={mejor['rps']:.4f}")
    print(f"Diferencia: {diferencia:.4f} ({'mejora' if diferencia > 0 else 'empeora'})")
 
    with open("fase2_barrido_resultado.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_rps": baseline_rps,
            "candidatos": resultados,
            "mejor_candidato": mejor,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
 
    print("\n✅ Guardado en fase2_barrido_resultado.json")
    print("   Este resultado NO es todavía la decisión final — ver Fase 3 antes")
    print("   de adoptar el candidato ganador en producción.")
 
 
if __name__ == "__main__":
    main()
