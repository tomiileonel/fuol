"""
FASE 1 DEL PROTOCOLO — Baseline.
 
Corre el walk-forward backtest con los valores DEFAULT del motor
(sin tocar half_life ni rho todavía) y guarda el resultado en un
archivo. Este número es el punto de comparación para todo lo que
sigue: si más adelante cambiás algo y el RPS empeora respecto a
este baseline, ese cambio se descarta, sin importar cuán razonable
haya sonado la justificación para hacerlo.
 
Requiere: haber completado la Fase 0 (tener el historial de partidos
guardado en un formato que este script pueda leer). Ajustá la función
cargar_historial() de más abajo para que lea tu formato real
(CSV, JSON, base de datos — lo que hayas usado en la Fase 0).
"""
import json
import csv
from pathlib import Path
from datetime import datetime
 
# ---------------------------------------------------------------
# AJUSTAR ESTO: cómo cargar el historial que armaste en la Fase 0.
# Dejo un ejemplo asumiendo un CSV con columnas:
# date, team, opponent, goals_for, goals_against, competition, venue
# Si usaste otro formato, esta es la única función que hay que cambiar.
# ---------------------------------------------------------------
def cargar_historial(csv_path: str, team_name: str) -> list[dict]:
    """
    Devuelve la lista de partidos de team_name en el formato que
    espera UnifiedEngine / WalkForwardBacktester: una lista de dicts
    con al menos 'date', 'gf' (goles a favor), 'gc' (goles en contra),
    'competition', 'venue'.
    """
    matches = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = row["home_team"].strip().upper()
            away = row["away_team"].strip().upper()
            team = team_name.strip().upper()
            
            if home != team and away != team:
                continue
                
            try:
                # Filtrar partidos donde falta el score (nulos)
                if not row["home_score"] or not row["away_score"]:
                    continue
                h_score = int(row["home_score"])
                a_score = int(row["away_score"])
            except ValueError:
                continue

            is_home = (home == team)
            is_neutral = (row["neutral"].strip().upper() == "TRUE")
            
            if is_neutral:
                venue = "N"
            else:
                venue = "H" if is_home else "A"
                
            gf = h_score if is_home else a_score
            gc = a_score if is_home else h_score
            
            matches.append({
                "date": row["date"],
                "gf": gf,
                "gc": gc,
                "competition": row["tournament"],
                "venue": venue,
            })
    # Walk-forward necesita orden cronológico estricto para que no haya
    # data leakage (que un partido "futuro" se use para predecir uno pasado).
    matches.sort(key=lambda m: datetime.fromisoformat(m["date"]))
    return matches
 
 
def main():
    # -------------------------------------------------------------
    # AJUSTAR: nombres de equipos y ruta del CSV según tu Fase 0.
    # -------------------------------------------------------------
    csv_path = "results.csv"
    team_a = "ARGENTINA"
    team_b = "FRANCE"
    venue = "N"  # o "team_a", "team_b" según cómo lo maneje tu UnifiedEngine
 
    if not Path(csv_path).exists():
        print(f"❌ No encontré {csv_path}. Este script asume que ya completaste")
        print("   la Fase 0 (construcción del dataset histórico). Si todavía no")
        print("   lo hiciste, ese es el paso anterior a este.")
        return
 
    matches_a = cargar_historial(csv_path, team_a)
    matches_b = cargar_historial(csv_path, team_b)
 
    print(f"Partidos cargados — {team_a}: {len(matches_a)}, {team_b}: {len(matches_b)}")
 
    if len(matches_a) < 30 or len(matches_b) < 30:
        print("⚠️  Con menos de ~30 partidos por selección, el walk-forward")
        print("   backtester va a tener muy pocas iteraciones de test, y el RPS")
        print("   resultante va a tener mucho ruido. El número igual se calcula,")
        print("   pero hay que leerlo con más cautela cuantos menos partidos haya.")
 
    # -------------------------------------------------------------
    # Este bloque asume que unified_engine.py está en el mismo directorio
    # o en el PYTHONPATH. Ajustar el import si tu estructura es distinta.
    # -------------------------------------------------------------
    from unified_engine import WalkForwardBacktester
 
    backtester = WalkForwardBacktester(min_train_size=10)
    resultado = backtester.run_walkforward(
        team_a=team_a,
        team_b=team_b,
        all_matches_a=matches_a,
        all_matches_b=matches_b,
        venue=venue,
        half_life=365.0,
        optimize_rho=True,
    )
 
    print("\n=== RESULTADO BASELINE (valores default) ===")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
 
    # Guardar el resultado para comparar después contra cualquier ajuste.
    with open("baseline_resultado.json", "w", encoding="utf-8") as f:
        json.dump({
            "team_a": team_a,
            "team_b": team_b,
            "n_matches_a": len(matches_a),
            "n_matches_b": len(matches_b),
            "resultado": resultado,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
 
    print("\n✅ Guardado en baseline_resultado.json — este es tu punto de comparación.")
    print("   NINGÚN ajuste de half_life/rho que hagas de acá en más se acepta")
    print("   si el RPS resultante es peor (más alto) que el de este archivo.")
 
 
if __name__ == "__main__":
    main()
