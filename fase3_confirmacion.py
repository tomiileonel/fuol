"""
FASE 3 DEL PROTOCOLO — Confirmación Institucional.

Idea: Comparamos el half_life ganador de la Fase 2 contra el baseline (365.0)
utilizando el WalkForwardPipeline institucional que garantiza evaluación fuera 
de muestra (Purge & Embargo) para evitar sobreajustes.
"""
import json
from pathlib import Path
from datetime import datetime
from walk_forward_pipeline import WalkForwardPipeline
from data_pipeline import DataPipeline

def main():
    csv_path = "results.csv"

    if not Path("fase2_barrido_resultado.json").exists():
        print("❌ No encontré fase2_barrido_resultado.json. Corré fase2_barrido.py primero.")
        return

    with open("fase2_barrido_resultado.json", encoding="utf-8") as f:
        fase2 = json.load(f)

    half_life_ganador = fase2["mejor_candidato"]["half_life"]
    print(f"Half_life ganador de la Fase 2 (a confirmar acá): {half_life_ganador}")

    print(f"Evaluando Baseline y Candidato usando WalkForwardPipeline...")

    def correr(half_life):
        pipeline = WalkForwardPipeline(
            half_life=half_life,
            prior_strength=2.0,
            lambda_scale=1.0
        )
        data_pipeline = DataPipeline(csv_path=csv_path)
        return pipeline.run(data_pipeline)

    # Baseline
    print("\n--- Corriendo Baseline (half_life = 365) ---")
    resultado_baseline = correr(half_life=365.0)
    rps_baseline = resultado_baseline.get("avg_rps", 1.0)

    # Candidato
    print(f"\n--- Corriendo Candidato (half_life = {half_life_ganador}) ---")
    resultado_candidato = correr(half_life=half_life_ganador)
    rps_candidato = resultado_candidato.get("avg_rps", 1.0)

    print(f"\n=== CONFIRMACIÓN WALK-FORWARD ===")
    print(f"Baseline (half_life=365):        RPS={rps_baseline:.4f}")
    print(f"Candidato (half_life={half_life_ganador}): RPS={rps_candidato:.4f}")

    diferencia = rps_baseline - rps_candidato

    print(f"\nDiferencia: {diferencia:.4f}")
    if diferencia > 0.005:
        print("✅ El candidato demuestra una mejora sólida en el backtest estricto.")
        print("   Hay base razonable para adoptarlo en producción.")
    elif diferencia > -0.005:
        print("➖ La diferencia es marginal. No hay evidencia fuerte de que el candidato sea")
        print("   realmente mejor que el default — considerá quedarte con el default.")
    else:
        print("❌ El candidato empeora el RPS general. Esto es la señal clásica de")
        print("   que la Fase 2 sobreajustó. NO adoptar este half_life en producción.")

    with open("fase3_confirmacion_resultado.json", "w", encoding="utf-8") as f:
        json.dump({
            "half_life_candidato": half_life_ganador,
            "rps_baseline": rps_baseline,
            "rps_candidato": rps_candidato,
            "diferencia": diferencia,
            "decision_sugerida": (
                "adoptar" if diferencia > 0.005 else
                "marginal_revisar" if diferencia > -0.005 else
                "descartar_usar_default"
            ),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print("\n✅ Guardado en fase3_confirmacion_resultado.json")

if __name__ == "__main__":
    main()
