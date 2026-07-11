"""
FASE 1 DEL PROTOCOLO — Baseline Institucional.

Corre el walk-forward backtest sobre el dataset histórico completo utilizando la
arquitectura institucional `WalkForwardPipeline` con los valores DEFAULT del motor
y calibración. Guarda el resultado en un archivo. Este número es el punto de comparación.
"""
import json
from datetime import datetime
from pathlib import Path
from walk_forward_pipeline import WalkForwardPipeline
from data_pipeline import DataPipeline

def main():
    csv_path = "results.csv"
    if not Path(csv_path).exists():
        print(f"❌ No encontré {csv_path}. Este script asume que ya completaste")
        print("   la Fase 0 (construcción del dataset histórico).")
        return

    print(f"Iniciando Walk-Forward Backtester Pipeline con valores default...")
    print(f"Esto evaluará el dataset completo con Purge & Embargo estricto. Puede tomar tiempo.\n")
    
    pipeline = WalkForwardPipeline(
        half_life=365.0,
        prior_strength=2.0,
        lambda_scale=1.0
    )
    
    data_pipeline = DataPipeline(csv_path=csv_path)
    resultado = pipeline.run(data_pipeline)

    print("\n=== RESULTADO BASELINE (valores default) ===")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
 
    with open("baseline_resultado.json", "w", encoding="utf-8") as f:
        json.dump({
            "resultado": resultado,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
 
    print("\n[OK] Guardado en baseline_resultado.json — este es tu punto de comparación.")
    print("   NINGÚN ajuste de parámetros se acepta si empeora este RPS.")

if __name__ == "__main__":
    main()
