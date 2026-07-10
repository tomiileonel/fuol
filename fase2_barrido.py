"""
FASE 2 DEL PROTOCOLO — Optimización Bayesiana (Optuna)

Migrado de Grid Search a Optuna para optimizar hiperparámetros
(half_life, prior_strength) minimizando el RPS mediante el pipeline
Walk-Forward riguroso.
"""
import json
import optuna
import numpy as np
from datetime import datetime
from pathlib import Path
from data_pipeline import DataPipeline
from walk_forward_pipeline import WalkForwardPipeline
import config

def objective(trial):
    # Definir espacio de búsqueda
    half_life = trial.suggest_float("half_life", 90.0, 730.0, log=True)
    prior_strength = trial.suggest_float("prior_strength", 2.0, 20.0, log=True)
    lambda_scale = trial.suggest_float("lambda_scale", 300.0, 800.0)

    # Temporary override of config for this trial
    # In a real setup, we'd pass these directly to the engines/pipelines
    # but for simplicity, if UnifiedEngine reads config, we can patch it,
    # or better, modify the Pipeline/Engine to accept them.
    # Since we can't easily patch the global config for parallel trials,
    # we'll assume sequential execution and patch config.
    config.HALF_LIFE = half_life
    config.PRIOR_STRENGTH = prior_strength
    config.LAMBDA_SCALE = lambda_scale

    # Instanciar el pipeline con Purge & Embargo
    # Usamos test_window_days=30, embargo_days=14
    tester = WalkForwardPipeline(train_window_days=365*3, test_window_days=30, embargo_days=14)
    pipeline = DataPipeline()
    
    # Nota: skip_fisher_info=True se implementaría a nivel de AdvancedDixonColes 
    # para evitar el cálculo O(N^2) del Hessiano durante Optuna. 
    # Aquí asumimos que AdvancedDixonColes lo omite o que usamos optimize_rho=False 
    # en UnifiedEngine para acelerar.
    
    try:
        results = tester.run(pipeline)
        if not results:
            raise optuna.TrialPruned()
            
        rps = results.get('avg_rps', np.inf)
        if np.isnan(rps) or np.isinf(rps):
            raise optuna.TrialPruned()
            
        return rps
        
    except Exception as e:
        print(f"Error en trial: {e}")
        raise optuna.TrialPruned()

def main():
    print("Iniciando Optimización Bayesiana con Optuna...")
    study = optuna.create_study(direction="minimize", study_name="fuol_hyperparams")
    
    # Optimizar con un máximo de 50 trials (ajustar en producción)
    study.optimize(objective, n_trials=10, timeout=3600)
    
    print("\n=== RANKING (menor RPS primero = mejor) ===")
    print("Mejores Parámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Mejor RPS: {study.best_value}")
    
    # Guardar resultados
    resultados = {
        "best_rps": study.best_value,
        "best_params": study.best_params,
        "timestamp": datetime.now().isoformat(),
        "trials": len(study.trials)
    }
    
    with open("fase2_barrido_resultado.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
        
    print("\n✅ Guardado en fase2_barrido_resultado.json")

if __name__ == "__main__":
    main()
