"""
calibrate_optuna.py — Calibración bayesiana de hiperparámetros del motor FUOL.

Usa Optuna (TPESampler + MedianPruner) para buscar los hiperparámetros del
WalkForwardPipeline que minimizan el RPS (Ranked Probability Score) sobre
walk-forward con Purge & Embargo estricto.

HIPERPARÁMETROS CALIBRADOS
--------------------------
  lambda_scale     : multiplicador Elo→λ en EloRegistry.expected_goal_ratio
                     (controla cuánto impacto tiene el Elo en la tasa de goles)
  prior_strength   : "partidos imaginarios" de confianza en el prior Bayesiano
                     (controla cuánto se contrae la posterior hacia el prior)
  half_life        : decaimiento temporal en días
                     (controla cuánto pesan los partidos recientes vs viejos)

DISEÑO ANTI-SOBAJUSTE
---------------------
El dataset se parte en 3 ventanas cronológicas:

  [Inicio ──────────────── Holdout-Start ──── Fin]
   ↑ Train + Validation      ↑ Holdout final
   (Optuna opera acá)         (se toca 1 sola vez al final)

Optuna solo ve la ventana Train+Validation. La ventana Holdout final
(últimos N meses, default 12) se reserva para verificar que el ganador
generaliza — si el RPS en holdout es significativamente peor que en
validation, hay sobreajuste y el trial ganador se descarta.

USAGE
-----
    # Calibración rápida (smoke test, ~5 min)
    python calibrate_optuna.py --trials 5 --holdout-months 12

    # Calibración completa (~3-5 horas sobre los 49k partidos)
    python calibrate_optuna.py --trials 50 --holdout-months 12

    # Reanudar un estudio interrumpido
    python calibrate_optuna.py --resume --study fuol_calibration

OUTPUT
------
  - calibracion_optuna_resultado.json: mejores hiperparámetros + métricas
  - optuna_study.db: base SQLite del estudio (para reanudar o visualizar)
  - Prints en stdout con progreso por trial

REQUISITOS
----------
  optuna >= 3.6 (ya en requirements.txt)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

# Silenciar warnings de Optuna (log excesivo)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Safety net para Windows PowerShell: si stdout no soporta UTF-8 por defecto
# (cp1252 en Windows en-US), reemplazarlo por un stream UTF-8 safe. Esto
# previene el UnicodeEncodeError al imprimir caracteres acentuados o simbolos
# Unicode (→, é, ó, etc.) sin necesidad de setear PYTHONIOENCODING afuera.
# ---------------------------------------------------------------------------
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        # Python < 3.7 o stdout redirigido a archivo sin reconfigure
        pass

from data_pipeline import DataPipeline
from walk_forward_pipeline import WalkForwardPipeline


# ===========================================================================
# CONFIGURACIÓN DEFAULT
# ===========================================================================

DEFAULT_TRIALS = 30
DEFAULT_HOLDOUT_MONTHS = 12
DEFAULT_TRAIN_WINDOW_DAYS = 365 * 4   # 4 años de train por fold
DEFAULT_TEST_WINDOW_DAYS = 30          # 1 mes de test por fold
DEFAULT_EMBARGO_DAYS = 14              # gap anti-leakage

# Espacio de búsqueda (rangos elegidos para cubrir los defaults actuales:
# lambda_scale=0.23, prior_strength=6.0, half_life=365)
SEARCH_SPACE = {
    'lambda_scale':    (0.10, 0.50),   # log-uniform
    'prior_strength':  (3.0, 10.0),    # uniform
    'half_life':       [180, 270, 365, 500, 730],  # categorical
}


# ===========================================================================
# CARGA Y PARTICIÓN TEMPORAL DEL DATASET
# ===========================================================================

def load_and_partition_data(
    csv_path: str = "results.csv",
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga el CSV y lo parte en:
      - train_val_df: todo menos los últimos `holdout_months` meses
      - holdout_df: últimos `holdout_months` meses (reservado para validación final)
      - full_df: dataset completo (se usa solo al final con el ganador)

    Retorna (train_val_df, holdout_df, full_df).
    """
    pipeline = DataPipeline(csv_path=csv_path)
    full_df, _ = pipeline.prepare_data()
    full_df['date'] = pd.to_datetime(full_df['date'])

    cutoff = full_df['date'].max() - pd.DateOffset(months=holdout_months)
    train_val_df = full_df[full_df['date'] < cutoff].copy()
    holdout_df = full_df[full_df['date'] >= cutoff].copy()

    print(f"Dataset cargado: {len(full_df)} partidos "
          f"({full_df['date'].min().date()} → {full_df['date'].max().date()})")
    print(f"  Train+Validation: {len(train_val_df)} partidos (hasta {cutoff.date()})")
    print(f"  Holdout final:    {len(holdout_df)} partidos (desde {cutoff.date()})")

    return train_val_df, holdout_df, full_df, cutoff


# ===========================================================================
# DATAPIPELINE ACOTADO (para inyectar el sub-DF en WalkForwardPipeline)
# ===========================================================================

class ScopedDataPipeline(DataPipeline):
    """
    DataPipeline que retorna un sub-DataFrame predefinido en lugar de leer
    todo el CSV. Permite que WalkForwardPipeline.run() opere sobre la
    ventana Train+Validation sin tocar el holdout.
    """
    def __init__(self, scoped_df: pd.DataFrame, csv_path: str = "results.csv"):
        super().__init__(csv_path=csv_path)
        self._scoped_df = scoped_df

    def prepare_data(self):
        # Respetar la firma de DataPipeline.prepare_data -> (df, elo_registry)
        # El WalkForwardPipeline solo usa el df.
        return self._scoped_df.copy(), None


# ===========================================================================
# OBJECTIVE FUNCTION PARA OPTUNA
# ===========================================================================

def make_objective(
    train_val_df: pd.DataFrame,
    train_window_days: int,
    test_window_days: int,
    embargo_days: int,
) -> optuna.study.StudyDirection:
    """
    Factory de la función objetivo. Recibe el DF de train+validation ya
    particionado y retorna una función `objective(trial)` lista para
    pasarse a `optuna.study.optimize()`.
    """
    scoped_pipeline = ScopedDataPipeline(train_val_df)

    def objective(trial: optuna.Trial) -> float:
        # Sampleo de hiperparámetros
        lambda_scale = trial.suggest_float(
            'lambda_scale',
            SEARCH_SPACE['lambda_scale'][0],
            SEARCH_SPACE['lambda_scale'][1],
            log=True,
        )
        prior_strength = trial.suggest_float(
            'prior_strength',
            SEARCH_SPACE['prior_strength'][0],
            SEARCH_SPACE['prior_strength'][1],
        )
        half_life = trial.suggest_categorical(
            'half_life',
            SEARCH_SPACE['half_life'],
        )

        # Log del trial
        trial.set_user_attr('params_str',
            f"λ={lambda_scale:.4f} | prior={prior_strength:.2f} | hl={half_life}")

        # Construir pipeline con los hiperparámetros sampleados
        wfp = WalkForwardPipeline(
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            embargo_days=embargo_days,
            min_train_size=100,
            half_life=half_life,
            prior_strength=prior_strength,
            lambda_scale=lambda_scale,
        )

        # Ejecutar walk-forward
        try:
            t0 = time.time()
            metrics = wfp.run(scoped_pipeline)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [Trial {trial.number}] ERROR: {e}")
            raise TrialPruned()

        if not metrics or 'avg_rps' not in metrics:
            print(f"  [Trial {trial.number}] Sin metricas (posible dataset insuficiente)")
            raise TrialPruned()

        avg_rps = float(metrics['avg_rps'])
        avg_brier = float(metrics['avg_brier'])
        n_folds = int(metrics['n_folds'])
        n_samples = int(metrics['total_test_samples'])

        trial.set_user_attr('avg_rps', avg_rps)
        trial.set_user_attr('avg_brier', avg_brier)
        trial.set_user_attr('n_folds', n_folds)
        trial.set_user_attr('n_samples', n_samples)
        trial.set_user_attr('elapsed_sec', round(elapsed, 1))

        # Pruning: si el RPS parcial está por encima de la mediana de los
        # trials completados hasta ahora, cortar temprano.
        trial.report(avg_rps, step=0)
        if trial.should_prune():
            print(f"  [Trial {trial.number}] PRUNED | RPS={avg_rps:.4f} "
                  f"({n_samples} samples, {n_folds} folds, {elapsed:.1f}s)")
            raise TrialPruned()

        print(f"  [Trial {trial.number}] DONE   | RPS={avg_rps:.4f} "
              f"Brier={avg_brier:.4f} | {n_samples} samples, {n_folds} folds, "
              f"{elapsed:.1f}s | {trial.user_attrs['params_str']}")

        return avg_rps

    return objective


# ===========================================================================
# VALIDACIÓN FINAL EN HOLDOUT
# ===========================================================================

def evaluate_on_holdout(
    best_params: dict,
    holdout_df: pd.DataFrame,
    full_df: pd.DataFrame,
    train_window_days: int,
    test_window_days: int,
    embargo_days: int,
) -> dict:
    """
    Corre el pipeline con los mejores hiperparámetros sobre el dataset
    completo (incluyendo el holdout). Compara el RPS en la ventana holdout
    vs el RPS en la ventana train+validation para detectar sobreajuste.
    """
    print("\n" + "=" * 70)
    print("VALIDACION FINAL EN HOLDOUT (ultimos 12 meses, nunca vistos por Optuna)")
    print("=" * 70)
    print(f"Hiperparametros ganadores: {best_params}")

    # Pipeline con los hiperparámetros ganadores, sobre el dataset completo
    full_pipeline = ScopedDataPipeline(full_df)
    wfp = WalkForwardPipeline(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        embargo_days=embargo_days,
        min_train_size=100,
        half_life=best_params['half_life'],
        prior_strength=best_params['prior_strength'],
        lambda_scale=best_params['lambda_scale'],
    )

    t0 = time.time()
    metrics_full = wfp.run(full_pipeline)
    elapsed = time.time() - t0

    # Filtrar rps_by_match a solo la ventana holdout
    rps_by_match = metrics_full.get('rps_by_match', {})
    holdout_cutoff = holdout_df['date'].min()
    rps_holdout = {d: v for d, v in rps_by_match.items()
                   if pd.to_datetime(d) >= holdout_cutoff}
    rps_trainval = {d: v for d, v in rps_by_match.items()
                    if pd.to_datetime(d) < holdout_cutoff}

    rps_holdout_mean = float(np.mean(list(rps_holdout.values()))) if rps_holdout else None
    rps_trainval_mean = float(np.mean(list(rps_trainval.values()))) if rps_trainval else None

    print(f"\nResultados sobre dataset completo ({len(rps_by_match)} partidos, {elapsed:.1f}s):")
    print(f"  RPS global:        {metrics_full['avg_rps']:.4f}")
    print(f"  Brier global:      {metrics_full['avg_brier']:.4f}")
    print(f"  RPS Train+Val:     {rps_trainval_mean:.4f}  (N={len(rps_trainval)})")
    print(f"  RPS Holdout:       {rps_holdout_mean:.4f}  (N={len(rps_holdout)})")

    # Detección de sobreajuste
    if rps_holdout_mean is not None and rps_trainval_mean is not None:
        delta = rps_holdout_mean - rps_trainval_mean
        # Un delta > 5% del RPS de trainval es señal amarilla
        rel_delta = delta / rps_trainval_mean if rps_trainval_mean > 0 else 0
        if rel_delta > 0.10:
            verdict = "[WARN] SOBREAJUSTE: RPS en holdout es >10% peor que en train+val"
        elif rel_delta > 0.05:
            verdict = "[+] Degradacion moderada (5-10%) - revisar si el N de holdout es suficiente"
        else:
            verdict = "[OK] Generaliza bien (degradacion < 5%)"
        print(f"  Delta (holdout - trainval): {delta:+.4f} ({rel_delta:+.1%})")
        print(f"  {verdict}")
    else:
        verdict = "Sin datos suficientes para evaluar sobreajuste"

    return {
        'rps_full': float(metrics_full['avg_rps']),
        'brier_full': float(metrics_full['avg_brier']),
        'rps_trainval': rps_trainval_mean,
        'rps_holdout': rps_holdout_mean,
        'n_holdout_samples': len(rps_holdout),
        'n_trainval_samples': len(rps_trainval),
        'verdict': verdict,
        'rps_by_match_holdout': rps_holdout,
        'rps_by_match_trainval': rps_trainval,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibración bayesiana de hiperparámetros FUOL con Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'Número de trials de Optuna (default {DEFAULT_TRIALS})')
    parser.add_argument('--holdout-months', type=int, default=DEFAULT_HOLDOUT_MONTHS,
                        help=f'Meses reservados para holdout final (default {DEFAULT_HOLDOUT_MONTHS})')
    parser.add_argument('--study', type=str, default='fuol_calibration',
                        help='Nombre del estudio Optuna (para SQLite)')
    parser.add_argument('--resume', action='store_true',
                        help='Reanudar estudio existente en lugar de crear uno nuevo')
    parser.add_argument('--csv', type=str, default='results.csv',
                        help='Ruta al CSV de partidos (default results.csv)')
    parser.add_argument('--train-window-days', type=int, default=DEFAULT_TRAIN_WINDOW_DAYS,
                        help=f'Ventana de train por fold en días (default {DEFAULT_TRAIN_WINDOW_DAYS})')
    parser.add_argument('--test-window-days', type=int, default=DEFAULT_TEST_WINDOW_DAYS,
                        help=f'Ventana de test por fold en días (default {DEFAULT_TEST_WINDOW_DAYS})')
    parser.add_argument('--embargo-days', type=int, default=DEFAULT_EMBARGO_DAYS,
                        help=f'Días de embargo anti-leakage (default {DEFAULT_EMBARGO_DAYS})')
    parser.add_argument('--no-holdout', action='store_true',
                        help='Saltar validación en holdout (solo Optuna)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  FUOL - Calibracion Optuna de Hiperparametros")
    print(f"  Trials: {args.trials} | Holdout: {args.holdout_months} meses | "
          f"Study: {args.study}")
    print("=" * 70)

    # 1. Cargar y partir datos
    train_val_df, holdout_df, full_df, cutoff = load_and_partition_data(
        csv_path=args.csv,
        holdout_months=args.holdout_months,
    )

    if len(holdout_df) < 200:
        print(f"\n[WARN] Holdout tiene solo {len(holdout_df)} partidos. Considera aumentar --holdout-months.")
    if len(train_val_df) < 1000:
        print(f"\n[WARN] Train+Val tiene solo {len(train_val_df)} partidos. Optuna puede no converger.")

    # 2. Configurar estudio Optuna
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
        interval_steps=1,
    )

    # SIEMPRE load_if_exists=True: si la DB existe (de una corrida previa
    # interrumpida o exitosa), carga el estudio; si no existe, lo crea.
    # Esto evita el DuplicatedStudyError cuando se relanza sin --resume
    # después de un crash, y es idempotente: correr dos veces el mismo
    # comando no rompe, simplemente continua agregando trials.
    # La flag --resume se mantiene por compatibilidad retroactiva pero
    # ya no hace falta.
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///optuna_study.db",
        direction='minimize',  # minimizar RPS
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    n_existing_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_existing_total = len(study.trials)
    if n_existing_total > 0:
        print(f"\nEstudio existente detectado: {n_existing_completed} completados, "
              f"{n_existing_total - n_existing_completed} pruned/running de antes.")
        if not args.resume:
            print(f"[INFO] Continuando sobre el estudio existente (load_if_exists=True por defecto).")
            print(f"       Si queres empezar de cero, borra optuna_study.db antes.")

    # Calcular trials adicionales a correr:
    #   --trials se interpreta como TRIALES NUEVOS a agregar al estudio.
    #   Si el estudio ya tiene N trials completados, el total final será
    #   N + args.trials.
    #   Para detenerse en un total exacto, usar --resume --trials 0 (solo
    #   corre la validación en holdout).
    n_new_trials = max(0, args.trials)

    print(f"\nIniciando optimizacion (objetivo: minimizar avg_rps)...")
    if n_existing_completed > 0:
        print(f"  Se agregaran {n_new_trials} trials nuevos a los {n_existing_completed} existentes.")
    else:
        print(f"  Se correran {n_new_trials} trials desde cero.")
    print(f"Espacio de busqueda:")
    print(f"  lambda_scale:    log-uniform [{SEARCH_SPACE['lambda_scale'][0]}, {SEARCH_SPACE['lambda_scale'][1]}]")
    print(f"  prior_strength:  uniform [{SEARCH_SPACE['prior_strength'][0]}, {SEARCH_SPACE['prior_strength'][1]}]")
    print(f"  half_life:       categorical {SEARCH_SPACE['half_life']}")
    print()

    # 3. Correr optimización
    objective = make_objective(
        train_val_df=train_val_df,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        embargo_days=args.embargo_days,
    )

    if n_new_trials == 0:
        print("[INFO] --trials 0: saltando optimizacion, solo se correra validacion en holdout.")
        elapsed = 0.0
    else:
        t0 = time.time()
        study.optimize(objective, n_trials=n_new_trials, show_progress_bar=False)
        elapsed = time.time() - t0

    # 4. Reportar resultados
    print("\n" + "=" * 70)
    print(f"  OPTUNA COMPLETADO - {args.trials} trials en {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Completados: {len(completed)} | Pruned: {len(pruned)}")

    if not completed:
        print("\n[FAIL] No hay trials completados. Revisa logs arriba.")
        sys.exit(1)

    best = study.best_trial
    print(f"\n  [BEST] Mejor trial #{best.number}:")
    print(f"     RPS:     {best.value:.4f}")
    print(f"     Brier:   {best.user_attrs.get('avg_brier', 'N/A')}")
    print(f"     Samples: {best.user_attrs.get('n_samples', 'N/A')}")
    print(f"     Folds:   {best.user_attrs.get('n_folds', 'N/A')}")
    print(f"     Tiempo:  {best.user_attrs.get('elapsed_sec', 'N/A')}s")
    print(f"     Params:")
    for k, v in best.params.items():
        print(f"       {k}: {v}")

    # Comparar con defaults actuales
    print(f"\n  [STATS] Comparacion con defaults actuales:")
    print(f"     lambda_scale:    default=0.23    | optimo={best.params['lambda_scale']:.4f}    "
          f"delta={best.params['lambda_scale'] - 0.23:+.4f}")
    print(f"     prior_strength:  default=6.0     | optimo={best.params['prior_strength']:.4f}   "
          f"delta={best.params['prior_strength'] - 6.0:+.4f}")
    print(f"     half_life:       default=365     | optimo={best.params['half_life']}    "
          f"delta={best.params['half_life'] - 365:+d}")

    # 5. Validación en holdout
    holdout_results = {}
    if not args.no_holdout:
        holdout_results = evaluate_on_holdout(
            best_params=best.params,
            holdout_df=holdout_df,
            full_df=full_df,
            train_window_days=args.train_window_days,
            test_window_days=args.test_window_days,
            embargo_days=args.embargo_days,
        )

    # 6. Persistir resultados
    output = {
        'best_trial': {
            'number': best.number,
            'value_rps': best.value,
            'params': best.params,
            'user_attrs': {k: v for k, v in best.user_attrs.items() if k != 'params_str'},
        },
        'n_completed': len(completed),
        'n_pruned': len(pruned),
        'total_elapsed_sec': round(elapsed, 1),
        'search_space': {
            'lambda_scale': list(SEARCH_SPACE['lambda_scale']),
            'prior_strength': list(SEARCH_SPACE['prior_strength']),
            'half_life': SEARCH_SPACE['half_life'],
        },
        'holdout_validation': holdout_results,
        'defaults_for_comparison': {
            'lambda_scale': 0.23,
            'prior_strength': 6.0,
            'half_life': 365,
        },
        'dataset_info': {
            'total_matches': len(full_df),
            'train_val_matches': len(train_val_df),
            'holdout_matches': len(holdout_df),
            'holdout_cutoff': str(cutoff.date()),
            'date_range': [str(full_df['date'].min().date()), str(full_df['date'].max().date())],
        },
        'timestamp': datetime.now().isoformat(),
    }

    out_path = 'calibracion_optuna_resultado.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Resultados guardados en {out_path}")
    print(f"   Estudio SQLite: optuna_study.db (para reanudar: --resume)")

    # 7. Proximos pasos sugeridos
    print("\n" + "=" * 70)
    print("  PROXIMOS PASOS")
    print("=" * 70)
    print("  1. Si el veredicto de holdout es '[OK] Generaliza bien':")
    print(f"     Actualizar config.py con los hiperparametros optimos:")
    print(f"       LAMBDA_SCALE = {best.params['lambda_scale']:.4f}")
    print(f"       PRIOR_STRENGTH = {best.params['prior_strength']:.4f}")
    print(f"       DEFAULT_HALF_LIFE = {best.params['half_life']}")
    print("  2. Si el veredicto es '[WARN] SOBREAJUSTE':")
    print("     - Ampliar --holdout-months a 18 o 24")
    print("     - Reducir search space (ej. fijar half_life=365)")
    print("     - Aumentar --trials con --resume para explorar mejor")
    print("  3. Para visualizar el estudio:")
    print("     pip install plotly")
    print("     optuna-dashboard sqlite:///optuna_study.db")
    print()


if __name__ == '__main__':
    main()
