import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from calibrate_optuna import evaluate_on_holdout, load_and_partition_data
from test_significancia import bootstrap_paired_rps

def main():
    HOLDOUT_MONTHS = 12
    # Cargar datos
    train_val_df, holdout_df, full_df, cutoff = load_and_partition_data(
        csv_path="results.csv",
        holdout_months=HOLDOUT_MONTHS
    )
    n_holdout_total = len(holdout_df)

    # Parametros
    baseline_params = {'lambda_scale': 0.23, 'prior_strength': 6.0, 'half_life': 365.0}
    optimized_params = {'lambda_scale': 0.4031, 'prior_strength': 7.21, 'half_life': 365.0}

    print(f"\n{'='*70}")
    print("--- CORRIENDO BASELINE ---")
    results_baseline = evaluate_on_holdout(
        best_params=baseline_params,
        holdout_df=holdout_df,
        full_df=full_df,
        train_window_days=1460,
        test_window_days=30,
        embargo_days=14
    )

    print(f"\n{'='*70}")
    print("--- CORRIENDO OPTIMIZADO ---")
    results_optimized = evaluate_on_holdout(
        best_params=optimized_params,
        holdout_df=holdout_df,
        full_df=full_df,
        train_window_days=1460,
        test_window_days=30,
        embargo_days=14
    )

    baseline = results_baseline['rps_by_match_holdout']
    candidato = results_optimized['rps_by_match_holdout']

    print(f"\n{'='*70}")
    print("--- CALCULANDO BOOTSTRAP PAREADO ---")
    res = bootstrap_paired_rps(baseline, candidato)
    
    baseline_tv = results_baseline['rps_by_match_trainval']
    candidato_tv = results_optimized['rps_by_match_trainval']
    res_tv = bootstrap_paired_rps(baseline_tv, candidato_tv) if baseline_tv and candidato_tv else None

    # Common keys computation for the JSON:
    common_keys = sorted(list(set(baseline.keys()) & set(candidato.keys())))
    common_keys_tv = sorted(list(set(baseline_tv.keys()) & set(candidato_tv.keys()))) if baseline_tv else []

    rps_mean_baseline = np.mean([baseline[k] for k in common_keys]) if common_keys else 0.0
    rps_mean_optimized = np.mean([candidato[k] for k in common_keys]) if common_keys else 0.0
    delta = rps_mean_optimized - rps_mean_baseline

    vec_b_tv = np.array([baseline_tv[k] for k in common_keys_tv]) if common_keys_tv else np.array([])
    vec_o_tv = np.array([candidato_tv[k] for k in common_keys_tv]) if common_keys_tv else np.array([])

    output = {
        'baseline_params': baseline_params,
        'optimized_params': optimized_params,
        'results_baseline': {
            'rps_full': results_baseline['rps_full'],
            'rps_holdout': results_baseline['rps_holdout'],
            'rps_trainval': results_baseline['rps_trainval'],
            'n_holdout': results_baseline['n_holdout_samples'],
            'n_trainval': results_baseline['n_trainval_samples'],
        },
        'results_optimized': {
            'rps_full': results_optimized['rps_full'],
            'rps_holdout': results_optimized['rps_holdout'],
            'rps_trainval': results_optimized['rps_trainval'],
            'n_holdout': results_optimized['n_holdout_samples'],
            'n_trainval': results_optimized['n_trainval_samples'],
        },
        'bootstrap_holdout': {
            'n_common': len(common_keys),
            'delta_mean': delta,
            'delta_relative_pct': (delta / rps_mean_baseline * 100) if rps_mean_baseline else 0,
            'ci95_lo': res['ci95_lo'],
            'ci95_hi': res['ci95_hi'],
            'verdict': res['veredicto'],
        },
        'bootstrap_trainval': {
            'n_common': len(common_keys_tv),
            'delta_mean': vec_b_tv.mean() - vec_o_tv.mean() if common_keys_tv else None,
            'ci95_lo': res_tv['ci95_lo'] if res_tv else None,
            'ci95_hi': res_tv['ci95_hi'] if res_tv else None,
            'verdict': res_tv['veredicto'] if res_tv else None,
        } if res_tv else None,
        'dataset_info': {
            'total_matches': len(full_df),
            'holdout_months': HOLDOUT_MONTHS,
            'holdout_cutoff': str(cutoff.date()),
            'holdout_total_matches': n_holdout_total,
            'date_range': [str(full_df['date'].min().date()), str(full_df['date'].max().date())],
        },
        'timestamp': datetime.now().isoformat(),
    }

    out_path = 'bootstrap_holdout_resultado.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"  Resultados guardados en {out_path}")
    print(f"{'='*70}")

    # 6. Resumen ejecutivo
    print(f"\n  RESUMEN EJECUTIVO:")
    print(f"  =================")
    print(f"  Holdout (N={len(common_keys)} partidos pareados):")
    print(f"    Baseline RPS:    {rps_mean_baseline:.4f}")
    print(f"    Optimizado RPS:  {rps_mean_optimized:.4f}")
    if rps_mean_baseline:
        print(f"    Delta:           {delta:+.4f} ({delta/rps_mean_baseline*100:+.2f}%)")
    print(f"    IC 95%:          [{res['ci95_lo']:.5f}, {res['ci95_hi']:.5f}]")
    print(f"    Significativo:   {'SI' if not res['cruza_cero'] else 'NO'}")
    print()
    if not res['cruza_cero'] and delta < 0:
        print(f"  -> La mejora del optimizado es ESTADISTICAMENTE SIGNIFICATIVA.")
        print(f"     Los hiperparametros lambda_scale={optimized_params['lambda_scale']}, prior_strength={optimized_params['prior_strength']}")
        print(f"     pueden adoptarse con confianza estadistica.")
    elif not res['cruza_cero'] and delta > 0:
        print(f"  -> El optimizado es ESTADISTICAMENTE PEOR que el baseline.")
        print(f"     REVERTIR config.py a los defaults.")
    else:
        print(f"  -> La diferencia NO es estadisticamente significativa.")
        print(f"     El delta de {delta:+.4f} podria ser ruido de muestra.")
        print(f"     Los hiperparametros optimizados no perjudican, pero la")
        print(f"     mejora no es concluyente.")

if __name__ == '__main__':
    main()
