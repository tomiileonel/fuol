import numpy as np
from walk_forward_pipeline import WalkForwardPipeline
from data_pipeline import DataPipeline
from test_significancia import bootstrap_paired_rps

def evaluate_lambda(lambda_scale: float) -> dict[str, float]:
    pipeline = WalkForwardPipeline(
        half_life=365.0,
        lambda_scale=lambda_scale
    )
    data_pipeline = DataPipeline(csv_path="results.csv")
    metrics = pipeline.run(data_pipeline)
    return metrics.get('rps_by_match', {})


def main():
    LAMBDA_VIEJO = 0.23
    LAMBDA_NUEVO = 0.2517  # beta1_top: fit restringido a Elo>=1900 ambos lados

    print(f"Evaluando con LAMBDA_SCALE = {LAMBDA_VIEJO} (Viejo)...")
    rps_old = evaluate_lambda(LAMBDA_VIEJO)

    print(f"Evaluando con LAMBDA_SCALE = {LAMBDA_NUEVO} (beta1_top)...")
    rps_new = evaluate_lambda(LAMBDA_NUEVO)

    fechas_comunes = sorted(set(rps_old) & set(rps_new))
    if not fechas_comunes:
        print("No hay fechas comunes para comparar.")
        return

    vec_old = np.array([rps_old[f] for f in fechas_comunes])
    vec_new = np.array([rps_new[f] for f in fechas_comunes])

    if np.array_equal(vec_old, vec_new):
        raise RuntimeError(
            "rps_old y rps_new son idénticos: LAMBDA_SCALE no tuvo "
            "ningún efecto entre las dos corridas. Verificá qué atributo lee "
            "expected_goal_ratio antes de confiar en cualquier RPS de este script."
        )

    rps_mean_old = vec_old.mean()
    rps_mean_new = vec_new.mean()

    print(f"\nResultados (Dataset Global):")
    print(f"RPS Medio (Viejo {LAMBDA_VIEJO}): {rps_mean_old:.4f}")
    print(f"RPS Medio (Nuevo {LAMBDA_NUEVO}): {rps_mean_new:.4f}")

    diferencia = rps_mean_old - rps_mean_new  # positivo = el nuevo es mejor
    print(f"Mejora (Viejo - Nuevo): {diferencia:.4f}")
    
    print("\nCalculando bootstrap pareado (B=1000)...")
    res = bootstrap_paired_rps(rps_old, rps_new)

    print(f"Intervalo de confianza 95% para la diferencia de RPS: "
          f"[{res['ci95_lo']:.5f}, {res['ci95_hi']:.5f}]")
    print(res['veredicto'])


if __name__ == "__main__":
    main()
