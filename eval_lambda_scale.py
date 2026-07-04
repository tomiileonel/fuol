import numpy as np
from unified_engine import EloRating, WalkForwardBacktester
from fase1_baseline import cargar_historial
from test_significancia import bootstrap_paired_rps

team_a, team_b = "ARGENTINA", "FRANCE"
all_matches_a = cargar_historial("results.csv", team_a)
all_matches_b = cargar_historial("results.csv", team_b)
venue = "N"


def evaluate_lambda(lambda_scale: float) -> dict[str, float]:
    # LAMBDA_SCALE se lee en EloRating.expected_goal_ratio (self.LAMBDA_SCALE),
    # NO en UnifiedEngine. Parchear UnifiedEngine.LAMBDA_SCALE crea un
    # atributo de clase que nadie lee: ambas corridas terminarían usando
    # silenciosamente el 0.23 de EloRating, sin importar qué se le pase acá.
    original = EloRating.LAMBDA_SCALE
    try:
        EloRating.LAMBDA_SCALE = lambda_scale
        backtester = WalkForwardBacktester()
        metrics = backtester.run_walkforward(
            team_a, team_b, all_matches_a, all_matches_b,
            venue=venue, half_life=365, eval_start_idx=0,
        )
        return metrics['rps_by_match']
    finally:
        # Restaurar SIEMPRE, incluso si run_walkforward lanza -- de lo
        # contrario el valor parcheado se filtra a cualquier código que
        # importe EloRating después en el mismo proceso (tests, notebook, etc).
        EloRating.LAMBDA_SCALE = original


def main():
    LAMBDA_VIEJO = 0.23
    LAMBDA_NUEVO = 0.2517  # beta1_top: fit restringido a Elo>=1900 ambos lados

    print(f"Evaluando con LAMBDA_SCALE = {LAMBDA_VIEJO} (Viejo)...")
    rps_old = evaluate_lambda(LAMBDA_VIEJO)

    print(f"Evaluando con LAMBDA_SCALE = {LAMBDA_NUEVO} (beta1_top)...")
    rps_new = evaluate_lambda(LAMBDA_NUEVO)

    # Guardia adicional: si por algún motivo futuro LAMBDA_SCALE dejara de
    # tener efecto (otro desacople de atributo), rps_old y rps_new saldrían
    # bit-a-bit idénticos. Eso ya nos pasó una vez en este mismo script --
    # que quede detectado en código, no en una relectura manual del output.
    fechas_comunes = sorted(set(rps_old) & set(rps_new))
    if not fechas_comunes:
        print("No hay fechas comunes para comparar.")
        return

    vec_old = np.array([rps_old[f] for f in fechas_comunes])
    vec_new = np.array([rps_new[f] for f in fechas_comunes])

    if np.array_equal(vec_old, vec_new):
        raise RuntimeError(
            "rps_old y rps_new son bit-a-bit idénticos: LAMBDA_SCALE no tuvo "
            "ningún efecto entre las dos corridas. Esto ya pasó antes por "
            "parchear la clase equivocada -- no asumas que 'da distinto' sin "
            "chequearlo explícitamente. Verificá qué atributo lee "
            "expected_goal_ratio antes de confiar en cualquier RPS de este script."
        )

    rps_mean_old = vec_old.mean()
    rps_mean_new = vec_new.mean()

    print(f"\nResultados (N = {len(fechas_comunes)} partidos pareados):")
    print(f"RPS Medio (Viejo {LAMBDA_VIEJO}): {rps_mean_old:.4f}")
    print(f"RPS Medio (Nuevo {LAMBDA_NUEVO}): {rps_mean_new:.4f}")

    diferencia = rps_mean_old - rps_mean_new  # positivo = el nuevo es mejor
    print(f"Mejora (Viejo - Nuevo): {diferencia:.4f}")

    print("\nCalculando bootstrap pareado (B=1000)...")
    res = bootstrap_paired_rps(rps_old, rps_new)

    # bootstrap_paired_rps devuelve 'ci95_lo'/'ci95_hi', no 'ci_95'.
    print(f"Intervalo de confianza 95% para la diferencia de RPS: "
          f"[{res['ci95_lo']:.5f}, {res['ci95_hi']:.5f}]")
    print(res['veredicto'])


if __name__ == "__main__":
    main()
