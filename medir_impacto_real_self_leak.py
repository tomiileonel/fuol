"""
El cálculo de "13 self-leaks * ~0.23 de ahorro / 1063 = 0.0028" ASUME que
el motor, bajo el filtro viejo (<=), obtuvo RPS≈0 en esos 13 folds. Esa
suposición no está medida -- se dedujo teóricamente del mecanismo del
leak, pero el prior Bayesiano (prior_strength=6) no colapsa a certeza
absoluta con un solo partido filtrado, y Dixon-Coles nunca asigna
probabilidad 1.0 a un marcador exacto. El leak sesga la predicción hacia
el resultado real, pero "sesga" no es "iguala".

Este script corre el walk-forward DOS VECES -- con train_b filtrado por
<= (el bug original) y por < (el fix) -- y extrae el RPS real de
CADA UNO de los 13 folds de self-leak confirmados, en ambas versiones.
Así reemplazamos la estimación teórica por una medición directa.

Requiere que unified_engine.py tenga run_walkforward parametrizable con
un filtro de train_b inyectable. Como el fix ya está aplicado en el
motor (usa '<'), este script re-implementa localmente la versión vieja
('<=') SOLO para esta comparación diagnóstica -- no para revertir nada
en producción.
"""
from datetime import datetime

from fase1_baseline import cargar_historial
from unified_engine import UnifiedEngine, WalkForwardBacktester

FECHAS_SELF_LEAK_CONFIRMADAS = [
    "1930-07-15", "1965-06-03", "1971-01-08", "1971-01-13", "1972-06-25",
    "1974-05-18", "1977-06-26", "1978-06-06", "1986-03-26", "2007-02-07",
    "2009-02-11", "2018-06-30", "2022-12-18"
]


def correr_walkforward_con_filtro(matches_a, matches_b, team_a, team_b, venue,
                                   half_life, filtro_inclusive: bool) -> dict:
    """
    Reimplementación mínima de run_walkforward, parametrizando solo el
    operador de corte en train_b, para aislar el efecto del fix sin
    tocar unified_engine.py. filtro_inclusive=True reproduce el bug
    original (<=); False reproduce el fix actual (<).
    """
    sorted_a = sorted(matches_a, key=lambda m: m.get('date', '1970-01-01'))
    bt = WalkForwardBacktester(min_train_size=10)

    resultados_por_fecha = {}

    for k in range(bt.min_train_size, len(sorted_a)):
        train_a = sorted_a[:k]
        test_m = sorted_a[k]
        test_date = test_m.get('date', '9999-99-99')

        if filtro_inclusive:
            train_b = [m for m in matches_b if m.get('date', '9999-99-99') <= test_date]
        else:
            train_b = [m for m in matches_b if m.get('date', '9999-99-99') < test_date]

        if len(train_b) < 3:
            continue

        try:
            engine = UnifiedEngine(
                team_a=team_a, team_b=team_b,
                matches_a=train_a, matches_b=train_b,
                venue=venue, half_life=half_life, optimize_rho=True,
            )
            pred = engine.predict()
            metrics = bt.evaluate_match(int(test_m.get('gf', 0)), int(test_m.get('gc', 0)), pred)
            resultados_por_fecha[test_date] = metrics['rps']
        except Exception as e:
            print(f"[diagnóstico] Error en fold {test_date}: {e}")
            continue

    return resultados_por_fecha


def main():
    if not FECHAS_SELF_LEAK_CONFIRMADAS:
        print("❌ Completá FECHAS_SELF_LEAK_CONFIRMADAS con las 13 fechas")
        print("   confirmadas por confirmar_self_leak_por_rival.py antes de correr esto.")
        return

    csv_path = "results.csv"
    team_a, team_b = "ARGENTINA", "FRANCE"
    matches_a = cargar_historial(csv_path, team_a)
    matches_b = cargar_historial(csv_path, team_b)

    print("Corriendo walk-forward con filtro VIEJO (<=, reproduce el bug)...")
    rps_viejo = correr_walkforward_con_filtro(
        matches_a, matches_b, team_a, team_b, venue="N",
        half_life=365, filtro_inclusive=True,
    )

    print("Corriendo walk-forward con filtro NUEVO (<, el fix actual)...")
    rps_nuevo = correr_walkforward_con_filtro(
        matches_a, matches_b, team_a, team_b, venue="N",
        half_life=365, filtro_inclusive=False,
    )

    print(f"\n{'='*70}")
    print("RPS real por fold de self-leak, filtro viejo vs nuevo:")
    print(f"{'fecha':<12} {'RPS (viejo <=)':>16} {'RPS (nuevo <)':>16} {'delta':>10}")

    suma_delta = 0.0
    n_medidos = 0
    for fecha in FECHAS_SELF_LEAK_CONFIRMADAS:
        rv = rps_viejo.get(fecha)
        rn = rps_nuevo.get(fecha)
        if rv is None or rn is None:
            print(f"{fecha:<12} {'(fold no evaluado -- ver por qué)':>44}")
            continue
        delta = rv - rn  # positivo = el viejo tenía RPS más bajo (mejor) que el nuevo
        suma_delta += delta
        n_medidos += 1
        print(f"{fecha:<12} {rv:>16.4f} {rn:>16.4f} {delta:>10.4f}")

    if n_medidos == 0:
        print("\n❌ Ningún fold de la lista se pudo medir en ambas corridas.")
        return

    impacto_medido_agregado = suma_delta / len(rps_nuevo)
    print(f"\nFolds medidos: {n_medidos} de {len(FECHAS_SELF_LEAK_CONFIRMADAS)}")
    print(f"Delta promedio en esos folds (RPS_viejo - RPS_nuevo): {suma_delta / n_medidos:.4f}")
    print(f"Impacto en el RPS AGREGADO (delta_total / n_total_evaluado): {impacto_medido_agregado:.4f}")
    print(f"\nComparar esto contra la estimación teórica de 0.0018-0.0028.")
    print("Si el impacto medido es sustancialmente menor, el leak de Tipo B")
    print("(173 folds restantes) o algún otro factor está aportando más de lo")
    print("estimado -- no cierres la causalidad solo con la aritmética teórica.")


if __name__ == "__main__":
    main()
