"""
Calibración de LAMBDA_SCALE por regresión Poisson.

LAMBDA_SCALE=0.23 en unified_engine.EloRating es una constante puesta a
mano (ver comentario "idealmente debe ajustarse por regresión" en el
propio código fuente). Este script la reemplaza por un coeficiente
ajustado sobre el dataset histórico completo.

Modelo:
    gf ~ Poisson(lambda),  log(lambda) = beta0 + beta1 * logit(expected_score_home)
    gc ~ Poisson(mu),      log(mu)     = gamma0 + gamma1 * logit(expected_score_away)

donde expected_score(...) es la MISMA función que corre en producción
(EloRating.expected_score, con HOME_ADV ya aplicado según venue), y el
regresor es su logit tras el mismo clip [0.05, 0.95] que usa
EloRating.expected_goal_ratio. Esto es deliberado y no cosmético:
expected_score usa base 10 (10**(-diff/400)), así que
logit(expected_score(diff)) = diff * ln(10)/400, NO diff/400. Ajustar
sobre diff/400 directo produciría un beta1 desalineado ~2.3x respecto al
espacio en que LAMBDA_SCALE realmente se multiplica en producción.

beta1 debería converger a valores cercanos entre la perspectiva local y
la visitante, si el modelo está bien especificado (misma relación
Elo->goles en ataque propio y en goles concedidos por el rival). Si
divergen de forma notoria, el script lo reporta explícitamente en lugar
de promediar en silencio.

Requiere: statsmodels, y un CSV con el formato usado por
DataPipeline (date, home_team, away_team, home_score,
away_score, tournament, neutral).

Uso:
    python calibrate_lambda.py results.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime

import numpy as np
import statsmodels.api as sm

from unified_engine import EloRating, ELO_INITIAL


def cargar_partidos_globales(csv_path: str) -> list[dict]:
    """
    A diferencia de DataPipeline.prepare_data, esto carga TODOS los partidos del CSV, porque la calibración
    de LAMBDA_SCALE necesita reconstruir el Elo de cada selección que
    aparece en el dataset, no solo de dos equipos.
    """
    partidos = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["home_score"] or not row["away_score"]:
                continue
            try:
                h_score = int(row["home_score"])
                a_score = int(row["away_score"])
            except ValueError:
                continue

            partidos.append({
                "date": row["date"],
                "home_team": row["home_team"].strip().upper(),
                "away_team": row["away_team"].strip().upper(),
                "home_score": h_score,
                "away_score": a_score,
                "competition": row["tournament"],
                "neutral": row["neutral"].strip().upper() == "TRUE",
            })

    partidos.sort(key=lambda m: datetime.fromisoformat(m["date"]))
    return partidos


def construir_dataset_elo_vs_goles(partidos: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorre todos los partidos cronológicamente manteniendo un EloRating
    vivo por selección. Por cada partido, registra:
      - logit_e: logit(expected_score(elo_diff)) tras el mismo clip [0.05,0.95]
        que usa EloRating.expected_goal_ratio en producción, PRE-partido
      - goles observados por cada lado

    CRÍTICO: el regresor debe ser exactamente logit(expected_score(...)),
    no elo_diff/400. expected_score() usa base 10 (10**(-diff/400)), así
    que logit(expected_score(diff)) = diff * ln(10) / 400, NO diff/400 --
    difieren por un factor ln(10)≈2.303. Si el fit se hace sobre diff/400
    y el coeficiente resultante se sustituye directo en LAMBDA_SCALE (que
    multiplica a logit(E), no a diff/400), el escalar queda ~2.3x-3.3x
    desalineado respecto al espacio en que expected_goal_ratio lo consume.
    Ajustar aquí exactamente lo que el modelo de producción evalúa evita
    ese desacople dimensional por construcción, no por convención.

    Retorna (logits, goles, es_de_local).
    """
    elos: dict[str, EloRating] = {}

    def get_elo(team: str) -> EloRating:
        if team not in elos:
            elos[team] = EloRating(team, initial_elo=ELO_INITIAL.get(team, ELO_INITIAL["DEFAULT"]))
        return elos[team]

    logits, goles, es_local = [], [], []

    for m in partidos:
        home, away = m["home_team"], m["away_team"]
        elo_home, elo_away = get_elo(home), get_elo(away)

        venue_home = "N" if m["neutral"] else "H"
        venue_away = "N" if m["neutral"] else "A"

        # expected_score() ya aplica HOME_ADV internamente según venue,
        # y ya está en base logística (0,1) -- usamos la MISMA función
        # que corre en producción, no una reconstrucción manual del diff.
        E_home = elo_home.expected_score(elo_away.rating, venue_home)
        E_away = elo_away.expected_score(elo_home.rating, venue_away)
        E_home_c = np.clip(E_home, 0.05, 0.95)
        E_away_c = np.clip(E_away, 0.05, 0.95)

        logits.append(np.log(E_home_c / (1.0 - E_home_c)))
        goles.append(m["home_score"])
        es_local.append(1)

        logits.append(np.log(E_away_c / (1.0 - E_away_c)))
        goles.append(m["away_score"])
        es_local.append(0)

        # Actualizar Elo DESPUÉS de registrar el snapshot pre-partido.
        match_home = {"gf": m["home_score"], "gc": m["away_score"],
                      "competition": m["competition"], "venue": venue_home}
        match_away = {"gf": m["away_score"], "gc": m["home_score"],
                      "competition": m["competition"], "venue": venue_away}

        new_elo_home = elo_home.update(match_home, elo_away.rating)
        elo_away.update(match_away, elo_home.rating)
        elo_home.rating = new_elo_home  # ya asignado por .update(), explícito por claridad

    return np.array(logits), np.array(goles), np.array(es_local)


def fit_poisson_lambda_scale(logits: np.ndarray, goles: np.ndarray) -> tuple[float, float, object]:
    """
    log(goles_esperados) = beta0 + beta1 * logit(expected_score)
    beta1 es el reemplazo calibrado de LAMBDA_SCALE, en el MISMO espacio
    que expected_goal_ratio lo consume (multiplica a logit(E), no a
    elo_diff/400 crudo).
    """
    X = sm.add_constant(logits)
    modelo = sm.GLM(goles, X, family=sm.families.Poisson()).fit()
    beta0, beta1 = modelo.params
    return float(beta0), float(beta1), modelo


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

    print(f"Cargando partidos desde {csv_path}...")
    partidos = cargar_partidos_globales(csv_path)
    print(f"Total de partidos con marcador válido: {len(partidos)}")

    print("Reconstruyendo Elo secuencial y armando dataset logit(E) -> goles...")
    logits, goles, es_local = construir_dataset_elo_vs_goles(partidos)
    print(f"Observaciones (2 por partido, local y visitante): {len(logits)}")

    beta0, beta1, modelo = fit_poisson_lambda_scale(logits, goles)

    print("\n=== Regresión Poisson: log(goles) ~ logit(expected_score) ===")
    print(modelo.summary())

    print(f"\nLAMBDA_SCALE calibrado (beta1): {beta1:.5f}")
    print(f"Valor actual hardcodeado en unified_engine.py: 0.23")
    print(f"Diferencia relativa: {abs(beta1 - 0.23) / 0.23 * 100:.1f}%")

    # exp(beta0) es el análogo fiteado de AVG_GOALS_WC_HISTORICAL (1.32).
    # Si difieren de forma notoria, sustituir SOLO beta1 en LAMBDA_SCALE y
    # dejar AVG_GOALS_WC_HISTORICAL=1.32 sin tocar mezcla una escala base
    # calibrada contra Mundiales 2010-2022 con un exponente calibrado
    # contra el dataset completo de Kaggle (partidos de todo tipo). El
    # exponente y la escala base deben salir de la MISMA fuente de datos
    # o el resultado combinado no es coherente con ninguno de los dos.
    escala_base_fiteada = float(np.exp(beta0))
    print(f"\nexp(beta0) [escala base implícita del fit]: {escala_base_fiteada:.4f}")
    print(f"AVG_GOALS_WC_HISTORICAL actual (unified_engine.py): 1.32")
    diff_escala_pct = abs(escala_base_fiteada - 1.32) / 1.32 * 100
    if diff_escala_pct > 15:
        print(f"⚠️  Difieren {diff_escala_pct:.1f}%. Sustituir solo beta1 y dejar")
        print("   AVG_GOALS_WC_HISTORICAL=1.32 sin tocar es inconsistente: ese 1.32")
        print("   fue calibrado contra Mundiales 2010-2022 (partidos de máxima")
        print("   competitividad), y el dataset de Kaggle incluye amistosos y")
        print("   clasificatorios con promedio de goles distinto. Reemplazar uno")
        print("   sin el otro deja el modelo con una escala base y un exponente")
        print("   estimados sobre poblaciones distintas.")
        print("   Opciones: (a) recalibrar AVG_GOALS_WC_HISTORICAL también desde")
        print("   este mismo fit (usar escala_base_fiteada), o (b) re-fittear")
        print("   beta1 filtrando el dataset a partidos de competición oficial")
        print("   equivalente a Mundial, para que ambas constantes vengan de la")
        print("   misma población de partidos.")
    else:
        print(f"✅ Escala base consistente (diff {diff_escala_pct:.1f}%) -- sustituir")
        print("   solo beta1 en LAMBDA_SCALE es razonable en este caso.")

    # Chequeo de consistencia: ajustar también solo con el subconjunto
    # "local" y solo "visitante" por separado. Si beta1 difiere mucho
    # entre ambos, el escalar único que usa EloRating.expected_goal_ratio
    # es una simplificación que pierde información direccional real.
    beta1_home = fit_poisson_lambda_scale(logits[es_local == 1], goles[es_local == 1])[1]
    beta1_away = fit_poisson_lambda_scale(logits[es_local == 0], goles[es_local == 0])[1]
    print(f"\nbeta1 solo-local:     {beta1_home:.5f}")
    print(f"beta1 solo-visitante: {beta1_away:.5f}")

    divergencia_pct = abs(beta1_home - beta1_away) / max(abs(beta1_home), abs(beta1_away), 1e-9) * 100
    if divergencia_pct > 20:
        print(f"\n⚠️  beta1 local y visitante difieren {divergencia_pct:.1f}%.")
        print("   Un solo LAMBDA_SCALE global puede estar promediando dos")
        print("   relaciones Elo->goles distintas. Considerá si HOME_ADV=100")
        print("   está bien calibrado, o si el modelo necesita un término")
        print("   de interacción venue * elo_diff en vez de un ajuste aditivo.")
    else:
        print(f"\n✅ beta1 local/visitante consistente (diff {divergencia_pct:.1f}%).")
        print("   Un único LAMBDA_SCALE global es una simplificación razonable.")

    # -----------------------------------------------------------------
    # Chequeo de heterogeneidad por nivel de Elo: el motor en producción
    # predice partidos entre selecciones TOP (Argentina, Francia, Brasil,
    # etc.), pero este fit corrió sobre TODO el dataset de Kaggle, que
    # incluye partidos con selecciones de Elo bajo (~1600) en volumen
    # mucho mayor. Si beta1 difiere fuerte entre "ambos equipos fuertes"
    # y la población general, un único LAMBDA_SCALE global calibrado
    # sobre TODO el dataset puede estar mal ajustado específicamente
    # donde el motor opera -- que es lo único que importa en la práctica.
    # -----------------------------------------------------------------
    ELO_ALTO_UMBRAL = 1900.0  # aprox. Elo de Portugal/Bélgica hacia arriba
    elos_por_partido: dict[str, EloRating] = {}

    def _get_elo_actual(team: str) -> float:
        if team not in elos_por_partido:
            elos_por_partido[team] = EloRating(
                team, initial_elo=ELO_INITIAL.get(team, ELO_INITIAL["DEFAULT"])
            )
        return elos_por_partido[team].rating

    mask_elo_alto = []
    for m in partidos:
        elo_h, elo_a = _get_elo_actual(m["home_team"]), _get_elo_actual(m["away_team"])
        ambos_top = (elo_h >= ELO_ALTO_UMBRAL) and (elo_a >= ELO_ALTO_UMBRAL)
        mask_elo_alto.append(ambos_top)
        mask_elo_alto.append(ambos_top)  # una entrada por cada observación (local y visitante)
        # Actualizar los Elos con el resultado real para la siguiente iteración
        venue_h = "N" if m["neutral"] else "H"
        venue_a = "N" if m["neutral"] else "A"
        elos_por_partido[m["home_team"]].update(
            {"gf": m["home_score"], "gc": m["away_score"], "competition": m["competition"], "venue": venue_h},
            elo_a,
        )
        elos_por_partido[m["away_team"]].update(
            {"gf": m["away_score"], "gc": m["home_score"], "competition": m["competition"], "venue": venue_a},
            elo_h,
        )

    mask_elo_alto = np.array(mask_elo_alto[:len(logits)])
    n_top = int(mask_elo_alto.sum())
    print(f"\n--- Chequeo de heterogeneidad: partidos Elo≥{ELO_ALTO_UMBRAL:.0f} ambos lados ---")
    print(f"Observaciones en ese subconjunto: {n_top} de {len(logits)}")

    if n_top >= 200:
        beta1_top = fit_poisson_lambda_scale(logits[mask_elo_alto], goles[mask_elo_alto])[1]
        print(f"beta1 restringido a selecciones top: {beta1_top:.5f}")
        print(f"beta1 población general (todo Kaggle): {beta1:.5f}")
        diff_top_pct = abs(beta1_top - beta1) / max(abs(beta1_top), abs(beta1), 1e-9) * 100
        if diff_top_pct > 20:
            print(f"\n🛑 beta1 difiere {diff_top_pct:.1f}% entre selecciones top y la población")
            print("   general. El motor predice específicamente partidos entre")
            print("   selecciones top -- usar el beta1 de la población general lo")
            print("   descalibraría justo donde importa. Usá beta1_top, no beta1,")
            print("   como reemplazo de LAMBDA_SCALE.")
        else:
            print(f"\n✅ beta1 consistente entre selecciones top y población general")
            print(f"   (diff {diff_top_pct:.1f}%). El valor global ({beta1:.5f}) es")
            print("   representativo también para el régimen donde el motor opera.")
    else:
        print(f"⚠️  Menos de 200 observaciones con ambos Elo≥{ELO_ALTO_UMBRAL:.0f} -- no hay")
        print("   suficiente muestra para un fit restringido confiable. No se puede")
        print("   descartar heterogeneidad por nivel de Elo con este dataset.")


if __name__ == "__main__":
    main()

    print(f"\n{'='*70}")
    print("SIGUIENTE PASO OBLIGATORIO -- no lo saltees:")
    print(f"{'='*70}")
    print("Este script calibra LAMBDA_SCALE en AISLAMIENTO (regresión directa")
    print("goles ~ logit(E)). Eso no garantiza mejor RPS dentro del motor")
    print("completo: LAMBDA_SCALE interactúa con BayesianGoalRates")
    print("(prior_strength=6), con el ajuste cruzado lam_final=sqrt(lam*mu_def)")
    print("y con Dixon-Coles. Un beta1 óptimo en aislamiento puede no serlo")
    print("una vez compuesto con el resto del pipeline.")
    print("\nAntes de aceptar el reemplazo en producción:")
    print("  1. Sustituir LAMBDA_SCALE por el beta1 reportado arriba.")
    print("  2. Correr fase1_baseline.py y fase2_barrido.py de nuevo (el RPS")
    print("     baseline con LAMBDA_SCALE=0.23 ya no es comparable una vez")
    print("     que la constante cambia -- hay que regenerar el punto de")
    print("     referencia, no comparar contra el JSON viejo).")
    print("  3. Correr test_significancia.py: bootstrap pareado entre RPS con")
    print("     LAMBDA_SCALE viejo vs nuevo, sobre el walk-forward completo,")
    print("     no sobre el fit de esta regresión aislada.")
    print("  4. Solo si el CI95 de esa diferencia NO cruza cero y favorece al")
    print("     valor nuevo, reemplazar en producción.")
