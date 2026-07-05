"""
risk_core.py
============
Refuerzo matemático de la gestión de riesgo (paper_trader.py, Kelly Criterion)
del sistema FUOL.

FRENTE 3 — TRADING, KELLY Y GESTIÓN DE RIESGO
------------------------------------------------
El resumen indica "Kelly Fraccional modulado por Brier Score histórico".
Esto es un buen punto de partida, pero el Kelly clásico tiene 3 supuestos
que rara vez se cumplen en apuestas deportivas reales, y que si se ignoran,
producen sobre-apuesta sistemática incluso con la fracción de Kelly
reducida:

  1) Asume que la probabilidad estimada p es EXACTA (sin incertidumbre
     de estimación). En la práctica, p viene de un modelo (Dixon-Coles +
     Bayesiano + Elo) con su propio error, y ese error debe descontarse
     del tamaño de la apuesta (Kelly-con-incertidumbre / "Kelly seguro").
  2) Asume apuestas independientes. Múltiples partidos de la MISMA
     jornada/liga NO son independientes si comparten un factor de riesgo
     común (ej. el modelo tiene un sesgo sistemático esa semana, clima
     regional, fatiga por calendario de Champions). Ignorar la
     correlación infla el Kelly total simultáneo.
  3) No tiene mecanismo explícito de control de drawdown máximo. Kelly
     puro maximiza el crecimiento logarítmico esperado a largo plazo,
     pero tiene varianza de camino (path variance) que puede ser
     inaceptable para un bankroll real incluso siendo "matemáticamente
     óptimo" en el límite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats, optimize


# ============================================================================
# 3.1 — KELLY CLÁSICO (baseline, para referencia y tests de regresión)
# ============================================================================

def kelly_fraction_classic(prob_win: float, decimal_odds: float) -> float:
    """
    Fórmula clásica de Kelly (Kelly, 1956):
        f* = (p * b - q) / b = p - q/b
    donde b = decimal_odds - 1 (ganancia neta por unidad apostada),
    q = 1 - p.

    Puede ser negativo (no apostar) o mayor a 1 (apalancamiento, que en
    apuestas deportivas no aplica: se trunca a [0, 1] en la práctica).
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob_win
    f_star = (prob_win * b - q) / b
    return max(f_star, 0.0)


# ============================================================================
# 3.2 — KELLY ROBUSTO A INCERTIDUMBRE DE PROBABILIDAD (Kelly-Bayes)
# ============================================================================
#
# Formulación: en vez de tratar p como un escalar fijo, se modela p como
# una variable aleatoria con distribución posterior (heredada directamente
# de GammaPoissonPosterior / UncertainEloSystem del módulo statistical_core).
# El tamaño de apuesta óptimo bajo incertidumbre de parámetro NO es
# f*(E[p]), sino la fracción que maximiza el crecimiento logarítmico
# ESPERADO SOBRE LA DISTRIBUCIÓN POSTERIOR de p (Baker & McHale, 2013;
# formulación de "Kelly con parámetro incierto" vía expansión de Taylor
# de segundo orden, equivalente a un shrinkage explícito y derivable,
# a diferencia de la modulación por Brier histórico que es una heurística
# post-hoc sin garantía de optimalidad).

def kelly_fraction_uncertainty_adjusted(
    prob_win_mean: float,
    prob_win_variance: float,
    decimal_odds: float,
    risk_aversion_lambda: float = 1.0,
) -> dict:
    """
    Expansión de segundo orden del crecimiento logarítmico esperado
    E[log(1 + f*R)] respecto a la incertidumbre de p, dando:

        f*_ajustado = f*_clasico(E[p]) - risk_aversion_lambda * Var(p) * C(b)

    donde C(b) es un factor de curvatura que depende de las odds (b),
    derivado de la segunda derivada de la función de crecimiento respecto
    a p. Esto castiga MATEMÁTICAMENTE (no heurísticamente) apuestas donde
    el modelo tiene alta incertidumbre epistémica, incluso si p_mean
    sugiere una ventaja aparente grande.

    risk_aversion_lambda=1.0 es el ajuste "puro" derivado de la expansión;
    valores >1 son más conservadores (útil si se quiere combinar con el
    Brier-Score-modulation existente sin doble-contar el mismo efecto:
    ver nota de integración al final del archivo).
    """
    f_star_point = kelly_fraction_classic(prob_win_mean, decimal_odds)
    if f_star_point <= 0:
        return {
            "f_star_classic": 0.0,
            "f_star_uncertainty_adjusted": 0.0,
            "uncertainty_penalty": 0.0,
            "prob_win_std": math.sqrt(prob_win_variance),
        }

    b = decimal_odds - 1.0
    q_mean = 1.0 - prob_win_mean

    # Curvatura de la log-utilidad de Kelly respecto a p, evaluada en el
    # punto (p_mean, f_star_point): d^2/dp^2 [E log(1+fR)] derivado
    # analíticamente para el caso binario ganar/perder.
    win_term = (b / (1 + f_star_point * b)) ** 2
    lose_term = (1.0 / (1 - f_star_point)) ** 2 if f_star_point < 1 else 0.0
    curvature = prob_win_mean * win_term + q_mean * lose_term

    penalty = risk_aversion_lambda * prob_win_variance * curvature * 0.5
    f_star_adjusted = max(f_star_point - penalty, 0.0)

    return {
        "f_star_classic": f_star_point,
        "f_star_uncertainty_adjusted": f_star_adjusted,
        "uncertainty_penalty": penalty,
        "prob_win_std": math.sqrt(prob_win_variance),
    }


# ============================================================================
# 3.3 — KELLY MULTI-ACTIVO CON CORRELACIÓN (matriz de covarianza)
# ============================================================================
#
# Generaliza el Kelly escalar a un vector de apuestas simultáneas
# (ej. 8 partidos de la misma jornada), usando la formulación de Kelly
# multivariado clásica de portfolio theory (Thorp, 1969; MacLean, Thorp
# & Ziemba, 2010): el vector óptimo de fracciones es
#
#       f* = Sigma^{-1} * mu
#
# donde mu es el vector de "edge" esperado por apuesta y Sigma es la
# matriz de covarianza de los retornos. Ignorar Sigma (tratar cada
# apuesta independientemente, sumando fracciones escalares) SISTEMÁTICAMENTE
# sobre-apuesta cuando hay correlación positiva entre partidos (factor de
# riesgo común), que es la situación típica de una misma jornada/liga.

def kelly_multivariate_correlated(
    edge_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    max_total_exposure: float = 1.0,
) -> dict:
    """
    edge_vector: mu_i = p_i * b_i - q_i para cada apuesta i (edge esperado
        por unidad, análogo al numerador de Kelly escalar).
    covariance_matrix: Sigma_ij = correlación entre outcomes de apuesta i,j
        multiplicada por sus varianzas individuales. Una correlación
        razonable a estimar empíricamente: partidos de la misma liga en
        la misma jornada suelen tener correlación positiva de 0.05-0.20
        en el error del modelo (sesgo sistemático compartido), no cero.

    Retorna el vector de fracciones óptimas, más el chequeo de exposición
    total: si sum(f*) excede max_total_exposure, se escala proporcionalmente
    (mismo mecanismo que "position sizing" con budget constraint en
    portfolio management clásico).
    """
    try:
        sigma_inv = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        # Matriz singular (posible con partidos perfectamente correlacionados
        # o edge_vector degenerado): usar pseudo-inversa como fallback seguro
        sigma_inv = np.linalg.pinv(covariance_matrix)

    f_star_vector = sigma_inv @ edge_vector
    f_star_vector = np.clip(f_star_vector, 0, None)  # no apuestas negativas (no hay "short" en este contexto)

    total_exposure = f_star_vector.sum()
    scaling_applied = False
    if total_exposure > max_total_exposure:
        f_star_vector = f_star_vector * (max_total_exposure / total_exposure)
        scaling_applied = True

    # Varianza total del portfolio de apuestas simultáneas, para reporting
    portfolio_variance = f_star_vector @ covariance_matrix @ f_star_vector

    return {
        "f_star_vector": f_star_vector,
        "total_exposure": f_star_vector.sum(),
        "scaling_applied": scaling_applied,
        "portfolio_variance": float(portfolio_variance),
        "portfolio_std": float(math.sqrt(max(portfolio_variance, 0))),
    }


def estimate_correlation_same_matchday(
    residuals_by_match: dict,
) -> np.ndarray:
    """
    Estima la matriz de covarianza empírica de residuales del modelo
    (p_predicha - resultado_real) entre partidos de la misma jornada,
    a partir del historial de production_logger.py / performance_tracker.py.

    residuals_by_match: {match_id: array_de_residuales_historicos_del_mismo_slot}

    Esto operacionaliza "correlación entre apuestas" con datos reales del
    propio sistema, en vez de asumir un valor arbitrario.
    """
    match_ids = list(residuals_by_match.keys())
    n = len(match_ids)
    matrix = np.zeros((n, n))

    lengths = [len(v) for v in residuals_by_match.values()]
    min_len = min(lengths) if lengths else 0
    if min_len < 2:
        # Sin suficiente historia: fallback a matriz diagonal (independencia),
        # el supuesto conservador cuando no hay evidencia de correlación
        return np.eye(n)

    data_matrix = np.array([
        residuals_by_match[mid][:min_len] for mid in match_ids
    ])
    return np.cov(data_matrix)


# ============================================================================
# 3.4 — CONTROL DE DRAWDOWN: KELLY FRACCIONAL ÓPTIMO BAJO RESTRICCIÓN
# ============================================================================
#
# Kelly puro maximiza crecimiento logarítmico esperado, pero la probabilidad
# de un drawdown severo en el camino (antes de alcanzar el óptimo asintótico)
# puede ser alta. Se deriva la fracción de Kelly (multiplicador c en
# f_usado = c * f_kelly) que mantiene P(drawdown > D) por debajo de un
# umbral aceptable, usando la fórmula analítica estándar de teoría de
# apuestas (ver: MacLean, Sanegre, Zhao & Ziemba, 2004 — "Never bet more
# than a quarter of your Kelly").

def max_drawdown_probability(kelly_multiplier: float, target_drawdown: float,
                              n_bets_horizon: int, win_rate: float,
                              avg_edge: float) -> float:
    """
    Aproximación de la probabilidad de alcanzar un drawdown de al menos
    `target_drawdown` (ej. 0.20 = 20% del bankroll) en un horizonte de
    n_bets_horizon apuestas, usando la fórmula clásica:

        P(drawdown >= D) ≈ ((1 - c*(2-c)) )^(algo)  [aprox. vía difusión]

    En su forma operacional (aproximación de random walk con drift,
    válida para c <= 1 y n grande):

        P(DD >= D) ≈ exp( -2 * D * drift / (c^2 * variance_per_bet) )

    donde drift y variance_per_bet se derivan del edge promedio y c es el
    multiplicador fraccional de Kelly (c=1: Kelly completo, c=0.5: half-Kelly).

    Esta función permite resolver INVERSAMENTE qué fracción de Kelly usar
    para no exceder una probabilidad de drawdown objetivo — en vez de
    elegir "medio Kelly" arbitrariamente, como es común en la práctica.
    """
    drift = avg_edge * kelly_multiplier
    variance_per_bet = win_rate * (1 - win_rate) * (kelly_multiplier ** 2)
    if variance_per_bet <= 0 or drift <= 0:
        return 1.0  # sin edge positivo, el drawdown es prácticamente seguro

    exponent = -2.0 * target_drawdown * drift / variance_per_bet
    return float(np.clip(math.exp(exponent), 0.0, 1.0))


def solve_kelly_multiplier_for_drawdown_budget(
    target_drawdown: float,
    max_drawdown_probability_allowed: float,
    n_bets_horizon: int,
    win_rate: float,
    avg_edge: float,
) -> float:
    """
    Resuelve c* (multiplicador fraccional, ej. 0.5 = "half Kelly") tal que
    P(drawdown >= target_drawdown) == max_drawdown_probability_allowed,
    vía búsqueda de raíz sobre max_drawdown_probability.

    Ejemplo de uso concreto para paper_trader.py: en vez de fijar un
    "Kelly fraccional" arbitrario (ej. 0.25 fijo), se calcula el
    multiplicador que garantiza, por ejemplo, "no más de 5% de probabilidad
    de perder 20% del bankroll en los próximos 500 partidos" — una
    restricción de riesgo explícita y auditable en vez de un número mágico.
    """
    def objective(c):
        return (max_drawdown_probability(c, target_drawdown, n_bets_horizon,
                                          win_rate, avg_edge)
                - max_drawdown_probability_allowed)

    try:
        result = optimize.brentq(objective, 1e-6, 2.0, xtol=1e-4)
        return float(np.clip(result, 0.0, 1.0))
    except ValueError:
        # objective no cambia de signo en el rango: devolver el extremo
        # conservador (nunca el agresivo) para no exponer capital de más
        return 0.1


# ============================================================================
# 3.5 — INTEGRACIÓN: PIPELINE COMPLETO DE SIZING PARA paper_trader.py
# ============================================================================

@dataclass
class BetSizingRecommendation:
    match_id: str
    f_star_uncertainty_adjusted: float
    f_star_after_correlation: float
    f_star_after_drawdown_control: float
    final_recommended_fraction: float
    warnings: list


def full_risk_adjusted_kelly_pipeline(
    prob_win_mean: float,
    prob_win_variance: float,
    decimal_odds: float,
    correlation_scaling_factor: float,
    drawdown_kelly_multiplier: float,
    brier_score_recent: float,
    match_id: str = "unknown",
) -> BetSizingRecommendation:
    """
    Pipeline de referencia que compone los 3 ajustes en la SECUENCIA
    correcta (el orden importa: aplicar drawdown control antes que
    incertidumbre doble-cuenta la penalización):

    1) Kelly base ajustado por incertidumbre de p (3.2)
    2) Escalado por exposición correlacionada de la jornada (3.3)
    3) Escalado final por presupuesto de drawdown (3.4)
    4) Nota de integración con Brier histórico existente: si
       brier_score_recent ya está capturando parte de la incertidumbre
       epistémica (porque el modelo falló recientemente), reducir
       risk_aversion_lambda en el paso 1 para NO penalizar dos veces la
       misma fuente de error (Brier histórico = error retrospectivo
       realizado; Var(p) = error prospectivo de estimación — se solapan
       parcialmente y NO deben sumarse ingenuamente).
    """
    warnings = []

    # Ajuste dinámico de risk_aversion_lambda: si Brier reciente ya es alto
    # (modelo fallando), reducir el peso adicional de Var(p) para no
    # castigar dos veces la misma señal de mala calibración.
    baseline_brier = 0.25  # Brier de un modelo sin skill (predicción constante 0.5)
    if brier_score_recent > baseline_brier:
        warnings.append(
            f"Brier reciente ({brier_score_recent:.3f}) peor que baseline "
            f"sin-skill ({baseline_brier}). El modelo puede estar mal "
            f"calibrado en el régimen actual; considerar pausar sizing "
            f"hasta re-calibración."
        )
        risk_aversion_lambda = 0.5  # reducir doble penalización
    else:
        risk_aversion_lambda = 1.0

    step1 = kelly_fraction_uncertainty_adjusted(
        prob_win_mean, prob_win_variance, decimal_odds, risk_aversion_lambda
    )
    f1 = step1["f_star_uncertainty_adjusted"]

    f2 = f1 * correlation_scaling_factor
    if correlation_scaling_factor < 0.5:
        warnings.append(
            f"Alta correlación detectada con otras apuestas simultáneas "
            f"(factor de escalado {correlation_scaling_factor:.2f}). "
            f"Exposición individual reducida significativamente."
        )

    f3 = f2 * drawdown_kelly_multiplier

    return BetSizingRecommendation(
        match_id=match_id,
        f_star_uncertainty_adjusted=f1,
        f_star_after_correlation=f2,
        f_star_after_drawdown_control=f3,
        final_recommended_fraction=max(f3, 0.0),
        warnings=warnings,
    )