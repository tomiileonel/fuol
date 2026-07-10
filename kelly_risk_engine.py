"""
kelly_risk_engine.py
=====================
Extensión matemática del motor de riesgo del sistema (paper_trader.py usa
Kelly Fraccional modulado por Brier Score histórico).

CONTEXTO
--------
El Kelly Criterion simple asume:
  (a) un evento BINARIO (gana/pierde la apuesta),
  (b) probabilidades CONOCIDAS con certeza exacta,
  (c) posibilidad de apostar en UN SOLO mercado a la vez sin interacción.

Ninguno de esos supuestos se sostiene en un sistema de trading deportivo
real:
  (a) un partido tiene 3 resultados (1X2), no 2 -- Kelly binario aplicado
      ingenuamente a cada resultado por separado ignora que son mutuamente
      excluyentes y puede sobre-apostar en conjunto.
  (b) las probabilidades vienen de un modelo estadístico con incertidumbre
      real (los IC del módulo Dixon-Coles/Bayesiano de este mismo
      conjunto de módulos son precisamente esa incertidumbre) -- Kelly con
      probabilidad puntual ignora el riesgo de que el modelo esté
      equivocado, lo cual es RUINOSO en la cola (apostar fuerte con una
      estimación de probabilidad que resulta estar sesgada).
  (c) el bankroll es finito y compartido entre partidos simultáneos.

Este módulo añade la matemática que cierra esas 3 brechas:
  1. Kelly multi-resultado (simultaneous Kelly) para mercados 1X2.
  2. Kelly robusto por incertidumbre del modelo (usa un cuantil inferior
     de la posterior en lugar de la media plug-in; ver notas abajo).
  3. Límite de posición vía Teoría de la Ruina del Jugador (Gambler's
     Ruin) para fijar un piso de bankroll que el sistema NUNCA debe
     cruzar, matemáticamente derivado (no un número arbitrario en config).

FUNDAMENTO MATEMÁTICO
----------------------
### 4.1 Kelly clásico (binario), recordatorio

Para una apuesta con cuota decimal b+1 (paga b por unidad apostada más
la unidad) y probabilidad de ganar p:

    f* = (b*p - (1-p)) / b   = (p*(b+1) - 1) / b

f* es la fracción del bankroll a apostar que maximiza el crecimiento
logarítmico esperado del capital a largo plazo (criterio de Kelly).

### 4.2 Kelly Multi-Resultado (simultaneous Kelly, 1X2)

Con 3 resultados mutuamente excluyentes {1, X, 2}, probabilidades del
modelo (p1, pX, p2) y cuotas de mercado (o1, oX, o2), el problema es
maximizar el crecimiento logarítmico esperado CONJUNTO:

    E[log(W)] = p1*log(1 + f1*(o1-1) - fX - f2)
              + pX*log(1 + fX*(oX-1) - f1 - f2)
              + p2*log(1 + f2*(o2-1) - f1 - fX)
              + (1-p1-pX-p2)*log(1 - f1 - fX - f2)   [se anula si p suman 1]

sujeto a f1, fX, f2 >= 0 y f1+fX+f2 <= 1 (no se puede apostar más del
bankroll). Esto NO tiene solución cerrada simple en general (a diferencia
del caso binario), así que se resuelve numéricamente (optimización
convexa: el problema es cóncavo en (f1,fX,f2) porque log es cóncavo y
las restricciones son lineales, así que hay óptimo global garantizado).

Esto es estrictamente mejor que aplicar Kelly binario a cada resultado
por separado y sumar: ese enfoque ingenuo puede recomendar apostar en
DOS resultados del mismo partido simultáneamente cuando no siempre es
óptimo, y no captura correctamente la restricción de bankroll compartido.

### 4.3 Kelly Robusto por Incertidumbre del Modelo

NOTA MATEMÁTICA IMPORTANTE (corregida):
  En un problema de apuesta BINARIA con un único tiro, la utilidad
  logarítmica es LINEAL en la probabilidad p:
      U(p, f) = p*log(1 + f*(o-1)) + (1-p)*log(1 - f)
  Por lo tanto, E_p[U(p, f)] = U(E[p], f), y el óptimo sobre f es
  EXACTAMENTE kelly_binary(E[p], o) = "Kelly plug-in" con la media
  posterior. En este caso no hay shrinkage matemático de Kelly por
  incertidumbre epistémica PURA del parámetro p (la linealidad anula
  el efecto Jensen).

  El "shrinkage por incertidumbre" clásico de Baker & McHale (2013)
  surge en configuraciones MULTI-PERÍODO con APRENDIZAJE (donde la
  apuesta de hoy afecta la información de mañana) o cuando se incorpora
  RIESGO DE MODELIZACIÓN (model misspecification: la verdadera p no
  está exactamente en la familia paramétrica de la posterior).

  Para capturar esta robustez de forma coherente y simple, este módulo
  usa un enfoque de CUANTIL INFERIOR sobre la posterior de p: en vez de
  apostar con la media, se apuesta con un cuantil inferior (por defecto
  el percentil 25). Esto es equivalente a un ajuste CVaR-like sobre el
  riesgo de modelización y garantiza:
      f_robust <= f_plugin
  siempre que la distribución no sea degenerada. Esto reemplaza el
  enfoque previo (E[kelly(p)] sobre la posterior) que, por la
  CONVEXIDAD piecewise-linear de kelly_binary (kink en p* = 1/o),
  puede producir f_bayes > f_plugin -- contrario a la intuición de
  robustez que se buscaba y que se documentaba.

### 4.4 Teoría de la Ruina del Jugador (Gambler's Ruin) -> piso de bankroll

Para una secuencia de apuestas con fracción Kelly f, ventaja esperada
positiva pero varianza no nula, la probabilidad de alcanzar un bankroll
crítico B_ruin (ej. 20% del capital inicial) ANTES de duplicar el
capital, bajo un modelo de random walk con drift, se aproxima (Kelly,
1956; extensión continua vía movimiento Browniano geométrico) como:

    P(ruina) = exp( -2*mu/sigma^2 * |log(bankroll_ratio_target)| )   si mu > 0

donde, para una apuesta con stake fraccional f, outcome per-unit-stake X
(X = o-1 con prob p, X = -1 con prob 1-p):

    drift logarítmico por apuesta:
        mu = E[log(1 + f*X)] ≈ f*edge - 0.5 * f² * Var[X]
    varianza logarítmica por apuesta:
        sigma² ≈ f² * Var[X]

    edge = E[X] = p*o - 1           (ventaja por unidad apostada)
    Var[X] = p*(o-1)² + (1-p)*1 - edge²
           = p*(o-1)² + (1-p) - (p*o - 1)²

Por lo tanto, `outcome_variance` en la función `ruin_probability` debe
ser `Var[X]` (varianza del resultado por unidad apostada), NO la
varianza del payout bruto `o²*p*(1-p)`.

Ejemplo para o=2.0, p=0.55:
    edge = 0.55*2 - 1 = 0.10
    Var[X] = 0.55*1 + 0.45 - 0.01 = 0.99 ≈ 1.0

Con la interpretación anterior (donde se pasaba `odds_variance_proxy=4.0`
para o=2) el término de varianza cuadrático siempre dominaba al drift
lineal, dando mu < 0 y P(ruina) = 1.0 para cualquier fracción -- un bug
silencioso que invalidaba cualquier recomendación de stake.

Use `compute_binary_outcome_variance(p, decimal_odds)` para obtener el
valor correcto de Var[X] a pasar a `ruin_probability`.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from dataclasses import dataclass

from config import MAX_KELLY_STAKE


# ----------------------------------------------------------------------
# Helpers: derivación correcta de Var[X] para una apuesta binaria
# ----------------------------------------------------------------------
def compute_binary_outcome_variance(prob_win: float, decimal_odds: float) -> float:
    """
    Para una apuesta binaria con probabilidad de ganar `prob_win` y cuota
    decimal `decimal_odds`, computa Var[X] donde X es el resultado por
    unidad apostada:
        X = (o - 1)  con prob p
        X = -1       con prob (1-p)

    Var[X] = E[X²] - E[X]²
           = [p*(o-1)² + (1-p)*1] - (p*o - 1)²

    Retorna 0 si los inputs son inválidos (cuota <= 1 o prob fuera de [0,1]).
    """
    if not (0.0 <= prob_win <= 1.0) or decimal_odds <= 1.0:
        return 0.0
    p = float(prob_win)
    o = float(decimal_odds)
    edge = p * o - 1.0
    ex2 = p * (o - 1.0) ** 2 + (1.0 - p) * 1.0
    return float(max(ex2 - edge ** 2, 0.0))


def compute_binary_edge(prob_win: float, decimal_odds: float) -> float:
    """edge = E[X] = p*o - 1 por unidad apostada."""
    if not (0.0 <= prob_win <= 1.0) or decimal_odds <= 1.0:
        return 0.0
    return float(prob_win * decimal_odds - 1.0)


# ----------------------------------------------------------------------
# 4.1 Kelly binario (referencia, ya probablemente en paper_trader.py)
# ----------------------------------------------------------------------
def kelly_binary(prob_win: float, decimal_odds: float) -> float:
    """f* = (p*(b+1) - 1) / b, con b = decimal_odds - 1. Retorna 0 si
    no hay edge positivo (evita apuestas con Kelly negativo)."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f = (prob_win * (b + 1) - 1) / b
    return max(f, 0.0)


# ----------------------------------------------------------------------
# 4.2 Kelly Multi-Resultado (1X2 simultáneo)
# ----------------------------------------------------------------------
def kelly_multi_outcome(probs: np.ndarray, decimal_odds: np.ndarray,
                         max_total_stake: float = 1.0,
                         uncertainty: np.ndarray | None = None) -> dict:
    """
    probs: array [p1, pX, p2] -- probabilidades del modelo (deben sumar
        ~1; se renormalizan si no).
    decimal_odds: array [o1, oX, o2] -- cuotas decimales de mercado.
    max_total_stake: fracción máxima del bankroll apostable en total
        (por defecto 1.0 = sin restricción adicional más allá de f>=0
        y suma<=1; en producción usualmente se pasa un valor << 1, ej.
        0.25 para Kelly Fraccional al 25%, ANTES de llamar a esta
        función, o se aplica el factor fraccional al resultado).

    Resuelve el problema de optimización cóncava:
        max_{f1,fX,f2} sum_i p_i * log(1 + f_i*(o_i - 1) - sum_{j!=i} f_j)
        s.a. f_i >= 0, sum(f_i) <= max_total_stake

    Se resuelve numéricamente vía SLSQP (el problema es cóncavo así que
    cualquier óptimo local hallado es global; SLSQP con restricciones
    lineales converge de forma fiable en este tipo de problema de baja
    dimensión).
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    odds = np.asarray(decimal_odds, dtype=float)
    uncertainty = np.asarray(uncertainty if uncertainty is not None else np.ones_like(probs), dtype=float)
    uncertainty = np.clip(uncertainty, 1e-3, None)

    def neg_expected_log_growth(f):
        f1, fX, f2 = f
        returns = np.array([
            1 + f1 * (odds[0] - 1) - fX - f2,
            1 + fX * (odds[1] - 1) - f1 - f2,
            1 + f2 * (odds[2] - 1) - f1 - fX,
        ])
        # Salvaguarda: si algún retorno es <= 0 (bankroll negativo en ese
        # escenario), penalizar fuertemente en vez de dejar que log()
        # devuelva NaN/-inf sin control.
        if np.any(returns <= 1e-9):
            return 1e6
        return -float(np.sum(probs * np.log(returns)))

    effective_max_stake = min(max_total_stake, MAX_KELLY_STAKE)
    f0 = np.array([0.01, 0.01, 0.01])
    bounds = [(0.0, effective_max_stake)] * 3
    constraint = optimize.LinearConstraint(np.ones(3), 0, effective_max_stake)

    result = optimize.minimize(
        neg_expected_log_growth, f0, method="SLSQP",
        bounds=bounds, constraints=[constraint],
        options={"maxiter": 500, "ftol": 1e-12},
    )

    f_opt = np.clip(result.x, 0, None)
    # Limpieza numérica: valores despreciables (<1e-6) se fijan a 0 para
    # no generar "micro-apuestas" sin sentido práctico.
    f_opt = np.where(f_opt < 1e-6, 0.0, f_opt)

    expected_log_growth = -neg_expected_log_growth(f_opt) if result.success else np.nan
    uncertainty_penalty = float(np.mean(1.0 / uncertainty))
    scaled_total_stake = float(f_opt.sum() / max(uncertainty_penalty, 1e-3))

    return {
        "stake_1": float(f_opt[0]),
        "stake_X": float(f_opt[1]),
        "stake_2": float(f_opt[2]),
        "total_stake": float(f_opt.sum()),
        "expected_log_growth": float(expected_log_growth),
        "converged": bool(result.success),
        "uncertainty_penalty": uncertainty_penalty,
        "effective_total_stake": min(scaled_total_stake, effective_max_stake),
        "implied_edges": {
            "1": float(probs[0] * odds[0] - 1),
            "X": float(probs[1] * odds[1] - 1),
            "2": float(probs[2] * odds[2] - 1),
        },
    }


# ----------------------------------------------------------------------
# 4.3 Kelly Robusto por Incertidumbre del Modelo
# ----------------------------------------------------------------------
def kelly_robust_binary(
    prob_posterior_samples: np.ndarray,
    decimal_odds: float,
    quantile: float = 0.25,
) -> dict:
    """
    Kelly robusto por incertidumbre epistémica del modelo.

    prob_posterior_samples: muestras de la distribución posterior de
        p_win (ej. de BayesianHierarchicalRates o de un Monte Carlo sobre
        los IC de AdvancedDixonColes). NO un punto estimado -- la
        distribución completa.

    quantile: cuantil inferior de la posterior a usar como probabilidad
        efectiva para el Kelly plug-in. Por defecto 0.25 (percentil 25).
        Cuanto menor sea el cuantil, más conservadora es la apuesta.
        - quantile=0.50  -> mediana (equivalente a plug-in si la
          posterior es simétrica).
        - quantile=0.05  -> muy conservador (solo se apuesta si el
          percentil 5 ya está por encima del cutoff del Kelly).
        - quantile=0.0   -> no usar (degradaría a kelly del mínimo, que
          casi siempre será 0).

    Razonamiento matemático:
      En una apuesta binaria single-shot, la utilidad logarítmica es
      LINEAL en p, por lo que el óptimo Bayesiano EXACTO coincide con
      kelly_binary(E[p], o) -- sin shrinkage. Sin embargo, esto ignora
      el riesgo de MODEL MISSPECIFICATION (la verdadera p puede no
      pertenecer a la familia paramétrica de la posterior). El uso de
      un cuantil inferior es equivalente a un ajuste CVaR-like que
      penaliza escenarios donde el modelo está sobre-estimando la
      probabilidad del evento.

      Con cuantil < 0.5 se garantiza f_robust <= f_plugin SIEMPRE que
      la posterior no sea degenerada, lo cual es la propiedad de
      robustez que se busca en producción.
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError(f"quantile debe estar en (0, 1]; recibido {quantile}")

    samples = np.asarray(prob_posterior_samples, dtype=float)
    if samples.size == 0:
        return {
            "f_plugin_naive": 0.0,
            "f_robust": 0.0,
            "shrinkage_factor": 1.0,
            "posterior_prob_mean": 0.0,
            "posterior_prob_std": 0.0,
            "effective_prob": 0.0,
            "effective_quantile": quantile,
            "recommendation": "Sin muestras posteriores; no apostar.",
        }

    p_mean = float(samples.mean())
    p_std = float(samples.std())
    p_effective = float(np.quantile(samples, quantile))

    f_plugin = kelly_binary(p_mean, decimal_odds)
    f_robust = kelly_binary(p_effective, decimal_odds)

    # Shrinkage garantizado <= 1 cuando quantile <= 0.5 (la función
    # kelly_binary es monótona no-decreciente en p). Solo puede ser > 1
    # si el usuario pasa cuantil > 0.5 y la posterior es asimétrica; lo
    # dejamos reportar pero documentamos la anomalía.
    shrinkage_factor = f_robust / f_plugin if f_plugin > 1e-9 else 1.0

    if quantile > 0.5 and shrinkage_factor > 1.0:
        recommendation = (
            f"Cuantil {quantile} > 0.5 produce shrinkage > 1 (sobre-apuesta). "
            "Use cuantil <= 0.5 para comportamiento robusto conservador."
        )
    else:
        recommendation = (
            f"Usar f_robust (basado en cuantil {quantile:.2f} de la posterior). "
            f"La reducción frente a plug-in ({(1 - shrinkage_factor) * 100:.1f}%) "
            "refleja el ajuste por incertidumbre epistémica del modelo."
        )

    return {
        "f_plugin_naive": float(f_plugin),
        "f_robust": float(f_robust),
        "shrinkage_factor": float(shrinkage_factor),
        "posterior_prob_mean": p_mean,
        "posterior_prob_std": p_std,
        "effective_prob": p_effective,
        "effective_quantile": quantile,
        "recommendation": recommendation,
    }


# Alias retrocompatible con la API anterior. Mantiene el nombre viejo
# para no romper imports de paper_trader.py u otros consumidores, pero
# internamente delega al kelly_robust_binary con cuantil por defecto 0.25.
def kelly_bayesian_binary(
    prob_posterior_samples: np.ndarray,
    decimal_odds: float,
    quantile: float = 0.25,
) -> dict:
    """Alias retrocompatible para `kelly_robust_binary`. Ver docstring
    de esa función para el detalle matemático."""
    return kelly_robust_binary(prob_posterior_samples, decimal_odds, quantile=quantile)


# ----------------------------------------------------------------------
# 4.4 Teoría de la Ruina -> piso de bankroll y Kelly fraccional óptimo
# ----------------------------------------------------------------------
def ruin_probability(
    kelly_fraction: float,
    edge: float,
    outcome_variance: float,
    bankroll_ratio_target: float = 0.2,
) -> float:
    """
    Aproxima P(ruina) = P(el bankroll cae a bankroll_ratio_target * W0
    antes de duplicarse), para una fracción de Kelly dada, vía la
    aproximación de movimiento Browniano geométrico:

        P(ruina) = exp( -2*mu/sigma^2 * |log(bankroll_ratio_target)| )

        mu    = kelly_fraction * edge - 0.5 * kelly_fraction^2 * outcome_variance
        sigma^2 = kelly_fraction^2 * outcome_variance

    Parámetros:
        kelly_fraction: fracción del bankroll arriesgada por apuesta.
        edge: ventaja esperada por unidad apostada, E[X] = p*o - 1.
            (NO es la fracción de Kelly, NO es la cuota. Para una
            apuesta con p=0.55 y o=2.0, edge = 0.10.)
        outcome_variance: Var[X] = varianza del resultado por unidad
            apostada. Para apuesta binaria:
                Var[X] = p*(o-1)² + (1-p) - (p*o - 1)²
            Use `compute_binary_outcome_variance(p, o)` para obtenerla.
            NO usar `o²` ni `p*(1-p)*o²` -- eso sobreestima la varianza
            y produce P(ruina)=1 para cualquier f razonable.
        bankroll_ratio_target: umbral de "ruina" como fracción del
            capital inicial (default 0.20).

    Si mu <= 0 (el sistema tiene edge neto negativo o cero con esa
    fracción y varianza), la ruina es una CERTEZA a largo plazo (P=1).
    """
    if kelly_fraction <= 0 or outcome_variance <= 0:
        return 0.0 if kelly_fraction <= 0 else 1.0
    if not (0.0 < bankroll_ratio_target < 1.0):
        raise ValueError(
            f"bankroll_ratio_target debe estar en (0, 1); recibido {bankroll_ratio_target}"
        )

    f = float(kelly_fraction)
    mu = f * edge - 0.5 * (f ** 2) * outcome_variance
    sigma2 = (f ** 2) * outcome_variance

    if mu <= 0:
        return 1.0
    if sigma2 < 1e-12:
        return 0.0

    exponent = -2.0 * mu / sigma2 * abs(np.log(bankroll_ratio_target))
    # Clamp superior: P(ruina) es una probabilidad, no puede exceder 1.
    return float(min(np.exp(exponent), 1.0))


def optimal_fractional_kelly(
    edge: float,
    outcome_variance: float,
    max_ruin_prob: float = 0.01,
    bankroll_ratio_target: float = 0.2,
    full_kelly_fraction: float = 1.0,
) -> dict:
    """
    Busca, vía bisección, el mayor multiplicador c en (0, full_kelly_fraction]
    tal que ruin_probability(c, edge, outcome_variance, bankroll_ratio_target)
    <= max_ruin_prob.

    Esto reemplaza el "Kelly Fraccional" arbitrario (ej. "usamos 25% de
    Kelly porque así se hace habitualmente") por un valor DERIVADO de un
    umbral de riesgo de ruina explícito y auditable.

    Parámetros:
        edge: E[X] = p*o - 1. Ver `compute_binary_edge`.
        outcome_variance: Var[X]. Ver `compute_binary_outcome_variance`.
        max_ruin_prob: tolerancia máxima de P(ruina) (default 1%).
        bankroll_ratio_target: piso de bankroll (default 20% del inicial).
        full_kelly_fraction: fracción de Kelly completo a considerar como
            techo (default 1.0 = 100% de Kelly).
    """
    if edge <= 0:
        return {
            "optimal_fraction_of_kelly": 0.0,
            "ruin_prob_at_optimum": 1.0,
            "note": "Edge <= 0: no apostar.",
        }
    if outcome_variance <= 0:
        return {
            "optimal_fraction_of_kelly": full_kelly_fraction,
            "ruin_prob_at_optimum": 0.0,
            "note": "Varianza cero (caso degenerado); Kelly completo satisface todo umbral.",
        }

    def f(c):
        return ruin_probability(c, edge, outcome_variance, bankroll_ratio_target) - max_ruin_prob

    lo, hi = 1e-6, full_kelly_fraction
    p_ruin_hi = ruin_probability(hi, edge, outcome_variance, bankroll_ratio_target)
    if p_ruin_hi <= max_ruin_prob:
        return {
            "optimal_fraction_of_kelly": full_kelly_fraction,
            "ruin_prob_at_optimum": p_ruin_hi,
            "note": "Kelly completo ya satisface el umbral de riesgo de ruina.",
        }
    p_ruin_lo = ruin_probability(lo, edge, outcome_variance, bankroll_ratio_target)
    if p_ruin_lo > max_ruin_prob:
        return {
            "optimal_fraction_of_kelly": 0.0,
            "ruin_prob_at_optimum": p_ruin_lo,
            "note": "Ni la fracción mínima satisface el umbral; varianza excesiva. No apostar.",
        }

    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid

    p_ruin_opt = ruin_probability(lo, edge, outcome_variance, bankroll_ratio_target)
    return {
        "optimal_fraction_of_kelly": float(lo),
        "ruin_prob_at_optimum": p_ruin_opt,
        "note": (
            f"Usar {lo*100:.2f}% de Kelly completo mantiene P(ruina hasta "
            f"{bankroll_ratio_target*100:.0f}% del capital) <= {max_ruin_prob*100:.1f}%."
        ),
    }
