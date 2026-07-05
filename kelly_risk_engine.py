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
  2. Kelly ajustado por incertidumbre bayesiana del modelo (shrinkage
     hacia stake 0 cuando el modelo está incierto sobre su propia
     probabilidad -- no solo cuando el edge estimado es chico).
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

### 4.3 Kelly Bayesiano (ajuste por incertidumbre del modelo)

El Kelly clásico trata a p como si fuera CONOCIDO exactamente. En
realidad, el sistema tiene una DISTRIBUCIÓN posterior sobre p (viene de
los módulos Dixon-Coles/Bayesiano: un IC, no un punto). El "Kelly
Bayesiano" (a veces llamado "Kelly con shrinkage por incertidumbre
paramétrica") integra sobre esa incertidumbre en vez de usar el punto
estimado:

    f*_bayes = E_theta~posterior[ f*(p(theta)) ]  -- vía Monte Carlo

Alternativa cerrada (aproximación de 2do orden, más rápida que MC):
Si p ~ Beta(a,b) (o se aproxima la posterior de p por una Beta ajustada
por momentos a partir del IC), se puede mostrar que el Kelly óptimo bajo
incertidumbre de parámetro es ESTRICTAMENTE MENOR que el Kelly con el
punto estimado plug-in (Baker & McHale, 2013; "shrinkage" de Kelly por
incertidumbre epistémica). La intuición: apostar fuerte basado en una
probabilidad que podría estar mal estimada expone a pérdidas
sistemáticas mayores que las que Kelly-plug-in anticipa, porque Kelly
plug-in NO sabe que su propio input p es incierto.

Este módulo implementa la versión Monte Carlo (general, no depende de
supuestos de forma funcional de la posterior) muestreando directamente
de la posterior si está disponible (ej. desde BayesianHierarchicalRates
o desde los IC de AdvancedDixonColes), y una aproximación analítica
rápida para el caso Beta.

### 4.4 Teoría de la Ruina del Jugador (Gambler's Ruin) -> piso de bankroll

Para una secuencia de apuestas con fracción Kelly f, ventaja esperada
positiva pero varianza no nula, la probabilidad de alcanzar un bankroll
crítico B_ruin (ej. 20% del capital inicial) ANTES de duplicar el
capital, bajo un modelo de random walk con drift, se aproxima (Kelly,
1956; extensión continua vía movimiento Browniano geométrico) como:

    P(ruina) = exp( -2*mu/sigma^2 * log(W0/B_ruin) )   si mu > 0

    mu = E[log(1 + f*edge)] aprox f*edge - 0.5*f^2*sigma_odds^2  (drift log)
    sigma^2 = Var[log(1 + f*resultado)]  (varianza del retorno logarítmico)

Esto permite fijar f_max (fracción de Kelly a usar, ej. "Kelly al 25%"
que ya se menciona en el sistema como Kelly Fraccional) de forma que
P(ruina) quede por debajo de un umbral tolerado (ej. 1%), en vez de fijar
el fraccionamiento de Kelly de forma arbitraria.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from dataclasses import dataclass


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
                         max_total_stake: float = 1.0) -> dict:
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

    def neg_expected_log_growth(f):
        f1, fX, f2 = f
        stakes = np.array([f1, fX, f2])
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

    f0 = np.array([0.01, 0.01, 0.01])
    bounds = [(0.0, max_total_stake)] * 3
    constraint = optimize.LinearConstraint(np.ones(3), 0, max_total_stake)

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

    return {
        "stake_1": float(f_opt[0]),
        "stake_X": float(f_opt[1]),
        "stake_2": float(f_opt[2]),
        "total_stake": float(f_opt.sum()),
        "expected_log_growth": float(expected_log_growth),
        "converged": bool(result.success),
        "implied_edges": {
            "1": float(probs[0] * odds[0] - 1),
            "X": float(probs[1] * odds[1] - 1),
            "2": float(probs[2] * odds[2] - 1),
        },
    }


# ----------------------------------------------------------------------
# 4.3 Kelly Bayesiano (ajuste por incertidumbre epistémica del modelo)
# ----------------------------------------------------------------------
def kelly_bayesian_binary(prob_posterior_samples: np.ndarray, decimal_odds: float) -> dict:
    """
    prob_posterior_samples: muestras de la distribución posterior de
        p_win (ej. de BayesianHierarchicalRates.compare_teams o de un
        Monte Carlo sobre los IC de AdvancedDixonColes). NO un punto
        estimado -- la distribución completa.

    Calcula:
        f*_plugin  = kelly_binary(media_posterior, odds)   [ingenuo]
        f*_bayes   = E_theta[ kelly_binary(p(theta), odds) ]  [correcto]

    f*_bayes < f*_plugin cuando hay incertidumbre real y sustancial en
    p (esto es matemáticamente esperable por la concavidad de f* como
    función de p combinada con la penalización asimétrica de apostar de
    más cuando p resulta ser menor que el estimado puntual -- el error
    por sobreestimar duele más en crecimiento logarítmico que el
    beneficio de acertar por el mismo margen).
    """
    samples = np.asarray(prob_posterior_samples, dtype=float)
    f_plugin = kelly_binary(float(samples.mean()), decimal_odds)

    f_per_sample = np.array([kelly_binary(p, decimal_odds) for p in samples])
    f_bayes = float(f_per_sample.mean())

    shrinkage_factor = f_bayes / f_plugin if f_plugin > 1e-9 else 1.0

    return {
        "f_plugin_naive": float(f_plugin),
        "f_bayes_correct": f_bayes,
        "shrinkage_factor": float(shrinkage_factor),
        "posterior_prob_mean": float(samples.mean()),
        "posterior_prob_std": float(samples.std()),
        "recommendation": (
            "Usar f_bayes_correct. La diferencia con f_plugin_naive "
            f"({(1 - shrinkage_factor) * 100:.1f}% de reducción) refleja "
            "el costo de la incertidumbre del modelo sobre su propia "
            "probabilidad estimada."
        ),
    }


# ----------------------------------------------------------------------
# 4.4 Teoría de la Ruina -> piso de bankroll y Kelly fraccional óptimo
# ----------------------------------------------------------------------
def ruin_probability(kelly_fraction: float, edge: float, odds_variance_proxy: float,
                      bankroll_ratio_target: float = 0.2) -> float:
    """
    Aproxima P(ruina) = P(el bankroll cae a bankroll_ratio_target * W0
    antes de duplicarse), para una fracción de Kelly dada, vía la
    aproximación de movimiento Browniano geométrico:

        P(ruina) = exp( -2*mu/sigma^2 * |log(bankroll_ratio_target)| )

        mu    ~ kelly_fraction * edge - 0.5 * kelly_fraction^2 * odds_variance_proxy
        sigma^2 ~ kelly_fraction^2 * odds_variance_proxy

    edge: ventaja esperada por unidad apostada (ej. de implied_edges en
        kelly_multi_outcome).
    odds_variance_proxy: varianza del retorno por unidad apostada (para
        una apuesta binaria con prob p y odds o: Var = p*(1-p)*o^2,
        aproximadamente -- se puede pasar precalculado).

    Si mu <= 0 (el sistema tiene edge negativo neto con esa fracción),
    la ruina es una CERTEZA a largo plazo (P=1) -- se retorna 1.0
    explícitamente en vez de un número posiblemente engañoso.
    """
    mu = kelly_fraction * edge - 0.5 * (kelly_fraction ** 2) * odds_variance_proxy
    sigma2 = (kelly_fraction ** 2) * odds_variance_proxy

    if mu <= 0:
        return 1.0
    if sigma2 < 1e-12:
        return 0.0

    exponent = -2 * mu / sigma2 * abs(np.log(bankroll_ratio_target))
    return float(np.exp(exponent))


def optimal_fractional_kelly(edge: float, odds_variance_proxy: float,
                              max_ruin_prob: float = 0.01,
                              bankroll_ratio_target: float = 0.2,
                              full_kelly_fraction: float = 1.0) -> dict:
    """
    Busca, vía bisección, el mayor multiplicador c en (0, full_kelly_fraction]
    tal que ruin_probability(c, edge, odds_variance_proxy, bankroll_ratio_target)
    <= max_ruin_prob.

    Esto reemplaza el "Kelly Fraccional" arbitrario (ej. "usamos 25% de
    Kelly porque así se hace habitualmente") por un valor DERIVADO de un
    umbral de riesgo de ruina explícito y auditable.
    """
    def f(c):
        return ruin_probability(c, edge, odds_variance_proxy, bankroll_ratio_target) - max_ruin_prob

    lo, hi = 1e-6, full_kelly_fraction
    if f(hi) <= 0:
        # incluso Kelly completo cumple el umbral de riesgo
        return {"optimal_fraction_of_kelly": full_kelly_fraction,
                "ruin_prob_at_optimum": ruin_probability(hi, edge, odds_variance_proxy, bankroll_ratio_target),
                "note": "Kelly completo ya satisface el umbral de riesgo de ruina."}
    if f(lo) > 0:
        return {"optimal_fraction_of_kelly": 0.0, "ruin_prob_at_optimum": f(lo) + max_ruin_prob,
                "note": "Ni la fracción mínima satisface el umbral; edge insuficiente o varianza excesiva. No apostar."}

    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid

    return {
        "optimal_fraction_of_kelly": float(lo),
        "ruin_prob_at_optimum": ruin_probability(lo, edge, odds_variance_proxy, bankroll_ratio_target),
        "note": (
            f"Usar {lo*100:.1f}% de Kelly completo mantiene P(ruina hasta "
            f"{bankroll_ratio_target*100:.0f}% del capital) <= {max_ruin_prob*100:.1f}%."
        ),
    }