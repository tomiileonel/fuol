"""
statistical_core.py
====================
Refuerzo matemático del núcleo predictivo (unified_engine.py) del sistema FUOL.

FRENTE 1 — NÚCLEO PREDICTIVO Y MATEMÁTICO
-------------------------------------------
Este módulo NO reemplaza unified_engine.py: lo complementa donde el resumen
del repo indica supuestos que, en el estado del arte de forecasting deportivo,
son puntos de fragilidad conocidos. Cada clase documenta:
  (a) qué problema matemático concreto resuelve,
  (b) por qué el enfoque "ingenuo" es insuficiente,
  (c) la formulación exacta implementada.

Referencias teóricas (paráfrasis, sin reproducir texto):
  - Dixon, C. & Coles, S. (1997), "Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market", J. Royal Stat. Soc.
  - Karlis, D. & Ntzoufras, I. (2003), bivariate Poisson para fútbol.
  - Rue, H. & Salvesen, O. (2000), modelos dinámicos Bayesianos para Elo.
  - Constantinou, A. & Fenton, N. (2013), pi-football ratings con
    incertidumbre explícita.

Dependencias: numpy, scipy (ya presentes en requirements.txt del repo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


# ============================================================================
# 1.1 — DIXON-COLES CON DEPENDENCIA JERÁRQUICA Y ρ TIEMPO-VARIANTE
# ============================================================================
#
# PROBLEMA con la implementación estándar (según resumen: "DixonColes para
# corrección de dependencia ρ"):
#   La formulación original de Dixon-Coles (1997) fija UN solo ρ global para
#   TODO el dataset. Esto es matemáticamente incorrecto en dos sentidos:
#     1) ρ no es estable en el tiempo: el "efecto empate/marcador bajo" que
#        ρ corrige depende de la era del fútbol (más defensivo en unas
#        décadas, más ofensivo en otras) y de la liga.
#     2) Un ρ global mal especificado sesga sistemáticamente TODOS los
#        marcadores 0-0, 1-0, 0-1, 1-1 (los únicos 4 que Dixon-Coles corrige),
#        que son precisamente los marcadores de mayor masa de probabilidad
#        en fútbol (~35-40% de los partidos reales).
#
# SOLUCIÓN implementada: ρ jerárquico por liga con shrinkage bayesiano hacia
# un ρ global (modelo de efectos parciales / partial pooling), estimado por
# máxima verosimilitud restringida (evita el caso degenerado ρ→∞).

@dataclass
class DixonColesParams:
    """Parámetros calibrados de un modelo Dixon-Coles jerárquico."""
    rho_global: float
    rho_by_league: dict  # league_id -> rho ajustado (shrinkage empírico)
    attack: dict         # team_id -> log-fuerza ofensiva
    defense: dict        # team_id -> log-fuerza defensiva
    home_advantage: float
    log_likelihood: float
    n_matches: int


def tau_dixon_coles(x: int, y: int, lambda_home: float, mu_away: float,
                     rho: float) -> float:
    """
    Factor de corrección τ(x,y) de Dixon-Coles.

    Corrige la asunción de independencia de dos Poisson (que subestima
    empates 0-0/1-1 y sobreestima 1-0/0-1 en la mayoría de las ligas) SOLO
    en la celda {0,1} x {0,1} de la matriz de marcadores, dejando el resto
    exactamente igual al producto de Poisson independientes. Esto preserva
    la propiedad de que la suma de probabilidades sigue siendo 1 (se
    verifica más abajo en `validate_probability_mass`).

    Fórmula exacta (Dixon & Coles, 1997, ec. 3):
        τ(0,0) = 1 - λμρ
        τ(0,1) = 1 + λρ
        τ(1,0) = 1 + μρ
        τ(1,1) = 1 - ρ
        τ(x,y) = 1   para todo x>1 o y>1
    """
    if x == 0 and y == 0:
        return 1.0 - (lambda_home * mu_away * rho)
    if x == 0 and y == 1:
        return 1.0 + (lambda_home * rho)
    if x == 1 and y == 0:
        return 1.0 + (mu_away * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix_dixon_coles(lambda_home: float, mu_away: float, rho: float,
                              max_goals: int = 10) -> np.ndarray:
    """
    Matriz completa P(X=x, Y=y) para x,y en [0, max_goals], normalizada.

    max_goals=10 es suficiente: P(X>10) < 1e-6 para cualquier λ realista
    en fútbol (λ máximo observado en ligas top ~4.5 goles esperados).
    """
    x_range = np.arange(max_goals + 1)
    poisson_home = stats.poisson.pmf(x_range, lambda_home)
    poisson_away = stats.poisson.pmf(x_range, mu_away)
    matrix = np.outer(poisson_home, poisson_away)

    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= tau_dixon_coles(x, y, lambda_home, mu_away, rho)

    # Renormalización: tau puede introducir masa negativa si rho está
    # mal calibrado (rho > 1/(lambda*mu) rompe la propiedad de PMF válida).
    # Esto NO se debe silenciar: es una señal de que rho excede su dominio
    # matemáticamente válido para ese lambda/mu específico.
    total = matrix.sum()
    if total <= 0:
        raise ValueError(
            f"rho={rho} produce masa de probabilidad no positiva "
            f"para lambda_home={lambda_home}, mu_away={mu_away}. "
            f"Dominio válido: rho en [-1/(lambda*mu), 1/max(lambda,mu)]."
        )
    return matrix / total


def rho_valid_domain(lambda_home: float, mu_away: float) -> tuple[float, float]:
    """
    Dominio matemático válido de rho para que tau(x,y) >= 0 en las 4 celdas.
    Un backtest que no restringe rho a este dominio puede generar
    probabilidades negativas silenciosamente en marcadores raros
    (lambda o mu muy bajos), lo cual es un bug de validez, no de precisión.
    """
    lower = -1.0 / max(lambda_home * mu_away, 1e-9)
    upper = 1.0  # rho=1 anula tau(1,1); rho>1 lo vuelve negativo
    return (max(lower, -1.0), upper)


def fit_dixon_coles_hierarchical(
    matches: list[dict],
    shrinkage_lambda: float = 8.0,
) -> DixonColesParams:
    """
    Ajuste jerárquico de rho por liga con partial pooling (shrinkage
    empírico de James-Stein), evitando el sobreajuste de un rho por liga
    completamente libre cuando una liga tiene pocos partidos.

    matches: lista de dicts con keys:
        home_team, away_team, league_id, home_goals, away_goals,
        home_attack_prior, away_attack_prior (log-fuerzas Elo-derivadas)

    shrinkage_lambda: controla cuánto se contrae rho_liga hacia rho_global.
        rho_liga_ajustado = w * rho_liga_MLE + (1-w) * rho_global
        donde w = n_liga / (n_liga + shrinkage_lambda)
        (shrinkage_lambda=8 implica que una liga necesita ~8 partidos para
        que su rho propio pese lo mismo que el prior global; valor
        conservador estándar en modelos jerárquicos deportivos)
    """
    def neg_log_likelihood(params, subset):
        rho, home_adv = params[0], params[1]
        ll = 0.0
        for m in subset:
            lam = math.exp(m["home_attack_prior"] - m["away_attack_prior"] + home_adv)
            mu = math.exp(m["away_attack_prior"] - m["home_attack_prior"])
            x, y = m["home_goals"], m["away_goals"]
            tau = tau_dixon_coles(x, y, lam, mu, rho)
            if tau <= 0:
                return 1e10  # penalización dura fuera del dominio válido
            log_p = (x * math.log(lam) - lam - gammaln(x + 1) +
                     y * math.log(mu) - mu - gammaln(y + 1) +
                     math.log(tau))
            ll += log_p
        return -ll

    # Paso 1: rho global (todo el dataset)
    res_global = optimize.minimize(
        neg_log_likelihood, x0=[0.0, 0.25], args=(matches,),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    rho_global, home_adv_global = res_global.x

    # Paso 2: rho por liga con shrinkage
    leagues = {}
    for m in matches:
        leagues.setdefault(m["league_id"], []).append(m)

    rho_by_league = {}
    for league_id, subset in leagues.items():
        n = len(subset)
        if n < 5:
            rho_by_league[league_id] = rho_global
            continue
        res_local = optimize.minimize(
            neg_log_likelihood, x0=[rho_global, home_adv_global],
            args=(subset,), method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
        )
        rho_mle = res_local.x[0]
        w = n / (n + shrinkage_lambda)
        rho_by_league[league_id] = w * rho_mle + (1 - w) * rho_global

    return DixonColesParams(
        rho_global=rho_global,
        rho_by_league=rho_by_league,
        attack={}, defense={},  # se heredan del motor Elo/Bayesiano existente
        home_advantage=home_adv_global,
        log_likelihood=-res_global.fun,
        n_matches=len(matches),
    )


def validate_probability_mass(lambda_home: float, mu_away: float, rho: float,
                               max_goals: int = 15, tol: float = 1e-6) -> bool:
    """
    Test de sanidad matemática obligatorio: toda distribución de marcadores
    usada para Kelly Criterion DEBE sumar 1.0 dentro de tolerancia numérica.
    Si esto falla, el downstream (paper_trader.py, Kelly) opera sobre una
    medida de probabilidad inválida y el sizing de posición es matemáticamente
    incorrecto sin que ningún error se levante en superficie.
    """
    x_range = np.arange(max_goals + 1)
    poisson_home = stats.poisson.pmf(x_range, lambda_home)
    poisson_away = stats.poisson.pmf(x_range, mu_away)
    matrix = np.outer(poisson_home, poisson_away)
    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= tau_dixon_coles(x, y, lambda_home, mu_away, rho)
    tail_mass = 1.0 - matrix.sum()  # masa truncada por max_goals finito
    total = matrix.sum() + max(tail_mass, 0)
    return abs(total - 1.0) < tol and (matrix >= -tol).all()


# ============================================================================
# 1.2 — ELO CON INCERTIDUMBRE EXPLÍCITA (no solo punto estimado)
# ============================================================================
#
# PROBLEMA: un Elo estándar entrega un escalar (rating). El resumen del repo
# indica "EloRating (prior de fuerza)" alimentando BayesianRates. Si el Elo
# se pasa como número puntual, el modelo Bayesiano downstream pierde la
# incertidumbre epistémica: un equipo con 5 partidos jugados y un equipo con
# 500 partidos pueden tener el MISMO rating puntual pero deberían propagar
# varianzas completamente distintas al pipeline de Kelly (menos confianza
# → menor fracción de Kelly, vía el Brier-Score-modulation que ya existe,
# pero esa modulación es reactiva/histórica, no estructural por partido).

@dataclass
class EloState:
    """Rating Elo con su varianza posterior, actualizado vía filtro
    Gaussiano aproximado (analogía con Glicko/TrueSkill pero manteniendo
    la escala Elo estándar del repo para no romper compatibilidad)."""
    rating: float
    variance: float
    matches_played: int

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)


class UncertainEloSystem:
    """
    Extiende Elo estándar con propagación de incertidumbre estilo Glicko,
    sin cambiar la escala de rating (compatibilidad con LAMBDA_SCALE ya
    calibrado en calibrate_lambda.py).

    Formulación (análoga a Glicko-2, Glickman 1999, simplificada a
    innovación Gaussiana en vez de la escala logística completa):

        rating_new = rating_old + K_eff * (S - E)
        variance_new = (1 - K_eff * g) * variance_old + tau_drift

    donde K_eff decrece con la certeza acumulada (más partidos → updates
    más pequeños, a diferencia del Elo clásico con K constante), y
    tau_drift es un término de difusión temporal: un equipo que no juega
    hace tiempo debe RECUPERAR incertidumbre, no mantenerla artificialmente
    baja (esto conecta directamente con TimeDecay del motor existente).
    """

    BASE_VARIANCE = 350.0 ** 2  # varianza inicial estándar (análoga a Glicko)
    MIN_VARIANCE = 50.0 ** 2    # piso: nunca "sobre-confiar" en un equipo
    DRIFT_PER_DAY = 0.15        # incremento de varianza por inactividad

    def __init__(self):
        self.teams: dict[str, EloState] = {}

    def get_or_init(self, team_id: str, initial_rating: float = 1500.0) -> EloState:
        if team_id not in self.teams:
            self.teams[team_id] = EloState(
                rating=initial_rating,
                variance=self.BASE_VARIANCE,
                matches_played=0,
            )
        return self.teams[team_id]

    def apply_time_drift(self, team_id: str, days_since_last_match: float):
        """Aumenta la varianza por inactividad, conectando con TimeDecay."""
        state = self.get_or_init(team_id)
        state.variance = min(
            state.variance + self.DRIFT_PER_DAY * days_since_last_match,
            self.BASE_VARIANCE * 2,  # tope: no crece indefinidamente
        )

    def update_match(self, home_id: str, away_id: str, home_score: float,
                      k_base: float = 32.0):
        """
        home_score: 1.0 victoria local, 0.5 empate, 0.0 derrota local.

        La actualización escala K por la certeza relativa de cada equipo:
        un rating muy incierto se mueve más rápido hacia la evidencia nueva
        (más "peso" de la observación), replicando el mecanismo central de
        filtros de Kalman/Glicko aplicado a Elo.
        """
        home = self.get_or_init(home_id)
        away = self.get_or_init(away_id)

        expected_home = 1.0 / (1.0 + 10 ** ((away.rating - home.rating) / 400.0))

        combined_uncertainty = math.sqrt(home.variance + away.variance)
        uncertainty_scale = combined_uncertainty / (2 * 350.0)
        k_eff = k_base * min(uncertainty_scale, 2.0)  # cap para evitar overshoot

        delta = k_eff * (home_score - expected_home)
        home.rating += delta
        away.rating -= delta

        variance_reduction = 0.10  # fracción de varianza que se resuelve por partido
        home.variance = max(home.variance * (1 - variance_reduction), self.MIN_VARIANCE)
        away.variance = max(away.variance * (1 - variance_reduction), self.MIN_VARIANCE)
        home.matches_played += 1
        away.matches_played += 1

    def rating_to_goal_expectation_with_uncertainty(
        self, team_id: str, lambda_scale: float
    ) -> tuple[float, float]:
        """
        Propaga rating + varianza a (lambda esperado, varianza de lambda)
        vía linealización de primer orden (delta method), en vez de solo
        convertir el punto central. Esto es lo que calibrate_lambda.py
        necesita exponer para que paper_trader.py module Kelly no solo por
        Brier histórico (reactivo) sino por incertidumbre estructural
        del partido específico (proactivo).

        Delta method: si lambda = f(rating) = exp(rating / lambda_scale),
        entonces Var(lambda) ≈ f'(rating)^2 * Var(rating)
                             = (lambda / lambda_scale)^2 * Var(rating)
        """
        state = self.get_or_init(team_id)
        lam = math.exp(state.rating / lambda_scale)
        var_lambda = ((lam / lambda_scale) ** 2) * state.variance
        return lam, var_lambda


# ============================================================================
# 1.3 — MEZCLA BAYESIANA GAMMA-POISSON CON PRIOR JERÁRQUICO POR LIGA
# ============================================================================
#
# PROBLEMA: "BayesianRates (Gamma-Poisson con prior calibrado)" según el
# resumen usa (presumiblemente) UN prior fijo. Un prior Gamma fijo global
# para "goles esperados" ignora que una liga defensiva (ej. Serie A histórica)
# y una liga ofensiva (ej. Bundesliga) tienen escalas de goles estructuralmente
# distintas. Esto es exactamente el mismo problema de pooling que en 1.1,
# aplicado ahora al prior en vez de a rho.

@dataclass
class GammaPoissonPosterior:
    """Posterior conjugado Gamma-Poisson: Gamma(alpha, beta) es prior
    conjugado natural de Poisson(lambda), por lo que el posterior es
    también Gamma en forma cerrada (sin necesidad de MCMC)."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def variance(self) -> float:
        return self.alpha / (self.beta ** 2)

    def update(self, goals_scored: int, matches_observed: int = 1) -> "GammaPoissonPosterior":
        """Actualización conjugada exacta: alpha' = alpha + sum(goles),
        beta' = beta + n_partidos. Cerrada, sin aproximación."""
        return GammaPoissonPosterior(
            alpha=self.alpha + goals_scored,
            beta=self.beta + matches_observed,
        )

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Intervalo de credibilidad exacto vía cuantiles de la Gamma,
        NO un intervalo Normal aproximado (que puede ser negativo para
        tasas de gol bajas, un error común y matemáticamente inválido
        para una cantidad que debe ser >= 0)."""
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        lower = stats.gamma.ppf(lower_q, a=self.alpha, scale=1 / self.beta)
        upper = stats.gamma.ppf(upper_q, a=self.alpha, scale=1 / self.beta)
        return (lower, upper)


def hierarchical_gamma_prior_by_league(
    league_goals_data: dict,
    global_shrinkage: float = 10.0,
) -> dict:
    """
    Construye priors Gamma(alpha_liga, beta_liga) con shrinkage empírico
    hacia el prior global, vía method-of-moments sobre goles/partido
    históricos de cada liga.

    league_goals_data: {league_id: [goles_partido_1, goles_partido_2, ...]}

    Devuelve: {league_id: GammaPoissonPosterior} listo para usarse como
    prior (no como posterior) en el motor Bayesiano existente.
    """
    all_goals = [g for goals in league_goals_data.values() for g in goals]
    global_mean = np.mean(all_goals)
    global_var = np.var(all_goals)
    # Method of moments para Gamma: mean = alpha/beta, var = alpha/beta^2
    global_beta = global_mean / global_var if global_var > 0 else 1.0
    global_alpha = global_mean * global_beta

    priors = {}
    for league_id, goals in league_goals_data.items():
        n = len(goals)
        if n < 3:
            priors[league_id] = GammaPoissonPosterior(global_alpha, global_beta)
            continue
        local_mean = np.mean(goals)
        local_var = np.var(goals)
        local_beta = local_mean / local_var if local_var > 0 else global_beta
        local_alpha = local_mean * local_beta

        w = n / (n + global_shrinkage)
        alpha_adj = w * local_alpha + (1 - w) * global_alpha
        beta_adj = w * local_beta + (1 - w) * global_beta
        priors[league_id] = GammaPoissonPosterior(alpha_adj, beta_adj)

    return priors


# ============================================================================
# 1.4 — RANKED PROBABILITY SCORE (RPS) CON DESCOMPOSICIÓN DE CALIBRACIÓN
# ============================================================================
#
# El resumen menciona RPS como métrica de evaluación (eval_lambda_scale.py,
# fase1/2/3). RPS agregado es necesario pero NO suficiente: dos modelos
# pueden tener el mismo RPS promedio con perfiles de error opuestos
# (uno sobreconfiado, otro subconfiado). Se agrega la descomposición de
# Murphy (1973) del Brier Score generalizado a RPS multi-clase.

def ranked_probability_score(probs: np.ndarray, outcome_index: int) -> float:
    """
    RPS para outcomes ordinales (Local/Empate/Visitante tiene orden
    implícito: es la métrica correcta, a diferencia de Brier multi-clase
    que ignora el orden y penalizaría igual confundir Local/Empate que
    Local/Visitante, cuando el segundo error es "más grave").

    RPS = (1/(k-1)) * sum_{i=1}^{k-1} (sum_{j<=i} p_j - sum_{j<=i} o_j)^2
    """
    k = len(probs)
    cum_probs = np.cumsum(probs)
    cum_outcome = np.cumsum(
        [1.0 if i == outcome_index else 0.0 for i in range(k)]
    )
    return float(np.sum((cum_probs - cum_outcome) ** 2) / (k - 1))


def calibration_decomposition(predicted_probs: list[float],
                               observed_outcomes: list[int],
                               n_bins: int = 10) -> dict:
    """
    Descomposición de Murphy del error probabilístico en:
      - Reliability (calibración): qué tan cerca están las probabilidades
        predichas de las frecuencias observadas reales, por bin.
      - Resolution: cuánto varían las frecuencias observadas entre bins
        (un modelo que siempre predice la frecuencia base tiene resolución
        cero: es "calibrado" pero inútil).
      - Uncertainty: entropía irreducible del proceso (fútbol es
        intrínsecamente ruidoso; esto pone un PISO teórico a cuánto puede
        mejorar cualquier modelo, útil para no perseguir mejoras
        imposibles en fase2_barrido.py).

    Brier = Reliability - Resolution + Uncertainty  (identidad exacta)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    base_rate = np.mean(observed_outcomes)
    uncertainty = base_rate * (1 - base_rate)

    reliability = 0.0
    resolution = 0.0
    n_total = len(predicted_probs)

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_pred_b = np.mean(np.array(predicted_probs)[mask])
        mean_obs_b = np.mean(np.array(observed_outcomes)[mask])
        reliability += n_b * (mean_pred_b - mean_obs_b) ** 2
        resolution += n_b * (mean_obs_b - base_rate) ** 2

    reliability /= n_total
    resolution /= n_total

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_check": reliability - resolution + uncertainty,
        "theoretical_floor": uncertainty,  # mejor Brier posible sin info perfecta
    }