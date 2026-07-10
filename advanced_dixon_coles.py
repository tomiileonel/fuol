"""
advanced_dixon_coles.py
========================
Extensión de grado institucional del núcleo Dixon-Coles del sistema.

CONTEXTO
--------
El unified_engine.py actual usa Dixon-Coles como corrección de dependencia
sobre un modelo Poisson doble (marginal para goles local/visitante). Este
módulo añade lo que normalmente falta en implementaciones "de manual":

  1. Estimación por Máxima Verosimilitud COMPLETA (ataque, defensa, home
     advantage, rho, y decaimiento temporal) vía optimización conjunta,
     no ajustes secuenciales desacoplados.
  2. Matriz de Información de Fisher -> errores estándar e intervalos de
     confianza para CADA parámetro (fuerza de ataque/defensa por equipo).
     Sin esto, no se puede saber si "Equipo A es mejor que Equipo B" es
     una diferencia real o ruido estadístico.
  3. Generalización de la corrección de dependencia de Dixon-Coles (que
     originalmente solo corrige los marcadores 0-0, 1-0, 0-1, 1-1) a una
     familia de cópulas discretas (Frank discreta) para capturar
     dependencia en TODO el soporte de marcadores, no solo el rincón bajo.

FUNDAMENTO MATEMÁTICO
----------------------
Modelo base (Dixon & Coles, 1997):
    X_ij ~ Poisson(lambda_ij)   goles del local i vs visitante j
    Y_ij ~ Poisson(mu_ij)       goles del visitante j vs local i

    lambda_ij = alpha_i * beta_j * gamma   (gamma = home advantage)
    mu_ij     = alpha_j * beta_i

donde alpha_i = fuerza de ataque del equipo i, beta_i = (debilidad de)
defensa del equipo i. Se impone la restricción de identificabilidad
  sum_i alpha_i = n_equipos  (o producto = 1, según convención)
para que el sistema no tenga infinitas soluciones equivalentes
(alpha_i * k, beta_i / k da la misma verosimilitud para cualquier k).

La probabilidad conjunta observada se corrige con tau (rho):
    P(X=x, Y=y) = tau_rho(x,y) * Poisson(x; lambda) * Poisson(y; mu)

    tau_rho(0,0) = 1 - lambda*mu*rho
    tau_rho(0,1) = 1 + lambda*rho
    tau_rho(1,0) = 1 + mu*rho
    tau_rho(1,1) = 1 - rho
    tau_rho(x,y) = 1   en cualquier otro caso

rho < 0 típicamente: empates 0-0 y 1-1 son MÁS frecuentes de lo que
predice el producto de Poissons independientes (los equipos "se cuidan"
en el marcador), y 1-0/0-1 son menos frecuentes.

LOG-VEROSIMILITUD PONDERADA POR TIEMPO (Dixon-Coles decay):
    L(theta) = sum_t phi(t) * log[ tau_rho(x_t,y_t) * Poisson(x_t;lambda_t)
                                   * Poisson(y_t;mu_t) ]

    phi(t) = exp(-xi * (T - t))     xi = tasa de decaimiento

theta = (alpha_1..alpha_n, beta_1..beta_n, gamma, rho, xi)

Se maximiza L vía L-BFGS-B con restricciones y luego se invierte el
Hessiano negativo (matriz de información observada) para obtener la
matriz de covarianza asintótica de los estimadores MLE:

    Cov(theta_hat) ~= [ -H(theta_hat) ]^{-1}

    SE(theta_i) = sqrt(Cov_ii)
    IC 95%(theta_i) = theta_i +/- 1.96 * SE(theta_i)

Esto es lo que permite responder con rigor: "¿la fuerza de ataque del
Equipo A es significativamente distinta de la del Equipo B?" en vez de
comparar puntos estimados desnudos.

EXTENSIÓN: CÓPULA DE FRANK DISCRETA
------------------------------------
La corrección tau de Dixon-Coles solo actúa sobre 4 celdas (0-0,0-1,1-0,1-1).
Para marcadores más altos, el modelo colapsa a independencia total, lo cual
es empíricamente cuestionable (existe dependencia negativa residual en todo
el soporte: un gol tempranero cambia la dinámica del partido completo).

Se ofrece una alternativa opcional: cópula de Frank discreta, que modela
dependencia en TODO el soporte con un solo parámetro theta_frank:

    C(u,v; theta) = -1/theta * log[1 + (e^{-theta*u}-1)(e^{-theta*v}-1)/(e^{-theta}-1)]

    P(X=x,Y=y) = C(F_X(x),F_Y(y)) - C(F_X(x-1),F_Y(y)) - C(F_X(x),F_Y(y-1))
                 + C(F_X(x-1),F_Y(y-1))

donde F_X, F_Y son las CDFs marginales Poisson. theta_frank > 0 implica
dependencia positiva, < 0 dependencia negativa. Se recomienda usar Dixon-
Coles clásico para producción (más simple, más testeado, menos riesgo de
overfitting) y la cópula de Frank como validación cruzada / diagnóstico de
si la corrección de 4 celdas es suficiente.

USO
---
    engine = AdvancedDixonColes(n_teams=20)
    result = engine.fit(matches)  # lista de dicts con home,away,gh,ga,date
    ci = engine.confidence_intervals()
    probs = engine.predict_match_probs('Equipo A', 'Equipo B')
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class DixonColesFitResult:
    team_index: dict
    alpha: np.ndarray          # fuerza de ataque por equipo
    beta: np.ndarray           # fuerza de defensa por equipo (menor = mejor defensa)
    gamma: float               # ventaja de local (multiplicativa, en espacio log)
    rho: float                 # parámetro de dependencia Dixon-Coles
    xi: float                  # tasa de decaimiento temporal
    log_likelihood: float
    n_params: int
    n_obs: int
    aic: float
    bic: float
    converged: bool
    std_errors: Optional[np.ndarray] = None
    cov_matrix: Optional[np.ndarray] = None
    theta_hat_raw: Optional[np.ndarray] = None  # theta_hat pre-renormalización (debug/introspección)


class AdvancedDixonColes:
    """
    Motor Dixon-Coles con MLE conjunta, inferencia de incertidumbre
    (Fisher Information) y decaimiento temporal integrado en una sola
    optimización (no en pasos separados como suele hacerse).
    """

    def __init__(self, n_teams: int, l2_reg: float = 0.001):
        """
        Parameters
        ----------
        n_teams : int
            Número de equipos distintos en el dataset.
        l2_reg : float
            Regularización L2 (ridge) sobre alpha y beta en espacio log.
            Evita que equipos con pocos partidos exploten a fuerzas
            infinitas (un problema real de MLE con datos escasos).
            lambda_ridge * sum(log(alpha_i)^2 + log(beta_i)^2)
        """
        self.n_teams = n_teams
        self.l2_reg = l2_reg
        self.team_index: dict = {}
        self.fit_result: Optional[DixonColesFitResult] = None

    # ------------------------------------------------------------------
    # Construcción de índices y preparación de datos
    # ------------------------------------------------------------------
    def _build_team_index(self, matches: list[dict]) -> dict:
        teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        return {t: i for i, t in enumerate(teams)}

    # ------------------------------------------------------------------
    # Log-verosimilitud negativa (función objetivo a minimizar)
    # ------------------------------------------------------------------
    def _unpack(self, theta: np.ndarray, n: int):
        log_alpha = theta[0:n]
        log_beta = theta[n:2 * n]
        gamma = theta[2 * n]
        # rho se parametriza internamente como tanh(raw) para mantenerlo
        # SIEMPRE en (-1, 1), que es el rango en el que tau_rho(x,y) puede
        # garantizarse positivo para lambda, mu razonables. Sin este
        # acotamiento, L-BFGS-B puede empujar rho a valores grandes que
        # hacen tau negativo y la log-verosimilitud deja de tener sentido
        # (esto es exactamente lo que causaba la no-convergencia inicial).
        rho_raw = theta[2 * n + 1]
        rho = np.tanh(rho_raw)
        xi = theta[2 * n + 2]
        return log_alpha, log_beta, gamma, rho, xi

    @staticmethod
    def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
              rho: float) -> np.ndarray:
        """Corrección de dependencia de Dixon-Coles, vectorizada."""
        tau = np.ones_like(lam)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        # Salvaguarda numérica: tau debe ser > 0 para que log(tau) exista.
        # Si la optimización empuja rho a una región inválida, se penaliza
        # en vez de producir NaN (evita que L-BFGS-B se rompa silenciosamente).
        tau = np.clip(tau, 1e-10, None)
        return tau

    def _neg_log_likelihood(self, theta: np.ndarray, home_idx, away_idx,
                             gh, ga, t_norm) -> float:
        n = self.n_teams
        log_alpha, log_beta, gamma, rho, xi = self._unpack(theta, n)

        log_lambda = log_alpha[home_idx] + log_beta[away_idx] + gamma
        log_mu = log_alpha[away_idx] + log_beta[home_idx]
        lam = np.exp(log_lambda)
        mu = np.exp(log_mu)

        # log P(X=x) para Poisson: x*log(lam) - lam - log(x!)
        log_pmf_x = gh * log_lambda - lam - gammaln(gh + 1)
        log_pmf_y = ga * log_mu - mu - gammaln(ga + 1)

        tau = self._tau(gh, ga, lam, mu, rho)
        log_tau = np.log(tau)

        # Ponderación temporal: phi(t) = exp(-xi * (T - t)), t_norm en [0,1]
        # donde 1 = partido más reciente. xi >= 0 forzado vía softplus
        # para que el decaimiento nunca sea "creciente hacia el pasado".
        xi_pos = np.log1p(np.exp(xi))  # softplus, siempre >= 0
        phi = np.exp(-xi_pos * (1.0 - t_norm))

        ll = np.sum(phi * (log_pmf_x + log_pmf_y + log_tau))

        # Regularización ridge en log_alpha, log_beta (shrink hacia 0,
        # es decir, hacia fuerza promedio). Evita blow-up con equipos
        # de pocos partidos (ej. recién ascendidos).
        reg = self.l2_reg * (np.sum(log_alpha ** 2) + np.sum(log_beta ** 2))

        return -(ll) + reg

    # ------------------------------------------------------------------
    # Ajuste (fit)
    # ------------------------------------------------------------------
    def fit(self, matches: list[dict], date_key: str = "date", skip_fisher_info: bool = False) -> DixonColesFitResult:
        """
        matches: lista de dicts con claves:
            'home' (str), 'away' (str), 'gh' (int goles local),
            'ga' (int goles visitante), date_key (algo ordenable, ej.
            timestamp o índice entero de jornada)

        Optimiza TODOS los parámetros conjuntamente: esto es más correcto
        que el patrón común de "calibrar rho por separado después de fijar
        alpha/beta", porque esa práctica produce estimadores sesgados
        (ignora la covarianza entre rho y las fuerzas de equipo).
        """
        self.team_index = self._build_team_index(matches)
        n = len(self.team_index)
        self.n_teams = n

        home_idx = np.array([self.team_index[m["home"]] for m in matches])
        away_idx = np.array([self.team_index[m["away"]] for m in matches])
        gh = np.array([m["gh"] for m in matches], dtype=float)
        ga = np.array([m["ga"] for m in matches], dtype=float)

        dates = np.array([m[date_key] for m in matches], dtype=float)
        d_min, d_max = dates.min(), dates.max()
        span = max(d_max - d_min, 1e-9)
        t_norm = (dates - d_min) / span

        n_params = 2 * n + 3
        theta0 = np.zeros(n_params)
        theta0[2 * n] = 0.3        # gamma inicial (home advantage positivo)
        theta0[2 * n + 1] = -0.05  # rho_raw inicial (tanh(-0.05)=~-0.05, dependencia negativa típica)
        theta0[2 * n + 2] = -2.0   # xi_raw inicial -> softplus pequeño (decay lento)

        # Bounds: log_alpha/log_beta acotados a un rango generoso pero
        # finito (evita blow-up con equipos de pocos partidos, incluso
        # con la regularización L2 ya presente). gamma acotado a un rango
        # físicamente razonable de home advantage. rho_raw acotado de
        # forma que tanh(rho_raw) cubra casi todo (-1,1) sin permitir que
        # el optimizador divague en la región plana de tanh (|rho_raw|>4
        # ya satura tanh, así que 5 es más que suficiente margen).
        bounds = (
            [(-4.0, 4.0)] * n +      # log_alpha
            [(-4.0, 4.0)] * n +      # log_beta
            [(-2.0, 2.0)] +          # gamma
            [(-5.0, 5.0)] +          # rho_raw (-> tanh en (-0.9999,0.9999))
            [(-10.0, 10.0)]          # xi_raw (softplus)
        )

        # Restricción de identificabilidad: en vez de imponer una
        # constraint dura, se resuelve libremente y luego se re-normaliza
        # (equivalente matemáticamente, más estable numéricamente para
        # L-BFGS-B sin restricciones de igualdad).
        result = optimize.minimize(
            self._neg_log_likelihood,
            theta0,
            args=(home_idx, away_idx, gh, ga, t_norm),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "maxfun": 50000, "ftol": 1e-12, "gtol": 1e-10},
        )

        theta_hat = result.x
        log_alpha, log_beta, gamma, rho, xi = self._unpack(theta_hat, n)

        # Re-normalización post-hoc: fijar mean(log_alpha) = 0 y
        # trasladar la diferencia a gamma, preservando lambda/mu exactos.
        shift = log_alpha.mean()
        log_alpha_norm = log_alpha - shift
        gamma_norm = gamma + shift

        alpha = np.exp(log_alpha_norm)
        beta = np.exp(log_beta)
        xi_pos = float(np.log1p(np.exp(xi)))

        # --- Matriz de Información de Fisher (para IC de los parámetros) ---
        if skip_fisher_info:
            std_errors, cov_matrix = None, None
        else:
            std_errors, cov_matrix = self._compute_std_errors(
                theta_hat, home_idx, away_idx, gh, ga, t_norm
            )

        n_obs = len(matches)
        ll = -result.fun
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_obs) - 2 * ll

        self.fit_result = DixonColesFitResult(
            team_index=self.team_index,
            alpha=alpha,
            beta=beta,
            gamma=float(gamma_norm),
            rho=float(rho),
            xi=xi_pos,
            log_likelihood=float(ll),
            n_params=n_params,
            n_obs=n_obs,
            aic=float(aic),
            bic=float(bic),
            converged=bool(result.success),
            std_errors=std_errors,
            cov_matrix=cov_matrix,
            theta_hat_raw=theta_hat.copy(),
        )

        if not result.success:
            warnings.warn(
                f"Optimización Dixon-Coles no convergió limpiamente: "
                f"{result.message}. Revisar datos de entrada (posibles "
                f"equipos con < 3 partidos, o colinealidad)."
            )

        return self.fit_result

    # ------------------------------------------------------------------
    # Inferencia: Matriz de información de Fisher -> errores estándar
    # ------------------------------------------------------------------
    def _compute_std_errors(self, theta_hat, home_idx, away_idx, gh, ga, t_norm):
        """
        Calcula el Hessiano numérico de la NEGATIVA log-verosimilitud en
        theta_hat (esto ES la matriz de información observada, por
        definición: I(theta) = -d^2 L / d theta^2 evaluado en el MLE).

        Cov(theta_hat) ~ I(theta_hat)^{-1}   (aproximación asintótica MLE)

        Se usa diferenciación numérica de segundo orden (más robusto que
        depender de autograd, que no está garantizado en este entorno).
        """
        n_params = len(theta_hat)

        def neg_ll(th):
            return self._neg_log_likelihood(th, home_idx, away_idx, gh, ga, t_norm)

        try:
            hess = self._numerical_hessian(neg_ll, theta_hat)

            # PROBLEMA DE IDENTIFICABILIDAD (gauge freedom): el modelo
            # Dixon-Coles tiene una dirección de log-verosimilitud
            # exactamente plana: (log_alpha_i -> log_alpha_i + c,
            # gamma -> gamma - c) para cualquier c deja lambda_ij, mu_ij
            # invariantes. Esto se resolvió post-hoc centrando log_alpha,
            # pero el Hessiano crudo en theta_hat SIGUE teniendo esa
            # dirección con curvatura ~0 (autovalor casi nulo), lo que
            # hace que su inversa directa sea numéricamente basura
            # (SE inflados e idénticos entre equipos, como se observó
            # empíricamente: todos los SE de log_alpha ~ 5.0).
            #
            # Solución estándar en modelos con gauge freedom (idéntico al
            # tratamiento en regresión con dummies colineales): usar la
            # pseudo-inversa de Moore-Penrose, que proyecta fuera del
            # espacio nulo (la dirección no identificada) y invierte solo
            # en el subespacio donde el modelo SÍ tiene curvatura. Esto da
            # el error estándar correcto para cualquier CONTRASTE
            # identificable (ej. log_alpha_i - log_alpha_j), que es
            # justamente lo que se usa en teams_significantly_different().
            # rcond de numpy es un umbral RELATIVO: cualquier autovalor
            # singular menor que rcond * autovalor_maximo se descarta. En
            # este modelo, la dirección de gauge tiene un autovalor ~1000x
            # más chico que el resto (verificado empíricamente: ~0.002
            # contra un rango de 0.6-32 para las direcciones identificadas),
            # así que un umbral de 1e-2 aísla limpiamente esa única
            # dirección sin descartar curvatura real del modelo.
            eigvals_hess = np.linalg.eigvalsh(hess)
            cov = np.linalg.pinv(hess, rcond=1e-2)
            variances = np.diag(cov)
            std_errors = np.where(variances >= 0, np.sqrt(np.abs(variances)), np.nan)
            return std_errors, cov
        except np.linalg.LinAlgError:
            warnings.warn(
                "Hessiano singular: no se pudieron calcular errores "
                "estándar. Esto suele indicar sobre-parametrización "
                "(demasiados equipos con muy pocos partidos)."
            )
            return np.full(n_params, np.nan), None

    @staticmethod
    def _numerical_hessian(f, x, eps: float = 1e-4) -> np.ndarray:
        """Hessiano numérico vía diferencias finitas centradas (O(eps^2))."""
        n = len(x)
        hess = np.zeros((n, n))
        fx = f(x)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                val = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
                hess[i, j] = val
                hess[j, i] = val
        return hess

    def confidence_intervals(self, alpha_level: float = 0.05) -> dict:
        """
        Devuelve IC (1 - alpha_level) para alpha_i, beta_i de cada equipo
        (en espacio log, donde el MLE es aproximadamente Gaussiano) y para
        gamma, rho.

        IC = theta_hat +/- z_{1-alpha/2} * SE(theta_hat)
        """
        if self.fit_result is None or self.fit_result.std_errors is None:
            raise RuntimeError("Debe llamar fit() antes de confidence_intervals().")

        z = stats.norm.ppf(1 - alpha_level / 2)
        se = self.fit_result.std_errors
        n = self.n_teams

        idx_to_team = {v: k for k, v in self.team_index.items()}
        out = {"teams": {}, "gamma": None, "rho": None}

        log_alpha = np.log(self.fit_result.alpha)
        log_beta = np.log(self.fit_result.beta)

        for i in range(n):
            team = idx_to_team[i]
            se_a = se[i]
            se_b = se[n + i]
            out["teams"][team] = {
                "attack_log": (float(log_alpha[i] - z * se_a), float(log_alpha[i] + z * se_a)),
                "attack": (float(np.exp(log_alpha[i] - z * se_a)), float(np.exp(log_alpha[i] + z * se_a))),
                "defense_log": (float(log_beta[i] - z * se_b), float(log_beta[i] + z * se_b)),
                "defense": (float(np.exp(log_beta[i] - z * se_b)), float(np.exp(log_beta[i] + z * se_b))),
            }

        se_gamma = se[2 * n]
        # rho = tanh(rho_raw). El Hessiano numérico se calculó en el
        # espacio de rho_raw, así que se[2*n+1] es SE(rho_raw), no
        # SE(rho). Se aplica el método delta:
        #   Var(g(theta)) ~= g'(theta)^2 * Var(theta)
        #   d/dx tanh(x) = 1 - tanh(x)^2
        se_rho_raw = se[2 * n + 1]
        rho_raw_hat = np.arctanh(np.clip(self.fit_result.rho, -0.999999, 0.999999))
        dtanh = 1 - np.tanh(rho_raw_hat) ** 2
        se_rho = abs(dtanh) * se_rho_raw

        out["gamma"] = (self.fit_result.gamma - z * se_gamma, self.fit_result.gamma + z * se_gamma)
        out["rho"] = (float(self.fit_result.rho - z * se_rho), float(self.fit_result.rho + z * se_rho))
        return out

    def teams_significantly_different(self, team_a: str, team_b: str,
                                       alpha_level: float = 0.05) -> dict:
        """
        Test de Wald para H0: log_alpha_A = log_alpha_B (mismo ataque).

        z_stat = (log_alpha_A - log_alpha_B) / sqrt(Var_A + Var_B - 2*Cov_AB)

        Esto es lo que le falta a la mayoría de sistemas de rating: saber
        si "A tiene más ataque que B" es una diferencia estadísticamente
        sólida o cae dentro del ruido de muestra.
        """
        if self.fit_result is None or self.fit_result.cov_matrix is None:
            raise RuntimeError("Requiere fit() con matriz de covarianza válida.")

        i = self.team_index[team_a]
        j = self.team_index[team_b]
        cov = self.fit_result.cov_matrix
        n = self.n_teams

        var_i = cov[i, i]
        var_j = cov[j, j]
        cov_ij = cov[i, j]

        diff = np.log(self.fit_result.alpha[i]) - np.log(self.fit_result.alpha[j])
        var_diff = var_i + var_j - 2 * cov_ij
        if var_diff <= 0:
            return {"z_stat": np.nan, "p_value": np.nan, "significant": False,
                     "note": "Varianza de la diferencia no positiva; datos insuficientes."}

        z_stat = diff / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "z_stat": float(z_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha_level),
            "attack_diff_log": float(diff),
        }

    # ------------------------------------------------------------------
    # Predicción de partido: matriz completa de marcadores
    # ------------------------------------------------------------------
    def predict_match_probs(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        """
        Devuelve matriz (max_goals+1) x (max_goals+1) con P(gh=i, ga=j)
        incluyendo la corrección tau de Dixon-Coles.
        """
        if self.fit_result is None:
            raise RuntimeError("Debe llamar fit() primero.")

        r = self.fit_result
        i = self.team_index[home]
        j = self.team_index[away]

        lam = r.alpha[i] * r.beta[j] * np.exp(r.gamma)
        mu = r.alpha[j] * r.beta[i]

        goals = np.arange(0, max_goals + 1)
        pmf_x = stats.poisson.pmf(goals, lam)
        pmf_y = stats.poisson.pmf(goals, mu)

        matrix = np.outer(pmf_x, pmf_y)

        # Aplicar corrección tau solo en las 4 celdas relevantes
        x_grid, y_grid = np.meshgrid(goals, goals, indexing="ij")
        lam_grid = np.full_like(x_grid, lam, dtype=float)
        mu_grid = np.full_like(y_grid, mu, dtype=float)
        tau_grid = self._tau(x_grid.astype(float), y_grid.astype(float),
                              lam_grid, mu_grid, r.rho)
        matrix = matrix * tau_grid
        matrix = matrix / matrix.sum()  # renormalizar (tau puede alterar la masa total levemente)
        return matrix

    def match_outcome_probs(self, home: str, away: str, max_goals: int = 10) -> dict:
        """P(1), P(X), P(2) integrando la matriz de marcadores completa."""
        m = self.predict_match_probs(home, away, max_goals)
        p1, px, p2 = self.extract_1x2(m)
        return {"home_win": p1, "draw": px, "away_win": p2}

    @staticmethod
    def extract_1x2(matrix: np.ndarray) -> tuple[float, float, float]:
        """P(1), P(X), P(2) desde matriz."""
        p1 = float(np.tril(matrix, -1).sum())
        px = float(np.trace(matrix))
        p2 = float(np.triu(matrix, 1).sum())
        return p1, px, p2

    @staticmethod
    def score_matrix(lam: float, mu: float, rho: float, max_goals: Optional[int] = None) -> np.ndarray:
        """
        Matriz de probabilidades P(goals_A=i, goals_B=j) para uso externo (UnifiedEngine).
        """
        import math
        if max_goals is None:
            max_goals = max(7, int(np.ceil(max(lam, mu) + 3.0 * np.sqrt(max(lam, mu)))))
            
        goals = np.arange(0, max_goals + 1)
        pmf_x = stats.poisson.pmf(goals, lam)
        pmf_y = stats.poisson.pmf(goals, mu)
        matrix = np.outer(pmf_x, pmf_y)
        
        x_grid, y_grid = np.meshgrid(goals, goals, indexing="ij")
        lam_grid = np.full_like(x_grid, lam, dtype=float)
        mu_grid = np.full_like(y_grid, mu, dtype=float)
        tau_grid = AdvancedDixonColes._tau(x_grid.astype(float), y_grid.astype(float), lam_grid, mu_grid, rho)
        
        matrix = matrix * tau_grid
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix


# ----------------------------------------------------------------------
# EXTENSIÓN: Cópula de Frank discreta (diagnóstico / validación cruzada)
# ----------------------------------------------------------------------
class FrankCopulaDependence:
    """
    Modela dependencia entre goles local/visitante en TODO el soporte
    (no solo las 4 celdas de Dixon-Coles) usando una cópula de Frank
    discretizada. Uso recomendado: diagnóstico, NO reemplazo directo en
    producción hasta validar que mejora el RPS out-of-sample (fase3).

    C(u,v;theta) = -(1/theta) * ln[1 + (e^{-theta u}-1)(e^{-theta v}-1)/(e^{-theta}-1)]
    """

    def __init__(self, theta_frank: float = 0.0):
        self.theta_frank = theta_frank

    @staticmethod
    def _C(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        if abs(theta) < 1e-6:
            return u * v  # límite: independencia
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        den = np.exp(-theta) - 1
        return -(1.0 / theta) * np.log1p(num / den)

    def joint_pmf(self, lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
        goals = np.arange(0, max_goals + 1)
        Fx = stats.poisson.cdf(goals, lam)
        Fy = stats.poisson.cdf(goals, mu)
        Fx_prev = np.concatenate(([0.0], Fx[:-1]))
        Fy_prev = np.concatenate(([0.0], Fy[:-1]))

        U, V = np.meshgrid(Fx, Fy, indexing="ij")
        Up, Vp = np.meshgrid(Fx_prev, Fy_prev, indexing="ij")

        theta = self.theta_frank
        pmf = (self._C(U, V, theta) - self._C(Up, V, theta)
               - self._C(U, Vp, theta) + self._C(Up, Vp, theta))
        pmf = np.clip(pmf, 0, None)
        return pmf / pmf.sum()

    def fit_theta(self, matches_lam_mu_xy: list[tuple[float, float, int, int]]) -> float:
        """
        Estima theta_frank por MLE dado (lambda, mu, x, y) por partido
        (lambda, mu ya vienen del ajuste Dixon-Coles previo; solo se
        calibra la dependencia residual).
        """
        def neg_ll(theta_arr):
            theta = theta_arr[0]
            ll = 0.0
            for lam, mu, x, y in matches_lam_mu_xy:
                self.theta_frank = theta
                pmf = self.joint_pmf(lam, mu, max_goals=max(15, x + 3, y + 3))
                p = pmf[x, y] if x < pmf.shape[0] and y < pmf.shape[1] else 1e-12
                ll += np.log(max(p, 1e-12))
            return -ll

        res = optimize.minimize_scalar(lambda t: neg_ll([t]), bounds=(-10, 10), method="bounded")
        self.theta_frank = res.x
        return float(res.x)