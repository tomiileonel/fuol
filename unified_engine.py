"""
unified_engine.py — FUOL v2.0
Motor unificado de predicción: Dixon-Coles + Bayesiano calibrado + Elo/SRS + Walk-forward CV
Reemplaza: prediction_model.py, prediction_engine.py, supreme_engine.py

Arquitectura:
  1. EloRating        — Prior de fuerza basado en Elo histórico (reemplaza PageRank)
  2. BayesianRates    — Gamma-Poisson con prior_strength calibrado (reemplaza Bayesiano circular)
  3. TimeDecay        — Decaimiento temporal con half_life optimizado por LOO-CV
  4. DixonColes       — Corrección ρ estimado por MLE (reemplaza ρ=-0.04)
  5. UnifiedEngine    — Orquestador único
  6. LOOCalibrator    — Optimización de hiperparámetros walk-forward
"""

from __future__ import annotations
import numpy as np
from scipy import stats, optimize
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONSTANTE GLOBAL — centralizada, no duplicada
# ---------------------------------------------------------------------------

# Formación por defecto para selecciones no especificadas
TEAM_FORMATIONS: dict[str, str] = {
    'BÉLGICA': '4-2-3-1', 'SENEGAL': '4-3-3', 'EEUU': '4-3-3',
    'BOSNIA-HERZ.': '3-4-2-1', 'ARGENTINA': '4-3-3', 'BRASIL': '4-2-3-1',
    'ALEMANIA': '4-2-3-1', 'ESPAÑA': '4-3-3', 'FRANCIA': '4-2-3-1',
    'PORTUGAL': '4-2-3-1', 'INGLATERRA': '4-3-3', 'PAÍSES BAJOS': '4-3-3',
    'DEFAULT': '4-3-3',
}

# Elo inicial FIFA-like para WC2026 (fuente: EloRatings.net + FIFA ranking ponderado)
# Se actualiza con `EloRating.update()` después de cada partido
ELO_INITIAL: dict[str, float] = {
    'ARGENTINA': 2076, 'BRASIL': 2049, 'ALEMANIA': 1989, 'ESPAÑA': 2002,
    'FRANCIA': 1991, 'PORTUGAL': 1939, 'PAÍSES BAJOS': 1950, 'BÉLGICA': 1901,
    'SENEGAL': 1691, 'EEUU': 1742, 'BOSNIA-HERZ.': 1598,
    'DEFAULT': 1600,
}

# Avg goles WC histórico: promedio de WC 2010-2022 (2.64 goles/partido total)
AVG_GOALS_WC_HISTORICAL: float = 1.32   # por equipo por partido (2.64 / 2)


# ===========================================================================
# COMPONENTE 1: ELO RATING (reemplaza PageRank y "ratio de dificultad")
# ===========================================================================

class EloRating:
    """
    Sistema Elo adaptado para fútbol internacional.
    
    Ventajas sobre PageRank en este contexto:
    - Calibrado históricamente (K=40 para WC, K=20 competiciones ordinarias)
    - Prior informativo desde ranking FIFA oficial
    - No requiere grafo conexo (funciona con 0 partidos registrados)
    - Expected score E_A = 1 / (1 + 10^((R_B - R_A)/400)) es una logística calibrada
    
    Referencia: Hvattum & Arntzen (2010), Journal of Quantitative Analysis in Sports
    """
    
    K_WC = 40.0           # Mayor impacto en torneos mundiales
    K_QUALIF = 30.0       # Clasificatorios
    K_FRIENDLY = 20.0     # Amistosos
    HOME_ADV = 100.0      # Ventaja de local en Elo (empírica: ≈ 65-100 puntos)
    
    # Multiplicador de conversión Elo → λ/μ
    # A_ELO_LAMBDA: factor escala para convertir ventaja Elo en ratio de goles
    ELO_SCALE = 400.0
    LAMBDA_SCALE = 0.7695   # Calibrado por regresión Poisson (statsmodels GLM) sobre historial Kaggle
    
    COMP_K = {
        'WC 2026 Telemetry': K_WC,
        'FIFA World Cup': K_WC,
        'CONMEBOL': K_QUALIF,
        'UEFA': K_QUALIF,
        'AFC': K_QUALIF,
        'CAF': K_QUALIF,
        'CONCACAF': K_QUALIF,
        'friendly': K_FRIENDLY,
        'Friendly': K_FRIENDLY,
        'N': K_FRIENDLY,
    }
    
    def __init__(self, team_name: str, initial_elo: Optional[float] = None):
        self.team_name = team_name
        self.rating = initial_elo or ELO_INITIAL.get(team_name, ELO_INITIAL['DEFAULT'])
    
    def expected_score(self, opp_elo: float, venue: str = 'N') -> float:
        """P(win + 0.5*draw) desde perspectiva de este equipo."""
        home_bonus = self.HOME_ADV if venue == 'H' else (-self.HOME_ADV if venue == 'A' else 0.0)
        diff = (self.rating + home_bonus) - opp_elo
        return 1.0 / (1.0 + 10.0 ** (-diff / self.ELO_SCALE))
    
    def update(self, match: dict, opp_elo: float) -> float:
        """Actualiza rating con resultado real. Retorna el nuevo rating."""
        result = match.get('res', '')
        if result == 'W':
            s = 1.0
        elif result == 'D':
            s = 0.5
        elif result == 'L':
            s = 0.0
        else:
            gf, gc = match.get('gf', 0), match.get('gc', 0)
            s = 1.0 if gf > gc else (0.5 if gf == gc else 0.0)
        
        comp = match.get('competition', match.get('comp', 'N'))
        K = self.COMP_K.get(comp, self.K_FRIENDLY)
        E = self.expected_score(opp_elo, match.get('venue', 'N'))
        
        self.rating += K * (s - E)
        return self.rating
    
    def elo_from_matches(self, matches: list[dict]) -> float:
        """
        Reconstruye el Elo actual jugando todos los partidos secuencialmente.
        Precondición: matches ordenados cronológicamente.
        """
        for m in sorted(matches, key=lambda x: x.get('date', '1970-01-01')):
            opp_name = m.get('opponent', m.get('Opponent', 'DEFAULT'))
            opp_elo = ELO_INITIAL.get(opp_name, ELO_INITIAL['DEFAULT'])
            self.update(m, opp_elo)
        return self.rating
    
    def expected_goal_ratio(self, opp_elo: float, venue: str = 'N') -> float:
        """
        Convierte ventaja Elo en ratio esperado de goles.
        """
        E = self.expected_score(opp_elo, venue)
        # Clip para evitar log(0)
        E_clipped = np.clip(E, 0.05, 0.95)
        log_ratio = np.log(E_clipped / (1.0 - E_clipped)) * self.LAMBDA_SCALE
        return np.exp(log_ratio)


# ===========================================================================
# COMPONENTE 2: DECAIMIENTO TEMPORAL (reemplaza half_life hardcoded)
# ===========================================================================

class TimeWeighter:
    """
    Decaimiento exponencial con half_life optimizado por LOO-CV y
    soporte para quiebres estructurales dinámicos.
    """
    
    def __init__(self, half_life: float = 365.0, structural_break_date: Optional[str] = None):
        """
        half_life: en días. Valor por defecto 365.
        structural_break_date: Fecha de un evento disruptivo (ej. cambio de DT).
                               Partidos previos a esta fecha decaerán más rápido.
        """
        self.half_life = half_life
        self.structural_break_date = structural_break_date
    
    def compute_weights(self, matches: list[dict],
                        reference_date: Optional[str] = None) -> np.ndarray:
        """
        Retorna array de pesos normalizados [0,1] preservando índice original.
        """
        import datetime
        if reference_date is None:
            today = datetime.date.today()
        else:
            today = datetime.date.fromisoformat(reference_date)
            
        break_date = None
        if self.structural_break_date:
            try:
                break_date = datetime.date.fromisoformat(self.structural_break_date)
            except ValueError:
                pass
        
        weights = np.ones(len(matches))
        for i, m in enumerate(matches):
            date_str = m.get('date', None)
            if date_str:
                try:
                    match_date = datetime.date.fromisoformat(date_str[:10])
                    days_ago = max(0, (today - match_date).days)
                    
                    # Dynamic Decay: Si el partido ocurrió antes del quiebre estructural
                    current_hl = self.half_life
                    if break_date and match_date < break_date:
                        current_hl = self.half_life / 4.0  # Decaimiento agresivo (4x más rápido)
                        
                    weights[i] = 2.0 ** (-days_ago / current_hl)
                except (ValueError, TypeError):
                    weights[i] = 0.5  # sin fecha: peso moderado
            else:
                weights[i] = 0.5
        
        total = weights.sum()
        return weights / total if total > 0 else np.ones(len(matches)) / len(matches)
    
    @classmethod
    def optimize_half_life(cls, matches: list[dict],
                           candidates: list[float] = None,
                           metric: str = 'log_loss') -> float:
        """
        Leave-One-Out CV sobre partidos históricos para encontrar half_life óptimo.
        """
        if candidates is None:
            candidates = [60, 90, 120, 180, 270, 365, 500, 730]
        
        if len(matches) < 8:
            return 365.0  # fallback con datos insuficientes
        
        sorted_matches = sorted(matches, key=lambda m: m.get('date', '1970-01-01'))
        
        best_hl, best_score = 365.0, float('inf')
        
        for hl in candidates:
            weighter = cls(half_life=hl)
            scores = []
            
            for k in range(5, len(sorted_matches)):  # mínimo 5 en train
                train = sorted_matches[:k]
                test  = sorted_matches[k]
                
                ref_date = test.get('date', None)
                w = weighter.compute_weights(train, ref_date)
                
                lam_hat = np.average([m['gf'] for m in train], weights=w)
                mu_hat  = np.average([m['gc'] for m in train], weights=w)
                
                gf_true = test.get('gf', 0)
                gc_true = test.get('gc', 0)
                
                if metric == 'log_loss':
                    p_gf = stats.poisson.pmf(int(gf_true), max(lam_hat, 0.01))
                    p_gc = stats.poisson.pmf(int(gc_true), max(mu_hat, 0.01))
                    scores.append(-np.log(max(p_gf * p_gc, 1e-10)))
                elif metric == 'mse':
                    scores.append((gf_true - lam_hat)**2 + (gc_true - mu_hat)**2)
            
            avg_score = np.mean(scores) if scores else float('inf')
            if avg_score < best_score:
                best_score = avg_score
                best_hl = hl
        
        return best_hl


# ===========================================================================
# COMPONENTE 3: TASAS BAYESIANAS (reemplaza prior circular)
# ===========================================================================

class BayesianGoalRates:
    """
    Inferencia Gamma-Poisson con prior informativo desde Elo.
    """
    
    PRIOR_STRENGTH: float = 6.0  # "6 partidos imaginarios" de confianza en el prior Elo
    
    def __init__(self, prior_strength: float = None):
        self.prior_strength = prior_strength or self.PRIOR_STRENGTH
    
    def compute_rates(
        self,
        matches: list[dict],
        elo_ratio: float,
        weighter: TimeWeighter,
        modifiers: dict = None,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Retorna (lam_post, mu_post, lam_std, mu_std, alpha_att_post, beta_att_post).
        """
        if modifiers is None:
            modifiers = {}
        
        sorted_matches = sorted(matches, key=lambda m: m.get('date', '1970-01-01'))
        
        # Pesos temporales pre-calculados (NO se recalculan dentro de bootstrap)
        w = weighter.compute_weights(sorted_matches)
        
        # Media ponderada de goles históricos (equivale a MLE con decaimiento)
        gf_list = np.array([m['gf'] for m in sorted_matches], dtype=float)
        gc_list = np.array([m['gc'] for m in sorted_matches], dtype=float)
        
        # Suavizado para datos escasos
        if len(w) == 0:
            n_eff = 1.0
        else:
            n = len(w)
            n_eff = max(1.0, (np.sum(w)**2) / np.sum(w**2) if np.sum(w**2) > 0 else n)
        
        # Prior basado en Elo (venue_modifier ya está integrado dentro de elo_ratio):
        prior_lam = AVG_GOALS_WC_HISTORICAL * elo_ratio
        prior_mu  = AVG_GOALS_WC_HISTORICAL / elo_ratio
        
        # Parámetros Gamma prior
        alpha_0_att = prior_lam * self.prior_strength
        beta_0_att  = self.prior_strength
        alpha_0_def = prior_mu * self.prior_strength
        beta_0_def  = self.prior_strength
        
        # Actualización Bayesiana con datos ponderados
        sum_gf_w = np.sum(w * gf_list) * n_eff  # pseudo-suma de goles atacante
        sum_gc_w = np.sum(w * gc_list) * n_eff
        
        alpha_att_post = alpha_0_att + sum_gf_w
        beta_att_post  = beta_0_att  + n_eff
        alpha_def_post = alpha_0_def + sum_gc_w
        beta_def_post  = beta_0_def  + n_eff
        
        lam_post = alpha_att_post / beta_att_post
        mu_post  = alpha_def_post / beta_def_post
        
        # Std de distribución predictiva Binomial Negativa (corrección Issue #16)
        lam_std = np.sqrt(lam_post * (1.0 + lam_post / beta_att_post))
        mu_std  = np.sqrt(mu_post  * (1.0 + mu_post  / beta_def_post))
        
        # Modificadores externos
        inj = modifiers.get('injury_modifier', 1.0)
        fat = modifiers.get('travel_fatigue', 1.0)
        
        lam_post = np.clip(lam_post * inj * fat, 0.1, 5.0)
        mu_post  = np.clip(mu_post  * inj * fat, 0.1, 5.0)
        
        return float(lam_post), float(mu_post), float(lam_std), float(mu_std), float(alpha_att_post), float(beta_att_post)
    
    def predictive_credible_interval(
        self, alpha_post: float, beta_post: float, credibility: float = 0.90
    ) -> tuple[float, float]:
        """
        Intervalo de credibilidad correcto desde distribución Gamma posterior.
        Usa ppf de Gamma en lugar de la heurística best_prob * 0.7/1.3 (Issue #18).
        """
        lo = stats.gamma.ppf((1 - credibility) / 2, a=alpha_post, scale=1.0 / beta_post)
        hi = stats.gamma.ppf((1 + credibility) / 2, a=alpha_post, scale=1.0 / beta_post)
        return float(lo), float(hi)


# ===========================================================================
# COMPONENTE 4: DIXON-COLES CON ρ ESTIMADO POR MLE
# ===========================================================================

class DixonColes:
    """
    Modelo Dixon-Coles (1997)
    """
    
    def __init__(self):
        self.rho_: Optional[float] = None
    
    @staticmethod
    def tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
        """Corrección Dixon-Coles para bajo marcador."""
        if   i == 0 and j == 0: return 1.0 - lam * mu * rho
        elif i == 1 and j == 0: return 1.0 + mu * rho
        elif i == 0 and j == 1: return 1.0 + lam * rho
        elif i == 1 and j == 1: return 1.0 - rho
        return 1.0
    
    def fit_rho(self, all_matches: list[dict],
                lam_hat: float, mu_hat: float,
                bounds: tuple[float, float] = (-0.5, 0.2)) -> float:
        """
        MLE de ρ dado λ̂ y μ̂ con prior informado.
        Prior: ρ ~ N(-0.13, 0.08) basado en literatura Dixon-Coles.
        """
        if len(all_matches) < 15:
            return -0.13  # sin datos suficientes, usar media del prior
        
        def neg_profile_ll(rho: float) -> float:
            ll = 0.0
            for m in all_matches:
                i_raw, j_raw = m.get('gf', 0), m.get('gc', 0)
                try:
                    i, j = int(round(float(i_raw))), int(round(float(j_raw)))
                except (ValueError, TypeError):
                    continue
                
                tau_val = self.tau(i, j, lam_hat, mu_hat, rho)
                if tau_val <= 0:
                    return 1e9
                
                import math
                lam_c = max(lam_hat, 0.01)
                mu_c = max(mu_hat, 0.01)
                pmf_i = math.exp(-lam_c) * (lam_c**i) / math.factorial(i)
                pmf_j = math.exp(-mu_c) * (mu_c**j) / math.factorial(j)
                p = tau_val * pmf_i * pmf_j
                ll += np.log(max(float(p), 1e-15))
            
            # Regularizador Bayesiano: log_prior(rho)
            log_prior = stats.norm.logpdf(rho, loc=-0.13, scale=0.08)
            return -(ll + log_prior)
        
        result = optimize.minimize_scalar(
            neg_profile_ll, bounds=bounds, method='bounded',
            options={'xatol': 1e-4, 'maxiter': 200}
        )
        self.rho_ = float(np.clip(result.x, *bounds))
        return self.rho_
    
    def score_matrix(self, lam: float, mu: float, rho: float,
                     max_goals: Optional[int] = None) -> np.ndarray:
        """
        Matriz de probabilidades P(goals_A=i, goals_B=j).
        """
        if max_goals is None:
            max_goals = max(7, int(np.ceil(max(lam, mu) + 3.0 * np.sqrt(max(lam, mu)))))
        
        N = max_goals + 1
        matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                tau_val = self.tau(i, j, lam, mu, rho)
                import math
                lam_c = max(lam, 0.01)
                mu_c = max(mu, 0.01)
                pmf_i = math.exp(-lam_c) * (lam_c**i) / math.factorial(i)
                pmf_j = math.exp(-mu_c) * (mu_c**j) / math.factorial(j)
                matrix[i, j] = tau_val * pmf_i * pmf_j
        
        # Renormalizar (suma ≠ 1.0 por truncación)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        
        return matrix
    
    @staticmethod
    def extract_1x2(matrix: np.ndarray) -> tuple[float, float, float]:
        """
        P(1), P(X), P(2) desde matriz.
        """
        p1 = float(np.tril(matrix, -1).sum())   # goles_A > goles_B
        px = float(np.trace(matrix))             # empate
        p2 = float(np.triu(matrix, 1).sum())     # goles_B > goles_A
        return p1, px, p2
    
    @staticmethod
    def top_k_scores(matrix: np.ndarray, k: int = 5) -> list[dict]:
        """Top-k marcadores exactos con probabilidad."""
        flat = matrix.flatten()
        indices = np.argsort(flat)[::-1][:k]
        results = []
        for idx in indices:
            i, j = np.unravel_index(idx, matrix.shape)
            results.append({
                'score': f'{i}-{j}',
                'goals_a': int(i), 'goals_b': int(j),
                'prob': float(flat[idx])
            })
        return results


# ===========================================================================
# COMPONENTE 5: ENGINE UNIFICADO
# ===========================================================================

class UnifiedEngine:
    """
    Motor unificado que consolida los 3 engines existentes.
    """
    
    def __init__(
        self,
        team_a: str,
        team_b: str,
        matches_a: list[dict],
        matches_b: list[dict],
        venue: str = 'N',
        modifiers_a: dict = None,
        modifiers_b: dict = None,
        prior_strength: float = 6.0,
        half_life: Optional[float] = None,  # None → se optimiza por LOO-CV
        optimize_rho: bool = True,
        precomputed_rho: Optional[float] = None,
        base_elo_a: Optional[float] = None,
        base_elo_b: Optional[float] = None,
    ):
        self.team_a = team_a
        self.team_b = team_b
        
        # Ordenar temporalmente UNA VEZ aquí (no en cada submódulo)
        self.matches_a = sorted(matches_a, key=lambda m: m.get('date', '1970-01-01'))
        self.matches_b = sorted(matches_b, key=lambda m: m.get('date', '1970-01-01'))
        
        self.venue = venue
        self.modifiers_a = modifiers_a or {}
        self.modifiers_b = modifiers_b or {}
        
        # Inicializar subcomponentes
        all_matches = self.matches_a + self.matches_b
        
        # Elo: reconstruir ratings desde historial o usar base precomputada
        self.elo_a = EloRating(team_a, initial_elo=base_elo_a)
        if base_elo_a is None:
            self.elo_a.elo_from_matches(self.matches_a)
            
        self.elo_b = EloRating(team_b, initial_elo=base_elo_b)
        if base_elo_b is None:
            self.elo_b.elo_from_matches(self.matches_b)
        
        # Optimizar half_life por LOO-CV si no se especifica
        if half_life is None and len(all_matches) >= 8:
            hl_a = TimeWeighter.optimize_half_life(self.matches_a)
            hl_b = TimeWeighter.optimize_half_life(self.matches_b)
            self.half_life = (hl_a + hl_b) / 2.0
        else:
            self.half_life = half_life or 365.0
        
        self.weighter = TimeWeighter(self.half_life)
        self.bayes = BayesianGoalRates(prior_strength=prior_strength)
        self.dc = DixonColes()
        
        # Estimar ρ por MLE sobre datos combinados
        if precomputed_rho is not None:
            self.rho = precomputed_rho
        else:
            lam_all = np.mean([m['gf'] for m in all_matches]) if all_matches else 1.3
            mu_all  = np.mean([m['gc'] for m in all_matches]) if all_matches else 1.3
            
            if optimize_rho and len(all_matches) >= 15:
                self.rho = self.dc.fit_rho(all_matches, lam_all, mu_all)
            else:
                self.rho = -0.13  # fallback basado en literatura
    
    def predict(self) -> dict:
        """
        Genera predicción completa.
        """
        # 1. Ratios Elo → modificador de goles
        elo_ratio_a = self.elo_a.expected_goal_ratio(self.elo_b.rating, self.venue)
        elo_ratio_b = 1.0 / elo_ratio_a
        
        # 2. Tasas Bayesianas con prior Elo (Home advantage ya incluido en elo_ratio)
        lam, lam_def, lam_std, _, alpha_lam, beta_lam = self.bayes.compute_rates(
            self.matches_a, elo_ratio_a, self.weighter, self.modifiers_a
        )
        mu, mu_def, mu_std, _, alpha_mu, beta_mu = self.bayes.compute_rates(
            self.matches_b, elo_ratio_b, self.weighter, self.modifiers_b
        )
        
        # Ajuste cruzado defensa: λ_final = √(λ_ataque_A × λ_concedido_B)
        # Esto pondera la capacidad ofensiva A contra la capacidad defensiva B
        lam_final = np.sqrt(lam * mu_def) if mu_def > 0 else lam
        mu_final  = np.sqrt(mu * lam_def) if lam_def > 0 else mu
        
        # 3. Matriz Dixon-Coles
        matrix = self.dc.score_matrix(lam_final, mu_final, self.rho)
        
        # 4. 1X2 + marcadores
        p1, px, p2 = self.dc.extract_1x2(matrix)
        top_5 = self.dc.top_k_scores(matrix, k=5)
        
        # 5. Intervalos de credibilidad - FIX #2
        ci_lam = self.bayes.predictive_credible_interval(alpha_lam, beta_lam)
        ci_mu = self.bayes.predictive_credible_interval(alpha_mu, beta_mu)
        
        return {
            # 1X2
            'p1': round(p1, 4),
            'px': round(px, 4),
            'p2': round(p2, 4),
            
            # Marcadores exactos
            'top_5_scores': top_5,
            
            # Goles esperados
            'lam': round(lam_final, 3),
            'mu':  round(mu_final, 3),
            'lam_std': round(lam_std, 3),
            'mu_std':  round(mu_std, 3),
            
            # Intervalos de credibilidad 90% para λ y μ
            'ci_lam_90': (round(ci_lam[0], 3), round(ci_lam[1], 3)),
            'ci_mu_90':  (round(ci_mu[0], 3), round(ci_mu[1], 3)),
            
            # Diagnóstico del modelo
            'elo_a': round(self.elo_a.rating, 1),
            'elo_b': round(self.elo_b.rating, 1),
            'rho':   round(self.rho, 4),
            'half_life_days': round(self.half_life, 1),
            
            # Matriz completa (para visualización)
            'score_matrix': matrix,
        }


# ===========================================================================
# COMPONENTE 6: BACKTESTING WALK-FORWARD (Fase 4)
# ===========================================================================

class WalkForwardBacktester:
    """
    Protocolo de evaluación walk-forward correcto.
    """
    
    def __init__(self, min_train_size: int = 8):
        self.min_train_size = min_train_size
        self.results: list[dict] = []
    
    @staticmethod
    def rps(p_pred: np.ndarray, outcome: int) -> float:
        """
        Ranked Probability Score para predicciones ordinales 1X2.
        """
        cdf_pred = np.cumsum(p_pred)[:2]  # CDF en puntos 1 y X
        cdf_true = np.cumsum(np.eye(3)[outcome])[:2]
        return float(np.mean((cdf_pred - cdf_true) ** 2))
    
    def evaluate_match(
        self,
        actual_gf: int, actual_gc: int,
        pred: dict
    ) -> dict:
        """Calcula todas las métricas para un partido dado."""
        p1, px, p2 = pred['p1'], pred['px'], pred['p2']
        p_vec = np.array([p1, px, p2])
        p_vec = np.clip(p_vec, 1e-10, 1.0)
        p_vec /= p_vec.sum()
        
        # Resultado real
        if actual_gf > actual_gc:
            y_true = np.array([1.0, 0.0, 0.0])
            outcome_idx = 0
        elif actual_gf == actual_gc:
            y_true = np.array([0.0, 1.0, 0.0])
            outcome_idx = 1
        else:
            y_true = np.array([0.0, 0.0, 1.0])
            outcome_idx = 2
        
        brier = float(np.sum((p_vec - y_true) ** 2))
        log_loss = float(-np.log(p_vec[outcome_idx]))
        rps_val = self.rps(p_vec, outcome_idx)
        hit = int(np.argmax(p_vec) == outcome_idx)
        
        return {
            'brier': brier, 'log_loss': log_loss,
            'rps': rps_val, 'hit': hit,
            'p1': float(p_vec[0]), 'px': float(p_vec[1]), 'p2': float(p_vec[2]),
            'y_true': int(outcome_idx),
            'lam_pred': pred.get('lam', 0), 'mu_pred': pred.get('mu', 0),
            'gf_true': actual_gf, 'gc_true': actual_gc,
        }
    
    def aggregate(self) -> dict:
        """Agrega métricas sobre todos los partidos evaluados."""
        if not self.results:
            return {}
        
        n = len(self.results)
        bs = np.mean([r['brier'] for r in self.results])
        ll = np.mean([r['log_loss'] for r in self.results])
        rps = np.mean([r['rps'] for r in self.results])
        hit = np.mean([r['hit'] for r in self.results])
        
        # Calibración: dividir predicciones en deciles y comparar con frecuencia real
        p1_preds = np.array([r['p1'] for r in self.results])
        y_true_1 = np.array([1.0 if r['y_true'] == 0 else 0.0 for r in self.results])
        
        calibration_bins = {}
        for bin_lo, bin_hi in [(i/10, (i+1)/10) for i in range(10)]:
            mask = (p1_preds >= bin_lo) & (p1_preds < bin_hi)
            if mask.sum() > 0:
                calibration_bins[f'{bin_lo:.1f}-{bin_hi:.1f}'] = {
                    'n': int(mask.sum()),
                    'pred_mean': float(p1_preds[mask].mean()),
                    'actual_freq': float(y_true_1[mask].mean()),
                }
        
        # ECE (Expected Calibration Error)
        ece = sum(
            abs(v['pred_mean'] - v['actual_freq']) * v['n'] / n
            for v in calibration_bins.values()
        )
        
        rps_by_match = {
            r['match'].get('date', f'unknown_{i}'): r['rps']
            for i, r in enumerate(self.results)
        }
        
        return {
            'n_matches': n,
            'brier_score': round(bs, 4),
            'log_loss': round(ll, 4),
            'rps': round(rps, 4),
            'hit_rate_pct': round(hit * 100, 1),
            'ece': round(ece, 4),
            'calibration': calibration_bins,
            'rps_by_match': rps_by_match,
        }
    
    def run_walkforward(
        self,
        team_a: str, team_b: str,
        all_matches_a: list[dict], all_matches_b: list[dict],
        venue: str = 'N',
        half_life: Optional[float] = None,
        optimize_rho: bool = True,
        eval_start_idx: Optional[int] = None,
    ) -> dict:
        """
        Simulación walk-forward completa.
        """
        if half_life is None:
            raise AssertionError(
                "run_walkforward requiere half_life explícito. "
                "half_life=None dispararía auto-optimización dentro de cada "
                "fold del walk-forward, filtrando información del propio "
                "fold hacia la selección del hiperparámetro (double dipping). "
                "Corré fase2_barrido.py para elegir un half_life primero."
            )

        sorted_a = sorted(all_matches_a, key=lambda m: m.get('date', '1970-01-01'))
        start_idx = max(self.min_train_size, eval_start_idx if eval_start_idx is not None else 0)
        
        for k in range(start_idx, len(sorted_a)):
            train_a = sorted_a[:k]
            test_m  = sorted_a[k]
            
            # matches_b estrictamente anteriores a test_date
            test_date = test_m.get('date', '9999-99-99')
            train_b = [m for m in all_matches_b if m.get('date', '9999-99-99') < test_date]
            
            if len(train_b) < 3:
                continue  # insuficientes datos del oponente
            
            try:
                engine = UnifiedEngine(
                    team_a=team_a, team_b=team_b,
                    matches_a=train_a, matches_b=train_b,
                    venue=venue,
                    half_life=half_life,
                    optimize_rho=optimize_rho,
                )
                pred = engine.predict()
                metrics = self.evaluate_match(
                    int(test_m.get('gf', 0)), int(test_m.get('gc', 0)), pred
                )
                metrics['match'] = test_m
                self.results.append(metrics)
            except Exception as e:
                print(f"[WalkForward] Error en partido {k}: {e}")
                continue
        
        return self.aggregate()


# ===========================================================================
# INTERFACE PÚBLICA — compatible con la API existente de supreme_engine
# ===========================================================================

def run_prediction(
    team_a: str, team_b: str,
    matches_a: list[dict], matches_b: list[dict],
    venue: str = 'N',
    modifiers_a: dict = None, modifiers_b: dict = None,
    verbose: bool = True,
    half_life: Optional[float] = None,
    precomputed_rho: Optional[float] = None,
    run_backtest: bool = True,
) -> dict:
    """
    Wrapper de alto nivel, compatible con el flujo existente.
    """
    engine = UnifiedEngine(
        team_a=team_a, team_b=team_b,
        matches_a=matches_a, matches_b=matches_b,
        venue=venue,
        modifiers_a=modifiers_a or {},
        modifiers_b=modifiers_b or {},
        half_life=half_life,
        precomputed_rho=precomputed_rho,
    )
    result = engine.predict()
    
    # Conectamos WalkForwardBacktester al flujo principal
    backtest_metrics = {}
    if run_backtest and len(matches_a) >= 8 and len(matches_b) >= 8:
        bt = WalkForwardBacktester(min_train_size=5)
        backtest_metrics = bt.run_walkforward(
            team_a, team_b, matches_a, matches_b, venue=venue
        )
        result['backtest_metrics'] = backtest_metrics
    
    if verbose:
        print(f"\n{'='*55}")
        print(f"  {team_a} vs {team_b} (Venue: {venue})")
        print(f"{'='*55}")
        print(f"  Elo: {result['elo_a']} vs {result['elo_b']}")
        print(f"  rho (Dixon-Coles MLE): {result['rho']}")
        print(f"  half_life (LOO-CV):  {result['half_life_days']} días")
        print(f"  lam = {result['lam']} ± {result['lam_std']}  (CI90: {result['ci_lam_90']})")
        print(f"  mu = {result['mu']} ± {result['mu_std']}  (CI90: {result['ci_mu_90']})")
        print(f"\n  P(1)={result['p1']:.1%}  P(X)={result['px']:.1%}  P(2)={result['p2']:.1%}")
        
        if backtest_metrics:
            print(f"\n  [Walk-Forward Validation]")
            print(f"  Matches Eval: {backtest_metrics.get('n_matches', 0)}")
            print(f"  Brier Score:  {backtest_metrics.get('brier_score', 'N/A')}")
            print(f"  Log-Loss:     {backtest_metrics.get('log_loss', 'N/A')}")
            print(f"  RPS:          {backtest_metrics.get('rps', 'N/A')}")
            print(f"  Hit Rate:     {backtest_metrics.get('hit_rate_pct', 'N/A')}%")
            
        print(f"\n  Top-5 marcadores exactos:")
        for s in result['top_5_scores']:
            print(f"    {s['score']}  →  {s['prob']:.2%}")
    
    return result


if __name__ == '__main__':
    # Test block
    print("Testing UnifiedEngine...")
    from performance_tracker import ModelTelemetry
    telemetry = ModelTelemetry()
    belgium = [{"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0}, {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3}, {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1}]
    senegal = [{"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1}, {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0}, {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0}]
    
    bel_dyn = telemetry.synchronize_knowledge_base("BÉLGICA", belgium)
    sen_dyn = telemetry.synchronize_knowledge_base("SENEGAL", senegal)
    
    run_prediction("BÉLGICA", "SENEGAL", bel_dyn, sen_dyn, venue='N')
    
    print("\n--- WALK-FORWARD BACKTESTING ---")
    bt = WalkForwardBacktester(min_train_size=8)
    metrics = bt.run_walkforward(
        team_a='BÉLGICA', team_b='SENEGAL',
        all_matches_a=bel_dyn,
        all_matches_b=sen_dyn,
        venue='N'
    )
    print(metrics)
