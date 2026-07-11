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

import math
from feature_engine import EloRegistry
from advanced_dixon_coles import AdvancedDixonColes
from probability_calibrator import ProbabilityCalibrator

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
        gf_list = []
        gc_list = []
        for m in sorted_matches:
            gf = m.get('gf', m.get('gh', 0))
            gc = m.get('gc', m.get('ga', 0))
            gf_list.append(float(gf))
            gc_list.append(float(gc))
        gf_list = np.array(gf_list, dtype=float)
        gc_list = np.array(gc_list, dtype=float)
        
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
        
        goals = np.arange(0, N)
        lam_c = max(lam, 0.01)
        mu_c = max(mu, 0.01)
        
        pmf_i = stats.poisson.pmf(goals, lam_c)
        pmf_j = stats.poisson.pmf(goals, mu_c)
        matrix = np.outer(pmf_i, pmf_j)
        
        tau_grid = np.ones_like(matrix)
        if N > 0:
            tau_grid[0, 0] = 1 - lam * mu * rho
        if N > 1:
            tau_grid[0, 1] = 1 + lam * rho
            tau_grid[1, 0] = 1 + mu * rho
            tau_grid[1, 1] = 1 - rho
            
        matrix = matrix * tau_grid
        
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
        calibrator: Optional[ProbabilityCalibrator] = None,
        lambda_scale: float = 0.23,
    ):
        self.team_a = team_a
        self.team_b = team_b
        
        # Ordenar temporalmente UNA VEZ aquí (no en cada submódulo)
        self.matches_a = sorted(matches_a, key=lambda m: m.get('date', '1970-01-01'))
        self.matches_b = sorted(matches_b, key=lambda m: m.get('date', '1970-01-01'))
        
        self.venue = venue
        self.modifiers_a = modifiers_a or {}
        self.modifiers_b = modifiers_b or {}
        self.calibrator = calibrator
        self.lambda_scale = lambda_scale
        
        # Inicializar subcomponentes
        all_matches = self.matches_a + self.matches_b
        
        # Elo: extract from matches if available, else use base_elo or defaults
        self.elo_a = base_elo_a
        if self.elo_a is None and self.matches_a:
            self.elo_a = self.matches_a[-1].get('elo_pre', 1600.0)
        elif self.elo_a is None:
            self.elo_a = 1600.0
            
        self.elo_b = base_elo_b
        if self.elo_b is None and self.matches_b:
            self.elo_b = self.matches_b[-1].get('elo_pre', 1600.0)
        elif self.elo_b is None:
            self.elo_b = 1600.0
        
        # Optimizar half_life por LOO-CV si no se especifica
        if half_life is None:
            if len(all_matches) >= 8:
                hl_a = TimeWeighter.optimize_half_life(self.matches_a)
                hl_b = TimeWeighter.optimize_half_life(self.matches_b)
                self.half_life = (hl_a + hl_b) / 2.0
            else:
                self.half_life = 365.0  # Fallback si no hay suficientes partidos
        else:
            self.half_life = half_life  # Usar el valor inyectado externamente
            
        self.weighter = TimeWeighter(self.half_life)
        self.bayes = BayesianGoalRates(prior_strength=prior_strength)
        
        # Estimar ρ por MLE sobre datos combinados o usar config
        if precomputed_rho is not None:
            self.rho = precomputed_rho
        else:
            # Fallback a un prior informado si no se computó globalmente
            self.rho = -0.05
    
    def predict(self) -> dict:
        """
        Genera predicción completa con extensiones opcionales Hawkes/cuánticas/topológicas.
        """
        # 1. Priors Bayesianos basados en Elo
        registry = EloRegistry(lambda_scale=self.lambda_scale)
        elo_ratio_a = registry.expected_goal_ratio(self.elo_a, self.elo_b, self.venue)
        elo_ratio_b = registry.expected_goal_ratio(self.elo_b, self.elo_a, 'A' if self.venue == 'H' else ('H' if self.venue == 'A' else 'N'))

        lam, lam_def, lam_std, _, alpha_lam, beta_lam = self.bayes.compute_rates(
            self.matches_a, elo_ratio_a, self.weighter, self.modifiers_a
        )
        mu, mu_def, mu_std, _, alpha_mu, beta_mu = self.bayes.compute_rates(
            self.matches_b, elo_ratio_b, self.weighter, self.modifiers_b
        )

        lam_final = np.sqrt(lam * mu_def) if mu_def > 0 else lam
        mu_final = np.sqrt(mu * lam_def) if lam_def > 0 else mu

        # 3. Probabilidades del partido (matriz completa)
        matrix = AdvancedDixonColes.score_matrix(lam_final, mu_final, self.rho)
        p1, px, p2 = AdvancedDixonColes.extract_1x2(matrix)
        
        # 4. Calibración (opcional)
        if self.calibrator is not None:
            raw_probs = np.array([[p1, px, p2]])
            calibrated = self.calibrator.predict_proba(raw_probs)[0]
            p1, px, p2 = calibrated[0], calibrated[1], calibrated[2]
            # Nota: Al calibrar el 1X2, la matriz detallada pierde coherencia directa 
            # con el 1X2 final si no se re-escala. Por ahora, reportamos 1X2 calibrado.
        
        top_scores = []
        flat = matrix.flatten()
        indices = np.argsort(flat)[::-1][:5]
        for idx in indices:
            i, j = np.unravel_index(idx, matrix.shape)
            top_scores.append({
                'score': f'{i}-{j}',
                'goals_a': int(i), 'goals_b': int(j),
                'prob': float(flat[idx])
            })

        ci_lam = self.bayes.predictive_credible_interval(alpha_lam, beta_lam)
        ci_mu = self.bayes.predictive_credible_interval(alpha_mu, beta_mu)

        network_features = None
        if hasattr(self, 'pass_matrix_a') and self.pass_matrix_a is not None:
            analyzer = PassNetworkAnalyzer()
            network_features = analyzer.analyze(self.pass_matrix_a).as_feature_vector().tolist()

        return {
            'p1': round(p1, 4),
            'px': round(px, 4),
            'p2': round(p2, 4),
            'top_5_scores': top_scores,
            'lam': round(lam_final, 3),
            'mu': round(mu_final, 3),
            'lam_std': round(lam_std, 3),
            'mu_std': round(mu_std, 3),
            'ci_lam_90': (round(ci_lam[0], 3), round(ci_lam[1], 3)),
            'ci_mu_90': (round(ci_mu[0], 3), round(ci_mu[1], 3)),
            'elo_a': round(self.elo_a, 1) if isinstance(self.elo_a, float) else round(getattr(self.elo_a, 'rating', self.elo_a), 1),
            'elo_b': round(self.elo_b, 1) if isinstance(self.elo_b, float) else round(getattr(self.elo_b, 'rating', self.elo_b), 1),
            'rho': round(self.rho, 4),
            'half_life_days': round(self.half_life, 1),
            'score_matrix': matrix,
        }


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