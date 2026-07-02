#!/usr/bin/env python3
"""
Motor Matemático de Predicción Futbolística
============================================
Modelos: Dixon-Coles Poisson | Bayesiano Gamma-Poisson | Teoría de Grafos (PageRank) | Cadenas de Markov
Partidos:
  1) Bélgica vs Senegal — Ronda de 32, FIFA World Cup 2026
  2) EEUU vs Bosnia y Herzegovina — Ronda de 32, FIFA World Cup 2026
"""

import numpy as np
from scipy.stats import poisson, gamma as gamma_dist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# DATOS HISTÓRICOS RECOPILADOS (fecha, rival, condición, competición, gf, gc, resultado)
# Condición: H=Home, A=Away, N=Neutral
# Resultado: W=Win, D=Draw, L=Loss
# ==============================================================================

# --- BÉLGICA (últimos ~30 partidos, 2024-2026) ---
belgium_matches = [
    # 2024
    {"date": "2024-03-22", "opponent": "Montenegro", "venue": "H", "comp": "EURO Q", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2024-03-26", "opponent": "Ireland", "venue": "A", "comp": "EURO Q", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2024-06-08", "opponent": "Luxembourg", "venue": "H", "comp": "Friendly", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2024-06-11", "opponent": "Montenegro", "venue": "A", "comp": "Friendly", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-06-17", "opponent": "Slovakia", "venue": "N", "comp": "EURO 2024", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-06-22", "opponent": "Romania", "venue": "N", "comp": "EURO 2024", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-06-26", "opponent": "Ukraine", "venue": "N", "comp": "EURO 2024", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2024-07-01", "opponent": "France", "venue": "N", "comp": "EURO 2024", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-09-06", "opponent": "Israel", "venue": "N", "comp": "Nations League", "gf": 3, "gc": 1, "res": "W"},
    {"date": "2024-09-09", "opponent": "France", "venue": "A", "comp": "Nations League", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-10-11", "opponent": "Italy", "venue": "A", "comp": "Nations League", "gf": 2, "gc": 2, "res": "D"},
    {"date": "2024-10-14", "opponent": "France", "venue": "H", "comp": "Nations League", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2024-11-14", "opponent": "Italy", "venue": "H", "comp": "Nations League", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-11-17", "opponent": "Israel", "venue": "N", "comp": "Nations League", "gf": 1, "gc": 0, "res": "W"},
    # 2025
    {"date": "2025-03-20", "opponent": "England", "venue": "A", "comp": "Friendly", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2025-03-24", "opponent": "Ireland", "venue": "H", "comp": "Friendly", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2025-06-06", "opponent": "North Macedonia", "venue": "A", "comp": "WC Qual", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2025-06-09", "opponent": "Wales", "venue": "H", "comp": "WC Qual", "gf": 4, "gc": 3, "res": "W"},
    {"date": "2025-09-06", "opponent": "Wales", "venue": "A", "comp": "WC Qual", "gf": 2, "gc": 1, "res": "W"},
    {"date": "2025-09-09", "opponent": "North Macedonia", "venue": "H", "comp": "WC Qual", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2025-10-10", "opponent": "Kazakhstan", "venue": "A", "comp": "WC Qual", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2025-10-13", "opponent": "Kazakhstan", "venue": "H", "comp": "WC Qual", "gf": 4, "gc": 0, "res": "W"},
    {"date": "2025-11-14", "opponent": "Montenegro", "venue": "H", "comp": "WC Qual", "gf": 3, "gc": 1, "res": "W"},
    {"date": "2025-11-17", "opponent": "Montenegro", "venue": "A", "comp": "WC Qual", "gf": 1, "gc": 0, "res": "W"},
    # 2026
    {"date": "2026-03-28", "opponent": "USA", "venue": "N", "comp": "Friendly", "gf": 5, "gc": 2, "res": "W"},
    {"date": "2026-06-02", "opponent": "Croatia", "venue": "A", "comp": "Friendly", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2026-06-06", "opponent": "Tunisia", "venue": "H", "comp": "Friendly", "gf": 5, "gc": 0, "res": "W"},
    {"date": "2026-06-15", "opponent": "Egypt", "venue": "N", "comp": "WC 2026", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2026-06-21", "opponent": "Iran", "venue": "N", "comp": "WC 2026", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2026-06-26", "opponent": "New Zealand", "venue": "N", "comp": "WC 2026", "gf": 5, "gc": 1, "res": "W"},
]

# --- SENEGAL (últimos ~30 partidos, 2024-2026) ---
senegal_matches = [
    # 2024 AFCON Qualifiers
    {"date": "2024-09-06", "opponent": "Burkina Faso", "venue": "H", "comp": "AFCON Q", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-09-09", "opponent": "Burundi", "venue": "A", "comp": "AFCON Q", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2024-10-11", "opponent": "Malawi", "venue": "H", "comp": "AFCON Q", "gf": 4, "gc": 0, "res": "W"},
    {"date": "2024-10-15", "opponent": "Malawi", "venue": "A", "comp": "AFCON Q", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2024-11-14", "opponent": "Burkina Faso", "venue": "A", "comp": "AFCON Q", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2024-11-19", "opponent": "Burundi", "venue": "H", "comp": "AFCON Q", "gf": 2, "gc": 0, "res": "W"},
    # 2024 WC Qualifiers Africa
    {"date": "2024-03-21", "opponent": "Congo DR", "venue": "H", "comp": "WC Qual", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-03-26", "opponent": "Mauritania", "venue": "A", "comp": "WC Qual", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-06-06", "opponent": "Congo DR", "venue": "A", "comp": "WC Qual", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2024-06-10", "opponent": "Mauritania", "venue": "H", "comp": "WC Qual", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-06-14", "opponent": "Togo", "venue": "A", "comp": "WC Qual", "gf": 1, "gc": 0, "res": "W"},
    # 2025 AFCON (Morocco)
    {"date": "2025-12-23", "opponent": "Botswana", "venue": "N", "comp": "AFCON 2025", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2025-12-27", "opponent": "DR Congo", "venue": "N", "comp": "AFCON 2025", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2025-12-30", "opponent": "Benin", "venue": "N", "comp": "AFCON 2025", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2026-01-14", "opponent": "Egypt", "venue": "N", "comp": "AFCON 2025", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2026-01-18", "opponent": "Morocco", "venue": "N", "comp": "AFCON 2025", "gf": 1, "gc": 0, "res": "W"},
    # 2025 Friendlies
    {"date": "2025-11-15", "opponent": "Brazil", "venue": "A", "comp": "Friendly", "gf": 0, "gc": 2, "res": "L"},
    {"date": "2025-11-18", "opponent": "Kenya", "venue": "H", "comp": "Friendly", "gf": 8, "gc": 0, "res": "W"},
    # 2026 pre-WC friendlies + WC
    {"date": "2026-05-31", "opponent": "USA", "venue": "A", "comp": "Friendly", "gf": 2, "gc": 3, "res": "L"},
    {"date": "2026-06-09", "opponent": "Saudi Arabia", "venue": "N", "comp": "Friendly", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2026-06-16", "opponent": "France", "venue": "N", "comp": "WC 2026", "gf": 1, "gc": 3, "res": "L"},
    {"date": "2026-06-22", "opponent": "Norway", "venue": "N", "comp": "WC 2026", "gf": 2, "gc": 3, "res": "L"},
    {"date": "2026-06-26", "opponent": "Iraq", "venue": "N", "comp": "WC 2026", "gf": 5, "gc": 0, "res": "W"},
]

# --- EEUU (últimos ~30 partidos, 2024-2026) ---
usa_matches = [
    # 2024
    {"date": "2024-01-20", "opponent": "Slovenia", "venue": "N", "comp": "Friendly", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-03-21", "opponent": "Jamaica", "venue": "H", "comp": "CNL", "gf": 3, "gc": 1, "res": "W"},
    {"date": "2024-03-24", "opponent": "Mexico", "venue": "N", "comp": "CNL", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-06-01", "opponent": "Colombia", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 5, "res": "L"},
    {"date": "2024-06-05", "opponent": "Brazil", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-06-15", "opponent": "Bolivia", "venue": "N", "comp": "Copa America", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2024-06-24", "opponent": "Panama", "venue": "N", "comp": "Copa America", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2024-06-27", "opponent": "Uruguay", "venue": "N", "comp": "Copa America", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-09-07", "opponent": "Canada", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2024-09-10", "opponent": "New Zealand", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-10-12", "opponent": "Panama", "venue": "A", "comp": "Friendly", "gf": 0, "gc": 1, "res": "L"},
    {"date": "2024-10-15", "opponent": "Mexico", "venue": "H", "comp": "Friendly", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2024-11-18", "opponent": "Jamaica", "venue": "H", "comp": "CNL", "gf": 4, "gc": 0, "res": "W"},
    # 2025
    {"date": "2025-03-20", "opponent": "Venezuela", "venue": "H", "comp": "Friendly", "gf": 3, "gc": 1, "res": "W"},
    {"date": "2025-03-25", "opponent": "Costa Rica", "venue": "A", "comp": "CNL", "gf": 3, "gc": 0, "res": "W"},
    {"date": "2025-06-05", "opponent": "Trinidad and Tobago", "venue": "H", "comp": "Gold Cup", "gf": 5, "gc": 0, "res": "W"},
    {"date": "2025-06-11", "opponent": "Saudi Arabia", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2025-06-14", "opponent": "Haiti", "venue": "N", "comp": "Gold Cup", "gf": 2, "gc": 1, "res": "W"},
    {"date": "2025-09-05", "opponent": "Turkey", "venue": "A", "comp": "Friendly", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2025-09-09", "opponent": "Switzerland", "venue": "A", "comp": "Friendly", "gf": 0, "gc": 4, "res": "L"},
    {"date": "2025-10-09", "opponent": "Japan", "venue": "H", "comp": "Friendly", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2025-10-14", "opponent": "Australia", "venue": "H", "comp": "Friendly", "gf": 2, "gc": 1, "res": "W"},
    {"date": "2025-11-14", "opponent": "Mexico", "venue": "A", "comp": "Friendly", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2025-11-18", "opponent": "Uruguay", "venue": "H", "comp": "Friendly", "gf": 5, "gc": 1, "res": "W"},
    # 2026
    {"date": "2026-03-28", "opponent": "Belgium", "venue": "N", "comp": "Friendly", "gf": 2, "gc": 5, "res": "L"},
    {"date": "2026-03-31", "opponent": "Portugal", "venue": "H", "comp": "Friendly", "gf": 0, "gc": 2, "res": "L"},
    {"date": "2026-05-31", "opponent": "Senegal", "venue": "H", "comp": "Friendly", "gf": 3, "gc": 2, "res": "W"},
    {"date": "2026-06-06", "opponent": "Germany", "venue": "H", "comp": "Friendly", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2026-06-12", "opponent": "Paraguay", "venue": "N", "comp": "WC 2026", "gf": 4, "gc": 1, "res": "W"},
    {"date": "2026-06-19", "opponent": "Australia", "venue": "N", "comp": "WC 2026", "gf": 2, "gc": 0, "res": "W"},
    {"date": "2026-06-25", "opponent": "Turkey", "venue": "N", "comp": "WC 2026", "gf": 2, "gc": 3, "res": "L"},
]

# --- BOSNIA Y HERZEGOVINA (últimos ~25 partidos, 2024-2026) ---
bosnia_matches = [
    # 2024 Nations League
    {"date": "2024-09-05", "opponent": "Netherlands", "venue": "A", "comp": "Nations League", "gf": 1, "gc": 5, "res": "L"},
    {"date": "2024-09-08", "opponent": "Hungary", "venue": "H", "comp": "Nations League", "gf": 0, "gc": 2, "res": "L"},
    {"date": "2024-10-10", "opponent": "Germany", "venue": "A", "comp": "Nations League", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2024-10-13", "opponent": "Hungary", "venue": "A", "comp": "Nations League", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2024-11-14", "opponent": "Germany", "venue": "H", "comp": "Nations League", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2024-11-17", "opponent": "Netherlands", "venue": "H", "comp": "Nations League", "gf": 1, "gc": 3, "res": "L"},
    # 2025 WC Qualifiers
    {"date": "2025-03-21", "opponent": "Romania", "venue": "H", "comp": "WC Qual", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2025-03-24", "opponent": "Cyprus", "venue": "A", "comp": "WC Qual", "gf": 2, "gc": 1, "res": "W"},
    {"date": "2025-06-07", "opponent": "San Marino", "venue": "A", "comp": "WC Qual", "gf": 1, "gc": 0, "res": "W"},
    {"date": "2025-09-06", "opponent": "San Marino", "venue": "H", "comp": "WC Qual", "gf": 6, "gc": 0, "res": "W"},
    {"date": "2025-09-09", "opponent": "Austria", "venue": "A", "comp": "WC Qual", "gf": 1, "gc": 2, "res": "L"},
    {"date": "2025-10-09", "opponent": "Cyprus", "venue": "H", "comp": "WC Qual", "gf": 2, "gc": 2, "res": "D"},
    {"date": "2025-11-15", "opponent": "Romania", "venue": "A", "comp": "WC Qual", "gf": 3, "gc": 1, "res": "W"},
    {"date": "2025-11-18", "opponent": "Austria", "venue": "H", "comp": "WC Qual", "gf": 1, "gc": 1, "res": "D"},
    # 2026 Play-offs + friendlies + WC
    {"date": "2026-03-26", "opponent": "Wales", "venue": "N", "comp": "WC Playoff", "gf": 1, "gc": 1, "res": "W"},  # pens
    {"date": "2026-03-31", "opponent": "Italy", "venue": "A", "comp": "WC Playoff", "gf": 1, "gc": 1, "res": "W"},  # pens
    {"date": "2026-05-29", "opponent": "North Macedonia", "venue": "H", "comp": "Friendly", "gf": 0, "gc": 0, "res": "D"},
    {"date": "2026-06-06", "opponent": "Panama", "venue": "N", "comp": "Friendly", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2026-06-12", "opponent": "Canada", "venue": "N", "comp": "WC 2026", "gf": 1, "gc": 1, "res": "D"},
    {"date": "2026-06-18", "opponent": "Switzerland", "venue": "N", "comp": "WC 2026", "gf": 1, "gc": 4, "res": "L"},
    {"date": "2026-06-24", "opponent": "Qatar", "venue": "N", "comp": "WC 2026", "gf": 3, "gc": 1, "res": "W"},
]

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def time_decay_weights(n, half_life=10):
    """Peso exponencial: partidos recientes pesan más. half_life en partidos."""
    weights = np.array([np.exp(-np.log(2) * (n - 1 - i) / half_life) for i in range(n)])
    return weights / weights.sum()

def compute_attack_defense(matches, weights):
    """Calcula tasas ponderadas de ataque (λ) y defensa (μ)."""
    gf = np.array([m["gf"] for m in matches], dtype=float)
    gc = np.array([m["gc"] for m in matches], dtype=float)
    attack = np.average(gf, weights=weights)
    defense = np.average(gc, weights=weights)
    return attack, defense

# ==============================================================================
# PASO 1: DISTRIBUCIÓN DE POISSON (MODELO DIXON-COLES)
# ==============================================================================

def dixon_coles_tau(x, y, lambda_home, mu_away, rho):
    """Corrección Dixon-Coles para marcadores bajos."""
    if x == 0 and y == 0:
        return 1 - lambda_home * mu_away * rho
    elif x == 1 and y == 0:
        return 1 + mu_away * rho
    elif x == 0 and y == 1:
        return 1 + lambda_home * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def build_score_matrix(lambda_home, mu_away, rho, max_goals=6):
    """Genera la matriz completa de probabilidades de marcador con corrección DC."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = dixon_coles_tau(i, j, lambda_home, mu_away, rho)
            matrix[i][j] = tau * poisson.pmf(i, lambda_home) * poisson.pmf(j, mu_away)
    # Normalizar
    matrix /= matrix.sum()
    return matrix

def extract_1x2(matrix):
    """Extrae probabilidades 1X2 de la matriz de marcadores."""
    max_g = matrix.shape[0]
    p_home = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i > j)
    p_draw = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i == j)
    p_away = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i < j)
    return p_home, p_draw, p_away

# ==============================================================================
# PASO 2: ACTUALIZACIÓN BAYESIANA (GAMMA-POISSON)
# ==============================================================================

def bayesian_update(prior_rate, prior_n, recent_goals, recent_n):
    """
    Actualización conjugada Gamma-Poisson.
    Prior: Gamma(alpha=prior_rate*prior_n, beta=prior_n)
    Posterior: Gamma(alpha + sum(goals), beta + recent_n)
    """
    alpha_prior = prior_rate * prior_n
    beta_prior = prior_n
    alpha_post = alpha_prior + sum(recent_goals)
    beta_post = beta_prior + recent_n
    posterior_mean = alpha_post / beta_post
    posterior_std = np.sqrt(alpha_post) / beta_post
    return posterior_mean, posterior_std

# ==============================================================================
# PASO 3: TEORÍA DE GRAFOS (PageRank sobre resultados)
# ==============================================================================

def build_result_graph_pagerank(matches, all_opponents_pool):
    """
    Escenario B: Grafo de resultados.
    Nodos = rivales enfrentados. Aristas = diferencia de goles ponderada.
    PageRank -> fuerza del calendario.
    """
    # Asignar un peso base a cada rival basado en la diferencia de goles
    opponent_scores = {}
    opponent_counts = {}
    for m in matches:
        opp = m["opponent"]
        diff = m["gf"] - m["gc"]
        if opp not in opponent_scores:
            opponent_scores[opp] = 0
            opponent_counts[opp] = 0
        opponent_scores[opp] += diff
        opponent_counts[opp] += 1
    
    # Clasificar rivales por "nivel" (proxy: dificultad observada)
    # Mayor diferencia negativa = rival más fuerte
    top_teams = {"France", "Germany", "Brazil", "England", "Italy", "Netherlands", 
                 "Spain", "Portugal", "Argentina", "Croatia", "Belgium", "USA",
                 "Switzerland", "Uruguay", "Turkey", "Colombia", "Mexico"}
    mid_teams = {"Romania", "Austria", "Hungary", "Egypt", "Iran", "Norway",
                 "Morocco", "Senegal", "Japan", "Australia", "Wales", "Panama",
                 "Canada", "DR Congo", "Bosnia", "Paraguay"}
    
    # Calcular fuerza del calendario
    total_weight = 0
    quality_sum = 0
    for m in matches:
        opp = m["opponent"]
        if opp in top_teams:
            q = 1.3
        elif opp in mid_teams:
            q = 1.0
        else:
            q = 0.7
        total_weight += 1
        quality_sum += q
    
    calendar_strength = quality_sum / total_weight if total_weight > 0 else 1.0
    return calendar_strength

# ==============================================================================
# PASO 4: CADENAS DE MARKOV
# ==============================================================================

def markov_1x2(matches):
    """
    Cadenas de Markov sobre estados {Local_Adelante, Empate, Visitante_Adelante}.
    Usando transición resultado-al-descanso → resultado-final (aproximación).
    Sin minuto de gol, usamos la distribución empírica de estados finales.
    """
    # Estados: 0=Empate, 1=Victoria equipo, 2=Derrota equipo
    states = {"W": 1, "D": 0, "L": 2}
    
    # Construir matriz de transición empírica
    # Usamos pares consecutivos de partidos como proxy de transiciones
    n = len(matches)
    trans = np.ones((3, 3)) * 0.1  # suavizado Laplace
    
    for i in range(n - 1):
        s_from = states[matches[i]["res"]]
        s_to = states[matches[i + 1]["res"]]
        trans[s_from][s_to] += 1
    
    # Normalizar filas
    for i in range(3):
        row_sum = trans[i].sum()
        if row_sum > 0:
            trans[i] /= row_sum
    
    # Distribución estacionaria
    eigenvalues, eigenvectors = np.linalg.eig(trans.T)
    # Buscar autovalor ~1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    
    # stationary[0]=Empate, stationary[1]=Victoria, stationary[2]=Derrota
    return stationary[1], stationary[0], stationary[2]  # W, D, L

# ==============================================================================
# PASO 5: BOOTSTRAP E INTEGRACIÓN
# ==============================================================================

def bootstrap_analysis(matches_home, matches_away, rho, n_bootstrap=2000, graph_adj_home=1.0, graph_adj_away=1.0):
    """
    Bootstrap sobre partidos históricos para obtener intervalos de confianza.
    """
    n_home = len(matches_home)
    n_away = len(matches_away)
    
    results_1x2 = []
    lambdas = []
    mus = []
    best_scores = []
    
    for _ in range(n_bootstrap):
        # Remuestreo con reemplazo
        idx_h = np.random.choice(n_home, size=n_home, replace=True)
        idx_a = np.random.choice(n_away, size=n_away, replace=True)
        
        sample_home = [matches_home[i] for i in idx_h]
        sample_away = [matches_away[i] for i in idx_a]
        
        w_h = time_decay_weights(n_home, half_life=10)
        w_a = time_decay_weights(n_away, half_life=10)
        
        att_h, def_h = compute_attack_defense(sample_home, w_h)
        att_a, def_a = compute_attack_defense(sample_away, w_a)
        
        # λ_home = ataque_local * debilidad_defensa_visitante
        # Ponderamos con la media global
        avg_goals = 1.35  # media aprox en WC
        
        lambda_h = att_h * (def_a / avg_goals) * graph_adj_home
        mu_a = att_a * (def_h / avg_goals) * graph_adj_away
        
        # Limitar a rangos razonables
        lambda_h = np.clip(lambda_h, 0.3, 4.5)
        mu_a = np.clip(mu_a, 0.3, 4.5)
        
        matrix = build_score_matrix(lambda_h, mu_a, rho, max_goals=6)
        p1, px, p2 = extract_1x2(matrix)
        
        results_1x2.append((p1, px, p2))
        lambdas.append(lambda_h)
        mus.append(mu_a)
        
        # Marcador más probable
        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        best_scores.append((best_idx[0], best_idx[1], matrix[best_idx]))
    
    return results_1x2, lambdas, mus, best_scores

def analyze_match(team_a_name, team_b_name, matches_a, matches_b, is_neutral=True):
    """Análisis completo de un partido."""
    print(f"\n{'='*80}")
    print(f"  ANÁLISIS: {team_a_name} vs {team_b_name}")
    print(f"  FIFA World Cup 2026 — Ronda de 32 | Sede: Neutral (USA)")
    print(f"{'='*80}")
    
    n_a = len(matches_a)
    n_b = len(matches_b)
    
    print(f"\n  Partidos analizados: {team_a_name}={n_a} | {team_b_name}={n_b}")
    print(f"  NOTA: xG, posesión%, remates y minuto de gol NO disponibles.")
    print(f"         Esto limita la granularidad de Markov (se usa Plan B) y la")
    print(f"         precisión del modelo. Las estimaciones se basan solo en goles reales.")
    
    # ---- PASO 1: POISSON DIXON-COLES ----
    print(f"\n{'─'*80}")
    print(f"  PASO 1: DISTRIBUCIÓN DE POISSON (DIXON-COLES)")
    print(f"{'─'*80}")
    
    w_a = time_decay_weights(n_a, half_life=10)
    w_b = time_decay_weights(n_b, half_life=10)
    
    att_a, def_a = compute_attack_defense(matches_a, w_a)
    att_b, def_b = compute_attack_defense(matches_b, w_b)
    
    avg_goals = 1.35  # media WC
    
    lambda_a = att_a * (def_b / avg_goals)
    mu_b = att_b * (def_a / avg_goals)
    
    # Ajuste por neutralidad (ambos en campo neutral, sin ventaja de localía)
    if is_neutral:
        lambda_a *= 0.97  # leve reducción por no ser local
        mu_b *= 0.97
    
    # Estimar rho (correlación Dixon-Coles)
    rho = -0.04  # Valor típico para torneos internacionales
    
    print(f"  Factor time-decay: half_life = 10 partidos")
    print(f"  Tasas iniciales (Poisson puro):")
    print(f"    {team_a_name}: Ataque={att_a:.3f} | Defensa(gc)={def_a:.3f}")
    print(f"    {team_b_name}: Ataque={att_b:.3f} | Defensa(gc)={def_b:.3f}")
    print(f"  λ({team_a_name})={lambda_a:.3f} | μ({team_b_name})={mu_b:.3f}")
    print(f"  ρ (Dixon-Coles) = {rho}")
    
    matrix_dc = build_score_matrix(lambda_a, mu_b, rho, max_goals=6)
    p1_dc, px_dc, p2_dc = extract_1x2(matrix_dc)
    
    print(f"\n  Matriz de marcador (hasta 5-5, cola agrupada en 6+):")
    print(f"  {'':>8}", end="")
    for j in range(7):
        label = f"{team_b_name[:3]}={j}" if j < 6 else "6+"
        print(f"  {label:>8}", end="")
    print()
    for i in range(7):
        label = f"{team_a_name[:3]}={i}" if i < 6 else "6+"
        print(f"  {label:>8}", end="")
        for j in range(7):
            print(f"  {matrix_dc[i][j]*100:>7.2f}%", end="")
        print()
    
    print(f"\n  1X2 (Dixon-Coles puro):")
    print(f"    {team_a_name}: {p1_dc*100:.1f}% | Empate: {px_dc*100:.1f}% | {team_b_name}: {p2_dc*100:.1f}%")
    
    # ---- PASO 2: ACTUALIZACIÓN BAYESIANA ----
    print(f"\n{'─'*80}")
    print(f"  PASO 2: ACTUALIZACIÓN BAYESIANA (GAMMA-POISSON)")
    print(f"{'─'*80}")
    
    # Últimos 7 partidos como evidencia
    recent_n = 7
    recent_a = matches_a[-recent_n:]
    recent_b = matches_b[-recent_n:]
    
    recent_gf_a = [m["gf"] for m in recent_a]
    recent_gc_a = [m["gc"] for m in recent_a]
    recent_gf_b = [m["gf"] for m in recent_b]
    recent_gc_b = [m["gc"] for m in recent_b]
    
    # Prior basado en todos los partidos
    prior_n = n_a
    lambda_a_post, lambda_a_std = bayesian_update(att_a, prior_n, recent_gf_a, recent_n)
    def_a_post, def_a_std = bayesian_update(def_a, prior_n, recent_gc_a, recent_n)
    
    prior_n_b = n_b
    att_b_post, att_b_std = bayesian_update(att_b, prior_n_b, recent_gf_b, recent_n)
    def_b_post, def_b_std = bayesian_update(def_b, prior_n_b, recent_gc_b, recent_n)
    
    lambda_a_bayes = lambda_a_post * (def_b_post / avg_goals)
    mu_b_bayes = att_b_post * (def_a_post / avg_goals)
    
    if is_neutral:
        lambda_a_bayes *= 0.97
        mu_b_bayes *= 0.97
    
    print(f"  Evidencia: últimos {recent_n} partidos de cada selección")
    print(f"  {team_a_name}:")
    print(f"    Ataque: Prior={att_a:.3f} → Posterior={lambda_a_post:.3f} (±{lambda_a_std:.3f})")
    print(f"    Defensa: Prior={def_a:.3f} → Posterior={def_a_post:.3f} (±{def_a_std:.3f})")
    print(f"  {team_b_name}:")
    print(f"    Ataque: Prior={att_b:.3f} → Posterior={att_b_post:.3f} (±{att_b_std:.3f})")
    print(f"    Defensa: Prior={def_b:.3f} → Posterior={def_b_post:.3f} (±{def_b_std:.3f})")
    print(f"  λ_Bayes({team_a_name})={lambda_a_bayes:.3f} | μ_Bayes({team_b_name})={mu_b_bayes:.3f}")
    
    matrix_bayes = build_score_matrix(lambda_a_bayes, mu_b_bayes, rho, max_goals=6)
    p1_bayes, px_bayes, p2_bayes = extract_1x2(matrix_bayes)
    
    print(f"  1X2 (Bayes):")
    print(f"    {team_a_name}: {p1_bayes*100:.1f}% | Empate: {px_bayes*100:.1f}% | {team_b_name}: {p2_bayes*100:.1f}%")
    
    # ---- PASO 3: TEORÍA DE GRAFOS ----
    print(f"\n{'─'*80}")
    print(f"  PASO 3: TEORÍA DE GRAFOS (ESCENARIO B — PageRank por calendario)")
    print(f"{'─'*80}")
    
    calendar_a = build_result_graph_pagerank(matches_a, None)
    calendar_b = build_result_graph_pagerank(matches_b, None)
    
    print(f"  Escenario: B (solo datos agregados, sin red de pases)")
    print(f"  Fuerza del calendario:")
    print(f"    {team_a_name}: {calendar_a:.3f}")
    print(f"    {team_b_name}: {calendar_b:.3f}")
    
    # Ajustar λ/μ con la fuerza del calendario
    # Si calendario más fuerte → los goles valen más → ajuste positivo
    ratio_calendar = calendar_a / calendar_b if calendar_b > 0 else 1.0
    graph_adj_a = min(max(ratio_calendar ** 0.3, 0.85), 1.15)  # Suavizado
    graph_adj_b = min(max((1/ratio_calendar) ** 0.3, 0.85), 1.15)
    
    lambda_a_adj = lambda_a_bayes * graph_adj_a
    mu_b_adj = mu_b_bayes * graph_adj_b
    
    print(f"  Ratio calendario: {ratio_calendar:.3f}")
    print(f"  Corrector multiplicativo {team_a_name}: {graph_adj_a:.3f}")
    print(f"  Corrector multiplicativo {team_b_name}: {graph_adj_b:.3f}")
    print(f"  λ_ajustado({team_a_name})={lambda_a_adj:.3f} | μ_ajustado({team_b_name})={mu_b_adj:.3f}")
    
    matrix_adj = build_score_matrix(lambda_a_adj, mu_b_adj, rho, max_goals=6)
    p1_adj, px_adj, p2_adj = extract_1x2(matrix_adj)
    
    print(f"  1X2 (Poisson-Bayes-Grafos):")
    print(f"    {team_a_name}: {p1_adj*100:.1f}% | Empate: {px_adj*100:.1f}% | {team_b_name}: {p2_adj*100:.1f}%")
    
    # ---- PASO 4: CADENAS DE MARKOV ----
    print(f"\n{'─'*80}")
    print(f"  PASO 4: CADENAS DE MARKOV (TIEMPO DISCRETO)")
    print(f"{'─'*80}")
    
    print(f"  Estados: {{Victoria, Empate, Derrota}}")
    print(f"  NOTA: Sin minuto de gol → Plan B (transición entre resultados consecutivos)")
    
    w_a_markov, d_a_markov, l_a_markov = markov_1x2(matches_a)
    w_b_markov, d_b_markov, l_b_markov = markov_1x2(matches_b)
    
    # Combinar perspectivas: victoria de A = derrota de B y viceversa
    p1_markov = (w_a_markov + l_b_markov) / 2
    px_markov = (d_a_markov + d_b_markov) / 2
    p2_markov = (l_a_markov + w_b_markov) / 2
    
    # Normalizar
    total_m = p1_markov + px_markov + p2_markov
    p1_markov /= total_m
    px_markov /= total_m
    p2_markov /= total_m
    
    print(f"  Distribución estacionaria {team_a_name}: W={w_a_markov:.3f} D={d_a_markov:.3f} L={l_a_markov:.3f}")
    print(f"  Distribución estacionaria {team_b_name}: W={w_b_markov:.3f} D={d_b_markov:.3f} L={l_b_markov:.3f}")
    print(f"  1X2 (Markov combinado):")
    print(f"    {team_a_name}: {p1_markov*100:.1f}% | Empate: {px_markov*100:.1f}% | {team_b_name}: {p2_markov*100:.1f}%")
    
    # Divergencia Poisson-Bayes vs Markov
    div_1 = abs(p1_adj - p1_markov) * 100
    div_x = abs(px_adj - px_markov) * 100
    div_2 = abs(p2_adj - p2_markov) * 100
    max_div = max(div_1, div_x, div_2)
    
    print(f"\n  Divergencia Poisson-Bayes vs Markov:")
    print(f"    {team_a_name}: {div_1:.1f}pp | Empate: {div_x:.1f}pp | {team_b_name}: {div_2:.1f}pp")
    if max_div > 5:
        print(f"  ⚠️  SEÑAL DE INCERTIDUMBRE: Divergencia máxima = {max_div:.1f}pp (> 5pp)")
    else:
        print(f"  ✅ Divergencia máxima = {max_div:.1f}pp (≤ 5pp) — modelos coherentes")
    
    # ---- PASO 5: INTEGRACIÓN + BOOTSTRAP ----
    print(f"\n{'─'*80}")
    print(f"  PASO 5: INTEGRACIÓN Y BOOTSTRAP (n=2000)")
    print(f"{'─'*80}")
    
    results_boot, lambdas_boot, mus_boot, scores_boot = bootstrap_analysis(
        matches_a, matches_b, rho, n_bootstrap=2000,
        graph_adj_home=graph_adj_a, graph_adj_away=graph_adj_b
    )
    
    p1s = [r[0] for r in results_boot]
    pxs = [r[1] for r in results_boot]
    p2s = [r[2] for r in results_boot]
    
    # Modelo integrado: 60% Poisson-Bayes-Grafos + 25% Bootstrap media + 15% Markov
    boot_p1 = np.mean(p1s)
    boot_px = np.mean(pxs)
    boot_p2 = np.mean(p2s)
    
    final_p1 = 0.60 * p1_adj + 0.25 * boot_p1 + 0.15 * p1_markov
    final_px = 0.60 * px_adj + 0.25 * boot_px + 0.15 * px_markov
    final_p2 = 0.60 * p2_adj + 0.25 * boot_p2 + 0.15 * p2_markov
    
    # Normalizar
    total_f = final_p1 + final_px + final_p2
    final_p1 /= total_f
    final_px /= total_f
    final_p2 /= total_f
    
    # Intervalos de confianza al 90% del bootstrap
    ci_p1 = (np.percentile(p1s, 5) * 100, np.percentile(p1s, 95) * 100)
    ci_px = (np.percentile(pxs, 5) * 100, np.percentile(pxs, 95) * 100)
    ci_p2 = (np.percentile(p2s, 5) * 100, np.percentile(p2s, 95) * 100)
    
    ci_lambda = (np.percentile(lambdas_boot, 5), np.percentile(lambdas_boot, 95))
    ci_mu = (np.percentile(mus_boot, 5), np.percentile(mus_boot, 95))
    
    lambda_mean = np.mean(lambdas_boot)
    lambda_std = np.std(lambdas_boot)
    mu_mean = np.mean(mus_boot)
    mu_std = np.std(mus_boot)
    
    # Marcador más probable (de la matriz integrada final)
    matrix_final = build_score_matrix(lambda_a_adj, mu_b_adj, rho, max_goals=6)
    best_idx = np.unravel_index(np.argmax(matrix_final), matrix_final.shape)
    best_prob = matrix_final[best_idx] * 100
    
    # Bootstrap del marcador más probable
    score_probs = {}
    for s in scores_boot:
        key = (s[0], s[1])
        score_probs[key] = score_probs.get(key, 0) + 1
    most_common_score = max(score_probs, key=score_probs.get)
    score_freq = score_probs[most_common_score] / len(scores_boot) * 100
    
    # CI del marcador más probable
    best_score_probs = [s[2] * 100 for s in scores_boot if (s[0], s[1]) == best_idx]
    if len(best_score_probs) > 10:
        ci_score = (np.percentile(best_score_probs, 5), np.percentile(best_score_probs, 95))
    else:
        ci_score = (best_prob * 0.7, best_prob * 1.3)
    
    # Masa de probabilidad del marcador más probable vs total
    mass_pct = matrix_final[best_idx] / matrix_final.sum() * 100
    
    print(f"  Pesos de integración: Poisson-Bayes-Grafos=60% | Bootstrap=25% | Markov=15%")
    print(f"  Bootstrap: n=2000 iteraciones con remuestreo")
    
    # =============== OUTPUT FINAL ===============
    print(f"\n{'='*80}")
    print(f"  RESULTADOS FINALES: {team_a_name} vs {team_b_name}")
    print(f"{'='*80}")
    
    print(f"\n  A) Tabla 1X2:")
    print(f"  ┌─────────────────────────────┬──────────────┬───────────────────────────┐")
    print(f"  │ Resultado                   │ Probabilidad │ IC 90%                    │")
    print(f"  ├─────────────────────────────┼──────────────┼───────────────────────────┤")
    print(f"  │ Local ({team_a_name:>12})      │    {final_p1*100:>5.1f}%    │ [{ci_p1[0]:>5.1f}% – {ci_p1[1]:>5.1f}%]       │")
    print(f"  │ Empate                      │    {final_px*100:>5.1f}%    │ [{ci_px[0]:>5.1f}% – {ci_px[1]:>5.1f}%]       │")
    print(f"  │ Visitante ({team_b_name:>12}) │    {final_p2*100:>5.1f}%    │ [{ci_p2[0]:>5.1f}% – {ci_p2[1]:>5.1f}%]       │")
    print(f"  └─────────────────────────────┴──────────────┴───────────────────────────┘")
    
    print(f"\n  B) Marcador exacto de mayor probabilidad:")
    print(f"     {team_a_name} {best_idx[0]} – {best_idx[1]} {team_b_name}")
    print(f"     Probabilidad: {best_prob:.1f}% (IC 90%: [{ci_score[0]:.1f}% – {ci_score[1]:.1f}%])")
    print(f"     Goles esperados: {team_a_name} λ={lambda_a_adj:.2f} ± {lambda_std:.2f} | {team_b_name} μ={mu_b_adj:.2f} ± {mu_std:.2f}")
    print(f"     Este marcador concentra el {mass_pct:.1f}% de la masa de probabilidad total.")
    
    print(f"\n  Limitaciones declaradas:")
    print(f"  - Columnas faltantes: xG_favor, xG_contra, posesión_%, remates_favor,")
    print(f"    remates_contra, minuto_de_gol, tarjetas_rojas")
    print(f"  - Markov opera con Plan B (transición entre resultados, no por bloques de 15 min)")
    print(f"  - Grafos opera con Escenario B (sin red de pases)")
    print(f"  - N partidos < 100 para ambas selecciones (declarado)")
    
    return {
        "p1": final_p1, "px": final_px, "p2": final_p2,
        "ci_p1": ci_p1, "ci_px": ci_px, "ci_p2": ci_p2,
        "lambda": lambda_a_adj, "mu": mu_b_adj,
        "best_score": best_idx, "best_prob": best_prob
    }

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  MOTOR MATEMÁTICO DE PREDICCIÓN FUTBOLÍSTICA")
    print("  FIFA World Cup 2026 — Ronda de 32 — 1 de julio de 2026")
    print("  Modelos: Dixon-Coles | Bayesiano | Grafos (PageRank) | Markov")
    print("=" * 80)
    
    # Partido 1: Bélgica vs Senegal
    result1 = analyze_match("Bélgica", "Senegal", belgium_matches, senegal_matches, is_neutral=True)
    
    print("\n" * 2)
    
    # Partido 2: EEUU vs Bosnia y Herzegovina
    result2 = analyze_match("EEUU", "Bosnia", usa_matches, bosnia_matches, is_neutral=True)
    
    print("\n" + "=" * 80)
    print("  FIN DEL ANÁLISIS")
    print("=" * 80)
