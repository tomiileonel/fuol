#!/usr/bin/env python3
"""
MOTOR MATEMÁTICO DE PREDICCIÓN FUTBOLÍSTICA - ARQUITECTURA RIGUROSA
===================================================================
1. Geometría Espacial: Voronoi Acotado (Shapely)
2. Geometría Dinámica: EDOs de Momentum
3. Interferencia Táctica: Eigenvalores de Matriz Hermitiana
4. Colapso Probabilístico: Poisson Bivariado de Dixon-Coles
"""

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# DATOS HISTÓRICOS VERIFICADOS
# ==============================================================================
belgium = [
    {"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0},
    {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1},
    {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2},
    {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3},
    {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1},
]

senegal = [
    {"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1},
    {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0},
    {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0},
    {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0},
]

usa = [
    {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5},
    {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1},
    {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0},
    {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3},
]

bosnia = [
    {"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5},
    {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7},
    {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4},
    {"gf":3,"gc":1},
]

FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
    "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]])
}

def derive_empirical_stats(data):
    """Deriva stats de Poisson (λ empírico) y perfil táctico."""
    gf = [m["gf"] for m in data]
    gc = [m["gc"] for m in data]
    mean_gf = np.mean(gf)
    mean_gc = np.mean(gc)
    
    pos_proxy = 40.0 + (mean_gf / (mean_gf + mean_gc + 1e-9)) * 20.0
    line_proxy = np.clip(35.0 + (mean_gf * 4.0) - (mean_gc * 2.0), 30.0, 50.0)
    def_k_proxy = np.clip(1.0 - (mean_gc / 3.0) * 0.15, 0.70, 0.95)
    
    if pos_proxy > 52.0: form = "4-3-3"
    elif pos_proxy < 48.0: form = "3-4-2-1"
    else: form = "4-2-3-1"
        
    return {"lam_base": mean_gf, "def_k": def_k_proxy, "form": form, "line": line_proxy}

# ==============================================================================
# 1. TOPOLOGÍA RIGUROSA: VORONOI ACOTADO
# ==============================================================================
def calculate_bounded_voronoi_control(fa, fb):
    """Calcula el control espacial (área en m^2) cruzando diagramas de Voronoi
    con los límites estrictos de un campo de fútbol estándar (105x68m)."""
    field_polygon = Polygon([(0, 0), (105, 0), (105, 68), (0, 68)])
    
    points = np.vstack([fa, fb])
    vor = Voronoi(points)
    
    area_a = 0.0
    area_b = 0.0
    
    for i, region_index in enumerate(vor.point_region):
        if region_index == -1:
            continue
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue
            
        polygon = Polygon([vor.vertices[v] for v in region])
        bounded_polygon = polygon.intersection(field_polygon)
        
        if i < len(fa):
            area_a += bounded_polygon.area
        else:
            area_b += bounded_polygon.area
            
    total_area = area_a + area_b
    if total_area == 0:
        return 0.5, 0.5
        
    return area_a / total_area, area_b / total_area

# ==============================================================================
# 2. EDOS DE MOMENTUM (Alimentadas por Geometría Real)
# ==============================================================================
def fatigue_ode(y, t, c_a, c_b):
    Fa, Fb, Ma, Mb = y
    dFa = 0.01 * (1 + c_b - c_a)
    dFb = 0.01 * (1 + c_a - c_b)
    dMa = 0.05 * c_a - 0.02 * Fa - 0.1 * Mb
    dMb = 0.05 * c_b - 0.02 * Fb - 0.1 * Ma
    return [dFa, dFb, dMa, dMb]

# ==============================================================================
# 3. INTERFERENCIA TÁCTICA (ÁLGEBRA LINEAL PURA)
# ==============================================================================
def apply_tactical_interference(lam_base, mu_base, m_a, m_b):
    """Usa el análisis espectral de una matriz Hermitiana para modelar 
    cómo el Momentum interfiere con las tasas de gol base."""
    damp_a = 1.0 + np.log1p(np.abs(m_a)/5.0) * np.sign(m_a) * 0.15
    damp_b = 1.0 + np.log1p(np.abs(m_b)/5.0) * np.sign(m_b) * 0.15
    
    F_A = max(0.01, lam_base * damp_a)
    F_B = max(0.01, mu_base * damp_b)
    
    V = (F_A * F_B)**0.5 * 0.1
    
    H = np.array([[F_A, -V], 
                  [-V, F_B]], dtype=np.complex128)
    
    eigenvalues, _ = np.linalg.eigh(H)
    
    lam_adj = max(0.01, np.real(eigenvalues[1])) 
    mu_adj = max(0.01, np.real(eigenvalues[0]))
    
    return lam_adj, mu_adj

# ==============================================================================
# 4. COLAPSO ESTADÍSTICO: DIXON-COLES
# ==============================================================================
def dixon_coles_tau(x, y, lam, mu, rho):
    if x==0 and y==0: return 1 - lam*mu*rho
    elif x==1 and y==0: return 1 + mu*rho
    elif x==0 and y==1: return 1 + lam*rho
    elif x==1 and y==1: return 1 - rho
    else: return 1.0

def build_score_matrix(lambda_a, mu_b, rho=-0.04, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = dixon_coles_tau(i, j, lambda_a, mu_b, rho)
            matrix[i][j] = tau * stats.poisson.pmf(i, lambda_a) * stats.poisson.pmf(j, mu_b)
    
    matrix /= matrix.sum() 
    return matrix

def extract_1x2(matrix):
    max_g = matrix.shape[0]
    p1 = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i > j)
    px = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i == j)
    p2 = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i < j)
    return p1, px, p2

# ==============================================================================
# ORQUESTADOR PRINCIPAL
# ==============================================================================
def predict_match(team_a_name, team_b_name, prior_lam, prior_mu, form_a, form_b):
    ctrl_a, ctrl_b = calculate_bounded_voronoi_control(form_a, form_b)
    
    t = np.linspace(0, 90, 90)
    sol = integrate.odeint(fatigue_ode, [0, 0, 1.0, 1.0], t, args=(ctrl_a, ctrl_b))
    m_a, m_b = np.mean(sol[:, 2]), np.mean(sol[:, 3])
    
    lam_final, mu_final = apply_tactical_interference(prior_lam, prior_mu, m_a, m_b)
    
    matrix = build_score_matrix(lam_final, mu_final, rho=-0.04)
    p1, px, p2 = extract_1x2(matrix)
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS RIGUROSO: {team_a_name} vs {team_b_name}")
    print(f"{'='*60}")
    print(f"[Capa 1] Geometría Voronoi (m²): Control {team_a_name}={ctrl_a*100:.1f}% | {team_b_name}={ctrl_b*100:.1f}%")
    print(f"[Capa 2] EDO Momentum: m_a={m_a:.3f} | m_b={m_b:.3f}")
    print(f"[Capa 3] Interferencia Hermitiana: lam_final={lam_final:.3f} | mu_final={mu_final:.3f}")
    print(f"[Capa 4] Colapso Dixon-Coles 1X2:")
    print(f"  {team_a_name:<15}: {p1*100:>5.2f}%")
    print(f"  EMPATE         : {px*100:>5.2f}%")
    print(f"  {team_b_name:<15}: {p2*100:>5.2f}%")
    print(f"{'-'*60}\n")
    
    return p1, px, p2

def run():
    # Calcular perfiles previos
    prof_bel = derive_empirical_stats(belgium)
    prof_sen = derive_empirical_stats(senegal)
    prof_usa = derive_empirical_stats(usa)
    prof_bos = derive_empirical_stats(bosnia)
    
    # λ base (Poisson original) ajustado por la defensa rival
    lam_bel_base = prof_bel["lam_base"] * (1 - prof_sen["def_k"] * 0.1)
    lam_sen_base = prof_sen["lam_base"] * (1 - prof_bel["def_k"] * 0.1)
    
    lam_usa_base = prof_usa["lam_base"] * (1 - prof_bos["def_k"] * 0.1)
    lam_bos_base = prof_bos["lam_base"] * (1 - prof_usa["def_k"] * 0.1)
    
    # Preparar formaciones geométricas
    fa_bel = FORMACIONES[prof_bel["form"]].copy().astype(float)
    fa_bel[:, 0] += (prof_bel["line"] - 40.0)
    fb_sen = FORMACIONES[prof_sen["form"]].copy().astype(float)
    fb_sen[:, 0] = 105 - fb_sen[:, 0] - (prof_sen["line"] - 40.0)
    
    fa_usa = FORMACIONES[prof_usa["form"]].copy().astype(float)
    fa_usa[:, 0] += (prof_usa["line"] - 40.0)
    fb_bos = FORMACIONES[prof_bos["form"]].copy().astype(float)
    fb_bos[:, 0] = 105 - fb_bos[:, 0] - (prof_bos["line"] - 40.0)
    
    # Ejecutar predicciones con el motor ortogonal
    predict_match("BÉLGICA", "SENEGAL", lam_bel_base, lam_sen_base, fa_bel, fb_sen)
    predict_match("EEUU", "BOSNIA-HERZ.", lam_usa_base, lam_bos_base, fa_usa, fb_bos)

if __name__ == "__main__":
    run()
