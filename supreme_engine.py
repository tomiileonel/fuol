import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
from performance_tracker import ModelTelemetry

warnings.filterwarnings('ignore')

class SupremePredictionEngine:
    def __init__(self, team_a, team_b, matches_a, matches_b, forms, modifiers_a=None, modifiers_b=None):
        self.team_a = team_a
        self.team_b = team_b
        self.matches_a = matches_a
        self.matches_b = matches_b
        self.forms = forms # Diccionario de formaciones
        self.modifiers_a = modifiers_a or {}
        self.modifiers_b = modifiers_b or {}
        self.field_bounds = Polygon([(0, 0), (105, 0), (105, 68), (0, 68)])
        self.avg_goals_wc = 1.35
        self.rho = -0.04

    def _calculate_pagerank_prior(self, matches):
        G = nx.DiGraph()
        for m in matches:
            opp = m.get("opponent", "Opponent")
            diff = m["gf"] - m["gc"]
            if not G.has_node("Base"): G.add_node("Base")
            if not G.has_node(opp): G.add_node(opp)
            weight = max(0.1, 1.0 + diff * 0.2)
            if diff > 0:
                G.add_edge(opp, "Base", weight=weight)
            else:
                G.add_edge("Base", opp, weight=max(0.1, 1.0 - diff * 0.2))
        try:
            pagerank = nx.pagerank(G, alpha=0.85, weight='weight')
            return max(0.5, pagerank.get("Base", 1.0) * len(G))
        except:
            return 1.0

    def _bayesian_update(self, prior_rate, prior_n, recent_goals, recent_n):
        alpha_prior = prior_rate * prior_n
        beta_prior = prior_n
        alpha_post = alpha_prior + sum(recent_goals)
        beta_post = beta_prior + recent_n
        return alpha_post / beta_post

    def _derive_bayesian_rates(self):
        pr_a = self._calculate_pagerank_prior(self.matches_a)
        pr_b = self._calculate_pagerank_prior(self.matches_b)
        rec_a, rec_b = self.matches_a[-7:], self.matches_b[-7:]
        
        att_a_prior = np.mean([m["gf"] for m in self.matches_a]) * (pr_a ** 0.1)
        def_a_prior = np.mean([m["gc"] for m in self.matches_a]) / (pr_a ** 0.1)
        att_b_prior = np.mean([m["gf"] for m in self.matches_b]) * (pr_b ** 0.1)
        def_b_prior = np.mean([m["gc"] for m in self.matches_b]) / (pr_b ** 0.1)
        
        n_a, n_b = len(self.matches_a), len(self.matches_b)
        att_a_post = self._bayesian_update(att_a_prior, n_a, [m["gf"] for m in rec_a], 7)
        def_a_post = self._bayesian_update(def_a_prior, n_a, [m["gc"] for m in rec_a], 7)
        att_b_post = self._bayesian_update(att_b_prior, n_b, [m["gf"] for m in rec_b], 7)
        def_b_post = self._bayesian_update(def_b_prior, n_b, [m["gc"] for m in rec_b], 7)
        
        # Aplicación del Tensor Cualitativo de Momentum (Home Advantage, Travel Fatigue, Injuries)
        momentum_a = self.modifiers_a.get('home_advantage', 1.0) * self.modifiers_a.get('injury_modifier', 1.0) / self.modifiers_a.get('travel_fatigue', 1.0)
        momentum_b = self.modifiers_b.get('home_advantage', 1.0) * self.modifiers_b.get('injury_modifier', 1.0) / self.modifiers_b.get('travel_fatigue', 1.0)
        
        att_a_post *= momentum_a
        def_a_post /= momentum_a
        att_b_post *= momentum_b
        def_b_post /= momentum_b
        
        lam_base = att_a_post * (def_b_post / self.avg_goals_wc)
        mu_base = att_b_post * (def_a_post / self.avg_goals_wc)
        return lam_base, mu_base

    def _bounded_voronoi(self, fa, fb):
        points = np.vstack([fa, fb])
        vor = Voronoi(points)
        area_a, area_b = 0.0, 0.0
        for i, region_index in enumerate(vor.point_region):
            if region_index == -1: continue
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0: continue
            poly = Polygon([vor.vertices[v] for v in region])
            bounded = poly.intersection(self.field_bounds)
            if i < len(fa): area_a += bounded.area
            else: area_b += bounded.area
        total = area_a + area_b
        return (area_a/total, area_b/total) if total > 0 else (0.5, 0.5)

    def _momentum_ode(self, y, t, c_a, c_b):
        Fa, Fb, Ma, Mb = y
        dFa = 0.01 * (1 + c_b - c_a)
        dFb = 0.01 * (1 + c_a - c_b)
        dMa = 0.05 * c_a - 0.02 * Fa - 0.1 * Mb
        dMb = 0.05 * c_b - 0.02 * Fb - 0.1 * Ma
        return [dFa, dFb, dMa, dMb]

    def _apply_hermitian_interference(self, lam_base, mu_base, c_a, c_b):
        t = np.linspace(0, 90, 90)
        sol = integrate.odeint(self._momentum_ode, [0, 0, 1.0, 1.0], t, args=(c_a, c_b))
        m_a_array, m_b_array = sol[:, 2], sol[:, 3]
        m_a, m_b = np.mean(m_a_array), np.mean(m_b_array)
        
        damp_a = 1.0 + np.log1p(np.abs(m_a)/5.0) * np.sign(m_a) * 0.15
        damp_b = 1.0 + np.log1p(np.abs(m_b)/5.0) * np.sign(m_b) * 0.15
        F_A = max(0.01, lam_base * damp_a)
        F_B = max(0.01, mu_base * damp_b)
        V = (F_A * F_B)**0.5 * 0.1
        
        H = np.array([[F_A, -V], [-V, F_B]], dtype=np.complex128)
        eigenvalues, _ = np.linalg.eigh(H)
        return max(0.01, np.real(eigenvalues[1])), max(0.01, np.real(eigenvalues[0])), m_a_array, m_b_array

    def _dixon_coles_matrix(self, lam, mu, max_goals=None):
        if max_goals is None:
            max_goals = max(6, int(np.ceil(max(lam, mu) + 4 * np.sqrt(max(lam, mu)))))
            
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = 1.0
                if i==0 and j==0: tau = 1 - lam*mu*self.rho
                elif i==1 and j==0: tau = 1 + mu*self.rho
                elif i==0 and j==1: tau = 1 + lam*self.rho
                elif i==1 and j==1: tau = 1 - self.rho
                matrix[i][j] = tau * stats.poisson.pmf(i, lam) * stats.poisson.pmf(j, mu)
        return matrix / matrix.sum()

    def run_pipeline(self, form_str_a="4-3-3", form_str_b="4-2-3-1"):
        lam_base, mu_base = self._derive_bayesian_rates()
        fa = self.forms[form_str_a].copy().astype(float)
        fb = self.forms[form_str_b].copy().astype(float)
        fb[:, 0] = 105 - fb[:, 0]
        ctrl_a, ctrl_b = self._bounded_voronoi(fa, fb)
        lam_final, mu_final, m_a_array, m_b_array = self._apply_hermitian_interference(lam_base, mu_base, ctrl_a, ctrl_b)
        matrix = self._dixon_coles_matrix(lam_final, mu_final)
        
        max_g = matrix.shape[0]
        p1 = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i > j)
        px = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i == j)
        p2 = sum(matrix[i][j] for i in range(max_g) for j in range(max_g) if i < j)
        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        return {
            "Control m2": (ctrl_a, ctrl_b),
            "Tasas de Poisson (Bayes+Grafos)": (lam_base, mu_base),
            "Tasas Finales (Interferencia)": (lam_final, mu_final),
            "1X2": (p1, px, p2),
            "Marcador Exacto": best_idx,
            "Probabilidad Marcador": matrix[best_idx],
            "Matriz Cruda": matrix
        }, (m_a_array, m_b_array)

def render_tactical_dashboard(team_a_name, team_b_name, engine_output, momentum_history):
    BG_COLOR = '#08080C'
    SURFACE_COLOR = '#12121A'
    ACCENT_A = '#059669'
    ACCENT_B = '#6366F1'
    TEXT_MAIN = '#F8FAFC'
    TEXT_MUTED = '#94A3B8'
    GRID_COLOR = '#1E293B'
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title(f"Dashboard: {team_a_name} vs {team_b_name}")
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # 2. PANEL 1: EVOLUCIÓN DEL MOMENTUM (EDOs)
    ax1 = fig.add_subplot(gs[0, :2], facecolor=SURFACE_COLOR)
    t = np.linspace(0, 90, 90)
    m_a_history, m_b_history = momentum_history
    
    ax1.plot(t, m_a_history, color=ACCENT_A, linewidth=2, label=f'Momentum {team_a_name}')
    ax1.plot(t, m_b_history, color=ACCENT_B, linewidth=2, label=f'Momentum {team_b_name}')
    ax1.fill_between(t, m_a_history, alpha=0.1, color=ACCENT_A)
    ax1.fill_between(t, m_b_history, alpha=0.1, color=ACCENT_B)
    
    ax1.set_title("Evolución de Presión y Momentum Táctico (90 Min)", color=TEXT_MAIN, fontsize=12, pad=15)
    ax1.set_xlabel("Minutos de Juego", color=TEXT_MUTED)
    ax1.set_ylabel("Fuerza Relativa (Ma, Mb)", color=TEXT_MUTED)
    ax1.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(frameon=False, labelcolor=TEXT_MAIN)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color(GRID_COLOR)
    ax1.spines['left'].set_color(GRID_COLOR)

    # 3. PANEL 2: CONTROL ESPACIAL (Voronoi Acotado)
    ax2 = fig.add_subplot(gs[0, 2], facecolor=BG_COLOR)
    ctrl_a, ctrl_b = engine_output["Control m2"]
    sizes = [ctrl_a, ctrl_b]
    labels = [f'{team_a_name}\n({ctrl_a*100:.1f}%)', f'{team_b_name}\n({ctrl_b*100:.1f}%)']
    colors = [ACCENT_A, ACCENT_B]
    
    ax2.pie(sizes, labels=labels, colors=colors, startangle=90, 
            wedgeprops=dict(width=0.3, edgecolor=BG_COLOR, linewidth=2),
            textprops=dict(color=TEXT_MAIN, fontsize=10, weight='bold'))
    ax2.set_title("Dominio Topológico ($m^2$)", color=TEXT_MAIN, fontsize=12, pad=15)

    # 4. PANEL 3: MAPA DE CALOR DIXON-COLES
    ax3 = fig.add_subplot(gs[1, :2], facecolor=SURFACE_COLOR)
    matrix = engine_output["Matriz Cruda"][:6, :6] 
    
    sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="mako", 
                cbar_kws={'label': 'Probabilidad (%)'}, ax=ax3,
                linewidths=0.5, linecolor=BG_COLOR, 
                annot_kws={"size": 9, "weight": "bold"})
    ax3.set_title("Matriz de Colapso Probabilístico (Dixon-Coles)", color=TEXT_MAIN, fontsize=12, pad=15)
    ax3.set_xlabel(f"Goles {team_b_name} (mu)", color=TEXT_MUTED, fontsize=10)
    ax3.set_ylabel(f"Goles {team_a_name} (lam)", color=TEXT_MUTED, fontsize=10)
    ax3.tick_params(colors=TEXT_MUTED)

    # 5. PANEL 4: PROBABILIDADES 1X2
    ax4 = fig.add_subplot(gs[1, 2], facecolor=SURFACE_COLOR)
    p1, px, p2 = engine_output["1X2"]
    categories = [team_a_name, 'EMPATE', team_b_name]
    probs = [p1 * 100, px * 100, p2 * 100]
    bar_colors = [ACCENT_A, '#475569', ACCENT_B]
    
    y_pos = np.arange(len(categories))
    bars = ax4.barh(y_pos, probs, color=bar_colors, height=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(categories, color=TEXT_MAIN, weight='bold')
    ax4.invert_yaxis() 
    ax4.set_xlabel('Probabilidad (%)', color=TEXT_MUTED)
    ax4.set_title("Vector de Salida (1X2)", color=TEXT_MAIN, fontsize=12, pad=15)
    ax4.grid(axis='x', color=GRID_COLOR, linestyle='--', alpha=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_color(GRID_COLOR)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(axis='y', length=0)
    
    for bar in bars:
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', color=TEXT_MAIN, weight='bold')

    plt.tight_layout()
    fig.patch.set_facecolor(BG_COLOR)

# ==============================================================================
# EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    belgium = [{"gf":0,"gc":0}, {"gf":2,"gc":2}, {"gf":2,"gc":0}, {"gf":3,"gc":0}, {"gf":0,"gc":1}, {"gf":2,"gc":0}, {"gf":0,"gc":0}, {"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":0,"gc":2}, {"gf":2,"gc":2}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":0,"gc":1}, {"gf":1,"gc":1}, {"gf":4,"gc":3}, {"gf":1,"gc":1}, {"gf":0,"gc":0}, {"gf":5,"gc":1}]
    senegal = [{"gf":3,"gc":0}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":1}, {"gf":3,"gc":0}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":0,"gc":0}, {"gf":2,"gc":0}, {"gf":1,"gc":3}, {"gf":2,"gc":3}, {"gf":5,"gc":0}]
    usa = [{"gf":0,"gc":1}, {"gf":3,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":5}, {"gf":1,"gc":1}, {"gf":2,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":1}, {"gf":1,"gc":2}, {"gf":1,"gc":1}, {"gf":3,"gc":1}, {"gf":3,"gc":0}, {"gf":4,"gc":1}, {"gf":2,"gc":0}, {"gf":2,"gc":3}]
    bosnia = [{"gf":1,"gc":2}, {"gf":0,"gc":3}, {"gf":0,"gc":1}, {"gf":2,"gc":5}, {"gf":0,"gc":0}, {"gf":1,"gc":2}, {"gf":0,"gc":2}, {"gf":0,"gc":7}, {"gf":1,"gc":1}, {"gf":1,"gc":0}, {"gf":1,"gc":1}, {"gf":1,"gc":4}, {"gf":3,"gc":1}]
    
    FORMACIONES = {
        "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
        "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]]),
        "3-4-2-1": np.array([[5,34], [20,20], [15,34], [20,48], [45,10], [40,26], [40,42], [45,58], [65,24], [65,44], [80,34]])
    }
    
    # ---------------------------------------------------------
    # TELEMETRÍA Y SINCRONIZACIÓN DE CONOCIMIENTO (ONLINE LEARNING)
    # ---------------------------------------------------------
    telemetry = ModelTelemetry()
    
    # Sincronizamos las listas base con cualquier resultado empírico ya guardado en la DB
    belgium_dynamic = telemetry.synchronize_knowledge_base("BÉLGICA", belgium)
    senegal_dynamic = telemetry.synchronize_knowledge_base("SENEGAL", senegal)
    usa_dynamic = telemetry.synchronize_knowledge_base("EEUU", usa)
    bosnia_dynamic = telemetry.synchronize_knowledge_base("BOSNIA-HERZ.", bosnia)
    
    # 1. BÉLGICA vs SENEGAL
    engine1 = SupremePredictionEngine("BÉLGICA", "SENEGAL", belgium_dynamic, senegal_dynamic, FORMACIONES)
    res1, momentum1 = engine1.run_pipeline("4-3-3", "4-3-3")
    print("Generando Dashboard: BÉLGICA vs SENEGAL...")
    telemetry.log_prediction("BÉLGICA", "SENEGAL", res1)
    render_tactical_dashboard("BÉLGICA", "SENEGAL", res1, momentum1)
    
    # 2. EEUU vs BOSNIA-HERZ.
    engine2 = SupremePredictionEngine("EEUU", "BOSNIA-HERZ.", usa_dynamic, bosnia_dynamic, FORMACIONES)
    res2, momentum2 = engine2.run_pipeline("4-3-3", "3-4-2-1")
    print("Generando Dashboard: EEUU vs BOSNIA-HERZ...")
    telemetry.log_prediction("EEUU", "BOSNIA-HERZ.", res2)
    render_tactical_dashboard("EEUU", "BOSNIA-HERZ.", res2, momentum2)
    
    # Simulamos inyectar resultados reales para la telemetría (Cierra el bucle de feedback empírico)
    telemetry.log_actual_result("BÉLGICA", "SENEGAL", goals_a=1, goals_b=0)
    telemetry.log_actual_result("EEUU", "BOSNIA-HERZ.", goals_a=4, goals_b=0)
    
    telemetry.calculate_metrics()
    
    # Mostrar todas las figuras (detiene la ejecución hasta cerrar las ventanas)
    plt.show()
