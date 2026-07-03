import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def render_matchup_summary():
    BG_COLOR = '#08080C'
    SURFACE_COLOR = '#12121A'
    ACCENT_MAIN = '#059669' # Verde (Australia)
    ACCENT_SEC = '#EAB308'  # Amarillo (Empate)
    ACCENT_TER = '#EF4444'  # Rojo (Egipto)
    TEXT_MAIN = '#F8FAFC'
    TEXT_MUTED = '#94A3B8'
    GRID_COLOR = '#1E293B'

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor=BG_COLOR)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], wspace=0.3, hspace=0.4)

    # Datos
    probs_90m = [37.88, 29.17, 32.96]
    labels_90m = ['Victoria AUSTRALIA', 'EMPATE', 'Victoria EGIPTO']
    colors_90m = [ACCENT_MAIN, ACCENT_SEC, ACCENT_TER]

    probs_adv = [52.47, 47.55]
    labels_adv = ['AUSTRALIA', 'EGIPTO']
    colors_adv = [ACCENT_MAIN, ACCENT_TER]

    # Panel 1: Probabilidades 90m (Donut Chart)
    ax1 = fig.add_subplot(gs[0, 0], facecolor=SURFACE_COLOR)
    wedges, texts, autotexts = ax1.pie(probs_90m, labels=labels_90m, colors=colors_90m, autopct='%1.2f%%',
                                       startangle=90, textprops=dict(color=TEXT_MAIN, weight='bold'),
                                       wedgeprops=dict(width=0.4, edgecolor=BG_COLOR))
    plt.setp(autotexts, size=10, weight="bold")
    ax1.set_title("PROBABILIDADES EN 90 MINUTOS", color=TEXT_MAIN, weight='bold', pad=15)

    # Panel 2: Probabilidad de Avanzar (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE_COLOR)
    y_pos = np.arange(len(labels_adv))
    bars = ax2.barh(y_pos, probs_adv, color=colors_adv, edgecolor=BG_COLOR, height=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_adv, color=TEXT_MAIN, weight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_title("PROBABILIDAD DE CLASIFICAR (Incl. Prórroga)", color=TEXT_MAIN, weight='bold', pad=15)
    ax2.grid(axis='x', color=GRID_COLOR, linestyle='-', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color(GRID_COLOR)
    ax2.spines['left'].set_visible(False)
    for bar, p in zip(bars, probs_adv):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                 f'{p:.2f}%', va='center', color=TEXT_MAIN, weight='bold')

    # Panel 3: Info General
    ax3 = fig.add_subplot(gs[1, :], facecolor=BG_COLOR)
    ax3.axis('off')
    
    info_text = (
        "ANÁLISIS TÁCTICO - MOTOR DIXON-COLES + GAMMA-POISSON\n\n"
        "Encuentro: AUSTRALIA vs EGIPTO\n"
        "Fase: 16avos de Final\n"
        "Sede: Dallas, Arlington (Neutral)\n"
        "Edge del Sistema: +4.92% (Favorito: Australia)\n\n"
        "Contexto: Partido altamente friccionado. Alta probabilidad de empate (29.17%)\n"
        "sugiere posible tiempo extra."
    )
    ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes, ha='center', va='center',
             color=TEXT_MUTED, fontsize=12, weight='bold', 
             bbox=dict(facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, alpha=0.8, boxstyle='round,pad=1.5'))

    plt.tight_layout()
    fig.patch.set_facecolor(BG_COLOR)
    plt.savefig('C:/Users/Usuario/Desktop/fuol/dashboard_aus_egy.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    print("Gráfico generado en C:/Users/Usuario/Desktop/fuol/dashboard_aus_egy.png")

if __name__ == "__main__":
    render_matchup_summary()
