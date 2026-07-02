import numpy as np
import matplotlib
matplotlib.use('Agg') # Uso de backend no interactivo
from tournament_simulator import MonteCarloTournament
from tournament_dashboard import render_tournament_center
import matplotlib.pyplot as plt

# Formaciones
FORMACIONES = {
    "4-3-3": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [45,20], [40,34], [45,48], [75,14], [80,34], [75,54]]),
    "4-2-3-1": np.array([[5,34], [25,14], [20,28], [20,40], [25,54], [40,25], [40,43], [60,14], [65,34], [60,54], [85,34]])
}

# Equipos de Octavos de Final (Ejemplo representativo en orden de llaves)
teams = [
    "Brasil", "Uruguay", 
    "Francia", "Senegal", 
    "Inglaterra", "EEUU", 
    "España", "Marruecos", 
    "Argentina", "Suiza", 
    "Paises Bajos", "EEUU",  # oops EEUU repetido, lo cambio por Japon
    "Portugal", "Corea del Sur", 
    "Alemania", "Bosnia y Herzegovina"
]
teams[11] = "Japon"

# Mock de historiales de xG (Goles Esperados a favor y en contra)
team_histories = {
    "Brasil": [{"gf": 2.2, "gc": 0.5, "res": "W"}, {"gf": 1.8, "gc": 0.4, "res": "W"}],
    "Uruguay": [{"gf": 1.5, "gc": 0.8, "res": "W"}, {"gf": 1.1, "gc": 1.0, "res": "W"}],
    "Francia": [{"gf": 2.5, "gc": 0.8, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}],
    "Senegal": [{"gf": 1.2, "gc": 1.2, "res": "D"}, {"gf": 1.0, "gc": 1.5, "res": "L"}],
    "Inglaterra": [{"gf": 2.0, "gc": 0.6, "res": "W"}, {"gf": 1.5, "gc": 0.5, "res": "W"}],
    "EEUU": [{"gf": 1.4, "gc": 1.1, "res": "W"}, {"gf": 1.0, "gc": 1.0, "res": "D"}],
    "España": [{"gf": 2.1, "gc": 0.7, "res": "W"}, {"gf": 1.8, "gc": 0.6, "res": "W"}],
    "Marruecos": [{"gf": 1.0, "gc": 0.5, "res": "W"}, {"gf": 0.8, "gc": 0.4, "res": "W"}],
    "Argentina": [{"gf": 2.4, "gc": 0.4, "res": "W"}, {"gf": 2.0, "gc": 0.5, "res": "W"}],
    "Suiza": [{"gf": 1.3, "gc": 1.0, "res": "W"}, {"gf": 1.1, "gc": 1.1, "res": "D"}],
    "Paises Bajos": [{"gf": 1.9, "gc": 0.8, "res": "W"}, {"gf": 1.6, "gc": 0.7, "res": "W"}],
    "Japon": [{"gf": 1.2, "gc": 0.9, "res": "W"}, {"gf": 1.5, "gc": 1.2, "res": "W"}],
    "Portugal": [{"gf": 2.2, "gc": 0.7, "res": "W"}, {"gf": 1.9, "gc": 0.8, "res": "W"}],
    "Corea del Sur": [{"gf": 1.1, "gc": 1.3, "res": "L"}, {"gf": 1.4, "gc": 1.0, "res": "W"}],
    "Alemania": [{"gf": 2.3, "gc": 0.9, "res": "W"}, {"gf": 1.8, "gc": 0.6, "res": "W"}],
    "Bosnia y Herzegovina": [{"gf": 0.8, "gc": 1.5, "res": "L"}, {"gf": 0.5, "gc": 1.2, "res": "L"}],
}

# Implied Probabilities del Mercado (Mock Pinnacle/Bet365 en %)
MARKET_ODDS = {
    "Francia": 15.0, "Inglaterra": 14.0, "Brasil": 12.0, "Argentina": 11.0,
    "Portugal": 9.0, "España": 8.0, "Alemania": 8.0, "Paises Bajos": 5.0,
    "Uruguay": 4.0, "EEUU": 3.0, "Suiza": 3.0, "Marruecos": 2.0,
    "Senegal": 2.0, "Japon": 2.0, "Corea del Sur": 1.0, "Bosnia y Herzegovina": 1.0
}

def run_simulation():
    # Correr iteraciones (N=100 para demo)
    n_sim = 100
    
    base_modifiers = {
        "EEUU": {"home_advantage": 1.15} # Bono de anfitrión
    }
    
    INJURY_DEBUFF_FACTOR = 0.8
    black_swan_config = {
        "team": "Francia",
        "probability": 0.15, # 15% de probabilidad estocástica por ronda
        "injury_multiplier": INJURY_DEBUFF_FACTOR # Reduce fuerza (Lambda) en -20%
    }
    
    simulator = MonteCarloTournament(
        teams, team_histories, FORMACIONES, 
        n_simulations=n_sim, 
        base_modifiers=base_modifiers, 
        black_swan_config=black_swan_config
    )
    dist = simulator.run()
    
    print("\n" + "="*50)
    print(" ANÁLISIS DE ARBITRAJE (MARKET GAP) Y CISNE NEGRO")
    print("="*50)
    print(f"{'Equipo':<20} | {'MC Engine':<10} | {'Mercado':<10} | {'Alpha (Edge)'}")
    print("-"*50)
    for t, p in dist.items():
        market_p = MARKET_ODDS.get(t, 0.0)
        edge = p - market_p
        if p > 0:
            edge_str = f"+{edge:.2f}%" if edge > 0 else f"{edge:.2f}%"
            print(f"{t:<20} | {p:>6.2f}%    | {market_p:>6.2f}%   | {edge_str}")
            
    # Guardar Dashboard
    render_tournament_center(dist, n_sim)
    plt.savefig('C:/Users/Usuario/.gemini/antigravity/brain/a1f0fdb0-392c-4a26-9266-ebb694530b47/monte_carlo_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#08080C')
    print("\n[+] Dashboard Cualitativo guardado exitosamente.")

if __name__ == "__main__":
    run_simulation()
