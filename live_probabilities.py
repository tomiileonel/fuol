import numpy as np
from scipy.stats import poisson

def get_probabilities(lam, mu, home_goals=0, away_goals=0, t_elapsed=0):
    t_remaining = (90 - t_elapsed) / 90.0
    lam_r = lam * t_remaining
    mu_r = mu * t_remaining
    
    p1 = 0
    px = 0
    p2 = 0
    
    for i in range(12):
        for j in range(12):
            prob = poisson.pmf(i, lam_r) * poisson.pmf(j, mu_r)
            final_home = home_goals + i
            final_away = away_goals + j
            
            if final_home > final_away:
                p1 += prob
            elif final_home == final_away:
                px += prob
            else:
                p2 += prob
                
    return p1, px, p2

def print_match_evolution(home_team, away_team, lam, mu):
    print(f"\n=======================================================")
    print(f" {home_team.upper()} vs {away_team.upper()} (Escenario: 0-0)")
    print(f"=======================================================")
    print(f"Minuto | P(1) {home_team[:3]} | P(X) Empate | P(2) {away_team[:3]} | Cuota Justa X")
    print("-" * 55)
    
    for t in [0, 15, 30, 45, 60, 75, 85, 90]:
        if t == 90:
            p1, px, p2 = 0, 1, 0
        else:
            p1, px, p2 = get_probabilities(lam, mu, 0, 0, t)
        
        cuota_x = 1/px if px > 0 else 0
        print(f"{t:02d}'    | {p1*100:5.1f}%     | {px*100:5.1f}%     | {p2*100:5.1f}%     | @ {cuota_x:.2f}")

# Partido 1: Inglaterra vs Noruega
print_match_evolution("Inglaterra", "Noruega", 1.595, 1.194)

# Partido 2: Argentina vs Suiza
print_match_evolution("Argentina", "Suiza", 1.698, 0.970)
