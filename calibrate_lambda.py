import csv
import sys
import numpy as np
import statsmodels.api as sm
from unified_engine import EloRating

def main():
    csv_path = "results.csv"
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matches = list(reader)
    except FileNotFoundError:
        print(f"❌ No se encontró el dataset {csv_path}")
        sys.exit(1)
        
    sorted_matches = sorted(matches, key=lambda m: m.get('date', '1970-01-01'))
    
    print(f"Procesando {len(sorted_matches)} partidos para calibrar LAMBDA_SCALE...")
    
    elos = {}
    
    def get_elo(team):
        if team not in elos:
            elos[team] = EloRating(team)
        return elos[team]
        
    diffs = []
    gf_observed = []
    gc_observed = []
    
    for m in sorted_matches:
        home = m.get('home_team', m.get('Home Team'))
        away = m.get('away_team', m.get('Away Team'))
        if not home or not away:
            continue
            
        gf_str = m.get('home_score', m.get('Home Goals'))
        gc_str = m.get('away_score', m.get('Away Goals'))
        
        try:
            gf = int(gf_str)
            gc = int(gc_str)
        except (ValueError, TypeError):
            continue
            
        elo_home = get_elo(home)
        elo_away = get_elo(away)
        
        venue_bonus = EloRating.HOME_ADV if m.get('neutral') != 'True' else 0.0
        
        diff = (elo_home.rating + venue_bonus) - elo_away.rating
        
        diffs.append(diff)
        gf_observed.append(gf)
        gc_observed.append(gc)
        
        # update elos
        elo_home.update({'res': 'W' if gf > gc else ('D' if gf == gc else 'L'), 'competition': m.get('tournament', 'N'), 'venue': 'H' if m.get('neutral') != 'True' else 'N'}, elo_away.rating)
        
        elo_away.update({'res': 'L' if gf > gc else ('D' if gf == gc else 'W'), 'competition': m.get('tournament', 'N'), 'venue': 'A' if m.get('neutral') != 'True' else 'N'}, elo_home.rating - venue_bonus)

    diffs = np.array(diffs)
    gf_observed = np.array(gf_observed)
    gc_observed = np.array(gc_observed)
    
    print("Ajustando modelos Poisson GLM...")
    
    # Model GF
    X = sm.add_constant(diffs / 400.0)
    model_gf = sm.GLM(gf_observed, X, family=sm.families.Poisson()).fit()
    lambda_scale_gf = model_gf.params[1]
    
    # Model GC (diffs are negative for away team, so we use -diffs/400.0)
    X_gc = sm.add_constant(-diffs / 400.0)
    model_gc = sm.GLM(gc_observed, X_gc, family=sm.families.Poisson()).fit()
    lambda_scale_gc = model_gc.params[1]
    
    print(f"LAMBDA_SCALE ajustado por goles locales (gf): {lambda_scale_gf:.4f}")
    print(f"LAMBDA_SCALE ajustado por goles visitantes (gc): {lambda_scale_gc:.4f}")
    
    avg_scale = (lambda_scale_gf + lambda_scale_gc) / 2.0
    
    diff_pct = abs(lambda_scale_gf - lambda_scale_gc) / avg_scale * 100
    if diff_pct > 20:
        print(f"⚠️  Divergencia alta ({diff_pct:.1f}%) entre modelo local y visitante.")
        print("   Considerá revisar la ventaja de localía (HOME_ADV) o el modelo subyacente.")
    else:
        print(f"\nSugerencia: Actualizá LAMBDA_SCALE en unified_engine.py a {avg_scale:.4f}")

if __name__ == "__main__":
    main()
