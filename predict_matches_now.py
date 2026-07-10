import sys
import pandas as pd
import numpy as np

sys.path.append(r"c:\Users\Usuario\Desktop\fuol")
from unified_engine_v3 import UnifiedEngineV3

def load_team_matches(df, team_name):
    # Filter matches where team is home or away
    home_matches = df[df['home_team'] == team_name].copy()
    away_matches = df[df['away_team'] == team_name].copy()
    
    matches = []
    
    for _, row in home_matches.dropna(subset=['home_score', 'away_score']).iterrows():
        gf = int(row['home_score'])
        gc = int(row['away_score'])
        res = 'W' if gf > gc else ('L' if gc > gf else 'D')
        matches.append({
            'date': row['date'],
            'gf': gf,
            'gc': gc,
            'res': res,
            'opponent': row['away_team'],
            'minute': 90
        })
        
    for _, row in away_matches.dropna(subset=['home_score', 'away_score']).iterrows():
        gf = int(row['away_score'])
        gc = int(row['home_score'])
        res = 'W' if gf > gc else ('L' if gc > gf else 'D')
        matches.append({
            'date': row['date'],
            'gf': gf,
            'gc': gc,
            'res': res,
            'opponent': row['home_team'],
            'minute': 90
        })
        
    # Sort by date and take the last 200 matches
    matches = sorted(matches, key=lambda x: x['date'])[-200:]
    return matches

import traceback
def main():
    try:
        df = pd.read_csv("results.csv")
        
        pairs = [("Mexico", "England"), ("Brazil", "Norway")]
        
        for team_a, team_b in pairs:
            matches_a = load_team_matches(df, team_a)
            matches_b = load_team_matches(df, team_b)
            
            engine = UnifiedEngineV3(
                team_a=team_a,
                team_b=team_b,
                matches_a=matches_a,
                matches_b=matches_b,
                venue='N'
            )
            
            pred = engine.predict()
            
            print(f"\n{'='*40}")
            print(f"PREDICCION: {team_a.upper()} vs {team_b.upper()}")
            print(f"{'='*40}")
            print(f"Probabilidades (1X2): 1={pred.p_home:.2%} | X={pred.p_draw:.2%} | 2={pred.p_away:.2%}")
            print(f"Lambdas V3: Local={pred.lambda_home:.3f} | Visita={pred.mu_away:.3f}")
            print(f"Lambdas Hawkes: Lam={pred.lambda_hawkes:.3f} | Mu={pred.mu_hawkes:.3f}")
            print(f"Lambdas Base (DC): Local={pred.p_home_dc:.2%} | Empate={pred.p_draw_dc:.2%} | Visita={pred.p_away_dc:.2%}")
            print(f"Coherencia Cuantica: {pred.quantum_coherence:.2%}")
            print(f"Interpretacion: {pred.interpretation}")
            print(f"Confianza: {pred.confidence_score:.2f}")
            print(f"Recomendacion: {pred.recommendation}")
            print("Top 5 Resultados Exactos:")
            for score in pred.top_5_scores:
                print(f"  {score}")
                
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
