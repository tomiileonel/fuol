import argparse
import json
import pandas as pd
from data_pipeline import DataPipeline
from unified_engine import UnifiedEngine
from walk_forward_pipeline import WalkForwardPipeline
import config

def main():
    parser = argparse.ArgumentParser(description="FUOL Prediction Engine")
    parser.add_argument("--home", type=str, required=True, help="Home team name")
    parser.add_argument("--away", type=str, required=True, help="Away team name")
    parser.add_argument("--neutral", action="store_true", help="Is neutral venue (World Cup)?")
    args = parser.parse_args()
    
    print("Iniciando DataPipeline para extraer dataset completo...")
    pipeline = DataPipeline()
    df, elo_registry = pipeline.prepare_data()
    df['date'] = pd.to_datetime(df['date'])
    
    home_team = args.home
    away_team = args.away
    
    # Filter dataset up to today
    today = pd.to_datetime("today")
    df = df[df['date'] <= today]
    
    print(f"Dataset cargado con {len(df)} partidos históricos. Última fecha: {df['date'].max().date()}")
    print("Construyendo historia (history_cache) para el motor...")
    
    # Build history using WalkForwardPipeline helper
    wfp = WalkForwardPipeline(
        train_window_days=1460, # 4 years of history as calibrated
        test_window_days=30,
        embargo_days=0
    )
    # The history cache requires the dataframe of the last 4 years
    train_start = today - wfp.train_window
    train_df = df[(df['date'] >= train_start) & (df['date'] < today)]
    
    history_cache = wfp._build_history_cache(train_df)
    
    print(f"Historial construido con {len(history_cache)} equipos activos en los últimos 4 años.")
    
    print("Inicializando UnifiedEngine con hiperparámetros calibrados (Optuna)...")
    
    h_hist = history_cache.get(home_team, pd.DataFrame())
    a_hist = history_cache.get(away_team, pd.DataFrame())
    
    engine = UnifiedEngine(
        home_team, away_team, h_hist, a_hist,
        lambda_scale=config.LAMBDA_SCALE,
        prior_strength=config.PRIOR_STRENGTH,
        half_life=config.DEFAULT_HALF_LIFE,
        venue='N' if args.neutral else 'H'
    )
    
    print(f"\nGenerando predicción para {home_team} vs {away_team}...")
    probs = engine.predict()
    
    if probs is None:
        print(f"ERROR: No se pudo generar la predicción. ¿Nombres correctos? {home_team}, {away_team}")
        return
        
    print("\n--- RESULTADOS DEL MOTOR FUOL ---")
    print(f"Probabilidad Local ({home_team}): {probs.get('p1', 0)*100:.2f}%")
    print(f"Probabilidad Empate: {probs.get('px', 0)*100:.2f}%")
    print(f"Probabilidad Visitante ({away_team}): {probs.get('p2', 0)*100:.2f}%")
    
    # Top 5 scores
    print("\n--- TOP 5 MARCADORES EXACTOS ---")
    try:
        for i, ((h_score, a_score), prob) in enumerate(probs.get('top_5_scores', [])):
            print(f"{i+1}. {h_score}-{a_score}: {prob*100:.2f}%")
    except Exception as e:
        print("Error parseando top scores", e)
    
    # Parametros base
    print("\n--- PARAMETROS BASE ---")
    print(f"Elo {home_team}: {probs.get('elo_a', 'N/A')}")
    print(f"Elo {away_team}: {probs.get('elo_b', 'N/A')}")
    print(f"Lambda ({home_team} Goles Esperados): {probs.get('lam', 'N/A')}")
    print(f"Mu ({away_team} Goles Esperados): {probs.get('mu', 'N/A')}")
    
    # Calculamos cuotas justas (Fair Odds)
    fair_odds = {'1': 1/probs['p1'] if probs['p1']>0 else 0, 
                 'X': 1/probs['px'] if probs['px']>0 else 0, 
                 '2': 1/probs['p2'] if probs['p2']>0 else 0}
    
    print("\n--- CUOTAS JUSTAS (FAIR ODDS sin margen) ---")
    print(f"Cuota 1 (Local): {fair_odds['1']:.2f}")
    print(f"Cuota X (Empate): {fair_odds['X']:.2f}")
    print(f"Cuota 2 (Visita): {fair_odds['2']:.2f}")
    
if __name__ == '__main__':
    main()
