import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from data_pipeline import DataPipeline
from walk_forward_pipeline import WalkForwardPipeline
import config

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting 5-year Backtest (2020-2026)")
    
    pipeline = DataPipeline()
    wfp = WalkForwardPipeline(
        train_window_days=1460, 
        test_window_days=30, 
        embargo_days=14,
        lambda_scale=config.LAMBDA_SCALE,
        prior_strength=config.PRIOR_STRENGTH,
        half_life=config.DEFAULT_HALF_LIFE
    )
    
    t0 = time.time()
    results = wfp.run(pipeline)
    elapsed = time.time() - t0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Run complete in {elapsed:.1f}s")
    
    all_metrics = results.get('all_metrics', [])
    if not all_metrics:
        print("ERROR: no matches in the backtest period. Check pipeline.")
        return
        
    start_date_str = '2020-01-01'
    start_dt = pd.to_datetime(start_date_str)
    
    filtered_rps = []
    filtered_hit = []
    filtered_brier = []
    
    for fold in all_metrics:
        brier_array = fold.get('brier_array', [])
        if not brier_array:
            continue
        for date_str, rps_val, hit_val, brier_val in zip(fold['test_dates'], fold['rps_array'], fold['hit_array'], brier_array):
            dt = pd.to_datetime(date_str.split('|')[0])
            if dt >= start_dt:
                filtered_rps.append(rps_val)
                filtered_hit.append(hit_val)
                filtered_brier.append(brier_val)
                
    n_matches = len(filtered_rps)
    avg_rps = float(np.mean(filtered_rps)) if n_matches else 0.0
    hit_rate = float(np.mean(filtered_hit)) if n_matches else 0.0
    avg_brier = float(np.mean(filtered_brier)) if n_matches else 0.0
    
    # Get max date from dataframe for reporting
    df, _ = pipeline.prepare_data()
    max_date = str(df['date'].max().date())
    
    output = {
      "period": f"{start_date_str} to {max_date}",
      "n_matches": n_matches,
      "hyperparameters": {
        "lambda_scale": config.LAMBDA_SCALE,
        "prior_strength": config.PRIOR_STRENGTH,
        "half_life": config.DEFAULT_HALF_LIFE
      },
      "metrics": {
        "rps": round(avg_rps, 4),
        "brier": round(avg_brier, 4),
        "hit_rate_pct": round(hit_rate * 100, 1)
      },
      "timestamp": datetime.now().isoformat()
    }
    
    print("\nRESULTS:")
    print(json.dumps(output, indent=2))
    
    out_file = "backtest_results_5y.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")

if __name__ == '__main__':
    main()
