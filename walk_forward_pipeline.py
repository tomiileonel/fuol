import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import timedelta
from probability_calibrator import ProbabilityCalibrator
from unified_engine import UnifiedEngine
from data_pipeline import DataPipeline

class WalkForwardPipeline:
    """
    Robust Walk-Forward backtester with Purge & Embargo to prevent data leakage,
    especially protecting against EWMA feature autocorrelation.
    """
    def __init__(self, 
                 train_window_days: int = 365 * 4,  # 4 years training
                 test_window_days: int = 30,        # 1 month testing
                 embargo_days: int = 14,            # 14 days gap to clear EWMA memory
                 min_train_size: int = 100,
                 half_life: float = 365.0,
                 prior_strength: float = 5.0,
                 lambda_scale: float = 500.0):
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.embargo = timedelta(days=embargo_days)
        self.min_train_size = min_train_size
        self.half_life = half_life
        self.prior_strength = prior_strength
        self.lambda_scale = lambda_scale

    def generate_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates (train_df, test_df) tuples adhering to Purge & Embargo.
        """
        if not np.issubdtype(df['date'].dtype, np.datetime64):
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date').reset_index(drop=True)
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        folds = []
        current_test_start = min_date + self.train_window + self.embargo
        
        while current_test_start < max_date:
            test_end = current_test_start + self.test_window
            
            # Train window strictly ends BEFORE the embargo period
            train_end = current_test_start - self.embargo
            train_start = train_end - self.train_window
            
            train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
            test_mask = (df['date'] >= current_test_start) & (df['date'] < test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(train_df) >= self.min_train_size and len(test_df) > 0:
                folds.append((train_df, test_df))
                
            current_test_start = test_end
            
        return folds

    def _compute_metrics(self, y_true: np.ndarray, probs: np.ndarray) -> dict:
        """
        y_true: 0 (Home), 1 (Draw), 2 (Away)
        probs: shape (n, 3)
        """
        n = len(y_true)
        if n == 0:
            return {}
            
        # Log-Loss
        # Clip to prevent log(0)
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        log_loss = -np.mean([np.log(probs_clipped[i, y_true[i]]) for i in range(n)])
        
        # Brier Score (Multi-class)
        targets = np.zeros_like(probs)
        targets[np.arange(n), y_true] = 1.0
        brier = np.mean(np.sum((probs - targets) ** 2, axis=1))
        
        # Ranked Probability Score (RPS)
        cum_probs = np.cumsum(probs, axis=1)
        cum_targets = np.cumsum(targets, axis=1)
        rps = np.mean(np.sum((cum_probs - cum_targets) ** 2, axis=1) / 2.0)
        
        return {
            'log_loss': log_loss,
            'brier': brier,
            'rps': rps,
            'n_samples': n
        }

    def run(self, pipeline: DataPipeline) -> dict:
        """
        Executes the walk-forward validation across the entire dataset.
        Trains a ProbabilityCalibrator inside each fold (on the training data predictions)
        and applies it to the test data.
        """
        # Load and compute features incrementally (without leakage up to any point in time)
        df, _ = pipeline.prepare_data()
        folds = self.generate_folds(df)
        
        all_metrics = []
        
        for fold_idx, (train_df, test_df) in enumerate(folds):
            # 1. Collect raw predictions for training set to fit the calibrator
            train_preds = []
            y_train = []
            
            unique_teams = pd.concat([train_df['home_team'], train_df['away_team']]).unique()
            history_cache = {team: pipeline.get_team_history(train_df, team) for team in unique_teams}
            
            # Note: For performance, predicting the entire train_df might be slow if UnifiedEngine is slow.
            # In a production environment, we'd cache base model predictions. 
            # For brevity, we sample the last 500 matches for calibration.
            calib_sample = train_df.tail(500) 
            
            for _, row in calib_sample.iterrows():
                h, a = row['home_team'], row['away_team']
                h_hist = history_cache.get(h, [])
                a_hist = history_cache.get(a, [])
                
                # Base model (no calibrator yet)
                engine = UnifiedEngine(h, a, h_hist, a_hist, optimize_rho=False, 
                                       half_life=self.half_life, 
                                       prior_strength=self.prior_strength, 
                                       lambda_scale=self.lambda_scale)
                try:
                    pred = engine.predict()
                    train_preds.append([pred.get('p1', 0.33), pred.get('px', 0.33), pred.get('p2', 0.33)])
                    
                    if row['home_score'] > row['away_score']:
                        y_train.append(0)
                    elif row['home_score'] == row['away_score']:
                        y_train.append(1)
                    else:
                        y_train.append(2)
                except Exception:
                    pass

            if not train_preds:
                continue
                
            calibrator = ProbabilityCalibrator(method='isotonic')
            calibrator.fit(np.array(train_preds), np.array(y_train))
            
            # Use calibrated training predictions as the reference distribution for drift monitoring
            drift_reference = calibrator.predict_proba(np.array(train_preds))
            
            # 2. Predict on Test Set USING the calibrator
            test_preds = []
            y_test = []
            
            for _, row in test_df.iterrows():
                h, a = row['home_team'], row['away_team']
                # We can use train_df + past test_df rows for history to be completely precise,
                # but using train_df history is strict and prevents leakage.
                h_hist = history_cache.get(h, [])
                a_hist = history_cache.get(a, [])
                
                engine = UnifiedEngine(h, a, h_hist, a_hist, calibrator=calibrator, optimize_rho=False,
                                       half_life=self.half_life, 
                                       prior_strength=self.prior_strength, 
                                       lambda_scale=self.lambda_scale)
                try:
                    pred = engine.predict()
                    test_preds.append([pred.get('p1', 0.33), pred.get('px', 0.33), pred.get('p2', 0.33)])
                    
                    if row['home_score'] > row['away_score']:
                        y_test.append(0)
                    elif row['home_score'] == row['away_score']:
                        y_test.append(1)
                    else:
                        y_test.append(2)
                        
                    # Circuit Breaker: Check for drift every 50 predictions
                    if len(test_preds) % 50 == 0:
                        ProbabilityCalibrator.check_drift(drift_reference, np.array(test_preds[-50:]))
                        
                except Exception as e:
                    if isinstance(e, RuntimeError) and "Structural drift detected" in str(e):
                        # Propagate circuit breaker exceptions
                        raise e
                    pass
                    
            if test_preds:
                metrics = self._compute_metrics(np.array(y_test), np.array(test_preds))
                all_metrics.append(metrics)
                
        if not all_metrics:
            return {}
            
        # Aggregate metrics
        agg = {
            'avg_rps': np.mean([m['rps'] for m in all_metrics]),
            'avg_brier': np.mean([m['brier'] for m in all_metrics]),
            'avg_log_loss': np.mean([m['log_loss'] for m in all_metrics]),
            'total_test_samples': sum(m['n_samples'] for m in all_metrics),
            'n_folds': len(all_metrics)
        }
        return agg

if __name__ == "__main__":
    pipeline = DataPipeline()
    tester = WalkForwardPipeline(train_window_days=365*4, test_window_days=30, embargo_days=14)
    print("Iniciando validación Walk-Forward (Purge & Embargo)...")
    results = tester.run(pipeline)
    print(f"Resultados Finales: {results}")
