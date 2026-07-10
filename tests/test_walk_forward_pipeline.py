import pytest
import pandas as pd
from datetime import timedelta
from walk_forward_pipeline import WalkForwardPipeline

def test_strict_purge_and_embargo():
    """
    Test asertivo riguroso de No-Fuga Temporal (No-Leakage).
    Verifica que el gap entre el último partido de train y el primer 
    partido de test es estrictamente >= embargo_days.
    """
    # Create synthetic data
    dates = pd.date_range(start="2020-01-01", end="2025-01-01", freq="D")
    df = pd.DataFrame({
        'date': dates,
        'home_team': 'A',
        'away_team': 'B',
        'home_score': 1,
        'away_score': 1
    })
    
    # Configure pipeline with 14 days embargo
    embargo_days = 14
    tester = WalkForwardPipeline(train_window_days=365, test_window_days=30, embargo_days=embargo_days, min_train_size=10)
    
    folds = tester.generate_folds(df)
    
    assert len(folds) > 0, "No folds generated"
    
    for train_df, test_df in folds:
        max_train_date = train_df['date'].max()
        min_test_date = test_df['date'].min()
        
        gap = (min_test_date - max_train_date).days
        
        # Test 1: Gap is strictly >= embargo_days
        assert gap >= embargo_days, f"Leakage detected! Gap is {gap} days, expected at least {embargo_days}"
        
        # Test 2: No overlapping dates
        assert max_train_date < min_test_date, "Train and Test dates overlap!"
