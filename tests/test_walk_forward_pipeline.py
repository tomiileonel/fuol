import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from walk_forward_pipeline import WalkForwardPipeline
from data_pipeline import DataPipeline
from unittest.mock import MagicMock

def test_walk_forward_pipeline_metrics_shape():
    # Mock data to avoid long execution
    mock_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'home_team': ['A', 'C'] * 50,
        'away_team': ['B', 'D'] * 50,
        'home_score': [1, 0] * 50,
        'away_score': [0, 1] * 50,
        'neutral': ['False'] * 100,
        'tournament': ['Friendly'] * 100
    })
    
    # Mock DataPipeline to return mock_df
    mock_pipeline = MagicMock(spec=DataPipeline)
    mock_pipeline.prepare_data.return_value = (mock_df, None)
    
    # Create Pipeline with very small windows
    pipeline = WalkForwardPipeline(
        train_window_days=10,
        test_window_days=5,
        embargo_days=2,
        min_train_size=10,
        half_life=365.0,
        prior_strength=2.0,
        lambda_scale=0.23
    )
    
    # Run
    metrics = pipeline.run(mock_pipeline)
    
    # Validations
    assert metrics is not None
    assert 'avg_rps' in metrics
    assert 'avg_brier' in metrics
    assert 'avg_log_loss' in metrics
    assert 'total_test_samples' in metrics
    
    # Ranges
    assert 0.0 <= metrics['avg_rps'] <= 1.0
    assert 0.0 <= metrics['avg_brier'] <= 2.0
    assert metrics['avg_log_loss'] >= 0.0
    assert metrics['total_test_samples'] > 0
    assert metrics['n_folds'] > 0
