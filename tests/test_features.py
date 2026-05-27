"""
Unit tests for features module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import (
    compute_statistical_features,
    compute_trend_features,
    compute_anomaly_features,
    compute_temporal_features,
    compute_other_features,
    build_features,
    normalize_features
)


def test_statistical_features():
    """Test statistical feature computation."""
    consumption = np.array([10, 20, 15, 25, 30, 0, 35, 40])
    
    features = compute_statistical_features(consumption)
    
    assert 'mean' in features
    assert 'median' in features
    assert 'std' in features
    assert 'coef_var' in features
    assert 'min' in features
    assert 'max' in features
    assert 'range' in features
    assert 'skewness' in features
    
    # Check reasonable values
    assert features['mean'] > 0
    assert features['min'] == 0
    assert features['max'] == 40
    assert features['range'] == 40


def test_trend_features():
    """Test trend feature computation."""
    # Linear increasing trend
    consumption = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    features = compute_trend_features(consumption)
    
    assert 'slope_full' in features
    assert 'slope_last_30d' in features
    assert 'slope_last_90d' in features
    
    # Should detect positive slope
    assert features['slope_full'] > 0


def test_anomaly_features():
    """Test anomaly feature computation."""
    # Series with zero days and sudden drop
    consumption = np.array([50, 50, 0, 0, 50, 20, 10, 50, 50])  # sudden drop from 50 to 20
    
    features = compute_anomaly_features(consumption, sudden_drop_threshold=0.5)
    
    assert 'zero_day_count' in features
    assert 'sudden_drop_count' in features
    assert 'volatility_index' in features
    
    # Should detect 2 zero days
    assert features['zero_day_count'] == 2
    
    # Should detect at least one sudden drop
    assert features['sudden_drop_count'] >= 1


def test_temporal_features():
    """Test temporal feature computation."""
    consumption = np.random.rand(100) * 50  # 100 days of data
    
    features = compute_temporal_features(consumption)
    
    assert 'weekday_vs_weekend_ratio' in features
    assert 'peak_day_ratio' in features
    
    # Ratio should be positive
    assert features['weekday_vs_weekend_ratio'] > 0
    
    # Peak ratio should be between 0 and 1
    assert 0 <= features['peak_day_ratio'] <= 1


def test_other_features():
    """Test other features (autocorr, missing sequences)."""
    consumption = np.array([10, 20, 30, np.nan, np.nan, np.nan, np.nan, 40, 50])
    
    features = compute_other_features(consumption)
    
    assert 'autocorr_lag1' in features
    assert 'missing_sequences_count' in features
    
    # Should detect one missing sequence (4 consecutive NaN)
    assert features['missing_sequences_count'] >= 1


def test_normalize_features(tmp_path):
    """Test feature normalization."""
    # Create synthetic features
    X_train = pd.DataFrame({
        'feature1': [0, 50, 100],
        'feature2': [10, 20, 30],
        'feature3': [0.1, 0.5, 1.0]
    })
    
    X_test = pd.DataFrame({
        'feature1': [25, 75],
        'feature2': [15, 25],
        'feature3': [0.3, 0.7]
    })
    
    scaler_path = tmp_path / "test_scaler.joblib"
    
    X_train_scaled, X_test_scaled = normalize_features(
        X_train, X_test, str(scaler_path)
    )
    
    # Check scaler was saved
    assert scaler_path.exists()
    
    # Check scaled data is between 0 and 1
    assert (X_train_scaled >= 0).all().all()
    assert (X_train_scaled <= 1).all().all()
    assert (X_test_scaled >= 0).all().all()
    assert (X_test_scaled <= 1).all().all()
    
    # Check min/max in training data
    assert X_train_scaled['feature1'].min() == 0
    assert X_train_scaled['feature1'].max() == 1


def test_build_features_mock():
    """Test build_features with mock data."""
    # Create mock long-format data
    df_long = pd.DataFrame({
        'customer_id': ['C1'] * 100 + ['C2'] * 100,
        'day_index': list(range(100)) * 2,
        'consumption_kwh': np.random.rand(200) * 50
    })
    
    labels = pd.Series([0, 1], index=['C1', 'C2'], name='label')
    
    X, y = build_features(df_long, labels)
    
    # Check structure
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 2  # 2 customers
    assert len(y) == 2
    
    # Check index alignment
    assert (X.index == y.index).all()
    
    # Check feature count (should have multiple features)
    assert X.shape[1] >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
