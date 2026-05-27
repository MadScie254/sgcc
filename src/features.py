"""
SGCC Theft Detector - Feature Engineering Module

Computes temporal, statistical, trend, and anomaly features from consumption time series.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def compute_statistical_features(consumption: np.ndarray) -> dict:
    """
    Compute statistical features from consumption time series.
    
    Args:
        consumption: Array of consumption values
    
    Returns:
        Dictionary of statistical features
    """
    # Remove NaN values
    valid_consumption = consumption[~np.isnan(consumption)]
    
    if len(valid_consumption) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'coef_var': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'skewness': 0.0
        }
    
    mean_val = np.mean(valid_consumption)
    std_val = np.std(valid_consumption)
    
    return {
        'mean': mean_val,
        'median': np.median(valid_consumption),
        'std': std_val,
        'coef_var': std_val / mean_val if mean_val > 0 else 0.0,
        'min': np.min(valid_consumption),
        'max': np.max(valid_consumption),
        'range': np.max(valid_consumption) - np.min(valid_consumption),
        'skewness': skew(valid_consumption) if len(valid_consumption) > 1 else 0.0
    }


def compute_trend_features(consumption: np.ndarray) -> dict:
    """
    Compute trend features using linear regression on time series.
    
    Args:
        consumption: Array of consumption values
    
    Returns:
        Dictionary of trend features
    """
    valid_indices = ~np.isnan(consumption)
    valid_consumption = consumption[valid_indices]
    
    if len(valid_consumption) < 2:
        return {
            'slope_full': 0.0,
            'slope_last_30d': 0.0,
            'slope_last_90d': 0.0
        }
    
    # Full series slope
    X = np.arange(len(valid_consumption)).reshape(-1, 1)
    y = valid_consumption
    
    lr = LinearRegression()
    lr.fit(X, y)
    slope_full = lr.coef_[0]
    
    # Last 30 days slope
    if len(valid_consumption) >= 30:
        X_30 = np.arange(30).reshape(-1, 1)
        y_30 = valid_consumption[-30:]
        lr.fit(X_30, y_30)
        slope_30 = lr.coef_[0]
    else:
        slope_30 = slope_full
    
    # Last 90 days slope
    if len(valid_consumption) >= 90:
        X_90 = np.arange(90).reshape(-1, 1)
        y_90 = valid_consumption[-90:]
        lr.fit(X_90, y_90)
        slope_90 = lr.coef_[0]
    else:
        slope_90 = slope_full
    
    return {
        'slope_full': slope_full,
        'slope_last_30d': slope_30,
        'slope_last_90d': slope_90
    }


def compute_anomaly_features(consumption: np.ndarray, sudden_drop_threshold: float = 0.5) -> dict:
    """
    Compute anomaly detection features.
    
    Args:
        consumption: Array of consumption values
        sudden_drop_threshold: Threshold for detecting sudden drops (default: 0.5 = 50%)
    
    Returns:
        Dictionary of anomaly features
    """
    valid_consumption = consumption[~np.isnan(consumption)]
    
    if len(valid_consumption) < 2:
        return {
            'zero_day_count': 0,
            'sudden_drop_count': 0,
            'volatility_index': 0.0
        }
    
    # Zero consumption days
    zero_count = np.sum(valid_consumption == 0)
    
    # Sudden drops (day-over-day decrease > threshold)
    sudden_drops = 0
    for i in range(1, len(valid_consumption)):
        if valid_consumption[i-1] > 0:
            pct_change = (valid_consumption[i-1] - valid_consumption[i]) / valid_consumption[i-1]
            if pct_change > sudden_drop_threshold:
                sudden_drops += 1
    
    # Volatility index (coefficient of variation)
    mean_val = np.mean(valid_consumption)
    std_val = np.std(valid_consumption)
    volatility = std_val / mean_val if mean_val > 0 else 0.0
    
    return {
        'zero_day_count': int(zero_count),
        'sudden_drop_count': sudden_drops,
        'volatility_index': volatility
    }


def compute_temporal_features(consumption: np.ndarray) -> dict:
    """
    Compute temporal pattern features.
    
    Note: Since we don't have actual dates, we simulate weekday/weekend patterns
    using a 7-day cycle assumption.
    
    Args:
        consumption: Array of consumption values
    
    Returns:
        Dictionary of temporal features
    """
    valid_consumption = consumption[~np.isnan(consumption)]
    
    if len(valid_consumption) < 7:
        return {
            'weekday_vs_weekend_ratio': 1.0,
            'peak_day_ratio': 0.0
        }
    
    # Simulate weekday/weekend pattern (assume 7-day cycle)
    # Days 5-6 in each week are "weekend"
    weekday_vals = []
    weekend_vals = []
    
    for i, val in enumerate(valid_consumption):
        day_of_week = i % 7
        if day_of_week < 5:
            weekday_vals.append(val)
        else:
            weekend_vals.append(val)
    
    weekday_mean = np.mean(weekday_vals) if len(weekday_vals) > 0 else 0.0
    weekend_mean = np.mean(weekend_vals) if len(weekend_vals) > 0 else 0.0
    
    weekday_ratio = weekday_mean / weekend_mean if weekend_mean > 0 else 1.0
    
    # Peak day ratio (frequency of top 10% consumption days)
    if len(valid_consumption) > 0:
        threshold_90 = np.percentile(valid_consumption, 90)
        peak_days = np.sum(valid_consumption >= threshold_90)
        peak_ratio = peak_days / len(valid_consumption)
    else:
        peak_ratio = 0.0
    
    return {
        'weekday_vs_weekend_ratio': weekday_ratio,
        'peak_day_ratio': peak_ratio
    }


def compute_other_features(consumption: np.ndarray) -> dict:
    """
    Compute additional features (autocorrelation, missing sequences).
    
    Args:
        consumption: Array of consumption values
    
    Returns:
        Dictionary of additional features
    """
    valid_indices = ~np.isnan(consumption)
    valid_consumption = consumption[valid_indices]
    
    if len(valid_consumption) < 2:
        return {
            'autocorr_lag1': 0.0,
            'missing_sequences_count': 0
        }
    
    # Autocorrelation at lag 1
    if len(valid_consumption) > 1:
        autocorr = pd.Series(valid_consumption).autocorr(lag=1)
        autocorr = autocorr if not np.isnan(autocorr) else 0.0
    else:
        autocorr = 0.0
    
    # Count missing sequences (>3 consecutive days)
    missing_sequences = 0
    current_sequence = 0
    
    for is_valid in valid_indices:
        if not is_valid:
            current_sequence += 1
        else:
            if current_sequence > 3:
                missing_sequences += 1
            current_sequence = 0
    
    # Check final sequence
    if current_sequence > 3:
        missing_sequences += 1
    
    return {
        'autocorr_lag1': autocorr,
        'missing_sequences_count': missing_sequences
    }


def build_features(
    df_long: pd.DataFrame,
    labels: pd.Series,
    config: dict = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build complete feature set from long-format consumption data.
    
    Args:
        df_long: Long-format DataFrame with columns [customer_id, day_index, consumption_kwh]
        labels: Series with customer_id as index and binary labels
        config: Configuration dictionary (optional)
    
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is labels Series
    """
    logger.info("Building features from consumption time series...")
    
    # Default config
    if config is None:
        config = {
            'sudden_drop_threshold': 0.5,
            'peak_day_percentile': 0.9,
            'missing_sequence_threshold': 3
        }
    
    # Group by customer and aggregate
    customers = df_long['customer_id'].unique()
    logger.info(f"Processing {len(customers)} customers...")
    
    feature_list = []
    
    for idx, customer_id in enumerate(customers):
        if idx % 100 == 0:
            logger.info(f"Processing customer {idx}/{len(customers)}...")
        
        customer_data = df_long[df_long['customer_id'] == customer_id]
        consumption = customer_data['consumption_kwh'].values
        
        # Compute all feature groups
        features = {}
        features.update(compute_statistical_features(consumption))
        features.update(compute_trend_features(consumption))
        features.update(compute_anomaly_features(consumption, config['sudden_drop_threshold']))
        features.update(compute_temporal_features(consumption))
        features.update(compute_other_features(consumption))
        
        features['customer_id'] = customer_id
        feature_list.append(features)
    
    # Create DataFrame
    X = pd.DataFrame(feature_list)
    X = X.set_index('customer_id')
    
    # Align labels with features
    y = labels.loc[X.index]
    
    logger.info(f"Built {X.shape[1]} features for {X.shape[0]} customers")
    logger.info(f"Features: {list(X.columns)}")
    
    # Check for any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    
    return X, y


def normalize_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    scaler_path: str = "artifacts/scaler.joblib"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Normalize features using MinMaxScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        scaler_path: Path to save/load scaler
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled)
    """
    logger.info("Normalizing features with MinMaxScaler...")
    
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    
    # Transform test data if provided
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
    else:
        X_test_scaled = None
    
    # Save scaler
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    return X_train_scaled, X_test_scaled


def load_scaler(scaler_path: str = "artifacts/scaler.joblib") -> MinMaxScaler:
    """
    Load saved scaler.
    
    Args:
        scaler_path: Path to scaler file
    
    Returns:
        Loaded MinMaxScaler
    """
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")
    return scaler


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_raw
    
    logger.info("Testing feature engineering...")
    
    # Load data
    df_long, labels = load_raw("data/datasetsmall.csv")
    
    # Build features
    X, y = build_features(df_long, labels)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING TEST")
    print("="*80)
    print(f"\nFeatures shape: {X.shape}")
    print(f"\nFeature names:\n{list(X.columns)}")
    print(f"\nFeature statistics:")
    print(X.describe())
    print(f"\nLabel distribution:")
    print(y.value_counts())
