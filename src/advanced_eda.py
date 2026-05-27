"""
SGCC Theft Detector - Advanced EDA Module

Advanced analytics and visualization functions for exploratory data analysis.
Includes time-series decomposition, geospatial analysis, statistical testing, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


def decompose_seasonality(
    consumption_series: pd.Series,
    period: int = 30,
    model: str = 'additive'
) -> Dict[str, pd.Series]:
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        consumption_series: Time series of consumption values
        period: Seasonality period (default 30 for monthly patterns)
        model: 'additive' or 'multiplicative'
        
    Returns:
        Dictionary with 'trend', 'seasonal', 'residual' components
    """
    try:
        result = seasonal_decompose(
            consumption_series,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    except Exception as e:
        logger.warning(f"Seasonality decomposition failed: {e}")
        return {
            'trend': consumption_series,
            'seasonal': pd.Series(0, index=consumption_series.index),
            'residual': pd.Series(0, index=consumption_series.index)
        }


def detect_changepoints(
    consumption_series: pd.Series,
    n_changepoints: int = 5,
    model: str = 'rbf'
) -> List[int]:
    """
    Detect change points in consumption time series.
    
    Args:
        consumption_series: Time series of consumption values
        n_changepoints: Maximum number of changepoints to detect
        model: Detection model ('l1', 'l2', 'rbf', 'linear')
        
    Returns:
        List of changepoint indices
    """
    try:
        # Use Pelt algorithm for change point detection
        signal = consumption_series.values.reshape(-1, 1)
        algo = rpt.Pelt(model=model, min_size=10).fit(signal)
        changepoints = algo.predict(n_bkps=n_changepoints)
        
        # Remove last point (always included by ruptures)
        if changepoints and changepoints[-1] == len(signal):
            changepoints = changepoints[:-1]
        
        return changepoints
    except Exception as e:
        logger.warning(f"Changepoint detection failed: {e}")
        return []


def calculate_acf_pacf(
    consumption_series: pd.Series,
    max_lags: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate autocorrelation and partial autocorrelation.
    
    Args:
        consumption_series: Time series of consumption values
        max_lags: Maximum number of lags
        
    Returns:
        Tuple of (ACF values, PACF values)
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    try:
        acf_values = acf(consumption_series.dropna(), nlags=max_lags)
        pacf_values = pacf(consumption_series.dropna(), nlags=max_lags)
        return acf_values, pacf_values
    except Exception as e:
        logger.warning(f"ACF/PACF calculation failed: {e}")
        return np.zeros(max_lags + 1), np.zeros(max_lags + 1)


def calculate_rolling_statistics(
    consumption_series: pd.Series,
    windows: List[int] = [7, 30]
) -> Dict[int, Dict[str, pd.Series]]:
    """
    Calculate rolling statistics for different window sizes.
    
    Args:
        consumption_series: Time series of consumption values
        windows: List of window sizes in days
        
    Returns:
        Dictionary mapping window size to dict of stats (mean, std, min, max)
    """
    results = {}
    
    for window in windows:
        results[window] = {
            'mean': consumption_series.rolling(window=window, min_periods=1).mean(),
            'std': consumption_series.rolling(window=window, min_periods=1).std(),
            'min': consumption_series.rolling(window=window, min_periods=1).min(),
            'max': consumption_series.rolling(window=window, min_periods=1).max()
        }
    
    return results


def calculate_anomaly_scores(
    features_df: pd.DataFrame,
    method: str = 'zscore'
) -> pd.Series:
    """
    Calculate anomaly scores for each sample.
    
    Args:
        features_df: DataFrame of features
        method: 'zscore' or 'iqr'
        
    Returns:
        Series of anomaly scores
    """
    if method == 'zscore':
        # Calculate z-scores for each feature
        z_scores = np.abs(stats.zscore(features_df, nan_policy='omit'))
        # Composite anomaly score (max z-score across features)
        anomaly_scores = np.max(z_scores, axis=1)
        
    elif method == 'iqr':
        # Calculate IQR-based anomaly scores
        Q1 = features_df.quantile(0.25)
        Q3 = features_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Count features outside 1.5*IQR range
        outlier_counts = ((features_df < (Q1 - 1.5 * IQR)) | 
                         (features_df > (Q3 + 1.5 * IQR))).sum(axis=1)
        anomaly_scores = outlier_counts / len(features_df.columns)
    else:
        anomaly_scores = pd.Series(0, index=features_df.index)
    
    return pd.Series(anomaly_scores, index=features_df.index)


def create_consumption_calendar(
    df_long: pd.DataFrame,
    customer_id: str
) -> pd.DataFrame:
    """
    Create calendar heatmap data (day-of-week vs week-of-year).
    
    Args:
        df_long: Long-format consumption data
        customer_id: Customer ID to analyze
        
    Returns:
        DataFrame with day_of_week, week_of_year, and consumption columns
    """
    customer_data = df_long[df_long['customer_id'] == customer_id].copy()
    
    # Add temporal features
    customer_data['day_of_week'] = customer_data['day_index'] % 7
    customer_data['week_of_year'] = customer_data['day_index'] // 7
    
    # Aggregate by week and day
    calendar_data = customer_data.groupby(['week_of_year', 'day_of_week'])['consumption_kwh'].mean().reset_index()
    
    return calendar_data


def calculate_feature_importance_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.
    
    Args:
        X: Features dataframe
        y: Target labels
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        DataFrame with feature names and importance scores
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10
    )
    
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def perform_hypothesis_tests(
    features_df: pd.DataFrame,
    labels: pd.Series
) -> Dict[str, Dict]:
    """
    Perform statistical hypothesis tests comparing theft vs honest customers.
    
    Args:
        features_df: Features dataframe
        labels: Binary labels (0=honest, 1=theft)
        
    Returns:
        Dictionary mapping feature names to test results
    """
    results = {}
    
    honest_data = features_df[labels == 0]
    theft_data = features_df[labels == 1]
    
    for col in features_df.columns:
        # T-test
        t_stat, t_pvalue = stats.ttest_ind(
            honest_data[col].dropna(),
            theft_data[col].dropna()
        )
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            honest_data[col].dropna(),
            theft_data[col].dropna(),
            alternative='two-sided'
        )
        
        results[col] = {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue,
            'significant_005': t_pvalue < 0.05
        }
    
    return results


def cluster_consumption_patterns(
    features_df: pd.DataFrame,
    method: str = 'kmeans',
    n_clusters: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster consumption patterns.
    
    Args:
        features_df: Features dataframe
        method: 'kmeans' or 'dbscan'
        n_clusters: Number of clusters (for kmeans)
        random_state: Random seed
        
    Returns:
        Array of cluster labels
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(features_df)
        
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(features_df)
    else:
        labels = np.zeros(len(features_df))
    
    return labels


# Geospatial helper functions

def generate_mock_coordinates(
    n_customers: int,
    center_lat: float = 36.7,
    center_lon: float = 3.2,
    spread: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate mock GPS coordinates for customers (for demonstration).
    
    Args:
        n_customers: Number of customers
        center_lat: Center latitude (default: Algiers, Algeria)
        center_lon: Center longitude
        spread: Geographic spread in degrees
        random_state: Random seed
        
    Returns:
        DataFrame with customer_id, latitude, longitude
    """
    np.random.seed(random_state)
    
    coords = pd.DataFrame({
        'customer_id': range(n_customers),
        'latitude': np.random.normal(center_lat, spread, n_customers),
        'longitude': np.random.normal(center_lon, spread, n_customers)
    })
    
    return coords


def calculate_theft_density(
    coordinates: pd.DataFrame,
    theft_labels: pd.Series,
    grid_size: int = 20
) -> pd.DataFrame:
    """
    Calculate theft density on a grid.
    
    Args:
        coordinates: DataFrame with latitude, longitude
        theft_labels: Binary theft labels
        grid_size: Grid resolution
        
    Returns:
        DataFrame with lat, lon, theft_density
    """
    # Create grid
    lat_bins = np.linspace(coordinates['latitude'].min(), coordinates['latitude'].max(), grid_size)
    lon_bins = np.linspace(coordinates['longitude'].min(), coordinates['longitude'].max(), grid_size)
    
    # Bin coordinates
    coordinates['lat_bin'] = pd.cut(coordinates['latitude'], bins=lat_bins, labels=False)
    coordinates['lon_bin'] = pd.cut(coordinates['longitude'], bins=lon_bins, labels=False)
    
    # Calculate theft rate per grid cell
    coordinates['is_theft'] = theft_labels.values
    
    density = coordinates.groupby(['lat_bin', 'lon_bin'])['is_theft'].agg(['sum', 'count']).reset_index()
    density['theft_rate'] = density['sum'] / density['count']
    
    # Map back to actual coordinates
    density['latitude'] = lat_bins[density['lat_bin']]
    density['longitude'] = lon_bins[density['lon_bin']]
    
    return density[['latitude', 'longitude', 'theft_rate']]


# Visualization functions

def plot_seasonality_decomposition(decomposition: Dict[str, pd.Series]) -> go.Figure:
    """Create interactive plotly figure for seasonality decomposition."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08
    )
    
    # Original (reconstructed)
    original = decomposition['trend'] + decomposition['seasonal'] + decomposition['residual']
    fig.add_trace(
        go.Scatter(y=original.values, mode='lines', name='Original', line=dict(color='#00C2A8')),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(y=decomposition['trend'].values, mode='lines', name='Trend', line=dict(color='#FFB020')),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(y=decomposition['seasonal'].values, mode='lines', name='Seasonal', line=dict(color='#8B5CF6')),
        row=3, col=1
    )
    
    # Residual
    fig.add_trace(
        go.Scatter(y=decomposition['residual'].values, mode='lines', name='Residual', line=dict(color='#EF4444')),
        row=4, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template='plotly_dark',
        title_text="Time Series Decomposition"
    )
    
    return fig


def plot_acf_pacf_combined(acf_values: np.ndarray, pacf_values: np.ndarray) -> go.Figure:
    """Create combined ACF/PACF plot."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)')
    )
    
    lags = np.arange(len(acf_values))
    
    # ACF
    fig.add_trace(
        go.Bar(x=lags, y=acf_values, name='ACF', marker_color='#00C2A8'),
        row=1, col=1
    )
    
    # PACF
    fig.add_trace(
        go.Bar(x=lags, y=pacf_values, name='PACF', marker_color='#FFB020'),
        row=1, col=2
    )
    
    # Add confidence intervals (±1.96/√n)
    n = len(acf_values)
    conf_interval = 1.96 / np.sqrt(n)
    
    for col in [1, 2]:
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=col)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=col)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig


def plot_consumption_calendar_heatmap(calendar_data: pd.DataFrame) -> go.Figure:
    """Create calendar heatmap of consumption patterns."""
    # Pivot for heatmap
    pivot = calendar_data.pivot(index='day_of_week', columns='week_of_year', values='consumption_kwh')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='Viridis',
        colorbar=dict(title="kWh")
    ))
    
    fig.update_layout(
        title="Consumption Calendar (Day of Week vs Week of Year)",
        xaxis_title="Week of Year",
        yaxis_title="Day of Week",
        template='plotly_dark',
        height=400
    )
    
    return fig
