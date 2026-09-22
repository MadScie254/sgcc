"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_raw, save_processed_features, load_processed_features


@pytest.fixture(scope="module")
def small_dataset(small_dataset_path):
    """Load datasetsmall.csv once for all tests in this module."""
    return load_raw(str(small_dataset_path))


def test_load_raw_structure(small_dataset):
    """Test that load_raw returns correct data structures."""
    df_long, labels = small_dataset
    
    # Check DataFrame structure
    assert isinstance(df_long, pd.DataFrame), "df_long should be a DataFrame"
    assert isinstance(labels, pd.Series), "labels should be a Series"
    
    # Check required columns
    assert 'customer_id' in df_long.columns, "df_long must have customer_id column"
    assert 'day_index' in df_long.columns, "df_long must have day_index column"
    assert 'consumption_kwh' in df_long.columns, "df_long must have consumption_kwh column"
    
    # Check labels
    assert labels.name == 'label', "labels Series must be named 'label'"
    assert set(labels.unique()).issubset({0, 1}), "labels should only contain 0 and 1"
    
    # Check non-empty
    assert len(df_long) > 0, "df_long should not be empty"
    assert len(labels) > 0, "labels should not be empty"


def test_load_raw_customer_count(small_dataset):
    """Test that customer count matches."""
    df_long, labels = small_dataset
    
    unique_customers = df_long['customer_id'].nunique()
    label_count = len(labels)
    
    assert unique_customers == label_count, "Number of customers should match label count"


def test_load_raw_consumption_values(small_dataset):
    """Test that consumption values are reasonable."""
    df_long, labels = small_dataset
    
    consumption = df_long['consumption_kwh']
    
    # Check data type
    assert pd.api.types.is_numeric_dtype(consumption), "consumption_kwh should be numeric"
    
    # Check for reasonable values (allowing NaN)
    valid_consumption = consumption.dropna()
    assert (valid_consumption >= 0).all(), "consumption should be non-negative"


def test_save_and_load_features(tmp_path):
    """Test saving and loading processed features."""
    # Create synthetic features
    X = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10)
    }, index=[f'customer_{i}' for i in range(10)])
    
    # load_processed_features returns int32 labels; np.random.randint's default
    # dtype is platform-dependent (int32 on Windows, int64 on Linux)
    y = pd.Series(np.random.randint(0, 2, 10), index=X.index, name='label', dtype='int32')
    
    # Save
    output_path = tmp_path / "test_features.csv"
    save_processed_features(X, y, str(output_path))
    
    # Check file exists
    assert output_path.exists(), "Features file should be created"
    
    # Load
    X_loaded, y_loaded = load_processed_features(str(output_path))
    
    # Check loaded data
    assert X_loaded.shape == X.shape, "Loaded features should have same shape"
    assert y_loaded.shape == y.shape, "Loaded labels should have same shape"
    pd.testing.assert_frame_equal(X, X_loaded)
    pd.testing.assert_series_equal(y, y_loaded)


def test_load_raw_index_types(small_dataset):
    """Test that day_index is sequential."""
    df_long, labels = small_dataset
    
    # Check day_index is integer-like
    assert pd.api.types.is_integer_dtype(df_long['day_index']), "day_index should be integer type"
    
    # Check day_index starts from 0
    assert df_long['day_index'].min() == 0, "day_index should start from 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
