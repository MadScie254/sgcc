"""
SGCC Theft Detector - Data Loader Module

Handles loading and preprocessing of the SGCC dataset from CSV.
Converts wide-format time series data to long format suitable for feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw(
    path: str,
    encoding: str = 'utf-8'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw SGCC dataset from CSV file.
    
    The dataset is expected in wide format where:
    - Each row represents one customer
    - Columns 0 to N-2 contain daily consumption readings
    - Column N-1 contains customer_id (32-char hex hash)
    - Column N contains binary label (0=honest, 1=theft)
    
    Args:
        path: Path to the CSV file
        encoding: File encoding (default: 'utf-8')
    
    Returns:
        Tuple of (df_long, labels) where:
        - df_long: DataFrame with columns [customer_id, day_index, consumption_kwh]
        - labels: Series with customer_id as index and binary labels (0/1)
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the data format is unexpected
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}...")
    
    # Try different encodings
    encodings_to_try = [encoding, 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            logger.info(f"Successfully loaded with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed with encoding {enc}: {str(e)}")
    
    if df is None:
        raise ValueError(f"Could not load file with any encoding: {encodings_to_try}")
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Check if data already has CUSTOMER_ID and FLAG columns (augmented data format)
    if 'CUSTOMER_ID' in df.columns and 'FLAG' in df.columns:
        logger.info("Detected augmented data format with CUSTOMER_ID and FLAG columns")
        customer_ids = df['CUSTOMER_ID'].astype(str)
        labels = df['FLAG'].astype(int)
        
        # Get consumption columns (date columns like '1/1/2014')
        # Exclude non-consumption columns
        exclude_cols = ['CUSTOMER_ID', 'FLAG', 'IS_SYNTHETIC', 'CUSTOMER_TYPE', 'THEFT_TYPE',
                       'MEAN_MONTHLY_CONSUMPTION', 'STD_MONTHLY_CONSUMPTION', 
                       'MAX_MONTHLY_CONSUMPTION', 'MIN_MONTHLY_CONSUMPTION', 'MEDIAN_MONTHLY_CONSUMPTION',
                       'CONSUMPTION_TREND', 'COEFFICIENT_OF_VARIATION', 'MAX_CONSUMPTION_DROP',
                       'MONTHS_WITH_ZERO', 'MONTHS_WITH_LOW_CONSUMPTION', 
                       'RECENT_VS_HISTORICAL_RATIO', 'QUARTERLY_STD']
        consumption_cols = [col for col in df.columns if col not in exclude_cols]
        
    else:
        # Original format: detect if we have headers or not
        # If first row contains numeric data only (except last 2 cols), it's headerless
        first_row_numeric = df.iloc[0, :-2].apply(lambda x: pd.api.types.is_numeric_dtype(type(x)) or isinstance(x, (int, float)))
        
        if not first_row_numeric.all():
            # Has headers, skip first row
            logger.info("Detected headers in first row")
            df.columns = [f'day_{i}' if i < len(df.columns)-2 else ('customer_id' if i == len(df.columns)-2 else 'FLAG') 
                         for i in range(len(df.columns))]
        else:
            # Headerless - assign column names
            logger.info("No headers detected - assigning column names")
            df.columns = [f'day_{i}' if i < len(df.columns)-2 else ('customer_id' if i == len(df.columns)-2 else 'FLAG') 
                         for i in range(len(df.columns))]
        
        # Extract customer IDs and labels
        customer_ids = df['customer_id'].astype(str)
        labels = df['FLAG'].astype(int)
        
        # Get consumption columns (all except customer_id and FLAG)
        consumption_cols = [col for col in df.columns if col not in ['customer_id', 'FLAG']]
    
    # Extract customer IDs and labels
    labels = labels.fillna(0).astype(int)  # Fill any NaN labels with 0 (honest)
    
    # Get consumption columns (all except customer_id and FLAG)
    consumption_cols = [col for col in df.columns if col not in ['customer_id', 'FLAG']]
    
    logger.info(f"Found {len(consumption_cols)} consumption columns")
    logger.info(f"Found {len(customer_ids)} customers")
    logger.info(f"Label distribution:\n{labels.value_counts()}")
    
    # Convert to long format
    logger.info("Converting to long format...")
    df_consumption = df[consumption_cols]
    
    # Replace non-numeric values with NaN
    df_consumption = df_consumption.apply(pd.to_numeric, errors='coerce')
    
    # Create long format
    records = []
    for idx, customer_id in enumerate(customer_ids):
        consumptions = df_consumption.iloc[idx].values
        for day_idx, consumption in enumerate(consumptions):
            records.append({
                'customer_id': customer_id,
                'day_index': day_idx,
                'consumption_kwh': consumption
            })
    
    df_long = pd.DataFrame(records)
    
    # Create labels series indexed by customer_id
    labels_series = pd.Series(labels.values, index=customer_ids.values, name='label')
    
    logger.info(f"Long format shape: {df_long.shape}")
    logger.info(f"Unique customers: {df_long['customer_id'].nunique()}")
    logger.info(f"Days per customer: {len(consumption_cols)}")
    
    # Basic data quality checks
    null_pct = (df_long['consumption_kwh'].isna().sum() / len(df_long)) * 100
    logger.info(f"Null consumption values: {null_pct:.2f}%")
    
    zero_pct = ((df_long['consumption_kwh'] == 0).sum() / len(df_long)) * 100
    logger.info(f"Zero consumption values: {zero_pct:.2f}%")
    
    return df_long, labels_series


def load_processed_features(path: str = "artifacts/features.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load pre-computed features from CSV.
    
    Args:
        path: Path to features CSV file
    
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is labels Series
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    
    logger.info(f"Loading processed features from {path}...")
    
    df = pd.read_csv(path, index_col=None)
    
    if 'label' not in df.columns:
        raise ValueError("Features file must contain 'label' column")
    
    y = df['label']
    X = df.drop('label', axis=1)
    
    logger.info(f"Loaded features shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    return X, y


def save_processed_features(
    X: pd.DataFrame,
    y: pd.Series,
    path: str = "artifacts/features.csv"
) -> None:
    """
    Save processed features to CSV.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        path: Output path for CSV file
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine features and labels
    df = X.copy()
    
    # Reset indices to avoid duplicate index errors
    df = df.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    df['label'] = y_reset
    
    df.to_csv(path, index=False)
    logger.info(f"Saved features to {path}")


if __name__ == "__main__":
    # Test data loader
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/datasetsmall.csv"
    
    try:
        df_long, labels = load_raw(data_path)
        print("\n" + "="*80)
        print("DATA LOADING SUCCESSFUL")
        print("="*80)
        print(f"\nLong format shape: {df_long.shape}")
        print(f"\nFirst few rows:")
        print(df_long.head(10))
        print(f"\nLabels distribution:")
        print(labels.value_counts())
        print(f"\nConsumption statistics:")
        print(df_long['consumption_kwh'].describe())
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
