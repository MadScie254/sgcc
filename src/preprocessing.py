"""
SGCC Theft Detector - Preprocessing Module

Handles SMOTE+ENN resampling for handling class imbalance.
"""

import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import json
import logging
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def apply_smote_enn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    enn_n_neighbors: int = 3,
    sampling_strategy: str = 'auto'
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply SMOTE+ENN preprocessing to handle class imbalance.
    
    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples
    for the minority class, while ENN (Edited Nearest Neighbours) removes noisy
    samples from both classes.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        random_state: Random seed for reproducibility
        smote_k_neighbors: Number of neighbors for SMOTE
        enn_n_neighbors: Number of neighbors for ENN
        sampling_strategy: Sampling strategy ('auto', 'minority', or ratio)
    
    Returns:
        Tuple of (X_resampled, y_resampled, report_dict)
    """
    logger.info("Applying SMOTE+ENN preprocessing...")
    
    # Get original class distribution
    original_dist = y_train.value_counts().to_dict()
    logger.info(f"Original class distribution: {original_dist}")
    
    # Check if we have enough samples for SMOTE
    min_class_count = y_train.value_counts().min()
    if min_class_count < smote_k_neighbors + 1:
        logger.warning(f"Minority class has only {min_class_count} samples. Adjusting k_neighbors...")
        smote_k_neighbors = max(1, min_class_count - 1)
    
    # Create SMOTE+ENN object
    try:
        smote_enn = SMOTEENN(
            smote=SMOTE(
                k_neighbors=smote_k_neighbors,
                random_state=random_state,
                sampling_strategy=sampling_strategy
            ),
            enn=EditedNearestNeighbours(
                n_neighbors=enn_n_neighbors,
                sampling_strategy='all'
            ),
            random_state=random_state
        )
        
        # Fit and resample
        X_res, y_res = smote_enn.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame and Series
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name='label')
        
        # Get resampled class distribution
        resampled_dist = y_res.value_counts().to_dict()
        logger.info(f"Resampled class distribution: {resampled_dist}")
        
        # Calculate statistics
        original_total = len(y_train)
        resampled_total = len(y_res)
        
        report = {
            'method': 'SMOTE+ENN',
            'original_distribution': original_dist,
            'resampled_distribution': resampled_dist,
            'original_total': int(original_total),
            'resampled_total': int(resampled_total),
            'samples_added': int(resampled_total - original_total),
            'smote_k_neighbors': smote_k_neighbors,
            'enn_n_neighbors': enn_n_neighbors,
            'sampling_strategy': sampling_strategy
        }
        
        logger.info(f"Preprocessing complete. Total samples: {original_total} -> {resampled_total}")
        logger.info(f"Samples added: {report['samples_added']}")
        
        return X_res, y_res, report
        
    except Exception as e:
        logger.error(f"SMOTE+ENN failed: {str(e)}")
        logger.warning("Returning original data without resampling")
        
        report = {
            'method': 'None (SMOTE+ENN failed)',
            'original_distribution': original_dist,
            'resampled_distribution': original_dist,
            'original_total': int(len(y_train)),
            'resampled_total': int(len(y_train)),
            'samples_added': 0,
            'error': str(e)
        }
        
        return X_train, y_train, report


def save_preprocessing_report(
    report: Dict,
    output_path: str = "artifacts/preprocess_report.json"
) -> None:
    """
    Save preprocessing report to JSON file.
    
    Args:
        report: Dictionary containing preprocessing statistics
        output_path: Path to save JSON report
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved preprocessing report to {output_path}")


def load_preprocessing_report(
    report_path: str = "artifacts/preprocess_report.json"
) -> Dict:
    """
    Load preprocessing report from JSON file.
    
    Args:
        report_path: Path to JSON report file
    
    Returns:
        Dictionary containing preprocessing statistics
    """
    report_file = Path(report_path)
    
    if not report_file.exists():
        raise FileNotFoundError(f"Preprocessing report not found: {report_path}")
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    logger.info(f"Loaded preprocessing report from {report_path}")
    return report


def get_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        y: Labels Series
    
    Returns:
        Dictionary mapping class labels to weights
    """
    class_counts = y.value_counts().to_dict()
    total = len(y)
    n_classes = len(class_counts)
    
    weights = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    
    logger.info(f"Calculated class weights: {weights}")
    return weights


if __name__ == "__main__":
    # Test preprocessing
    from sklearn.datasets import make_classification
    
    logger.info("Testing SMOTE+ENN preprocessing...")
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=42
    )
    
    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_train = pd.Series(y)
    
    print("\n" + "="*80)
    print("PREPROCESSING TEST")
    print("="*80)
    print(f"\nOriginal distribution:")
    print(y_train.value_counts())
    
    # Apply SMOTE+ENN
    X_res, y_res, report = apply_smote_enn(X_train, y_train)
    
    print(f"\nResampled distribution:")
    print(y_res.value_counts())
    
    print(f"\nPreprocessing report:")
    print(json.dumps(report, indent=2))
    
    # Save report
    save_preprocessing_report(report)
    print(f"\nReport saved to artifacts/preprocess_report.json")
