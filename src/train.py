"""
SGCC Theft Detector - Training Module

End-to-end training pipeline: load data, engineer features, preprocess, train model.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_raw, save_processed_features
from features import build_features, normalize_features
from preprocessing import apply_smote_enn, save_preprocessing_report
from modeling import get_xgb_model, train_xgb_with_optuna, save_model, save_optuna_study

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_pipeline(
    config_path: str = "config.yaml",
    quick_mode: bool = False
) -> dict:
    """
    Run complete training pipeline.
    
    Args:
        config_path: Path to configuration YAML file
        quick_mode: If True, use quick training settings (smaller sample, fewer trials)
    
    Returns:
        Dictionary containing training results and metrics
    """
    # Load configuration
    logger.info("="*80)
    logger.info("SGCC THEFT DETECTOR - TRAINING PIPELINE")
    logger.info("="*80)
    
    config = load_config(config_path)
    random_state = config['random_state']
    
    if quick_mode:
        logger.info("[QUICK MODE ENABLED] Using reduced dataset and fewer trials")
    
    # Step 1: Load raw data
    logger.info("\n[STEP 1/7] Loading raw data...")
    data_path = config['data']['raw_data_path']
    
    try:
        df_long, labels = load_raw(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please download the dataset first using: bash scripts/download_data.sh")
        raise
    
    # Quick mode: sample data
    if quick_mode:
        sample_frac = config['model']['quick_train']['sample_fraction']
        logger.info(f"Sampling {sample_frac*100}% of data for quick training...")
        
        # Sample customers
        unique_customers = df_long['customer_id'].unique()
        n_sample = int(len(unique_customers) * sample_frac)
        sampled_customers = np.random.choice(unique_customers, size=n_sample, replace=False)
        
        df_long = df_long[df_long['customer_id'].isin(sampled_customers)]
        labels = labels.loc[sampled_customers]
        
        logger.info(f"Sampled {len(sampled_customers)} customers")
    
    # Step 2: Build features (or load cached)
    features_path = config['data']['processed_features_path']
    features_file = Path(features_path)
    
    if features_file.exists() and not quick_mode:
        logger.info("\n[STEP 2/7] Loading cached features from previous run...")
        logger.info(f"Reading from: {features_path}")
        cached_df = pd.read_csv(features_path)
        
        # Split into X and y
        y = cached_df['label']
        X = cached_df.drop(columns=['label'])
        
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Features: {X.columns.tolist()}")
    else:
        logger.info("\n[STEP 2/7] Engineering features...")
        feature_config = {
            'sudden_drop_threshold': config['features']['sudden_drop_threshold'],
            'peak_day_percentile': config['features']['peak_day_percentile'],
            'missing_sequence_threshold': config['features']['missing_sequence_threshold']
        }
        
        X, y = build_features(df_long, labels, config=feature_config)
        
        # Save features
        logger.info("Saving processed features...")
        save_processed_features(X, y, config['data']['processed_features_path'])
    
    # Step 3: Train-test split
    logger.info("\n[STEP 3/7] Splitting data into train and test sets...")
    test_size = config['evaluation']['test_size']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Train label distribution:\n{y_train.value_counts()}")
    logger.info(f"Test label distribution:\n{y_test.value_counts()}")
    
    smote_config = config['preprocessing']['smote_enn']

    # Step 4: Train model with Optuna
    logger.info("\n[STEP 4/7] Training XGBoost with Optuna optimization on the original split...")
    
    if quick_mode:
        n_trials = config['model']['quick_train']['n_trials']
        cv_folds = config['model']['quick_train']['cv_folds']
    else:
        n_trials = config['model']['optuna']['n_trials']
        cv_folds = config['model']['optuna']['cv_folds']
    
    scoring_weights = config['model']['optuna']['scoring_weights']
    
    best_params, study = train_xgb_with_optuna(
        X_train, y_train,
        n_trials=n_trials,
        cv=cv_folds,
        random_state=random_state,
        smote_enn_params=smote_config,
        recall_weight=scoring_weights['recall'],
        precision_weight=scoring_weights['precision'],
        f1_weight=scoring_weights['f1'],
        timeout=config['model']['optuna']['timeout']
    )
    
    # Step 5: Apply SMOTE+ENN for the final fit
    logger.info("\n[STEP 5/7] Applying SMOTE+ENN for the final model fit...")
    X_train_res, y_train_res, preprocess_report = apply_smote_enn(
        X_train, y_train,
        random_state=random_state,
        smote_k_neighbors=smote_config['smote']['k_neighbors'],
        enn_n_neighbors=smote_config['enn']['n_neighbors'],
        sampling_strategy=smote_config['smote']['sampling_strategy']
    )

    # Save preprocessing report
    save_preprocessing_report(preprocess_report)

    # Step 6: Normalize features and fit the final model
    logger.info("\n[STEP 6/7] Normalizing features and fitting the final model...")
    X_train_scaled, X_test_scaled = normalize_features(
        X_train_res, X_test,
        scaler_path="artifacts/scaler.joblib"
    )
    assert X_test_scaled is not None

    final_model = get_xgb_model(best_params)
    final_model.fit(X_train_scaled, y_train_res, verbose=False)

    # Step 7: Save model and study
    logger.info("\n[STEP 7/7] Saving model and artifacts...")
    model = final_model
    save_model(model, "models/xgb_best.joblib")
    save_optuna_study(study, "artifacts/optuna_study.pkl")
    
    # Save best parameters
    params_file = Path("artifacts/best_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters to {params_file}")
    
    # Save feature names
    feature_names_file = Path("artifacts/feature_names.json")
    with open(feature_names_file, 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    logger.info(f"Saved feature names to {feature_names_file}")
    
    # Return results
    results = {
        'best_score': study.best_value,
        'best_params': best_params,
        'n_trials': n_trials,
        'train_samples': len(X_train_scaled),
        'test_samples': len(X_test_scaled),
        'n_features': X_train_scaled.shape[1],
        'preprocessing_report': preprocess_report
    }
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best composite score: {results['best_score']:.4f}")
    logger.info(f"Model saved to: models/xgb_best.joblib")
    logger.info(f"Artifacts saved to: artifacts/")
    logger.info("="*80)
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test_scaled,
        'y_test': y_test
    }
    import joblib
    joblib.dump(test_data, "artifacts/test_data.pkl")
    logger.info("Saved test data for evaluation")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SGCC Theft Detector')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--quick', action='store_true', help='Enable quick training mode')
    
    args = parser.parse_args()
    
    try:
        results = train_pipeline(
            config_path=args.config,
            quick_mode=args.quick
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
