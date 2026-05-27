"""
SGCC Theft Detector - Modeling Module

XGBoost classifier with Optuna hyperparameter optimization.
Optimizes composite score weighted toward recall for theft detection.
"""

import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def get_xgb_model(params: Optional[Dict] = None) -> xgb.XGBClassifier:
    """
    Get XGBoost classifier with specified or default parameters.
    
    Args:
        params: Dictionary of XGBoost parameters (optional)
    
    Returns:
        Configured XGBClassifier instance
    """
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params is not None:
        default_params.update(params)
    
    return xgb.XGBClassifier(**default_params)


def composite_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recall_weight: float = 0.60,
    precision_weight: float = 0.25,
    f1_weight: float = 0.15
) -> float:
    """
    Calculate composite score weighted toward recall.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        recall_weight: Weight for recall (default: 0.60)
        precision_weight: Weight for precision (default: 0.25)
        f1_weight: Weight for F1 score (default: 0.15)
    
    Returns:
        Weighted composite score
    """
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    score = (recall_weight * recall + 
             precision_weight * precision + 
             f1_weight * f1)
    
    return score


def train_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 60,
    cv: int = 5,
    random_state: int = 42,
    recall_weight: float = 0.60,
    precision_weight: float = 0.25,
    f1_weight: float = 0.15,
    timeout: Optional[int] = None
) -> Tuple[xgb.XGBClassifier, Dict, optuna.Study]:
    """
    Train XGBoost with Optuna hyperparameter optimization.
    
    Optimizes composite score: 0.60*recall + 0.25*precision + 0.15*f1
    
    Args:
        X: Training features
        y: Training labels
        n_trials: Number of Optuna trials
        cv: Number of cross-validation folds
        random_state: Random seed
        recall_weight: Weight for recall in composite score
        precision_weight: Weight for precision in composite score
        f1_weight: Weight for F1 in composite score
        timeout: Timeout in seconds (optional)
    
    Returns:
        Tuple of (best_model, best_params, study)
    """
    logger.info(f"Starting Optuna optimization with {n_trials} trials, {cv}-fold CV...")
    logger.info(f"Composite score weights: recall={recall_weight}, precision={precision_weight}, f1={f1_weight}")
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        param = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 15.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Create model
        model = xgb.XGBClassifier(**param)
        
        # Cross-validation with composite scoring
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train
            model.fit(X_train_fold, y_train_fold, verbose=False)
            
            # Predict
            y_pred = model.predict(X_val_fold)
            
            # Calculate composite score
            score = composite_score(
                y_val_fold, y_pred,
                recall_weight, precision_weight, f1_weight
            )
            scores.append(score)
        
        # Return mean score
        return np.mean(scores)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best composite score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'use_label_encoder': False,
        'random_state': random_state,
        'n_jobs': -1
    })
    
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X, y, verbose=False)
    
    logger.info("Training complete!")
    
    return best_model, best_params, study


def save_model(
    model: xgb.XGBClassifier,
    model_path: str = "models/xgb_best.joblib"
) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained XGBoost model
        model_path: Path to save model
    """
    output_file = Path(model_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")


def load_model(model_path: str = "models/xgb_best.joblib") -> xgb.XGBClassifier:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded XGBoost model
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    return model


def save_optuna_study(
    study: optuna.Study,
    study_path: str = "artifacts/optuna_study.pkl"
) -> None:
    """
    Save Optuna study to disk.
    
    Args:
        study: Optuna study object
        study_path: Path to save study
    """
    output_file = Path(study_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(study, study_path)
    logger.info(f"Saved Optuna study to {study_path}")


def load_optuna_study(study_path: str = "artifacts/optuna_study.pkl") -> optuna.Study:
    """
    Load Optuna study from disk.
    
    Args:
        study_path: Path to study file
    
    Returns:
        Loaded Optuna study
    """
    study_file = Path(study_path)
    
    if not study_file.exists():
        raise FileNotFoundError(f"Optuna study not found: {study_path}")
    
    study = joblib.load(study_path)
    logger.info(f"Loaded Optuna study from {study_path}")
    
    return study


if __name__ == "__main__":
    # Test modeling with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    logger.info("Testing XGBoost with Optuna optimization...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    print("\n" + "="*80)
    print("MODELING TEST")
    print("="*80)
    print(f"\nDataset shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    # Quick optimization (5 trials for testing)
    print("\n[INFO] Running quick optimization (5 trials)...")
    model, params, study = train_xgb_with_optuna(
        X, y,
        n_trials=5,
        cv=3,
        random_state=42
    )
    
    print(f"\n[SUCCESS] Best composite score: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in params.items():
        if key not in ['objective', 'eval_metric', 'use_label_encoder', 'random_state', 'n_jobs']:
            print(f"  {key}: {value}")
    
    # Save model
    save_model(model)
    print(f"\n[INFO] Model saved to models/xgb_best.joblib")
    
    # Save study
    save_optuna_study(study)
    print(f"[INFO] Study saved to artifacts/optuna_study.pkl")
