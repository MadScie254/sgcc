"""
SGCC Theft Detector - Modeling Module

XGBoost classifier with Optuna hyperparameter optimization.
Optimizes composite score weighted toward recall for theft detection.
"""

import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, List
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
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    score = (recall_weight * recall + 
             precision_weight * precision + 
             f1_weight * f1)
    
    return float(score)


def _build_smote_enn(
    smote_enn_params: Optional[Dict] = None,
    random_state: int = 42
) -> SMOTEENN:
    """Create a configured SMOTEENN resampler from config-style parameters."""
    smote_enn_params = smote_enn_params or {}
    smote_config = smote_enn_params.get('smote', smote_enn_params)
    enn_config = smote_enn_params.get('enn', {})

    sampling_strategy = smote_config.get(
        'sampling_strategy',
        smote_enn_params.get('sampling_strategy', 'auto')
    )
    smote_k_neighbors = smote_config.get(
        'k_neighbors',
        smote_enn_params.get('k_neighbors', 5)
    )
    enn_n_neighbors = enn_config.get(
        'n_neighbors',
        smote_enn_params.get('enn_n_neighbors', 3)
    )

    return SMOTEENN(
        smote=SMOTE(
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        ),
        enn=EditedNearestNeighbours(
            n_neighbors=enn_n_neighbors,
            sampling_strategy='all',
        ),
        random_state=random_state,
    )


def build_cv_pipeline(
    smote_enn_params: Optional[Dict],
    xgb_params: Dict,
    random_state: int = 42
) -> ImbPipeline:
    """Build the per-fold pipeline used during Optuna cross-validation."""
    return ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('resample', _build_smote_enn(smote_enn_params, random_state=random_state)),
        ('clf', xgb.XGBClassifier(**xgb_params)),
    ])


def evaluate_pipeline_cv(
    pipeline_factory: Callable[[], ImbPipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    random_state: int,
    recall_weight: float = 0.60,
    precision_weight: float = 0.25,
    f1_weight: float = 0.15
) -> List[float]:
    """Run per-fold evaluation for a pipeline factory and return fold scores."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        pipeline = pipeline_factory()

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_val_fold)

        score = composite_score(
            y_val_fold,
            y_pred,
            recall_weight,
            precision_weight,
            f1_weight,
        )
        scores.append(score)

    return scores


def train_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 60,
    cv: int = 5,
    random_state: int = 42,
    smote_enn_params: Optional[Dict] = None,
    recall_weight: float = 0.60,
    precision_weight: float = 0.25,
    f1_weight: float = 0.15,
    timeout: Optional[int] = None
) -> Tuple[Dict, optuna.Study]:
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
        Tuple of (best_params, study)
    """
    logger.info(f"Starting Optuna optimization with {n_trials} trials, {cv}-fold CV...")
    logger.info(f"Composite score weights: recall={recall_weight}, precision={precision_weight}, f1={f1_weight}")
    
    smote_enn_params = smote_enn_params or {}
    
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
        
        pipeline_factory = lambda: build_cv_pipeline(
            smote_enn_params,
            param,
            random_state=random_state,
        )

        scores = evaluate_pipeline_cv(
            pipeline_factory,
            X,
            y,
            cv=cv,
            random_state=random_state,
            recall_weight=recall_weight,
            precision_weight=precision_weight,
            f1_weight=f1_weight,
        )

        mean_score = float(sum(scores) / len(scores))
        return mean_score
    
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
    
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'use_label_encoder': False,
        'random_state': random_state,
        'n_jobs': -1
    })
    
    logger.info("Training complete!")
    
    return best_params, study


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
    params, study = train_xgb_with_optuna(
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
    
    # Save a quick demo model fit for the module smoke test
    demo_model = get_xgb_model(params)
    demo_model.fit(X, y, verbose=False)
    save_model(demo_model)
    print(f"\n[INFO] Model saved to models/xgb_best.joblib")
    
    # Save study
    save_optuna_study(study)
    print(f"[INFO] Study saved to artifacts/optuna_study.pkl")
