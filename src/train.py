"""
SGCC Theft Detector - Training Module

End-to-end training pipeline: load data, engineer features, tune, fit the final
pipeline, and evaluate it. One run writes one consistent set of artifacts: the
model, its held-out metrics, the test split, and the feature matrix it serves.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Dict, Optional
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_raw, save_processed_features
from features import build_features
from preprocessing import save_preprocessing_report
from modeling import (
    train_xgb_with_optuna,
    fit_final_pipeline,
    out_of_fold_probabilities,
    threshold_for_budget,
    get_classifier,
    save_model,
    save_optuna_study,
)
from eval import evaluate_model, ranking_metrics, save_metrics, save_feature_importance
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_output_paths(config: dict, quick_mode: bool = False) -> Dict[str, Path]:
    """
    Where a training run writes its artifacts.

    Quick runs are for smoke-testing the pipeline on a sample, so they write
    under models/quick/ and artifacts/quick/ and never replace the deployed
    model or its metrics.
    """
    paths = config.get('paths', {})
    models_dir = Path(paths.get('models', 'models'))
    artifacts_dir = Path(paths.get('artifacts', 'artifacts'))
    features_path = Path(config['data']['processed_features_path'])

    if quick_mode:
        models_dir = models_dir / 'quick'
        artifacts_dir = artifacts_dir / 'quick'
        features_path = artifacts_dir / features_path.name

    model_filename = config.get('persistence', {}).get('model_filename', 'xgb_best.joblib')

    return {
        'model': models_dir / model_filename,
        'features': features_path,
        'metrics': artifacts_dir / 'metrics.json',
        'test_data': artifacts_dir / 'test_data.pkl',
        'feature_names': artifacts_dir / 'feature_names.json',
        'feature_importance': artifacts_dir / 'feature_importance.csv',
        'best_params': artifacts_dir / 'best_params.json',
        'optuna_study': artifacts_dir / 'optuna_study.pkl',
        'preprocess_report': artifacts_dir / 'preprocess_report.json',
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> Dict[str, str]:
    import imblearn
    import sklearn
    import xgboost
    return {
        'xgboost': xgboost.__version__,
        'scikit-learn': sklearn.__version__,
        'imbalanced-learn': imblearn.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
    }


def _resampling_report(
    pipeline, X_train: pd.DataFrame, y_train: pd.Series, smote_config: dict, resampling: str
) -> dict:
    """Recreate the pipeline's fit-time resampling to report class counts."""
    step = pipeline.named_steps['resample']
    if step == 'passthrough':
        y_resampled = y_train.to_numpy()
    else:
        scaled = pipeline.named_steps['scaler'].transform(X_train)
        _, y_resampled = clone(step).fit_resample(scaled, y_train)
    original = y_train.value_counts().sort_index()
    resampled = pd.Series(y_resampled).value_counts().sort_index()
    return {
        'method': f'{resampling} (after MinMax scaling, inside the model pipeline)',
        'resampling': resampling,
        'original_distribution': {str(k): int(v) for k, v in original.items()},
        'resampled_distribution': {str(k): int(v) for k, v in resampled.items()},
        'original_total': int(len(y_train)),
        'resampled_total': int(len(y_resampled)),
        'samples_added': int(len(y_resampled) - len(y_train)),
        'smote_k_neighbors': smote_config['smote']['k_neighbors'],
        'enn_n_neighbors': smote_config['enn']['n_neighbors'],
        'sampling_strategy': smote_config['smote']['sampling_strategy'],
    }


def _select_operating_point(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    smote_config: dict,
    cv_folds: int,
    random_state: int,
    resampling: str
) -> dict:
    """
    Decision threshold and risk-tier cutoffs, chosen without the test split.

    method "inspection_budget" flags the top budget_fraction of customers by
    out-of-fold score on the training split; "medium" covers the next band up
    to medium_budget_fraction. method "fixed" uses the configured threshold.
    """
    evaluation = config['evaluation']
    settings = evaluation.get('operating_point', {'method': 'fixed', 'threshold': evaluation.get('threshold', 0.5)})
    method = settings.get('method', 'fixed')

    if method == 'fixed':
        threshold = float(settings.get('threshold', 0.5))
        return {
            'method': 'fixed',
            'threshold': threshold,
            'medium_threshold': float(settings.get('medium_threshold', 0.4)),
        }
    if method != 'inspection_budget':
        raise ValueError(f"Unknown operating_point.method {method!r}")

    budget = float(settings['budget_fraction'])
    medium_budget = float(settings.get('medium_budget_fraction', min(3 * budget, 1.0)))
    logger.info(f"Choosing threshold for a {budget:.0%} inspection budget from {cv_folds}-fold out-of-fold scores...")
    oof = out_of_fold_probabilities(
        X_train, y_train, best_params, smote_config,
        cv=cv_folds, random_state=random_state, resampling=resampling
    )
    threshold = threshold_for_budget(oof, budget)
    medium_threshold = threshold_for_budget(oof, medium_budget)
    logger.info(f"Threshold {threshold:.4f} (high), {medium_threshold:.4f} (medium)")
    return {
        'method': 'inspection_budget',
        'budget_fraction': budget,
        'medium_budget_fraction': medium_budget,
        'threshold': threshold,
        'medium_threshold': medium_threshold,
        'selected_on': f'{cv_folds}-fold out-of-fold scores on the training split',
        'oof_average_precision': float(average_precision_score(y_train, oof)),
        'oof_flag_rate': float((oof >= threshold).mean()),
    }


def train_pipeline(
    config_path: str = "config.yaml",
    quick_mode: bool = False
) -> dict:
    """
    Run complete training pipeline.

    Args:
        config_path: Path to configuration YAML file
        quick_mode: If True, use quick training settings (smaller sample, fewer
            trials) and write artifacts under models/quick and artifacts/quick

    Returns:
        Dictionary containing training results and held-out test metrics
    """
    logger.info("="*80)
    logger.info("SGCC THEFT DETECTOR - TRAINING PIPELINE")
    logger.info("="*80)

    config = load_config(config_path)
    random_state = config['random_state']
    output_paths = resolve_output_paths(config, quick_mode)
    trained_at = datetime.now(timezone.utc)

    if quick_mode:
        logger.info("[QUICK MODE ENABLED] Using reduced dataset and fewer trials")
        logger.info(f"Quick-mode artifacts go to {output_paths['model'].parent} and {output_paths['metrics'].parent}")

    # Step 1: Load raw data
    logger.info("\n[STEP 1/7] Loading raw data...")
    data_path = config['data']['raw_data_path']

    try:
        df_long, labels = load_raw(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please download the dataset first using: bash scripts/download_data.sh")
        raise

    # Quick mode: sample data (seeded, so quick runs are reproducible)
    if quick_mode:
        sample_frac = config['model']['quick_train']['sample_fraction']
        logger.info(f"Sampling {sample_frac*100}% of data for quick training...")

        unique_customers = df_long['customer_id'].unique()
        n_sample = int(len(unique_customers) * sample_frac)
        rng = np.random.default_rng(random_state)
        sampled_customers = rng.choice(unique_customers, size=n_sample, replace=False)

        df_long = df_long[df_long['customer_id'].isin(sampled_customers)]
        labels = labels.loc[sampled_customers]

        logger.info(f"Sampled {len(sampled_customers)} customers")

    # Step 2: Build features. Always rebuilt from the raw data so a stale cache
    # from another run (e.g. a quick-mode sample) can never be trained on.
    logger.info("\n[STEP 2/7] Engineering features...")
    feature_config = {
        'sudden_drop_threshold': config['features']['sudden_drop_threshold'],
        'peak_day_percentile': config['features']['peak_day_percentile'],
        'missing_sequence_threshold': config['features']['missing_sequence_threshold']
    }
    X, y = build_features(df_long, labels, config=feature_config)
    save_processed_features(X, y, str(output_paths['features']))

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
    resampling = config['preprocessing'].get('resampling', 'smote_enn')
    scoring = config['model']['optuna'].get('scoring', 'composite')
    logger.info(f"Resampling: {resampling}; Optuna scoring: {scoring}")

    # Step 4: Tune hyperparameters with Optuna (scaler -> resample -> XGBoost per fold)
    logger.info("\n[STEP 4/7] Tuning XGBoost with Optuna on the training split...")

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
        timeout=config['model']['optuna']['timeout'],
        scoring=scoring,
        resampling=resampling
    )

    # Step 5: Fit the final model as the same pipeline that was cross-validated
    logger.info(f"\n[STEP 5/7] Fitting the final scaler -> {resampling} -> XGBoost pipeline...")
    model = fit_final_pipeline(
        X_train, y_train,
        xgb_params=best_params,
        smote_enn_params=smote_config,
        random_state=random_state,
        resampling=resampling
    )
    preprocess_report = _resampling_report(model, X_train, y_train, smote_config, resampling)
    save_preprocessing_report(preprocess_report, str(output_paths['preprocess_report']))

    # Choose the operating threshold on the training split only
    operating_point = _select_operating_point(
        config, X_train, y_train, best_params, smote_config, cv_folds, random_state, resampling
    )
    threshold = operating_point['threshold']

    # Step 6: Evaluate the fitted pipeline on the untouched test split
    logger.info("\n[STEP 6/7] Evaluating on the held-out test split...")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    test_scores = model.predict_proba(X_test)[:, 1]
    operating_point['test_flag_rate'] = float((test_scores >= threshold).mean())
    metrics['operating_point'] = operating_point
    metrics['ranking'] = ranking_metrics(y_test, test_scores)

    git_commit = _git_commit()
    metrics.update({
        'model_version': f"{trained_at:%Y%m%dT%H%M%SZ}-{git_commit or 'nogit'}",
        'trained_at': trained_at.isoformat(timespec='seconds'),
        'git_commit': git_commit,
        'config_path': str(config_path),
        'quick_mode': bool(quick_mode),
        'model_path': output_paths['model'].as_posix(),
        'data': {
            'path': str(data_path),
            'sha256': _file_sha256(data_path),
            'n_customers': int(len(X)),
            'n_days': int(df_long['day_index'].nunique()),
            'theft_rate': float(y.mean()),
        },
        'split': {
            'test_size': test_size,
            'random_state': random_state,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        },
        'optuna': {
            'n_trials': int(n_trials),
            'cv_folds': int(cv_folds),
            'scoring': scoring,
            'best_cv_score': float(study.best_value),
            'scoring_weights': scoring_weights if scoring == 'composite' else None,
        },
        'resampling': resampling,
        'libraries': _library_versions(),
    })

    # Step 7: Save model and artifacts
    logger.info("\n[STEP 7/7] Saving model and artifacts...")
    save_model(model, str(output_paths['model']))
    save_metrics(metrics, str(output_paths['metrics']))
    save_feature_importance(get_classifier(model), list(X.columns), str(output_paths['feature_importance']))
    save_optuna_study(study, str(output_paths['optuna_study']))

    with open(output_paths['best_params'], 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters to {output_paths['best_params']}")

    with open(output_paths['feature_names'], 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    logger.info(f"Saved feature names to {output_paths['feature_names']}")

    # Raw (unscaled) test features: the saved pipeline applies its own scaler
    import joblib
    joblib.dump({'X_test': X_test, 'y_test': y_test}, output_paths['test_data'])
    logger.info(f"Saved test split to {output_paths['test_data']}")

    results = {
        'best_score': study.best_value,
        'best_params': best_params,
        'n_trials': n_trials,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'preprocessing_report': preprocess_report,
        'metrics': metrics,
        'output_paths': {name: path.as_posix() for name, path in output_paths.items()},
    }

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Model version: {metrics['model_version']}")
    logger.info(f"Best CV {scoring} score: {results['best_score']:.4f}")
    logger.info(
        f"Test (threshold {threshold:.4f}, flags {operating_point['test_flag_rate']:.1%}): "
        f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
        f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} pr_auc={metrics['average_precision']:.4f}"
    )
    logger.info(f"Model saved to: {output_paths['model']}")
    logger.info("="*80)

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
