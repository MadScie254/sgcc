"""
SGCC Theft Detector - Evaluation Module

Model evaluation, metrics computation, error analysis, and explainability.
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, roc_curve, precision_recall_curve
)
from imblearn.metrics import geometric_mean_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'threshold': threshold,
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'gmean': float(geometric_mean_score(y_test, y_pred)),
        'mcc': float(matthews_corrcoef(y_test, y_pred))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    # Per-class support
    metrics['support'] = {
        'class_0': int((y_test == 0).sum()),
        'class_1': int((y_test == 1).sum())
    }
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"G-Mean: {metrics['gmean']:.4f}")
    logger.info(f"MCC: {metrics['mcc']:.4f}")
    
    return metrics


def save_metrics(
    metrics: Dict,
    output_path: str = "artifacts/metrics.json"
) -> None:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {output_path}")


def save_feature_importance(
    model,
    feature_names: List[str],
    output_path: str = "artifacts/feature_importance.csv"
) -> pd.DataFrame:
    """
    Save feature importance to CSV.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        output_path: Path to save CSV file
    
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_importance.to_csv(output_file, index=False)
    logger.info(f"Saved feature importance to {output_path}")
    
    return df_importance


def error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    max_samples: int = 100
) -> Dict:
    """
    Perform error analysis and save false positive/negative samples.
    
    Args:
        X_test: Test features with customer_id index
        y_test: True labels
        y_pred: Predicted labels
        max_samples: Maximum number of error samples to save
    
    Returns:
        Dictionary with error analysis results
    """
    logger.info("Performing error analysis...")
    
    # Get error indices
    errors = y_test != y_pred
    
    # False positives (predicted 1, actually 0)
    fp_mask = (y_pred == 1) & (y_test == 0)
    fp_ids = X_test.index[fp_mask].tolist()[:max_samples]
    
    # False negatives (predicted 0, actually 1)
    fn_mask = (y_pred == 0) & (y_test == 1)
    fn_ids = X_test.index[fn_mask].tolist()[:max_samples]
    
    logger.info(f"False positives: {len(fp_ids)} (saved: {min(len(fp_ids), max_samples)})")
    logger.info(f"False negatives: {len(fn_ids)} (saved: {min(len(fn_ids), max_samples)})")
    
    # Save error samples
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    
    with open("artifacts/fp_ids.json", 'w') as f:
        json.dump(fp_ids, f, indent=2)
    
    with open("artifacts/fn_ids.json", 'w') as f:
        json.dump(fn_ids, f, indent=2)
    
    logger.info("Saved error samples to artifacts/")
    
    return {
        'total_errors': int(errors.sum()),
        'false_positives': len(fp_ids),
        'false_negatives': len(fn_ids),
        'fp_ids': fp_ids,
        'fn_ids': fn_ids
    }


def generate_shap_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 1000,
    output_dir: str = "artifacts"
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X_train: Training data for background (can be sample)
        X_test: Test data to explain
        max_samples: Maximum samples for SHAP summary
        output_dir: Directory to save SHAP artifacts
    
    Returns:
        Tuple of (explainer, shap_values)
    """
    logger.info("Generating SHAP explanations...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (limit to max_samples)
    X_explain = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    shap_values = explainer.shap_values(X_explain)
    
    logger.info(f"Computed SHAP values for {len(X_explain)} samples")
    
    # Save SHAP values
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/shap_values.npy", shap_values)
    logger.info(f"Saved SHAP values to {output_dir}/shap_values.npy")
    
    # Generate and save summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_explain,
        show=False,
        plot_type='bar'
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_dir}/shap_summary.png")
    
    # Generate beeswarm plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_explain,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to {output_dir}/shap_beeswarm.png")
    
    return explainer, shap_values


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str = "artifacts/confusion_matrix.png"
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Honest', 'Theft'],
        yticklabels=['Honest', 'Theft']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_pr_curves(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = "artifacts"
) -> None:
    """
    Plot and save ROC and PR curves.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plots
    """
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC and PR curves to {output_dir}/")


if __name__ == "__main__":
    # Test evaluation with saved model
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent))
    from modeling import load_model
    
    logger.info("Running evaluation on saved model...")
    
    try:
        # Load model and test data
        model = load_model("models/xgb_best.joblib")
        test_data = joblib.load("artifacts/test_data.pkl")
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics)
        
        # Feature importance
        save_feature_importance(model, list(X_test.columns))
        
        # Error analysis
        y_pred = model.predict(X_test)
        error_analysis(X_test, y_test, y_pred)
        
        # SHAP
        generate_shap_explanations(model, X_test[:100], X_test)
        
        # Plots
        plot_confusion_matrix(y_test, y_pred)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        plot_roc_pr_curves(y_test, y_pred_proba)
        
        logger.info("Evaluation complete!")
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {str(e)}")
        logger.error("Please run training first: python -m src.train")
