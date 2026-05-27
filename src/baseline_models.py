"""
Baseline Models for Comparison

Implements Logistic Regression, Random Forest, and SVM for comparative evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import joblib
from pathlib import Path
import json
import time


class BaselineModels:
    """Train and evaluate baseline models for comparison."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, random_state=42):
        """Train Logistic Regression."""
        print("Training Logistic Regression...")
        start_time = time.time()
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='liblinear',
            C=1.0
        )
        
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        training_time = time.time() - start_time
        
        print(f"Logistic Regression trained in {training_time:.2f}s")
        return model, training_time
    
    def train_random_forest(self, X_train, y_train, random_state=42):
        """Train Random Forest."""
        print("Training Random Forest...")
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        training_time = time.time() - start_time
        
        print(f"Random Forest trained in {training_time:.2f}s")
        return model, training_time
    
    def train_svm(self, X_train, y_train, random_state=42):
        """Train Support Vector Machine."""
        print("Training SVM...")
        start_time = time.time()
        
        # Use subset for SVM if dataset is large (SVM is slow)
        if len(X_train) > 5000:
            print(f"Using subset of 5000 samples for SVM (original: {len(X_train)})")
            from sklearn.utils import resample
            X_train_subset, y_train_subset = resample(
                X_train, y_train,
                n_samples=5000,
                random_state=random_state,
                stratify=y_train
            )
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=random_state,
            class_weight='balanced'
        )
        
        model.fit(X_train_subset, y_train_subset)
        
        self.models['svm'] = model
        training_time = time.time() - start_time
        
        print(f"SVM trained in {training_time:.2f}s")
        return model, training_time
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model."""
        print(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        self.results[model_name] = metrics
        
        print(f"{model_name} - Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
        
        return metrics
    
    def train_all_baselines(self, X_train, y_train, X_test, y_test, random_state=42):
        """Train and evaluate all baseline models."""
        results = {}
        
        # Logistic Regression
        lr_model, lr_time = self.train_logistic_regression(X_train, y_train, random_state)
        lr_metrics = self.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
        lr_metrics['training_time'] = lr_time
        results['logistic_regression'] = lr_metrics
        
        # Random Forest
        rf_model, rf_time = self.train_random_forest(X_train, y_train, random_state)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        rf_metrics['training_time'] = rf_time
        results['random_forest'] = rf_metrics
        
        # SVM
        svm_model, svm_time = self.train_svm(X_train, y_train, random_state)
        svm_metrics = self.evaluate_model(svm_model, X_test, y_test, 'svm')
        svm_metrics['training_time'] = svm_time
        results['svm'] = svm_metrics
        
        return results
    
    def save_models(self, output_dir="models/baselines"):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_path / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save results
        results_path = output_path / "comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved comparison results to {results_path}")
    
    def load_models(self, input_dir="models/baselines"):
        """Load trained baseline models."""
        input_path = Path(input_dir)
        
        for model_file in input_path.glob("*.joblib"):
            model_name = model_file.stem
            self.models[model_name] = joblib.load(model_file)
            print(f"Loaded {model_name}")
        
        # Load results
        results_path = input_path / "comparison_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded comparison results")
    
    def get_best_model(self, metric='recall'):
        """Identify the best performing model."""
        if not self.results:
            return None
        
        best_score = -1
        best_model_name = None
        
        for name, metrics in self.results.items():
            score = metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_name = name
        
        return best_model_name, best_score
    
    def compare_to_xgboost(self, xgboost_metrics):
        """Compare baseline models to XGBoost."""
        comparison = {}
        
        for name, metrics in self.results.items():
            comparison[name] = {
                'recall_diff': metrics['recall'] - xgboost_metrics.get('recall', 0),
                'precision_diff': metrics['precision'] - xgboost_metrics.get('precision', 0),
                'f1_diff': metrics['f1'] - xgboost_metrics.get('f1', 0),
                'roc_auc_diff': metrics['roc_auc'] - xgboost_metrics.get('auc', 0)
            }
        
        return comparison


def train_baseline_comparison(config_path="config.yaml"):
    """
    Full baseline comparison pipeline.
    
    Loads preprocessed data, trains all baseline models, evaluates them,
    and saves results for comparison with XGBoost.
    """
    from src.data_loader import load_raw
    from src.features import build_features, normalize_features
    from src.preprocessing import apply_smote_enn, save_preprocessing_report
    from sklearn.model_selection import train_test_split
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*50)
    print("BASELINE MODELS COMPARISON")
    print("="*50)
    
    # Load data
    print("\nStep 1: Loading data...")
    df_long, labels = load_raw(config['data']['raw_data_path'])
    
    # Engineer features
    print("\nStep 2: Engineering features...")
    feature_config = {
        'sudden_drop_threshold': config['features']['sudden_drop_threshold'],
        'peak_day_percentile': config['features']['peak_day_percentile'],
        'missing_sequence_threshold': config['features']['missing_sequence_threshold']
    }
    X, y = build_features(df_long, labels, config=feature_config)
    
    # Split data
    print("\nStep 3: Splitting data...")
    test_size = config['evaluation']['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=config['random_state']
    )
    
    # Apply SMOTE+ENN
    print("\nStep 4: Applying SMOTE+ENN...")
    smote_config = config['preprocessing']['smote_enn']
    X_train_resampled, y_train_resampled, preprocess_report = apply_smote_enn(
        X_train, y_train,
        random_state=config['random_state'],
        smote_k_neighbors=smote_config['smote']['k_neighbors'],
        enn_n_neighbors=smote_config['enn']['n_neighbors'],
        sampling_strategy=smote_config['smote']['sampling_strategy']
    )
    save_preprocessing_report(preprocess_report)
    
    # Normalize
    print("\nStep 5: Normalizing features...")
    X_train_scaled, X_test_scaled = normalize_features(
        X_train_resampled, X_test,
        scaler_path="artifacts/baseline_scaler.joblib"
    )
    
    # Train baselines
    print("\nStep 6: Training baseline models...")
    baseline_trainer = BaselineModels()
    results = baseline_trainer.train_all_baselines(
        X_train_scaled, y_train_resampled,
        X_test_scaled, y_test,
        random_state=config['random_state']
    )
    
    # Save models
    print("\nStep 7: Saving models...")
    baseline_trainer.save_models()
    
    print("\n" + "="*50)
    print("BASELINE COMPARISON COMPLETE")
    print("="*50)
    
    # Print summary
    print("\nModel Performance Summary:")
    print(f"{'Model':<20} {'Recall':<10} {'Precision':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['recall']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['roc_auc']:<10.4f}")
    
    # Best model
    best_name, best_score = baseline_trainer.get_best_model('recall')
    print(f"\nBest model by recall: {best_name} ({best_score:.4f})")
    
    return results


if __name__ == "__main__":
    results = train_baseline_comparison()
