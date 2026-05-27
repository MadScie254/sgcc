"""
Model Monitoring & Drift Detection

Track model performance over time and detect data/concept drift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, precision_score
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, reference_data_path="artifacts/test_data.pkl"):
        self.reference_data_path = reference_data_path
        self.monitoring_log = []
        self.drift_alerts = []
        
    def detect_data_drift_ks(self, reference_data, current_data, feature_name, threshold=0.05):
        """
        Detect data drift using Kolmogorov-Smirnov test.
        
        Returns:
            drift_detected (bool), p_value (float), ks_statistic (float)
        """
        # Remove NaNs
        ref = reference_data[~np.isnan(reference_data)]
        cur = current_data[~np.isnan(current_data)]
        
        if len(ref) == 0 or len(cur) == 0:
            return False, 1.0, 0.0
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        
        # Drift if p-value < threshold (reject null hypothesis of same distribution)
        drift_detected = p_value < threshold
        
        return drift_detected, p_value, ks_stat
    
    def detect_data_drift_psi(self, reference_data, current_data, bins=10):
        """
        Calculate Population Stability Index (PSI) for drift detection.
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change (drift)
        
        Returns:
            psi (float), drift_level (str)
        """
        # Create bins based on reference data
        try:
            _, bin_edges = np.histogram(reference_data, bins=bins)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(reference_data, bins=bin_edges)
            cur_hist, _ = np.histogram(current_data, bins=bin_edges)
            
            # Convert to percentages (avoid division by zero)
            ref_pct = (ref_hist + 1) / (len(reference_data) + bins)
            cur_pct = (cur_hist + 1) / (len(current_data) + bins)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            # Determine drift level
            if psi < 0.1:
                drift_level = 'None'
            elif psi < 0.25:
                drift_level = 'Moderate'
            else:
                drift_level = 'Significant'
            
            return psi, drift_level
        
        except Exception as e:
            return 0.0, 'Error'
    
    def analyze_feature_drift(self, X_reference, X_current, method='ks', threshold=0.05):
        """
        Analyze drift across all features.
        
        Returns:
            DataFrame with drift analysis for each feature
        """
        drift_results = []
        
        for col in X_reference.columns:
            ref_data = X_reference[col].values
            cur_data = X_current[col].values
            
            if method == 'ks':
                drift_detected, p_value, ks_stat = self.detect_data_drift_ks(
                    ref_data, cur_data, col, threshold
                )
                
                drift_results.append({
                    'feature': col,
                    'drift_detected': drift_detected,
                    'p_value': p_value,
                    'ks_statistic': ks_stat,
                    'method': 'KS-Test'
                })
            
            elif method == 'psi':
                psi, drift_level = self.detect_data_drift_psi(ref_data, cur_data)
                
                drift_results.append({
                    'feature': col,
                    'psi': psi,
                    'drift_level': drift_level,
                    'drift_detected': drift_level in ['Moderate', 'Significant'],
                    'method': 'PSI'
                })
        
        return pd.DataFrame(drift_results)
    
    def detect_concept_drift(self, model, X_reference, y_reference, X_current, y_current, threshold=0.1):
        """
        Detect concept drift by comparing model performance.
        
        Returns:
            drift_detected (bool), performance_drop (float), metrics_dict (dict)
        """
        # Evaluate on reference data
        y_pred_ref = model.predict(X_reference)
        recall_ref = recall_score(y_reference, y_pred_ref, zero_division=0)
        precision_ref = precision_score(y_reference, y_pred_ref, zero_division=0)
        
        # Evaluate on current data
        y_pred_cur = model.predict(X_current)
        recall_cur = recall_score(y_current, y_pred_cur, zero_division=0)
        precision_cur = precision_score(y_current, y_pred_cur, zero_division=0)
        
        # Calculate performance drop
        recall_drop = recall_ref - recall_cur
        precision_drop = precision_ref - precision_cur
        
        # Drift if recall drops more than threshold
        drift_detected = recall_drop > threshold
        
        metrics = {
            'reference': {
                'recall': recall_ref,
                'precision': precision_ref
            },
            'current': {
                'recall': recall_cur,
                'precision': precision_cur
            },
            'drops': {
                'recall': recall_drop,
                'precision': precision_drop
            }
        }
        
        return drift_detected, recall_drop, metrics
    
    def generate_monitoring_report(self, model, X_reference, y_reference, X_current, y_current):
        """
        Generate comprehensive monitoring report.
        
        Returns:
            Dictionary with drift analysis, performance metrics, and alerts
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': {},
            'concept_drift': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Data drift analysis (KS test)
        print("Analyzing data drift (KS test)...")
        drift_df_ks = self.analyze_feature_drift(X_reference, X_current, method='ks')
        n_drifted_features = drift_df_ks['drift_detected'].sum()
        drift_pct = (n_drifted_features / len(drift_df_ks)) * 100
        
        report['data_drift']['ks_test'] = {
            'n_features': len(drift_df_ks),
            'n_drifted': int(n_drifted_features),
            'drift_pct': drift_pct,
            'drifted_features': drift_df_ks[drift_df_ks['drift_detected']]['feature'].tolist(),
            'top_5_drifted': drift_df_ks.nsmallest(5, 'p_value')[['feature', 'p_value', 'ks_statistic']].to_dict('records')
        }
        
        # Data drift analysis (PSI)
        print("Analyzing data drift (PSI)...")
        drift_df_psi = self.analyze_feature_drift(X_reference, X_current, method='psi')
        significant_psi = drift_df_psi[drift_df_psi['drift_level'] == 'Significant']
        
        report['data_drift']['psi'] = {
            'n_significant': len(significant_psi),
            'significant_features': significant_psi['feature'].tolist(),
            'top_5_psi': drift_df_psi.nlargest(5, 'psi')[['feature', 'psi', 'drift_level']].to_dict('records')
        }
        
        # Concept drift
        print("Analyzing concept drift...")
        concept_drift, recall_drop, performance_metrics = self.detect_concept_drift(
            model, X_reference, y_reference, X_current, y_current
        )
        
        report['concept_drift'] = {
            'detected': concept_drift,
            'recall_drop': recall_drop,
            'metrics': performance_metrics
        }
        
        # Generate alerts
        if drift_pct > 30:
            report['alerts'].append({
                'severity': 'HIGH',
                'type': 'Data Drift',
                'message': f'{drift_pct:.1f}% of features showing drift'
            })
        elif drift_pct > 15:
            report['alerts'].append({
                'severity': 'MEDIUM',
                'type': 'Data Drift',
                'message': f'{drift_pct:.1f}% of features showing drift'
            })
        
        if concept_drift:
            report['alerts'].append({
                'severity': 'HIGH',
                'type': 'Concept Drift',
                'message': f'Recall dropped by {recall_drop:.1%}'
            })
        
        if len(significant_psi) > 5:
            report['alerts'].append({
                'severity': 'MEDIUM',
                'type': 'Distribution Shift',
                'message': f'{len(significant_psi)} features with significant PSI'
            })
        
        # Generate recommendations
        if concept_drift:
            report['recommendations'].append('URGENT: Retrain model with recent data')
        
        if drift_pct > 20:
            report['recommendations'].append('Review feature engineering pipeline')
            report['recommendations'].append('Investigate data quality issues')
        
        if len(significant_psi) > 0:
            report['recommendations'].append(f'Monitor features: {", ".join(significant_psi["feature"].tolist()[:5])}')
        
        if len(report['alerts']) == 0:
            report['recommendations'].append('Model performing well - continue monitoring')
        
        return report
    
    def save_monitoring_log(self, report, output_path="artifacts/monitoring_log.json"):
        """Save monitoring report to disk."""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        # Load existing log
        if output_file.exists():
            with open(output_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new report
        logs.append(report)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Monitoring report saved to {output_path}")
    
    def load_monitoring_history(self, input_path="artifacts/monitoring_log.json"):
        """Load historical monitoring data."""
        input_file = Path(input_path)
        
        if input_file.exists():
            with open(input_file, 'r') as f:
                logs = json.load(f)
            return logs
        else:
            return []


def monitor_model_performance(model_path="models/xgb_best.joblib",
                              reference_data_path="artifacts/test_data.pkl",
                              current_data_path="artifacts/test_data.pkl"):
    """
    Run model monitoring pipeline.
    
    In production, current_data_path would be recent production data.
    For demo, we use test data as both reference and current.
    """
    import joblib
    from src.modeling import load_model
    
    print("="*50)
    print("MODEL MONITORING PIPELINE")
    print("="*50)
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path)
    
    # Load reference data
    print("Loading reference data...")
    ref_data = joblib.load(reference_data_path)
    X_ref = ref_data['X_test']
    y_ref = ref_data['y_test']
    
    # Load current data (in production, this would be recent data)
    print("Loading current data...")
    cur_data = joblib.load(current_data_path)
    X_cur = cur_data['X_test']
    y_cur = cur_data['y_test']
    
    # Run monitoring
    monitor = ModelMonitor()
    report = monitor.generate_monitoring_report(model, X_ref, y_ref, X_cur, y_cur)
    
    # Save report
    monitor.save_monitoring_log(report)
    
    print("\n" + "="*50)
    print("MONITORING COMPLETE")
    print("="*50)
    
    # Print summary
    print(f"\nAlerts: {len(report['alerts'])}")
    for alert in report['alerts']:
        print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return report


if __name__ == "__main__":
    report = monitor_model_performance()
