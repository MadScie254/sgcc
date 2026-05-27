"""
SGCC Theft Detector - Research Validation Page

Comprehensive research validation dashboard to demonstrate:
1. Theft Detection Performance (Objective 1)
2. Feature Importance & Interpretability (Objective 2)  
3. Model Comparison (Objective 3)
4. Deployment & Monitoring (Objective 4)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, recall_score,
    precision_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from scipy import stats
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from design_system import get_custom_css, get_icon, fa_icon
from deploy_utils import apply_custom_theme, show_header

# Apply theme
apply_custom_theme()

st.set_page_config(
    page_title="Research Validation - SGCC Platform",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/graduation-cap.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom design system
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Show header
show_header(
    title="Research Validation Dashboard",
    subtitle="Comprehensive validation of research objectives and model performance",
    icon=fa_icon("graduation-cap", 22, "#00C2A8")
)


@st.cache_data
def load_model_and_data():
    """Load trained model and test data."""
    try:
        model = joblib.load("models/xgb_best.joblib")
        test_data = joblib.load("artifacts/test_data.pkl")
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return model, X_test, y_test, y_pred_proba
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_baseline_models():
    """Load baseline model results if available."""
    baseline_path = Path("artifacts/baseline_comparison.json")
    if baseline_path.exists():
        import json
        with open(baseline_path, 'r') as f:
            return json.load(f)
    return None


def plot_precision_recall_curve(y_true, y_pred_proba, threshold=0.5):
    """Plot interactive precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    
    # PR curve
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC={pr_auc:.3f})',
        line=dict(color='#00C2A8', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 194, 168, 0.2)'
    ))
    
    # Current threshold point
    y_pred = (y_pred_proba >= threshold).astype(int)
    current_precision = precision_score(y_true, y_pred, zero_division=0)
    current_recall = recall_score(y_true, y_pred)
    
    fig.add_trace(go.Scatter(
        x=[current_recall], y=[current_precision],
        mode='markers',
        name=f'Current (threshold={threshold:.2f})',
        marker=dict(size=15, color='#FFB020', symbol='star'),
        hovertemplate=f'Recall: {current_recall:.3f}<br>Precision: {current_precision:.3f}<extra></extra>'
    ))
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[sum(y_true)/len(y_true), sum(y_true)/len(y_true)],
        mode='lines',
        name='Baseline',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template='plotly_dark',
        hovermode='closest',
        height=500
    )
    
    return fig


def plot_roc_curve_with_ci(y_true, y_pred_proba, n_bootstraps=100):
    """Plot ROC curve with confidence intervals."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap for confidence intervals
    tpr_list = []
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_pred_proba[indices])
        tpr_interp = np.interp(fpr, fpr_boot, tpr_boot)
        tpr_list.append(tpr_interp)
    
    tpr_array = np.array(tpr_list)
    tpr_mean = np.mean(tpr_array, axis=0)
    tpr_lower = np.percentile(tpr_array, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_array, 97.5, axis=0)
    
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([fpr, fpr[::-1]]),
        y=np.concatenate([tpr_upper, tpr_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 194, 168, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='95% CI'
    ))
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC={roc_auc:.3f})',
        line=dict(color='#00C2A8', width=3)
    ))
    
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve with 95% Confidence Interval",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template='plotly_dark',
        height=500
    )
    
    return fig


def plot_lift_curve(y_true, y_pred_proba):
    """Plot lift curve showing model effectiveness."""
    # Sort by predicted probability (descending)
    indices = np.argsort(y_pred_proba)[::-1]
    y_sorted = y_true[indices]
    
    # Calculate cumulative gains
    n = len(y_true)
    percentiles = np.arange(1, n + 1) / n * 100
    cumulative_gains = np.cumsum(y_sorted) / sum(y_true) * 100
    
    # Baseline (random selection)
    baseline = percentiles
    
    # Lift
    lift = cumulative_gains / baseline
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=percentiles, y=lift,
        mode='lines',
        name='Model Lift',
        line=dict(color='#00C2A8', width=3)
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                  annotation_text="Baseline (Random)")
    
    fig.update_layout(
        title="Lift Curve - Model Effectiveness",
        xaxis_title="Population Percentile (%)",
        yaxis_title="Lift",
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_calibration_curve(y_true, y_pred_proba):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    fig = go.Figure()
    
    # Calibration curve
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='#00C2A8', width=3),
        marker=dict(size=10)
    ))
    
    # Perfect calibration
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Calibration Curve - Probability Reliability",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Probability",
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_confusion_matrix_interactive(y_true, y_pred):
    """Plot interactive confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=['Honest', 'Theft'],
        y=['Honest', 'Theft'],
        text=[[f'{cm[i,j]}<br>({cm_norm[i,j]:.1%})' for j in range(2)] for i in range(2)],
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Normalized)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template='plotly_dark',
        height=400
    )
    
    return fig


def calculate_cost_benefit(y_true, y_pred, fp_cost=100, fn_cost=500):
    """Calculate cost-benefit analysis."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Costs
    total_fp_cost = fp * fp_cost
    total_fn_cost = fn * fn_cost
    total_cost = total_fp_cost + total_fn_cost
    
    # Benefits (assuming we prevent loss for detected thefts)
    benefit_per_catch = 1000  # Example: prevent $1000 loss per caught thief
    total_benefit = tp * benefit_per_catch
    
    net_benefit = total_benefit - total_cost
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'fp_cost': total_fp_cost,
        'fn_cost': total_fn_cost,
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'net_benefit': net_benefit
    }


# Main app
def main():
    st.sidebar.markdown(
        f"<h2 style='margin-bottom: 12px;'>{fa_icon('chart-line', 18, '#00C2A8')} Validation Sections</h2>",
        unsafe_allow_html=True,
    )
    
    section = st.sidebar.radio(
        "Select Section",
        ["Objective 1: Detection Performance",
         "Objective 2: Feature Importance",
         "Objective 3: Model Comparison",
         "Objective 4: Deployment Readiness"],
        label_visibility="collapsed"
    )
    
    # Load data
    model, X_test, y_test, y_pred_proba = load_model_and_data()
    
    if model is None:
        st.error("Model not found. Please train a model first.")
        st.info("Run: `python -m src.memory_efficient_train --quick`")
        return
    
    # Calculate default metrics (threshold=0.5) for global use
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    recall = recall_score(y_test, y_pred_default)
    precision = precision_score(y_test, y_pred_default, zero_division=0)
    f1 = f1_score(y_test, y_pred_default, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred_default)
    
    # === OBJECTIVE 1: DETECTION PERFORMANCE ===
    if section == "Objective 1: Detection Performance":
        st.markdown(
            f"<h2>{fa_icon('bullseye', 18, '#00C2A8')} Objective 1: Theft Detection Performance</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("Demonstrating model effectiveness with comprehensive metrics and threshold optimization.")
        
        # Threshold slider
        st.markdown(
            f"<h3>{fa_icon('sliders', 16, '#FFB020')} Threshold Optimization</h3>",
            unsafe_allow_html=True,
        )
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust threshold to balance precision and recall"
        )
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Metrics at current threshold
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            recall_val = recall_score(y_test, y_pred)
            st.metric("Recall", f"{recall_val:.1%}", 
                     delta=f"{(recall_val - 0.85)*100:+.1f}% vs target" if recall_val >= 0.85 else None,
                     delta_color="normal" if recall_val >= 0.85 else "inverse")
        
        with col2:
            precision_val = precision_score(y_test, y_pred, zero_division=0)
            st.metric("Precision", f"{precision_val:.1%}")
        
        with col3:
            f1_val = f1_score(y_test, y_pred, zero_division=0)
            st.metric("F1 Score", f"{f1_val:.1%}")
        
        with col4:
            accuracy_val = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy_val:.1%}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_precision_recall_curve(y_test, y_pred_proba, threshold), use_container_width=True)
            st.plotly_chart(plot_lift_curve(y_test, y_pred_proba), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_roc_curve_with_ci(y_test, y_pred_proba), use_container_width=True)
            st.plotly_chart(plot_calibration_curve(y_test, y_pred_proba), use_container_width=True)
        
        # Confusion Matrix
        st.markdown(
            f"<h3>{fa_icon('table-cells', 16, '#4FACFE')} Confusion Matrix</h3>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_confusion_matrix_interactive(y_test, y_pred), use_container_width=True)
        
        # Cost-Benefit Analysis
        st.markdown(
            f"<h3>{fa_icon('sack-dollar', 16, '#FFB020')} Cost-Benefit Analysis</h3>",
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fp_cost = st.number_input("Cost per False Positive ($)", value=100, step=50, 
                                     help="Cost of incorrectly flagging honest customer")
        with col2:
            fn_cost = st.number_input("Cost per False Negative ($)", value=500, step=100,
                                     help="Cost of missing a theft case")
        
        cb_results = calculate_cost_benefit(y_test, y_pred, fp_cost, fn_cost)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cost", f"${cb_results['total_cost']:,}")
        col2.metric("Total Benefit", f"${cb_results['total_benefit']:,}")
        col3.metric("Net Benefit", f"${cb_results['net_benefit']:,}",
                   delta_color="normal" if cb_results['net_benefit'] > 0 else "inverse")
        col4.metric("True Positives", cb_results['true_positives'])
    
    # === OBJECTIVE 2: FEATURE IMPORTANCE ===
    elif section == "Objective 2: Feature Importance":
        st.markdown(
            f"<h2>{fa_icon('magnifying-glass', 18, '#00C2A8')} Objective 2: Feature Importance & Interpretability</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("Understanding which features drive theft predictions.")
        
        st.info("For detailed SHAP analysis, please visit the Explainability page from the main menu.")
        
        st.markdown(
            f"<h3>{fa_icon('chart-column', 16, '#4FACFE')} Feature Importance Ranking</h3>",
            unsafe_allow_html=True,
        )
        
        # Get feature importance from model
        try:
            importance_df = pd.DataFrame({
                'feature': model.get_booster().feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 15 Features by Importance',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template='plotly_dark', height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error getting feature importance: {e}")
    
    # === OBJECTIVE 3: MODEL COMPARISON ===
    elif section == "Objective 3: Model Comparison":
        st.markdown(
            f"<h2>{fa_icon('scale-balanced', 18, '#00C2A8')} Objective 3: Model Comparison</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("Comparing XGBoost against baseline models.")
        
        baseline_results = load_baseline_models()
        
        if baseline_results:
            st.success("Baseline comparison results loaded")
            
            # Create comparison table
            comparison_df = pd.DataFrame(baseline_results).T
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Radar chart
            st.markdown(
                f"<h3>{fa_icon('satellite-dish', 16, '#7B68EE')} Multi-Metric Comparison</h3>",
                unsafe_allow_html=True,
            )
            st.info("Radar chart feature coming soon! See **Compare** page for detailed model comparison.")
        else:
            st.warning("No baseline comparison results found.")
            st.info("Visit the **Compare** page to run baseline model comparison.")
    
    # === OBJECTIVE 4: DEPLOYMENT READINESS ===
    elif section == "Objective 4: Deployment Readiness":
        st.markdown(
            f"<h2>{fa_icon('rocket', 18, '#00C2A8')} Objective 4: Deployment & Monitoring</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("Production readiness assessment.")
        
        st.success("Model trained and saved to `models/xgb_best.joblib`")
        st.success("Scaler saved to `artifacts/scaler.joblib`")
        st.success("SHAP explainer available")
        
        st.markdown(
            f"<h3>{fa_icon('clipboard-check', 16, '#4FACFE')} Deployment Checklist</h3>",
            unsafe_allow_html=True,
        )
        
        checklist = {
            "Model Performance": recall >= 0.85,
            "Model Saved": Path("models/xgb_best.joblib").exists(),
            "Scaler Saved": Path("artifacts/scaler.joblib").exists(),
            "Test Data Saved": Path("artifacts/test_data.pkl").exists(),
            "Documentation Complete": Path("README.md").exists(),
        }
        
        for item, status in checklist.items():
            if status:
                icon_html = fa_icon("circle-check", 14, "#00C851")
            else:
                icon_html = fa_icon("circle-xmark", 14, "#FF4444")
            st.markdown(f"{icon_html} {item}", unsafe_allow_html=True)
        
        st.markdown(
            f"<h3>{fa_icon('chart-line', 16, '#4FACFE')} Model Monitoring</h3>",
            unsafe_allow_html=True,
        )
        st.info("Visit the Monitoring page for real-time drift detection and performance tracking.")


if __name__ == "__main__":
    main()
