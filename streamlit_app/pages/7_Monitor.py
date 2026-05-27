"""
SGCC Platform - Performance Monitoring

Monitor model performance and detect drift over time.
"""

import streamlit as st
from design_system import get_custom_css, get_icon, fa_icon, LOTTIE_ANIMATIONS, load_lottie_url
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from deploy_utils import apply_custom_theme, show_header
from monitoring import ModelMonitor, monitor_model_performance

# Apply theme
apply_custom_theme()

st.set_page_config(
    page_title="Performance Monitoring - SGCC Platform",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/eye.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_monitoring_logs():
    """Load monitoring history."""
    log_path = Path("artifacts/monitoring_log.json")
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            logs = json.load(f)
        return logs
    return []


def plot_performance_over_time(logs):
    """Plot performance metrics over time."""
    if not logs:
        return None
    
    timestamps = []
    recalls = []
    precisions = []
    
    for log in logs:
        try:
            timestamps.append(datetime.fromisoformat(log['timestamp']))
            recalls.append(log['concept_drift']['metrics']['current']['recall'])
            precisions.append(log['concept_drift']['metrics']['current']['precision'])
        except:
            continue
    
    if not timestamps:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=recalls,
        mode='lines+markers',
        name='Recall',
        line=dict(color='#00C2A8', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=precisions,
        mode='lines+markers',
        name='Precision',
        line=dict(color='#FFB020', width=3),
        marker=dict(size=8)
    ))
    
    # Add target line for recall
    fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                  annotation_text="Target Recall (85%)")
    
    fig.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        template="plotly_dark",
        height=500,
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def plot_drift_heatmap(logs):
    """Create heatmap of drift over time."""
    if not logs or len(logs) < 2:
        return None
    
    timestamps = []
    drift_pcts = []
    
    for log in logs:
        try:
            timestamps.append(datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M'))
            drift_pcts.append(log['data_drift']['ks_test']['drift_pct'])
        except:
            continue
    
    if not timestamps:
        return None
    
    fig = go.Figure(data=go.Bar(
        x=timestamps,
        y=drift_pcts,
        marker=dict(
            color=drift_pcts,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Drift %")
        ),
        text=[f"{v:.1f}%" for v in drift_pcts],
        textposition='outside'
    ))
    
    # Add warning zones
    fig.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="High Alert (30%)")
    fig.add_hline(y=15, line_dash="dash", line_color="orange",
                  annotation_text="Warning (15%)")
    
    fig.update_layout(
        title="Data Drift Percentage Over Time",
        xaxis_title="Timestamp",
        yaxis_title="% Features Drifted",
        template="plotly_dark",
        height=500
    )
    
    return fig


def plot_feature_drift_distribution(latest_log):
    """Plot distribution of drift across features."""
    if not latest_log or 'data_drift' not in latest_log:
        return None
    
    try:
        top_drifted = latest_log['data_drift']['ks_test']['top_5_drifted']
        
        features = [item['feature'] for item in top_drifted]
        ks_stats = [item['ks_statistic'] for item in top_drifted]
        
        fig = go.Figure(go.Bar(
            x=ks_stats,
            y=features,
            orientation='h',
            marker=dict(color='#FF4444'),
            text=[f"{v:.4f}" for v in ks_stats],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 5 Features with Highest Drift (KS Statistic)",
            xaxis_title="KS Statistic",
            yaxis_title="Feature",
            template="plotly_dark",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    except:
        return None



# Apply custom design system
st.markdown(get_custom_css(), unsafe_allow_html=True)

def main():
    """Main monitoring dashboard."""
    
    show_header(
        "Performance Monitoring Dashboard",
        "Track model performance and detect data/concept drift (Objective 4)",
        icon=fa_icon("eye", 22, "#00C2A8")
    )
    
    # Check if monitoring logs exist
    logs = load_monitoring_logs()
    
    if not logs:
        st.warning("Status: No monitoring data. Run monitoring analysis first.")
        
        st.markdown("""
        This dashboard tracks model performance over time and detects:
        
        • **Data Drift**: Changes in input feature distributions (KS test, PSI)
        • **Concept Drift**: Degradation in model performance
        • **Performance Tracking**: Recall, precision, F1 trends
        • **Alert System**: Automated warnings for critical issues
        
        Click below to run the first monitoring analysis.
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("Run Monitoring Analysis", type="primary", use_container_width=True):
                with st.spinner("Running monitoring analysis..."):
                    try:
                        report = monitor_model_performance()
                        st.success("Monitoring complete.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Monitoring failed: {str(e)}")
                        st.exception(e)
        
        return
    
    # Latest report
    latest_log = logs[-1]
    
    st.success(f"Monitoring active: {len(logs)} reports generated")
    
    # Create tabs
    tabs = st.tabs(["OVERVIEW", "DATA DRIFT", "CONCEPT DRIFT", "ALERTS", "HISTORY"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown("""
        <h3 style="color: #00C2A8; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
            SYSTEM HEALTH OVERVIEW
        </h3>
        """, unsafe_allow_html=True)
        
        # Status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate status
        n_alerts = len(latest_log.get('alerts', []))
        concept_drift = latest_log.get('concept_drift', {}).get('detected', False)
        drift_pct = latest_log.get('data_drift', {}).get('ks_test', {}).get('drift_pct', 0)
        current_recall = latest_log.get('concept_drift', {}).get('metrics', {}).get('current', {}).get('recall', 0)
        
        with col1:
            status_color = "#FF4444" if n_alerts > 0 else "#00C851"
            status_text = "ALERT" if n_alerts > 0 else "HEALTHY"
            
            st.markdown(f"""
            <div style="
                class="modern-card"
                border-left: 4px solid {status_color};
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                text-align: center;
            ">
                <div style="color: #AAAAAA; font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 10px;">
                    SYSTEM STATUS
                </div>
                <div style="color: {status_color}; font-size: 28px; font-weight: 700;">
                    {status_text}
                </div>
                <div style="color: #AAAAAA; font-size: 14px; margin-top: 8px;">
                    {n_alerts} active alerts
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            drift_color = "#FF4444" if drift_pct > 30 else "#FFB020" if drift_pct > 15 else "#00C851"
            
            st.markdown(f"""
            <div style="
                class="modern-card"
                border-left: 4px solid {drift_color};
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                text-align: center;
            ">
                <div style="color: #AAAAAA; font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 10px;">
                    DATA DRIFT
                </div>
                <div style="color: #FAFAFA; font-size: 36px; font-weight: 700;">
                    {drift_pct:.1f}%
                </div>
                <div style="color: {drift_color}; font-size: 14px; margin-top: 8px;">
                    features drifted
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            concept_color = "#FF4444" if concept_drift else "#00C851"
            concept_text = "DETECTED" if concept_drift else "STABLE"
            
            st.markdown(f"""
            <div style="
                class="modern-card"
                border-left: 4px solid {concept_color};
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                text-align: center;
            ">
                <div style="color: #AAAAAA; font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 10px;">
                    CONCEPT DRIFT
                </div>
                <div style="color: {concept_color}; font-size: 28px; font-weight: 700;">
                    {concept_text}
                </div>
                <div style="color: #AAAAAA; font-size: 14px; margin-top: 8px;">
                    performance check
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recall_color = "#00C851" if current_recall >= 0.85 else "#FFB020" if current_recall >= 0.7 else "#FF4444"
            
            st.markdown(f"""
            <div style="
                class="modern-card"
                border-left: 4px solid {recall_color};
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                text-align: center;
            ">
                <div style="color: #AAAAAA; font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 10px;">
                    CURRENT RECALL
                </div>
                <div style="color: #FAFAFA; font-size: 36px; font-weight: 700;">
                    {current_recall:.1%}
                </div>
                <div style="color: {recall_color}; font-size: 14px; margin-top: 8px;">
                    target: ≥85%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance over time
        if len(logs) > 1:
            fig = plot_performance_over_time(logs)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run multiple analyses to see trends.")
    
    # Tab 2: Data Drift
    with tabs[1]:
        st.markdown("""
        <h3 style="color: #FFB020; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
            DATA DRIFT ANALYSIS
        </h3>
        """, unsafe_allow_html=True)
        
        # KS Test results
        ks_results = latest_log.get('data_drift', {}).get('ks_test', {})
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Features", ks_results.get('n_features', 0))
        col2.metric("Drifted Features", ks_results.get('n_drifted', 0))
        col3.metric("Drift Percentage", f"{ks_results.get('drift_pct', 0):.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top drifted features
        fig = plot_feature_drift_distribution(latest_log)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Drift over time
        if len(logs) > 1:
            fig = plot_drift_heatmap(logs)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # PSI results
        st.markdown("---")
        st.markdown("**Population Stability Index (PSI) Analysis**")
        
        psi_results = latest_log.get('data_drift', {}).get('psi', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Features with Significant PSI", psi_results.get('n_significant', 0))
            
            if psi_results.get('significant_features'):
                st.markdown("**Significant Features:**")
                for feat in psi_results['significant_features'][:10]:
                    st.markdown(f"• {feat}")
        
        with col2:
            top_psi = psi_results.get('top_5_psi', [])
            if top_psi:
                psi_df = pd.DataFrame(top_psi)
                st.markdown("**Top 5 PSI Values:**")
                st.dataframe(psi_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **INTERPRETATION**:
        • PSI < 0.1: No significant change
        • 0.1 ≤ PSI < 0.25: Moderate change
        • PSI ≥ 0.25: Significant change (drift detected)
        """)
    
    # Tab 3: Concept Drift
    with tabs[2]:
        st.markdown("""
        <h3 style="color: #7B68EE; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
            CONCEPT DRIFT ANALYSIS
        </h3>
        """, unsafe_allow_html=True)
        
        concept_data = latest_log.get('concept_drift', {})
        metrics = concept_data.get('metrics', {})
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Reference Performance**")
            ref_metrics = metrics.get('reference', {})
            
            st.markdown(f"""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <div style="font-size: 18px; color: #00C2A8; margin-bottom: 10px;">Recall: {ref_metrics.get('recall', 0):.4f}</div>
                <div style="font-size: 18px; color: #FFB020;">Precision: {ref_metrics.get('precision', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Current Performance**")
            cur_metrics = metrics.get('current', {})
            
            st.markdown(f"""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <div style="font-size: 18px; color: #00C2A8; margin-bottom: 10px;">Recall: {cur_metrics.get('recall', 0):.4f}</div>
                <div style="font-size: 18px; color: #FFB020;">Precision: {cur_metrics.get('precision', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance drops
        st.markdown("**Performance Changes**")
        drops = metrics.get('drops', {})
        
        col1, col2 = st.columns(2)
        
        recall_drop = drops.get('recall', 0)
        precision_drop = drops.get('precision', 0)
        
        col1.metric("Recall Change", f"{recall_drop:.4f}", 
                   delta=f"{-recall_drop:.2%}", delta_color="inverse")
        col2.metric("Precision Change", f"{precision_drop:.4f}",
                   delta=f"{-precision_drop:.2%}", delta_color="inverse")
        
        if concept_data.get('detected', False):
            st.error(f"""
            Concept drift detected.

            Recall dropped by {recall_drop:.2%}. Model performance degraded.
            **Recommended action**: Retrain model with recent data.
            """)
        else:
            st.success("No concept drift detected. Model performing within acceptable range.")
    
    # Tab 4: Alerts
    with tabs[3]:
        st.markdown("""
        <h3 style="color: #FF4444; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
            ACTIVE ALERTS & RECOMMENDATIONS
        </h3>
        """, unsafe_allow_html=True)
        
        alerts = latest_log.get('alerts', [])
        
        if alerts:
            for alert in alerts:
                severity = alert['severity']
                alert_type = alert['type']
                message = alert['message']
                
                if severity == 'HIGH':
                    st.error(f"**[{severity}] {alert_type}**: {message}")
                else:
                    st.warning(f"**[{severity}] {alert_type}**: {message}")
        else:
            st.success("No active alerts")
        
        st.markdown("---")
        st.markdown("**Recommendations**")
        
        recommendations = latest_log.get('recommendations', [])
        
        for rec in recommendations:
            st.info(f"• {rec}")
        
        # Retraining button
        if concept_data.get('detected', False):
            st.markdown("---")
            st.markdown("**Actions**")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("Retrain Model", type="primary", use_container_width=True):
                    st.info("Navigate to Model Training page to retrain")
                    st.page_link("pages/2_Train.py", label="Go to Training")
    
    # Tab 5: History
    with tabs[4]:
        st.markdown("""
        <h3 style="color: #00C851; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
            MONITORING HISTORY
        </h3>
        """, unsafe_allow_html=True)
        
        # Summary table
        history_data = []
        
        for log in logs:
            try:
                timestamp = datetime.fromisoformat(log['timestamp'])
                n_alerts = len(log.get('alerts', []))
                drift_pct = log.get('data_drift', {}).get('ks_test', {}).get('drift_pct', 0)
                concept_drift = log.get('concept_drift', {}).get('detected', False)
                recall = log.get('concept_drift', {}).get('metrics', {}).get('current', {}).get('recall', 0)
                
                history_data.append({
                    'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Alerts': n_alerts,
                    'Data Drift %': f"{drift_pct:.1f}%",
                    'Concept Drift': '■ YES' if concept_drift else '□ NO',
                    'Recall': f"{recall:.4f}"
                })
            except:
                continue
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Re-run monitoring
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("▶ RUN NEW ANALYSIS", use_container_width=True):
                with st.spinner("Running monitoring analysis..."):
                    try:
                        report = monitor_model_performance()
                        st.success("■ MONITORING COMPLETE")
                        st.rerun()
                    except Exception as e:
                        st.error(f"■ MONITORING FAILED: {str(e)}")


if __name__ == "__main__":
    main()
