"""
SGCC Theft Detector - EDA Page

Exploratory Data Analysis with interactive visualizations.
"""

import streamlit as st
from design_system import get_custom_css, get_icon, fa_icon, LOTTIE_ANIMATIONS, load_lottie_url
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_loader import load_raw, load_processed_features
from deploy_utils import apply_custom_theme, show_header

# Apply theme
apply_custom_theme()

st.set_page_config(
    page_title="Data Analytics - SGCC Platform",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/chart-line.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load and cache data."""
    try:
        # Try loading processed features first
        X, y = load_processed_features("artifacts/features.csv")
        df_features = X.copy()
        df_features['label'] = y
        return df_features, None
    except:
        # Fall back to raw data
        try:
            df_long, labels = load_raw("data/datasetsmall.csv")
            return None, (df_long, labels)
        except:
            return None, None


def plot_class_distribution(y):
    """Plot class distribution."""
    counts = y.value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Honest (0)', 'Theft (1)'],
            y=[counts.get(0, 0), counts.get(1, 0)],
            marker=dict(
                color=['#00C851', '#FF4444'],
                line=dict(color='#FAFAFA', width=2)
            ),
            text=[counts.get(0, 0), counts.get(1, 0)],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_feature_distributions(df, features, label_col='label'):
    """Plot feature distributions by class."""
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Honest customers
        honest_data = df[df[label_col] == 0][feature].dropna()
        # Theft customers
        theft_data = df[df[label_col] == 1][feature].dropna()
        
        fig.add_trace(
            go.Violin(
                y=honest_data,
                name='Honest',
                line_color='#00C851',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Violin(
                y=theft_data,
                name='Theft',
                line_color='#FF4444',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=300 * n_rows,
        showlegend=True,
        title_text="Feature Distributions by Class"
    )
    
    return fig


def plot_correlation_heatmap(df, label_col='label'):
    """Plot correlation heatmap."""
    # Calculate correlations
    corr_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_dark",
        height=800,
        width=800
    )
    
    return fig


def plot_consumption_timeseries(df_long, customer_id, max_days=365):
    """Plot consumption time series for a customer."""
    customer_data = df_long[df_long['customer_id'] == customer_id].sort_values('day_index')
    
    if len(customer_data) > max_days:
        customer_data = customer_data.iloc[:max_days]
    
    # Calculate rolling mean
    customer_data['rolling_mean'] = customer_data['consumption_kwh'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Raw consumption
    fig.add_trace(go.Scatter(
        x=customer_data['day_index'],
        y=customer_data['consumption_kwh'],
        mode='lines',
        name='Daily Consumption',
        line=dict(color='#00C2A8', width=1),
        opacity=0.6
    ))
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=customer_data['day_index'],
        y=customer_data['rolling_mean'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='#FFB020', width=2)
    ))
    
    # Highlight zero days
    zero_days = customer_data[customer_data['consumption_kwh'] == 0]
    if len(zero_days) > 0:
        fig.add_trace(go.Scatter(
            x=zero_days['day_index'],
            y=zero_days['consumption_kwh'],
            mode='markers',
            name='Zero Consumption',
            marker=dict(color='#FF4444', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f"Consumption Timeline - Customer {customer_id[:12]}...",
        xaxis_title="Day Index",
        yaxis_title="Consumption (kWh)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig



# Apply custom design system
st.markdown(get_custom_css(), unsafe_allow_html=True)

def main():
    """Main EDA page."""
    
    show_header(
        "Exploratory Data Analysis",
        "Interactive visualizations and statistical insights",
        icon=fa_icon("chart-line", 22, "#00C2A8")
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df_features, raw_data = load_data()
    
    if df_features is None and raw_data is None:
        st.error("Data not available. Please download the dataset first.")
        st.code("bash scripts/download_data.sh")
        return
    
    # Create tabs
    if df_features is not None:
        tabs = st.tabs(["OVERVIEW", "DISTRIBUTIONS", "CORRELATIONS", "TIME SERIES", "COHORTS"])
    else:
        tabs = st.tabs(["TIME SERIES", "OVERVIEW"])
    
    # Tab 1: Overview
    with tabs[0]:
        if df_features is not None:
            st.markdown("### Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{len(df_features):,}")
            with col2:
                theft_pct = (df_features['label'].sum() / len(df_features)) * 100
                st.metric("Theft Rate", f"{theft_pct:.2f}%")
            with col3:
                st.metric("Features", len(df_features.columns) - 1)
            with col4:
                st.metric("Classes", df_features['label'].nunique())
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Class distribution
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = plot_class_distribution(df_features['label'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.info("""
                **Interpretation**: The dataset shows class imbalance typical of fraud detection scenarios. 
                SMOTE+ENN preprocessing balances training data while preserving real-world test distribution.
                """)
            
            with col2:
                # Statistical summary
                st.markdown("#### Key Statistics")
                
                honest_count = (df_features['label'] == 0).sum()
                theft_count = (df_features['label'] == 1).sum()
                imbalance_ratio = honest_count / theft_count if theft_count > 0 else 0
                
                stats_df = pd.DataFrame({
                    'Metric': ['Honest Customers', 'Theft Cases', 'Imbalance Ratio', 'Minority %'],
                    'Value': [
                        f"{honest_count:,}",
                        f"{theft_count:,}",
                        f"{imbalance_ratio:.2f}:1",
                        f"{theft_pct:.2f}%"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Feature list
                with st.expander("Feature List", expanded=False):
                    features = [col for col in df_features.columns if col != 'label']
                    for i, feat in enumerate(features, 1):
                        st.text(f"{i}. {feat}")
        
        else:
            df_long, labels = raw_data
            st.markdown("### Raw Data Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{df_long['customer_id'].nunique():,}")
            with col2:
                st.metric("Total Days", f"{df_long['day_index'].max() + 1:,}")
            with col3:
                theft_pct = (labels.sum() / len(labels)) * 100
                st.metric("Theft Rate", f"{theft_pct:.2f}%")
            
            st.info("Raw data loaded. Run feature engineering to see full EDA.")
    
    # Tab 2: Distributions (features only)
    if df_features is not None and len(tabs) > 1:
        with tabs[1]:
            st.markdown("### Feature Distributions by Class")
            
            # Select features to plot
            all_features = [col for col in df_features.columns if col != 'label']
            selected_features = st.multiselect(
                "Select features to visualize",
                all_features,
                default=all_features[:6]
            )
            
            if selected_features:
                fig = plot_feature_distributions(df_features, selected_features)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation**: Violin plots show distribution differences between honest and theft customers. 
                Look for features with clear separation - these are strong predictors. 
                Overlapping distributions indicate weaker individual predictors.
                """)
    
    # Tab 3: Correlations
    if df_features is not None and len(tabs) > 2:
        with tabs[2]:
            st.markdown("### Feature Correlations")
            
            # Correlation with target
            correlations = df_features.corr()['label'].drop('label').sort_values(ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Top Positive Correlations with Theft")
                top_pos = correlations.head(10)
                
                fig = go.Figure(go.Bar(
                    x=top_pos.values,
                    y=top_pos.index,
                    orientation='h',
                    marker=dict(color='#FF4444')
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Correlation with Theft Label",
                    yaxis_title="Feature"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Top Negative Correlations with Theft")
                top_neg = correlations.tail(10)
                
                fig = go.Figure(go.Bar(
                    x=top_neg.values,
                    y=top_neg.index,
                    orientation='h',
                    marker=dict(color='#00C851')
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Correlation with Theft Label",
                    yaxis_title="Feature"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretation**: Features with strong positive correlation increase theft probability, 
            while negative correlation decreases it. The model learns complex non-linear combinations 
            beyond these simple correlations.
            """)
            
            # Full correlation matrix
            with st.expander("Full Correlation Matrix", expanded=False):
                fig = plot_correlation_heatmap(df_features)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Advanced Time Series Analysis
    time_tab_idx = 3 if df_features is not None else 0
    if raw_data is not None or df_features is not None:
        with tabs[time_tab_idx]:
            st.markdown("### Advanced Time Series & Anomaly Detection")
            
            if raw_data:
                df_long, labels = raw_data
                from advanced_eda import (
                    decompose_seasonality,
                    detect_changepoints,
                    calculate_acf_pacf,
                    plot_seasonality_decomposition,
                    plot_acf_pacf_combined
                )
                
                # Customer selector
                all_customers = df_long['customer_id'].unique()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_customer = st.selectbox(
                        "Select Customer ID",
                        all_customers,
                        format_func=lambda x: f"{x[:12]}..."
                    )
                
                with col2:
                    customer_label = labels.loc[selected_customer] if selected_customer in labels.index else "Unknown"
                    if customer_label == 1:
                        st.error("Theft detected")
                    elif customer_label == 0:
                        st.success("Honest customer")
                    else:
                        st.info("Unknown status")
                
                # Get usage series
                customer_series = df_long[df_long['customer_id'] == selected_customer].set_index('day_index')['consumption_kwh']
                
                # 1. Seasonality Decomposition
                st.markdown("#### 1. Seasonality Decomposition")
                with st.spinner("Decomposing time series..."):
                    decomposition = decompose_seasonality(customer_series, period=30)
                    fig = plot_seasonality_decomposition(decomposition)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("**Trend**: Underlying direction. **Seasonal**: Repeated patterns. **Residual**: Noise/Anomalies.")
                
                # 2. Changepoint Detection
                st.markdown("#### 2. Structural Change Points")
                with st.spinner("Detecting changepoints..."):
                    changepoints = detect_changepoints(customer_series, n_changepoints=5)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=customer_series, mode='lines', name='Consumption', line=dict(color='#00C2A8')))
                    
                    for cp in changepoints:
                        fig.add_vline(x=cp, line_dash="dash", line_color="#FF4444", annotation_text="Change")
                    
                    fig.update_layout(title="Detected Structural Changes", template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3. Autocorrelation
                st.markdown("#### 3. Autocorrelation Analysis")
                with st.spinner("Calculating ACF/PACF..."):
                    acf, pacf = calculate_acf_pacf(customer_series)
                    fig = plot_acf_pacf_combined(acf, pacf)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Time series visualization requires raw data. Features have been pre-aggregated.")
    
    # Tab 5: Advanced Statistical Analysis
    if df_features is not None and len(tabs) > 4:
        with tabs[4]:
            st.markdown("### Advanced Statistical Validation")
            
            from advanced_eda import perform_hypothesis_tests
            
            if st.button("Run Statistical Tests (t-test & Mann-Whitney U)"):
                with st.spinner("Calculating statistics..."):
                    results = perform_hypothesis_tests(df_features.drop(columns=['label']), df_features['label'])
                    
                    res_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Feature'})
                    res_df['Significant'] = res_df['significant_005'].apply(lambda x: 'Yes' if x else 'No')
                    
                    st.dataframe(
                        res_df[['Feature', 't_statistic', 't_pvalue', 'u_statistic', 'u_pvalue', 'Significant']]
                        .sort_values('t_pvalue'),
                        width=1000,
                        hide_index=True
                    )
            
            st.markdown("### Anomaly Clustering")
            st.info("Geospatial clustering coming in Phase 2!")


if __name__ == "__main__":
    main()
