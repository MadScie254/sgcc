"""
SGCC Theft Detector - Explain Page

Model explainability with SHAP analysis.
"""

import streamlit as st
from design_system import get_custom_css, get_icon, fa_icon, LOTTIE_ANIMATIONS, load_lottie_url
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from deploy_utils import apply_custom_theme, show_header
from modeling import load_model

# Apply theme
apply_custom_theme()

st.set_page_config(
    page_title="Model Explainability - SGCC Platform",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/lightbulb.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_cached():
    """Load trained model."""
    try:
        return load_model("models/xgb_best.joblib")
    except:
        return None


@st.cache_data
def load_shap_artifacts():
    """Load pre-computed SHAP artifacts."""
    artifacts = {}
    
    # Load SHAP values
    shap_path = Path("artifacts/shap_values.npy")
    if shap_path.exists():
        artifacts['shap_values'] = np.load(shap_path)
    
    # Load feature importance
    fi_path = Path("artifacts/feature_importance.csv")
    if fi_path.exists():
        artifacts['feature_importance'] = pd.read_csv(fi_path)
    
    # Load SHAP plots
    for plot_name in ['shap_summary', 'shap_beeswarm']:
        plot_path = Path(f"artifacts/{plot_name}.png")
        if plot_path.exists():
            artifacts[plot_name] = plot_path
    
    return artifacts


@st.cache_data
def load_test_data():
    """Load test data."""
    try:
        test_data = joblib.load("artifacts/test_data.pkl")
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(X_test, np.ndarray):
            try:
                import json
                with open("artifacts/feature_names.json", 'r') as f:
                    feature_names = json.load(f)
                X_test = pd.DataFrame(X_test, columns=feature_names)
            except:
                # Fallback
                X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
                
        # Convert y_test to Series if needed
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, name='label')
            
        return X_test, y_test
    except:
        return None, None


def plot_feature_importance_interactive(df_importance, top_k=15):
    """Create interactive feature importance plot."""
    df_top = df_importance.head(top_k)
    
    fig = go.Figure(go.Bar(
        x=df_top['importance'],
        y=df_top['feature'],
        orientation='h',
        marker=dict(
            color=df_top['importance'],
            colorscale='Teal',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=df_top['importance'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {top_k} Most Important Features",
        xaxis_title="Feature Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig



# Apply custom design system
st.markdown(get_custom_css(), unsafe_allow_html=True)

def main():
    """Main explanation page."""
    
    show_header(
        "Model Explainability",
        "Understand how the model makes theft detection decisions",
        icon=fa_icon("lightbulb", 22, "#00C2A8")
    )
    
    # Load model
    model = load_model_cached()
    
    if model is None:
        st.error("■ MODEL NOT DETECTED - Please train a model first.")
        return
    
    # Create tabs
    tabs = st.tabs(["FEATURE IMPORTANCE", "GLOBAL SHAP", "LOCAL EXPLANATION", "MODEL INFO"])
    
    # Tab 1: Feature Importance
    with tabs[0]:
        st.markdown("""
        <h3 style="color: #00C2A8; font-size: 24px; font-weight: 700; letter-spacing: 0.5px;">
            FEATURE IMPORTANCE ANALYSIS
        </h3>
        """, unsafe_allow_html=True)
        
        artifacts = load_shap_artifacts()
        feature_importance = artifacts.get('feature_importance')
        
        if feature_importance is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Interactive plot
                top_k = st.slider("Number of features to display", 5, 30, 15)
                fig = plot_feature_importance_interactive(feature_importance, top_k)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <h4 style="color: #FFB020; font-size: 18px; font-weight: 700;">
                    TOP 10 FEATURES
                </h4>
                """, unsafe_allow_html=True)
                
                top_10 = feature_importance.head(10)
                
                for i, row in top_10.iterrows():
                    st.markdown(f"""
                    <div style="background: #262730; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                        <div style="color: #00C2A8; font-weight: 600; font-size: 14px;">
                            {i+1}. {row['feature']}
                        </div>
                        <div style="color: #AAAAAA; font-size: 12px; margin-top: 4px;">
                            Importance: {row['importance']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interpretation
            st.markdown("---")
            st.info("""
            **INTERPRETATION**: Feature importance shows which features the model relies on most for predictions. 
            Higher values indicate stronger influence on the model's decisions. The top features are the most 
            critical for detecting electricity theft patterns.
            """)
            
            # Download
            csv = feature_importance.to_csv(index=False)
            st.download_button(
                "Download Feature Importance",
                csv,
                "feature_importance.csv",
                "text/csv"
            )
        
        else:
            st.warning("Feature importance unavailable. Run training and evaluation first.")
    
    # Tab 2: Global SHAP
    with tabs[1]:
        st.markdown("""
        <h3 style="color: #7B68EE; font-size: 24px; font-weight: 700; letter-spacing: 0.5px;">
            GLOBAL SHAP ANALYSIS
        </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to predictions 
        across all samples. This provides a global view of model behavior.
        """)
        
        artifacts = load_shap_artifacts()
        
        # Summary plot
        col1, col2 = st.columns([3, 1])
        
        with col1:
            summary_path = artifacts.get('shap_summary')
            if summary_path:
                st.markdown("""
                <h4 style="color: #00C2A8; font-size: 18px; font-weight: 700;">
                    SHAP SUMMARY (BAR)
                </h4>
                """, unsafe_allow_html=True)
                image = Image.open(summary_path)
                st.image(image, use_container_width=True)
            else:
                st.warning("SHAP summary plot not available")
        
        with col2:
            st.markdown("""
            <h4 style="color: #FFB020; font-size: 16px; font-weight: 700;">
                HOW TO READ
            </h4>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background: #262730; padding: 15px; border-radius: 8px;">
                <p style="color: #FAFAFA; font-size: 13px; line-height: 1.6;">
                    <strong>Bar heights</strong> show average absolute SHAP value 
                    (mean impact on model output).
                    <br><br>
                    <strong>Longer bars</strong> = more important features for theft detection.
                    <br><br>
                    <strong>Color</strong> indicates feature value direction in beeswarm plot.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Beeswarm plot
        beeswarm_path = artifacts.get('shap_beeswarm')
        if beeswarm_path:
            st.markdown("""
            <h4 style="color: #00C851; font-size: 18px; font-weight: 700;">
                SHAP BEESWARM PLOT
            </h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                image = Image.open(beeswarm_path)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("""
                <h4 style="color: #FFB020; font-size: 16px; font-weight: 700;">
                    HOW TO READ
                </h4>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div style="background: #262730; padding: 15px; border-radius: 8px;">
                    <p style="color: #FAFAFA; font-size: 13px; line-height: 1.6;">
                        <strong>X-axis</strong>: SHAP value (impact on prediction)
                        <br><br>
                        <strong>Color</strong>: Feature value (red=high, blue=low)
                        <br><br>
                        <strong>Positive SHAP</strong> → increases theft probability
                        <br><br>
                        <strong>Negative SHAP</strong> → decreases theft probability
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interpretation guide
        with st.expander("SHAP Interpretation Guide", expanded=False):
            st.markdown("""
            ### Understanding SHAP Values
            
            **What are SHAP values?**
            - SHAP values explain individual predictions by computing feature contributions
            - Based on game theory (Shapley values) - fair attribution of prediction to features
            - Sum of all SHAP values = model output - base value
            
            **How to interpret:**
            
            1. **Positive SHAP value** (red):
               - Feature increases theft probability
               - Pushes prediction toward "theft" class
            
            2. **Negative SHAP value** (blue):
               - Feature decreases theft probability
               - Pushes prediction toward "honest" class
            
            3. **Magnitude**:
               - Larger absolute value = stronger influence
               - Small values = minimal impact
            
            **Example**:
            If `sudden_drop_count` has SHAP value +0.15 for a customer, it means this feature 
            increases the model's theft probability prediction by 0.15 (on log-odds scale).
            """)
    
    # Tab 3: Local Explanation
    with tabs[2]:
        st.markdown("""
        <h3 style="color: #FFB020; font-size: 24px; font-weight: 700; letter-spacing: 0.5px;">
            PER-CUSTOMER EXPLANATION
        </h3>
        """, unsafe_allow_html=True)
        
        X_test, y_test = load_test_data()
        
        if X_test is None:
            st.warning("Test data not available. Please run training first.")
        else:
            st.markdown("Select a customer to see detailed SHAP explanation:")
            
            # Customer selector
            customer_ids = X_test.index.tolist()
            selected_idx = st.selectbox(
                "Customer ID",
                range(min(100, len(customer_ids))),  # Limit to 100 for performance
                format_func=lambda i: f"{customer_ids[i][:16]}... (Label: {'THEFT' if y_test.iloc[i] == 1 else 'HONEST'})"
            )
            
            if st.button("Explain Prediction", type="primary"):
                with st.spinner("Generating explanation..."):
                    customer_id = customer_ids[selected_idx]
                    customer_data = X_test.iloc[selected_idx:selected_idx+1]
                    actual_label = y_test.iloc[selected_idx]
                    
                    # Predict
                    prob = model.predict_proba(customer_data)[0, 1]
                    pred_label = 1 if prob >= 0.5 else 0
                    
                    # SHAP
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(customer_data)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    # Display
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### Customer Info")
                        st.markdown(f"**ID**: `{customer_id[:16]}...`")
                        actual_html = "<span style='color:#FF4444; font-weight:700;'>THEFT</span>" if actual_label == 1 else "<span style='color:#00C851; font-weight:700;'>HONEST</span>"
                        pred_html = "<span style='color:#FF4444; font-weight:700;'>THEFT</span>" if pred_label == 1 else "<span style='color:#00C851; font-weight:700;'>HONEST</span>"
                        st.markdown(f"**Actual Label**: {actual_html}", unsafe_allow_html=True)
                        st.markdown(f"**Predicted**: {pred_html}", unsafe_allow_html=True)
                        st.markdown(f"**Probability**: {prob:.1%}")
                        
                        if pred_label == actual_label:
                            st.success("Correct")
                        else:
                            st.error("Incorrect")
                    
                    with col2:
                        st.markdown("#### Prediction Breakdown")
                        
                        # Base value
                        base_value = explainer.expected_value
                        if isinstance(base_value, list):
                            base_value = base_value[1]
                        
                        # Calculate contribution
                        from scipy.special import expit
                        
                        st.markdown(f"**Base probability**: {expit(base_value):.1%}")
                        st.markdown(f"**Final probability**: {prob:.1%}")
                        st.markdown(f"**Total SHAP impact**: {shap_values[0].sum():.4f}")
                    
                    # Top features
                    st.markdown("---")
                    st.markdown("""
                    <h4 style="color: #00C851; font-size: 18px; font-weight: 700;">
                        TOP CONTRIBUTING FEATURES
                    </h4>
                    """, unsafe_allow_html=True)
                    
                    # Get top features by absolute SHAP
                    feature_names = list(customer_data.columns)
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': customer_data.iloc[0].values,
                        'SHAP': shap_values[0],
                        'Abs_SHAP': np.abs(shap_values[0])
                    }).sort_values('Abs_SHAP', ascending=False)
                    
                    # Plot top 10
                    top_features = shap_df.head(10)
                    
                    fig = go.Figure(go.Bar(
                        x=top_features['SHAP'],
                        y=top_features['Feature'],
                        orientation='h',
                        marker=dict(
                            color=top_features['SHAP'],
                            colorscale='RdBu',
                            cmid=0,
                            showscale=True,
                            colorbar=dict(title="SHAP Value")
                        ),
                        text=top_features['SHAP'].round(4),
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Top 10 Feature Contributions",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        template="plotly_dark",
                        height=400,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature table
                    with st.expander("All Features", expanded=False):
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                    
                    # Waterfall (text-based)
                    st.markdown("---")
                    st.markdown(
                        f"<h4>{fa_icon('droplet', 16, '#4FACFE')} Contribution Waterfall</h4>",
                        unsafe_allow_html=True,
                    )
                    
                    st.info(f"""
                    Starting from base probability {expit(base_value):.1%}, the model applies feature contributions:
                    
                    **Top 3 Increasing Theft Probability:**
                    {chr(10).join([f"• {row['Feature']}: +{row['SHAP']:.4f}" for _, row in shap_df[shap_df['SHAP'] > 0].head(3).iterrows()])}
                    
                    **Top 3 Decreasing Theft Probability:**
                    {chr(10).join([f"• {row['Feature']}: {row['SHAP']:.4f}" for _, row in shap_df[shap_df['SHAP'] < 0].head(3).iterrows()])}
                    
                    Final prediction: {prob:.1%} theft probability
                    """)
    
    # Tab 4: Model Info
    with tabs[3]:
        st.markdown(
            f"<h3>{fa_icon('book', 18, '#4FACFE')} Model Information</h3>",
            unsafe_allow_html=True,
        )
        
        # Model architecture
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"<h4>{fa_icon('sitemap', 16, '#00C2A8')} Architecture</h4>",
                unsafe_allow_html=True,
            )
            st.markdown("""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <p style="color: #FAFAFA; line-height: 1.8;">
                    <strong>Algorithm</strong>: XGBoost Classifier<br>
                    <strong>Objective</strong>: Binary Logistic<br>
                    <strong>Optimization</strong>: Optuna TPE Sampler<br>
                    <strong>Composite Score</strong>: 0.6×Recall + 0.25×Precision + 0.15×F1<br>
                    <strong>Preprocessing</strong>: SMOTE+ENN<br>
                    <strong>Scaling</strong>: MinMaxScaler [0,1]
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(
                f"<h4>{fa_icon('bullseye', 16, '#FFB020')} Performance Targets</h4>",
                unsafe_allow_html=True,
            )
            st.markdown("""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <p style="color: #FAFAFA; line-height: 1.8;">
                    <strong>Minimum Recall</strong>: ≥85%<br>
                    <strong>Minimum Precision</strong>: ≥75%<br>
                    <strong>Minimum F1</strong>: ≥75%<br>
                    <strong>Focus</strong>: Recall-optimized (catch theft)<br>
                    <strong>Trade-off</strong>: Accept false positives to minimize false negatives
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature engineering
        st.markdown(
            f"<h4>{fa_icon('gears', 16, '#7B68EE')} Feature Engineering</h4>",
            unsafe_allow_html=True,
        )
        
        with st.expander("20+ Engineered Features", expanded=True):
            st.markdown("""
            **Statistical Features (8)**:
            - Mean, Median, Std Dev, Coefficient of Variation
            - Min, Max, Range, Skewness
            
            **Trend Features (3)**:
            - Linear slope over full series
            - Linear slope over last 30 days
            - Linear slope over last 90 days
            
            **Temporal Features (2)**:
            - Weekday vs weekend consumption ratio
            - Peak day ratio (top 10% consumption frequency)
            
            **Anomaly Features (3)**:
            - Zero consumption day count
            - Sudden drop count (>50% day-over-day)
            - Volatility index (std/mean)
            
            **Other Features (2)**:
            - Autocorrelation at lag 1
            - Missing sequence count (>3 consecutive days)
            """)
        
        # Explainability methods
        st.markdown("---")
        st.markdown(
            f"<h4>{fa_icon('brain', 16, '#00C2A8')} Explainability Methods</h4>",
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <h4 style="color: #00C2A8; margin-top: 0;">SHAP (TreeExplainer)</h4>
                <p style="color: #FAFAFA; line-height: 1.6;">
                    • Game-theoretic feature attribution<br>
                    • Exact Shapley values for tree models<br>
                    • Global and local explanations<br>
                    • Mathematically consistent
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #262730; padding: 20px; border-radius: 10px;">
                <h4 style="color: #FFB020; margin-top: 0;">Feature Importance</h4>
                <p style="color: #FAFAFA; line-height: 1.6;">
                    • XGBoost native importance<br>
                    • Based on split gain<br>
                    • Aggregated across all trees<br>
                    • Fast computation
                </p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
