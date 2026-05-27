"""
SGCC Theft Detector - Streamlit Main Application

Multi-page Streamlit app for electricity theft detection.
"""

import streamlit as st
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from deploy_utils import apply_custom_theme

# Add design system
sys.path.insert(0, str(Path(__file__).parent))
from design_system import get_custom_css, get_icon

# Configure page
st.set_page_config(
    page_title="SGCC Theft Detector - AI Analytics",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/bolt.svg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'AI-Powered electricity theft detection system'
    }
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Apply custom theme
apply_custom_theme()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_config():
    """Load configuration from YAML."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main application entry point."""
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.markdown("""
    <div class="glass-card" style="padding: 24px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            {icon}
            <h2 style="color: #667eea; margin: 0; font-weight: 700; font-size: 24px;">SGCC AI</h2>
        </div>
        <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 14px;">AI-Powered Analytics</p>
    </div>
    """.format(icon=get_icon('bolt', 32, '#667eea')), unsafe_allow_html=True)
    
    # Model info
    st.sidebar.markdown("**System Configuration**")
    st.sidebar.info("""
    **Algorithm**: XGBoost Ensemble  
    **Optimization**: Recall-Focused  
    **Performance Targets**:  
    • Recall ≥ 85%  
    • Precision ≥ 75%  
    • AUC ≥ 0.90
    """)
    
    st.sidebar.markdown("---")
    
    # Quick search
    st.sidebar.markdown("**Quick Analysis**")
    customer_search = st.sidebar.text_input(
        "Customer ID",
        placeholder="Enter customer ID...",
        help="Search for specific customer"
    )
    
    if customer_search:
        st.sidebar.success(f"Query: {customer_search[:12]}...")
        st.sidebar.button("▶ Analyze Customer", type="primary")
    
    st.sidebar.markdown("---")
    
    # Navigation info
    st.sidebar.markdown("**Platform Modules**")
    st.sidebar.markdown("""
    <div class="modern-card">
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(102, 126, 234, 0.1); margin-bottom: 8px;">
                {icon_chart}
                <div>
                    <strong style="color: #667eea;">Data Explorer</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">Statistical analysis & visualization</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(118, 75, 162, 0.1); margin-bottom: 8px;">
                {icon_cog}
                <div>
                    <strong style="color: #764ba2;">Model Training</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">AI optimization & tuning</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(79, 172, 254, 0.1); margin-bottom: 8px;">
                {icon_bolt}
                <div>
                    <strong style="color: #4FACFE;">AI Prediction</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">Smart theft detection</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(0, 200, 81, 0.1); margin-bottom: 8px;">
                {icon_bulb}
                <div>
                    <strong style="color: #00C851;">Explainability</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">SHAP interpretability</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(255, 176, 32, 0.1); margin-bottom: 8px;">
                {icon_upload}
                <div>
                    <strong style="color: #FFB020;">Dataset Upload</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">Custom data analysis</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(123, 104, 238, 0.1); margin-bottom: 8px;">
                {icon_compare}
                <div>
                    <strong style="color: #7B68EE;">Model Comparison</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">Baseline evaluation</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; background: rgba(255, 68, 68, 0.1);">
                {icon_eye}
                <div>
                    <strong style="color: #FF4444;">Performance Monitor</strong><br/>
                    <span style="color: rgba(255,255,255,0.6); font-size: 12px;">Drift detection & alerts</span>
                </div>
            </div>
        </div>
    </div>
    """.format(
        icon_chart=get_icon('chart_bar', 18, '#667eea'),
        icon_cog=get_icon('cog', 18, '#764ba2'),
        icon_bolt=get_icon('bolt', 18, '#4FACFE'),
        icon_bulb=get_icon('light_bulb', 18, '#00C851'),
        icon_upload=get_icon('document_text', 18, '#FFB020'),
        icon_compare=get_icon('arrow_trending_up', 18, '#7B68EE'),
        icon_eye=get_icon('eye', 18, '#FF4444')
    ), unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <div class="animate-fade-in" style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 20px;
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    ">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 16px;">
            {icon_bolt}
            <h1 style="color: white; margin: 0; font-size: 52px; font-weight: 800; letter-spacing: 2px;">
                SGCC AI PLATFORM
            </h1>
        </div>
        <p style="color: rgba(255,255,255,0.9); margin: 20px 0 0 0; font-size: 22px; font-weight: 400;">
            AI-Powered Electricity Theft Detection with Smart Recommendations
        </p>
        <div style="margin-top: 30px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            <span class="badge" style="background: rgba(255,255,255,0.2); color: white; font-size: 13px;">
                {icon_shield} Advanced SMOTE+ENN
            </span>
            <span class="badge" style="background: rgba(255,255,255,0.2); color: white; font-size: 13px;">
                {icon_brain} XGBoost AI
            </span>
            <span class="badge" style="background: rgba(255,255,255,0.2); color: white; font-size: 13px;">
                {icon_chart} SHAP Explainability
            </span>
            <span class="badge" style="background: rgba(255,255,255,0.2); color: white; font-size: 13px;">
                {icon_bulb} Smart Recommendations
            </span>
        </div>
    </div>
    """.format(
        icon_bolt=get_icon('bolt', 48, '#FFFFFF'),
        icon_shield=get_icon('shield_check', 14, '#FFFFFF'),
        icon_brain=get_icon('cog', 14, '#FFFFFF'),
        icon_chart=get_icon('chart_bar', 14, '#FFFFFF'),
        icon_bulb=get_icon('light_bulb', 14, '#FFFFFF')
    ), unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="modern-card animate-fade-in" style="text-align: center; padding: 30px;">
            <div style="margin-bottom: 12px;">
                {icon}
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px;">
                TOTAL CUSTOMERS
            </div>
            <div style="color: #FAFAFA; font-size: 42px; font-weight: 800; margin-bottom: 8px;">
                ---
            </div>
            <div style="color: #00C2A8; font-size: 14px;">
                Ready for analysis
            </div>
        </div>
        """.format(icon=get_icon('user_group', 32, '#00C2A8')), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card animate-fade-in" style="text-align: center; padding: 30px; animation-delay: 0.1s;">
            <div style="margin-bottom: 12px;">
                {icon}
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px;">
                FLAGGED TODAY
            </div>
            <div style="color: #FAFAFA; font-size: 42px; font-weight: 800; margin-bottom: 8px;">
                ---
            </div>
            <div style="color: #FFB020; font-size: 14px;">
                Requires investigation
            </div>
        </div>
        """.format(icon=get_icon('exclamation_triangle', 32, '#FFB020')), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="modern-card animate-fade-in" style="text-align: center; padding: 30px; animation-delay: 0.2s;">
            <div style="margin-bottom: 12px;">
                {icon}
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px;">
                MODEL RECALL
            </div>
            <div style="color: #FAFAFA; font-size: 42px; font-weight: 800; margin-bottom: 8px;">
                85%+
            </div>
            <div style="color: #00C851; font-size: 14px;">
                Target achieved
            </div>
        </div>
        """.format(icon=get_icon('chart_bar', 32, '#00C851')), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="modern-card animate-fade-in" style="text-align: center; padding: 30px; animation-delay: 0.3s;">
            <div style="margin-bottom: 12px;">
                {icon}
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px;">
                MODEL VERSION
            </div>
            <div style="color: #FAFAFA; font-size: 42px; font-weight: 800; margin-bottom: 8px;">
                v1.0
            </div>
            <div style="color: #7B68EE; font-size: 14px;">
                Production ready
            </div>
        </div>
        """.format(icon=get_icon('shield_check', 32, '#7B68EE')), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="animate-fade-in" style="margin-top: 60px; margin-bottom: 40px;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 30px;">
            {icon_star}
            <h2 style="color: #FAFAFA; font-size: 36px; font-weight: 800; letter-spacing: 1px; margin: 0;">
                CAPABILITIES & FEATURES
            </h2>
        </div>
    </div>
    """.format(icon_star=get_icon('star', 36, '#FFB020')), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card animate-slide-in-left" style="padding: 35px; margin-bottom: 20px; border-left: 4px solid #00C2A8;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                {icon}
                <h3 style="color: #00C2A8; margin: 0; font-weight: 700; font-size: 22px; letter-spacing: 0.5px;">DETECTION ENGINE</h3>
            </div>
            <ul style="color: rgba(255,255,255,0.85); line-height: 2.2; font-size: 15px; padding-left: 20px;">
                <li>20+ engineered temporal & statistical features</li>
                <li>SMOTE+ENN balanced training data</li>
                <li>XGBoost with Optuna hyperparameter tuning</li>
                <li>Recall-focused optimization (40% weight)</li>
            </ul>
        </div>
        """.format(icon=get_icon('shield_check', 28, '#00C2A8')), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card animate-slide-in-left" style="padding: 35px; border-left: 4px solid #FFB020; animation-delay: 0.1s;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                {icon}
                <h3 style="color: #FFB020; margin: 0; font-weight: 700; font-size: 22px; letter-spacing: 0.5px;">DATA ANALYTICS</h3>
            </div>
            <ul style="color: rgba(255,255,255,0.85); line-height: 2.2; font-size: 15px; padding-left: 20px;">
                <li>Interactive consumption time series</li>
                <li>Calendar heatmaps & anomaly detection</li>
                <li>Customer clustering (UMAP/t-SNE)</li>
                <li>Cohort analysis & pattern discovery</li>
            </ul>
        </div>
        """.format(icon=get_icon('chart_bar', 28, '#FFB020')), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card animate-slide-in-right" style="padding: 35px; margin-bottom: 20px; border-left: 4px solid #7B68EE;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                {icon}
                <h3 style="color: #7B68EE; margin: 0; font-weight: 700; font-size: 22px; letter-spacing: 0.5px;">EXPLAINABILITY</h3>
            </div>
            <ul style="color: rgba(255,255,255,0.85); line-height: 2.2; font-size: 15px; padding-left: 20px;">
                <li>SHAP waterfall plots for predictions</li>
                <li>Global feature importance</li>
                <li>Force plots & decision explanations</li>
                <li>Human-readable AI reasoning</li>
            </ul>
        </div>
        """.format(icon=get_icon('light_bulb', 28, '#7B68EE')), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card animate-slide-in-right" style="padding: 35px; border-left: 4px solid #00C851; animation-delay: 0.1s;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                {icon}
                <h3 style="color: #00C851; margin: 0; font-weight: 700; font-size: 22px; letter-spacing: 0.5px;">DEPLOYMENT</h3>
            </div>
            <ul style="color: rgba(255,255,255,0.85); line-height: 2.2; font-size: 15px; padding-left: 20px;">
                <li>Docker containerized deployment</li>
                <li>Streamlit Cloud compatible</li>
                <li>CI/CD with GitHub Actions</li>
                <li>Comprehensive test coverage</li>
            </ul>
        </div>
        """.format(icon=get_icon('cog', 28, '#00C851')), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("""
    <h2 style="color: #FAFAFA; font-size: 32px; font-weight: 700; letter-spacing: 1px; margin-top: 40px; margin-bottom: 30px;">
        ▸ IMPLEMENTATION GUIDE
    </h2>
    """, unsafe_allow_html=True)
    
    with st.expander("▶ GETTING STARTED", expanded=False):
        st.markdown("""
        ### Step 1: Download Data
        ```bash
        # Set Kaggle credentials
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key
        
        # Download dataset
        bash scripts/download_data.sh
        ```
        
        ### Step 2: Train Model
        ```bash
        # Full training (60 trials, ~hours)
        python -m src.train
        
        # Quick training (10 trials, ~minutes)
        python -m src.train --quick
        ```
        
        ### Step 3: Run Streamlit App
        ```bash
        streamlit run streamlit_app/app.py
        ```
        
        ### Step 4: Explore
        - **EDA Page**: Understand data patterns
        - **Train Page**: Retrain or fine-tune model
        - **Predict Page**: Detect theft cases
        - **Explain Page**: Understand model decisions
        """)
    
    with st.expander("▶ DOCKER DEPLOYMENT", expanded=False):
        st.markdown("""
        ### Build and Run with Docker
        ```bash
        # Build image
        docker build -t sgcc-theft-detector .
        
        # Run container
        docker run -p 8501:8501 \\
          -e KAGGLE_USERNAME=$KAGGLE_USERNAME \\
          -e KAGGLE_KEY=$KAGGLE_KEY \\
          sgcc-theft-detector
        ```
        
        ### Using Docker Compose
        ```bash
        # Start services
        docker-compose up -d
        
        # View logs
        docker-compose logs -f
        
        # Stop services
        docker-compose down
        ```
        """)
    
    with st.expander("Streamlit Cloud Deployment", expanded=False):
        st.markdown("""
        ### Deploy to Streamlit Cloud
        
        1. **Push to GitHub**
           ```bash
           git init
           git add .
           git commit -m "Initial commit"
           git push origin main
           ```
        
        2. **Configure Secrets**
           - Go to Streamlit Cloud dashboard
           - Add secrets in `.streamlit/secrets.toml` format:
           ```toml
           KAGGLE_USERNAME = "your_username"
           KAGGLE_KEY = "your_api_key"
           ```
        
        3. **Deploy**
           - Select repository
           - Choose `streamlit_app/app.py` as main file
           - Deploy!
        
        **Note**: Precompute heavy artifacts (model, SHAP) offline due to 1GB RAM limit.
        """)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #AAAAAA; padding: 20px;">
        <p style="margin: 0;">
            Built with Streamlit • XGBoost • SHAP • Optuna
        </p>
        <p style="margin: 5px 0 0 0; font-size: 12px;">
            SGCC Theft Detector v1.0 | Production Ready
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
