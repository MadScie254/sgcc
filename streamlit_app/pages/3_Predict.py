"""
SGCC Theft Detector - Enhanced Predict Page

AI-Powered Theft Prediction with Smart Recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from modeling import load_model
from features import load_scaler
from recommendation_engine import RecommendationEngine

# Add design system
sys.path.insert(0, str(Path(__file__).parent.parent))
from design_system import get_custom_css, get_icon, fa_icon, LOTTIE_ANIMATIONS

st.set_page_config(
    page_title="AI Prediction - SGCC Theft Detector",
    page_icon="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/svgs/solid/bullseye.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize recommendation engine
if 'rec_engine' not in st.session_state:
    st.session_state.rec_engine = RecommendationEngine()


@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler."""
    try:
        model = load_model("models/xgb_best.joblib")
        scaler = load_scaler("artifacts/scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_test_data():
    """Load test data if available."""
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
                # Fallback if feature names missing
                X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
                
        # Convert y_test to Series if needed
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, name='label')
            
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None


def create_risk_gauge(probability: float) -> go.Figure:
    """Create animated risk gauge chart."""
    if probability >= 0.7:
        color = "#FF4444"
        risk = "CRITICAL"
    elif probability >= 0.5:
        color = "#FFB020"
        risk = "HIGH"
    elif probability >= 0.3:
        color = "#4FACFE"
        risk = "MEDIUM"
    else:
        color = "#00C851"
        risk = "LOW"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk} RISK</b>", 'font': {'size': 24, 'color': color}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#FAFAFA', 'family': 'Inter'}},
        delta={'reference': 50, 'increasing': {'color': "#FF4444"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#FAFAFA", 'tickfont': {'size': 14}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(26, 26, 46, 0.5)",
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 200, 81, 0.15)'},
                {'range': [30, 50], 'color': 'rgba(79, 172, 254, 0.15)'},
                {'range': [50, 70], 'color': 'rgba(255, 176, 32, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(255, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 4},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#FAFAFA")
    )
    
    return fig


def display_theft_recommendations(recommendations: dict, probability: float):
    """Display AI-powered theft recommendations with professional design."""
    rec = recommendations
    
    # Risk Overview Card
    st.markdown(f"""
    <div class="modern-card animate-fade-in" style="background: linear-gradient(135deg, rgba(255, 68, 68, 0.1) 0%, rgba(238, 9, 121, 0.1) 100%); border-left: 4px solid #FF4444; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            {get_icon('shield_exclamation', 32, '#FF4444')}
            <div>
                <h2 style="margin: 0; color: #FF4444; font-size: 24px; font-weight: 700;">THEFT DETECTED</h2>
                <p style="margin: 4px 0 0 0; color: rgba(255, 255, 255, 0.8); font-size: 14px;">Risk Level: {rec['risk_level']} | Confidence: {rec['confidence']} | Priority: {rec['priority']}</p>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px;">
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value" style="color: #FF4444;">{rec['risk_score']}/100</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Estimated Loss</div>
                <div class="metric-value" style="color: #FFB020;">${rec['estimated_loss_usd']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Response Time</div>
                <div class="metric-value" style="color: #4FACFE;">{rec['recommended_timeline']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for detailed information
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"{get_icon('bolt', 20, '#FFB020')} Immediate Actions",
        f"{get_icon('clipboard_document_check', 20, '#4FACFE')} Investigation",
        f"{get_icon('chart_bar', 20, '#7B68EE')} Pattern Analysis",
        f"{get_icon('calendar', 20, '#00C851')} Follow-Up Plan",
        f"{get_icon('shield_check', 20, '#FF4444')} Prevention"
    ])
    
    with tab1:
        st.markdown("### Prioritized Actions")
        for action in rec['immediate_actions']:
            priority_colors = {1: '#FF4444', 2: '#FFB020', 3: '#4FACFE'}
            color = priority_colors.get(action['priority'], '#7B68EE')
            
            st.markdown(f"""
            <div class="modern-card animate-slide-in-left" style="border-left: 4px solid {color}; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="badge" style="background: {color};">P{action['priority']}</span>
                        <h4 style="margin: 0; color: #FAFAFA; font-size: 18px;">{action['action']}</h4>
                    </div>
                    <span style="color: rgba(255, 255, 255, 0.6); font-size: 13px;">{action['timeline']}</span>
                </div>
                <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 8px;">{action['description']}</p>
                <div style="display: flex; align-items: center; gap: 8px; color: #4FACFE; font-size: 13px;">
                    {get_icon('user_group', 16, '#4FACFE')}
                    <span>{action['responsible']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Investigation Checklist")
        for item in rec['investigation_checklist']:
            icon_name = 'exclamation_triangle' if item['critical'] else 'information_circle'
            icon_color = '#FF4444' if item['critical'] else '#4FACFE'
            
            st.markdown(f"""
            <div class="modern-card" style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    {get_icon(icon_name, 24, icon_color)}
                    <h4 style="margin: 0; color: #FAFAFA;">{item['item']}</h4>
                    {f'<span class="badge badge-danger">CRITICAL</span>' if item['critical'] else ''}
                </div>
                <ul style="color: rgba(255, 255, 255, 0.8); margin: 0; padding-left: 20px;">
            """, unsafe_allow_html=True)
            
            for check in item['checks']:
                st.markdown(f"<li style='margin: 6px 0;'>{check}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Consumption Pattern Analysis")
        patterns = rec['pattern_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; margin-bottom: 16px;">
                <h4 style="color: #FAFAFA; margin-bottom: 16px;">{get_icon('arrow_trending_down', 20, '#FF4444')} Anomaly Indicators</h4>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: rgba(255, 255, 255, 0.8);">Sudden Drops</span>
                        <span style="color: #FAFAFA; font-weight: 600;">{patterns['sudden_drops']['count']} events</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(patterns['sudden_drops']['count'] * 20, 100)}%; background: #FF4444;"></div>
                    </div>
                    <span style="font-size: 12px; color: rgba(255, 255, 255, 0.6);">{patterns['sudden_drops']['indicator']}</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: rgba(255, 255, 255, 0.8);">Volatility</span>
                        <span style="color: #FAFAFA; font-weight: 600;">{patterns['volatility']['value']}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(patterns['volatility']['value'] * 100, 100)}%; background: #FFB020;"></div>
                    </div>
                    <span style="font-size: 12px; color: rgba(255, 255, 255, 0.6);">{patterns['volatility']['indicator']}</span>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: rgba(255, 255, 255, 0.8);">Zero Days</span>
                        <span style="color: #FAFAFA; font-weight: 600;">{patterns['zero_consumption_days']['count']} days ({patterns['zero_consumption_days']['percentage']}%)</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {patterns['zero_consumption_days']['percentage']}%; background: #4FACFE;"></div>
                    </div>
                    <span style="font-size: 12px; color: rgba(255, 255, 255, 0.6);">{patterns['zero_consumption_days']['indicator']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; margin-bottom: 16px;">
                <h4 style="color: #FAFAFA; margin-bottom: 16px;">{get_icon('chart_bar', 20, '#00C851')} Behavioral Patterns</h4>
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        {get_icon('arrow_trending_' + ('down' if patterns['trend']['direction'] == 'declining' else 'up'), 20, '#FF4444' if patterns['trend']['direction'] == 'declining' else '#00C851')}
                        <div>
                            <div style="color: rgba(255, 255, 255, 0.7); font-size: 13px;">Consumption Trend</div>
                            <div style="color: #FAFAFA; font-weight: 600; text-transform: capitalize;">{patterns['trend']['direction']}</div>
                        </div>
                    </div>
                    <span style="font-size: 12px; color: rgba(255, 255, 255, 0.6);">{patterns['trend']['indicator']}</span>
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        {get_icon('cog', 20, '#7B68EE')}
                        <div>
                            <div style="color: rgba(255, 255, 255, 0.7); font-size: 13px;">Pattern Consistency</div>
                            <div style="color: #FAFAFA; font-weight: 600; text-transform: capitalize;">{patterns['consistency']['level']}</div>
                        </div>
                    </div>
                    <span style="font-size: 12px; color: rgba(255, 255, 255, 0.6);">{patterns['consistency']['indicator']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Timeline & Follow-Up")
        for period, tasks in rec['follow_up_plan'].items():
            period_display = period.replace('_', ' ').title()
            
            st.markdown(f"""
            <div class="modern-card" style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    {get_icon('calendar', 20, '#4FACFE')}
                    <h4 style="margin: 0; color: #4FACFE;">{period_display}</h4>
                </div>
                <ul style="color: rgba(255, 255, 255, 0.8); margin: 0; padding-left: 20px;">
            """, unsafe_allow_html=True)
            
            for task in tasks:
                st.markdown(f"<li style='margin: 8px 0;'>{task}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### Preventive Measures")
        for measure in rec['preventive_measures']:
            effectiveness_colors = {
                'Very High': '#00C851',
                'High': '#4FACFE',
                'Medium': '#FFB020',
                'Low': '#FF4444'
            }
            color = effectiveness_colors.get(measure['effectiveness'], '#7B68EE')
            
            st.markdown(f"""
            <div class="modern-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 12px;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 8px 0; color: #FAFAFA;">{measure['measure']}</h4>
                        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">{measure['description']}</p>
                    </div>
                    <span class="badge" style="background: {color}; margin-left: 16px;">{measure['effectiveness']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px;">Cost</div>
                        <div style="color: #FAFAFA; font-weight: 600;">{measure['cost']}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px;">Benefit</div>
                        <div style="color: #4FACFE; font-size: 13px;">{measure['benefit']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_honest_recommendations(recommendations: dict, probability: float):
    """Display engagement strategies for honest customers."""
    rec = recommendations
    
    # Status Card
    st.markdown(f"""
    <div class="modern-card animate-fade-in" style="background: linear-gradient(135deg, rgba(0, 200, 81, 0.1) 0%, rgba(17, 153, 142, 0.1) 100%); border-left: 4px solid #00C851; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            {get_icon('shield_check', 32, '#00C851')}
            <div>
                <h2 style="margin: 0; color: #00C851; font-size: 24px; font-weight: 700;">HONEST CUSTOMER</h2>
                <p style="margin: 4px 0 0 0; color: rgba(255, 255, 255, 0.8); font-size: 14px;">Confidence: {rec['confidence']}% | Profile: {rec['usage_profile']} | Efficiency: Grade {rec['efficiency_analysis']['grade']}</p>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px;">
            <div class="metric-card">
                <div class="metric-label">Efficiency Score</div>
                <div class="metric-value" style="color: #00C851;">{rec['efficiency_score']}/100</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Potential Savings</div>
                <div class="metric-value" style="color: #FFB020;">${rec['potential_savings_usd']}/mo</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Usage</div>
                <div class="metric-value" style="color: #4FACFE;">{rec['efficiency_analysis']['average_daily_usage']:.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        f"{get_icon('light_bulb', 20, '#FFB020')} Optimization Tips",
        f"{get_icon('star', 20, '#7B68EE')} Engagement",
        f"{get_icon('currency_dollar', 20, '#00C851')} Rewards"
    ])
    
    with tab1:
        st.markdown("### Personalized Recommendations")
        for tip in rec['optimization_tips']:
            st.markdown(f"""
            <div class="modern-card animate-slide-in-left" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <div>
                        <span class="badge badge-info">{tip['category']}</span>
                        <h4 style="margin: 8px 0; color: #FAFAFA;">{tip['tip']}</h4>
                    </div>
                    <span class="badge badge-success">Save {tip['potential_savings']}</span>
                </div>
                <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 8px;">{tip['description']}</p>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge" style="background: rgba(79, 172, 254, 0.2); color: #4FACFE; font-size: 11px;">Difficulty: {tip['difficulty']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Engagement Strategies")
        for strategy in rec['engagement_strategies']:
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; margin-bottom: 16px;">
                <h4 style="color: #FAFAFA; margin-bottom: 12px;">{strategy['strategy']}</h4>
                <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 12px;">{strategy['description']}</p>
                <div style="margin-bottom: 12px;">
                    <div style="color: #4FACFE; font-size: 13px; font-weight: 600; margin-bottom: 6px;">Benefits:</div>
                    <ul style="margin: 0; padding-left: 20px; color: rgba(255, 255, 255, 0.8);">
            """, unsafe_allow_html=True)
            
            for benefit in strategy['benefits']:
                st.markdown(f"<li style='margin: 4px 0;'>{benefit}</li>", unsafe_allow_html=True)
            
            st.markdown(f"""
                    </ul>
                </div>
                <div style="padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <span style="color: #00C851; font-weight: 600;">Action: </span>
                    <span style="color: rgba(255, 255, 255, 0.8);">{strategy['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Loyalty & Rewards Programs")
        for program in rec['loyalty_recommendations']:
            st.markdown(f"""
            <div class="modern-card" style="margin-bottom: 16px; border-left: 4px solid #FFB020;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    {get_icon('star', 20, '#FFB020')}
                    <h4 style="margin: 0; color: #FAFAFA;">{program['program']}</h4>
                </div>
                <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 12px;">{program['description']}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px;">Benefit</div>
                        <div style="color: #00C851; font-weight: 600;">{program['benefit']}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px;">Eligibility</div>
                        <div style="color: #4FACFE; font-size: 13px;">{program['eligibility']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="animate-fade-in" style="text-align: center; margin-bottom: 40px;">
        <h1 style="
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        ">AI-Powered Prediction</h1>
        <p style="font-size: 18px; color: rgba(255, 255, 255, 0.7);">
            Intelligent theft detection with smart recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error(f"{get_icon('exclamation_triangle', 20, '#FF4444')} Model not found. Please train a model first.")
        return
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Prediction Mode Selection
    st.markdown("### Select Prediction Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Mode",
            ["Quick Test (Sample Data)", "Manual Input"],
            label_visibility="collapsed"
        )
    
    if mode == "Quick Test (Sample Data)" and X_test is not None:
        st.markdown("### Select Customer from Test Set")
        
        n_samples = len(X_test)
        sample_idx = st.slider(
            "Customer Index",
            0,
            n_samples - 1,
            0,
            help=f"Browse through {n_samples} test customers"
        )
        
        if st.button(f"{get_icon('bolt', 16, '#FFFFFF')} Predict", type="primary"):
            with st.spinner("Analyzing customer data..."):
                # Get customer data
                X_customer = X_test.iloc[sample_idx:sample_idx+1]
                
                # Predict
                prob = model.predict_proba(X_customer)[0][1]
                prediction = 1 if prob >= 0.5 else 0
                actual = y_test.iloc[sample_idx] if y_test is not None else None
                
                # Display results
                st.markdown("---")
                
                col_gauge, col_info = st.columns([1, 1])
                
                with col_gauge:
                    fig = create_risk_gauge(prob)
                    st.plotly_chart(fig, use_column_width=True)
                
                with col_info:
                    # Status card
                    status_color = "#FF4444" if prediction == 1 else "#00C851"
                    status_text = "THEFT DETECTED" if prediction == 1 else "HONEST CUSTOMER"
                    
                    st.markdown(f"""
                    <div class="modern-card" style="border-left: 4px solid {status_color};">
                        <h3 style="color: {status_color}; margin-bottom: 16px;">{status_text}</h3>
                        <div style="margin-bottom: 12px;">
                            <span style="color: rgba(255, 255, 255, 0.7);">Theft Probability:</span>
                            <span style="color: #FAFAFA; font-weight: 700; font-size: 24px; margin-left: 8px;">{prob:.1%}</span>
                        </div>
                        {f'''<div style="margin-bottom: 12px;">
                            <span style="color: rgba(255, 255, 255, 0.7);">Actual Label:</span>
                            <span style="color: {'#FF4444' if actual == 1 else '#00C851'}; font-weight: 600; margin-left: 8px;">{'Theft' if actual == 1 else 'Honest'}</span>
                        </div>''' if actual is not None else ''}
                        <div>
                            <span style="color: rgba(255, 255, 255, 0.7);">Prediction:</span>
                            <span style="color: {status_color}; font-weight: 600; margin-left: 8px;">{'Theft' if prediction == 1 else 'Honest'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Get AI recommendations
                features_dict = X_customer.iloc[0].to_dict()
                customer_id = f"TEST_{sample_idx:05d}"
                
                if prediction == 1:
                    recommendations = st.session_state.rec_engine.generate_theft_recommendations(
                        prob, features_dict, customer_id
                    )
                    display_theft_recommendations(recommendations, prob)
                else:
                    recommendations = st.session_state.rec_engine.generate_honest_recommendations(
                        prob, features_dict, customer_id
                    )
                    display_honest_recommendations(recommendations, prob)
    
    else:
        st.info(f"{get_icon('information_circle', 16, '#4FACFE')} Manual input mode coming soon. Use test data for now.")


if __name__ == "__main__":
    main()
