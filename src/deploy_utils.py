"""
SGCC Theft Detector - Deployment Utilities

Helper functions for Streamlit app: caching, API calls, formatters.
"""

import pandas as pd
import numpy as np
import requests
import streamlit as st
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def get_weather(lat: float, lon: float) -> Optional[Dict]:
    """
    Get weather forecast from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Weather data dictionary or None if failed
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Weather API failed: {str(e)}")
        return None


@st.cache_data(ttl=86400)
def geocode_location(query: str) -> Optional[Tuple[float, float]]:
    """
    Geocode location using Nominatim.
    
    Args:
        query: Location query string
    
    Returns:
        Tuple of (lat, lon) or None if failed
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "SGCC-Theft-Detector/1.0"
        }
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
        return None
    except Exception as e:
        logger.warning(f"Geocoding failed: {str(e)}")
        return None


def get_placeholder_image(width: int = 200, height: int = 200, seed: Optional[int] = None) -> str:
    """
    Get placeholder image URL from Lorem Picsum.
    
    Args:
        width: Image width
        height: Image height
        seed: Random seed for consistent image
    
    Returns:
        Image URL
    """
    if seed is not None:
        return f"https://picsum.photos/seed/{seed}/{width}/{height}"
    return f"https://picsum.photos/{width}/{height}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"


def format_metric(value: float, decimals: int = 2) -> str:
    """
    Format metric value.
    
    Args:
        value: Metric value
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"


def get_risk_color(probability: float) -> str:
    """
    Get color based on risk probability.
    
    Args:
        probability: Theft probability (0-1)
    
    Returns:
        Color hex code
    """
    if probability < 0.3:
        return "#00C851"  # Green (low risk)
    elif probability < 0.6:
        return "#FFB020"  # Amber (medium risk)
    else:
        return "#FF4444"  # Red (high risk)


def get_risk_label(probability: float) -> str:
    """
    Get risk label based on probability.
    
    Args:
        probability: Theft probability (0-1)
    
    Returns:
        Risk label string
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def create_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    color: str = "#00C2A8"
) -> str:
    """
    Create HTML for metric card.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
        color: Accent color
    
    Returns:
        HTML string
    """
    delta_html = f'<div style="color: {color}; font-size: 14px; margin-top: 5px;">{delta}</div>' if delta else ''
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, #262730 0%, #1a1c24 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
        <div style="color: #AAAAAA; font-size: 12px; font-weight: 500; letter-spacing: 1px; margin-bottom: 8px;">
            {title.upper()}
        </div>
        <div style="color: #FAFAFA; font-size: 32px; font-weight: 700;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    return html


def create_customer_card(
    customer_id: str,
    risk_score: float,
    risk_label: str,
    image_url: str
) -> str:
    """
    Create HTML for customer portrait card.
    
    Args:
        customer_id: Customer ID
        risk_score: Risk probability (0-1)
        risk_label: Risk label
        image_url: Customer image URL
    
    Returns:
        HTML string
    """
    risk_color = get_risk_color(risk_score)
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, #262730 0%, #1a1c24 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        text-align: center;
    ">
        <img src="{image_url}" style="
            width: 120px;
            height: 120px;
            border-radius: 60px;
            margin-bottom: 15px;
            border: 3px solid {risk_color};
        "/>
        <div style="color: #FAFAFA; font-size: 14px; font-weight: 600; margin-bottom: 5px;">
            ID: {customer_id[:12]}...
        </div>
        <div style="
            color: {risk_color};
            font-size: 24px;
            font-weight: 700;
            margin: 10px 0;
        ">
            {risk_score*100:.1f}%
        </div>
        <div style="
            color: {risk_color};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1px;
        ">
            {risk_label.upper()}
        </div>
    </div>
    """
    return html


def apply_custom_theme():
    """Apply custom dark theme CSS to Streamlit app."""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #00C2A8;
        --warning-color: #FFB020;
        --background-color: #0E1117;
        --secondary-bg: #262730;
        --text-color: #FAFAFA;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #00C2A8 0%, #00A890 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #00A890 0%, #008878 100%);
        box-shadow: 0 4px 12px rgba(0, 194, 168, 0.3);
    }
    
    /* Card styling */
    .element-container {
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--secondary-bg);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-bg);
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
    }
    </style>
    """, unsafe_allow_html=True)


def show_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None):
    """
    Show app header with title and subtitle.
    
    Args:
        title: Page title
        subtitle: Optional subtitle
        icon: Optional HTML icon string
    """
    icon_html = f'<span style="margin-right: 15px;">{icon}</span>' if icon else ''
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #262730 0%, #1a1c24 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #00C2A8;
    ">
        <h1 style="color: #FAFAFA; margin: 0; font-size: 36px; font-weight: 700; display: flex; align-items: center;">
            {icon_html}{title}
        </h1>
        {f'<p style="color: #AAAAAA; margin: 10px 0 0 0; font-size: 16px; padding-left: {60 if icon else 0}px;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Test utilities
    print("Testing deployment utilities...")
    
    # Test formatting
    print(f"Percentage: {format_percentage(0.8542)}")
    print(f"Metric: {format_metric(0.8542)}")
    
    # Test risk functions
    for prob in [0.2, 0.5, 0.8]:
        print(f"Prob {prob}: {get_risk_label(prob)} ({get_risk_color(prob)})")
    
    # Test placeholder image
    print(f"Image URL: {get_placeholder_image(200, 200, 42)}")
    
    print("\nUtilities test complete!")
