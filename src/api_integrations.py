"""
SGCC Theft Detector - Public API Integrations

No-authentication API integrations for weather, geocoding, holidays, and more.
All APIs include caching, rate limiting, and error handling.
"""

import requests
import streamlit as st
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


# Rate limiting decorator
def rate_limit(min_interval: float = 1.0):
    """
    Decorator to enforce minimum time interval between API calls.
    
    Args:
        min_interval: Minimum seconds between calls
    """
    def decorator(func):
        last_called = [0.0]
        
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator


# Weather API (Open-Meteo)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str
) -> Optional[Dict]:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dictionary with weather data or None if failed
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        return None


# Geocoding API (Nominatim/OpenStreetMap)

@st.cache_data(ttl=604800)  # Cache for 1 week
@rate_limit(min_interval=1.0)  # Nominatim requires 1 req/second
def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """
    Convert address to GPS coordinates using Nominatim.
    
    Args:
        address: Address string
        
    Returns:
        Tuple of (latitude, longitude) or None if failed
    """
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'SGCC-Theft-Detector/1.0'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None
    except Exception as e:
        logger.warning(f"Geocoding API failed: {e}")
        return None


# Holiday Calendar API (Nager.Date)

@st.cache_data(ttl=604800)  # Cache for 1 week
def get_public_holidays(country_code: str = 'DZ', year: int = 2024) -> Optional[List[Dict]]:
    """
    Get public holidays from Nager.Date API.
    
    Args:
        country_code: ISO 3166-1 alpha-2 country code (DZ = Algeria)
        year: Year to fetch holidays for
        
    Returns:
        List of holiday dictionaries or None if failed
    """
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Holiday API failed: {e}")
        return None


# Exchange Rates API

@st.cache_data(ttl=3600)  # Cache for 1 hour
def convert_currency(
    amount: float,
    from_currency: str = 'DZD',
    to_currency: str = 'USD'
) -> Optional[float]:
    """
    Convert currency using ExchangeRate-API.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code
        
    Returns:
        Converted amount or None if failed
    """
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if to_currency in data['rates']:
            rate = data['rates'][to_currency]
            return amount * rate
        return None
    except Exception as e:
        logger.warning(f"Currency API failed: {e}")
        return None


# IP Geolocation API

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_user_location() -> Optional[Dict]:
    """
    Get user's location from IP address using ipapi.co.
    
    Returns:
        Dictionary with location data or None if failed
    """
    url = "https://ipapi.co/json/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"IP geolocation API failed: {e}")
        return None


# Random User Generator (for demos)

@st.cache_data(ttl=3600)
def generate_customer_profile(seed: Optional[str] = None) -> Optional[Dict]:
    """
    Generate realistic customer profile using RandomUser.me.
    
    Args:
        seed: Optional seed for reproducibility
        
    Returns:
        Dictionary with customer data or None if failed
    """
    url = "https://randomuser.me/api/"
    
    params = {}
    if seed:
        params['seed'] = seed
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            user = data['results'][0]
            return {
                'name': f"{user['name']['first']} {user['name']['last']}",
                'email': user['email'],
                'phone': user['phone'],
                'address': f"{user['location']['street']['number']} {user['location']['street']['name']}, {user['location']['city']}",
                'picture': user['picture']['large'],
                'country': user['location']['country']
            }
        return None
    except Exception as e:
        logger.warning(f"Random user API failed: {e}")
        return None


# Country Metadata API

@st.cache_data(ttl=604800)  # Cache for 1 week
def get_country_info(country_code: str = 'DZ') -> Optional[Dict]:
    """
    Get country metadata from REST Countries API.
    
    Args:
        country_code: ISO 3166-1 alpha-2 country code
        
    Returns:
        Dictionary with country data or None if failed
    """
    url = f"https://restcountries.com/v3.1/alpha/{country_code}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            country = data[0]
            return {
                'name': country['name']['common'],
                'capital': country.get('capital', ['N/A'])[0],
                'population': country.get('population', 0),
                'region': country.get('region', 'N/A'),
                'currency': list(country.get('currencies', {}).keys())[0] if country.get('currencies') else 'N/A',
                'flag': country['flags']['png']
            }
        return None
    except Exception as e:
        logger.warning(f"Country API failed: {e}")
        return None


# NASA APOD (Astronomy Picture of the Day)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_nasa_apod() -> Optional[Dict]:
    """
    Get NASA Astronomy Picture of the Day.
    
    Returns:
        Dictionary with APOD data or None if failed
    """
    url = "https://api.nasa.gov/planetary/apod"
    
    params = {
        'api_key': 'DEMO_KEY'  # Demo key allows limited requests
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"NASA APOD API failed: {e}")
        return None


# Numbers API (fun facts)

@st.cache_data(ttl=3600)
def get_number_fact(number: Optional[int] = None) -> Optional[str]:
    """
    Get interesting fact about a number.
    
    Args:
        number: Number to get fact about (random if None)
        
    Returns:
        Fact string or None if failed
    """
    if number is None:
        url = "http://numbersapi.com/random/trivia"
    else:
        url = f"http://numbersapi.com/{number}/trivia"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Numbers API failed: {e}")
        return None


# Helper functions for Streamlit integration

def display_weather_widget(latitude: float, longitude: float):
    """Display weather widget in Streamlit sidebar."""
    try:
        # Get weather for today
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        weather = fetch_weather_data(latitude, longitude, today, tomorrow)
        
        if weather and 'daily' in weather:
            st.sidebar.markdown("### Weather")
            temps = weather['daily'].get('temperature_2m_max', [])
            if temps:
                st.sidebar.metric("Temperature", f"{temps[0]}°C")
                
            precip = weather['daily'].get('precipitation_sum', [])
            if precip:
                st.sidebar.metric("Precipitation", f"{precip[0]} mm")
    except Exception as e:
        logger.debug(f"Weather widget failed: {e}")


def display_location_widget():
    """Display user location widget in Streamlit sidebar."""
    try:
        location = get_user_location()
        if location:
            st.sidebar.markdown("### Your Location")
            st.sidebar.text(f"{location.get('city', 'Unknown')}, {location.get('country_name', 'Unknown')}")
    except Exception as e:
        logger.debug(f"Location widget failed: {e}")


def display_fun_fact_widget():
    """Display fun fact widget in Streamlit sidebar."""
    try:
        fact = get_number_fact()
        if fact:
            st.sidebar.markdown("### Did You Know?")
            st.sidebar.info(fact)
    except Exception as e:
        logger.debug(f"Fun fact widget failed: {e}")
