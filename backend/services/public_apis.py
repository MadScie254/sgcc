from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

import requests


def _request_json(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> Dict[str, Any] | List[Dict[str, Any]]:
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=32)
def get_country_context(country_code: str = "DZ") -> Dict[str, Any]:
    try:
        payload = _request_json(f"https://restcountries.com/v3.1/alpha/{country_code}")
        country = payload[0]
        currencies = country.get("currencies", {})
        return {
            "country_code": country_code.upper(),
            "name": country.get("name", {}).get("common", country_code.upper()),
            "capital": country.get("capital", ["N/A"])[0],
            "region": country.get("region", "N/A"),
            "population": int(country.get("population", 0) or 0),
            "currency": next(iter(currencies.keys()), "N/A"),
            "flag": country.get("flags", {}).get("png"),
        }
    except Exception:
        return {
            "country_code": country_code.upper(),
            "name": country_code.upper(),
            "capital": "N/A",
            "region": "N/A",
            "population": 0,
            "currency": "N/A",
            "flag": None,
        }


@lru_cache(maxsize=32)
def get_public_holidays(country_code: str = "DZ", year: int | None = None) -> List[Dict[str, Any]]:
    target_year = year or datetime.utcnow().year
    try:
        payload = _request_json(f"https://date.nager.at/api/v3/PublicHolidays/{target_year}/{country_code.upper()}")
        return [dict(item) for item in payload]
    except Exception:
        return []


@lru_cache(maxsize=32)
def get_weather_context(latitude: float = 36.7538, longitude: float = 3.0588) -> Dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    try:
        payload = _request_json("https://api.open-meteo.com/v1/forecast", params=params)
        current = payload.get("current", {}) if isinstance(payload, dict) else {}
        return {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m": current.get("temperature_2m"),
            "weather_code": current.get("weather_code"),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "time": current.get("time"),
        }
    except Exception:
        return {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m": None,
            "weather_code": None,
            "wind_speed_10m": None,
            "time": None,
        }