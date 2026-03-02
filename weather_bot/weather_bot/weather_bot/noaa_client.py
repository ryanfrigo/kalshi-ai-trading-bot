# ============================================================
# noaa_client.py  -  Free NOAA weather API fetcher
# Uses https://api.weather.gov (no key needed)
# Returns today's forecast high temp for a city
# ============================================================
import requests
import json
from datetime import datetime, date
from typing import Optional, Dict
from config import City

NOAA_AGENT = "ClawdbotWeatherTrader/1.0 (paper-trade)"

# Cache gridpoints so we don't re-fetch every loop
_GRID_CACHE: Dict[str, Dict] = {}


def _get_gridpoint(city: City) -> Optional[Dict]:
    """Resolve lat/lon -> NOAA grid office + gridX/gridY (cached)."""
    key = f"{city.noaa_lat},{city.noaa_lon}"
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    try:
        r = requests.get(
            f"https://api.weather.gov/points/{city.noaa_lat},{city.noaa_lon}",
            headers={"User-Agent": NOAA_AGENT},
            timeout=10
        )
        r.raise_for_status()
        props = r.json()["properties"]
        grid = {
            "office": props["gridId"],
            "gridX": props["gridX"],
            "gridY": props["gridY"],
            "forecast_url": props["forecast"],
            "hourly_url": props["forecastHourly"],
        }
        _GRID_CACHE[key] = grid
        return grid
    except Exception as e:
        print(f"[NOAA] gridpoint error for {city.name}: {e}")
        return None


def get_forecast_high(city: City) -> Optional[float]:
    """
    Returns the forecast HIGH temperature (F) for today.
    Uses NOAA's /gridpoints/{office}/{x},{y}/forecast endpoint.
    This is what Kalshi's temperature markets reference.
    """
    grid = _get_gridpoint(city)
    if not grid:
        return None
    try:
        r = requests.get(
            grid["forecast_url"],
            headers={"User-Agent": NOAA_AGENT},
            timeout=10
        )
        r.raise_for_status()
        periods = r.json()["properties"]["periods"]
        today_str = date.today().strftime("%Y-%m-%d")
        # Find daytime periods for today
        for p in periods:
            start = p.get("startTime", "")
            is_day = p.get("isDaytime", False)
            if today_str in start and is_day:
                return float(p["temperature"])
        # Fallback: first period
        return float(periods[0]["temperature"]) if periods else None
    except Exception as e:
        print(f"[NOAA] forecast error for {city.name}: {e}")
        return None


def get_forecast_summary(city: City) -> Dict:
    """
    Returns a dict with high temp + short forecast text for the bot's
    trade reason log.
    """
    grid = _get_gridpoint(city)
    if not grid:
        return {"high": None, "summary": "NOAA unavailable"}
    try:
        r = requests.get(
            grid["forecast_url"],
            headers={"User-Agent": NOAA_AGENT},
            timeout=10
        )
        r.raise_for_status()
        periods = r.json()["properties"]["periods"]
        today_str = date.today().strftime("%Y-%m-%d")
        for p in periods:
            if today_str in p.get("startTime", "") and p.get("isDaytime", False):
                return {
                    "high": float(p["temperature"]),
                    "summary": p.get("shortForecast", ""),
                    "wind": p.get("windSpeed", ""),
                    "period_name": p.get("name", "Today"),
                }
        if periods:
            p = periods[0]
            return {
                "high": float(p["temperature"]),
                "summary": p.get("shortForecast", ""),
                "wind": p.get("windSpeed", ""),
                "period_name": p.get("name", ""),
            }
    except Exception as e:
        print(f"[NOAA] summary error for {city.name}: {e}")
    return {"high": None, "summary": "error"}


def temp_matches_bucket(forecast_high: float, bucket_str: str) -> bool:
    """
    Given a Kalshi market title like '66 to 67' or '65 or below' or '74 or above',
    check if the NOAA forecast high falls in that bucket.
    Returns True if NOAA forecast matches the bucket (good for YES bet).
    """
    s = bucket_str.lower().strip()
    try:
        if "or below" in s:
            threshold = float(s.replace("or below", "").replace("°", "").strip())
            return forecast_high <= threshold
        elif "or above" in s:
            threshold = float(s.replace("or above", "").replace("°", "").strip())
            return forecast_high >= threshold
        elif " to " in s:
            parts = s.replace("°", "").split(" to ")
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
            return lo <= forecast_high <= hi
    except Exception:
        pass
    return False