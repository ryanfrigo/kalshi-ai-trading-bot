# ============================================================
# config.py  -  Clawdbot Weather Trader (Paper Mode)
# ============================================================
import os
from dataclasses import dataclass, field
from typing import List

# --- Kalshi REST base -------------------------------------------------
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# --- Auth (loaded from .env) ------------------------------------------
KALSHI_API_KEY_ID  = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi.pem")

# --- Paper-trade flag -------------------------------------------------
PAPER_TRADE = True          # set False ONLY when you want live execution

# --- Strategy knobs ---------------------------------------------------
ENTRY_THRESHOLD  = 0.15    # buy YES or NO if price <= this (cents/100)
EXIT_MULTIPLIER  = 3.0     # exit when current_price >= entry * 3x
MAX_POSITION_USD = 2.00    # max dollars per single position
MAX_TRADES_PER_RUN = 5     # new entries allowed per 2-min scan
SCAN_INTERVAL_SEC  = 120   # 2 minutes

# --- Safeguards -------------------------------------------------------
DAILY_LOSS_LIMIT_USD = 20.00   # stop opening new trades if daily PnL < -this
MIN_DAYS_TO_EXPIRY   = 0       # allow same-day contracts
MAX_DAYS_TO_EXPIRY   = 7       # skip contracts expiring > 7 days out
MIN_VOLUME           = 10      # skip markets with < 10 total volume
MAX_SPREAD_CENTS     = 20      # skip if best_ask - best_bid > 0.20

# --- Trailing stop (optional) -----------------------------------------
TRAILING_STOP_ENABLED = True
TRAILING_STOP_CENTS   = 0.05   # trail 5 cents below peak after 3x hit

# --- Cities + NOAA station IDs ----------------------------------------
# NOAA gridpoint URLs: https://api.weather.gov/points/{lat},{lon}
@dataclass
class City:
    name: str
    noaa_lat: float
    noaa_lon: float
    kalshi_series_prefix: str   # prefix to filter tickers e.g. "KXHIGHNYC"

CITIES: List[City] = [
    City("NYC",     40.7128, -74.0060, "KXHIGH"),
    City("Chicago", 41.8781, -87.6298, "KXHIGH"),
    City("Seattle", 47.6062, -122.3321, "KXHIGH"),
    City("Atlanta", 33.7490, -84.3880, "KXHIGH"),
    City("Dallas",  32.7767, -96.7970, "KXHIGH"),
    City("Miami",   25.7617, -80.1918, "KXHIGH"),
]

# --- Logging ----------------------------------------------------------
LOG_FILE       = "weather_bot/trade_log.json"
PNL_FILE       = "weather_bot/pnl_history.json"
POSITIONS_FILE = "weather_bot/positions.json"