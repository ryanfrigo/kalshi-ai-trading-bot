# ============================================================
# strategy.py  -  Entry/exit decision engine
# Entry:  price <= 15%  AND  NOAA forecast matches bucket
# Exit:   current_price >= entry_price * 3x  (take profit)
# Trend:  skip if last 3 price ticks trending against us
# ============================================================
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timezone
import config as cfg
from noaa_client import temp_matches_bucket, get_forecast_summary
from kalshi_client import KalshiClient


@dataclass
class Candidate:
    """A market that passed all entry filters."""
    ticker: str
    side: str           # 'yes' or 'no'
    entry_price: float  # 0.0-1.0
    contracts: int      # how many to buy
    city_name: str
    market_title: str
    noaa_high: float
    noaa_summary: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Simple in-memory price trend cache: ticker -> list of last 3 mid prices
_price_history: Dict[str, List[float]] = {}


def _update_trend(ticker: str, price: float):
    hist = _price_history.setdefault(ticker, [])
    hist.append(price)
    if len(hist) > 3:
        hist.pop(0)


def _trend_ok(ticker: str, side: str) -> bool:
    """
    Returns True if price momentum is NOT strongly moving against our side.
    For YES:  we want price NOT falling consistently (last 3 ticks all down)
    For NO:   same (NO price = 1 - YES price, so inverse)
    """
    if not cfg.SCAN_INTERVAL_SEC:  # trend detection gate
        return True
    hist = _price_history.get(ticker, [])
    if len(hist) < 3:
        return True  # not enough data, allow
    if side == "yes":
        # block if price fell 3 ticks in a row
        return not (hist[0] > hist[1] > hist[2])
    else:
        return not (hist[0] < hist[1] < hist[2])


def find_entries(
    markets: List[Dict],
    city_forecast: Dict,   # {city_name: {high, summary}}
    existing_tickers: set,
    kalshi: KalshiClient,
    max_new: int = cfg.MAX_TRADES_PER_RUN,
) -> List[Candidate]:
    """
    Scan open markets, return up to max_new Candidate entries.
    Filters:
      1. Price (YES or NO ask) <= ENTRY_THRESHOLD (0.15)
      2. NOAA forecast matches the bucket for that side
      3. Not already in an open position
      4. Not trending against us
      5. Volume >= MIN_VOLUME
      6. Spread <= MAX_SPREAD_CENTS
    """
    candidates = []
    for mkt in markets:
        ticker = mkt.get("ticker", "")
        if ticker in existing_tickers:
            continue
        if mkt.get("status") != "open":
            continue
        # Volume filter
        if mkt.get("volume", 0) < cfg.MIN_VOLUME:
            continue
        # Identify which city this market belongs to
        city_name = None
        for city in cfg.CITIES:
            if city.name.lower() in mkt.get("title", "").lower() or \
               city.name.lower() in ticker.lower():
                city_name = city.name
                break
        if not city_name:
            continue
        forecast = city_forecast.get(city_name, {})
        noaa_high = forecast.get("high")
        noaa_summary = forecast.get("summary", "")
        if noaa_high is None:
            continue
        # Get orderbook
        try:
            ob = kalshi.get_orderbook(ticker, depth=1)
        except Exception:
            continue
        prices = KalshiClient.parse_best_prices(ob)
        yes_ask = prices.get("yes_ask")
        no_ask  = prices.get("no_ask")
        # Spread filter
        if yes_ask and no_ask:
            spread = abs(yes_ask - (1 - no_ask))
            if spread > cfg.MAX_SPREAD_CENTS / 100:
                continue
        # Parse bucket from market title (e.g. "66 to 67")
        title = mkt.get("title", "")
        bucket = _extract_bucket(title)
        # --- Check YES side ---
        if yes_ask and yes_ask <= cfg.ENTRY_THRESHOLD:
            forecast_favors_yes = bucket and temp_matches_bucket(noaa_high, bucket)
            _update_trend(ticker + "_yes", yes_ask)
            if forecast_favors_yes and _trend_ok(ticker + "_yes", "yes"):
                contracts = max(1, int(cfg.MAX_POSITION_USD / 1.0))  # $1 per contract
                reason = (f"YES ask {yes_ask:.2f} <= {cfg.ENTRY_THRESHOLD} threshold. "
                          f"NOAA {city_name} forecast high {noaa_high}F matches bucket '{bucket}'. "
                          f"Forecast: {noaa_summary}.")
                candidates.append(Candidate(
                    ticker=ticker, side="yes",
                    entry_price=yes_ask,
                    contracts=contracts,
                    city_name=city_name,
                    market_title=title,
                    noaa_high=noaa_high,
                    noaa_summary=noaa_summary,
                    reason=reason,
                ))
        # --- Check NO side ---
        elif no_ask and no_ask <= cfg.ENTRY_THRESHOLD:
            forecast_favors_no = bucket and not temp_matches_bucket(noaa_high, bucket)
            _update_trend(ticker + "_no", no_ask)
            if forecast_favors_no and _trend_ok(ticker + "_no", "no"):
                contracts = max(1, int(cfg.MAX_POSITION_USD / 1.0))
                reason = (f"NO ask {no_ask:.2f} <= {cfg.ENTRY_THRESHOLD} threshold. "
                          f"NOAA {city_name} forecast high {noaa_high}F does NOT match bucket '{bucket}'. "
                          f"Forecast: {noaa_summary}.")
                candidates.append(Candidate(
                    ticker=ticker, side="no",
                    entry_price=no_ask,
                    contracts=contracts,
                    city_name=city_name,
                    market_title=title,
                    noaa_high=noaa_high,
                    noaa_summary=noaa_summary,
                    reason=reason,
                ))
        if len(candidates) >= max_new:
            break
    return candidates[:max_new]


def check_exit(
    position: Dict,
    kalshi: KalshiClient,
) -> Optional[Dict]:
    """
    Given an open paper position, decide if we should exit now.
    Returns exit dict with reason, or None to hold.

    Exit rules:
      1. Current price >= entry_price * EXIT_MULTIPLIER (3x)  -> sell
      2. Trailing stop: if peak was hit and price fell back > TRAILING_STOP_CENTS -> sell
      3. Market closed/settled -> forced exit at settlement
    """
    ticker      = position["ticker"]
    side        = position["side"]
    entry_price = position["entry_price"]
    peak_price  = position.get("peak_price", entry_price)
    target      = entry_price * cfg.EXIT_MULTIPLIER

    try:
        ob = kalshi.get_orderbook(ticker, depth=1)
    except Exception:
        return None
    prices = KalshiClient.parse_best_prices(ob)

    # Current value of our side
    if side == "yes":
        current_price = prices.get("yes_bid") or prices.get("yes_ask") or entry_price
    else:
        current_price = prices.get("no_bid") or prices.get("no_ask") or entry_price

    if current_price is None:
        return None

    # Update peak
    new_peak = max(peak_price, current_price)

    # 3x target hit -> exit
    if current_price >= target:
        return {
            "exit_price": current_price,
            "new_peak": new_peak,
            "reason": (f"3x target reached. Entry {entry_price:.2f} -> "
                       f"Current {current_price:.2f} (target was {target:.2f}).")
        }

    # Trailing stop: only armed after peak >= target
    if cfg.TRAILING_STOP_ENABLED and peak_price >= target:
        stop = peak_price - cfg.TRAILING_STOP_CENTS
        if current_price <= stop:
            return {
                "exit_price": current_price,
                "new_peak": new_peak,
                "reason": (f"Trailing stop triggered. Peak {peak_price:.2f}, "
                           f"stop {stop:.2f}, current {current_price:.2f}.")
            }

    # Update peak in position (caller must persist)
    return {"hold": True, "new_peak": new_peak, "current_price": current_price}


def _extract_bucket(title: str) -> Optional[str]:
    """Extract temperature bucket string from a Kalshi market title.
    e.g. 'Highest temperature in NYC today? 66° to 67°' -> '66 to 67'
    """
    import re
    title = title.lower()
    # Match patterns: '66 to 67', '65 or below', '74 or above'
    m = re.search(r'(\d+)\s*°?\s*to\s*(\d+)', title)
    if m:
        return f"{m.group(1)} to {m.group(2)}"
    m = re.search(r'(\d+)\s*°?\s*or\s*below', title)
    if m:
        return f"{m.group(1)} or below"
    m = re.search(r'(\d+)\s*°?\s*or\s*above', title)
    if m:
        return f"{m.group(1)} or above"
    return None