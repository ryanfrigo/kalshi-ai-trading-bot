#!/usr/bin/env python3
# ============================================================
# main.py  -  Clawdbot Weather Trader  (Paper Mode)
#
# Run:  cd weather_bot && python main.py
#
# What it does every 2 minutes:
#   1. Pull NOAA forecasts for 6 cities
#   2. Scan Kalshi weather markets
#   3. Find undervalued YES/NO prices <= 15%
#      that match the NOAA forecast bucket
#   4. Paper-enter up to 5 new positions
#   5. Check all open positions for 3x exit or trailing stop
#   6. Print dashboard to terminal
# ============================================================
import sys
import os
import time
import traceback
from datetime import datetime, timezone

# Make sure we can import sibling modules when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from kalshi_client import KalshiClient
from noaa_client import get_forecast_summary
from strategy import find_entries, check_exit
from paper_trader import PaperTrader
from dashboard import display


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def fetch_forecasts() -> dict:
    """Pull NOAA forecast summary for all 6 cities. Returns {city_name: {high, summary}}."""
    forecasts = {}
    for city in cfg.CITIES:
        try:
            f = get_forecast_summary(city)
            forecasts[city.name] = f
            log(f"NOAA {city.name}: high={f.get('high')}F  {f.get('summary', '')}")
        except Exception as e:
            log(f"NOAA ERROR {city.name}: {e}")
            forecasts[city.name] = {"high": None, "summary": "error"}
    return forecasts


def fetch_weather_markets(kalshi: KalshiClient) -> list:
    """
    Fetch open markets matching our city prefixes.
    We filter by series tickers that contain 'KXHIGH' (Kalshi temperature series).
    Also try KXRAIN, KXSNOW for each city.
    """
    all_markets = []
    series_prefixes = ["KXHIGH", "KXRAIN", "KXSNOW"]
    for prefix in series_prefixes:
        try:
            # Get markets with this series ticker prefix
            mkts = kalshi.get_markets(series_ticker=prefix, status="open", limit=200)
            all_markets.extend(mkts)
        except Exception as e:
            log(f"Market fetch error ({prefix}): {e}")
    # Deduplicate by ticker
    seen = set()
    unique = []
    for m in all_markets:
        t = m.get("ticker", "")
        if t not in seen:
            seen.add(t)
            unique.append(m)
    log(f"Found {len(unique)} open weather markets")
    return unique


def run_once(kalshi: KalshiClient, trader: PaperTrader):
    """Single scan cycle."""
    log("=== SCAN START ===")

    # ------------------------------------------------------------------
    # Safeguard: daily loss limit
    # ------------------------------------------------------------------
    if trader.daily_pnl < -cfg.DAILY_LOSS_LIMIT_USD:
        log(f"SAFEGUARD: Daily loss limit hit (${trader.daily_pnl:.2f}). "
            f"Skipping new entries. Managing exits only.")
        _manage_exits(kalshi, trader)
        return

    # ------------------------------------------------------------------
    # 1. Fetch NOAA forecasts
    # ------------------------------------------------------------------
    forecasts = fetch_forecasts()
    noaa_ok = any(v.get("high") is not None for v in forecasts.values())
    if not noaa_ok:
        log("SAFEGUARD: All NOAA forecasts failed. Skipping new entries.")
        _manage_exits(kalshi, trader)
        return

    # ------------------------------------------------------------------
    # 2. Fetch open weather markets from Kalshi
    # ------------------------------------------------------------------
    markets = fetch_weather_markets(kalshi)
    if not markets:
        log("No markets found. Skipping.")
        return

    # ------------------------------------------------------------------
    # 3. Find entry candidates
    # ------------------------------------------------------------------
    candidates = find_entries(
        markets=markets,
        city_forecast=forecasts,
        existing_tickers=trader.open_tickers,
        kalshi=kalshi,
        max_new=cfg.MAX_TRADES_PER_RUN,
    )
    log(f"Found {len(candidates)} entry candidates")

    # ------------------------------------------------------------------
    # 4. Enter positions
    # ------------------------------------------------------------------
    for c in candidates:
        entry = trader.enter(c)
        log(f"PAPER ENTER: {c.ticker} {c.side.upper()} @ ${c.entry_price:.2f} "
            f"| {c.city_name} | {c.reason[:80]}")

    # ------------------------------------------------------------------
    # 5. Manage exits on all open positions
    # ------------------------------------------------------------------
    _manage_exits(kalshi, trader)

    # ------------------------------------------------------------------
    # 6. Print dashboard
    # ------------------------------------------------------------------
    display(clear=False)
    log(f"=== SCAN COMPLETE | Open: {len(trader.open_tickers)} | "
        f"PnL: ${trader.running_pnl:+.4f} ===")


def _manage_exits(kalshi: KalshiClient, trader: PaperTrader):
    """Check all open positions for 3x or trailing stop exit."""
    for ticker in list(trader.open_tickers):
        pos = trader.positions.get(ticker)
        if not pos:
            continue
        result = check_exit(pos, kalshi)
        if result is None:
            continue
        # Update peak regardless
        new_peak = result.get("new_peak", pos["peak_price"])
        trader.update_peak(ticker, new_peak)

        if result.get("exit_price") is not None and not result.get("hold"):
            exit_log = trader.exit(ticker, result["exit_price"], result["reason"])
            pnl = exit_log.get("pnl", 0) if exit_log else 0
            log(f"PAPER EXIT : {ticker} @ ${result['exit_price']:.2f} "
                f"PnL=${pnl:+.4f} | {result['reason'][:80]}")


def main():
    print("\n" + "="*60)
    print("  CLAWDBOT Weather Trader")
    print(f"  Paper Mode: {'ON' if cfg.PAPER_TRADE else 'OFF - LIVE TRADING'}")
    print(f"  Cities:     {', '.join(c.name for c in cfg.CITIES)}")
    print(f"  Entry:      <= {cfg.ENTRY_THRESHOLD*100:.0f}%  |  Exit: {cfg.EXIT_MULTIPLIER}x")
    print(f"  Max/Run:    {cfg.MAX_TRADES_PER_RUN}  |  Interval: {cfg.SCAN_INTERVAL_SEC}s")
    print("="*60 + "\n")

    kalshi = KalshiClient()
    trader = PaperTrader()

    log("Clawdbot started. Press Ctrl+C to stop.")
    while True:
        try:
            run_once(kalshi, trader)
        except KeyboardInterrupt:
            log("Stopping Clawdbot.")
            break
        except Exception as e:
            log(f"ERROR in run_once: {e}")
            traceback.print_exc()
        log(f"Sleeping {cfg.SCAN_INTERVAL_SEC}s...")
        time.sleep(cfg.SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()