# ============================================================
# paper_trader.py  -  Paper trade state machine
# Tracks positions, logs entries/exits, computes running PnL
# All state persisted to JSON files (no DB needed)
# ============================================================
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import config as cfg
from strategy import Candidate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class PaperTrader:
    """
    Manages paper positions + trade log + running PnL.

    Position dict schema:
    {
      ticker:       str
      side:         'yes' | 'no'
      entry_price:  float   (0.0-1.0)
      contracts:    int
      cost_usd:     float   (entry_price * contracts)
      peak_price:   float
      city_name:    str
      market_title: str
      noaa_high:    float
      reason_entry: str
      entry_time:   str (ISO)
      status:       'open'
    }

    Trade log entry schema:
    {
      type:          'entry' | 'exit'
      ticker:        str
      side:          str
      price:         float
      contracts:     int
      usd:           float
      pnl:           float   (only on exit)
      city_name:     str
      market_title:  str
      noaa_high:     float
      reason:        str
      timestamp:     str
    }
    """

    def __init__(self):
        self.positions: Dict[str, Dict] = _load_json(cfg.POSITIONS_FILE, {})
        self.trade_log: List[Dict] = _load_json(cfg.LOG_FILE, [])
        self.pnl_history: List[Dict] = _load_json(cfg.PNL_FILE, [])
        self._running_pnl: float = sum(
            t.get("pnl", 0) for t in self.trade_log if t.get("type") == "exit"
        )

    # ---------------------------------------------------------------
    @property
    def open_tickers(self) -> set:
        return set(self.positions.keys())

    @property
    def running_pnl(self) -> float:
        return self._running_pnl

    @property
    def daily_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return sum(
            t.get("pnl", 0)
            for t in self.trade_log
            if t.get("type") == "exit" and t.get("timestamp", "").startswith(today)
        )

    # ---------------------------------------------------------------
    def enter(self, candidate: Candidate) -> Dict:
        """
        Record a paper entry. Returns the trade log entry.
        """
        cost = candidate.entry_price * candidate.contracts
        pos = {
            "ticker":       candidate.ticker,
            "side":         candidate.side,
            "entry_price":  candidate.entry_price,
            "contracts":    candidate.contracts,
            "cost_usd":     cost,
            "peak_price":   candidate.entry_price,
            "city_name":    candidate.city_name,
            "market_title": candidate.market_title,
            "noaa_high":    candidate.noaa_high,
            "reason_entry": candidate.reason,
            "entry_time":   _now(),
            "status":       "open",
        }
        self.positions[candidate.ticker] = pos

        entry_log = {
            "type":         "entry",
            "ticker":       candidate.ticker,
            "side":         candidate.side,
            "price":        candidate.entry_price,
            "contracts":    candidate.contracts,
            "usd":          cost,
            "city_name":    candidate.city_name,
            "market_title": candidate.market_title,
            "noaa_high":    candidate.noaa_high,
            "reason":       candidate.reason,
            "timestamp":    _now(),
        }
        self.trade_log.append(entry_log)
        self._persist()
        return entry_log

    def exit(self, ticker: str, exit_price: float, reason: str) -> Optional[Dict]:
        """
        Record a paper exit. Returns the trade log entry with PnL.
        """
        pos = self.positions.pop(ticker, None)
        if not pos:
            return None

        cost_usd   = pos["cost_usd"]
        contracts  = pos["contracts"]
        entry_price = pos["entry_price"]
        # PnL = (exit - entry) * contracts
        pnl = (exit_price - entry_price) * contracts
        self._running_pnl += pnl

        exit_log = {
            "type":         "exit",
            "ticker":       ticker,
            "side":         pos["side"],
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "contracts":    contracts,
            "usd":          exit_price * contracts,
            "pnl":          round(pnl, 4),
            "pnl_pct":      round((pnl / cost_usd) * 100, 1) if cost_usd else 0,
            "city_name":    pos["city_name"],
            "market_title": pos["market_title"],
            "noaa_high":    pos["noaa_high"],
            "reason":       reason,
            "entry_time":   pos["entry_time"],
            "exit_time":    _now(),
            "timestamp":    _now(),
        }
        self.trade_log.append(exit_log)

        # Append to PnL history for graph
        self.pnl_history.append({
            "timestamp": _now(),
            "pnl": round(self._running_pnl, 4),
            "trade_pnl": round(pnl, 4),
        })
        self._persist()
        return exit_log

    def update_peak(self, ticker: str, peak_price: float):
        """Update the peak price tracked for trailing stop."""
        if ticker in self.positions:
            self.positions[ticker]["peak_price"] = peak_price
            self._persist()

    # ---------------------------------------------------------------
    def _persist(self):
        _save_json(cfg.POSITIONS_FILE, self.positions)
        _save_json(cfg.LOG_FILE, self.trade_log)
        _save_json(cfg.PNL_FILE, self.pnl_history)

    def summary(self) -> Dict:
        return {
            "open_positions": len(self.positions),
            "total_trades": len([t for t in self.trade_log if t["type"] == "exit"]),
            "running_pnl": round(self._running_pnl, 4),
            "daily_pnl": round(self.daily_pnl, 4),
            "open_cost": round(sum(p["cost_usd"] for p in self.positions.values()), 4),
        }