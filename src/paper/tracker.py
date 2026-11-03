"""
Paper Trading Signal Tracker

Logs hypothetical trades to SQLite and checks outcomes when markets settle.
No real money is ever risked.
"""

import math
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


DB_PATH = os.environ.get(
    "PAPER_TRADING_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_trades.db"),
)


@dataclass
class Signal:
    """A single paper-trading signal."""
    id: Optional[int]
    timestamp: str          # ISO-8601
    market_id: str
    market_title: str
    side: str               # YES / NO
    entry_price: float      # 0-1 scale (e.g. 0.85 = 85¢)
    confidence: float       # model confidence 0-1
    reasoning: str
    strategy: str           # e.g. directional, market_making
    # Outcome fields (filled after settlement)
    outcome: Optional[str]  # win / loss / push / void / pending
    settlement_price: Optional[float]
    pnl: Optional[float]    # per-contract P&L in dollars
    settled_at: Optional[str]


def _ensure_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            market_id       TEXT NOT NULL,
            market_title    TEXT NOT NULL,
            side            TEXT NOT NULL DEFAULT 'NO',
            entry_price     REAL NOT NULL,
            confidence      REAL,
            reasoning       TEXT,
            strategy        TEXT,
            outcome         TEXT DEFAULT 'pending',
            settlement_price REAL,
            pnl             REAL,
            settled_at      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_market
        ON signals(market_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_outcome
        ON signals(outcome)
    """)
    conn.commit()


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_db(conn)
    return conn


def log_signal(
    market_id: str,
    market_title: str,
    side: str,
    entry_price: float,
    confidence: float,
    reasoning: str,
    strategy: str = "directional",
) -> int:
    """Record a new paper-trading signal. Returns the signal id.

    ``side`` must be yes/no in any case (stored upper-case); ``entry_price`` must be
    a finite number strictly between 0 and 1. Anything else raises ValueError.
    """
    if not isinstance(side, str) or side.strip().upper() not in ("YES", "NO"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    side = side.strip().upper()
    try:
        entry_price = float(entry_price)
    except (TypeError, ValueError):
        raise ValueError(f"entry_price must be a number, got {entry_price!r}") from None
    if not math.isfinite(entry_price) or not 0.0 < entry_price < 1.0:
        raise ValueError(f"entry_price must be finite and in (0, 1), got {entry_price!r}")
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO signals
           (timestamp, market_id, market_title, side, entry_price, confidence, reasoning, strategy)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            market_id,
            market_title,
            side,
            entry_price,
            confidence,
            reasoning,
            strategy,
        ),
    )
    conn.commit()
    signal_id = cur.lastrowid
    conn.close()
    return signal_id


def settle_signal(signal_id: int, settlement_price: Optional[float], *, void: bool = False):
    """
    Settle a pending signal. ``settlement_price`` is the YES side's settlement value
    in [0, 1] (1.0 YES won, 0.0 NO won, 0.5 a half-and-half payout).

    Payout per contract is ``s`` for a YES holder and ``1 - s`` for a NO holder, and
    P&L is payout minus entry price (so 0.5 pays both sides 0.5, nobody gets a full
    win). ``void=True`` is a refund: P&L 0, outcome 'void'. A non-finite or
    out-of-range price raises ValueError and leaves the signal pending. Only
    pending signals are touched; settling twice is a no-op.
    """
    if void:
        outcome, pnl, settlement_price = "void", 0.0, None
    else:
        try:
            settlement_price = float(settlement_price)
        except (TypeError, ValueError):
            raise ValueError(f"settlement_price must be a number, got {settlement_price!r}") from None
        if not math.isfinite(settlement_price) or not 0.0 <= settlement_price <= 1.0:
            raise ValueError(f"settlement_price must be finite and in [0, 1], got {settlement_price!r}")
    conn = get_connection()
    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    if not row:
        conn.close()
        return

    if not void:
        payout = settlement_price if row["side"] == "YES" else 1.0 - settlement_price
        pnl = payout - row["entry_price"]
        outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "push"

    conn.execute(
        """UPDATE signals
           SET outcome = ?, settlement_price = ?, pnl = ?, settled_at = ?
           WHERE id = ? AND outcome = 'pending'""",
        (outcome, settlement_price, round(pnl, 4), datetime.now(timezone.utc).isoformat(), signal_id),
    )
    conn.commit()
    conn.close()


def get_pending_signals() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals WHERE outcome = 'pending' ORDER BY timestamp").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all_signals() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals ORDER BY timestamp DESC").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_stats() -> Dict[str, Any]:
    """Compute summary statistics over all settled signals."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signals WHERE outcome != 'pending'").fetchall()
    settled = [dict(r) for r in rows]
    pending = conn.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'pending'").fetchone()[0]
    conn.close()

    if not settled:
        return {
            "total_signals": pending,
            "settled": 0,
            "pending": pending,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    wins = sum(1 for s in settled if s["outcome"] == "win")
    losses = sum(1 for s in settled if s["outcome"] == "loss")
    pnls = [s["pnl"] for s in settled if s["pnl"] is not None]
    total_pnl = sum(pnls)

    return {
        "total_signals": len(settled) + pending,
        "settled": len(settled),
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(total_pnl / len(settled), 4) if settled else 0.0,
        "best_trade": round(max(pnls), 4) if pnls else 0.0,
        "worst_trade": round(min(pnls), 4) if pnls else 0.0,
    }
