"""Local-only backtest data collector.

Captures point-in-time account + position snapshots (and, later, candidate sets
and fills) to an append-only JSONL store under ``data/backtest/`` — which is
git-ignored. This is the raw material for backtesting and the self-learning
loop; per the mission it is NEVER committed to the public repo.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SNAPSHOT_PATH = "data/backtest/snapshots.jsonl"


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_snapshot(
    balance_resp: Dict[str, Any],
    positions_resp: Dict[str, Any],
    tag: str,
    ts: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pure: assemble a snapshot record from raw Kalshi responses."""
    cash = int(balance_resp.get("balance", 0) or 0)
    portfolio = int(balance_resp.get("portfolio_value", 0) or 0)

    positions = []
    for mp in positions_resp.get("market_positions", []):
        fp = _to_float(mp.get("position_fp", "0"))
        if abs(fp) < 0.0001:
            continue
        positions.append({
            "ticker": mp.get("ticker"),
            "side": "YES" if fp > 0 else "NO",
            "count": int(round(abs(fp))),
            "exposure_dollars": _to_float(mp.get("market_exposure_dollars")),
            "traded_dollars": _to_float(mp.get("total_traded_dollars")),
            "realized_pnl_dollars": _to_float(mp.get("realized_pnl_dollars")),
        })

    return {
        "ts": ts,
        "tag": tag,
        "cash_cents": cash,
        "portfolio_cents": portfolio,
        "equity_cents": cash + portfolio,
        "n_positions": len(positions),
        "positions": positions,
        "meta": meta or {},
    }


def append_snapshot(snapshot: Dict[str, Any], path: str = DEFAULT_SNAPSHOT_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(snapshot) + "\n")


async def snapshot_account(
    kalshi_client,
    tag: str,
    meta: Optional[Dict[str, Any]] = None,
    path: str = DEFAULT_SNAPSHOT_PATH,
) -> Dict[str, Any]:
    """Fetch live balance + positions, build a snapshot, append it, return it."""
    balance = await kalshi_client.get_balance()
    # Paginate non-zero positions across all pages.
    market_positions = []
    cursor = None
    for _ in range(30):
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = await kalshi_client._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/positions", params=params)
        market_positions.extend(resp.get("market_positions", []))
        cursor = resp.get("cursor")
        if not cursor:
            break
    ts = datetime.now(timezone.utc).isoformat()
    snap = build_snapshot(balance, {"market_positions": market_positions}, tag, ts, meta)
    append_snapshot(snap, path)
    return snap
