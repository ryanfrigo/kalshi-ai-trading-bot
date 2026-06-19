"""Settlement tracking — measures REALIZED edge from Kalshi's authoritative
/portfolio/settlements records. Appends settled outcomes to a git-ignored local
log and summarizes win-rate / P&L. This closes the learning loop: it's how we
find out whether the near-certain-NO edge is real on Kalshi.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SETTLEMENTS_PATH = "data/runtime/settlements.jsonl"


def _f(rec: Dict[str, Any], k: str) -> float:
    try:
        return float(rec.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def settlement_pnl(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Realized outcome for one settlement record, or None if no position held.

    revenue is in cents (value=100 == $1.00 settlement); cost/fee in dollars.
    pnl = revenue - cost - fees. won = held side matched market_result.
    """
    yc, nc = _f(rec, "yes_count_fp"), _f(rec, "no_count_fp")
    if yc < 0.5 and nc < 0.5:
        return None
    held_side = "yes" if yc >= 0.5 else "no"
    count = int(round(yc if held_side == "yes" else nc))
    result = rec.get("market_result")
    cost = _f(rec, "yes_total_cost_dollars") + _f(rec, "no_total_cost_dollars")
    fee = _f(rec, "fee_cost")
    revenue = _f(rec, "revenue") / 100.0
    return {
        "ticker": rec.get("ticker"),
        "event": rec.get("event_ticker"),
        "held_side": held_side,
        "count": count,
        "won": held_side == result,
        "result": result,
        "cost": round(cost, 4),
        "revenue": round(revenue, 4),
        "fee": fee,
        "pnl": round(revenue - cost - fee, 4),
        "settled_time": rec.get("settled_time"),
    }


async def fetch_settlements(kalshi_client, limit: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(10):
        params = {"limit": min(limit, 200)}
        if cursor:
            params["cursor"] = cursor
        resp = await kalshi_client._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/settlements", params=params)
        out.extend(resp.get("settlements", []))
        cursor = resp.get("cursor")
        if not cursor or len(out) >= limit:
            break
    return out


def load_settlements(path: str = DEFAULT_SETTLEMENTS_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def record_settlements(rows: List[Dict[str, Any]], path: str = DEFAULT_SETTLEMENTS_PATH) -> List[Dict[str, Any]]:
    """Append new settled rows, deduped by (ticker, settled_time). Returns the new ones."""
    seen = {(r.get("ticker"), r.get("settled_time")) for r in load_settlements(path)}
    new = [r for r in rows if (r.get("ticker"), r.get("settled_time")) not in seen]
    if new:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return new


def summarize_settlements(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    wins = sum(1 for r in rows if r.get("won"))
    pnl = sum(r.get("pnl", 0.0) for r in rows)
    by_side: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = by_side.setdefault(r.get("held_side", "?"), {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1
        d["wins"] += 1 if r.get("won") else 0
        d["pnl"] = round(d["pnl"] + r.get("pnl", 0.0), 4)
    return {
        "n": n,
        "wins": wins,
        "win_rate": (wins / n if n else None),
        "pnl": round(pnl, 4),
        "by_side": by_side,
    }
