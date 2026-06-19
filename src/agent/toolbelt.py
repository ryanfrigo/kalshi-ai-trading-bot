"""Agent toolbelt — the guarded primitives Claude drives each /loop tick.

- cap_count: hard size backstop (per-position equity cap + cash), independent of
  whatever size is requested.
- account_brief: one structured situational-awareness snapshot.
- place_guarded_order: one order through the governor + price/balance guards,
  journaling the agent's prediction (est_prob, edge, rationale) for later
  calibration measurement.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple


def cap_count(
    count: int,
    price_cents: int,
    equity_cents: int,
    cash_cents: int,
    max_position_pct: float = 0.10,
) -> Tuple[int, str]:
    """Cap an intended contract count to the per-position equity limit and cash.

    Returns (allowed_count, reason). reason is "" when nothing was capped.
    """
    n = int(count)
    reasons = []
    if price_cents <= 0:
        return 0, "invalid price"

    max_by_pos = int(equity_cents * max_position_pct) // price_cents
    if n > max_by_pos:
        n = max_by_pos
        reasons.append(f"position cap {max_position_pct:.0%}")

    max_by_cash = cash_cents // price_cents
    if n > max_by_cash:
        n = max_by_cash
        reasons.append("cash")

    return max(0, n), "; ".join(reasons)


async def account_brief(kalshi_client, governor=None) -> Dict[str, Any]:
    """Structured situational awareness: governor verdict, equity, positions,
    resting orders. Read-only."""
    from src.risk.risk_governor import RiskGovernor

    gov = governor or RiskGovernor(kalshi_client=kalshi_client)
    decision = await gov.check()

    bal = await kalshi_client.get_balance()
    cash = int(bal.get("balance", 0) or 0)
    pv = int(bal.get("portfolio_value", 0) or 0)

    positions = []
    cursor = None
    for _ in range(30):
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = await kalshi_client._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/positions", params=params)
        for mp in resp.get("market_positions", []):
            try:
                fp = float(mp.get("position_fp", "0") or 0)
            except ValueError:
                fp = 0.0
            if abs(fp) > 0.0001:
                positions.append({
                    "ticker": mp.get("ticker"),
                    "side": "YES" if fp > 0 else "NO",
                    "count": int(round(abs(fp))),
                    "exposure": float(mp.get("market_exposure_dollars", 0) or 0),
                    "traded": float(mp.get("total_traded_dollars", 0) or 0),
                })
        cursor = resp.get("cursor")
        if not cursor:
            break

    try:
        orders = (await kalshi_client.get_orders(status="resting")).get("orders", [])
    except Exception:
        orders = []
    resting = [{
        "ticker": o.get("ticker"),
        "outcome_side": o.get("outcome_side"),
        "no_price": o.get("no_price_dollars"),
        "remaining": o.get("remaining_count_fp"),
        "order_id": o.get("order_id"),
    } for o in orders]

    return {
        "governor": decision.to_dict(),
        "cash": cash / 100.0,
        "portfolio": pv / 100.0,
        "equity": (cash + pv) / 100.0,
        "positions": positions,
        "resting_orders": resting,
    }


async def place_guarded_order(
    kalshi_client,
    ticker: str,
    side: str,
    count: int,
    price: Optional[float] = None,
    type_: str = "limit",
    rationale: str = "",
    est_prob: Optional[float] = None,
    category: str = "",
    governor=None,
    max_position_pct: float = 0.10,
    dry: bool = False,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Place ONE order through the full guard stack, journaling the prediction.

    Guards (abort on first failure): governor not halted -> market active &
    tradeable -> price in 1..99c -> size capped to position+cash limits >= 1.
    On success (live or dry) appends a decision-journal record.
    """
    from src.risk.risk_governor import RiskGovernor
    from src.utils.market_prices import get_market_prices, is_tradeable_market
    from src.agent.journal import make_decision_record, append_decision, DEFAULT_JOURNAL_PATH

    side = side.lower()
    gov = governor or RiskGovernor(kalshi_client=kalshi_client)
    decision = await gov.check()
    if decision.halted:
        return {"ok": False, "reason": "governor_halted", "governor": decision.to_dict()}

    md = await kalshi_client.get_market(ticker)
    m = md.get("market", {})
    if m.get("status") != "active" or not is_tradeable_market(m):
        return {"ok": False, "reason": f"market_not_tradeable(status={m.get('status')})"}

    yb, ya, nb, na = get_market_prices(m)
    chosen = price if price is not None else (na if side == "no" else ya)
    price_cents = int(round(chosen * 100))
    if not (1 <= price_cents <= 99):
        return {"ok": False, "reason": f"price {price_cents}c out of range 1..99"}

    bal = await kalshi_client.get_balance()
    cash = int(bal.get("balance", 0) or 0)
    n, cap_reason = cap_count(count, price_cents, decision.current_equity_cents,
                              cash, max_position_pct)
    if n < 1:
        return {"ok": False, "reason": f"size capped to 0 ({cap_reason or 'insufficient cash'})"}

    edge = round(est_prob - price_cents / 100.0, 4) if est_prob is not None else None
    coid = str(uuid.uuid4())

    if dry:
        result: Dict[str, Any] = {
            "ok": True, "dry": True, "ticker": ticker, "side": side, "count": n,
            "price_cents": price_cents, "edge": edge, "cap_reason": cap_reason,
            "order_id": None,
        }
    else:
        kwargs = {"ticker": ticker, "client_order_id": coid, "side": side,
                  "action": "buy", "count": n, "type_": type_}
        kwargs["no_price" if side == "no" else "yes_price"] = price_cents
        resp = await kalshi_client.place_order(**kwargs)
        order = resp.get("order", resp) if isinstance(resp, dict) else {}
        result = {
            "ok": True, "dry": False, "ticker": ticker, "side": side, "count": n,
            "price_cents": price_cents, "edge": edge, "cap_reason": cap_reason,
            "order_id": order.get("order_id"), "fill_count": order.get("fill_count"),
        }

    if not dry:
        append_decision(make_decision_record(
            ticker=ticker, side=side, count=n, price=price_cents / 100.0,
            est_prob=est_prob, edge=edge, rationale=rationale, category=category,
            strategy="claude", order_id=result.get("order_id"),
        ), journal_path or DEFAULT_JOURNAL_PATH)
    result["journaled"] = not dry
    return result


def resolve_sell(position_fp, requested_count: Optional[int] = None):
    """Resolve a close request to (side, count). Sells the side actually held
    (YES if position_fp>0 else NO) and never more than is held. Returns
    (None, 0) for a flat position."""
    fp = float(position_fp)
    if abs(fp) < 0.5:
        return None, 0
    side = "yes" if fp > 0 else "no"
    held = int(round(abs(fp)))
    count = held if requested_count is None else min(int(requested_count), held)
    return side, max(0, count)


async def close_position(
    kalshi_client,
    ticker: str,
    count: Optional[int] = None,
    price: Optional[float] = None,
    rationale: str = "close position",
    dry: bool = False,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Sell (close) an existing position with a marketable limit at the bid.

    Reads the live holding, sells the side held (capped to held count), at the
    current bid for that side (or `price` if given). Journals the close. Selling
    reduces risk, so it is allowed even when the governor is halted.
    """
    from src.utils.market_prices import get_market_prices
    from src.agent.journal import make_decision_record, append_decision, DEFAULT_JOURNAL_PATH

    pos = await kalshi_client.get_positions(ticker=ticker)
    mine = [p for p in pos.get("market_positions", []) if p.get("ticker") == ticker]
    fp = float(mine[0].get("position_fp", "0")) if mine else 0.0
    side, n = resolve_sell(fp, count)
    if not side or n < 1:
        return {"ok": False, "reason": "no position to close", "ticker": ticker}

    m = (await kalshi_client.get_market(ticker)).get("market", {})
    yb, ya, nb, na = get_market_prices(m)
    bid = yb if side == "yes" else nb
    sell_price = price if price is not None else bid
    price_cents = int(round(sell_price * 100))
    if price_cents < 1:
        return {"ok": False, "reason": f"no bid to sell into ({side} bid {bid:.2f})",
                "ticker": ticker, "side": side, "count": n}
    price_cents = min(99, price_cents)

    coid = str(uuid.uuid4())
    if dry:
        result: Dict[str, Any] = {"ok": True, "dry": True, "ticker": ticker,
                                  "side": side, "action": "sell", "count": n,
                                  "price_cents": price_cents, "order_id": None}
    else:
        kwargs = {"ticker": ticker, "client_order_id": coid, "side": side,
                  "action": "sell", "count": n, "type_": "limit"}
        kwargs["yes_price" if side == "yes" else "no_price"] = price_cents
        resp = await kalshi_client.place_order(**kwargs)
        order = resp.get("order", resp) if isinstance(resp, dict) else {}
        result = {"ok": True, "dry": False, "ticker": ticker, "side": side,
                  "action": "sell", "count": n, "price_cents": price_cents,
                  "order_id": order.get("order_id"), "fill_count": order.get("fill_count")}

    if not dry:
        append_decision(make_decision_record(
            ticker=ticker, side=side, count=n, price=price_cents / 100.0,
            rationale=rationale, strategy="claude", order_id=result.get("order_id"),
            action="sell",
        ), journal_path or DEFAULT_JOURNAL_PATH)
    result["journaled"] = not dry
    return result
