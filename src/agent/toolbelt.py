"""Agent toolbelt — the guarded primitives Claude drives each /loop tick.

- cap_count: hard size backstop (per-position equity cap + cash), independent of
  whatever size is requested.
- account_brief: one structured situational-awareness snapshot.
- place_guarded_order: one order through the governor + price/balance guards,
  journaling the agent's prediction (est_prob, edge, rationale) for later
  calibration measurement.
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.clients.kalshi_client import KalshiOrderOutcomeUnknown

logger = logging.getLogger(__name__)


_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,99}")


def _normalize_ticker(ticker) -> Optional[str]:
    """``ticker`` stripped and upper-cased if it is then a plain Kalshi ticker, else
    None. It is spliced into request paths, so URL syntax (/ ? # %) must never
    get through."""
    if not isinstance(ticker, str):
        return None
    t = ticker.strip().upper()
    return t if _TICKER_RE.fullmatch(t) else None


def cap_count(
    count: int,
    price_cents: int,
    equity_cents: int,
    cash_cents: int,
    max_position_pct: float = 0.10,
    committed_cents: int = 0,
) -> Tuple[int, str]:
    """Cap an intended contract count to the per-position equity limit and cash.

    ``committed_cents`` (held + resting cost already on this ticker) counts against
    the position limit.

    Returns (allowed_count, reason). reason is "" when nothing was capped.
    """
    n = int(count)
    reasons = []
    if price_cents <= 0:
        return 0, "invalid price"

    max_by_pos = max(0, int(equity_cents * max_position_pct) - committed_cents) // price_cents
    if n > max_by_pos:
        n = max_by_pos
        reasons.append(f"position cap {max_position_pct:.0%}")

    max_by_cash = cash_cents // price_cents
    if n > max_by_cash:
        n = max_by_cash
        reasons.append("cash")

    return max(0, n), "; ".join(reasons)


_EXPOSURE_MAX_PAGES = 20


def _contracts(row: Dict[str, Any], fp_key: str, legacy_key: str) -> float:
    """Contract count of ``row``: the v2 ``*_fp`` field, else the legacy int field.
    Raises ValueError if neither is present and finite, so the caller fails closed."""
    for key in (fp_key, legacy_key):
        v = row.get(key)
        if v is None or v == "" or isinstance(v, bool):
            continue
        try:
            n = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"unparseable {key}={v!r}")
        if not math.isfinite(n):
            raise ValueError(f"non-finite {key}={v!r}")
        return abs(n)
    raise ValueError(f"no {fp_key}/{legacy_key} on row for {row.get('ticker')}")


def _fill_count(order: Dict[str, Any]) -> Optional[float]:
    """fill_count_fp, else legacy fill_count; None if unreadable. Never raises:
    the order is already placed, so the result must still carry its order_id."""
    try:
        return _contracts(order, "fill_count_fp", "fill_count")
    except ValueError:
        return None


async def _sum_pages(fetch, list_key: str, ticker: str, fp_key: str, legacy_key: str) -> float:
    """Sum contracts on ``ticker`` across every page of ``fetch``. Raises on a bad
    list shape, a malformed row, an unreadable count, or more than
    _EXPOSURE_MAX_PAGES pages."""
    total, cursor = 0.0, None
    for _ in range(_EXPOSURE_MAX_PAGES):
        resp = await (fetch(ticker=ticker, cursor=cursor) if cursor else fetch(ticker=ticker))
        rows = resp.get(list_key) if isinstance(resp, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"{list_key} missing or not a list")
        for r in rows:
            if not isinstance(r, dict) or not isinstance(r.get("ticker"), str) or not r["ticker"]:
                raise ValueError(f"malformed row in {list_key}")
        total += sum(_contracts(r, fp_key, legacy_key)
                     for r in rows if r.get("ticker") == ticker)
        cursor = resp.get("cursor")
        if not cursor:
            return total
    raise ValueError(f"more than {_EXPOSURE_MAX_PAGES} pages of {list_key}")


async def _committed_cents(kalshi_client, ticker: str, price_cents: int) -> int:
    """Cost already tied up in ``ticker`` (held + resting), valued at ``price_cents``.
    Raises if either read fails or is unreadable, so the caller can fail closed."""
    held = await _sum_pages(kalshi_client.get_positions, "market_positions", ticker,
                            "position_fp", "position")
    resting = await _sum_pages(
        lambda **kw: kalshi_client.get_orders(status="resting", **kw),
        "orders", ticker, "remaining_count_fp", "remaining_count")
    return int(round((held + resting) * price_cents))


async def account_brief(kalshi_client, governor=None) -> Dict[str, Any]:
    """Structured situational awareness: governor verdict, equity, positions,
    resting orders. Read-only. ``read_errors`` names reads this noticed failed, came back
    malformed or were cut short. It holds tickers, so don't publish it as is. An empty
    list isn't proof of completeness: it only means none of these checks tripped."""
    from src.risk.risk_governor import RiskGovernor

    gov = governor or RiskGovernor(kalshi_client=kalshi_client)
    decision = await gov.check()

    bal = await kalshi_client.get_balance()
    cash = int(bal.get("balance", 0) or 0)
    pv = int(bal.get("portfolio_value", 0) or 0)

    read_errors = []
    if bal.get("balance") is None or bal.get("portfolio_value") is None:
        read_errors.append("balance")
    positions = []
    cursor = None
    for _ in range(30):
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = await kalshi_client._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/positions", params=params)
        rows = resp.get("market_positions")
        if not isinstance(rows, list):
            read_errors.append("positions: no list")
            rows = []
        for mp in rows:
            if not isinstance(mp, dict):
                read_errors.append("positions: bad row")
                continue
            try:
                if isinstance(mp["position_fp"], bool):
                    raise ValueError(mp["position_fp"])
                fp = float(mp["position_fp"])
                if not math.isfinite(fp):
                    raise ValueError(fp)
            except (KeyError, TypeError, ValueError, OverflowError):
                read_errors.append(f"position {mp.get('ticker')}")
                continue
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
    else:
        read_errors.append("positions: page cap")

    try:
        resp = await kalshi_client.get_orders(status="resting")
        orders = resp.get("orders")
        if not isinstance(orders, list):
            raise ValueError("no orders list")
        if resp.get("cursor"):
            read_errors.append("resting_orders: first page only")
    except Exception:
        orders = []
        read_errors.append("resting_orders")
    if not all(isinstance(o, dict) for o in orders):
        read_errors.append("resting_orders: bad row")
        orders = [o for o in orders if isinstance(o, dict)]
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
        "read_errors": read_errors,
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
    dry: bool = True,
    journal_path: Optional[str] = None,
    policy_path: Optional[str] = None,
    override_policy: bool = False,
    expiration_ts: Optional[int] = None,
) -> Dict[str, Any]:
    """Place ONE order through the full guard stack, journaling the prediction.

    Guards (abort on first failure): governor not halted -> Edge Policy gate
    (block a category/method your settled record proves loses; haircut an
    overconfident est_prob) -> market active & tradeable -> price in 1..99c ->
    size capped to position+cash limits >= 1. On success (live or dry) appends a
    decision-journal record.

    The Edge Policy gate closes the self-improvement loop: it consults the active
    policy (``data/runtime/edge_policy.json``, kept fresh by ``cli improve``).
    No policy file -> no opinion -> proceeds (backward-compatible). A file that
    exists but can't be read as a policy, or holds a malformed rule, refuses the order
    (``reason="policy_unreadable"``): its blocks are unknown, so proceeding would
    silently switch the gate off. ``override_policy=True`` passes it, journaled
    via ``policy_note``. A BLOCK is a
    hard refusal unless ``override_policy=True`` (the agent retains full authority,
    but the override is recorded in the journal via ``policy_note``). A HAIRCUT
    shrinks the *journaled* est_prob to the policy's calibrated value; if that
    correction pushes edge <= 0 at the chosen price, the order is aborted
    (``reason="haircut_erased_edge"``) rather than placed as a -EV trade —
    ``override_policy=True`` forces it through anyway.
    """
    from src.risk.risk_governor import RiskGovernor
    from src.utils.market_prices import get_market_prices, is_tradeable_market
    from src.agent.journal import (make_decision_record, append_decision,
                                   mark_outcome_unknown, DEFAULT_JOURNAL_PATH)
    from src.agent.policy import (load_policy, apply_policy, policy_is_well_formed,
                                  DEFAULT_POLICY_PATH)
    from src.agent.settle import series_category

    from src.config.settings import settings

    # The caller may only tighten the configured per-position cap, never widen it.
    cap_pct = settings.trading.max_position_size_pct / 100.0
    max_position_pct = float(max_position_pct)
    if not max_position_pct <= cap_pct:  # also catches NaN
        max_position_pct = cap_pct

    clean = _normalize_ticker(ticker)
    if clean is None:
        return {"ok": False, "reason": f"invalid ticker {ticker!r}"}
    ticker = clean
    side = side.lower() if isinstance(side, str) else ""
    if side not in ("yes", "no"):
        return {"ok": False, "reason": "side must be 'yes' or 'no'"}
    gov = governor or RiskGovernor(kalshi_client=kalshi_client)
    decision = await gov.check()
    if decision.halted:
        return {"ok": False, "reason": "governor_halted", "governor": decision.to_dict()}

    # Edge Policy gate — the settled record constrains the next decision.
    policy_note: Optional[Dict[str, Any]] = None
    haircut_applied = False
    resolved_policy_path = policy_path or DEFAULT_POLICY_PATH
    policy = load_policy(resolved_policy_path)
    if (not policy_is_well_formed(policy) if policy is not None
            else Path(resolved_policy_path).exists()):
        if not override_policy:
            return {"ok": False, "reason": "policy_unreadable",
                    "policy_path": resolved_policy_path}
        policy_note = {"overridden_unreadable_policy": resolved_policy_path}
        policy = None
    if policy:
        verdict = apply_policy(policy, {
            "ticker": ticker, "side": side,
            "category": category or series_category(ticker), "est_prob": est_prob,
        })
        if verdict["verdict"] == "BLOCK":
            if not override_policy:
                return {"ok": False, "reason": "blocked_by_policy",
                        "policy_verdict": verdict}
            policy_note = {"overridden_block": verdict["reasons"]}
        elif verdict["verdict"] == "HAIRCUT":
            est_prob = verdict.get("adjusted_est_prob", est_prob)
            haircut_applied = True
            policy_note = {"haircut": verdict["reasons"],
                           "est_prob_shrunk_to": est_prob}

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
    try:
        committed = await _committed_cents(kalshi_client, ticker, price_cents)
    except Exception as e:
        return {"ok": False, "reason": f"exposure_unreadable ({e})"}
    n, cap_reason = cap_count(count, price_cents, decision.current_equity_cents,
                              cash, max_position_pct, committed)
    if n < 1:
        return {"ok": False, "reason": f"size capped to 0 ({cap_reason or 'insufficient cash'})"}

    edge = round(est_prob - price_cents / 100.0, 4) if est_prob is not None else None

    # If the policy haircut shrank est_prob below the price, the corrected
    # estimate says this is a losing entry — honoring the gate means NOT placing
    # it (the whole point was "your record proves you're overconfident here").
    # The agent can still force it with override_policy.
    if haircut_applied and not override_policy and edge is not None and edge <= 0:
        return {"ok": False, "reason": "haircut_erased_edge",
                "policy_note": policy_note, "edge_after_haircut": edge,
                "price_cents": price_cents}

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
        if expiration_ts is not None:
            kwargs["expiration_ts"] = expiration_ts  # resting quotes must not go stale
        try:
            resp = await kalshi_client.place_order(**kwargs)
        except KalshiOrderOutcomeUnknown:
            # The order may be live: journal it as pending, report it, never resend.
            unknown = {"ok": False, "reason": "outcome_unknown", "ticker": ticker,
                       "side": side, "count": n, "client_order_id": coid, "journaled": True}
            try:
                append_decision(mark_outcome_unknown(make_decision_record(
                    ticker=ticker, side=side, count=n, price=price_cents / 100.0,
                    est_prob=est_prob, edge=edge, rationale=rationale, category=category,
                    strategy="claude", order_id=None, policy_note=policy_note,
                ), coid), journal_path or DEFAULT_JOURNAL_PATH)
            except Exception as e:  # never lose the client_order_id to a journal error
                logger.warning("outcome_unknown buy %s not journaled: %s", coid, e)
                unknown.update(journaled=False, journal_error=str(e))
            return unknown
        order = resp.get("order", resp) if isinstance(resp, dict) else {}
        result = {
            "ok": True, "dry": False, "ticker": ticker, "side": side, "count": n,
            "price_cents": price_cents, "edge": edge, "cap_reason": cap_reason,
            "order_id": order.get("order_id"), "fill_count": _fill_count(order),
        }

    if not dry:
        append_decision(make_decision_record(
            ticker=ticker, side=side, count=n, price=price_cents / 100.0,
            est_prob=est_prob, edge=edge, rationale=rationale, category=category,
            strategy="claude", order_id=result.get("order_id"),
            policy_note=policy_note,
        ), journal_path or DEFAULT_JOURNAL_PATH)
    if policy_note:
        result["policy_note"] = policy_note
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
    dry: bool = True,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Sell (close) an existing position with a marketable limit at the bid.

    Reads the live holding, sells the side held (capped to held count net of our
    resting orders; nothing if those can't be read), at the current bid for that
    side (or `price` if given). Journals the close. Under a governor halt,
    ``KalshiClient.place_order`` blocks the sell on MANUAL_HALT and allows a
    reduce-only IOC sell up to held on DAILY_LOSS / DRAWDOWN.
    """
    from src.utils.market_prices import get_market_prices
    from src.agent.journal import (make_decision_record, append_decision,
                                   mark_outcome_unknown, DEFAULT_JOURNAL_PATH)

    clean = _normalize_ticker(ticker)
    if clean is None:
        return {"ok": False, "reason": f"invalid ticker {ticker!r}", "ticker": ticker}
    ticker = clean
    pos = await kalshi_client.get_positions(ticker=ticker)
    mine = [p for p in pos.get("market_positions", []) if p.get("ticker") == ticker]
    fp = float(mine[0].get("position_fp", "0")) if mine else 0.0
    side, n = resolve_sell(fp, count)
    if not side or n < 1:
        return {"ok": False, "reason": "no position to close", "ticker": ticker}

    # Net our resting orders against the holding (same rules as the exit jobs),
    # so a repeated close cannot sell past the position. Unreadable -> sell nothing.
    from src.jobs.execute import _sellable_contracts
    sellable = await _sellable_contracts(kalshi_client, ticker, side)
    if sellable is None:
        return {"ok": False, "reason": "sellable_unreadable", "ticker": ticker, "side": side}
    n = min(n, int(math.floor(sellable + 1e-9)))
    if n < 1:
        return {"ok": False, "reason": "nothing sellable (resting orders cover the holding)",
                "ticker": ticker, "side": side}

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
        try:
            resp = await kalshi_client.place_order(**kwargs)
        except KalshiOrderOutcomeUnknown:
            # The sell may be live: journal it as pending, report it, never resend.
            unknown = {"ok": False, "reason": "outcome_unknown", "ticker": ticker,
                       "side": side, "action": "sell", "count": n, "client_order_id": coid,
                       "price_cents": price_cents, "journaled": True}
            try:
                append_decision(mark_outcome_unknown(make_decision_record(
                    ticker=ticker, side=side, count=n, price=price_cents / 100.0,
                    rationale=rationale, strategy="claude", order_id=None, action="sell",
                ), coid), journal_path or DEFAULT_JOURNAL_PATH)
            except Exception as e:  # never lose the client_order_id to a journal error
                logger.warning("outcome_unknown sell %s not journaled: %s", coid, e)
                unknown.update(journaled=False, journal_error=str(e))
            return unknown
        order = resp.get("order", resp) if isinstance(resp, dict) else {}
        result = {"ok": True, "dry": False, "ticker": ticker, "side": side,
                  "action": "sell", "count": n, "price_cents": price_cents,
                  "order_id": order.get("order_id"), "fill_count": _fill_count(order)}

    if not dry:
        append_decision(make_decision_record(
            ticker=ticker, side=side, count=n, price=price_cents / 100.0,
            rationale=rationale, strategy="claude", order_id=result.get("order_id"),
            action="sell",
        ), journal_path or DEFAULT_JOURNAL_PATH)
    result["journaled"] = not dry
    return result
