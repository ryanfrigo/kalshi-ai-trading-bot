#!/usr/bin/env python3
"""Read-only preview of the next run_tracking pass. Run it before `cli.py run --live`.

It runs the real src.jobs.track.run_tracking (profit-take, stop-loss, the
should_exit_position exits and the sellable guard) against two stand-ins:

- a client that forwards only GET reads. place_order is recorded and never
  forwarded; anything else (cancel_order, amend, batch, raw requests) is refused.
  As a backstop, the HTTP client refuses any non-GET request before it is sent.
- a database that forwards only get_open_live_positions; writes are recorded
  and dropped.

A recorded sell is treated like a resting GTC order that filled nothing, so a
later step in the same pass sees it through the sellable guard, as it would live.

It then calls GET /portfolio/orders?status=resting once and reports the fields
the sellable guard reads on each resting order, and whether orders the bot
placed through POST /portfolio/events/orders (ids in the decision journal)
show up in that list.

Usage:  python scripts/exit_preview.py [--db trading_system.db]
"""
import argparse
import asyncio
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import structlog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.journal import DEFAULT_JOURNAL_PATH, load_journal  # noqa: E402
from src.jobs import execute as execute_mod  # noqa: E402
from src.jobs import track as track_mod  # noqa: E402
from src.utils.database import DatabaseManager  # noqa: E402
from src.utils.market_prices import get_market_prices  # noqa: E402

NOT_SENT_ID = "PREVIEW-NOT-SENT"


class ReadOnlyViolation(RuntimeError):
    pass


class ReadOnlyClient:
    """The client run_tracking sees. Only the reads it needs reach the exchange."""

    def __init__(self, inner):
        self._inner = inner
        self.phase = "start"
        self.markets: Dict[str, Dict] = {}
        self.sells: List[Dict] = []
        self.violations: List[str] = []
        self._not_sent: Dict[str, List[Dict]] = {}
        hooks = inner.client.event_hooks
        hooks.setdefault("request", []).append(_async_refuse)
        inner.client.event_hooks = hooks
        real_request = inner._make_authenticated_request

        async def _get_only(method, endpoint, *a, **k):
            if str(method).upper() != "GET":
                self.violations.append(f"{method} {endpoint}")
                raise ReadOnlyViolation(f"refused {method} {endpoint}")
            return await real_request(method, endpoint, *a, **k)

        inner._make_authenticated_request = _get_only

    async def get_market(self, ticker: str) -> Dict:
        resp = await self._inner.get_market(ticker)
        self.markets[ticker] = (resp or {}).get("market") or {}
        return resp

    async def _held_on_side(self, ticker: str, side: str) -> Optional[float]:
        return await self._inner._held_on_side(ticker, side)

    async def get_orders(self, ticker: Optional[str] = None,
                         status: Optional[str] = None) -> Dict:
        resp = await self._inner.get_orders(ticker=ticker, status=status)
        extra = self._not_sent.get(ticker) if status == "resting" else None
        if extra and isinstance(resp, dict) and isinstance(resp.get("orders"), list):
            resp = {**resp, "orders": resp["orders"] + extra}
        return resp

    async def place_order(self, **params) -> Dict:
        """Record the order run_tracking would send. Never forwarded."""
        self.sells.append({"phase": self.phase, **params})
        held = str(params.get("side", "")).lower()
        self._not_sent.setdefault(params.get("ticker"), []).append({
            "ticker": params.get("ticker"), "order_id": NOT_SENT_ID,
            "outcome_side": "no" if held == "yes" else "yes",
            "remaining_count_fp": f"{float(params.get('count', 0)):.2f}",
        })
        return {"order": {"order_id": NOT_SENT_ID, "fill_count_fp": "0.00",
                          "status": "resting"}}

    async def close(self) -> None:
        await self._inner.close()

    def __getattr__(self, name: str):
        self.violations.append(name)
        raise ReadOnlyViolation(f"exit_preview does not allow client.{name}")


async def _async_refuse(request: httpx.Request) -> None:
    """Last line: no non-GET request leaves the HTTP client, whatever the caller."""
    if request.method.upper() != "GET":
        raise ReadOnlyViolation(f"refused {request.method} {request.url.path}")


class ReadOnlyDB:
    """Forwards the one read run_tracking makes; records and drops every write."""

    def __init__(self, inner: DatabaseManager):
        self._inner = inner
        self.writes: List[tuple] = []

    async def get_open_live_positions(self):
        return await self._inner.get_open_live_positions()

    def __getattr__(self, name: str):
        async def _dropped(*args, **kwargs):
            self.writes.append((name, args, kwargs))
            return None
        return _dropped


async def preview_exits(client: ReadOnlyClient, db: ReadOnlyDB) -> Dict[str, Any]:
    """Run the real run_tracking pass read-only; return what it would have done."""
    exits: Dict[str, tuple] = {}
    sellable: List[Dict] = []
    real_profit = execute_mod.place_profit_taking_orders
    real_stop = execute_mod.place_stop_loss_orders
    real_sellable = execute_mod._sellable_contracts
    real_should_exit = track_mod.should_exit_position

    async def profit(*a, **k):
        client.phase = "profit_take"
        return await real_profit(*a, **k)

    async def stop(*a, **k):
        client.phase = "stop_loss"
        return await real_stop(*a, **k)

    async def should_exit(position, *a, **k):
        result = await real_should_exit(position, *a, **k)
        exits[position.market_id] = result
        client.phase = f"exit:{result[1]}" if result[0] else "exit"
        return result

    async def sellable_spy(kalshi_client, ticker, side):
        value = await real_sellable(kalshi_client, ticker, side)
        sellable.append({"phase": client.phase, "ticker": ticker, "sellable": value})
        return value

    with patch.object(track_mod, "KalshiClient", lambda *a, **k: client), \
         patch.object(execute_mod, "place_profit_taking_orders", profit), \
         patch.object(execute_mod, "place_stop_loss_orders", stop), \
         patch.object(execute_mod, "_sellable_contracts", sellable_spy), \
         patch.object(track_mod, "should_exit_position", should_exit):
        await track_mod.run_tracking(db)
    return {"exits": exits, "sellable": sellable}


def _fmt(v, nd=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in cells])


def exit_rows(positions, client: ReadOnlyClient, db: ReadOnlyDB,
              result: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for p in positions:
        t = p.market_id
        market = client.markets.get(t)
        bid = None
        if market:
            yb, _, nb, _ = get_market_prices(market)
            bid = yb if p.side == "YES" else nb
        steps = []
        legacy = "yes_price" if p.side == "YES" else "no_price"
        if market is not None and legacy not in market:
            steps.append(f"profit/stop skipped (market has no {legacy} field)")
        for phase in ("profit_take", "stop_loss"):
            steps += _phase_steps(t, phase, client, result)
        ex = result["exits"].get(t)
        if market is None:
            steps.append("no market data: skipped")
        elif ex is None:
            steps.append("exit check not reached (error; see stderr)")
        elif not ex[0]:
            steps.append("hold")
        elif ex[1] == "market_resolution":
            closes = [w for w in db.writes if w[0] == "update_position_status"
                      and w[1] and w[1][0] == p.id]
            steps.append(f"resolution: DB close at {ex[2]:.2f}, no order"
                         + ("" if closes else " (not written)"))
        elif ex[2] <= 0:
            steps.append(f"{ex[1]}: refused, exit price {ex[2]}")
        else:
            steps += _phase_steps(t, f"exit:{ex[1]}", client, result) or [f"{ex[1]}: no sell"]
        sells = [s for s in client.sells if s.get("ticker") == t]
        first = sells[0] if sells else None
        price = None
        if first:
            cents = first.get("yes_price") if first.get("side") == "yes" else first.get("no_price")
            price = f"{cents}c"
        rows.append([t, p.side, p.quantity, p.entry_price, bid,
                     "SELL" if first else ("CLOSE" if ex and ex[0] and ex[1] == "market_resolution" else "-"),
                     first.get("count") if first else None, price, "; ".join(steps)])
    return rows


def _phase_steps(ticker, phase, client, result) -> List[str]:
    out = []
    name = phase.split(":", 1)[-1]
    for s in result["sellable"]:
        if s["ticker"] != ticker or s["phase"] != phase:
            continue
        sold = [o for o in client.sells if o.get("ticker") == ticker and o["phase"] == phase]
        if s["sellable"] is None:
            out.append(f"{name}: blocked, sellable guard could not confirm holdings/resting orders")
        elif s["sellable"] <= 0 or not sold:
            out.append(f"{name}: blocked, nothing sellable (held minus resting = {s['sellable']})")
        else:
            o = sold[0]
            cents = o.get("yes_price") if o.get("side") == "yes" else o.get("no_price")
            out.append(f"{name}: sell {o.get('count')} @ {cents}c (sellable {s['sellable']})")
    return out


def _has(o: Dict, key: str) -> str:
    return "yes" if o.get(key) not in (None, "") else "NO"


def _remaining(o: Dict) -> Optional[float]:
    for key in ("remaining_count_fp", "remaining_count"):
        try:
            v = float(o.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 0:
            return v
    return None


def resting_report(resp: Any, journal: List[Dict]) -> str:
    orders = resp.get("orders") if isinstance(resp, dict) else None
    if not isinstance(orders, list):
        return "GET /portfolio/orders?status=resting returned no 'orders' list; cannot answer."
    rows = []
    for o in orders:
        if not isinstance(o, dict):
            continue
        readable = (execute_mod._order_outcome(o) is not None and _remaining(o) is not None)
        rows.append([o.get("order_id"), o.get("ticker"), _has(o, "outcome_side"),
                     _has(o, "book_side"), _has(o, "remaining_count_fp"),
                     "yes" if readable else "NO (guard blocks this ticker)"])
    out = [f"Resting orders: {len(rows)}"
           + (" (response has a cursor: more pages exist, only page 1 shown)"
              if resp.get("cursor") and len(orders) >= 100 else "")]
    if rows:
        out.append(_table(["order_id", "ticker", "outcome_side", "book_side",
                           "remaining_count_fp", "guard_can_read"], rows))
    out.append("")
    out.append(events_endpoint_verdict(orders, journal))
    return "\n".join(out)


def events_endpoint_verdict(orders: List[Dict], journal: List[Dict]) -> str:
    """Do orders placed via POST /portfolio/events/orders appear in the resting list?

    Every order this bot places goes through place_order, which posts only to
    /portfolio/events/orders, and the agent journal records its order_id.
    """
    resting_ids = {o.get("order_id") for o in orders if isinstance(o, dict)}
    placed = [r for r in journal if r.get("order_id")]
    open_ids = {r["order_id"] for r in placed if not r.get("outcome") and not r.get("voided")}
    seen = sorted(open_ids & resting_ids)
    q = "Do orders placed via POST /portfolio/events/orders appear in GET /portfolio/orders?status=resting?"
    if seen:
        return (f"{q}\nANSWER: YES. {len(seen)} order id(s) the bot placed through the events "
                f"endpoint (from the journal) are in the resting list: {', '.join(seen)}")
    return (f"{q}\nANSWER: NOT SHOWN BY THIS ACCOUNT. The journal holds {len(placed)} events-endpoint "
            f"order id(s), {len(open_ids)} not yet settled or voided, and none is in the resting "
            f"list. That is consistent with all of them having filled or been cancelled, so it is "
            f"not evidence either way. It needs one known resting events-endpoint order to settle.")


def _quiet_logs() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )


async def run(db_path: str, journal_path: str, client_factory=None) -> int:
    if not os.path.exists(db_path):
        print(f"No database at {db_path}; pass --db.", file=sys.stderr)
        return 2
    if client_factory is None:
        from src.clients.kalshi_client import KalshiClient
        client_factory = KalshiClient
    client = ReadOnlyClient(client_factory())
    db = ReadOnlyDB(DatabaseManager(db_path=db_path))
    positions = await db.get_open_live_positions()
    resting = await client.get_orders(status="resting")  # the one unscoped call
    print(f"EXIT PREVIEW: next run_tracking pass, nothing sent. {len(positions)} open live position(s).")
    print("Sells are as sent before the risk governor; under a DAILY_LOSS/DRAWDOWN halt "
          "a sell goes out IOC reduce-only.\n")
    result = await preview_exits(client, db)
    if positions:
        print(_table(["ticker", "side", "qty", "entry", "bid", "action", "contracts",
                      "price", "why"], exit_rows(positions, client, db, result)))
    dropped = [w[0] for w in db.writes]
    if dropped:
        print(f"\nDB writes the pass would make (dropped): {', '.join(dropped)}")
    print()
    print(resting_report(resting, load_journal(journal_path)))
    violations = client.violations
    if violations:
        print(f"\nREAD-ONLY VIOLATION, refused and not sent: {violations}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="trading_system.db")
    ap.add_argument("--journal", default=DEFAULT_JOURNAL_PATH)
    args = ap.parse_args()
    _quiet_logs()
    sys.exit(asyncio.run(run(args.db, args.journal)))


if __name__ == "__main__":
    main()
