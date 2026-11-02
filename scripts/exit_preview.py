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
the sellable guard reads on each resting order. Last, it lists every order
(GET /portfolio/orders, no status filter, all pages) and checks whether order
ids the bot placed through POST /portfolio/events/orders (from the decision
journal) appear there, whatever their status.

Usage:  python scripts/exit_preview.py [--db trading_system.db]
"""
import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import aiosqlite
import httpx
import structlog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.journal import DEFAULT_JOURNAL_PATH, load_journal  # noqa: E402
from src.jobs import execute as execute_mod  # noqa: E402
from src.jobs import track as track_mod  # noqa: E402
from src.utils.database import DatabaseManager  # noqa: E402
from src.utils.market_prices import get_market_prices  # noqa: E402

NOT_SENT_ID = "PREVIEW-NOT-SENT"
ORDERS_PATH = "/trade-api/v2/portfolio/orders"
PAGE_LIMIT = 200
MAX_PAGES = 50


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

    async def list_all_orders(self) -> List[Dict]:
        """Every order, any status: GET /portfolio/orders paginated by cursor.

        Fails closed (raises) on a malformed page, a repeated cursor or more than
        MAX_PAGES pages, so a partial listing is never mistaken for the whole.
        """
        orders: List[Dict] = []
        cursor, seen = None, set()
        for _ in range(MAX_PAGES):
            params = {"limit": PAGE_LIMIT, **({"cursor": cursor} if cursor else {})}
            resp = await self._inner._make_authenticated_request("GET", ORDERS_PATH, params=params)
            page = resp.get("orders") if isinstance(resp, dict) else None
            if not isinstance(page, list):
                raise ValueError("a page of GET /portfolio/orders has no 'orders' list")
            orders += [o for o in page if isinstance(o, dict)]
            cursor = resp.get("cursor") or None
            if not cursor:
                return orders
            if cursor in seen:
                raise ValueError("GET /portfolio/orders repeated a cursor")
            seen.add(cursor)
        raise ValueError(f"GET /portfolio/orders has more than {MAX_PAGES} pages")

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
    """Forwards the one read run_tracking makes; records and drops every write.

    The read goes through the real DatabaseManager reader on a mode=ro URI
    connection, so even a SELECT cannot write (no hot-journal recovery).
    """

    def __init__(self, db_path: str):
        self._inner = DatabaseManager(db_path=Path(db_path).resolve().as_uri() + "?mode=ro")
        self.writes: List[tuple] = []

    async def get_open_live_positions(self):
        real_connect = aiosqlite.connect
        with patch.object(aiosqlite, "connect",
                          lambda path, **k: real_connect(path, uri=True, **k)):
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
    real_sell = execute_mod.place_sell_limit_order
    attempts: List[Dict] = []
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

    async def sell_spy(position, limit_price, *a, **k):
        attempts.append({"phase": client.phase, "ticker": position.market_id,
                         "limit_price": limit_price})
        return await real_sell(position, limit_price, *a, **k)

    with patch.object(track_mod, "KalshiClient", lambda *a, **k: client), \
         patch.object(execute_mod, "place_profit_taking_orders", profit), \
         patch.object(execute_mod, "place_stop_loss_orders", stop), \
         patch.object(execute_mod, "_sellable_contracts", sellable_spy), \
         patch.object(execute_mod, "place_sell_limit_order", sell_spy), \
         patch.object(track_mod, "should_exit_position", should_exit):
        await track_mod.run_tracking(db)
    return {"exits": exits, "sellable": sellable, "attempts": attempts}


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
    """One line per sell attempt run_tracking made for ``ticker`` in ``phase``."""
    out = []
    name = phase.split(":", 1)[-1]
    pick = lambda xs: [x for x in xs if x.get("ticker") == ticker and x["phase"] == phase]
    sold, guard = pick(client.sells), pick(result["sellable"])
    for a in pick(result["attempts"]):
        if sold:
            o = sold.pop(0)
            cents = o.get("yes_price") if o.get("side") == "yes" else o.get("no_price")
            g = guard.pop(0)["sellable"] if guard else None
            out.append(f"{name}: sell {o.get('count')} @ {cents}c (sellable {g})")
        elif guard:
            g = guard.pop(0)["sellable"]
            out.append(f"{name}: blocked, sellable guard could not confirm holdings/resting orders"
                       if g is None else
                       f"{name}: blocked, nothing sellable (held minus resting = {g})")
        else:
            out.append(f"{name}: not sent, limit {int(round(a['limit_price'] * 100))}c "
                       f"outside 1-99c")
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


def resting_report(resp: Any) -> str:
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
    return "\n".join(out)


def events_endpoint_verdict(all_orders: Optional[List[Dict]], error: Optional[str],
                            journal: List[Dict]) -> str:
    """Do orders placed via POST /portfolio/events/orders appear in GET /portfolio/orders?

    Every order this bot places goes through place_order, which posts only to
    /portfolio/events/orders, and the decision journal records its order_id. Any
    journal id in the full listing, whatever its status, proves they appear.
    """
    q = "Do orders placed via POST /portfolio/events/orders appear in GET /portfolio/orders?"
    ids = {r["order_id"] for r in journal if r.get("order_id")}
    if not ids:
        return (f"{q}\nANSWER: CANNOT TELL. The decision journal holds no order ids, so there "
                f"is no known events-endpoint order to look for.")
    if all_orders is None:
        return f"{q}\nANSWER: CANNOT TELL. The full order listing failed: {error}"
    matched = [o for o in all_orders if o.get("order_id") in ids]
    if matched:
        by_status: Dict[str, int] = {}
        for o in matched:
            by_status[str(o.get("status") or "?")] = by_status.get(str(o.get("status") or "?"), 0) + 1
        detail = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
        return (f"{q}\nANSWER: YES. {len(matched)} of the journal's {len(ids)} events-endpoint "
                f"order id(s) are in GET /portfolio/orders ({detail}).")
    return (f"{q}\nANSWER: NO MATCH. None of the journal's {len(ids)} events-endpoint order id(s) "
            f"is among the {len(all_orders)} orders GET /portfolio/orders returned. Either they are "
            f"not listed there or the listing no longer reaches back to them.")


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
    db = ReadOnlyDB(db_path)
    positions = await db.get_open_live_positions()
    resting = await client.get_orders(status="resting")
    try:
        all_orders, list_error = await client.list_all_orders(), None
    except Exception as e:  # fail closed: no verdict from a partial listing
        all_orders, list_error = None, str(e)
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
    print(resting_report(resting))
    print()
    print(events_endpoint_verdict(all_orders, list_error, load_journal(journal_path)))
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
