"""scripts/exit_preview.py is read-only and reports the real run_tracking decisions.

A real KalshiClient talks to an httpx.MockTransport that logs every request, so
"only GET" is checked on the wire, not on a mock of the client.
"""
import json
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from scripts import exit_preview as ep
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Position

pytestmark = pytest.mark.asyncio

MARKETS = {
    # NO held at .40, NO bid .30: stored levels are the old inverted kind; recomputed stop .37 fires
    "T-NO": {"status": "active", "yes_bid_dollars": 0.69, "yes_ask_dollars": 0.71,
             "no_bid_dollars": 0.30, "no_ask_dollars": 0.31},
    # YES held at .40, bid .41: nothing fires
    "T-HOLD": {"status": "active", "yes_bid_dollars": 0.41, "yes_ask_dollars": 0.43,
               "no_bid_dollars": 0.57, "no_ask_dollars": 0.59},
    # settled YES: closes in the DB, no order
    "T-RES": {"status": "closed", "result": "yes", "yes_bid_dollars": 0.0,
              "no_bid_dollars": 0.0},
    # YES held at .40, legacy yes_price 60c (+50%): profit-take sells, then the
    # take_profit exit is blocked because that sell is (simulated) resting
    "T-PT": {"status": "active", "yes_price": 60, "yes_bid_dollars": 0.60,
             "yes_ask_dollars": 0.62, "no_bid_dollars": 0.38, "no_ask_dollars": 0.40},
}
HELD = {"T-NO": "-5.00", "T-HOLD": "3.00", "T-RES": "2.00", "T-PT": "4.00"}
RESTING = [
    {"order_id": "J-1", "ticker": "T-OTHER", "outcome_side": "yes", "book_side": "bid",
     "remaining_count_fp": "2.00"},
    {"order_id": "X-9", "ticker": "T-OTHER2", "side": "yes", "action": "buy"},
]


def _handler(log):
    def handle(request: httpx.Request) -> httpx.Response:
        log.append((request.method, request.url.path))
        path, q = request.url.path, parse_qs(urlparse(str(request.url)).query)
        if path.startswith("/trade-api/v2/markets/"):
            t = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"market": {"ticker": t, **MARKETS[t]}})
        if path == "/trade-api/v2/portfolio/positions":
            t = q["ticker"][0]
            return httpx.Response(200, json={"market_positions": [
                {"ticker": t, "position_fp": HELD[t]}]})
        if path == "/trade-api/v2/portfolio/orders":
            t = q.get("ticker", [None])[0]
            orders = [o for o in RESTING if t is None or o["ticker"] == t]
            return httpx.Response(200, json={"orders": orders, "cursor": ""})
        return httpx.Response(500, json={"error": f"unexpected {request.method} {path}"})
    return handle


@pytest.fixture
def wire():
    log = []

    def factory():
        c = KalshiClient()
        c.client = httpx.AsyncClient(transport=httpx.MockTransport(_handler(log)))
        return c

    with patch.object(KalshiClient, "_load_private_key", lambda self: None), \
         patch.object(KalshiClient, "_sign_request", lambda self, *a: "sig"), \
         patch("src.clients.kalshi_client.asyncio.sleep", AsyncMock()):
        yield log, factory


async def _db(tmp_path):
    path = str(tmp_path / "t.db")
    db = DatabaseManager(db_path=path)
    await db.initialize()
    now = datetime.now()
    for pos in [
        Position("T-NO", "NO", 0.40, 5, now, live=True, confidence=0.7,
                 stop_loss_price=0.43, take_profit_price=0.32),
        Position("T-HOLD", "YES", 0.40, 3, now, live=True, confidence=0.7,
                 stop_loss_price=0.37, take_profit_price=0.48),
        Position("T-RES", "YES", 0.40, 2, now, live=True, stop_loss_price=0.37),
        Position("T-PT", "YES", 0.40, 4, now, live=True, confidence=0.7,
                 stop_loss_price=0.37, take_profit_price=0.48),
    ]:
        await db.add_position(pos)
    return path


def _journal(tmp_path, ids):
    p = tmp_path / "j.jsonl"
    p.write_text("".join(json.dumps({"ticker": "T", "order_id": i, "outcome": None}) + "\n"
                         for i in ids))
    return str(p)


async def test_preview_sends_only_gets_and_never_calls_order_methods(tmp_path, wire, capsys):
    log, factory = wire
    db_path = await _db(tmp_path)
    before = sqlite3.connect(db_path).execute(
        "SELECT id, status, quantity FROM positions ORDER BY id").fetchall()
    with patch.object(KalshiClient, "place_order", AsyncMock()) as place, \
         patch.object(KalshiClient, "cancel_order", AsyncMock()) as cancel:
        rc = await ep.run(db_path, _journal(tmp_path, ["J-1"]), client_factory=factory)
    assert rc == 0
    place.assert_not_called()
    cancel.assert_not_called()
    assert log and all(m == "GET" for m, _ in log), log
    assert not any("/events/orders" in p for _, p in log)
    assert sum(1 for m, p in log if p == "/trade-api/v2/portfolio/orders") >= 1
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT id, status, quantity FROM positions ORDER BY id").fetchall() == before
    assert con.execute("SELECT COUNT(*) FROM trade_logs").fetchone()[0] == 0


async def test_preview_reports_the_real_exit_decisions(tmp_path, wire, capsys):
    _, factory = wire
    rc = await ep.run(await _db(tmp_path), _journal(tmp_path, ["J-1"]), client_factory=factory)
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("T-")}
    assert "SELL" in rows["T-NO"] and "stop_loss_triggered" in rows["T-NO"]
    assert "sell 5 @ 30c" in rows["T-NO"]
    assert "hold" in rows["T-HOLD"] and "SELL" not in rows["T-HOLD"]
    assert "CLOSE" in rows["T-RES"] and "resolution: DB close at 1.00" in rows["T-RES"]
    assert "profit_take: sell 4 @ 58c" in rows["T-PT"]
    assert "take_profit: blocked, nothing sellable" in rows["T-PT"]
    assert "profit/stop skipped (market has no no_price field)" in rows["T-NO"]
    assert "update_position_status" in out  # resolution write reported, not made
    assert rc == 0


async def test_resting_report_fields_and_events_endpoint_answer(tmp_path, wire, capsys):
    _, factory = wire
    await ep.run(await _db(tmp_path), _journal(tmp_path, ["J-1"]), client_factory=factory)
    out = capsys.readouterr().out
    j1 = next(l for l in out.splitlines() if l.startswith("J-1"))
    x9 = next(l for l in out.splitlines() if l.startswith("X-9"))
    assert j1.split()[2:5] == ["yes", "yes", "yes"]
    assert x9.split()[2:5] == ["NO", "NO", "NO"] and "guard blocks" in x9
    assert "ANSWER: YES" in out and "J-1" in out


async def test_events_endpoint_answer_is_inconclusive_without_a_match(tmp_path, wire, capsys):
    _, factory = wire
    await ep.run(await _db(tmp_path), _journal(tmp_path, ["GONE-1"]), client_factory=factory)
    out = capsys.readouterr().out
    assert "ANSWER: NOT SHOWN BY THIS ACCOUNT" in out and "not evidence either way" in out


async def test_proxy_refuses_mutating_methods_and_non_get_on_the_wire(wire):
    log, factory = wire
    client = ep.ReadOnlyClient(factory())
    for name in ("cancel_order", "amend_order", "batch_create_orders", "batch_cancel_orders"):
        with pytest.raises(ep.ReadOnlyViolation):
            getattr(client, name)
    with pytest.raises(ep.ReadOnlyViolation):
        await client._inner._make_authenticated_request("POST", "/trade-api/v2/portfolio/events/orders")
    with pytest.raises(ep.ReadOnlyViolation):  # even bypassing the wrapper, httpx refuses
        await client._inner.client.request("DELETE", "https://x/trade-api/v2/portfolio/events/orders/1")
    resp = await client.place_order(ticker="T", side="no", action="sell", count=1, no_price=30)
    assert resp["order"]["order_id"] == ep.NOT_SENT_ID
    assert log == []
    await client.close()


async def test_missing_db_is_refused_without_creating_it(tmp_path, wire):
    _, factory = wire
    path = tmp_path / "nope.db"
    assert await ep.run(str(path), str(tmp_path / "j"), client_factory=factory) == 2
    assert not path.exists()
