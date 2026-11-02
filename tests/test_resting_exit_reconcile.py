"""A GTC exit sell that rests and fills in a later cycle is booked by its fill count.

The exit's order id is stored on the position row when it rests; each
run_tracking cycle reads that order (GET /portfolio/orders/{id}) and logs and
closes or shrinks the row by what newly filled. Read problems change nothing.
Kalshi is faked (no network); the DB is a temp SQLite file.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from src.jobs.execute import (place_profit_taking_orders, place_sell_limit_order,
                              reconcile_resting_exit)
from src.utils.database import DatabaseManager, Position
from tests.test_phantom_exit_callers import FakeKalshi


def _order(fill, status, ticker="RX-1", price="0.6000", oid="o1"):
    """GET /portfolio/orders/{order_id} as documented: _fp counts, _dollars prices."""
    return {"order": {"order_id": oid, "ticker": ticker, "status": status,
                      "outcome_side": "no", "book_side": "ask",
                      "fill_count_fp": f"{fill:.2f}", "remaining_count_fp": "0.00",
                      "yes_price_dollars": price, "no_price_dollars": "0.4000"}}


async def _db(tmp_path, quantity=10, exit_order=("o1", 0.0)):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    pos = Position(market_id="RX-1", side="YES", entry_price=0.40, quantity=quantity,
                   timestamp=datetime.now(), rationale="r", live=True, status="open")
    pos.id = await db.add_position(pos)
    if exit_order:
        await db.set_exit_order(pos.id, *exit_order)
    return db


async def _row(db):
    async with aiosqlite.connect(db.db_path) as conn:
        cur = await conn.execute("SELECT status, quantity, exit_order_id, exit_order_filled "
                                 "FROM positions WHERE market_id = 'RX-1'")
        return await cur.fetchone()


def _reconcile(tmp_path, get_order, **setup):
    async def go():
        db = await _db(tmp_path, **setup)
        client = AsyncMock()
        client.get_order = get_order
        (pos,) = await db.get_open_live_positions()
        result = await reconcile_resting_exit(pos, client, db)
        return result, await _row(db), await db.get_all_trade_logs()
    return asyncio.run(go())


def test_late_full_fill_closes_with_no_settlement_log(tmp_path):
    from src.jobs.track import run_tracking
    client = AsyncMock()
    client.get_order = AsyncMock(return_value=_order(10, "executed"))
    # the market has since settled YES: settlement must not log these contracts again
    client.get_market = AsyncMock(return_value={"market": {"status": "settled", "result": "yes"}})

    async def go():
        db = await _db(tmp_path)
        with patch("src.jobs.track.KalshiClient", return_value=client):
            await run_tracking(db_manager=db)
        return await _row(db), await db.get_all_trade_logs()
    row, logs = asyncio.run(go())
    assert row == ("closed", 10, None, None)
    assert len(logs) == 1
    assert logs[0].quantity == 10 and logs[0].exit_price == pytest.approx(0.60)
    assert logs[0].pnl == pytest.approx(2.0)
    client.get_order.assert_awaited_once_with("o1")


def test_late_partial_fill_shrinks_and_keeps_tracking(tmp_path):
    result, row, logs = _reconcile(tmp_path, AsyncMock(return_value=_order(4, "resting")))
    assert result == "shrunk"
    assert row == ("open", 6, "o1", 4.0)
    assert len(logs) == 1 and logs[0].quantity == 4


def test_fills_already_recorded_are_not_booked_twice(tmp_path):
    # placement booked 4 of 10 and left 6; the order has now filled all 10
    result, row, logs = _reconcile(tmp_path, AsyncMock(return_value=_order(10, "executed")),
                                   quantity=6, exit_order=("o1", 4.0))
    assert result == "closed" and row[0] == "closed"
    assert len(logs) == 1 and logs[0].quantity == 6


def test_no_new_fill_on_a_resting_order_changes_nothing(tmp_path):
    result, row, logs = _reconcile(tmp_path, AsyncMock(return_value=_order(4, "resting")),
                                   quantity=6, exit_order=("o1", 4.0))
    assert result == "unchanged" and row == ("open", 6, "o1", 4.0) and logs == []


def test_cancelled_exit_clears_the_id_so_a_new_exit_goes_out(tmp_path):
    async def go():
        db = await _db(tmp_path)
        client = FakeKalshi(fill=0)
        client.get_order = AsyncMock(return_value=_order(0, "canceled"))
        (pos,) = await db.get_open_live_positions()
        blocked = await place_sell_limit_order(pos, 0.30, db, client)
        sent_while_tracked = list(client.sent)
        result = await reconcile_resting_exit(pos, client, db)
        (pos,) = await db.get_open_live_positions()
        ok = await place_sell_limit_order(pos, 0.30, db, client)
        return blocked, sent_while_tracked, result, ok, client.sent, await db.get_all_trade_logs()
    blocked, sent_while_tracked, result, ok, sent, logs = asyncio.run(go())
    assert blocked is False and sent_while_tracked == []
    assert result == "cleared" and logs == []
    assert ok is True and sent == [10]


@pytest.mark.parametrize("get_order", [
    AsyncMock(side_effect=RuntimeError("HTTP 503")),
    AsyncMock(return_value={}),
    AsyncMock(return_value={"order": {"ticker": "RX-1", "status": "executed"}}),  # no counts
    AsyncMock(return_value=_order(10, "executed", ticker="OTHER-1")),
    AsyncMock(return_value=_order(10, "pending")),
    AsyncMock(return_value=_order(10, "executed", price="junk")),
    AsyncMock(return_value=_order(float("nan"), "executed")),
], ids=["raises", "no-order", "no-fields", "wrong-ticker", "unknown-status", "bad-price",
        "nan-fill"])
def test_unreadable_order_changes_nothing(tmp_path, get_order):
    result, row, logs = _reconcile(tmp_path, get_order)
    assert result == "unchanged"
    assert row == ("open", 10, "o1", 0.0) and logs == []


def test_resting_exit_stores_its_order_id(tmp_path):
    async def go():
        db = await _db(tmp_path, exit_order=None)
        await place_profit_taking_orders(db, FakeKalshi(fill=0, yes_cents=60))
        return await _row(db)
    assert asyncio.run(go()) == ("open", 10, "o1", 0.0)


def test_partly_filled_exit_stores_id_and_recorded_fill(tmp_path):
    async def go():
        db = await _db(tmp_path, exit_order=None)
        await place_profit_taking_orders(db, FakeKalshi(fill=4, yes_cents=60))
        return await _row(db)
    assert asyncio.run(go()) == ("open", 6, "o1", 4.0)


def test_old_db_without_exit_columns_migrates(tmp_path):
    path = str(tmp_path / "old.db")

    async def go():
        async with aiosqlite.connect(path) as conn:  # positions as created before this change
            await conn.execute("""CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT NOT NULL,
                side TEXT NOT NULL, entry_price REAL NOT NULL, quantity INTEGER NOT NULL,
                timestamp TEXT NOT NULL, rationale TEXT, confidence REAL,
                live BOOLEAN NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'open',
                strategy TEXT, stop_loss_price REAL, take_profit_price REAL,
                max_hold_hours INTEGER, target_confidence_change REAL,
                UNIQUE(market_id, side))""")
            await conn.execute("INSERT INTO positions (market_id, side, entry_price, quantity, "
                               "timestamp, live, status) VALUES ('RX-1', 'YES', 0.4, 10, ?, 1, "
                               "'open')", (datetime.now().isoformat(),))
            await conn.commit()
        db = DatabaseManager(db_path=path)
        await db.initialize()
        await db.initialize()  # idempotent
        (pos,) = await db.get_open_live_positions()
        await db.set_exit_order(pos.id, "o9", 0.0)
        return pos, await _row(db)
    pos, row = asyncio.run(go())
    assert pos.exit_order_id is None and pos.quantity == 10
    assert row == ("open", 10, "o9", 0.0)
