"""An exit sell whose POST outcome is unknown is tracked by its client_order_id.

place_order raises KalshiOrderOutcomeUnknown when the POST may have landed but
the order could not be found. The exit's client_order_id is then stored as a
pending marker in exit_order_id, so new exits and settlement stay blocked
exactly as for an unread exit order. Each pass looks it up by client_order_id:
found -> its order_id is stored and reconciled as usual; absent after a full
lookup -> cleared; lookup failure -> kept. Kalshi is faked; the DB is temp SQLite.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from src.clients.kalshi_client import KalshiAPIError, KalshiOrderOutcomeUnknown
from src.jobs.execute import (PENDING_EXIT_PREFIX, place_sell_limit_order,
                              reconcile_resting_exit)
from src.utils.database import DatabaseManager, Position


class Exchange:
    """Holds 10 YES on UX-1. place_order raises unknown; reads are scripted."""

    def __init__(self, pages=None, history=None, orders_fail=False, history_fail=False,
                 place_error=None):
        self.place_error = place_error
        self.pages = pages if pages is not None else [{"orders": [], "cursor": ""}]
        self.history = history if history is not None else [{"orders": [], "cursor": ""}]
        self.orders_fail, self.history_fail = orders_fail, history_fail
        self.sent, self.order_reads, self.history_reads = [], [], []
        self.get_order = AsyncMock(side_effect=AssertionError("get_order not expected"))

    async def _held_on_side(self, ticker, side):
        return 10.0

    async def get_market(self, ticker):
        return {"market": {"status": "finalized", "result": "yes"}}

    async def close(self):
        pass

    async def get_orders(self, ticker=None, status=None, limit=None, cursor=None):
        if status == "resting":  # the sellable guard
            return {"orders": [], "cursor": ""}
        self.order_reads.append((ticker, cursor))
        if self.orders_fail:
            raise RuntimeError("HTTP 503")
        return self.pages[int(cursor or 0)]

    async def get_historical_orders(self, ticker, limit=1000, cursor=None):
        self.history_reads.append((ticker, cursor))
        if self.history_fail:
            raise RuntimeError("HTTP 503")
        return self.history[int(cursor or 0)]

    async def place_order(self, **kw):
        self.sent.append(kw)
        if self.place_error:
            raise self.place_error
        raise KalshiOrderOutcomeUnknown(f"order client_order_id={kw['client_order_id']} "
                                        f"outcome unknown; not resent")


def _found(coid, oid="real-1", **extra):
    return {"order_id": oid, "client_order_id": coid, "ticker": "UX-1", **extra}


def _get_order(fill, status, oid="real-1"):
    return {"order": {"order_id": oid, "ticker": "UX-1", "status": status,
                      "fill_count_fp": f"{fill:.2f}", "yes_price_dollars": "0.6000"}}


async def _db(tmp_path, exit_order_id=None, read_failures=0):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    pos = Position(market_id="UX-1", side="YES", entry_price=0.40, quantity=10,
                   timestamp=datetime.now(), rationale="r", live=True, status="open")
    pos.id = await db.add_position(pos)
    if exit_order_id:
        await db.set_exit_order(pos.id, exit_order_id, None, read_failures)
    return db


async def _row(db):
    async with aiosqlite.connect(db.db_path) as conn:
        cur = await conn.execute("SELECT status, quantity, exit_order_id, exit_order_filled, "
                                 "exit_order_read_failures FROM positions WHERE market_id='UX-1'")
        return await cur.fetchone()


def test_unknown_exit_is_stored_pending_and_blocks_a_second_exit(tmp_path):
    async def go():
        db = await _db(tmp_path)
        ex = Exchange()
        (pos,) = await db.get_open_live_positions()
        first = await place_sell_limit_order(pos, 0.60, db, ex, live=True)
        (pos,) = await db.get_open_live_positions()
        second = await place_sell_limit_order(pos, 0.60, db, ex, live=True)
        return first, second, ex, await _row(db)
    first, second, ex, row = asyncio.run(go())
    assert first is False and second is False
    assert len(ex.sent) == 1  # never a second sell while the first is unresolved
    assert row[2] == PENDING_EXIT_PREFIX + ex.sent[0]["client_order_id"]
    assert row[0] == "open" and row[1] == 10
    ex.get_order.assert_not_awaited()  # the pending marker is never read as an order id


@pytest.mark.parametrize("pending", [True, False])
def test_pending_exit_blocks_settlement_while_lookup_fails(tmp_path, pending):
    from src.jobs.track import run_tracking
    ex = Exchange(orders_fail=True)

    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + "c-1" if pending else None)
        with patch("src.jobs.track.KalshiClient", return_value=ex):
            await run_tracking(db_manager=db, live=True)
        return await _row(db), await db.get_all_trade_logs()
    row, logs = asyncio.run(go())
    assert ex.sent == []
    if pending:
        assert row[:3] == ("open", 10, PENDING_EXIT_PREFIX + "c-1")
        assert logs == []
    else:  # control: the same market does settle a row with no pending exit
        assert row[0] == "closed" and len(logs) == 1


def _reconcile(tmp_path, ex, coid="c-1"):
    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + coid)
        (pos,) = await db.get_open_live_positions()
        result = await reconcile_resting_exit(pos, ex, db)
        return result, await _row(db), await db.get_all_trade_logs()
    return asyncio.run(go())


def test_found_exit_stores_order_id_and_books_its_fills(tmp_path):
    ex = Exchange(pages=[{"orders": [_found("other")], "cursor": "1"},
                         {"orders": [_found("c-1")], "cursor": ""}])
    ex.get_order = AsyncMock(return_value=_get_order(10, "executed"))
    result, row, logs = _reconcile(tmp_path, ex)
    assert result == "closed"
    assert row[:4] == ("closed", 10, None, None)
    assert len(logs) == 1 and logs[0].quantity == 10
    assert logs[0].exit_price == pytest.approx(0.60)
    ex.get_order.assert_awaited_once_with("real-1")
    assert ex.order_reads == [("UX-1", None), ("UX-1", "1")]


def test_found_resting_exit_is_tracked_by_its_order_id(tmp_path):
    ex = Exchange(pages=[{"orders": [_found("c-1")], "cursor": ""}])
    ex.get_order = AsyncMock(return_value=_get_order(0, "resting"))
    result, row, logs = _reconcile(tmp_path, ex)
    assert result == "unchanged"
    assert row[:4] == ("open", 10, "real-1", 0.0)
    assert logs == []


def test_exit_found_only_in_historical_orders_is_tracked(tmp_path):
    ex = Exchange(history=[{"orders": [_found("c-1")], "cursor": ""}])
    ex.get_order = AsyncMock(return_value=_get_order(4, "canceled"))
    result, row, logs = _reconcile(tmp_path, ex)
    assert result == "shrunk"
    assert row[:4] == ("open", 6, None, None)
    assert len(logs) == 1 and logs[0].quantity == 4


def test_exit_absent_after_full_lookup_clears_pending_and_allows_a_new_exit(tmp_path):
    ex = Exchange(pages=[{"orders": [_found("other")], "cursor": "1"},
                         {"orders": [], "cursor": ""}],
                  history=[{"orders": [], "cursor": "1"}, {"orders": [], "cursor": ""}])

    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + "c-1")
        (pos,) = await db.get_open_live_positions()
        result = await reconcile_resting_exit(pos, ex, db)
        row = await _row(db)
        (pos,) = await db.get_open_live_positions()
        await place_sell_limit_order(pos, 0.60, db, ex, live=True)
        return result, row, await db.get_all_trade_logs()
    result, row, logs = asyncio.run(go())
    assert result == "cleared"
    assert row[:4] == ("open", 10, None, None)
    assert logs == []
    assert len(ex.sent) == 1  # the new exit went out
    assert ex.history_reads == [("UX-1", None), ("UX-1", "1")]


@pytest.mark.parametrize("ex", [
    Exchange(orders_fail=True),
    Exchange(history_fail=True),
    # current orders never stop paging: not a full lookup
    Exchange(pages=[{"orders": [], "cursor": str(i + 1)} for i in range(50)]),
    Exchange(history=[{"orders": [], "cursor": str(i + 1)} for i in range(50)]),
    # found, but without an order_id
    Exchange(pages=[{"orders": [{"client_order_id": "c-1"}], "cursor": ""}]),
], ids=["orders-fail", "history-fail", "orders-truncated", "history-truncated", "no-oid"])
def test_lookup_failure_keeps_pending_and_counts_a_failure(tmp_path, ex):
    result, row, logs = _reconcile(tmp_path, ex)
    assert result == "unchanged"
    assert row == ("open", 10, PENDING_EXIT_PREFIX + "c-1", None, 1)
    assert logs == []
    ex.get_order.assert_not_awaited()


@pytest.mark.parametrize("status,row_after", [
    ("canceled", ("open", 6, None, None)),
    ("resting", ("open", 6, "real-1", 4.0)),
])
def test_fills_already_on_the_found_order_are_booked_once(tmp_path, status, row_after):
    # The found order already shows 4 filled; none of it was booked when the POST
    # went unknown, so all 4 are booked now, and never again on a later pass.
    ex = Exchange(pages=[{"orders": [_found("c-1", fill_count_fp="4.00")], "cursor": ""}])
    ex.get_order = AsyncMock(return_value=_get_order(4, status))

    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + "c-1")
        (pos,) = await db.get_open_live_positions()
        first = await reconcile_resting_exit(pos, ex, db)
        row1, logs1 = await _row(db), await db.get_all_trade_logs()
        (pos,) = await db.get_open_live_positions()
        second = None
        if pos.exit_order_id:  # track's Step 0 reconciles only a stored exit
            second = await reconcile_resting_exit(pos, ex, db)
        return first, row1, logs1, second, await _row(db), await db.get_all_trade_logs()
    first, row1, logs1, second, row2, logs2 = asyncio.run(go())
    assert first == "shrunk"
    assert row1[:4] == row_after
    assert len(logs1) == 1 and logs1[0].quantity == 4
    assert second in (None, "unchanged")
    assert row2[:4] == row_after and len(logs2) == 1


def test_plain_api_error_on_exit_stores_no_marker(tmp_path):
    async def go():
        db = await _db(tmp_path)
        ex = Exchange(place_error=KalshiAPIError("HTTP 400: bad request", status_code=400))
        (pos,) = await db.get_open_live_positions()
        ok = await place_sell_limit_order(pos, 0.60, db, ex, live=True)
        return ok, ex, await _row(db)
    ok, ex, row = asyncio.run(go())
    assert ok is False and len(ex.sent) == 1
    assert row[2] is None and row[3] is None


def test_resolving_the_marker_resets_read_failures(tmp_path):
    ex = Exchange(pages=[{"orders": [_found("c-1")], "cursor": ""}])
    ex.get_order = AsyncMock(return_value=_get_order(0, "resting"))

    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + "c-1", read_failures=2)
        (pos,) = await db.get_open_live_positions()
        await reconcile_resting_exit(pos, ex, db)
        resolved = await _row(db)
        # the next read of the real order fails: the count starts again at 1, not 3
        ex.get_order = AsyncMock(side_effect=RuntimeError("HTTP 503"))
        (pos,) = await db.get_open_live_positions()
        await reconcile_resting_exit(pos, ex, db)
        return resolved, await _row(db)
    resolved, after_fail = asyncio.run(go())
    assert resolved == ("open", 10, "real-1", 0.0, 0)
    assert after_fail == ("open", 10, "real-1", 0.0, 1)


def test_read_failure_right_after_resolving_counts_from_zero(tmp_path):
    # resolved, then the real order's first read (same pass) fails: 1 failure, not 3
    ex = Exchange(pages=[{"orders": [_found("c-1")], "cursor": ""}])
    ex.get_order = AsyncMock(side_effect=RuntimeError("HTTP 503"))

    async def go():
        db = await _db(tmp_path, exit_order_id=PENDING_EXIT_PREFIX + "c-1", read_failures=2)
        (pos,) = await db.get_open_live_positions()
        result = await reconcile_resting_exit(pos, ex, db)
        return result, await _row(db), await db.get_all_trade_logs()
    result, row, logs = asyncio.run(go())
    assert result == "unchanged" and logs == []
    assert row == ("open", 10, "real-1", 0.0, 1)
