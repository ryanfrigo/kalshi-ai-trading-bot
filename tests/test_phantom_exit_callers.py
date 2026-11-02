"""Exit sells must reconcile what filled and never sell what we don't hold.

place_profit_taking_orders / place_stop_loss_orders sent sells but never logged
the exit or closed/shrank the position, so a filled sell left a phantom open
row that a later cycle could sell again, opening the other side. Every exit
sell now caps its count at contracts held on the exchange minus our resting
sells (fail closed), and the callers record only what the fill count says sold.

Kalshi is faked (no network); the DB is a temp SQLite file. Prices: entry 40c;
a 60c YES price trips profit-take, 30c trips stop-loss.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.jobs.execute import (place_profit_taking_orders, place_sell_limit_order,
                              place_stop_loss_orders)
from src.utils.database import DatabaseManager, Position

CALLERS = {
    "profit": (place_profit_taking_orders, 60),
    "stop": (place_stop_loss_orders, 30),
}


class FakeKalshi:
    """Exchange state: held contracts and our resting sells; fills move both."""

    def __init__(self, yes_cents=60, held=10.0, resting=None, ioc=False, fill=None,
                 positions_fail=False, orders_fail=False):
        self.yes_cents, self.held, self.ioc, self.fill = yes_cents, held, ioc, fill
        self.resting = list(resting or [])
        self.positions_fail, self.orders_fail = positions_fail, orders_fail
        self.sent = []  # count of every sell actually sent

    async def get_market(self, ticker):
        return {"market": {"yes_price": self.yes_cents, "no_price": 100 - self.yes_cents}}

    async def _held_on_side(self, ticker, side):
        return None if self.positions_fail else self.held

    async def get_orders(self, ticker=None, status=None):
        if self.orders_fail:
            raise RuntimeError("HTTP 503")
        return {"orders": list(self.resting), "cursor": ""}

    async def place_order(self, **kw):
        count = kw["count"]
        self.sent.append(count)
        filled = count if self.fill is None else min(self.fill, count)
        self.held -= filled
        if not self.ioc and count > filled:
            self.resting.append({"ticker": kw["ticker"], "side": kw["side"], "action": "sell",
                                 "remaining_count_fp": f"{count - filled:.2f}"})
        resp = {"order": {"order_id": "o1", "fill_count_fp": f"{filled:.2f}"}}
        if self.ioc:
            resp["reduce_only_ioc"] = True
        return resp


async def _setup(tmp_path, quantity=10):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    pos = Position(market_id="PH-1", side="YES", entry_price=0.40, quantity=quantity,
                   timestamp=datetime.now(), rationale="r", live=True, status="open")
    pos.id = await db.add_position(pos)
    return db, pos


def _run(tmp_path, kind, client, cycles=1):
    fn, _ = CALLERS[kind]
    client.yes_cents = CALLERS[kind][1]

    async def go():
        db, _ = await _setup(tmp_path)
        results = [await fn(db_manager=db, kalshi_client=client) for _ in range(cycles)]
        return results, await db.get_position_by_market_id("PH-1"), await db.get_all_trade_logs()
    return asyncio.run(go())


@pytest.mark.parametrize("kind", CALLERS)
@pytest.mark.parametrize("ioc", [True, False], ids=["ioc", "gtc"])
def test_full_fill_closes_position_and_logs_exit(tmp_path, kind, ioc):
    client = FakeKalshi(ioc=ioc)
    results, after, logs = _run(tmp_path, kind, client)
    assert after is None  # get_position_by_market_id returns open rows only
    assert len(logs) == 1 and logs[0].quantity == pytest.approx(10)
    assert results[0]["orders_placed"] == 1


@pytest.mark.parametrize("kind", CALLERS)
@pytest.mark.parametrize("ioc", [True, False], ids=["ioc", "gtc"])
def test_partial_fill_logs_sold_part_and_shrinks(tmp_path, kind, ioc, capsys):
    client = FakeKalshi(ioc=ioc, fill=4)
    _, after, logs = _run(tmp_path, kind, client)
    assert after.status == "open" and after.quantity == pytest.approx(6)
    assert len(logs) == 1 and logs[0].quantity == pytest.approx(4)
    assert logs[0].pnl == pytest.approx((logs[0].exit_price - 0.40) * 4)
    assert "Failed to place" not in capsys.readouterr().out


@pytest.mark.parametrize("kind", CALLERS)
def test_full_fill_is_not_sold_again_next_cycle(tmp_path, kind):
    client = FakeKalshi(ioc=True)
    _run(tmp_path, kind, client, cycles=2)
    assert client.sent == [10]


@pytest.mark.parametrize("kind", CALLERS)
def test_resting_gtc_is_not_recorded_as_sold_or_sold_twice(tmp_path, kind):
    client = FakeKalshi(fill=0)
    _, after, logs = _run(tmp_path, kind, client, cycles=2)
    assert after.status == "open" and after.quantity == 10
    assert logs == []
    assert client.sent == [10]


@pytest.mark.parametrize("kind", CALLERS)
def test_gtc_partial_remainder_resting_is_not_resold(tmp_path, kind):
    client = FakeKalshi(fill=4)
    _run(tmp_path, kind, client, cycles=2)
    assert client.sent == [10]  # 4 filled, 6 rest; held 6 - resting 6 = nothing to sell


# --- the sellable guard, at place_sell_limit_order (every exit path uses it) ---

def _sell(tmp_path, client):
    async def go():
        db, pos = await _setup(tmp_path)
        order = {}
        ok = await place_sell_limit_order(position=pos, limit_price=0.30, db_manager=db,
                                          kalshi_client=client, order_out=order)
        return ok, order
    return asyncio.run(go())


def test_db_above_held_caps_sell_at_held(tmp_path):
    client = FakeKalshi(held=6.0, fill=0)
    _sell(tmp_path, client)
    assert client.sent == [6]


def test_resting_sell_on_book_blocks_a_second_sell(tmp_path):
    client = FakeKalshi(resting=[{"ticker": "PH-1", "side": "yes", "action": "sell",
                                  "remaining_count_fp": "10.00"}])
    ok, order = _sell(tmp_path, client)
    assert ok is False and order == {} and client.sent == []


def test_resting_orders_on_other_side_or_buys_do_not_block(tmp_path):
    client = FakeKalshi(fill=0, resting=[
        {"ticker": "PH-1", "side": "no", "action": "sell", "remaining_count_fp": "5.00"},
        {"ticker": "PH-1", "side": "yes", "action": "buy", "remaining_count_fp": "5.00"}])
    _sell(tmp_path, client)
    assert client.sent == [10]


@pytest.mark.parametrize("client", [
    FakeKalshi(positions_fail=True),
    FakeKalshi(orders_fail=True),
    FakeKalshi(held=0.0),
    FakeKalshi(resting=[{"ticker": "PH-1", "side": "yes", "action": "sell"}]),  # no count
    FakeKalshi(resting=[{"ticker": "PH-1", "remaining_count_fp": "1.00"}]),     # no side
], ids=["positions-read-fails", "orders-read-fails", "nothing-held",
        "unreadable-remaining", "unreadable-side"])
def test_unconfirmable_exchange_state_sends_nothing(tmp_path, client):
    ok, order = _sell(tmp_path, client)
    assert ok is False and client.sent == []


def test_paginated_resting_orders_fail_closed(tmp_path):
    client = FakeKalshi()

    async def paged(ticker=None, status=None):
        return {"orders": [], "cursor": "next"}
    client.get_orders = paged
    ok, _ = _sell(tmp_path, client)
    assert ok is False and client.sent == []


# --- track.py close loop: no close on mere acceptance ---

def _track(tmp_path, client):
    from src.jobs.track import run_tracking
    client.get_market = AsyncMock(return_value={"market": {
        "status": "active", "result": None, "yes_bid_dollars": 0.65, "no_bid_dollars": 0.35}})
    client.close = AsyncMock()

    async def go():
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.initialize()
        pos = Position(market_id="PH-1", side="YES", entry_price=0.40, quantity=10,
                       timestamp=datetime.now(), rationale="r", live=True, status="open",
                       take_profit_price=0.60, stop_loss_price=0.30)
        await db.add_position(pos)
        with patch("src.jobs.track.KalshiClient", return_value=client):
            await run_tracking(db_manager=db)
        return await db.get_position_by_market_id("PH-1"), await db.get_all_trade_logs()
    return asyncio.run(go())


def test_close_loop_resting_sell_keeps_position_open(tmp_path):
    client = FakeKalshi(fill=0)
    after, logs = _track(tmp_path, client)
    assert after.status == "open" and after.quantity == 10 and logs == []
    assert client.sent == [10]


def test_close_loop_filled_sell_closes_position(tmp_path):
    client = FakeKalshi()
    after, logs = _track(tmp_path, client)
    assert after is None
    assert len(logs) == 1 and logs[0].quantity == pytest.approx(10)
