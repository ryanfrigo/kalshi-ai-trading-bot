"""A GTC exit the market moved away from must not lock the position.

Before: the stop-loss sell was priced off entry (entry * 0.835), so after a gap
it rested above the bid. Its id was stored as the position's exit order,
place_sell_limit_order refused every later sell while an id was stored, and
nothing cancelled it: the position rode to settlement.
Now the stop sells at the bid, and a later live exit priced below a stored
resting exit books its fills, cancels it, books any fill that raced the cancel,
and sells. Paper mode never cancels. Kalshi is faked; the DB is a temp SQLite file.
"""
import asyncio
from datetime import datetime

import pytest

from src.jobs.execute import place_sell_limit_order, place_stop_loss_orders
from src.utils.database import DatabaseManager, Position

T = "SX-1"


class FakeKalshi:
    """One NO holding and our stored resting exit "o1"; new sells fill in full."""

    def __init__(self, no_bid_cents=30, held=10.0, o1_cents=50, o1_filled=0.0,
                 cancel_fails=False, race_fill=0.0):
        self.no_bid, self.held = no_bid_cents, held
        self.o1 = {"order_id": "o1", "ticker": T, "status": "resting",
                   "no_price_dollars": f"{o1_cents / 100:.4f}", "filled": o1_filled}
        self.cancel_fails, self.race_fill = cancel_fails, race_fill
        self.calls, self.sent, self.on_cancel = [], [], None

    async def get_market(self, ticker):
        return {"market": {"status": "active", "yes_bid_dollars": "0.0000",
                           "no_bid_dollars": f"{self.no_bid / 100:.4f}"}}

    async def _held_on_side(self, ticker, side):
        return self.held

    async def get_orders(self, ticker=None, status=None):
        o = self.o1
        rest = [] if o["status"] != "resting" else [{
            "order_id": "o1", "ticker": T, "outcome_side": "yes",
            "remaining_count_fp": f"{10 - o['filled']:.2f}"}]
        return {"orders": rest, "cursor": ""}

    async def get_order(self, oid):
        self.calls.append(("get_order", oid))
        o = self.o1
        return {"order": {"order_id": oid, "ticker": T, "status": o["status"],
                          "no_price_dollars": o["no_price_dollars"],
                          "yes_price_dollars": "0.5000",
                          "fill_count_fp": f"{o['filled']:.2f}"}}

    async def cancel_order(self, oid):
        self.calls.append(("cancel_order", oid))
        if self.on_cancel:
            await self.on_cancel()
        if self.cancel_fails:
            raise RuntimeError("HTTP 503")
        self.o1["filled"] += self.race_fill  # filled while the cancel was in flight
        self.held -= self.race_fill
        self.o1["status"] = "canceled"
        return {"order": {"order_id": oid, "status": "canceled"}}

    async def place_order(self, **kw):
        self.calls.append(("place_order", kw.get("no_price")))
        self.sent.append(kw)
        self.held -= kw["count"]
        return {"order": {"order_id": "o2", "fill_count_fp": f"{kw['count']:.2f}",
                          "status": "executed"}}


async def _db(tmp_path, exit_order=("o1", 0.0)):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    pos = Position(market_id=T, side="NO", entry_price=0.60, quantity=10,
                   timestamp=datetime.now(), rationale="r", live=True, status="open")
    pos.id = await db.add_position(pos)
    if exit_order:
        await db.set_exit_order(pos.id, *exit_order)
    return db


def _stop(tmp_path, client, live=True, exit_order=("o1", 0.0)):
    async def go():
        db = await _db(tmp_path, exit_order)
        res = await place_stop_loss_orders(db, client, stop_loss_threshold=-0.15, live=live)
        return res, await db.get_position_by_market_id(T), await db.get_all_trade_logs()
    return asyncio.run(go())


def test_stop_loss_sells_at_the_bid_not_above_it(tmp_path):
    # Entry 60c, NO bid gapped to 40c. Old code sent 50c (entry * 0.835) and rested.
    c = FakeKalshi(no_bid_cents=40)
    c.o1["status"] = "canceled"  # no live resting exit in this case
    _, pos, logs = _stop(tmp_path, c, exit_order=None)
    assert [kw["no_price"] for kw in c.sent] == [40]
    assert pos is None and [(l.quantity, l.exit_price) for l in logs] == [(10, 0.40)]


def test_stop_cancels_a_stored_exit_resting_above_the_bid_and_sells(tmp_path):
    # The stuck case: exit o1 rests at 50c, the NO bid is now 30c.
    c = FakeKalshi(no_bid_cents=30, o1_cents=50)
    res, pos, logs = _stop(tmp_path, c)
    assert ("cancel_order", "o1") in c.calls
    assert c.calls.index(("cancel_order", "o1")) < c.calls.index(("place_order", 30))
    assert [(kw["no_price"], kw["count"]) for kw in c.sent] == [(30, 10)]
    assert res["orders_placed"] == 1
    assert pos is None and [(l.quantity, l.exit_price) for l in logs] == [(10, 0.30)]


def test_fills_are_booked_before_the_cancel_and_a_racing_fill_after(tmp_path):
    # o1 filled 4 at 50c since we last looked; 2 more fill while the cancel is in flight.
    c = FakeKalshi(no_bid_cents=30, o1_cents=50, o1_filled=4.0, race_fill=2.0)
    c.held = 6.0
    seen = {}

    async def snapshot():
        seen["pos"] = await seen["db"].get_position_by_market_id(T)
        seen["logs"] = await seen["db"].get_all_trade_logs()

    c.on_cancel = snapshot

    async def go():
        db = seen["db"] = await _db(tmp_path)
        await place_stop_loss_orders(db, c, stop_loss_threshold=-0.15, live=True)
        return await db.get_position_by_market_id(T), await db.get_all_trade_logs()

    pos, logs = asyncio.run(go())
    # At cancel time the 4 earlier fills were already booked.
    assert seen["pos"].quantity == 6 and seen["pos"].exit_order_id == "o1"
    assert [(l.quantity, l.exit_price) for l in seen["logs"]] == [(4, 0.50)]
    # The racing 2 are booked from the canceled order; the new sell is only the rest.
    assert [(kw["no_price"], kw["count"]) for kw in c.sent] == [(30, 4)]
    assert pos is None
    assert sorted((l.quantity, l.exit_price) for l in logs) == [(2, 0.50), (4, 0.30), (4, 0.50)]


def test_cancel_failure_keeps_the_exit_id_and_sends_no_sell(tmp_path):
    c = FakeKalshi(no_bid_cents=30, o1_cents=50, cancel_fails=True)
    res, pos, logs = _stop(tmp_path, c)
    assert ("cancel_order", "o1") in c.calls
    assert c.sent == [] and res["orders_placed"] == 0
    assert pos.exit_order_id == "o1" and pos.quantity == 10 and logs == []


def test_exit_resting_at_or_below_the_new_price_is_left_alone(tmp_path):
    c = FakeKalshi(no_bid_cents=30, o1_cents=30)
    _, pos, _ = _stop(tmp_path, c)
    assert ("cancel_order", "o1") not in c.calls and c.sent == []
    assert pos.exit_order_id == "o1"


@pytest.mark.parametrize("call", ["stop_loss", "sell_helper"])
def test_paper_mode_never_cancels_or_sends(tmp_path, call):
    c = FakeKalshi(no_bid_cents=30, o1_cents=50)

    async def go():
        db = await _db(tmp_path)
        if call == "stop_loss":
            await place_stop_loss_orders(db, c, stop_loss_threshold=-0.15, live=False)
        else:
            (p,) = await db.get_open_live_positions()
            assert await place_sell_limit_order(p, 0.30, db, c, live=False) is False
        return await db.get_position_by_market_id(T)

    pos = asyncio.run(go())
    assert c.calls == [] and c.sent == []
    assert pos.exit_order_id == "o1" and pos.quantity == 10


def test_cancel_lands_but_response_lost_sends_nothing(tmp_path):
    # The cancel reaches the exchange but its response times out: we can't tell
    # it landed, so the id stays, nothing is booked and no new sell goes out.
    c = FakeKalshi(no_bid_cents=30, o1_cents=50)

    async def lost():
        c.o1["status"] = "canceled"
        raise TimeoutError("read timeout")

    c.on_cancel = lost
    res, pos, logs = _stop(tmp_path, c)
    assert c.sent == [] and pos.exit_order_id == "o1" and logs == []
