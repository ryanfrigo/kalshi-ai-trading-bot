"""Exit sells must reconcile what filled and never sell what we don't hold.

place_profit_taking_orders / place_stop_loss_orders sent sells but never logged
the exit or closed/shrank the position, so a filled sell left a phantom open
row that a later cycle could sell again, opening the other side. Every exit
sell now caps its count at contracts held on the exchange minus our resting
sells (fail closed), and the callers record only what the fill count says sold.

Kalshi is faked (no network); the DB is a temp SQLite file. Prices: entry 40c;
a 60c YES bid trips profit-take, 30c trips stop-loss.
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


def _rest_sell(ticker, side, n):
    """A resting sell as GET /portfolio/orders documents it: no side/action, _fp counts.

    https://docs.kalshi.com/api-reference/orders/get-orders - sell-yes is
    outcome_side 'no' / book_side 'ask'; sell-no is 'yes' / 'bid'.
    """
    outcome = "no" if side == "yes" else "yes"
    return {"order_id": "r1", "ticker": ticker, "outcome_side": outcome,
            "book_side": "bid" if outcome == "yes" else "ask",
            "remaining_count_fp": f"{n:.2f}", "status": "resting"}


class FakeKalshi:
    """Exchange state: held contracts and our resting sells; fills move both."""

    def __init__(self, yes_cents=60, held=10.0, resting=None, ioc=False, fill=None,
                 positions_fail=False, orders_fail=False):
        self.yes_cents, self.held, self.ioc, self.fill = yes_cents, held, ioc, fill
        self.resting = list(resting or [])
        self.positions_fail, self.orders_fail = positions_fail, orders_fail
        self.sent = []  # count of every sell actually sent

    async def get_market(self, ticker):
        return {"market": {"yes_bid_dollars": f"{self.yes_cents / 100:.4f}",
                           "no_bid_dollars": f"{(100 - self.yes_cents) / 100:.4f}"}}

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
            self.resting.append(_rest_sell(kw["ticker"], kw["side"], count - filled))
        resp = {"order": {"order_id": "o1", "fill_count_fp": f"{filled:.2f}"}}
        if self.ioc:
            resp["reduce_only_ioc"] = True
        return resp


def _logged(capsys, caplog):
    """Log text from either sink: structlog prints or goes via stdlib logging,
    depending on which test configured it first."""
    return capsys.readouterr().out + caplog.text


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
def test_partial_fill_logs_sold_part_and_shrinks(tmp_path, kind, ioc, capsys, caplog):
    client = FakeKalshi(ioc=ioc, fill=4)
    _, after, logs = _run(tmp_path, kind, client)
    assert after.status == "open" and after.quantity == pytest.approx(6)
    assert len(logs) == 1 and logs[0].quantity == pytest.approx(4)
    assert logs[0].pnl == pytest.approx((logs[0].exit_price - 0.40) * 4)
    out = _logged(capsys, caplog)
    assert "filled 4.0 of 10" in out  # warning-level: proves capture, so the next check bites
    assert "Failed to place" not in out


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


@pytest.mark.parametrize("rest", [
    _rest_sell("PH-1", "yes", 10),                                           # documented
    {"ticker": "PH-1", "book_side": "ask", "remaining_count_fp": "10.00"},   # book_side only
    {"ticker": "PH-1", "side": "yes", "action": "sell", "remaining_count": 10},  # legacy
], ids=["outcome_side", "book_side", "legacy-side-action"])
def test_resting_sell_on_book_blocks_a_second_sell(tmp_path, rest):
    client = FakeKalshi(resting=[rest])
    ok, order = _sell(tmp_path, client)
    assert ok is False and order == {} and client.sent == []


def test_resting_buy_no_counts_against_a_yes_holding(tmp_path):
    # buy-no is outcome_side 'no' like sell-yes; Kalshi nets it against our YES
    client = FakeKalshi(fill=0, resting=[{"ticker": "PH-1", "outcome_side": "no",
                                          "remaining_count_fp": "4.00"}])
    _sell(tmp_path, client)
    assert client.sent == [6]


def test_resting_orders_for_our_own_outcome_do_not_block(tmp_path):
    # buy-yes / sell-no are positioned for YES: they add to, never sell, our YES
    client = FakeKalshi(fill=0, resting=[
        _rest_sell("PH-1", "no", 5),
        {"ticker": "PH-1", "outcome_side": "yes", "book_side": "bid",
         "remaining_count_fp": "5.00"},
        _rest_sell("OTHER-1", "yes", 10)])
    _sell(tmp_path, client)
    assert client.sent == [10]


def test_unreadable_resting_order_logs_the_missing_fields(tmp_path, capsys, caplog):
    client = FakeKalshi(resting=[{"order_id": "r9", "ticker": "PH-1",
                                  "remaining_count_fp": "1.00"}])
    _sell(tmp_path, client)
    out = _logged(capsys, caplog)
    assert client.sent == []
    assert "Exit blocked on PH-1" in out and "r9" in out and "outcome_side" in out


@pytest.mark.parametrize("client", [
    FakeKalshi(positions_fail=True),
    FakeKalshi(orders_fail=True),
    FakeKalshi(held=0.0),
    FakeKalshi(resting=[{"ticker": "PH-1", "outcome_side": "no"}]),             # no count
    FakeKalshi(resting=[{"ticker": "PH-1", "remaining_count_fp": "1.00"}]),     # no side
], ids=["positions-read-fails", "orders-read-fails", "nothing-held",
        "unreadable-remaining", "unreadable-side"])
def test_unconfirmable_exchange_state_sends_nothing(tmp_path, client):
    ok, order = _sell(tmp_path, client)
    assert ok is False and client.sent == []


def test_full_page_with_cursor_fails_closed(tmp_path):
    client = FakeKalshi()

    async def paged(ticker=None, status=None):
        return {"orders": [_rest_sell("PH-1", "no", 1)] * 100, "cursor": "next"}
    client.get_orders = paged
    ok, _ = _sell(tmp_path, client)
    assert ok is False and client.sent == []


def test_short_page_with_cursor_is_complete(tmp_path):
    # cursor is a required field in the response; a page under the 100 default is the last
    client = FakeKalshi(fill=0)

    async def last_page(ticker=None, status=None):
        return {"orders": [], "cursor": "abc"}
    client.get_orders = last_page
    _sell(tmp_path, client)
    assert client.sent == [10]


@pytest.mark.parametrize("ioc,reason", [(True, "partial IOC fill under risk halt"),
                                        (False, "partial fill (profit_take)")],
                         ids=["halt-ioc", "normal-gtc"])
def test_partial_exit_rationale_names_the_real_reason(tmp_path, ioc, reason):
    _, _, logs = _run(tmp_path, "profit", FakeKalshi(ioc=ioc, fill=4))
    assert logs[0].rationale.endswith(f"EXIT: {reason}")


# --- track.py close loop: no close on mere acceptance ---

def _track(tmp_path, client, yes_bid=0.46):
    from src.jobs.track import run_tracking
    # default +15%: hits take_profit_price 0.45 but not run_tracking's 20%
    # profit-taking job, so the sell comes from the close loop under test.
    client.get_market = AsyncMock(return_value={"market": {
        "status": "active", "result": None, "yes_bid_dollars": yes_bid,
        "no_bid_dollars": round(1 - yes_bid, 2)}})
    client.close = AsyncMock()

    async def go():
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.initialize()
        pos = Position(market_id="PH-1", side="YES", entry_price=0.40, quantity=10,
                       timestamp=datetime.now(), rationale="r", live=True, status="open",
                       take_profit_price=0.45, stop_loss_price=0.30)
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


def test_profit_job_and_close_loop_in_one_run_sell_once(tmp_path):
    # +62% trips both the profit-taking job and take_profit_price; the job's
    # sell rests, so the close loop must find nothing sellable.
    client = FakeKalshi(fill=0)
    after, logs = _track(tmp_path, client, yes_bid=0.65)
    assert client.sent == [10]
    assert after.status == "open" and logs == []
