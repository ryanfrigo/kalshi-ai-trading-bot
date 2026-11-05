"""The stop-loss job holds instead of selling into a gutted or wide book.

It skips the stop when the held side's bid is under max(5c, 25% of entry) or
its spread is over 10c; a missing ask means the spread is unknown, so it skips.
Only place_stop_loss_orders applies this. Kalshi is mocked; nothing is sent.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.jobs.execute import place_stop_loss_orders
from src.utils.database import Position
from tests.test_phantom_exit_callers import _logged

pytestmark = pytest.mark.asyncio


def _client(bid, ask):
    """A NO holding's market: NO bid/ask in cents; ask None = no ask on the book."""
    m = {"yes_bid_dollars": "0.0000", "no_bid_dollars": f"{bid / 100:.4f}"}
    if ask is not None:
        m["no_ask_dollars"] = f"{ask / 100:.4f}"
    c = MagicMock()
    c.get_market = AsyncMock(return_value={"market": m})
    c._held_on_side = AsyncMock(return_value=10.0)
    c.get_orders = AsyncMock(return_value={"orders": [], "cursor": ""})
    c.place_order = AsyncMock(return_value={"order": {"order_id": "o1",
                                                      "fill_count_fp": "10.00"}})
    return c


def _db(entry):
    d = MagicMock()
    d.get_open_live_positions = AsyncMock(return_value=[Position(
        market_id="FL-1", side="NO", entry_price=entry, quantity=10,
        timestamp=datetime.now(), live=True, status="open", id=7)])
    d.add_trade_log = AsyncMock()
    d.update_position_status = AsyncMock()
    d.set_exit_order = AsyncMock()
    return d


async def _stop(entry, bid, ask):
    # threshold -0.15 with bids well under entry: the stop always triggers here.
    c = _client(bid, ask)
    res = await place_stop_loss_orders(_db(entry), c, stop_loss_threshold=-0.15, live=True)
    return res, c


@pytest.mark.parametrize("entry, bid, ask, sells", [
    (0.60, 15, 16, True),    # floor 25% of 60c = 15c: exactly at it sells
    (0.60, 14, 15, False),   # 1c under skips
    (0.16, 5, 6, True),      # 25% of 16c is 4c, so the 5c minimum applies: at it sells
    (0.16, 4, 5, False),     # 1c under skips
    (0.60, 30, 40, True),    # spread 10c sells
    (0.60, 30, 41, False),   # spread 11c skips
    (0.60, 30, None, False), # no ask: spread unknown, skip
    (0.28, 7, 8, True),      # 0.28*25 = 7.000000000000001: exactly 25% sells
    (0.60, 29, 39, True),    # 0.29*100 = 28.999...: a 10c spread sells
    (0.60, 46, 57, False),   # 0.57*100 = 56.999...: an 11c spread holds
])
async def test_floor_and_spread_boundaries(entry, bid, ask, sells):
    res, c = await _stop(entry, bid, ask)
    assert res["orders_placed"] == (1 if sells else 0)
    if sells:
        assert c.place_order.call_args.kwargs["no_price"] == bid
    else:
        c.place_order.assert_not_called()


async def test_skip_logs_bid_floor_and_spread(capsys, caplog):
    await _stop(0.60, 14, 15)
    await _stop(0.60, 30, None)
    out = _logged(capsys, caplog)
    assert "stop skipped: bid 14c vs floor 15c / spread 1c" in out
    assert "stop skipped: bid 30c vs floor 15c / spread unknown" in out


async def test_paper_mode_skip_sends_nothing():
    c = _client(14, 15)
    res = await place_stop_loss_orders(_db(0.60), c, stop_loss_threshold=-0.15, live=False)
    assert res.get("simulated", 0) == 0
    c.place_order.assert_not_called()


# ---- run_tracking step 2: should_exit_position's stop exits use the same floor
def _track_pos(**kw):
    base = dict(market_id="FL-2", side="YES", entry_price=0.50, quantity=5,
                timestamp=datetime.now(), live=True, status="open", id=8,
                stop_loss_price=0.40, take_profit_price=0.99)
    base.update(kw)
    return Position(**base)


async def _track(pos, bid_c, ask_c):
    from unittest.mock import patch
    from src.jobs.track import run_tracking
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[pos])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    market = {"status": "active", "yes_bid_dollars": f"{bid_c / 100:.4f}",
              "yes_ask_dollars": f"{ask_c / 100:.4f}", "no_bid_dollars": "0.0000"}
    sell = AsyncMock(return_value=False)
    with patch("src.jobs.track.KalshiClient") as client, \
         patch("src.jobs.execute.place_sell_limit_order", sell), \
         patch("src.jobs.execute.place_profit_taking_orders",
               AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders",
               AsyncMock(return_value={"orders_placed": 0})):
        client.return_value.get_market = AsyncMock(return_value={"market": market})
        client.return_value.close = AsyncMock()
        await run_tracking(db, live=True)
    return sell


@pytest.mark.parametrize("pos", [
    _track_pos(),                                         # stop_loss_triggered_pnl_*
    _track_pos(stop_loss_price=None),                     # emergency_stop_loss_10pct
], ids=["stop", "emergency"])
async def test_step2_stop_on_gutted_bid_holds(pos, capsys, caplog):
    # Bid 10c on a 50c entry: under the floor max(5c, 12.5c).
    sell = await _track(pos, 10, 11)
    sell.assert_not_called()
    assert "stop skipped: bid 10c vs floor 12.5c / spread 1c" in _logged(capsys, caplog)


async def test_step2_stop_above_the_floor_sells():
    sell = await _track(_track_pos(), 30, 31)
    sell.assert_awaited_once()
    assert sell.call_args.kwargs["limit_price"] == 0.30


async def test_step2_time_exit_with_untriggered_stop_on_gutted_bid_still_sells():
    from datetime import timedelta
    # Stop at 5c is not hit by a 10c bid; held 2h past a 1h limit: a time exit.
    pos = _track_pos(stop_loss_price=0.05, max_hold_hours=1,
                     timestamp=datetime.now() - timedelta(hours=2))
    sell = await _track(pos, 10, 11)
    sell.assert_awaited_once()
    assert sell.call_args.kwargs["limit_price"] == 0.10


async def test_step2_triggered_stop_under_floor_holds_even_past_max_hold(capsys, caplog):
    from datetime import timedelta
    # Stop at 40c is hit by a 10c bid and the position is 2h past a 1h limit.
    # should_exit_position returns the stop reason before max_hold, so the floor holds it.
    pos = _track_pos(stop_loss_price=0.40, max_hold_hours=1,
                     timestamp=datetime.now() - timedelta(hours=2))
    sell = await _track(pos, 10, 11)
    sell.assert_not_called()
    assert "stop skipped: bid 10c vs floor 12.5c / spread 1c" in _logged(capsys, caplog)
