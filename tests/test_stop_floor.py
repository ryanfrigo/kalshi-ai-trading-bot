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
    assert "stop skipped: bid 14c under floor 15c / spread 1c" in out
    assert "stop skipped: bid 30c under floor 15c / spread unknown" in out


async def test_paper_mode_skip_sends_nothing():
    c = _client(14, 15)
    res = await place_stop_loss_orders(_db(0.60), c, stop_loss_threshold=-0.15, live=False)
    assert res.get("simulated", 0) == 0
    c.place_order.assert_not_called()
