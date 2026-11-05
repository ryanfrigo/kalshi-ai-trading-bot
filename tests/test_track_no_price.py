"""#97: an empty book (bid 0 / missing / None) must never trigger a stop-loss sell."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.jobs.track import run_tracking, should_exit_position
from src.utils.database import Position

pytestmark = pytest.mark.asyncio


def _pos(side="YES", **kw):
    return Position(
        market_id="NP-1", side=side, entry_price=0.50, quantity=5,
        timestamp=datetime.now(), live=True, status="open",
        stop_loss_price=kw.pop("stop_loss_price", 0.40), **kw,
    )


@pytest.mark.parametrize("price", [0.0, None, -0.01])
async def test_no_price_never_exits_yes(price):
    should_exit, _, exit_price = await should_exit_position(_pos("YES"), price, price, "active")
    assert should_exit is False and exit_price == 0.0


@pytest.mark.parametrize("price", [0.0, None])
async def test_no_price_never_exits_no_side_or_legacy(price):
    no_stop = _pos("NO", stop_loss_price=None, take_profit_price=0.10)
    assert (await should_exit_position(no_stop, price, price, "active"))[0] is False


async def test_real_bid_below_stop_still_triggers():
    should_exit, reason, exit_price = await should_exit_position(_pos("YES"), 0.30, 0.65, "active")
    assert should_exit and reason.startswith("stop_loss_triggered") and exit_price == 0.30


async def test_resolution_with_zero_bid_still_exits():
    should_exit, reason, _ = await should_exit_position(_pos("YES"), 0.0, 0.0, "closed", "YES")
    assert should_exit and reason == "market_resolution"


@pytest.mark.parametrize("market", [
    {"status": "active", "yes_bid_dollars": 0.0, "no_bid_dollars": 0.0},
    {"status": "active"},
])
@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_run_tracking_empty_book_does_not_sell(mock_client, mock_sell, market):
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[_pos("YES")])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    mock_client.return_value.get_market = AsyncMock(return_value={"market": market})
    mock_client.return_value.close = AsyncMock()
    with patch("src.jobs.execute.place_profit_taking_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders", AsyncMock(return_value={"orders_placed": 0})):
        await run_tracking(db)
    mock_sell.assert_not_called()
    db.add_trade_log.assert_not_called()
    db.update_position_status.assert_not_called()
