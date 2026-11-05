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
        await run_tracking(db, live=True)
    mock_sell.assert_not_called()
    db.add_trade_log.assert_not_called()
    db.update_position_status.assert_not_called()


# ---- non-finite prices on the exit path: that one position is skipped, the run goes on

@pytest.mark.parametrize("cur,stop", [(float("nan"), 0.40), (0.30, float("nan")), (float("inf"), 0.40)])
async def test_should_exit_raises_on_non_finite_price_instead_of_failing_open(cur, stop):
    with pytest.raises(ValueError):
        await should_exit_position(_pos("YES", stop_loss_price=stop), cur, cur, "active")


@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_run_tracking_nan_entry_or_stop_skips_that_position_and_continues(mock_client, mock_sell):
    nan = float("nan")
    bad_entry = _pos("YES", stop_loss_price=None)
    bad_entry.entry_price = nan
    bad_entry.market_id = "BAD-ENTRY"
    bad_stop = _pos("YES", stop_loss_price=nan)
    bad_stop.market_id = "BAD-STOP"
    good = _pos("YES", stop_loss_price=0.40)
    good.market_id = "GOOD"
    seen = []

    async def get_market(ticker):
        seen.append(ticker)
        return {"market": {"status": "active", "yes_bid_dollars": 0.30, "no_bid_dollars": 0.60}}

    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[bad_entry, bad_stop, good])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    mock_client.return_value.get_market = get_market
    mock_client.return_value.close = AsyncMock()
    mock_sell.return_value = False
    with patch("src.jobs.execute.place_profit_taking_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders", AsyncMock(return_value={"orders_placed": 0})):
        await run_tracking(db, live=False)
    assert seen == ["BAD-ENTRY", "BAD-STOP", "GOOD"]  # loop reached the last position


# ---- issue #49: a $0 exit on a non-resolution path is refused, live or paper ----------
LIVE_BOOK = {"status": "active", "yes_bid_dollars": 0.55, "no_bid_dollars": 0.45}


async def _track(mock_client, market, decision, live):
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[_pos("YES")])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    mock_client.return_value.get_market = AsyncMock(return_value={"market": market})
    mock_client.return_value.close = AsyncMock()
    paper = MagicMock(return_value=True)
    with patch("src.jobs.execute.place_profit_taking_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.log_paper_exit", paper), \
         patch("src.jobs.track.should_exit_position", AsyncMock(return_value=decision)):
        await run_tracking(db, live=live)
    return db, paper


@pytest.mark.parametrize("price", [0.0, -0.01])
@pytest.mark.parametrize("reason", ["time_based", "stop_loss_triggered"])
@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_live_zero_price_non_resolution_exit_is_refused(mock_client, mock_sell, reason, price):
    db, paper = await _track(mock_client, LIVE_BOOK, (True, reason, price), live=True)
    mock_sell.assert_not_called()
    paper.assert_not_called()
    db.add_trade_log.assert_not_called()
    db.update_position_status.assert_not_called()


@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_paper_zero_price_exit_is_refused_too(mock_client, mock_sell):
    db, paper = await _track(mock_client, LIVE_BOOK, (True, "time_based", 0.0), live=False)
    paper.assert_not_called()
    mock_sell.assert_not_called()


@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_paper_priced_exit_is_logged_as_paper_only(mock_client, mock_sell):
    # control: the same path with a real price reaches the paper exit, so the guard above is what stops it
    db, paper = await _track(mock_client, LIVE_BOOK, (True, "time_based", 0.55), live=False)
    paper.assert_called_once()
    mock_sell.assert_not_called()
    db.add_trade_log.assert_not_called()
    db.update_position_status.assert_not_called()


@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_missing_market_data_skips_the_position(mock_client, mock_sell):
    exit_check = AsyncMock(return_value=(True, "time_based", 0.55))
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[_pos("YES")])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    mock_client.return_value.get_market = AsyncMock(return_value={"market": {}})
    mock_client.return_value.close = AsyncMock()
    with patch("src.jobs.execute.place_profit_taking_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.track.should_exit_position", exit_check):
        await run_tracking(db, live=True)
    exit_check.assert_not_awaited()
    mock_sell.assert_not_called()
    db.update_position_status.assert_not_called()


@patch("src.jobs.execute.place_sell_limit_order", new_callable=AsyncMock)
@patch("src.jobs.track.KalshiClient")
async def test_paper_exit_that_fails_leaves_the_position_open(mock_client, mock_sell):
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[_pos("YES")])
    db.add_trade_log = AsyncMock()
    db.update_position_status = AsyncMock()
    mock_client.return_value.get_market = AsyncMock(return_value={"market": LIVE_BOOK})
    mock_client.return_value.close = AsyncMock()
    with patch("src.jobs.execute.place_profit_taking_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.place_stop_loss_orders", AsyncMock(return_value={"orders_placed": 0})), \
         patch("src.jobs.execute.log_paper_exit", MagicMock(return_value=False)) as paper, \
         patch("src.jobs.track.should_exit_position", AsyncMock(return_value=(True, "time_based", 0.55))):
        await run_tracking(db, live=False)
    paper.assert_called_once()
    mock_sell.assert_not_called()
    db.add_trade_log.assert_not_called()
    db.update_position_status.assert_not_called()


@patch("src.jobs.track.KalshiClient")
async def test_a_failed_db_read_still_closes_the_client(mock_client):
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(side_effect=RuntimeError("db down"))
    mock_client.return_value.close = AsyncMock()
    await run_tracking(db, live=False)
    mock_client.return_value.close.assert_awaited_once()
