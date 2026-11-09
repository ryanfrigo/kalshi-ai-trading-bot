from datetime import datetime

import pytest

from src.jobs.track import should_exit_position
from src.utils.database import Position

pytestmark = pytest.mark.asyncio


def _pos():
    return Position(market_id="T-1", side="NO", entry_price=0.9, quantity=10,
                    timestamp=datetime.now(), live=True, status="open", id=1)


@pytest.mark.parametrize("status", ["closed", "determined", "finalized", "FINALIZED", "Determined"])
async def test_settled_statuses_book_resolution(status):
    assert await should_exit_position(_pos(), 0.0, 0.0, status, "no") == (True, "market_resolution", 1.0)
    assert (await should_exit_position(_pos(), 1.0, 0.0, status, "yes"))[2] == 0.0


async def test_active_does_not_resolve():
    assert (await should_exit_position(_pos(), 0.1, 0.9, "active", "no"))[0] is False


async def test_resolution_without_result_or_price_is_safe():
    assert await should_exit_position(_pos(), None, None, "finalized", None) == (False, "", 0.0)


async def test_settled_empty_book_without_result_books_nothing():
    """A finalized market with result "" and an empty book reads 0.0 from
    get_market_prices (never None); that must not book a total loss."""
    from src.utils.market_prices import get_market_prices
    market = {"status": "finalized", "result": "",
              "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000",
              "no_bid_dollars": "0.0000", "no_ask_dollars": "0.0000"}
    yes_bid, _, no_bid, _ = get_market_prices(market)
    for side in ("YES", "NO"):
        pos = _pos()
        pos.side = side
        assert await should_exit_position(
            pos, yes_bid, no_bid, market["status"], market["result"]) == (False, "", 0.0)


@pytest.mark.parametrize("status", ["closed", "determined", "finalized"])
@pytest.mark.parametrize("result", ["", None, "scalar"])
async def test_settled_status_without_authoritative_result_books_nothing(status, result):
    """status closed + empty result + stale 0.40 bid must not book a loss: the
    market may still settle YES. Closed markets take no orders either, so no
    stop/time/profit exit can run; the row waits."""
    pos = _pos()
    pos.side = "YES"
    pos.entry_price = 0.50
    assert await should_exit_position(pos, 0.40, 0.55, status, result) == (False, "", 0.0)


async def test_settlement_value_dollars_is_authoritative_without_result():
    """get-market settlement_value_dollars is the YES side's settlement value
    (https://docs.kalshi.com/api-reference/market/get-market)."""
    yes_pos, no_pos = _pos(), _pos()
    yes_pos.side = "YES"
    assert await should_exit_position(
        yes_pos, 0.40, 0.55, "determined", "", "1.0000") == (True, "market_resolution", 1.0)
    assert await should_exit_position(
        no_pos, 0.40, 0.55, "determined", "", "1.0000") == (True, "market_resolution", 0.0)
    assert (await should_exit_position(
        yes_pos, 0.40, 0.55, "finalized", "scalar", "0.3000"))[2] == pytest.approx(0.3)
    assert (await should_exit_position(
        no_pos, 0.40, 0.55, "finalized", "scalar", "0.3000"))[2] == pytest.approx(0.7)


async def test_result_is_case_insensitive_and_unknown_result_waits():
    pos = _pos()
    pos.side = "YES"
    assert (await should_exit_position(pos, 0.0, 0.0, "closed", " YES "))[2] == 1.0
    assert (await should_exit_position(pos, 0.4, 0.5, "closed", "voided"))[0] is False


async def test_active_market_ignores_settlement_fields_and_keeps_exits():
    """Non-settled status: stop-loss still fires off the live bid."""
    pos = _pos()
    pos.side = "YES"
    pos.entry_price = 0.50
    pos.stop_loss_price = 0.45
    should_exit, reason, price = await should_exit_position(pos, 0.40, 0.55, "active", "")
    assert should_exit and reason.startswith("stop_loss_triggered") and price == 0.40


async def test_closed_ignores_settlement_value_default():
    """settlement_value_dollars is filled only after determination; a closed
    market's "0.0000" must not book a close."""
    pos = _pos()
    pos.side = "YES"
    assert await should_exit_position(pos, 0.40, 0.55, "closed", "", "0.0000") == (False, "", 0.0)
    assert (await should_exit_position(pos, 0.0, 0.0, "closed", "yes", "0.0000"))[2] == 1.0


# --- through run_tracking -------------------------------------------------

from unittest.mock import AsyncMock, patch

from src.jobs.track import run_tracking
from src.utils.database import DatabaseManager


async def _run(tmp_path, market, side="NO"):
    db = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await db.initialize()
    pos = _pos()
    pos.side, pos.id, pos.entry_price = side, None, 0.50
    await db.add_position(pos)
    with patch("src.jobs.track.KalshiClient") as client:
        client.return_value.get_market = AsyncMock(return_value={"market": market})
        client.return_value.close = AsyncMock()
        await run_tracking(db_manager=db, live=True)
    return await db.get_open_live_positions(), await db.get_all_trade_logs()


async def test_run_tracking_books_scalar_settlement_value(tmp_path):
    market = {"status": "finalized", "result": "scalar", "settlement_value_dollars": "0.3000",
              "yes_bid_dollars": "0.0000", "no_bid_dollars": "0.0000"}
    open_, logs = await _run(tmp_path, market, side="NO")
    assert open_ == [] and len(logs) == 1
    assert logs[0].exit_price == pytest.approx(0.7)
    assert logs[0].pnl == pytest.approx((0.7 - 0.50) * 10)


@pytest.mark.parametrize("bad", ["30", "-0.1", "nan"])
async def test_run_tracking_out_of_range_settlement_value_waits(tmp_path, bad):
    market = {"status": "finalized", "result": "scalar", "settlement_value_dollars": bad,
              "yes_bid_dollars": "0.4000", "no_bid_dollars": "0.5500"}
    open_, logs = await _run(tmp_path, market, side="NO")
    assert len(open_) == 1 and logs == []


# --- non-active statuses, warn-once, case-insensitivity -------------------

@pytest.mark.parametrize("status", ["disputed", "amended", "initialized", "inactive", "unknown", "", None])
async def test_only_active_runs_price_exits(status):
    """Stop-loss/time/profit exits run on 'active' only; any other status waits."""
    pos = _pos()
    pos.side = "YES"
    pos.entry_price = 0.50
    pos.stop_loss_price = 0.45
    pos.max_hold_hours = 1
    pos.timestamp = datetime(2020, 1, 1)
    assert await should_exit_position(pos, 0.40, 0.55, status, "") == (False, "", 0.0)


async def test_active_is_case_insensitive_for_exits():
    pos = _pos()
    pos.side = "YES"
    pos.entry_price = 0.50
    pos.stop_loss_price = 0.45
    assert (await should_exit_position(pos, 0.40, 0.55, "ACTIVE", ""))[0] is True


async def test_unresolved_row_warns_once_across_passes():
    import src.jobs.track as track
    track._warned_unresolved.clear()
    with patch("src.jobs.track.get_trading_logger") as get_logger:
        pos = _pos()
        for _ in range(2):
            assert await should_exit_position(pos, 0.4, 0.5, "closed", "") == (False, "", 0.0)
        assert get_logger.return_value.warning.call_count == 1
    track._warned_unresolved.clear()


async def test_uppercase_result_and_mixed_case_status_settle():
    pos = _pos()
    pos.side = "YES"
    assert await should_exit_position(pos, 0.0, 0.0, "closed", "YES") == (True, "market_resolution", 1.0)
    assert await should_exit_position(
        pos, 0.4, 0.5, "Determined", "", "1.0000") == (True, "market_resolution", 1.0)


@pytest.mark.parametrize("status", ["disputed", "Disputed", "DISPUTED"])
@pytest.mark.parametrize("result", ["yes", "no"])
async def test_disputed_waits_even_with_result(status, result):
    """A dispute can flip the result, so it is not authoritative yet."""
    import src.jobs.track as track
    track._warned_unresolved.clear()
    pos = _pos()
    pos.side = "YES"
    assert await should_exit_position(pos, 0.0, 0.0, status, result, "1.0000") == (False, "", 0.0)
    track._warned_unresolved.clear()


async def test_amended_with_result_settles():
    pos = _pos()
    pos.side = "YES"
    assert await should_exit_position(pos, 0.0, 0.0, "amended", "yes") == (True, "market_resolution", 1.0)
