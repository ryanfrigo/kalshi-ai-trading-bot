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
