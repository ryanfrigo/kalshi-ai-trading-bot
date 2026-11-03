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
