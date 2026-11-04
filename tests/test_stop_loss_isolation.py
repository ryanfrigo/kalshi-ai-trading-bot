"""place_stop_loss_orders: one bad position never stops the stop-loss for the rest.

The DB, the client and the sell helper are mocks: no network, no keys, no orders.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.jobs.execute import place_stop_loss_orders
from src.utils.database import Position

EX = "src.jobs.execute"


def pos(ticker):
    return Position(market_id=ticker, side="NO", entry_price=0.60, quantity=10,
                    timestamp=datetime.now(), rationale="r", live=True, status="open")


def market(no_bid="0.3000", status="active"):
    return {"market": {"status": status, "yes_bid_dollars": "0.0000", "no_bid_dollars": no_bid}}


def run(responses, sell_raises=(), sell_refuses=()):
    """responses: ticker -> market body, or an Exception for get_market to raise."""
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(return_value=[pos(t) for t in responses])

    async def get_market(ticker):
        r = responses[ticker]
        if isinstance(r, Exception):
            raise r
        return r

    client = MagicMock()
    client.get_market = AsyncMock(side_effect=get_market)

    async def fake_sell(**kw):
        if kw["position"].market_id in sell_raises:
            raise RuntimeError("sell helper crashed")
        if kw["position"].market_id in sell_refuses:
            return False  # refused before sending: order_out stays empty
        kw["order_out"].update({"order_id": f"o-{kw['position'].market_id}"})
        return True

    sell = AsyncMock(side_effect=fake_sell)
    with patch(f"{EX}.place_sell_limit_order", sell), \
         patch(f"{EX}.reconcile_exit_fill", AsyncMock(return_value="filled")):
        res = asyncio.run(place_stop_loss_orders(db, client, stop_loss_threshold=-0.15, live=True))
    return res, [c.kwargs["position"].market_id for c in sell.await_args_list]


def test_both_positions_past_the_stop_are_sold():
    res, sold = run({"A": market(), "B": market()})
    assert sold == ["A", "B"]
    assert res == {"orders_placed": 2, "positions_processed": 2}


def test_a_failing_market_read_does_not_stop_the_next_stop_loss():
    res, sold = run({"A": RuntimeError("503"), "B": market()})
    assert sold == ["B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}


@pytest.mark.parametrize("body", [{}, {"market": {}}, {"market": None}])
def test_missing_market_data_skips_only_that_position(body):
    res, sold = run({"A": body, "B": market()})
    assert sold == ["B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}


def test_an_inactive_market_is_skipped_and_the_next_is_sold():
    res, sold = run({"A": market(status="closed"), "B": market()})
    assert sold == ["B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}


def test_a_sell_helper_crash_on_one_position_still_sells_the_next():
    res, sold = run({"A": market(), "B": market()}, sell_raises={"A"})
    assert sold == ["A", "B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}


def test_a_position_above_the_stop_is_not_sold():
    res, sold = run({"A": market(no_bid="0.5500"), "B": market()})
    assert sold == ["B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}


def test_a_failing_position_read_returns_empty_results():
    db = MagicMock()
    db.get_open_live_positions = AsyncMock(side_effect=RuntimeError("db locked"))
    with patch(f"{EX}.place_sell_limit_order", AsyncMock()) as sell:
        res = asyncio.run(place_stop_loss_orders(db, MagicMock(), stop_loss_threshold=-0.15, live=True))
    sell.assert_not_awaited()
    assert res == {"orders_placed": 0, "positions_processed": 0}


def test_a_refused_sell_is_not_counted_as_an_order_placed():
    res, sold = run({"A": market(), "B": market()}, sell_refuses={"A"})
    assert sold == ["A", "B"]
    assert res == {"orders_placed": 1, "positions_processed": 2}
