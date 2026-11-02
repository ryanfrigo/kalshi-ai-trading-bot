"""Safety tests for the order-building paths in src/jobs/execute.py.

Covers the sell-limit payload, profit-taking, stop-loss, and the fail-closed
guards of execute_position. Kalshi client and DB are mocks; no network.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.kalshi_client import KalshiAPIError
from src.jobs.execute import (
    execute_position,
    place_profit_taking_orders,
    place_sell_limit_order,
    place_stop_loss_orders,
)
from src.utils.database import Position

pytestmark = pytest.mark.asyncio


def _pos(side="YES", entry=0.50, qty=10, market_id="T-1"):
    return Position(market_id=market_id, side=side, entry_price=entry, quantity=qty,
                    timestamp=datetime.now(), live=True, status="open", id=7)


_DEFAULT = object()


def _client(order_resp=_DEFAULT, market=None, balance=1_000_000):
    c = MagicMock()
    c.place_order = AsyncMock(return_value={"order": {"order_id": "o1"}}
                              if order_resp is _DEFAULT else order_resp)
    c.get_market = AsyncMock(return_value={"market": market if market is not None else {}})
    c.get_balance = AsyncMock(return_value={"balance": balance})
    # Exit sells first check what the exchange says is sellable: hold plenty, nothing resting.
    c._held_on_side = AsyncMock(return_value=1_000_000.0)
    c.get_orders = AsyncMock(return_value={"orders": [], "cursor": ""})
    return c


def _bids(yes=0, no=0):
    """A v2 market object: *_dollars fixed-point strings only, bids given in cents."""
    return {"yes_bid_dollars": f"{yes / 100:.4f}", "no_bid_dollars": f"{no / 100:.4f}"}


def _db(positions=()):
    d = MagicMock()
    d.get_open_live_positions = AsyncMock(return_value=list(positions))
    d.update_position_to_live = AsyncMock()
    d.set_exit_order = AsyncMock()  # a resting exit sell stores its order id
    return d


# ---------------------------------------------------------------- sell limit
async def test_sell_limit_yes_payload():
    c = _client()
    assert await place_sell_limit_order(_pos("YES", qty=4), 0.62, _db(), c) is True
    kw = c.place_order.call_args.kwargs
    assert (kw["side"], kw["action"], kw["type_"], kw["count"]) == ("yes", "sell", "limit", 4)
    assert kw["yes_price"] == 62 and "no_price" not in kw


async def test_sell_limit_no_payload_uses_no_price_only():
    c = _client()
    assert await place_sell_limit_order(_pos("NO", qty=3), 0.31, _db(), c) is True
    kw = c.place_order.call_args.kwargs
    assert (kw["side"], kw["action"]) == ("no", "sell")
    assert kw["no_price"] == 31 and "yes_price" not in kw


@pytest.mark.parametrize("resp", [None, {}, {"error": "x"}])
async def test_sell_limit_without_order_in_response_is_failure(resp):
    assert await place_sell_limit_order(_pos(), 0.5, _db(), _client(order_resp=resp)) is False


async def test_sell_limit_swallows_api_error_and_reports_failure():
    c = _client()
    c.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: x"))
    assert await place_sell_limit_order(_pos(), 0.5, _db(), c) is False


async def test_sell_limit_price_does_not_truncate_float_error():
    c = _client()
    await place_sell_limit_order(_pos("YES"), 0.29, _db(), c)
    assert c.place_order.call_args.kwargs["yes_price"] == 29


@pytest.mark.parametrize("price", [0.004, 0.0, 1.0, 1.5])
async def test_sell_limit_out_of_range_price_is_not_sent(price):
    c = _client()
    ok = await place_sell_limit_order(_pos("YES"), price, _db(), c)
    assert ok is False
    c.place_order.assert_not_called()
    c._held_on_side.assert_not_called()  # rejected before any exchange read


# -------------------------------------------------------------- profit taking
async def test_profit_taking_places_sell_at_98pct_of_current():
    c = _client(market=_bids(yes=80))  # entry 0.50 -> +60%
    res = await place_profit_taking_orders(_db([_pos("YES", 0.50)]), c, profit_threshold=0.25)
    assert res == {"orders_placed": 1, "positions_processed": 1}
    kw = c.place_order.call_args.kwargs
    assert kw["action"] == "sell" and kw["yes_price"] == 78  # 0.80*0.98=0.784 -> 78c


async def test_profit_taking_below_threshold_places_nothing():
    c = _client(market=_bids(yes=55))  # +10%
    res = await place_profit_taking_orders(_db([_pos("YES", 0.50)]), c, profit_threshold=0.25)
    assert res == {"orders_placed": 0, "positions_processed": 1}
    c.place_order.assert_not_called()


async def test_profit_taking_exactly_at_threshold_triggers():
    c = _client(market=_bids(yes=75))  # (0.75-0.60)/0.60 = 25%
    res = await place_profit_taking_orders(_db([_pos("YES", 0.60)]), c, profit_threshold=0.25)
    assert res["orders_placed"] == 1


async def test_profit_taking_exact_threshold_survives_float_error():
    c = _client(market=_bids(yes=12))  # (0.12-0.10)/0.10 == 0.1999999999999999
    res = await place_profit_taking_orders(_db([_pos("YES", 0.10)]), c, profit_threshold=0.20)
    assert res["orders_placed"] == 1


async def test_profit_taking_no_side_reads_no_bid():
    c = _client(market=_bids(yes=5, no=90))
    res = await place_profit_taking_orders(_db([_pos("NO", 0.60)]), c, profit_threshold=0.25)
    assert res["orders_placed"] == 1
    kw = c.place_order.call_args.kwargs
    assert kw["side"] == "no" and kw["no_price"] == 88 and "yes_price" not in kw


@pytest.mark.parametrize("market", [{}, _bids(yes=0)])
async def test_profit_taking_skips_missing_or_zero_price(market):
    c = _client(market=market)
    res = await place_profit_taking_orders(_db([_pos()]), c)
    assert res["orders_placed"] == 0
    c.place_order.assert_not_called()


async def test_profit_taking_one_failure_does_not_stop_the_rest():
    c = _client()
    c.get_market = AsyncMock(side_effect=[RuntimeError("boom"), {"market": _bids(yes=90)}])
    res = await place_profit_taking_orders(
        _db([_pos(market_id="A"), _pos(market_id="B")]), c, profit_threshold=0.25)
    assert res == {"orders_placed": 1, "positions_processed": 2}
    assert c.place_order.call_args.kwargs["ticker"] == "B"


async def test_profit_taking_db_failure_returns_zero_counts():
    d = _db()
    d.get_open_live_positions = AsyncMock(side_effect=RuntimeError("db"))
    assert await place_profit_taking_orders(d, _client()) == {"orders_placed": 0,
                                                              "positions_processed": 0}


@pytest.mark.parametrize("side,job,kw", [
    ("YES", place_profit_taking_orders, {"profit_threshold": 0.25}),
    ("NO", place_profit_taking_orders, {"profit_threshold": 0.25}),
    ("YES", place_stop_loss_orders, {"stop_loss_threshold": -0.10}),
    ("NO", place_stop_loss_orders, {"stop_loss_threshold": -0.10}),
])
async def test_exit_jobs_price_a_v2_market_at_the_held_bid(side, job, kw):
    # Get Market as v2 sends it: *_dollars strings, no yes_price/no_price keys.
    # Entry 0.50: the held side's bid is 0.80 for profit, 0.40 for stop; the
    # ask and the other side are set to prices that would not trip the job.
    win = job is place_profit_taking_orders
    held_bid, held_ask = ("0.8000", "0.8100") if win else ("0.4000", "0.4100")
    other_bid, other_ask = ("0.1900", "0.2000") if win else ("0.5900", "0.6000")
    y = (held_bid, held_ask) if side == "YES" else (other_bid, other_ask)
    n = (other_bid, other_ask) if side == "YES" else (held_bid, held_ask)
    market = {"ticker": "T-1", "status": "active",
              "yes_bid_dollars": y[0], "yes_ask_dollars": y[1],
              "no_bid_dollars": n[0], "no_ask_dollars": n[1],
              "last_price_dollars": "0.5000", "yes_bid_size_fp": "10.00"}
    c = _client(market=market)
    res = await job(_db([_pos(side, 0.50)]), c, **kw)
    assert res["orders_placed"] == 1
    sent = c.place_order.call_args.kwargs
    assert sent["side"] == side.lower() and sent["action"] == "sell"


# ------------------------------------------------------------------ stop loss
async def test_stop_loss_triggers_and_prices_below_entry():
    c = _client(market=_bids(yes=40))  # entry .50 -> -20%
    res = await place_stop_loss_orders(_db([_pos("YES", 0.50)]), c, stop_loss_threshold=-0.10)
    assert res["orders_placed"] == 1
    # entry*(1 + -0.10*1.1) = 0.445 -> round(44.5) = 44c (half to even)
    assert c.place_order.call_args.kwargs["yes_price"] == 44


async def test_stop_loss_not_triggered_above_threshold():
    c = _client(market=_bids(yes=46))  # -8%
    res = await place_stop_loss_orders(_db([_pos("YES", 0.50)]), c, stop_loss_threshold=-0.10)
    assert res["orders_placed"] == 0
    c.place_order.assert_not_called()


async def test_stop_loss_exactly_at_threshold_triggers():
    c = _client(market=_bids(yes=45))  # exactly -10%
    res = await place_stop_loss_orders(_db([_pos("YES", 0.50)]), c, stop_loss_threshold=-0.10)
    assert res["orders_placed"] == 1


async def test_stop_loss_price_floors_at_one_cent():
    c = _client(market=_bids(yes=1))
    # threshold -0.9 -> stop = entry*(1-0.99) = 0.005 -> clamped to 0.01
    res = await place_stop_loss_orders(_db([_pos("YES", 0.50)]), c, stop_loss_threshold=-0.9)
    assert res["orders_placed"] == 1
    assert c.place_order.call_args.kwargs["yes_price"] == 1


async def test_stop_loss_no_side_sells_no_contract():
    c = _client(market=_bids(no=30))  # entry .50 NO -> -40%
    res = await place_stop_loss_orders(_db([_pos("NO", 0.50)]), c, stop_loss_threshold=-0.10)
    assert res["orders_placed"] == 1
    kw = c.place_order.call_args.kwargs
    assert kw["side"] == "no" and "yes_price" not in kw


async def test_stop_loss_halt_rejection_counts_no_order():
    c = _client(market=_bids(yes=10))
    c.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: MANUAL_HALT: x"))
    res = await place_stop_loss_orders(_db([_pos("YES", 0.50)]), c)
    assert res == {"orders_placed": 0, "positions_processed": 1}


# ------------------------------------------------------ execute_position guards
def _market(yes_ask=0.60, no_ask=0.42):
    return {"yes_bid_dollars": 0.58, "yes_ask_dollars": yes_ask,
            "no_bid_dollars": 0.40, "no_ask_dollars": no_ask}


async def test_execute_fails_closed_when_balance_check_errors():
    c = _client(market=_market())
    c.get_balance = AsyncMock(side_effect=RuntimeError("timeout"))
    d = _db()
    assert await execute_position(_pos(), True, d, c) is False
    c.place_order.assert_not_called()
    d.update_position_to_live.assert_not_called()


async def test_execute_blocks_when_cost_exceeds_balance():
    c = _client(market=_market(yes_ask=0.60), balance=599)  # needs 600c
    d = _db()
    assert await execute_position(_pos("YES", qty=10), True, d, c) is False
    c.place_order.assert_not_called()
    d.update_position_to_live.assert_not_called()


async def test_execute_cost_exactly_equal_to_balance_is_allowed():
    c = _client(market=_market(yes_ask=0.60), balance=600)
    assert await execute_position(_pos("YES", qty=10), True, _db(), c) is True
    c.place_order.assert_called_once()


async def test_execute_missing_balance_key_is_zero_and_blocks():
    c = _client(market=_market())
    c.get_balance = AsyncMock(return_value={})
    assert await execute_position(_pos(), True, _db(), c) is False
    c.place_order.assert_not_called()


async def test_execute_no_side_prices_and_costs_the_no_leg():
    c = _client(market=_market(yes_ask=0.60, no_ask=0.42), balance=420)  # 10 * 42c
    assert await execute_position(_pos("NO", qty=10), True, _db(), c) is True
    kw = c.place_order.call_args.kwargs
    assert kw["side"] == "no" and kw["action"] == "buy" and kw["no_price"] == 42
    assert "yes_price" not in kw


@pytest.mark.parametrize("side,yes_ask,no_ask", [("YES", 0.0, 0.42), ("NO", 0.60, 0.0),
                                                 ("YES", 1.0, 0.42), ("NO", 0.60, 1.0)])
async def test_execute_rejects_out_of_range_ask_on_chosen_side(side, yes_ask, no_ask):
    c = _client(market=_market(yes_ask=yes_ask, no_ask=no_ask))
    d = _db()
    assert await execute_position(_pos(side), True, d, c) is False
    c.place_order.assert_not_called()
    d.update_position_to_live.assert_not_called()


async def test_execute_api_error_leaves_position_not_live():
    c = _client(market=_market())
    c.place_order = AsyncMock(side_effect=KalshiAPIError("RISK_HALT: DRAWDOWN"))
    d = _db()
    assert await execute_position(_pos(), True, d, c) is False
    d.update_position_to_live.assert_not_called()


async def test_execute_paper_mode_never_touches_exchange():
    c = _client()
    d = _db()
    assert await execute_position(_pos(), False, d, c) is True
    c.place_order.assert_not_called()
    c.get_market.assert_not_called()
    c.get_balance.assert_not_called()
    d.update_position_to_live.assert_awaited_once()
